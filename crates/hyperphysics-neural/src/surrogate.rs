//! Surrogate physics modeling with neural networks
//!
//! Provides fast approximations of physics simulations for real-time applications.
//! Uses neural networks trained on physics simulation data to provide 100-1000x
//! faster inference while maintaining acceptable accuracy.
//!
//! ## Use Cases
//!
//! - Real-time physics estimation in HFT (market microstructure dynamics)
//! - Fast approximation of computationally expensive simulations
//! - Physics-informed neural networks (PINNs)
//! - Model predictive control with neural surrogates

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::activation::Activation;
use crate::core::{Tensor, TensorShape};
use crate::error::{NeuralError, NeuralResult};
use crate::network::{Network, NetworkBuilder};

/// Surrogate model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateConfig {
    /// State dimension (positions + velocities + any other state variables)
    pub state_dim: usize,
    /// Number of bodies/particles in the simulation
    pub num_bodies: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Whether to use physics-informed loss (residual physics)
    pub physics_informed: bool,
    /// Time step for predictions
    pub dt: f64,
    /// Maximum inference latency target (microseconds)
    pub max_latency_us: u64,
    /// Error tolerance for physics accuracy
    pub error_tolerance: f64,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            state_dim: 6, // 3D position + 3D velocity per body
            num_bodies: 10,
            hidden_dims: vec![128, 64, 32],
            physics_informed: false,
            dt: 0.001, // 1ms timestep
            max_latency_us: 100,
            error_tolerance: 0.01,
        }
    }
}

impl SurrogateConfig {
    /// Config for N-body gravitational simulation
    pub fn nbody(num_bodies: usize) -> Self {
        Self {
            state_dim: 6 * num_bodies, // 3D pos + 3D vel per body
            num_bodies,
            hidden_dims: vec![256, 128, 64],
            physics_informed: true,
            dt: 0.001,
            max_latency_us: 50,
            error_tolerance: 0.001,
        }
    }

    /// Config for fluid dynamics approximation
    pub fn fluid(grid_size: usize) -> Self {
        Self {
            state_dim: grid_size * grid_size * 3, // 2D grid with 3 components (vx, vy, p)
            num_bodies: grid_size * grid_size,
            hidden_dims: vec![512, 256, 128],
            physics_informed: true,
            dt: 0.01,
            max_latency_us: 500,
            error_tolerance: 0.05,
        }
    }

    /// Config for market microstructure dynamics
    pub fn market_dynamics(num_levels: usize) -> Self {
        Self {
            state_dim: num_levels * 4, // bid/ask price + bid/ask size per level
            num_bodies: num_levels,
            hidden_dims: vec![64, 32],
            physics_informed: false,
            dt: 0.0001, // 100μs market tick
            max_latency_us: 10,
            error_tolerance: 0.001,
        }
    }
}

/// Physics surrogate model
#[derive(Debug)]
pub struct SurrogatePhysics {
    /// Configuration
    config: SurrogateConfig,
    /// Neural network for state prediction
    network: Network,
    /// Inference statistics
    inference_count: u64,
    total_latency_us: u64,
    max_observed_latency_us: u64,
    /// Accumulated error for monitoring
    accumulated_error: f64,
    error_samples: u64,
}

impl SurrogatePhysics {
    /// Create new surrogate physics model
    pub fn new(config: SurrogateConfig) -> NeuralResult<Self> {
        let network = Self::build_network(&config)?;
        Ok(Self {
            config,
            network,
            inference_count: 0,
            total_latency_us: 0,
            max_observed_latency_us: 0,
            accumulated_error: 0.0,
            error_samples: 0,
        })
    }

    fn build_network(config: &SurrogateConfig) -> NeuralResult<Network> {
        let mut builder = Network::builder()
            .name("SurrogatePhysics")
            .input_dim(config.state_dim)
            .output_dim(config.state_dim) // Predict next state
            .hidden_activation(Activation::Swish) // Good for physics
            .output_activation(Activation::Linear)
            .dropout(0.0); // No dropout for deterministic physics

        for &dim in &config.hidden_dims {
            builder = builder.hidden(dim);
        }

        builder.build()
    }

    /// Predict next state from current state
    ///
    /// # Arguments
    /// * `state` - Current state vector (positions, velocities, etc.)
    ///
    /// # Returns
    /// Predicted next state after time `dt`
    pub fn step(&mut self, state: &[f64]) -> NeuralResult<SurrogateResult> {
        let start = Instant::now();

        if state.len() != self.config.state_dim {
            return Err(NeuralError::DimensionMismatch {
                input_dim: state.len(),
                expected_dim: self.config.state_dim,
            });
        }

        let input = Tensor::new(state.to_vec(), TensorShape::d2(1, self.config.state_dim))?;
        let output = self.network.forward(&input)?;

        let latency_us = start.elapsed().as_micros() as u64;
        self.update_stats(latency_us);

        Ok(SurrogateResult {
            next_state: output.data().to_vec(),
            dt: self.config.dt,
            latency_us,
        })
    }

    /// Predict multiple steps forward
    pub fn multi_step(&mut self, initial_state: &[f64], num_steps: usize) -> NeuralResult<Vec<SurrogateResult>> {
        let mut results = Vec::with_capacity(num_steps);
        let mut current_state = initial_state.to_vec();

        for _ in 0..num_steps {
            let result = self.step(&current_state)?;
            current_state = result.next_state.clone();
            results.push(result);
        }

        Ok(results)
    }

    /// Predict trajectory over a time span
    pub fn trajectory(&mut self, initial_state: &[f64], total_time: f64) -> NeuralResult<Trajectory> {
        let num_steps = (total_time / self.config.dt).ceil() as usize;
        let results = self.multi_step(initial_state, num_steps)?;

        let states: Vec<Vec<f64>> = std::iter::once(initial_state.to_vec())
            .chain(results.iter().map(|r| r.next_state.clone()))
            .collect();

        let times: Vec<f64> = (0..=num_steps)
            .map(|i| i as f64 * self.config.dt)
            .collect();

        let total_latency_us: u64 = results.iter().map(|r| r.latency_us).sum();

        Ok(Trajectory {
            states,
            times,
            total_latency_us,
        })
    }

    /// Record error observation (for monitoring accuracy)
    pub fn record_error(&mut self, predicted: &[f64], actual: &[f64]) {
        if predicted.len() != actual.len() {
            return;
        }

        let error: f64 = predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            .sqrt() / predicted.len() as f64;

        self.accumulated_error += error;
        self.error_samples += 1;
    }

    /// Get average prediction error
    pub fn avg_error(&self) -> f64 {
        if self.error_samples == 0 {
            0.0
        } else {
            self.accumulated_error / self.error_samples as f64
        }
    }

    /// Check if error is within tolerance
    pub fn within_tolerance(&self) -> bool {
        self.avg_error() <= self.config.error_tolerance
    }

    fn update_stats(&mut self, latency_us: u64) {
        self.inference_count += 1;
        self.total_latency_us += latency_us;
        if latency_us > self.max_observed_latency_us {
            self.max_observed_latency_us = latency_us;
        }

        // Warn if exceeding target
        if latency_us > self.config.max_latency_us {
            tracing::warn!(
                "Surrogate inference {}μs exceeded target {}μs",
                latency_us,
                self.config.max_latency_us
            );
        }
    }

    /// Get average inference latency
    pub fn avg_latency_us(&self) -> f64 {
        if self.inference_count == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.inference_count as f64
        }
    }

    /// Get maximum observed latency
    pub fn max_latency_us(&self) -> u64 {
        self.max_observed_latency_us
    }

    /// Get configuration
    pub fn config(&self) -> &SurrogateConfig {
        &self.config
    }

    /// Get network
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Get mutable network (for training)
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.inference_count = 0;
        self.total_latency_us = 0;
        self.max_observed_latency_us = 0;
        self.accumulated_error = 0.0;
        self.error_samples = 0;
    }
}

/// Result of a single surrogate prediction step
#[derive(Debug, Clone)]
pub struct SurrogateResult {
    /// Predicted next state
    pub next_state: Vec<f64>,
    /// Time step used
    pub dt: f64,
    /// Inference latency in microseconds
    pub latency_us: u64,
}

/// Trajectory from multi-step prediction
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// States at each time point (including initial)
    pub states: Vec<Vec<f64>>,
    /// Time values for each state
    pub times: Vec<f64>,
    /// Total inference latency
    pub total_latency_us: u64,
}

impl Trajectory {
    /// Get position at a specific body index (assumes state = [pos, vel] per body)
    pub fn positions(&self, body_idx: usize, dims: usize) -> Vec<Vec<f64>> {
        self.states.iter()
            .map(|state| {
                let start = body_idx * dims * 2; // pos + vel per body
                state[start..start + dims].to_vec()
            })
            .collect()
    }

    /// Get velocities at a specific body index
    pub fn velocities(&self, body_idx: usize, dims: usize) -> Vec<Vec<f64>> {
        self.states.iter()
            .map(|state| {
                let start = body_idx * dims * 2 + dims;
                state[start..start + dims].to_vec()
            })
            .collect()
    }

    /// Total simulation time
    pub fn total_time(&self) -> f64 {
        *self.times.last().unwrap_or(&0.0)
    }

    /// Number of time steps
    pub fn num_steps(&self) -> usize {
        self.states.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surrogate_creation() {
        let config = SurrogateConfig::default();
        let surrogate = SurrogatePhysics::new(config).unwrap();

        assert_eq!(surrogate.config().state_dim, 60); // 6 * 10 bodies
    }

    #[test]
    fn test_surrogate_step() {
        let config = SurrogateConfig {
            state_dim: 6,
            num_bodies: 1,
            hidden_dims: vec![16, 8],
            ..Default::default()
        };

        let mut surrogate = SurrogatePhysics::new(config).unwrap();
        let state = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // pos + vel

        let result = surrogate.step(&state).unwrap();
        assert_eq!(result.next_state.len(), 6);
        assert!(result.latency_us > 0);
    }

    #[test]
    fn test_surrogate_multi_step() {
        let config = SurrogateConfig {
            state_dim: 6,
            num_bodies: 1,
            hidden_dims: vec![16],
            ..Default::default()
        };

        let mut surrogate = SurrogatePhysics::new(config).unwrap();
        let state = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let results = surrogate.multi_step(&state, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_surrogate_trajectory() {
        let config = SurrogateConfig {
            state_dim: 6,
            num_bodies: 1,
            hidden_dims: vec![16],
            dt: 0.1,
            ..Default::default()
        };

        let mut surrogate = SurrogatePhysics::new(config).unwrap();
        let state = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let trajectory = surrogate.trajectory(&state, 1.0).unwrap();
        assert!(trajectory.num_steps() >= 10);
        assert!(trajectory.total_time() >= 1.0 - 0.01);
    }

    #[test]
    fn test_market_config() {
        let config = SurrogateConfig::market_dynamics(5);
        let surrogate = SurrogatePhysics::new(config).unwrap();

        assert_eq!(surrogate.config().state_dim, 20); // 5 levels * 4
        assert_eq!(surrogate.config().max_latency_us, 10);
    }

    #[test]
    fn test_error_tracking() {
        let config = SurrogateConfig {
            state_dim: 4,
            num_bodies: 1,
            hidden_dims: vec![8],
            error_tolerance: 0.1,
            ..Default::default()
        };

        let mut surrogate = SurrogatePhysics::new(config).unwrap();

        surrogate.record_error(&[1.0, 2.0, 3.0, 4.0], &[1.1, 2.1, 3.1, 4.1]);
        assert!(surrogate.avg_error() > 0.0);
        assert!(surrogate.within_tolerance());
    }
}
