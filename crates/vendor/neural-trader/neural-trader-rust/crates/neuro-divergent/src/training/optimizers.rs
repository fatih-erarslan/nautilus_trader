//! Advanced optimizers for neural network training
//!
//! Comprehensive optimizer suite including:
//! - AdamW (Adam with decoupled weight decay)
//! - SGD with Nesterov momentum
//! - RMSprop
//! - Support for gradient clipping and learning rate scheduling

use ndarray::Array2;
use crate::{Result, NeuroDivergentError};
use serde::{Deserialize, Serialize};

/// Configuration for all optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub epsilon: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0001,
            epsilon: 1e-8,
        }
    }
}

/// Base optimizer trait
pub trait Optimizer: Send + Sync {
    /// Perform single optimization step
    fn step(&mut self, params: &mut [Array2<f64>], gradients: &[Array2<f64>]) -> Result<()>;

    /// Zero out gradients (if needed)
    fn zero_grad(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f64;

    /// Set learning rate (for schedulers)
    fn set_lr(&mut self, lr: f64);

    /// Get optimizer name
    fn name(&self) -> &str;
}

/// AdamW optimizer - Adam with decoupled weight decay
///
/// AdamW improves upon Adam by decoupling weight decay from gradient-based updates.
/// This leads to better generalization in many cases.
#[derive(Debug, Clone)]
pub struct AdamW {
    config: OptimizerConfig,
    /// First moment estimates (moving averages of gradients)
    m: Vec<Array2<f64>>,
    /// Second moment estimates (moving averages of squared gradients)
    v: Vec<Array2<f64>>,
    /// Time step
    t: usize,
    /// Exponential decay rate for first moment
    beta1: f64,
    /// Exponential decay rate for second moment
    beta2: f64,
}

impl AdamW {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    fn initialize_moments(&mut self, params: &[Array2<f64>]) {
        if self.m.is_empty() {
            self.m = params.iter().map(|p| Array2::zeros(p.dim())).collect();
            self.v = params.iter().map(|p| Array2::zeros(p.dim())).collect();
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [Array2<f64>], gradients: &[Array2<f64>]) -> Result<()> {
        if params.len() != gradients.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Number of parameters and gradients must match".to_string()
            ));
        }

        self.initialize_moments(params);
        self.t += 1;

        let lr = self.config.learning_rate;
        let weight_decay = self.config.weight_decay;
        let epsilon = self.config.epsilon;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            // Update biased first moment estimate
            self.m[i] = &self.m[i] * self.beta1 + grad * (1.0 - self.beta1);

            // Update biased second moment estimate
            let grad_sq = grad.mapv(|g| g.powi(2));
            self.v[i] = &self.v[i] * self.beta2 + &grad_sq * (1.0 - self.beta2);

            // Compute bias-corrected moments
            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // Update parameters with AdamW (decoupled weight decay)
            // θ_t = θ_{t-1} - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * θ_{t-1})
            let update = m_hat.mapv(|m| m) / v_hat.mapv(|v| v.sqrt() + epsilon);
            *param = &*param - &(&update * lr) - &(param.mapv(|p| p * lr * weight_decay));
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Moments are maintained across steps in Adam
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn name(&self) -> &str {
        "AdamW"
    }
}

/// SGD optimizer with Nesterov momentum
///
/// Stochastic Gradient Descent with optional momentum and Nesterov acceleration.
/// Nesterov momentum provides a "look-ahead" that often improves convergence.
#[derive(Debug, Clone)]
pub struct SGD {
    config: OptimizerConfig,
    /// Velocity (momentum buffer)
    velocity: Vec<Array2<f64>>,
    /// Momentum coefficient
    momentum: f64,
    /// Use Nesterov momentum
    nesterov: bool,
}

impl SGD {
    pub fn new(config: OptimizerConfig, momentum: f64) -> Self {
        Self {
            config,
            velocity: Vec::new(),
            momentum,
            nesterov: false,
        }
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    fn initialize_velocity(&mut self, params: &[Array2<f64>]) {
        if self.velocity.is_empty() {
            self.velocity = params.iter().map(|p| Array2::zeros(p.dim())).collect();
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Array2<f64>], gradients: &[Array2<f64>]) -> Result<()> {
        if params.len() != gradients.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Number of parameters and gradients must match".to_string()
            ));
        }

        self.initialize_velocity(params);

        let lr = self.config.learning_rate;
        let weight_decay = self.config.weight_decay;

        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay to gradient
            let grad_with_decay = if weight_decay > 0.0 {
                grad + &param.mapv(|p| p * weight_decay)
            } else {
                grad.clone()
            };

            if self.momentum > 0.0 {
                // Update velocity
                self.velocity[i] = &self.velocity[i] * self.momentum + &grad_with_decay;

                if self.nesterov {
                    // Nesterov momentum: θ = θ - lr * (momentum * v + grad)
                    let update = &self.velocity[i] * self.momentum + &grad_with_decay;
                    *param = &*param - &(&update * lr);
                } else {
                    // Standard momentum: θ = θ - lr * v
                    *param = &*param - &(&self.velocity[i] * lr);
                }
            } else {
                // Standard SGD: θ = θ - lr * grad
                *param = &*param - &(&grad_with_decay * lr);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Velocity is maintained across steps
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn name(&self) -> &str {
        if self.nesterov {
            "SGD-Nesterov"
        } else if self.momentum > 0.0 {
            "SGD-Momentum"
        } else {
            "SGD"
        }
    }
}

/// RMSprop optimizer
///
/// Root Mean Square Propagation - adapts learning rates based on
/// moving average of squared gradients. Good for non-stationary objectives.
#[derive(Debug, Clone)]
pub struct RMSprop {
    config: OptimizerConfig,
    /// Moving average of squared gradients
    square_avg: Vec<Array2<f64>>,
    /// Decay rate for moving average
    alpha: f64,
    /// Momentum coefficient (optional)
    momentum: f64,
    /// Momentum buffer
    momentum_buffer: Vec<Array2<f64>>,
}

impl RMSprop {
    pub fn new(config: OptimizerConfig, alpha: f64) -> Self {
        Self {
            config,
            square_avg: Vec::new(),
            alpha,
            momentum: 0.0,
            momentum_buffer: Vec::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    fn initialize_state(&mut self, params: &[Array2<f64>]) {
        if self.square_avg.is_empty() {
            self.square_avg = params.iter().map(|p| Array2::zeros(p.dim())).collect();
            if self.momentum > 0.0 {
                self.momentum_buffer = params.iter().map(|p| Array2::zeros(p.dim())).collect();
            }
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &mut [Array2<f64>], gradients: &[Array2<f64>]) -> Result<()> {
        if params.len() != gradients.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Number of parameters and gradients must match".to_string()
            ));
        }

        self.initialize_state(params);

        let lr = self.config.learning_rate;
        let weight_decay = self.config.weight_decay;
        let epsilon = self.config.epsilon;

        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if weight_decay > 0.0 {
                grad + &param.mapv(|p| p * weight_decay)
            } else {
                grad.clone()
            };

            // Update moving average of squared gradients
            let grad_sq = grad_with_decay.mapv(|g| g.powi(2));
            self.square_avg[i] = &self.square_avg[i] * self.alpha + &grad_sq * (1.0 - self.alpha);

            // Compute update
            let rms = self.square_avg[i].mapv(|v| (v + epsilon).sqrt());
            let update = &grad_with_decay / &rms;

            if self.momentum > 0.0 {
                // Update momentum buffer
                self.momentum_buffer[i] = &self.momentum_buffer[i] * self.momentum + &update;
                *param = &*param - &(&self.momentum_buffer[i] * lr);
            } else {
                *param = &*param - &(&update * lr);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // State is maintained across steps
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn name(&self) -> &str {
        if self.momentum > 0.0 {
            "RMSprop-Momentum"
        } else {
            "RMSprop"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_adamw_step() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.01,
            epsilon: 1e-8,
        };

        let mut optimizer = AdamW::new(config);

        let mut params = vec![arr2(&[[1.0, 2.0], [3.0, 4.0]])];
        let gradients = vec![arr2(&[[0.1, 0.2], [0.3, 0.4]])];

        // First step
        optimizer.step(&mut params, &gradients).unwrap();

        // Parameters should have changed
        assert!(params[0][[0, 0]] < 1.0);
        assert!(params[0][[1, 1]] < 4.0);

        // Second step - should continue updating
        let old_params = params[0].clone();
        optimizer.step(&mut params, &gradients).unwrap();

        assert_ne!(params[0][[0, 0]], old_params[[0, 0]]);
    }

    #[test]
    fn test_sgd_momentum() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.0,
            epsilon: 1e-8,
        };

        let mut optimizer = SGD::new(config, 0.9);

        let mut params = vec![arr2(&[[1.0]])];
        let gradients = vec![arr2(&[[1.0]])];

        // First step: v = 0.9*0 + 1 = 1, θ = 1 - 0.01*1 = 0.99
        optimizer.step(&mut params, &gradients).unwrap();
        assert_relative_eq!(params[0][[0, 0]], 0.99, epsilon = 1e-10);

        // Second step: v = 0.9*1 + 1 = 1.9, θ = 0.99 - 0.01*1.9 = 0.971
        optimizer.step(&mut params, &gradients).unwrap();
        assert_relative_eq!(params[0][[0, 0]], 0.971, epsilon = 1e-10);
    }

    #[test]
    fn test_sgd_nesterov() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.0,
            epsilon: 1e-8,
        };

        let mut optimizer = SGD::new(config, 0.9).with_nesterov(true);

        let mut params = vec![arr2(&[[1.0]])];
        let gradients = vec![arr2(&[[1.0]])];

        optimizer.step(&mut params, &gradients).unwrap();

        // Nesterov should give different results than standard momentum
        assert!(params[0][[0, 0]] < 1.0);
    }

    #[test]
    fn test_rmsprop_step() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.0,
            epsilon: 1e-8,
        };

        let mut optimizer = RMSprop::new(config, 0.9);

        let mut params = vec![arr2(&[[1.0, 2.0]])];
        let gradients = vec![arr2(&[[0.1, 0.2]])];

        optimizer.step(&mut params, &gradients).unwrap();

        // Parameters should have changed
        assert!(params[0][[0, 0]] < 1.0);
        assert!(params[0][[0, 1]] < 2.0);
    }

    #[test]
    fn test_optimizer_lr_update() {
        let config = OptimizerConfig::default();
        let mut optimizer = AdamW::new(config);

        assert_eq!(optimizer.get_lr(), 0.001);

        optimizer.set_lr(0.0005);
        assert_eq!(optimizer.get_lr(), 0.0005);
    }

    #[test]
    fn test_optimizer_names() {
        let config = OptimizerConfig::default();

        assert_eq!(AdamW::new(config.clone()).name(), "AdamW");
        assert_eq!(SGD::new(config.clone(), 0.0).name(), "SGD");
        assert_eq!(SGD::new(config.clone(), 0.9).name(), "SGD-Momentum");
        assert_eq!(SGD::new(config.clone(), 0.9).with_nesterov(true).name(), "SGD-Nesterov");
        assert_eq!(RMSprop::new(config.clone(), 0.9).name(), "RMSprop");
        assert_eq!(RMSprop::new(config, 0.9).with_momentum(0.9).name(), "RMSprop-Momentum");
    }
}
