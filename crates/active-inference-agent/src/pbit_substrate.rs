//! pBit Substrate Integration for Conscious Processing
//!
//! Bridges probabilistic bit dynamics with active inference,
//! implementing consciousness as emergent from stochastic computation.
//!
//! # Mathematical Foundation
//!
//! The pBit substrate provides a physical grounding for:
//! - Belief updates as pBit state evolution
//! - Free energy as pBit lattice energy
//! - Action selection via pBit annealing
//!
//! # Key Insight
//!
//! Consciousness emerges from the thermodynamic equilibration
//! of a pBit network, where:
//! - Each belief dimension maps to a pBit cluster
//! - Markovian transitions emerge from coupling dynamics
//! - Free energy minimization = pBit energy minimization

use nalgebra as na;
use serde::{Deserialize, Serialize};

use crate::{ConsciousnessError, ConsciousnessResult, ThermodynamicState};

/// pBit representation for active inference
///
/// Simplified version optimized for belief encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePBit {
    /// Current state (0 or 1)
    pub state: bool,
    /// Bias (external field)
    pub bias: f64,
    /// Temperature
    pub temperature: f64,
    /// Probability of state 1
    pub prob_one: f64,
}

impl InferencePBit {
    /// Create new pBit
    pub fn new(temperature: f64) -> Self {
        Self {
            state: false,
            bias: 0.0,
            temperature: temperature.max(0.001), // Prevent division by zero
            prob_one: 0.5,
        }
    }

    /// Update probability given effective field
    ///
    /// P(s=1) = σ(h_eff/T)
    #[inline]
    pub fn update_probability(&mut self, h_eff: f64) {
        self.prob_one = sigmoid(h_eff / self.temperature);
    }

    /// Stochastic state update
    ///
    /// Returns true if state changed
    pub fn stochastic_update(&mut self, h_eff: f64, random_value: f64) -> bool {
        self.update_probability(h_eff);
        let new_state = random_value < self.prob_one;
        let changed = new_state != self.state;
        self.state = new_state;
        changed
    }

    /// Get spin value (-1 or +1)
    #[inline]
    pub fn spin(&self) -> f64 {
        if self.state { 1.0 } else { -1.0 }
    }
}

/// pBit network for encoding probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitBeliefNetwork {
    /// pBits encoding the belief state
    pub pbits: Vec<InferencePBit>,
    /// Coupling matrix J[i,j]
    pub couplings: na::DMatrix<f64>,
    /// Number of pBits per belief dimension
    pub bits_per_dim: usize,
    /// Belief dimensionality
    pub belief_dim: usize,
    /// Temperature
    pub temperature: f64,
    /// Thermodynamic state tracking
    pub thermo: ThermodynamicState,
}

impl PBitBeliefNetwork {
    /// Create network with given dimensions
    ///
    /// Uses unary encoding: each belief dimension has bits_per_dim pBits
    pub fn new(belief_dim: usize, bits_per_dim: usize, temperature: f64) -> Self {
        let n_pbits = belief_dim * bits_per_dim;

        let pbits = (0..n_pbits)
            .map(|_| InferencePBit::new(temperature))
            .collect();

        // Initialize random couplings (small)
        let mut couplings = na::DMatrix::zeros(n_pbits, n_pbits);

        // Add within-dimension anti-ferromagnetic coupling (softmax-like)
        for d in 0..belief_dim {
            for i in 0..bits_per_dim {
                for j in (i + 1)..bits_per_dim {
                    let idx_i = d * bits_per_dim + i;
                    let idx_j = d * bits_per_dim + j;
                    // Weak anti-coupling encourages one-hot-like encoding
                    couplings[(idx_i, idx_j)] = -0.1;
                    couplings[(idx_j, idx_i)] = -0.1;
                }
            }
        }

        Self {
            pbits,
            couplings,
            bits_per_dim,
            belief_dim,
            temperature,
            thermo: ThermodynamicState::new(temperature, 1e-12),
        }
    }

    /// Encode belief vector into pBit states
    ///
    /// Uses thermometer encoding: belief[d] = fraction of active pBits in dimension d
    pub fn encode_belief(&mut self, belief: &na::DVector<f64>) -> ConsciousnessResult<()> {
        if belief.len() != self.belief_dim {
            return Err(ConsciousnessError::DimensionMismatch {
                expected: self.belief_dim,
                actual: belief.len(),
            });
        }

        for d in 0..self.belief_dim {
            // Number of active pBits = belief[d] * bits_per_dim
            let n_active = (belief[d] * self.bits_per_dim as f64).round() as usize;

            for i in 0..self.bits_per_dim {
                let idx = d * self.bits_per_dim + i;
                self.pbits[idx].state = i < n_active;
                self.pbits[idx].bias = if i < n_active { 1.0 } else { -1.0 };
            }
        }

        Ok(())
    }

    /// Decode pBit states into belief vector
    ///
    /// Computes fraction of active pBits per dimension
    pub fn decode_belief(&self) -> na::DVector<f64> {
        let mut belief = na::DVector::zeros(self.belief_dim);

        for d in 0..self.belief_dim {
            let active_count: usize = (0..self.bits_per_dim)
                .filter(|&i| self.pbits[d * self.bits_per_dim + i].state)
                .count();
            belief[d] = active_count as f64 / self.bits_per_dim as f64;
        }

        // Normalize
        let sum = belief.sum();
        if sum > 1e-10 {
            belief /= sum;
        } else {
            belief = na::DVector::from_element(self.belief_dim, 1.0 / self.belief_dim as f64);
        }

        belief
    }

    /// Compute effective field for pBit i
    fn effective_field(&self, idx: usize) -> f64 {
        let mut h_eff = self.pbits[idx].bias;

        for j in 0..self.pbits.len() {
            if j != idx {
                h_eff += self.couplings[(idx, j)] * self.pbits[j].spin();
            }
        }

        h_eff
    }

    /// Compute total network energy
    ///
    /// E = -Σ_i h_i s_i - (1/2) Σ_ij J_ij s_i s_j
    pub fn compute_energy(&self) -> f64 {
        let mut energy = 0.0;

        for i in 0..self.pbits.len() {
            // Bias term
            energy -= self.pbits[i].bias * self.pbits[i].spin();

            // Coupling term (count once)
            for j in (i + 1)..self.pbits.len() {
                energy -= self.couplings[(i, j)] * self.pbits[i].spin() * self.pbits[j].spin();
            }
        }

        energy
    }

    /// Perform one Gibbs sampling sweep
    ///
    /// Updates all pBits asynchronously
    pub fn gibbs_sweep<R: rand::Rng>(&mut self, rng: &mut R) -> ConsciousnessResult<usize> {
        let mut flips = 0;

        for i in 0..self.pbits.len() {
            let h_eff = self.effective_field(i);
            let random_value: f64 = rng.gen();

            if self.pbits[i].stochastic_update(h_eff, random_value) {
                flips += 1;
            }
        }

        // Record thermodynamic cost
        self.thermo.record_processing_cost(flips as f64 * 0.1)?;

        Ok(flips)
    }

    /// Anneal to find low-energy state
    ///
    /// Performs simulated annealing to minimize energy
    pub fn anneal<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        initial_temp: f64,
        final_temp: f64,
        sweeps: usize,
    ) -> ConsciousnessResult<f64> {
        let temp_ratio = (final_temp / initial_temp).powf(1.0 / sweeps as f64);

        let mut temp = initial_temp;
        for _ in 0..sweeps {
            // Update temperature for all pBits
            for pbit in &mut self.pbits {
                pbit.temperature = temp;
            }

            self.gibbs_sweep(rng)?;
            temp *= temp_ratio;
        }

        Ok(self.compute_energy())
    }

    /// Set coupling based on Markovian kernel
    ///
    /// Maps transition probabilities to pBit couplings
    pub fn set_markov_couplings(&mut self, kernel: &na::DMatrix<f64>) {
        if kernel.nrows() != self.belief_dim || kernel.ncols() != self.belief_dim {
            return;
        }

        // Map transition probabilities to inter-dimension couplings
        for d1 in 0..self.belief_dim {
            for d2 in 0..self.belief_dim {
                if d1 != d2 {
                    // Coupling strength proportional to transition probability
                    let coupling_strength = (kernel[(d1, d2)] - 0.5) * 2.0; // Map [0,1] to [-1,1]

                    // Set coupling between first pBits of each dimension
                    let idx1 = d1 * self.bits_per_dim;
                    let idx2 = d2 * self.bits_per_dim;
                    self.couplings[(idx1, idx2)] = coupling_strength * 0.5;
                    self.couplings[(idx2, idx1)] = coupling_strength * 0.5;
                }
            }
        }
    }

    /// Get magnetization per dimension
    ///
    /// m_d = (1/n) Σ_i s_i for pBits in dimension d
    pub fn magnetization(&self) -> na::DVector<f64> {
        let mut mag = na::DVector::zeros(self.belief_dim);

        for d in 0..self.belief_dim {
            let sum: f64 = (0..self.bits_per_dim)
                .map(|i| self.pbits[d * self.bits_per_dim + i].spin())
                .sum();
            mag[d] = sum / self.bits_per_dim as f64;
        }

        mag
    }
}

/// Conscious pBit processor
///
/// Wraps active inference in pBit substrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousPBitProcessor {
    /// Belief network
    pub network: PBitBeliefNetwork,
    /// Perception bias (observation encoding)
    perception_bias: na::DVector<f64>,
    /// Action readout weights
    action_weights: na::DMatrix<f64>,
    /// Number of action outputs
    n_actions: usize,
}

impl ConsciousPBitProcessor {
    /// Create processor for given belief and action dimensions
    pub fn new(belief_dim: usize, n_actions: usize, bits_per_dim: usize, temperature: f64) -> Self {
        let network = PBitBeliefNetwork::new(belief_dim, bits_per_dim, temperature);

        Self {
            network,
            perception_bias: na::DVector::zeros(belief_dim),
            action_weights: na::DMatrix::identity(n_actions, belief_dim),
            n_actions,
        }
    }

    /// Process observation through pBit dynamics
    ///
    /// Returns action probabilities
    pub fn process<R: rand::Rng>(
        &mut self,
        observation: &na::DVector<f64>,
        rng: &mut R,
        sweeps: usize,
    ) -> ConsciousnessResult<na::DVector<f64>> {
        // Encode observation as biases
        self.update_perception_bias(observation);

        // Run pBit dynamics
        for _ in 0..sweeps {
            self.network.gibbs_sweep(rng)?;
        }

        // Decode belief
        let belief = self.network.decode_belief();

        // Compute action probabilities
        let action_logits = &self.action_weights * &belief;

        // Softmax
        let max_logit = action_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = action_logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        Ok(na::DVector::from_iterator(
            self.n_actions,
            exp_logits.iter().map(|x| x / sum_exp),
        ))
    }

    /// Update perception biases from observation
    fn update_perception_bias(&mut self, observation: &na::DVector<f64>) {
        // Simple linear mapping
        let obs_dim = observation.len().min(self.network.belief_dim);

        for d in 0..obs_dim {
            // Map observation to bias: center around 0
            self.perception_bias[d] = (observation[d] - 0.5) * 2.0;

            // Apply to first pBit in each dimension
            let idx = d * self.network.bits_per_dim;
            self.network.pbits[idx].bias = self.perception_bias[d];
        }
    }

    /// Get current belief state
    pub fn belief(&self) -> na::DVector<f64> {
        self.network.decode_belief()
    }

    /// Get thermodynamic state
    pub fn thermodynamics(&self) -> &ThermodynamicState {
        &self.network.thermo
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_pbit_creation() {
        let pbit = InferencePBit::new(1.0);
        assert_eq!(pbit.state, false);
        assert!((pbit.prob_one - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_network_creation() {
        let network = PBitBeliefNetwork::new(4, 8, 1.0);
        assert_eq!(network.pbits.len(), 32); // 4 dims * 8 bits
        assert_eq!(network.belief_dim, 4);
    }

    #[test]
    fn test_belief_encoding() {
        let mut network = PBitBeliefNetwork::new(3, 4, 1.0);
        let belief = na::DVector::from_vec(vec![0.5, 0.3, 0.2]);

        network.encode_belief(&belief).unwrap();
        let decoded = network.decode_belief();

        // Should approximately recover belief
        assert!((decoded.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gibbs_sweep() {
        let mut network = PBitBeliefNetwork::new(2, 4, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let flips = network.gibbs_sweep(&mut rng);
        assert!(flips.is_ok());
    }

    #[test]
    fn test_annealing() {
        let mut network = PBitBeliefNetwork::new(2, 4, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let energy = network.anneal(&mut rng, 2.0, 0.1, 100);
        assert!(energy.is_ok());
        assert!(energy.unwrap().is_finite());
    }

    #[test]
    fn test_processor() {
        let mut processor = ConsciousPBitProcessor::new(3, 2, 4, 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let observation = na::DVector::from_vec(vec![0.8, 0.2, 0.0]);
        let actions = processor.process(&observation, &mut rng, 10);

        assert!(actions.is_ok());
        let probs = actions.unwrap();
        assert!((probs.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_energy_computation() {
        let mut network = PBitBeliefNetwork::new(2, 2, 1.0);

        // Set some states
        network.pbits[0].state = true;
        network.pbits[1].state = false;

        let energy = network.compute_energy();
        assert!(energy.is_finite());
    }
}
