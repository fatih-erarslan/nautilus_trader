//! pBit Hidden Markov Model for Regime Detection
//!
//! Uses Ising model dynamics to model regime transitions with
//! physically-grounded Boltzmann statistics.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Transition Matrix**: T_ij = exp(-E_ij/T) / Z
//! - **Stationary Distribution**: π where πT = π  
//! - **Forward Algorithm**: α_t(j) = Σ_i α_{t-1}(i) × T_ij × B_j(O_t)
//! - **Ising Energy**: E = -J Σ s_i s_j - h Σ s_i
//!
//! Validated stationary distribution for T=1.0:
//! - Bull: 22.4%, Bear: 21.1%, Ranging: 24.8%, Volatile: 17.6%, Crisis: 14.1%

use crate::types::MarketRegime;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Number of market regimes
pub const N_REGIMES: usize = 5;

/// Regime indices
pub const BULL: usize = 0;
pub const BEAR: usize = 1;
pub const RANGING: usize = 2;
pub const VOLATILE: usize = 3;
pub const CRISIS: usize = 4;

/// pBit HMM configuration
#[derive(Debug, Clone)]
pub struct PBitHmmConfig {
    /// Temperature (controls transition randomness)
    pub temperature: f64,
    /// External field (bias toward certain regimes)
    pub external_field: f64,
    /// Observation noise level
    pub observation_noise: f64,
    /// Number of thermalization steps
    pub thermalization_steps: usize,
}

impl Default for PBitHmmConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            external_field: 0.0,
            observation_noise: 0.1,
            thermalization_steps: 100,
        }
    }
}

/// pBit Hidden Markov Model for regime detection
pub struct PBitHmm {
    /// Configuration
    config: PBitHmmConfig,
    /// Energy matrix (E_ij = energy to transition from i to j)
    energy_matrix: Array2<f64>,
    /// Transition matrix (computed from energy)
    transition_matrix: Array2<f64>,
    /// Emission parameters (mean features per regime)
    emission_means: Array2<f64>,
    /// Emission variances
    emission_vars: Array2<f64>,
    /// Current state probabilities (belief state)
    belief_state: Array1<f64>,
    /// Forward probabilities (α)
    forward_probs: Vec<Array1<f64>>,
}

impl PBitHmm {
    /// Create new pBit HMM with default energy matrix
    pub fn new(config: PBitHmmConfig) -> Self {
        // Wolfram-validated energy matrix
        // Lower energy = higher transition probability
        let energy_matrix = Array2::from_shape_vec(
            (N_REGIMES, N_REGIMES),
            vec![
                0.0, 2.0, 1.0, 3.0, 5.0,  // From Bull
                2.0, 0.0, 1.0, 2.0, 3.0,  // From Bear
                1.0, 1.0, 0.0, 2.0, 4.0,  // From Ranging
                2.0, 2.0, 2.0, 0.0, 1.0,  // From Volatile
                4.0, 3.0, 3.0, 1.0, 0.0,  // From Crisis
            ],
        ).unwrap();

        // Compute transition matrix: T_ij = exp(-E_ij/T) / Z
        let transition_matrix = Self::compute_transition_matrix(&energy_matrix, config.temperature);

        // Default emission parameters (features: volatility, trend, volume)
        let emission_means = Array2::from_shape_vec(
            (N_REGIMES, 3),
            vec![
                0.02, 0.8, 1.2,   // Bull: low vol, strong up trend, high volume
                0.03, -0.7, 1.1,  // Bear: med vol, down trend, high volume
                0.01, 0.0, 0.8,   // Ranging: very low vol, no trend, low volume
                0.05, 0.1, 1.5,   // Volatile: high vol, slight up, very high volume
                0.10, -0.5, 2.0,  // Crisis: extreme vol, down, extreme volume
            ],
        ).unwrap();

        let emission_vars = Array2::from_shape_vec(
            (N_REGIMES, 3),
            vec![
                0.01, 0.3, 0.3,  // Bull
                0.01, 0.3, 0.3,  // Bear
                0.005, 0.2, 0.2, // Ranging
                0.02, 0.4, 0.4,  // Volatile
                0.03, 0.5, 0.5,  // Crisis
            ],
        ).unwrap();

        // Uniform initial belief
        let belief_state = Array1::from_elem(N_REGIMES, 1.0 / N_REGIMES as f64);

        Self {
            config,
            energy_matrix,
            transition_matrix,
            emission_means,
            emission_vars,
            belief_state,
            forward_probs: Vec::new(),
        }
    }

    /// Compute transition matrix from energy matrix using Boltzmann distribution
    /// T_ij = exp(-E_ij / T) / Σ_k exp(-E_ik / T)
    fn compute_transition_matrix(energy: &Array2<f64>, temperature: f64) -> Array2<f64> {
        let mut trans = Array2::zeros((N_REGIMES, N_REGIMES));
        
        for i in 0..N_REGIMES {
            let mut row_sum = 0.0;
            for j in 0..N_REGIMES {
                let p = (-energy[[i, j]] / temperature).exp();
                trans[[i, j]] = p;
                row_sum += p;
            }
            // Normalize
            for j in 0..N_REGIMES {
                trans[[i, j]] /= row_sum;
            }
        }
        
        trans
    }

    /// Update belief state with new observation (Forward step)
    /// α_t(j) = [Σ_i α_{t-1}(i) × T_ij] × B_j(O_t)
    pub fn update(&mut self, observation: &[f64; 3]) {
        // Compute emission probabilities B_j(O_t)
        let emissions = self.compute_emissions(observation);

        // Forward step: α_new(j) = Σ_i α(i) × T_ij × B_j(O)
        let mut new_belief = Array1::zeros(N_REGIMES);
        
        for j in 0..N_REGIMES {
            let mut prob = 0.0;
            for i in 0..N_REGIMES {
                prob += self.belief_state[i] * self.transition_matrix[[i, j]];
            }
            new_belief[j] = prob * emissions[j];
        }

        // Normalize
        let sum: f64 = new_belief.sum();
        if sum > 1e-10 {
            new_belief /= sum;
        } else {
            // Reset to uniform if underflow
            new_belief.fill(1.0 / N_REGIMES as f64);
        }

        // Store for Viterbi
        self.forward_probs.push(self.belief_state.clone());
        self.belief_state = new_belief;
    }

    /// Compute emission probabilities (Gaussian likelihood)
    fn compute_emissions(&self, observation: &[f64; 3]) -> [f64; N_REGIMES] {
        let mut emissions = [0.0; N_REGIMES];
        
        for j in 0..N_REGIMES {
            let mut log_prob = 0.0;
            for k in 0..3 {
                let mean = self.emission_means[[j, k]];
                let var = self.emission_vars[[j, k]] + self.config.observation_noise;
                let diff = observation[k] - mean;
                log_prob -= 0.5 * (diff * diff) / var;
            }
            emissions[j] = log_prob.exp();
        }
        
        // Normalize
        let sum: f64 = emissions.iter().sum();
        if sum > 1e-10 {
            for e in &mut emissions {
                *e /= sum;
            }
        }
        
        emissions
    }

    /// Get most likely current regime
    pub fn get_regime(&self) -> (MarketRegime, f64) {
        let (idx, &confidence) = self.belief_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let regime = match idx {
            BULL => MarketRegime::Bullish,
            BEAR => MarketRegime::Bearish,
            RANGING => MarketRegime::Ranging,
            VOLATILE => MarketRegime::Volatile,
            CRISIS => MarketRegime::Crisis,
            _ => MarketRegime::Ranging,
        };

        (regime, confidence)
    }

    /// Get transition probabilities from current regime
    pub fn get_transition_probs(&self) -> HashMap<MarketRegime, f64> {
        let (current_regime, _) = self.get_regime();
        let current_idx = self.regime_to_idx(current_regime);
        
        let mut probs = HashMap::new();
        probs.insert(MarketRegime::Bullish, self.transition_matrix[[current_idx, BULL]]);
        probs.insert(MarketRegime::Bearish, self.transition_matrix[[current_idx, BEAR]]);
        probs.insert(MarketRegime::Ranging, self.transition_matrix[[current_idx, RANGING]]);
        probs.insert(MarketRegime::Volatile, self.transition_matrix[[current_idx, VOLATILE]]);
        probs.insert(MarketRegime::Crisis, self.transition_matrix[[current_idx, CRISIS]]);
        probs
    }

    /// Get stationary distribution (long-term regime probabilities)
    pub fn get_stationary_distribution(&self) -> HashMap<MarketRegime, f64> {
        // Power iteration to find stationary distribution
        let mut pi = Array1::from_elem(N_REGIMES, 1.0 / N_REGIMES as f64);
        
        for _ in 0..100 {
            let mut new_pi = Array1::zeros(N_REGIMES);
            for j in 0..N_REGIMES {
                for i in 0..N_REGIMES {
                    new_pi[j] += pi[i] * self.transition_matrix[[i, j]];
                }
            }
            pi = new_pi;
        }

        let mut dist = HashMap::new();
        dist.insert(MarketRegime::Bullish, pi[BULL]);
        dist.insert(MarketRegime::Bearish, pi[BEAR]);
        dist.insert(MarketRegime::Ranging, pi[RANGING]);
        dist.insert(MarketRegime::Volatile, pi[VOLATILE]);
        dist.insert(MarketRegime::Crisis, pi[CRISIS]);
        dist
    }

    /// Set temperature (affects transition randomness)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.config.temperature = temperature;
        self.transition_matrix = Self::compute_transition_matrix(&self.energy_matrix, temperature);
    }

    /// Reset belief state to uniform
    pub fn reset(&mut self) {
        self.belief_state.fill(1.0 / N_REGIMES as f64);
        self.forward_probs.clear();
    }

    fn regime_to_idx(&self, regime: MarketRegime) -> usize {
        match regime {
            MarketRegime::Bullish => BULL,
            MarketRegime::Bearish => BEAR,
            MarketRegime::Ranging => RANGING,
            MarketRegime::Volatile => VOLATILE,
            MarketRegime::Crisis => CRISIS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_matrix_normalization() {
        let hmm = PBitHmm::new(PBitHmmConfig::default());
        
        // Each row should sum to 1
        for i in 0..N_REGIMES {
            let row_sum: f64 = (0..N_REGIMES).map(|j| hmm.transition_matrix[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
        }
    }

    #[test]
    fn test_stationary_distribution() {
        let hmm = PBitHmm::new(PBitHmmConfig::default());
        let dist = hmm.get_stationary_distribution();
        
        // Wolfram validated: Bull ~22.4%, Ranging ~24.8%
        let bull = dist.get(&MarketRegime::Bullish).unwrap();
        let ranging = dist.get(&MarketRegime::Ranging).unwrap();
        
        assert!((bull - 0.224).abs() < 0.01, "Bull = {}", bull);
        assert!((ranging - 0.248).abs() < 0.01, "Ranging = {}", ranging);
    }

    #[test]
    fn test_regime_detection() {
        let mut hmm = PBitHmm::new(PBitHmmConfig::default());
        
        // Simulate bull market observations
        for _ in 0..10 {
            hmm.update(&[0.015, 0.9, 1.3]); // Low vol, strong uptrend, high volume
        }
        
        let (regime, confidence) = hmm.get_regime();
        assert_eq!(regime, MarketRegime::Bullish);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_temperature_effect() {
        let mut hmm_low_t = PBitHmm::new(PBitHmmConfig { temperature: 0.5, ..Default::default() });
        let mut hmm_high_t = PBitHmm::new(PBitHmmConfig { temperature: 2.0, ..Default::default() });
        
        // Low temperature = more deterministic (higher self-transition)
        // High temperature = more random (more uniform)
        let low_t_self = hmm_low_t.transition_matrix[[BULL, BULL]];
        let high_t_self = hmm_high_t.transition_matrix[[BULL, BULL]];
        
        assert!(low_t_self > high_t_self, "Low T should have higher self-transition");
    }
}
