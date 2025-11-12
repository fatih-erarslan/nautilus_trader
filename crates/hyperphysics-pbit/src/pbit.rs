//! Individual pBit implementation

use hyperphysics_geometry::PoincarePoint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{PBitError, Result};

/// Probabilistic bit with state and coupling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBit {
    /// Position in hyperbolic space
    position: PoincarePoint,

    /// Current state: false=0, true=1
    state: bool,

    /// Probability of being in state 1
    prob_one: f64,

    /// Bias (external field)
    bias: f64,

    /// Temperature (controls randomness)
    temperature: f64,

    /// Couplings to other pBits: (neighbor_index, strength)
    couplings: HashMap<usize, f64>,
}

impl PBit {
    /// Create new pBit at given position
    pub fn new(position: PoincarePoint, temperature: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(PBitError::InvalidTemperature { temp: temperature });
        }

        Ok(Self {
            position,
            state: false,
            prob_one: 0.5,
            bias: 0.0,
            temperature,
            couplings: HashMap::new(),
        })
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> bool {
        self.state
    }

    /// Get state as float: 0.0 or 1.0
    #[inline]
    pub fn state_f64(&self) -> f64 {
        if self.state { 1.0 } else { 0.0 }
    }

    /// Get state as spin: -1.0 or +1.0 (for Ising model)
    #[inline]
    pub fn spin(&self) -> f64 {
        if self.state { 1.0 } else { -1.0 }
    }

    /// Set state directly
    pub fn set_state(&mut self, state: bool) {
        self.state = state;
    }

    /// Get probability of state 1
    #[inline]
    pub fn prob_one(&self) -> f64 {
        self.prob_one
    }

    /// Get position in hyperbolic space
    #[inline]
    pub fn position(&self) -> &PoincarePoint {
        &self.position
    }

    /// Get temperature
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) -> Result<()> {
        if temperature <= 0.0 {
            return Err(PBitError::InvalidTemperature { temp: temperature });
        }
        self.temperature = temperature;
        Ok(())
    }

    /// Get bias
    #[inline]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Set bias
    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    /// Add coupling to another pBit
    pub fn add_coupling(&mut self, neighbor_idx: usize, strength: f64) {
        self.couplings.insert(neighbor_idx, strength);
    }

    /// Get couplings
    pub fn couplings(&self) -> &HashMap<usize, f64> {
        &self.couplings
    }

    /// Calculate effective field h_eff given neighbor states
    ///
    /// h_eff = bias + Σ_j J_ij s_j
    pub fn effective_field(&self, neighbor_states: &[bool]) -> f64 {
        let mut h_eff = self.bias;

        for (neighbor_idx, strength) in &self.couplings {
            if *neighbor_idx < neighbor_states.len() {
                let neighbor_spin = if neighbor_states[*neighbor_idx] { 1.0 } else { -1.0 };
                h_eff += strength * neighbor_spin;
            }
        }

        h_eff
    }

    /// Update probability based on effective field
    ///
    /// P(s=1) = σ(h_eff/T) = 1/(1 + exp(-h_eff/T))
    pub fn update_probability(&mut self, h_eff: f64) {
        self.prob_one = sigmoid(h_eff / self.temperature);
    }

    /// Stochastic update using given random value
    ///
    /// Returns true if state changed
    pub fn stochastic_update(&mut self, h_eff: f64, random_value: f64) -> bool {
        self.update_probability(h_eff);

        let new_state = random_value < self.prob_one;
        let changed = new_state != self.state;
        self.state = new_state;

        changed
    }

    /// Get flip rate (for Gillespie algorithm)
    ///
    /// Rate = 1/τ where τ is characteristic time
    pub fn flip_rate(&self) -> f64 {
        // Use transition probability as rate (simplified)
        if self.state {
            1.0 - self.prob_one // Rate to flip from 1 to 0
        } else {
            self.prob_one // Rate to flip from 0 to 1
        }
    }
}

/// Sigmoid function: σ(x) = 1/(1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    #[test]
    fn test_pbit_creation() {
        let pos = PoincarePoint::new(na::Vector3::zeros()).unwrap();
        let pbit = PBit::new(pos, 1.0).unwrap();

        assert_eq!(pbit.state(), false);
        assert_eq!(pbit.prob_one(), 0.5);
    }

    #[test]
    fn test_invalid_temperature() {
        let pos = PoincarePoint::new(na::Vector3::zeros()).unwrap();
        assert!(PBit::new(pos, 0.0).is_err());
        assert!(PBit::new(pos, -1.0).is_err());
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_effective_field() {
        let pos = PoincarePoint::new(na::Vector3::zeros()).unwrap();
        let mut pbit = PBit::new(pos, 1.0).unwrap();

        pbit.set_bias(1.0);
        pbit.add_coupling(0, 0.5);
        pbit.add_coupling(1, -0.3);

        let states = vec![true, false];
        let h_eff = pbit.effective_field(&states);

        // h_eff = 1.0 + 0.5*1.0 + (-0.3)*(-1.0) = 1.0 + 0.5 + 0.3 = 1.8
        assert!((h_eff - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_spin_conversion() {
        let pos = PoincarePoint::new(na::Vector3::zeros()).unwrap();
        let mut pbit = PBit::new(pos, 1.0).unwrap();

        pbit.set_state(false);
        assert_eq!(pbit.spin(), -1.0);

        pbit.set_state(true);
        assert_eq!(pbit.spin(), 1.0);
    }
}
