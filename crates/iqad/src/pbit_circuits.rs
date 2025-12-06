//! pBit-based Quantum Circuits for IQAD
//!
//! Replaces roqoqo quantum circuits with pBit probabilistic computing.
//! Implements immune system affinity calculations via pBit correlations.
//!
//! ## Key Mappings
//!
//! - Quantum affinity circuit → pBit spin correlation
//! - State encoding → pBit probability distribution
//! - Measurement → Boltzmann sampling

use crate::error::{IqadError, IqadResult};
use quantum_core::{PBitState, PBitConfig, PBitCoupling, QuantumError};
use ndarray::Array1;
use std::f64::consts::PI;
use std::sync::Arc;
use parking_lot::RwLock;

/// pBit-based quantum backend for IQAD
pub struct PBitBackend {
    /// Number of pBits (qubits equivalent)
    num_pbits: usize,
    /// pBit state
    state: PBitState,
    /// Temperature for Boltzmann sampling
    temperature: f64,
    /// Number of sampling shots
    shots: usize,
}

impl PBitBackend {
    /// Create new pBit backend
    pub fn new(num_pbits: usize, shots: Option<usize>) -> IqadResult<Self> {
        if num_pbits == 0 || num_pbits > 30 {
            return Err(IqadError::InvalidParameter(
                "Number of pBits must be between 1 and 30".to_string()
            ));
        }
        
        let config = PBitConfig {
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: None,
        };
        
        let state = PBitState::with_config(num_pbits, config)
            .map_err(|e| IqadError::QuantumError(e.to_string()))?;
        
        Ok(Self {
            num_pbits,
            state,
            temperature: 1.0,
            shots: shots.unwrap_or(1024),
        })
    }
    
    /// Get number of pBits
    pub fn num_pbits(&self) -> usize {
        self.num_pbits
    }
    
    /// Execute pBit circuit and return expectation values
    pub fn execute(&mut self, sweeps: usize) -> IqadResult<Vec<f64>> {
        for _ in 0..sweeps {
            self.state.sweep();
        }
        
        // Return magnetization per pBit
        let mut measurements = Vec::with_capacity(self.num_pbits);
        for i in 0..self.num_pbits {
            if let Some(pbit) = self.state.get_pbit(i) {
                measurements.push(pbit.spin);
            } else {
                measurements.push(0.0);
            }
        }
        
        Ok(measurements)
    }
    
    /// Set pBit probability (RY encoding equivalent)
    pub fn set_probability(&mut self, index: usize, prob: f64) {
        if let Some(pbit) = self.state.get_pbit_mut(index) {
            pbit.probability_up = prob.clamp(0.0, 1.0);
        }
    }
    
    /// Set pBit bias (RZ encoding equivalent)
    pub fn set_bias(&mut self, index: usize, bias: f64) {
        if let Some(pbit) = self.state.get_pbit_mut(index) {
            pbit.bias = bias;
        }
    }
    
    /// Add coupling between pBits (CNOT equivalent)
    pub fn add_coupling(&mut self, i: usize, j: usize, strength: f64) {
        if strength > 0.0 {
            self.state.add_coupling(PBitCoupling::bell_coupling(i, j, strength.abs()));
        } else {
            self.state.add_coupling(PBitCoupling::anti_bell_coupling(i, j, strength.abs()));
        }
    }
    
    /// Get overall magnetization
    pub fn magnetization(&self) -> f64 {
        self.state.magnetization()
    }
    
    /// Get entropy
    pub fn entropy(&self) -> f64 {
        self.state.entropy()
    }
}

/// pBit-based quantum circuits for IQAD
pub struct PBitCircuits {
    /// Number of pBits
    num_pbits: usize,
    /// Sweeps per circuit execution
    sweeps: usize,
}

impl PBitCircuits {
    /// Create new pBit circuits instance
    pub fn new(num_pbits: usize) -> Self {
        Self {
            num_pbits,
            sweeps: 20,
        }
    }
    
    /// Encode value to pBit probability using RY mapping
    /// RY(θ) → P(↑) = sin²(θ/2)
    fn encode_to_probability(&self, value: f64) -> f64 {
        let normalized = value.clamp(0.0, 1.0);
        let theta = normalized * PI;
        (theta / 2.0).sin().powi(2)
    }
    
    /// Build and execute affinity circuit between pattern and detector
    /// Returns quantum affinity score (0-1)
    pub fn calculate_affinity(
        &self,
        pattern: &[f64],
        detector: &[f64],
    ) -> IqadResult<f64> {
        if pattern.len() != detector.len() {
            return Err(IqadError::QuantumError(
                "Pattern and detector must have same length".to_string()
            ));
        }
        
        let n = pattern.len().min(self.num_pbits);
        
        // Create pBit state
        let config = PBitConfig {
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: None,
        };
        
        let mut state = PBitState::with_config(n, config)
            .map_err(|e| IqadError::QuantumError(e.to_string()))?;
        
        // Encode pattern as pBit probabilities
        for i in 0..n {
            let pattern_prob = self.encode_to_probability(pattern[i % pattern.len()]);
            let detector_prob = self.encode_to_probability(detector[i % detector.len()]);
            
            if let Some(pbit) = state.get_pbit_mut(i) {
                // Combine pattern and detector encoding
                pbit.probability_up = (pattern_prob + detector_prob) / 2.0;
                pbit.bias = (pattern[i % pattern.len()] - detector[i % detector.len()]) * 0.5;
            }
        }
        
        // Add entangling couplings (CNOT chain equivalent)
        for i in 0..n.saturating_sub(1) {
            state.add_coupling(PBitCoupling::bell_coupling(i, i + 1, 0.8));
        }
        
        // Add CZ-like couplings
        for i in 0..n.saturating_sub(1) {
            state.add_coupling(PBitCoupling::bell_coupling(i, i + 1, 0.3));
        }
        
        // Equilibrate
        for _ in 0..self.sweeps {
            state.sweep();
        }
        
        // Calculate affinity from correlation
        // High correlation = high affinity
        let magnetization = state.magnetization().abs();
        
        // Also compute overlap-based affinity
        let mut overlap = 0.0;
        let mut norm_pattern = 0.0;
        let mut norm_detector = 0.0;
        
        for i in 0..pattern.len() {
            overlap += pattern[i] * detector[i];
            norm_pattern += pattern[i] * pattern[i];
            norm_detector += detector[i] * detector[i];
        }
        
        let cosine_affinity = if norm_pattern > 0.0 && norm_detector > 0.0 {
            overlap / (norm_pattern.sqrt() * norm_detector.sqrt())
        } else {
            0.0
        };
        
        // Apply quantum interference pattern (pBit-style)
        let quantum_affinity = 0.5 * (1.0 + cosine_affinity.cos());
        
        // Blend magnetization and cosine affinity
        let final_affinity = 0.3 * magnetization + 0.7 * quantum_affinity;
        
        Ok(final_affinity.clamp(0.0, 1.0))
    }
    
    /// Build detector generation circuit
    /// Generates a new random detector pattern using pBit sampling
    pub fn generate_detector(&self, features: &[f64], existing_patterns: &[Vec<f64>]) -> IqadResult<Vec<f64>> {
        let n = features.len().min(self.num_pbits);
        
        let config = PBitConfig {
            temperature: 2.0, // Higher temperature for exploration
            coupling_strength: 0.5,
            external_field: 0.0,
            seed: None,
        };
        
        let mut state = PBitState::with_config(n, config)
            .map_err(|e| IqadError::QuantumError(e.to_string()))?;
        
        // Initialize with feature-based probabilities
        for i in 0..n {
            let base_prob = self.encode_to_probability(features[i % features.len()]);
            if let Some(pbit) = state.get_pbit_mut(i) {
                // Add randomness for exploration
                pbit.probability_up = (base_prob + 0.5) / 2.0;
            }
        }
        
        // Add negative selection pressure from existing patterns
        for pattern in existing_patterns {
            for i in 0..n.min(pattern.len()) {
                if let Some(pbit) = state.get_pbit_mut(i) {
                    // Push away from existing patterns
                    pbit.bias -= pattern[i] * 0.1;
                }
            }
        }
        
        // Sample to generate new detector
        for _ in 0..self.sweeps * 2 {
            state.sweep();
        }
        
        // Extract detector from sampled state
        let mut detector = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(pbit) = state.get_pbit(i) {
                // Map spin to [0, 1] range
                detector.push((pbit.spin + 1.0) / 2.0);
            } else {
                detector.push(0.5);
            }
        }
        
        Ok(detector)
    }
    
    /// Calculate anomaly score using pBit ensemble
    pub fn calculate_anomaly_score(
        &self,
        pattern: &[f64],
        detectors: &[Vec<f64>],
    ) -> IqadResult<(f64, Vec<f64>)> {
        let mut affinities = Vec::with_capacity(detectors.len());
        let mut max_affinity = 0.0f64;
        
        for detector in detectors {
            let affinity = self.calculate_affinity(pattern, detector)?;
            affinities.push(affinity);
            max_affinity = max_affinity.max(affinity);
        }
        
        // Anomaly score: inverse of max affinity to any detector
        // High affinity to a detector = low anomaly score
        let anomaly_score = 1.0 - max_affinity;
        
        Ok((anomaly_score, affinities))
    }
}

/// Thread-safe wrapper for pBit circuits
pub type SharedPBitCircuits = Arc<RwLock<PBitCircuits>>;

/// Create shared pBit circuits
pub fn create_shared_circuits(num_pbits: usize) -> SharedPBitCircuits {
    Arc::new(RwLock::new(PBitCircuits::new(num_pbits)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pbit_backend_creation() {
        let backend = PBitBackend::new(4, Some(1024)).unwrap();
        assert_eq!(backend.num_pbits(), 4);
    }
    
    #[test]
    fn test_affinity_calculation() {
        let circuits = PBitCircuits::new(8);
        
        // Identical patterns should have high affinity
        let pattern = vec![0.5, 0.5, 0.5, 0.5];
        let affinity = circuits.calculate_affinity(&pattern, &pattern).unwrap();
        assert!(affinity > 0.7, "Identical patterns should have high affinity: {}", affinity);
        
        // Opposite patterns should have lower affinity
        let pattern1 = vec![0.0, 0.0, 0.0, 0.0];
        let pattern2 = vec![1.0, 1.0, 1.0, 1.0];
        let affinity2 = circuits.calculate_affinity(&pattern1, &pattern2).unwrap();
        assert!(affinity2 < affinity, "Opposite patterns should have lower affinity");
    }
    
    #[test]
    fn test_detector_generation() {
        let circuits = PBitCircuits::new(8);
        
        let features = vec![0.5, 0.6, 0.4, 0.3];
        let existing: Vec<Vec<f64>> = vec![];
        
        let detector = circuits.generate_detector(&features, &existing).unwrap();
        assert_eq!(detector.len(), 4);
        
        for &val in &detector {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_anomaly_score() {
        let circuits = PBitCircuits::new(8);
        
        let pattern = vec![0.5, 0.5, 0.5, 0.5];
        let detectors = vec![
            vec![0.5, 0.5, 0.5, 0.5], // Similar
            vec![0.1, 0.9, 0.1, 0.9], // Different
        ];
        
        let (score, affinities) = circuits.calculate_anomaly_score(&pattern, &detectors).unwrap();
        
        assert_eq!(affinities.len(), 2);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
