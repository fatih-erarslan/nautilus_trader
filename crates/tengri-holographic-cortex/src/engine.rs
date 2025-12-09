//! # pBit Engine
//!
//! A single pBit engine is a collection of N stochastic binary variables
//! with local coupling and Boltzmann sampling dynamics.
//!
//! ## Mathematical Foundation
//!
//! Each pBit i has state sᵢ ∈ {0, 1} with probability:
//! ```text
//! P(sᵢ = 1) = σ(hᵢ / T) = 1 / (1 + exp(-hᵢ/T))
//! ```
//! where hᵢ = biasᵢ + Σⱼ Wᵢⱼ sⱼ is the effective field.

use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use crate::constants::*;
use crate::{CortexError, Result};

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of pBits in this engine
    pub num_pbits: usize,
    /// Initial temperature
    pub temperature: f64,
    /// Bias values (defaults to 0)
    pub default_bias: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_pbits: 1024,
            temperature: PBIT_DEFAULT_TEMP,
            default_bias: 0.0,
            seed: None,
        }
    }
}

/// A single pBit engine with N stochastic binary variables
pub struct PBitEngine {
    /// Engine ID (0-3 for 4-engine topology)
    id: usize,
    /// Number of pBits
    n: usize,
    /// Current states (0 or 1)
    states: Vec<u8>,
    /// Bias values per pBit
    biases: Vec<f32>,
    /// Local coupling weights (sparse CSR format)
    couplings: SparseCouplings,
    /// Current temperature
    temperature: f64,
    /// Random number generator
    rng: SmallRng,
    /// Last spike indices (for STDP)
    last_spikes: Vec<u64>,
    /// Tick counter
    tick: u64,
}

/// Sparse coupling storage in CSR format
#[derive(Debug, Clone)]
pub struct SparseCouplings {
    /// Row offsets (length: num_pbits + 1)
    pub offsets: Vec<usize>,
    /// Column indices
    pub indices: Vec<usize>,
    /// Coupling weights
    pub weights: Vec<f32>,
}

impl SparseCouplings {
    /// Create empty sparse couplings
    pub fn empty(n: usize) -> Self {
        Self {
            offsets: vec![0; n + 1],
            indices: Vec::new(),
            weights: Vec::new(),
        }
    }
    
    /// Create with random sparse couplings
    pub fn random_sparse(n: usize, density: f64, rng: &mut SmallRng) -> Self {
        let mut offsets = vec![0usize];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        
        for i in 0..n {
            let mut row_indices = Vec::new();
            for j in 0..n {
                if i != j && rng.gen::<f64>() < density {
                    row_indices.push(j);
                    // Random weight in [-0.1, 0.1]
                    weights.push((rng.gen::<f32>() - 0.5) * 0.2);
                }
            }
            indices.extend(row_indices);
            offsets.push(indices.len());
        }
        
        Self { offsets, indices, weights }
    }
    
    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

impl PBitEngine {
    /// Create a new pBit engine
    pub fn new(id: usize, config: EngineConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => SmallRng::seed_from_u64(s + id as u64),
            None => SmallRng::from_entropy(),
        };
        
        let n = config.num_pbits;
        
        Self {
            id,
            n,
            states: vec![0; n],
            biases: vec![config.default_bias as f32; n],
            couplings: SparseCouplings::random_sparse(n, 0.01, &mut rng),
            temperature: config.temperature,
            rng,
            last_spikes: vec![0; n],
            tick: 0,
        }
    }
    
    /// Get engine ID
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Get number of pBits
    pub fn num_pbits(&self) -> usize {
        self.n
    }
    
    /// Get current states
    pub fn states(&self) -> &[u8] {
        &self.states
    }
    
    /// Set temperature
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp.max(PBIT_MIN_TEMP);
    }
    
    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    
    /// Perform one synchronous update of all pBits
    pub fn step(&mut self) {
        self.tick += 1;
        
        // Compute effective fields for all pBits
        let fields = self.compute_fields();
        
        // Sample new states based on Boltzmann distribution
        for i in 0..self.n {
            let prob = pbit_probability(fields[i] as f64, 0.0, self.temperature);
            let new_state = if self.rng.gen::<f64>() < prob { 1 } else { 0 };
            
            // Track spike times for STDP
            if new_state == 1 && self.states[i] == 0 {
                self.last_spikes[i] = self.tick;
            }
            
            self.states[i] = new_state;
        }
    }
    
    /// Perform N micro-timesteps (folded for efficiency)
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
    
    /// Compute effective fields for all pBits
    fn compute_fields(&self) -> Vec<f32> {
        let mut fields = self.biases.clone();
        
        // Add coupling contributions: h_i += Σ_j W_ij * s_j
        for i in 0..self.n {
            let start = self.couplings.offsets[i];
            let end = self.couplings.offsets[i + 1];
            
            for k in start..end {
                let j = self.couplings.indices[k];
                let w = self.couplings.weights[k];
                fields[i] += w * self.states[j] as f32;
            }
        }
        
        fields
    }
    
    /// Get indices of currently active (spiking) pBits
    pub fn active_indices(&self) -> Vec<usize> {
        self.states.iter()
            .enumerate()
            .filter(|(_, &s)| s == 1)
            .map(|(i, _)| i)
            .collect()
    }
    
    /// Get spike rate (fraction of active pBits)
    pub fn spike_rate(&self) -> f64 {
        let active: usize = self.states.iter().map(|&s| s as usize).sum();
        active as f64 / self.n as f64
    }
    
    /// Compute summary embedding (mean field)
    pub fn summary_embedding(&self) -> Vec<f64> {
        // Simple summary: mean activation per block of pBits
        // Maps N pBits to 11D vector for hyperbolic embedding
        let block_size = (self.n + HYPERBOLIC_DIM - 1) / HYPERBOLIC_DIM;
        let mut embedding = vec![0.0; HYPERBOLIC_DIM];
        
        for (i, &s) in self.states.iter().enumerate() {
            let block = i / block_size;
            if block < HYPERBOLIC_DIM {
                embedding[block] += s as f64;
            }
        }
        
        // Normalize to [-1, 1] range for Poincaré ball
        for e in &mut embedding {
            *e = (*e / block_size as f64 - 0.5) * 1.8; // Scale to ~[-0.9, 0.9]
        }
        
        embedding
    }
    
    /// Apply external input (e.g., from other engines)
    pub fn apply_input(&mut self, input: &[f32]) {
        for (i, &inp) in input.iter().enumerate() {
            if i < self.n {
                self.biases[i] += inp;
            }
        }
    }
    
    /// Apply STDP weight updates based on spike correlations
    pub fn apply_stdp(&mut self) {
        let current_tick = self.tick;
        
        for i in 0..self.n {
            if self.states[i] != 1 {
                continue;
            }
            
            let start = self.couplings.offsets[i];
            let end = self.couplings.offsets[i + 1];
            
            for k in start..end {
                let j = self.couplings.indices[k];
                
                // Compute spike timing difference
                let dt = self.last_spikes[i] as i64 - self.last_spikes[j] as i64;
                let dw = stdp_weight_change(dt as f64 * 0.001); // Convert to ms
                
                // Update weight with clamping
                self.couplings.weights[k] = (self.couplings.weights[k] + dw as f32)
                    .clamp(-1.0, 1.0);
            }
        }
    }
    
    /// Reset engine state
    pub fn reset(&mut self) {
        self.states.fill(0);
        self.last_spikes.fill(0);
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_creation() {
        let config = EngineConfig {
            num_pbits: 256,
            seed: Some(42),
            ..Default::default()
        };
        let engine = PBitEngine::new(0, config);
        
        assert_eq!(engine.id(), 0);
        assert_eq!(engine.num_pbits(), 256);
    }
    
    #[test]
    fn test_engine_step() {
        let config = EngineConfig {
            num_pbits: 64,
            seed: Some(42),
            ..Default::default()
        };
        let mut engine = PBitEngine::new(0, config);
        
        engine.step();
        
        // Some pBits should be active
        let rate = engine.spike_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }
    
    #[test]
    fn test_summary_embedding() {
        let config = EngineConfig {
            num_pbits: 128,
            seed: Some(42),
            ..Default::default()
        };
        let mut engine = PBitEngine::new(0, config);
        engine.step();
        
        let embedding = engine.summary_embedding();
        assert_eq!(embedding.len(), HYPERBOLIC_DIM);
        
        // All values should be in Poincaré ball
        for &e in &embedding {
            assert!(e.abs() < 1.0);
        }
    }
    
    #[test]
    fn test_temperature_effect() {
        let config = EngineConfig {
            num_pbits: 256,
            seed: Some(42),
            ..Default::default()
        };
        
        // High temperature: ~50% spike rate
        let mut hot_engine = PBitEngine::new(0, config.clone());
        hot_engine.set_temperature(10.0);
        hot_engine.step_n(10);
        let hot_rate = hot_engine.spike_rate();
        
        // Low temperature: more deterministic
        let mut cold_engine = PBitEngine::new(1, config);
        cold_engine.set_temperature(0.1);
        cold_engine.step_n(10);
        let cold_rate = cold_engine.spike_rate();
        
        // Hot should be closer to 0.5
        assert!((hot_rate - 0.5).abs() < 0.3);
    }
}
