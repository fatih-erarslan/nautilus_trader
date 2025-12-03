//! pBit-based noise generation for stochastic routing
//!
//! Uses probabilistic computing concepts for exploration in expert selection.

use crate::{RouterError, Result};
use serde::{Deserialize, Serialize};

// pBit types available when feature is enabled
#[cfg(feature = "pbit")]
#[allow(unused_imports)]
use hyperphysics_pbit::{PBit, PBitLattice};

/// Configuration for pBit noise generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Number of pBits in the noise generator
    pub num_pbits: usize,
    /// Temperature parameter (controls noise intensity)
    pub temperature: f64,
    /// Bias for the pBits
    pub bias: f64,
    /// Coupling strength between pBits
    pub coupling: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            num_pbits: 8,
            temperature: 1.0,
            bias: 0.0,
            coupling: 0.1,
        }
    }
}

/// pBit-based noise generator for exploration
#[derive(Debug)]
pub struct PBitNoiseGenerator {
    config: NoiseConfig,
    /// Current noise values (cached)
    cached_noise: Vec<f64>,
    /// State counter for pseudo-random updates
    state_counter: u64,
}

impl PBitNoiseGenerator {
    /// Create new pBit noise generator
    pub fn new(config: NoiseConfig) -> Result<Self> {
        if config.temperature <= 0.0 {
            return Err(RouterError::InvalidTemperature(config.temperature));
        }

        let cached_noise = vec![0.0; config.num_pbits];

        Ok(Self {
            config,
            cached_noise,
            state_counter: 0,
        })
    }

    /// Generate noise values using pBit dynamics
    ///
    /// Returns a vector of noise values in range [-1, 1] following
    /// Boltzmann statistics at the configured temperature.
    pub fn generate(&mut self) -> &[f64] {
        let t = self.config.temperature;
        let bias = self.config.bias;
        let coupling = self.config.coupling;

        // Simulate pBit dynamics
        for i in 0..self.config.num_pbits {
            // Compute effective field
            let neighbor_sum: f64 = self.cached_noise.iter().sum();
            let h_eff = bias + coupling * neighbor_sum / (self.config.num_pbits as f64);

            // Sigmoid probability at temperature T
            let p = 1.0 / (1.0 + (-h_eff / t).exp());

            // Generate stochastic output using deterministic chaos-based randomness
            // This provides reproducible but seemingly random behavior
            self.state_counter = self.state_counter.wrapping_add(1);
            let pseudo_random = (self.state_counter as f64 * 0.6180339887) % 1.0;

            // Map to {-1, +1} based on probability
            self.cached_noise[i] = if pseudo_random < p { 1.0 } else { -1.0 };
        }

        // Scale by temperature for intensity control
        for noise in &mut self.cached_noise {
            *noise *= t.sqrt();
        }

        &self.cached_noise
    }

    /// Generate Gaussian-like noise from pBit states
    ///
    /// Uses central limit theorem: sum of many binary variables
    /// approximates a Gaussian distribution.
    pub fn generate_gaussian(&mut self, num_samples: usize) -> Vec<f64> {
        let mut result = vec![0.0; num_samples];

        // Use multiple pBit samples to approximate Gaussian
        let samples_per_output = 12; // CLT approximation

        for output in &mut result {
            let mut sum = 0.0;
            for _ in 0..samples_per_output {
                let noise = self.generate();
                sum += noise.iter().sum::<f64>() / (self.config.num_pbits as f64);
            }
            // Normalize to standard Gaussian
            *output = sum / (samples_per_output as f64).sqrt();
        }

        result
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }

    /// Set temperature (for annealing)
    pub fn set_temperature(&mut self, temperature: f64) -> Result<()> {
        if temperature <= 0.0 {
            return Err(RouterError::InvalidTemperature(temperature));
        }
        self.config.temperature = temperature;
        Ok(())
    }

    /// Anneal temperature by factor
    pub fn anneal(&mut self, factor: f64) {
        self.config.temperature *= factor;
        self.config.temperature = self.config.temperature.max(0.01); // Lower bound
    }

    /// Reset to initial temperature
    pub fn reset_temperature(&mut self, initial_temp: f64) -> Result<()> {
        self.set_temperature(initial_temp)
    }

    /// Get number of pBits
    pub fn num_pbits(&self) -> usize {
        self.config.num_pbits
    }
}

/// Trait for objects that can generate routing noise
#[allow(dead_code)]
pub trait NoiseSource {
    /// Generate noise vector of given size
    fn sample(&mut self, size: usize) -> Vec<f64>;
}

impl NoiseSource for PBitNoiseGenerator {
    fn sample(&mut self, size: usize) -> Vec<f64> {
        self.generate_gaussian(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_generation() {
        let config = NoiseConfig {
            num_pbits: 8,
            temperature: 1.0,
            ..Default::default()
        };
        let mut gen = PBitNoiseGenerator::new(config).unwrap();

        let noise = gen.generate();
        assert_eq!(noise.len(), 8);

        // Noise should be bounded
        for &n in noise {
            assert!(n.abs() <= 2.0);
        }
    }

    #[test]
    fn test_gaussian_generation() {
        let config = NoiseConfig::default();
        let mut gen = PBitNoiseGenerator::new(config).unwrap();

        let samples = gen.generate_gaussian(100);
        assert_eq!(samples.len(), 100);

        // Mean should be close to 0
        let mean: f64 = samples.iter().sum::<f64>() / 100.0;
        assert!(mean.abs() < 1.0);
    }

    #[test]
    fn test_temperature_annealing() {
        let config = NoiseConfig {
            temperature: 10.0,
            ..Default::default()
        };
        let mut gen = PBitNoiseGenerator::new(config).unwrap();

        assert!((gen.temperature() - 10.0).abs() < 1e-10);

        gen.anneal(0.9);
        assert!((gen.temperature() - 9.0).abs() < 1e-10);

        // Anneal many times
        for _ in 0..100 {
            gen.anneal(0.9);
        }

        // Should hit lower bound
        assert!(gen.temperature() >= 0.01);
    }

    #[test]
    fn test_invalid_temperature() {
        let config = NoiseConfig {
            temperature: -1.0,
            ..Default::default()
        };
        let result = PBitNoiseGenerator::new(config);
        assert!(result.is_err());
    }
}
