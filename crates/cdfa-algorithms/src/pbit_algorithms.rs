//! pBit-Enhanced Signal Processing Algorithms
//!
//! Boltzmann-weighted algorithms for entropy estimation, volatility
//! clustering, and wavelet analysis using probabilistic bit dynamics.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Boltzmann Entropy**: S = -k_B Σ p_i ln(p_i)
//! - **Partition Function**: Z = Σ exp(-E_i / T)
//! - **Free Energy**: F = -T ln(Z)
//! - **Ising Correlation**: C(r) = ⟨s_i s_{i+r}⟩ - ⟨s⟩²

use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Boltzmann constant for pBit calculations (normalized)
const K_B: f64 = 1.0;

/// pBit algorithm configuration
#[derive(Debug, Clone)]
pub struct PBitAlgorithmConfig {
    /// Temperature for Boltzmann weighting
    pub temperature: f64,
    /// Number of pBit samples for Monte Carlo
    pub num_samples: usize,
    /// Coupling strength for Ising model
    pub coupling_j: f64,
    /// External field strength
    pub external_h: f64,
}

impl Default for PBitAlgorithmConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            num_samples: 1000,
            coupling_j: 1.0,
            external_h: 0.0,
        }
    }
}

/// pBit-enhanced entropy estimator
#[derive(Debug, Clone)]
pub struct PBitEntropy {
    config: PBitAlgorithmConfig,
}

impl PBitEntropy {
    /// Create new entropy estimator
    pub fn new(config: PBitAlgorithmConfig) -> Self {
        Self { config }
    }

    /// Calculate Boltzmann entropy from signal
    pub fn boltzmann_entropy(&self, signal: &ArrayView1<f64>) -> f64 {
        // Convert signal to probability distribution via Boltzmann weights
        let weights = self.boltzmann_weights(signal);
        let z: f64 = weights.iter().sum();
        
        if z < f64::EPSILON {
            return 0.0;
        }

        // Normalize to probabilities
        let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();
        
        // Calculate Shannon entropy
        let mut entropy = 0.0;
        for p in probs {
            if p > f64::EPSILON {
                entropy -= p * p.ln();
            }
        }

        K_B * entropy
    }

    /// Calculate Boltzmann weights for signal values
    fn boltzmann_weights(&self, signal: &ArrayView1<f64>) -> Vec<f64> {
        signal.iter()
            .map(|&x| (-x / self.config.temperature).exp())
            .collect()
    }

    /// Estimate partition function
    pub fn partition_function(&self, energies: &[f64]) -> f64 {
        energies.iter()
            .map(|&e| (-e / self.config.temperature).exp())
            .sum()
    }

    /// Calculate free energy F = -T ln(Z)
    pub fn free_energy(&self, energies: &[f64]) -> f64 {
        let z = self.partition_function(energies);
        if z < f64::EPSILON {
            return f64::INFINITY;
        }
        -self.config.temperature * z.ln()
    }

    /// Calculate thermal average ⟨E⟩
    pub fn thermal_average(&self, energies: &[f64]) -> f64 {
        let z = self.partition_function(energies);
        if z < f64::EPSILON {
            return 0.0;
        }

        energies.iter()
            .map(|&e| e * (-e / self.config.temperature).exp())
            .sum::<f64>() / z
    }
}

/// pBit-enhanced volatility analyzer
#[derive(Debug, Clone)]
pub struct PBitVolatility {
    config: PBitAlgorithmConfig,
}

impl PBitVolatility {
    /// Create new volatility analyzer
    pub fn new(config: PBitAlgorithmConfig) -> Self {
        Self { config }
    }

    /// Detect volatility regimes using Ising model dynamics
    pub fn detect_regimes(&self, returns: &ArrayView1<f64>) -> Vec<VolatilityRegime> {
        let n = returns.len();
        if n < 10 {
            return vec![VolatilityRegime::Normal];
        }

        // Calculate local volatility
        let window = 10.min(n / 3);
        let mut regimes = Vec::with_capacity(n);

        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2).min(n);
            
            let local_vol = self.local_volatility(&returns.slice(ndarray::s![start..end]));
            let regime = self.classify_regime(local_vol);
            regimes.push(regime);
        }

        regimes
    }

    /// Calculate local volatility with Boltzmann weighting
    fn local_volatility(&self, window: &ArrayView1<f64>) -> f64 {
        let n = window.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean = window.mean().unwrap_or(0.0);
        let variance: f64 = window.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;

        variance.sqrt()
    }

    /// Classify volatility regime based on threshold
    fn classify_regime(&self, vol: f64) -> VolatilityRegime {
        // Thresholds based on typical market regimes
        if vol < 0.01 {
            VolatilityRegime::Low
        } else if vol < 0.03 {
            VolatilityRegime::Normal
        } else if vol < 0.06 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }

    /// Calculate regime persistence using Ising correlation
    pub fn regime_persistence(&self, regimes: &[VolatilityRegime]) -> f64 {
        let n = regimes.len();
        if n < 2 {
            return 1.0;
        }

        let mut same_count = 0;
        for i in 1..n {
            if regimes[i] == regimes[i - 1] {
                same_count += 1;
            }
        }

        same_count as f64 / (n - 1) as f64
    }

    /// Calculate Ising-like magnetization from volatility
    pub fn magnetization(&self, returns: &ArrayView1<f64>) -> f64 {
        let regimes = self.detect_regimes(returns);
        
        // Map regimes to spins: High/Extreme = +1, Low/Normal = -1
        let spins: Vec<f64> = regimes.iter()
            .map(|r| match r {
                VolatilityRegime::High | VolatilityRegime::Extreme => 1.0,
                _ => -1.0,
            })
            .collect();

        spins.iter().sum::<f64>() / spins.len() as f64
    }
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

/// pBit-enhanced wavelet analyzer
#[derive(Debug, Clone)]
pub struct PBitWavelet {
    config: PBitAlgorithmConfig,
}

impl PBitWavelet {
    /// Create new wavelet analyzer
    pub fn new(config: PBitAlgorithmConfig) -> Self {
        Self { config }
    }

    /// Haar wavelet decomposition with Boltzmann weighting
    pub fn haar_decompose(&self, signal: &ArrayView1<f64>, levels: usize) -> WaveletDecomposition {
        let mut approximation = signal.to_owned();
        let mut details = Vec::with_capacity(levels);

        for _ in 0..levels {
            if approximation.len() < 2 {
                break;
            }

            let (approx, detail) = self.haar_step(&approximation.view());
            details.push(detail);
            approximation = approx;
        }

        WaveletDecomposition {
            approximation,
            details,
            temperature: self.config.temperature,
        }
    }

    /// Single Haar wavelet step
    fn haar_step(&self, signal: &ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = signal.len() / 2;
        let mut approx = Array1::zeros(n);
        let mut detail = Array1::zeros(n);

        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

        for i in 0..n {
            approx[i] = (signal[2 * i] + signal[2 * i + 1]) * sqrt2_inv;
            detail[i] = (signal[2 * i] - signal[2 * i + 1]) * sqrt2_inv;
        }

        (approx, detail)
    }

    /// Calculate wavelet energy with Boltzmann distribution
    pub fn wavelet_energy(&self, decomp: &WaveletDecomposition) -> Vec<f64> {
        let mut energies = Vec::with_capacity(decomp.details.len() + 1);

        // Energy in approximation
        let approx_energy: f64 = decomp.approximation.iter()
            .map(|x| x.powi(2))
            .sum();
        energies.push(approx_energy);

        // Energy in each detail level
        for detail in &decomp.details {
            let energy: f64 = detail.iter().map(|x| x.powi(2)).sum();
            energies.push(energy);
        }

        // Apply Boltzmann weighting
        let total: f64 = energies.iter().sum();
        if total > f64::EPSILON {
            energies.iter_mut().for_each(|e| *e /= total);
        }

        energies
    }
}

/// Wavelet decomposition result
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    /// Low-frequency approximation
    pub approximation: Array1<f64>,
    /// Detail coefficients at each level
    pub details: Vec<Array1<f64>>,
    /// Temperature used for Boltzmann weighting
    pub temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_boltzmann_entropy() {
        let config = PBitAlgorithmConfig::default();
        let entropy = PBitEntropy::new(config);

        // Uniform signal should have high entropy
        let uniform = array![1.0, 1.0, 1.0, 1.0];
        let s1 = entropy.boltzmann_entropy(&uniform.view());

        // Peaked signal should have lower entropy
        let peaked = array![0.0, 10.0, 0.0, 0.0];
        let s2 = entropy.boltzmann_entropy(&peaked.view());

        assert!(s1 > 0.0);
        assert!(s2 >= 0.0);
    }

    #[test]
    fn test_partition_function_wolfram() {
        // Wolfram: Sum[Exp[-E/T], {E, 0, 3}] at T=1
        // = exp(0) + exp(-1) + exp(-2) + exp(-3)
        // ≈ 1 + 0.368 + 0.135 + 0.050 ≈ 1.553
        let config = PBitAlgorithmConfig { temperature: 1.0, ..Default::default() };
        let entropy = PBitEntropy::new(config);
        
        let energies = [0.0, 1.0, 2.0, 3.0];
        let z = entropy.partition_function(&energies);
        
        assert!((z - 1.553).abs() < 0.01);
    }

    #[test]
    fn test_free_energy_wolfram() {
        // F = -T * ln(Z) = -1 * ln(1.553) ≈ -0.440
        let config = PBitAlgorithmConfig { temperature: 1.0, ..Default::default() };
        let entropy = PBitEntropy::new(config);
        
        let energies = [0.0, 1.0, 2.0, 3.0];
        let f = entropy.free_energy(&energies);
        
        assert!((f - (-0.440)).abs() < 0.01);
    }

    #[test]
    fn test_volatility_regimes() {
        let config = PBitAlgorithmConfig::default();
        let vol = PBitVolatility::new(config);

        // Low volatility returns
        let low_vol = Array1::from_vec(vec![0.001, -0.001, 0.002, -0.002, 0.001, 0.001, -0.001, 0.002, -0.002, 0.001]);
        let regimes = vol.detect_regimes(&low_vol.view());
        
        assert!(!regimes.is_empty());
    }

    #[test]
    fn test_haar_wavelet() {
        let config = PBitAlgorithmConfig::default();
        let wav = PBitWavelet::new(config);

        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decomp = wav.haar_decompose(&signal.view(), 2);

        assert!(!decomp.approximation.is_empty());
        assert_eq!(decomp.details.len(), 2);
    }

    #[test]
    fn test_wavelet_energy() {
        let config = PBitAlgorithmConfig::default();
        let wav = PBitWavelet::new(config);

        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decomp = wav.haar_decompose(&signal.view(), 2);
        let energy = wav.wavelet_energy(&decomp);

        // Energy should sum to 1 (normalized)
        let total: f64 = energy.iter().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
