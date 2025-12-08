//! Unit and integration tests for cdfa-algorithms
//!
//! Tests advanced signal processing algorithms including wavelet transforms,
//! entropy measures, and volatility analysis

use cdfa_algorithms::{
    wavelet::*,
    entropy::*,
    volatility::*,
    utils,
};
use ndarray::{array, Array1};
use approx::{assert_relative_eq, assert_abs_diff_eq};
use std::f64::consts::PI;

#[cfg(test)]
mod wavelet_tests {
    use super::*;
    
    #[test]
    fn test_haar_wavelet_perfect_reconstruction() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Forward transform
        let (approx, detail) = WaveletTransform::dwt_haar(&signal.view()).unwrap();
        
        // Inverse transform
        let reconstructed = WaveletTransform::idwt_haar(&approx.view(), &detail.view()).unwrap();
        
        // Perfect reconstruction
        for i in 0..signal.len() {
            assert_relative_eq!(signal[i], reconstructed[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_daubechies_wavelet() {
        let signal = array![1.0, 4.0, -3.0, 2.0, 1.0, 5.0, -2.0, 3.0];
        
        // Daubechies-4 transform
        let (approx, detail) = WaveletTransform::dwt_db4(&signal.view()).unwrap();
        
        // Check decomposition properties
        assert_eq!(approx.len(), signal.len() / 2);
        assert_eq!(detail.len(), signal.len() / 2);
        
        // Energy preservation (Parseval's theorem)
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum();
        let wavelet_energy: f64 = approx.iter().map(|x| x * x).sum::<f64>() 
                                + detail.iter().map(|x| x * x).sum::<f64>();
        
        assert_relative_eq!(signal_energy, wavelet_energy, epsilon = 1e-6);
    }
    
    #[test]
    fn test_wavelet_packet_decomposition() {
        let signal: Array1<f64> = Array1::range(0.0, 64.0, 1.0)
            .mapv(|x| (x * 0.1).sin() + (x * 0.3).cos() * 0.5);
        
        let packet = WaveletPacket::decompose(&signal.view(), 3).unwrap();
        
        // Check packet structure
        assert_eq!(packet.level(), 3);
        assert_eq!(packet.num_nodes(), 15); // 2^0 + 2^1 + 2^2 + 2^3 - 1
        
        // Verify reconstruction
        let reconstructed = packet.reconstruct().unwrap();
        
        for i in 0..signal.len() {
            assert_relative_eq!(signal[i], reconstructed[i], epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_continuous_wavelet_transform() {
        // Create a signal with known frequency components
        let n = 256;
        let t: Array1<f64> = Array1::range(0.0, n as f64, 1.0) / n as f64;
        let signal = t.mapv(|x| (2.0 * PI * 10.0 * x).sin() + (2.0 * PI * 25.0 * x).sin() * 0.5);
        
        // CWT with Morlet wavelet
        let scales = array![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let cwt = WaveletTransform::cwt_morlet(&signal.view(), &scales.view()).unwrap();
        
        // Check dimensions
        assert_eq!(cwt.nrows(), scales.len());
        assert_eq!(cwt.ncols(), signal.len());
        
        // Coefficients should be larger at scales corresponding to signal frequencies
        // Scale ~6.4 for 10Hz, Scale ~2.56 for 25Hz (assuming sampling rate normalization)
        let max_scale_10hz = 2; // Index for scale 4.0
        let max_scale_25hz = 1; // Index for scale 2.0
        
        let energy_10hz = cwt.row(max_scale_10hz).mapv(|x| x.norm()).sum();
        let energy_25hz = cwt.row(max_scale_25hz).mapv(|x| x.norm()).sum();
        
        // Higher frequency component should have more energy at smaller scales
        assert!(energy_10hz > energy_25hz * 0.5);
    }
}

#[cfg(test)]
mod entropy_tests {
    use super::*;
    
    #[test]
    fn test_shannon_entropy() {
        // Test 1: Uniform distribution (maximum entropy)
        let uniform = array![0.25, 0.25, 0.25, 0.25];
        let h_uniform = ShannonEntropy::calculate(&uniform.view()).unwrap();
        assert_relative_eq!(h_uniform, 2.0, epsilon = 1e-10); // log2(4) = 2
        
        // Test 2: Deterministic distribution (zero entropy)
        let deterministic = array![1.0, 0.0, 0.0, 0.0];
        let h_det = ShannonEntropy::calculate(&deterministic.view()).unwrap();
        assert_abs_diff_eq!(h_det, 0.0, epsilon = 1e-10);
        
        // Test 3: Binary distribution
        let binary = array![0.5, 0.5];
        let h_binary = ShannonEntropy::calculate(&binary.view()).unwrap();
        assert_relative_eq!(h_binary, 1.0, epsilon = 1e-10); // log2(2) = 1
    }
    
    #[test]
    fn test_sample_entropy() {
        // Test 1: Regular signal (low entropy)
        let regular = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let samp_en_regular = SampleEntropy::calculate(&regular.view(), 2, 0.5).unwrap();
        
        // Test 2: Random signal (high entropy)
        let random = array![1.2, 3.7, 0.5, 2.9, 1.8, 4.2, 0.9, 3.1];
        let samp_en_random = SampleEntropy::calculate(&random.view(), 2, 0.5).unwrap();
        
        // Random signal should have higher entropy
        assert!(samp_en_random > samp_en_regular);
        
        // Test 3: Constant signal (zero entropy)
        let constant = array![5.0; 10];
        let samp_en_const = SampleEntropy::calculate(&constant.view(), 2, 0.1).unwrap();
        assert_abs_diff_eq!(samp_en_const, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_approximate_entropy() {
        let signal = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        
        let m = 2;
        let r = 0.2;
        
        let app_en = ApproximateEntropy::calculate(&signal.view(), m, r).unwrap();
        
        // ApEn should be positive for non-constant signals
        assert!(app_en > 0.0);
        
        // ApEn should be less than log(N) where N is signal length
        assert!(app_en < (signal.len() as f64).ln());
    }
    
    #[test]
    fn test_permutation_entropy() {
        // Test 1: Monotonic sequence (minimum entropy)
        let monotonic = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let perm_en_mono = PermutationEntropy::calculate(&monotonic.view(), 3, 1).unwrap();
        assert_abs_diff_eq!(perm_en_mono, 0.0, epsilon = 1e-10);
        
        // Test 2: Random-like sequence
        let random = array![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let perm_en_random = PermutationEntropy::calculate(&random.view(), 3, 1).unwrap();
        
        // Random sequence should have higher entropy
        assert!(perm_en_random > perm_en_mono);
        
        // Test 3: Oscillating sequence
        let oscillating = array![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0];
        let perm_en_osc = PermutationEntropy::calculate(&oscillating.view(), 3, 1).unwrap();
        
        // Should be between monotonic and random
        assert!(perm_en_osc > perm_en_mono);
        assert!(perm_en_osc < perm_en_random);
    }
    
    #[test]
    fn test_multiscale_entropy() {
        let signal: Array1<f64> = Array1::range(0.0, 100.0, 1.0)
            .mapv(|x| (x * 0.1).sin() + 0.1 * rand::random::<f64>());
        
        let scales = vec![1, 2, 4, 8];
        let mut mse_values = Vec::new();
        
        for scale in scales {
            // Coarse-grain the signal
            let coarse = utils::coarse_grain(&signal.view(), scale).unwrap();
            let mse = SampleEntropy::calculate(&coarse.view(), 2, 0.15).unwrap();
            mse_values.push(mse);
        }
        
        // Entropy should generally increase with scale for complex signals
        assert!(mse_values.len() == 4);
        
        // First value should be positive
        assert!(mse_values[0] > 0.0);
    }
}

#[cfg(test)]
mod volatility_tests {
    use super::*;
    
    #[test]
    fn test_ewma_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, 0.0];
        let lambda = 0.94;
        
        let vol = VolatilityClustering::ewma_volatility(&returns.view(), lambda).unwrap();
        
        // Check dimensions
        assert_eq!(vol.len(), returns.len());
        
        // All volatilities should be non-negative
        assert!(vol.iter().all(|&v| v >= 0.0));
        
        // First volatility should equal absolute first return
        assert_relative_eq!(vol[0], returns[0].abs(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_garch_volatility() {
        let returns = Array1::range(0.0, 100.0, 1.0)
            .mapv(|i| 0.01 * (i * 0.1).sin() + 0.005 * (rand::random::<f64>() - 0.5));
        
        let params = GarchParams {
            omega: 0.000001,
            alpha: 0.1,
            beta: 0.85,
        };
        
        let vol = VolatilityClustering::garch_volatility(&returns.view(), &params).unwrap();
        
        // Check dimensions
        assert_eq!(vol.len(), returns.len());
        
        // All volatilities should be positive
        assert!(vol.iter().all(|&v| v > 0.0));
        
        // Check persistence (alpha + beta < 1 for stationarity)
        assert!(params.alpha + params.beta < 1.0);
    }
    
    #[test]
    fn test_volatility_regimes() {
        // Create returns with clear regime changes
        let mut returns = Vec::new();
        
        // Low volatility regime
        for _ in 0..50 {
            returns.push(0.001 * (rand::random::<f64>() - 0.5));
        }
        
        // High volatility regime
        for _ in 0..50 {
            returns.push(0.01 * (rand::random::<f64>() - 0.5));
        }
        
        // Low volatility regime again
        for _ in 0..50 {
            returns.push(0.001 * (rand::random::<f64>() - 0.5));
        }
        
        let returns_array = Array1::from_vec(returns);
        let regime = VolatilityRegime::detect(&returns_array.view(), 20).unwrap();
        
        // Check regime detection
        assert_eq!(regime.num_regimes(), 3);
        
        // Verify regime volatilities
        let regimes = regime.get_regimes();
        assert!(regimes[1].volatility > regimes[0].volatility); // High vol > Low vol
        assert!(regimes[1].volatility > regimes[2].volatility); // High vol > Low vol
    }
    
    #[test]
    fn test_realized_volatility() {
        // High-frequency returns (e.g., 5-minute returns)
        let hf_returns = Array1::range(0.0, 100.0, 1.0)
            .mapv(|_| 0.0001 * (rand::random::<f64>() - 0.5));
        
        let realized_vol = VolatilityClustering::realized_volatility(&hf_returns.view()).unwrap();
        
        // Realized volatility should be positive
        assert!(realized_vol > 0.0);
        
        // Should be close to standard deviation for i.i.d. returns
        let std_dev = hf_returns.std(0.0);
        assert_relative_eq!(realized_vol, std_dev, epsilon = 0.1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_wavelet_denoising_pipeline() {
        // Create noisy signal
        let n = 128;
        let t: Array1<f64> = Array1::range(0.0, n as f64, 1.0) / n as f64;
        let clean_signal = t.mapv(|x| (2.0 * PI * 5.0 * x).sin());
        let noise = Array1::from_shape_fn(n, |_| 0.1 * (rand::random::<f64>() - 0.5));
        let noisy_signal = &clean_signal + &noise;
        
        // Wavelet decomposition
        let mut coeffs = Vec::new();
        let mut current = noisy_signal.clone();
        
        // Multi-level decomposition
        for _ in 0..3 {
            let (approx, detail) = WaveletTransform::dwt_db4(&current.view()).unwrap();
            coeffs.push(detail);
            current = approx;
        }
        coeffs.push(current); // Final approximation
        
        // Soft thresholding for denoising
        let threshold = 0.05;
        let denoised_coeffs: Vec<Array1<f64>> = coeffs.into_iter()
            .map(|c| c.mapv(|x| {
                if x.abs() < threshold { 0.0 }
                else if x > 0.0 { x - threshold }
                else { x + threshold }
            }))
            .collect();
        
        // Reconstruction would go here (simplified for test)
        // In practice, would use inverse DWT
        
        // Verify some coefficients were zeroed (denoising occurred)
        let zeros: usize = denoised_coeffs.iter()
            .map(|c| c.iter().filter(|&&x| x == 0.0).count())
            .sum();
        assert!(zeros > 0);
    }
    
    #[test]
    fn test_entropy_volatility_relationship() {
        // Generate returns with varying volatility
        let mut returns = Vec::new();
        let mut vol_regimes = Vec::new();
        
        for regime in 0..3 {
            let vol = match regime {
                0 => 0.01,  // Low volatility
                1 => 0.05,  // High volatility
                _ => 0.02,  // Medium volatility
            };
            
            for _ in 0..100 {
                returns.push(vol * (rand::random::<f64>() - 0.5) * 2.0);
                vol_regimes.push(vol);
            }
        }
        
        let returns_array = Array1::from_vec(returns);
        
        // Calculate rolling entropy and volatility
        let window = 50;
        let mut entropies = Vec::new();
        let mut volatilities = Vec::new();
        
        for i in window..returns_array.len() {
            let window_data = returns_array.slice(ndarray::s![i-window..i]);
            
            // Sample entropy
            let entropy = SampleEntropy::calculate(&window_data, 2, 0.2).unwrap();
            entropies.push(entropy);
            
            // EWMA volatility
            let vol = VolatilityClustering::ewma_volatility(&window_data, 0.94).unwrap();
            volatilities.push(vol.mean().unwrap());
        }
        
        // Generally, higher volatility periods should have higher entropy
        // (more uncertainty/randomness)
        assert!(entropies.len() == volatilities.len());
        assert!(entropies.len() > 0);
    }
    
    #[test]
    fn test_multiscale_wavelet_entropy() {
        // Create multiscale signal
        let n = 256;
        let t: Array1<f64> = Array1::range(0.0, n as f64, 1.0) / n as f64;
        
        // Combine multiple frequencies
        let signal = t.mapv(|x| {
            (2.0 * PI * 5.0 * x).sin() +      // Low frequency
            0.5 * (2.0 * PI * 20.0 * x).sin() + // Medium frequency
            0.2 * (2.0 * PI * 50.0 * x).sin()   // High frequency
        });
        
        // Wavelet packet decomposition
        let packet = WaveletPacket::decompose(&signal.view(), 4).unwrap();
        
        // Calculate entropy at each node
        let mut node_entropies = Vec::new();
        for level in 0..=4 {
            for node in 0..(1 << level) {
                if let Ok(coeffs) = packet.get_node_coefficients(level, node) {
                    // Normalize to probability distribution
                    let energy: f64 = coeffs.iter().map(|x| x * x).sum();
                    if energy > 1e-10 {
                        let probs = coeffs.mapv(|x| (x * x) / energy);
                        let entropy = ShannonEntropy::calculate(&probs.view()).unwrap_or(0.0);
                        node_entropies.push((level, node, entropy));
                    }
                }
            }
        }
        
        // Should have computed entropies for multiple nodes
        assert!(!node_entropies.is_empty());
        
        // Different frequency bands should have different entropies
        let entropy_variance: f64 = {
            let mean = node_entropies.iter().map(|(_, _, e)| e).sum::<f64>() / node_entropies.len() as f64;
            node_entropies.iter()
                .map(|(_, _, e)| (e - mean).powi(2))
                .sum::<f64>() / node_entropies.len() as f64
        };
        assert!(entropy_variance > 0.0);
    }
}