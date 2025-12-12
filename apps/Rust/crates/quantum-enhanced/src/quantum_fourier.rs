//! Quantum Fourier Transform for frequency domain pattern analysis

use crate::types::*;
use crate::Result;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;
use tracing::{debug, info};

/// Quantum Fourier Transform engine for frequency domain analysis
#[derive(Clone)]
pub struct QuantumFourierTransform {
    /// Configuration
    config: QuantumConfig,
    /// FFT planner for classical computations
    fft_planner: Arc<FftPlanner<f64>>,
    /// QFT parameters
    qft_params: QftParams,
}

#[derive(Debug, Clone)]
struct QftParams {
    /// Frequency resolution
    frequency_resolution: f64,
    /// Maximum frequency to analyze
    max_frequency: f64,
    /// Spectral window function
    window_type: WindowType,
    /// Quantum phase precision
    phase_precision: f64,
}

#[derive(Debug, Clone)]
enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    Gaussian,
}

impl QuantumFourierTransform {
    /// Create a new quantum Fourier transform engine
    pub async fn new(config: &QuantumConfig) -> Result<Self> {
        info!("Initializing Quantum Fourier Transform Engine");

        let qft_params = QftParams {
            frequency_resolution: config.frequency_resolution,
            max_frequency: 0.5, // Nyquist frequency
            window_type: WindowType::Hanning,
            phase_precision: 1e-8,
        };

        Ok(Self {
            config: config.clone(),
            fft_planner: Arc::new(FftPlanner::new()),
            qft_params,
        })
    }

    /// Perform quantum Fourier transform on entangled patterns
    pub async fn transform(&self, entanglement: &EntanglementCorrelation) -> Result<QuantumFourierResult> {
        debug!("Performing quantum Fourier transform on {} entangled pairs", 
               entanglement.entangled_pairs.len());

        // Step 1: Prepare quantum signal for transformation
        let quantum_signal = self.prepare_quantum_signal(entanglement).await?;
        
        // Step 2: Apply quantum Fourier transform
        let frequency_amplitudes = self.apply_quantum_fft(&quantum_signal).await?;
        
        // Step 3: Extract phase spectrum
        let phase_spectrum = self.extract_phase_spectrum(&frequency_amplitudes).await?;
        
        // Step 4: Identify dominant frequencies
        let dominant_frequencies = self.identify_dominant_frequencies(&frequency_amplitudes).await?;
        
        // Step 5: Calculate frequency-domain entanglement
        let frequency_entanglement = self.calculate_frequency_entanglement(
            &frequency_amplitudes, entanglement
        ).await?;
        
        // Step 6: Compute spectral coherence
        let spectral_coherence = self.compute_spectral_coherence(&frequency_amplitudes).await?;

        let result = QuantumFourierResult {
            frequency_amplitudes,
            phase_spectrum,
            dominant_frequencies,
            frequency_entanglement,
            spectral_coherence,
        };

        debug!("QFT completed: {} dominant frequencies, coherence: {:.3}",
               result.dominant_frequencies.len(), result.spectral_coherence);

        Ok(result)
    }

    /// Perform inverse quantum Fourier transform
    pub async fn inverse_transform(&self, qft_result: &QuantumFourierResult) -> Result<Array1<Complex64>> {
        debug!("Performing inverse quantum Fourier transform");

        // Apply inverse QFT to frequency amplitudes
        let mut time_domain_signal = qft_result.frequency_amplitudes.clone();
        
        // Use inverse FFT algorithm
        let mut scratch = vec![Complex64::new(0.0, 0.0); time_domain_signal.len()];
        let fft_size = time_domain_signal.len();
        
        if fft_size > 0 {
            let fft = self.fft_planner.plan_fft_inverse(fft_size);
            
            // Convert ndarray to vec for FFT
            let mut signal_vec: Vec<Complex64> = time_domain_signal.iter().cloned().collect();
            fft.process_with_scratch(&mut signal_vec, &mut scratch);
            
            // Convert back to ndarray
            time_domain_signal = Array1::from_vec(signal_vec);
            
            // Normalize
            let normalization = 1.0 / (fft_size as f64).sqrt();
            time_domain_signal *= Complex64::new(normalization, 0.0);
        }

        Ok(time_domain_signal)
    }

    /// Analyze frequency-domain quantum interference patterns
    pub async fn analyze_quantum_interference(
        &self,
        qft_result: &QuantumFourierResult,
    ) -> Result<QuantumInterferencePattern> {
        
        // Detect interference patterns in frequency domain
        let interference_frequencies = self.detect_interference_frequencies(qft_result).await?;
        
        // Calculate interference strength
        let interference_strength = self.calculate_interference_strength(&interference_frequencies).await?;
        
        // Identify constructive and destructive interference
        let (constructive_regions, destructive_regions) = self.identify_interference_regions(
            &interference_frequencies
        ).await?;

        Ok(QuantumInterferencePattern {
            interference_frequencies,
            interference_strength,
            constructive_regions,
            destructive_regions,
            coherence_length: self.calculate_coherence_length(qft_result).await?,
        })
    }

    // Private helper methods

    async fn prepare_quantum_signal(&self, entanglement: &EntanglementCorrelation) -> Result<Array1<Complex64>> {
        let correlation_size = entanglement.correlation_matrix.nrows();
        if correlation_size == 0 {
            return Ok(Array1::zeros(1));
        }

        // Convert correlation matrix to 1D signal for FFT
        let mut signal = Array1::zeros(correlation_size * correlation_size);
        
        for i in 0..correlation_size {
            for j in 0..correlation_size {
                let idx = i * correlation_size + j;
                signal[idx] = entanglement.correlation_matrix[[i, j]];
            }
        }

        // Apply window function for better frequency resolution
        self.apply_window_function(&mut signal).await?;

        Ok(signal)
    }

    async fn apply_window_function(&self, signal: &mut Array1<Complex64>) -> Result<()> {
        let signal_length = signal.len();
        
        match self.qft_params.window_type {
            WindowType::Rectangular => {
                // No modification needed
            },
            WindowType::Hamming => {
                for (i, sample) in signal.iter_mut().enumerate() {
                    let window_value = 0.54 - 0.46 * (2.0 * PI * i as f64 / signal_length as f64).cos();
                    *sample *= Complex64::new(window_value, 0.0);
                }
            },
            WindowType::Hanning => {
                for (i, sample) in signal.iter_mut().enumerate() {
                    let window_value = 0.5 * (1.0 - (2.0 * PI * i as f64 / signal_length as f64).cos());
                    *sample *= Complex64::new(window_value, 0.0);
                }
            },
            WindowType::Blackman => {
                for (i, sample) in signal.iter_mut().enumerate() {
                    let arg = 2.0 * PI * i as f64 / signal_length as f64;
                    let window_value = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
                    *sample *= Complex64::new(window_value, 0.0);
                }
            },
            WindowType::Gaussian => {
                let sigma = signal_length as f64 / 6.0; // 3-sigma window
                let center = signal_length as f64 / 2.0;
                for (i, sample) in signal.iter_mut().enumerate() {
                    let distance = (i as f64 - center).abs();
                    let window_value = (-0.5 * (distance / sigma).powi(2)).exp();
                    *sample *= Complex64::new(window_value, 0.0);
                }
            },
        }

        Ok(())
    }

    async fn apply_quantum_fft(&self, signal: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        let signal_length = signal.len();
        if signal_length == 0 {
            return Ok(Array1::zeros(1));
        }

        // Convert to format expected by rustfft
        let mut fft_input: Vec<Complex64> = signal.iter().cloned().collect();
        
        // Pad to next power of 2 for efficiency
        let padded_length = signal_length.next_power_of_two();
        fft_input.resize(padded_length, Complex64::new(0.0, 0.0));

        // Plan and execute FFT
        let fft = self.fft_planner.plan_fft_forward(padded_length);
        let mut scratch = vec![Complex64::new(0.0, 0.0); padded_length];
        fft.process_with_scratch(&mut fft_input, &mut scratch);

        // Apply quantum corrections to standard FFT
        for (i, amplitude) in fft_input.iter_mut().enumerate() {
            // Quantum phase correction
            let quantum_phase = (i as f64 * self.qft_params.phase_precision) % (2.0 * PI);
            let quantum_correction = Complex64::new(quantum_phase.cos(), quantum_phase.sin());
            *amplitude *= quantum_correction;
        }

        // Normalize
        let normalization = 1.0 / (padded_length as f64).sqrt();
        for amplitude in fft_input.iter_mut() {
            *amplitude *= normalization;
        }

        // Convert back to ndarray, keeping only original length
        let result_length = signal_length.min(padded_length);
        Ok(Array1::from_vec(fft_input[..result_length].to_vec()))
    }

    async fn extract_phase_spectrum(&self, frequency_amplitudes: &Array1<Complex64>) -> Result<Array1<f64>> {
        let phase_spectrum: Vec<f64> = frequency_amplitudes.iter()
            .map(|amplitude| amplitude.arg())
            .collect();

        Ok(Array1::from_vec(phase_spectrum))
    }

    async fn identify_dominant_frequencies(&self, frequency_amplitudes: &Array1<Complex64>) -> Result<Vec<f64>> {
        let mut frequency_powers: Vec<(usize, f64)> = frequency_amplitudes.iter()
            .enumerate()
            .map(|(i, amplitude)| (i, amplitude.norm_sqr()))
            .collect();

        // Sort by power (descending)
        frequency_powers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top frequencies above threshold
        let power_threshold = frequency_powers.first().map(|(_, p)| p * 0.1).unwrap_or(0.0);
        let dominant_indices: Vec<usize> = frequency_powers.iter()
            .filter(|(_, power)| *power > power_threshold)
            .take(10) // Limit to top 10 frequencies
            .map(|(idx, _)| *idx)
            .collect();

        // Convert indices to actual frequencies
        let sampling_rate = 1.0 / self.qft_params.frequency_resolution;
        let num_samples = frequency_amplitudes.len();
        
        let dominant_frequencies: Vec<f64> = dominant_indices.iter()
            .map(|&idx| (idx as f64 * sampling_rate) / (num_samples as f64))
            .filter(|&freq| freq <= self.qft_params.max_frequency)
            .collect();

        Ok(dominant_frequencies)
    }

    async fn calculate_frequency_entanglement(
        &self,
        frequency_amplitudes: &Array1<Complex64>,
        entanglement: &EntanglementCorrelation,
    ) -> Result<Array2<Complex64>> {
        
        let num_frequencies = frequency_amplitudes.len();
        let num_pairs = entanglement.entangled_pairs.len().max(1);
        
        let mut frequency_entanglement = Array2::zeros((num_frequencies, num_pairs));

        // Calculate entanglement in frequency domain
        for freq_idx in 0..num_frequencies {
            for pair_idx in 0..num_pairs {
                if pair_idx < entanglement.bell_coefficients.len() {
                    let freq_amplitude = frequency_amplitudes[freq_idx];
                    let bell_coefficient = entanglement.bell_coefficients[pair_idx];
                    
                    // Frequency-domain entanglement measure
                    let entanglement_amplitude = freq_amplitude * bell_coefficient.conj();
                    frequency_entanglement[[freq_idx, pair_idx]] = entanglement_amplitude;
                }
            }
        }

        Ok(frequency_entanglement)
    }

    async fn compute_spectral_coherence(&self, frequency_amplitudes: &Array1<Complex64>) -> Result<f64> {
        // Calculate spectral coherence as measure of frequency domain organization
        
        if frequency_amplitudes.is_empty() {
            return Ok(0.0);
        }

        // Compute power spectral density
        let power_spectrum: Vec<f64> = frequency_amplitudes.iter()
            .map(|amplitude| amplitude.norm_sqr())
            .collect();

        // Calculate spectral entropy
        let total_power: f64 = power_spectrum.iter().sum();
        if total_power <= 0.0 {
            return Ok(0.0);
        }

        let mut spectral_entropy = 0.0;
        for &power in &power_spectrum {
            if power > 0.0 {
                let normalized_power = power / total_power;
                spectral_entropy -= normalized_power * normalized_power.ln();
            }
        }

        // Normalize entropy to get coherence (1 - normalized_entropy)
        let max_entropy = (power_spectrum.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            spectral_entropy / max_entropy
        } else {
            0.0
        };

        let coherence = (1.0 - normalized_entropy).max(0.0).min(1.0);
        
        Ok(coherence)
    }

    async fn detect_interference_frequencies(&self, qft_result: &QuantumFourierResult) -> Result<Vec<f64>> {
        let mut interference_frequencies = Vec::new();
        
        // Look for frequencies where phase relationships indicate interference
        for (i, &freq) in qft_result.dominant_frequencies.iter().enumerate() {
            if i < qft_result.phase_spectrum.len() {
                let phase = qft_result.phase_spectrum[i];
                
                // Check for interference patterns (phase relationships)
                if (phase % PI).abs() < 0.1 || ((phase % PI) - PI).abs() < 0.1 {
                    interference_frequencies.push(freq);
                }
            }
        }

        Ok(interference_frequencies)
    }

    async fn calculate_interference_strength(&self, interference_frequencies: &[f64]) -> Result<f64> {
        // Simple interference strength calculation based on number and spread of frequencies
        if interference_frequencies.is_empty() {
            return Ok(0.0);
        }

        let frequency_spread = if interference_frequencies.len() > 1 {
            let max_freq = interference_frequencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_freq = interference_frequencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            max_freq - min_freq
        } else {
            0.0
        };

        // Strength increases with number of interfering frequencies and their spread
        let strength = (interference_frequencies.len() as f64).sqrt() * (1.0 + frequency_spread);
        
        Ok(strength.min(1.0))
    }

    async fn identify_interference_regions(
        &self,
        interference_frequencies: &[f64],
    ) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        
        let mut constructive_regions = Vec::new();
        let mut destructive_regions = Vec::new();

        // Simple region identification based on frequency clustering
        if interference_frequencies.len() >= 2 {
            let mut sorted_freqs = interference_frequencies.to_vec();
            sorted_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for window in sorted_freqs.windows(2) {
                let freq_diff = window[1] - window[0];
                let region_center = (window[0] + window[1]) / 2.0;
                let region_width = freq_diff / 2.0;

                // Classify as constructive or destructive based on frequency difference
                if freq_diff < 0.01 { // Close frequencies = constructive
                    constructive_regions.push((region_center - region_width, region_center + region_width));
                } else { // Separated frequencies = destructive
                    destructive_regions.push((region_center - region_width, region_center + region_width));
                }
            }
        }

        Ok((constructive_regions, destructive_regions))
    }

    async fn calculate_coherence_length(&self, qft_result: &QuantumFourierResult) -> Result<f64> {
        // Calculate coherence length from spectral characteristics
        
        if qft_result.dominant_frequencies.is_empty() {
            return Ok(0.0);
        }

        // Coherence length is inversely related to spectral width
        let min_freq = qft_result.dominant_frequencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_freq = qft_result.dominant_frequencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let spectral_width = max_freq - min_freq;
        let coherence_length = if spectral_width > 0.0 {
            1.0 / spectral_width
        } else {
            f64::INFINITY
        };

        // Scale by spectral coherence
        Ok(coherence_length * qft_result.spectral_coherence)
    }
}

/// Quantum interference pattern analysis result
#[derive(Debug, Clone)]
pub struct QuantumInterferencePattern {
    pub interference_frequencies: Vec<f64>,
    pub interference_strength: f64,
    pub constructive_regions: Vec<(f64, f64)>,
    pub destructive_regions: Vec<(f64, f64)>,
    pub coherence_length: f64,
}

/// Initialize quantum Fourier transform system
pub async fn init() -> Result<()> {
    info!("Quantum Fourier Transform subsystem initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[tokio::test]
    async fn test_quantum_fft() {
        let config = QuantumConfig::default();
        let qft_engine = QuantumFourierTransform::new(&config).await.unwrap();

        // Create test signal
        let test_signal = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ]);

        let frequency_amplitudes = qft_engine.apply_quantum_fft(&test_signal).await.unwrap();
        
        assert_eq!(frequency_amplitudes.len(), test_signal.len());
        
        // Test inverse transform
        let reconstructed = qft_engine.inverse_transform(&QuantumFourierResult {
            frequency_amplitudes: frequency_amplitudes.clone(),
            phase_spectrum: Array1::zeros(frequency_amplitudes.len()),
            dominant_frequencies: vec![],
            frequency_entanglement: Array2::zeros((0, 0)),
            spectral_coherence: 0.5,
        }).await.unwrap();

        assert_eq!(reconstructed.len(), test_signal.len());
    }

    #[tokio::test]
    async fn test_dominant_frequency_detection() {
        let config = QuantumConfig::default();
        let qft_engine = QuantumFourierTransform::new(&config).await.unwrap();

        // Create signal with known frequency content
        let mut signal = Array1::zeros(64);
        for i in 0..64 {
            let phase = 2.0 * PI * i as f64 / 8.0; // Frequency component at 1/8 of sampling rate
            signal[i] = Complex64::new(phase.cos(), phase.sin());
        }

        let dominant_frequencies = qft_engine.identify_dominant_frequencies(&signal).await.unwrap();
        
        assert!(!dominant_frequencies.is_empty());
    }

    #[tokio::test]
    async fn test_spectral_coherence() {
        let config = QuantumConfig::default();
        let qft_engine = QuantumFourierTransform::new(&config).await.unwrap();

        // Test with coherent signal (single frequency)
        let coherent_signal = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let coherence1 = qft_engine.compute_spectral_coherence(&coherent_signal).await.unwrap();

        // Test with incoherent signal (random phases)
        let incoherent_signal = Array1::from_vec(vec![
            Complex64::new(0.25, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(0.25, 0.0),
        ]);

        let coherence2 = qft_engine.compute_spectral_coherence(&incoherent_signal).await.unwrap();

        // Coherent signal should have higher coherence
        assert!(coherence1 >= coherence2);
    }
}