//! Utility functions for quantum enhanced pattern recognition

use crate::types::*;
use crate::Result;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use chrono::{DateTime, Utc};
use serde_json;

/// Quantum state utilities
pub struct QuantumUtils;

impl QuantumUtils {
    /// Convert classical price data to quantum amplitudes
    pub fn price_to_quantum_amplitude(price: f64, base_price: f64) -> Complex64 {
        let relative_change = (price / base_price - 1.0).clamp(-1.0, 1.0);
        let magnitude = relative_change.abs().sqrt();
        let phase = if relative_change >= 0.0 { 0.0 } else { PI };
        
        Complex64::new(magnitude * phase.cos(), magnitude * phase.sin())
    }

    /// Convert quantum amplitude back to price prediction
    pub fn quantum_amplitude_to_price(amplitude: Complex64, base_price: f64) -> f64 {
        let magnitude = amplitude.norm();
        let phase = amplitude.arg();
        
        let relative_change = magnitude.powi(2);
        let direction = if phase.abs() < PI / 2.0 { 1.0 } else { -1.0 };
        
        base_price * (1.0 + direction * relative_change)
    }

    /// Calculate quantum fidelity between two states
    pub fn calculate_fidelity(state1: &Array1<Complex64>, state2: &Array1<Complex64>) -> f64 {
        if state1.len() != state2.len() || state1.is_empty() {
            return 0.0;
        }

        let overlap: Complex64 = state1.iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        overlap.norm_sqr()
    }

    /// Normalize quantum state to unit length
    pub fn normalize_quantum_state(state: &mut Array1<Complex64>) {
        let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        
        if norm_sq > 1e-10 {
            let norm = norm_sq.sqrt();
            *state /= norm;
        }
    }

    /// Calculate von Neumann entropy of quantum state
    pub fn von_neumann_entropy(density_matrix: &Array2<Complex64>) -> f64 {
        if density_matrix.nrows() != density_matrix.ncols() {
            return 0.0;
        }

        // For pure states, we can calculate entropy from diagonal elements
        let mut entropy = 0.0;
        for i in 0..density_matrix.nrows() {
            let eigenvalue = density_matrix[[i, i]].norm_sqr();
            if eigenvalue > 1e-10 {
                entropy -= eigenvalue * eigenvalue.ln();
            }
        }

        entropy
    }

    /// Generate Bell states for entanglement
    pub fn generate_bell_states() -> Vec<Array1<Complex64>> {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        
        vec![
            // |Φ+⟩ = (|00⟩ + |11⟩)/√2
            Array1::from_vec(vec![
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
            ]),
            // |Φ-⟩ = (|00⟩ - |11⟩)/√2
            Array1::from_vec(vec![
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-sqrt2_inv, 0.0),
            ]),
            // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
            Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(-sqrt2_inv, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
        ]
    }

    /// Apply quantum noise to state (decoherence simulation)
    pub fn apply_quantum_noise(state: &mut Array1<Complex64>, noise_level: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for amplitude in state.iter_mut() {
            let noise_real = rng.gen::<f64>() * noise_level * 2.0 - noise_level;
            let noise_imag = rng.gen::<f64>() * noise_level * 2.0 - noise_level;
            
            *amplitude += Complex64::new(noise_real, noise_imag);
        }
        
        // Renormalize after adding noise
        Self::normalize_quantum_state(state);
    }

    /// Calculate quantum discord (measure of quantum correlations)
    pub fn quantum_discord(joint_state: &Array2<Complex64>) -> f64 {
        // Simplified quantum discord calculation
        // In practice, this would require more sophisticated eigenvalue decomposition
        
        let von_neumann_joint = Self::von_neumann_entropy(joint_state);
        
        // Calculate classical correlation (simplified)
        let mut classical_correlation = 0.0;
        for i in 0..joint_state.nrows() {
            for j in 0..joint_state.ncols() {
                let prob = joint_state[[i, j]].norm_sqr();
                if prob > 1e-10 {
                    classical_correlation += prob * prob.ln();
                }
            }
        }
        
        (von_neumann_joint + classical_correlation).max(0.0)
    }
}

/// Signal processing utilities
pub struct SignalUtils;

impl SignalUtils {
    /// Calculate rolling statistics for time series
    pub fn rolling_statistics(data: &[f64], window: usize) -> Vec<(f64, f64)> {
        let mut results = Vec::new();
        
        for i in window..=data.len() {
            let window_data = &data[i-window..i];
            let mean = window_data.iter().sum::<f64>() / window as f64;
            let variance = window_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / window as f64;
            
            results.push((mean, variance.sqrt()));
        }
        
        results
    }

    /// Detect regime changes in time series
    pub fn detect_regime_changes(data: &[f64], threshold: f64) -> Vec<usize> {
        let mut change_points = Vec::new();
        
        if data.len() < 10 {
            return change_points;
        }

        let rolling_stats = Self::rolling_statistics(data, 5);
        
        for i in 1..rolling_stats.len() {
            let prev_vol = rolling_stats[i-1].1;
            let curr_vol = rolling_stats[i].1;
            
            let vol_change = (curr_vol - prev_vol).abs() / prev_vol.max(1e-6);
            
            if vol_change > threshold {
                change_points.push(i + 4); // Adjust for window offset
            }
        }
        
        change_points
    }

    /// Calculate correlation matrix with confidence intervals
    pub fn correlation_matrix_with_confidence(
        data: &HashMap<String, Vec<f64>>,
        confidence_level: f64,
    ) -> HashMap<(String, String), (f64, f64, f64)> {
        let mut correlations = HashMap::new();
        let instruments: Vec<String> = data.keys().cloned().collect();
        
        for i in 0..instruments.len() {
            for j in (i+1)..instruments.len() {
                let series1 = &data[&instruments[i]];
                let series2 = &data[&instruments[j]];
                
                if let Some((correlation, lower_bound, upper_bound)) = 
                    Self::pearson_correlation_with_confidence(series1, series2, confidence_level) {
                    
                    correlations.insert(
                        (instruments[i].clone(), instruments[j].clone()),
                        (correlation, lower_bound, upper_bound)
                    );
                }
            }
        }
        
        correlations
    }

    /// Calculate Pearson correlation with confidence interval
    fn pearson_correlation_with_confidence(
        x: &[f64], 
        y: &[f64], 
        confidence_level: f64
    ) -> Option<(f64, f64, f64)> {
        if x.len() != y.len() || x.len() < 3 {
            return None;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        let correlation = sum_xy / (sum_x2 * sum_y2).sqrt();
        
        // Fisher transformation for confidence interval
        let fisher_z = 0.5 * ((1.0 + correlation) / (1.0 - correlation)).ln();
        let se = 1.0 / (n - 3.0).sqrt();
        
        // Z-score for confidence level (approximation)
        let z_score = match confidence_level {
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };
        
        let lower_z = fisher_z - z_score * se;
        let upper_z = fisher_z + z_score * se;
        
        let lower_bound = (lower_z.exp() * 2.0 - 1.0) / (lower_z.exp() * 2.0 + 1.0);
        let upper_bound = (upper_z.exp() * 2.0 - 1.0) / (upper_z.exp() * 2.0 + 1.0);
        
        Some((correlation, lower_bound, upper_bound))
    }

    /// Detect anomalies using statistical methods
    pub fn detect_anomalies(data: &[f64], sensitivity: f64) -> Vec<(usize, f64)> {
        let mut anomalies = Vec::new();
        
        if data.len() < 10 {
            return anomalies;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        let threshold = sensitivity * std_dev;
        
        for (i, &value) in data.iter().enumerate() {
            let deviation = (value - mean).abs();
            if deviation > threshold {
                anomalies.push((i, deviation / std_dev));
            }
        }
        
        anomalies
    }
}

/// Configuration utilities
pub struct ConfigUtils;

impl ConfigUtils {
    /// Load quantum configuration from JSON
    pub fn load_quantum_config_from_json(json_str: &str) -> Result<QuantumConfig> {
        serde_json::from_str(json_str)
            .map_err(|e| crate::QuantumError::Config(format!("JSON parsing error: {}", e)))
    }

    /// Save quantum configuration to JSON
    pub fn save_quantum_config_to_json(config: &QuantumConfig) -> Result<String> {
        serde_json::to_string_pretty(config)
            .map_err(|e| crate::QuantumError::Config(format!("JSON serialization error: {}", e)))
    }

    /// Validate quantum configuration
    pub fn validate_quantum_config(config: &QuantumConfig) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        if config.max_superposition_states > 2048 {
            warnings.push("Large number of superposition states may impact performance".to_string());
        }

        if config.coherence_threshold < 0.1 {
            warnings.push("Very low coherence threshold may produce unreliable signals".to_string());
        }

        if config.entanglement_sensitivity > 0.9 {
            warnings.push("Very high entanglement sensitivity may produce false positives".to_string());
        }

        if config.performance.target_latency_us < 50 {
            warnings.push("Extremely low latency target may be unrealistic".to_string());
        }

        Ok(warnings)
    }

    /// Generate optimal quantum configuration for given constraints
    pub fn generate_optimal_config(
        target_latency_us: u64,
        max_memory_mb: usize,
        accuracy_preference: f64, // 0.0 = speed, 1.0 = accuracy
    ) -> QuantumConfig {
        let max_states = if target_latency_us < 100 {
            256
        } else if target_latency_us < 500 {
            512
        } else {
            1024
        };

        let coherence_threshold = 0.5 + 0.3 * accuracy_preference;
        let entanglement_sensitivity = 0.3 + 0.4 * accuracy_preference;

        QuantumConfig {
            max_superposition_states: max_states.min(max_memory_mb * 2),
            coherence_threshold,
            entanglement_sensitivity,
            frequency_resolution: if accuracy_preference > 0.7 { 0.0005 } else { 0.001 },
            claude_flow: ClaudeFlowQuantumConfig {
                enable_swarm: true,
                quantum_agents: if accuracy_preference > 0.5 { 6 } else { 4 },
                memory_namespace: "quantum-patterns".to_string(),
                real_time_sharing: target_latency_us < 200,
            },
            performance: QuantumPerformanceConfig {
                target_latency_us,
                max_concurrent_calculations: if target_latency_us < 100 { 4 } else { 8 },
                memory_pool_size_mb: max_memory_mb,
                enable_gpu: max_memory_mb > 1024,
            },
            simd: QuantumSimdConfig {
                enable_simd: true,
                vector_width: 8,
                parallel_threads: num_cpus::get(),
            },
        }
    }
}

/// Formatting utilities
pub struct FormatUtils;

impl FormatUtils {
    /// Format quantum signal for display
    pub fn format_quantum_signal(signal: &QuantumSignal) -> String {
        format!(
            "🔮 Quantum Signal [{}]\n\
             ├─ Pattern: {}\n\
             ├─ Strength: {:.3} | Confidence: {:.3} | Coherence: {:.3}\n\
             ├─ Instruments: {}\n\
             ├─ Entanglement: {} pairs\n\
             ├─ Execution: {}μs\n\
             └─ Timestamp: {}",
            signal.id,
            signal.pattern_type,
            signal.strength,
            signal.confidence,
            signal.coherence,
            signal.affected_instruments.join(", "),
            signal.entanglement_map.len(),
            signal.execution_time_us,
            signal.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    /// Format performance metrics for display
    pub fn format_performance_metrics(metrics: &QuantumPerformanceMetrics) -> String {
        format!(
            "📊 Quantum Performance Metrics\n\
             ├─ Latency: {:.1}μs (avg) | Target: <100μs\n\
             ├─ Success Rate: {:.1}% | Coherence: {:.1}%\n\
             ├─ Throughput: {:.1} patterns/sec\n\
             ├─ Memory Efficiency: {:.1}%\n\
             ├─ CPU Usage: {:.1}%{}\n\
             ├─ Total Calculations: {}\n\
             └─ Error Rate: {:.2}%",
            metrics.avg_detection_latency_us,
            metrics.detection_success_rate * 100.0,
            metrics.coherence_preservation_rate * 100.0,
            metrics.patterns_per_second,
            metrics.memory_efficiency * 100.0,
            metrics.cpu_utilization,
            metrics.gpu_utilization
                .map(|gpu| format!(" | GPU: {:.1}%", gpu))
                .unwrap_or_default(),
            metrics.total_calculations,
            metrics.quantum_error_rate * 100.0
        )
    }

    /// Format entanglement correlation for display
    pub fn format_entanglement_correlation(entanglement: &EntanglementCorrelation) -> String {
        format!(
            "🔗 Quantum Entanglement Analysis\n\
             ├─ Strength: {:.3} | Fidelity: {:.3}\n\
             ├─ Entangled Pairs: {}\n\
             ├─ Decoherence Rate: {:.3}/sec\n\
             └─ Bell Coefficients: {} states",
            entanglement.strength,
            entanglement.fidelity,
            entanglement.entangled_pairs.len(),
            entanglement.decoherence_rate,
            entanglement.bell_coefficients.len()
        )
    }

    /// Create timestamp for filenames
    pub fn timestamp_filename() -> String {
        Utc::now().format("%Y%m%d_%H%M%S").to_string()
    }

    /// Format duration in human-readable format
    pub fn format_duration_us(microseconds: u64) -> String {
        if microseconds < 1_000 {
            format!("{}μs", microseconds)
        } else if microseconds < 1_000_000 {
            format!("{:.1}ms", microseconds as f64 / 1_000.0)
        } else {
            format!("{:.2}s", microseconds as f64 / 1_000_000.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_quantum_amplitude_conversion() {
        let base_price = 50000.0;
        let test_price = 51000.0;
        
        let amplitude = QuantumUtils::price_to_quantum_amplitude(test_price, base_price);
        let recovered_price = QuantumUtils::quantum_amplitude_to_price(amplitude, base_price);
        
        // Should be approximately equal (within numerical precision)
        assert!((recovered_price - test_price).abs() < base_price * 0.01);
    }

    #[test]
    fn test_quantum_fidelity() {
        let state1 = Array1::from_vec(vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
        ]);
        
        let state2 = Array1::from_vec(vec![
            Complex64::new(0.8, 0.0),
            Complex64::new(0.6, 0.0),
        ]);
        
        let fidelity = QuantumUtils::calculate_fidelity(&state1, &state2);
        assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }

    #[test]
    fn test_bell_states() {
        let bell_states = QuantumUtils::generate_bell_states();
        assert_eq!(bell_states.len(), 4);
        
        // Each Bell state should be normalized
        for state in &bell_states {
            let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
            assert!((norm_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rolling_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = SignalUtils::rolling_statistics(&data, 3);
        
        assert!(!stats.is_empty());
        // First window: [1, 2, 3] -> mean = 2.0
        assert!((stats[0].0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_detection() {
        let mut data = vec![1.0; 10]; // Stable period
        data.extend(vec![5.0; 10]);   // Regime change
        
        let changes = SignalUtils::detect_regime_changes(&data, 0.5);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_correlation_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        
        let result = SignalUtils::pearson_correlation_with_confidence(&x, &y, 0.95);
        assert!(result.is_some());
        
        let (correlation, _lower, _upper) = result.unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut data = vec![1.0; 20]; // Normal data
        data.push(10.0); // Anomaly
        
        let anomalies = SignalUtils::detect_anomalies(&data, 2.0);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].0, 20); // Anomaly at index 20
    }

    #[test]
    fn test_config_validation() {
        let config = QuantumConfig::default();
        let warnings = ConfigUtils::validate_quantum_config(&config).unwrap();
        
        // Default config should have minimal warnings
        assert!(warnings.len() <= 2);
    }

    #[test]
    fn test_optimal_config_generation() {
        let config = ConfigUtils::generate_optimal_config(50, 1024, 0.8);
        
        assert_eq!(config.performance.target_latency_us, 50);
        assert!(config.coherence_threshold > 0.7); // High accuracy preference
        assert!(config.performance.enable_gpu); // Large memory available
    }

    #[test]
    fn test_signal_formatting() {
        let signal = QuantumSignal::new(0.8, 0.9, QuantumPatternType::SuperpositionMomentum, 0.85);
        let formatted = FormatUtils::format_quantum_signal(&signal);
        
        assert!(formatted.contains("Quantum Signal"));
        assert!(formatted.contains("Superposition Momentum"));
        assert!(formatted.contains("0.800")); // Strength
        assert!(formatted.contains("0.900")); // Confidence
    }
}