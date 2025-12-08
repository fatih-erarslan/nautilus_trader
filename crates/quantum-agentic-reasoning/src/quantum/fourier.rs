//! Quantum Fourier Transform implementations
//!
//! This module provides advanced QFT implementations including:
//! - Standard QFT
//! - Inverse QFT
//! - Approximate QFT
//! - Windowed QFT for large systems

use crate::core::{QarResult, QarError, constants};
use crate::quantum::{QuantumState, Gate, StandardGates};
use std::f64::consts::PI;

/// Quantum Fourier Transform implementation
#[derive(Debug, Clone)]
pub struct QuantumFourierTransform {
    /// Number of qubits
    num_qubits: usize,
    /// Whether to use approximate QFT
    approximate: bool,
    /// Approximation threshold for angle rotations
    angle_threshold: f64,
}

impl QuantumFourierTransform {
    /// Create a new QFT instance
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            approximate: false,
            angle_threshold: 1e-10,
        }
    }

    /// Enable approximate QFT with given threshold
    pub fn with_approximation(mut self, threshold: f64) -> Self {
        self.approximate = true;
        self.angle_threshold = threshold;
        self
    }

    /// Apply QFT to quantum state
    pub fn apply_qft(&self, state: &mut QuantumState) -> QarResult<()> {
        if state.num_qubits != self.num_qubits {
            return Err(QarError::QuantumError(
                "State and QFT qubit count mismatch".to_string()
            ));
        }

        self.apply_qft_recursive(state, 0)?;
        self.reverse_qubits(state)?;
        
        Ok(())
    }

    /// Apply inverse QFT to quantum state
    pub fn apply_inverse_qft(&self, state: &mut QuantumState) -> QarResult<()> {
        if state.num_qubits != self.num_qubits {
            return Err(QarError::QuantumError(
                "State and QFT qubit count mismatch".to_string()
            ));
        }

        self.reverse_qubits(state)?;
        self.apply_inverse_qft_recursive(state, 0)?;
        
        Ok(())
    }

    /// Recursive QFT implementation
    fn apply_qft_recursive(&self, state: &mut QuantumState, start_qubit: usize) -> QarResult<()> {
        if start_qubit >= self.num_qubits {
            return Ok(());
        }

        // Apply Hadamard to current qubit
        let h_gate = StandardGates::hadamard();
        state.apply_single_qubit_gate(start_qubit, &h_gate)?;

        // Apply controlled phase gates
        for control_qubit in (start_qubit + 1)..self.num_qubits {
            let phase_power = control_qubit - start_qubit;
            let angle = 2.0 * PI / (2.0_f64.powi(phase_power as i32));
            
            // Skip small angles in approximate mode
            if self.approximate && angle.abs() < self.angle_threshold {
                continue;
            }

            let cp_gate = StandardGates::cphase(angle);
            state.apply_two_qubit_gate(control_qubit, start_qubit, &cp_gate)?;
        }

        // Recursively apply to remaining qubits
        self.apply_qft_recursive(state, start_qubit + 1)
    }

    /// Recursive inverse QFT implementation
    fn apply_inverse_qft_recursive(&self, state: &mut QuantumState, start_qubit: usize) -> QarResult<()> {
        if start_qubit >= self.num_qubits {
            return Ok(());
        }

        // Recursively apply to remaining qubits first
        self.apply_inverse_qft_recursive(state, start_qubit + 1)?;

        // Apply controlled phase gates in reverse order
        for control_qubit in ((start_qubit + 1)..self.num_qubits).rev() {
            let phase_power = control_qubit - start_qubit;
            let angle = -2.0 * PI / (2.0_f64.powi(phase_power as i32)); // Negative for inverse
            
            // Skip small angles in approximate mode
            if self.approximate && angle.abs() < self.angle_threshold {
                continue;
            }

            let cp_gate = StandardGates::cphase(angle);
            state.apply_two_qubit_gate(control_qubit, start_qubit, &cp_gate)?;
        }

        // Apply Hadamard to current qubit
        let h_gate = StandardGates::hadamard();
        state.apply_single_qubit_gate(start_qubit, &h_gate)?;

        Ok(())
    }

    /// Reverse qubit order by applying SWAP gates
    fn reverse_qubits(&self, state: &mut QuantumState) -> QarResult<()> {
        for i in 0..(self.num_qubits / 2) {
            let j = self.num_qubits - 1 - i;
            let swap_gate = StandardGates::swap();
            state.apply_two_qubit_gate(i, j, &swap_gate)?;
        }
        Ok(())
    }

    /// Extract frequency domain information
    pub fn extract_frequencies(&self, state: &QuantumState) -> Vec<f64> {
        let probabilities = state.probabilities();
        let n = probabilities.len();
        
        // Convert probability amplitudes to frequency domain
        let mut frequencies = Vec::with_capacity(n);
        for (k, &prob) in probabilities.iter().enumerate() {
            let frequency = k as f64 / n as f64;
            frequencies.push(frequency * prob.sqrt()); // Use amplitude, not probability
        }
        
        frequencies
    }

    /// Get the most significant frequencies
    pub fn get_dominant_frequencies(&self, state: &QuantumState, count: usize) -> Vec<(usize, f64)> {
        let probabilities = state.probabilities();
        let mut freq_probs: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        
        // Sort by probability (descending)
        freq_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top frequencies
        freq_probs.into_iter().take(count).collect()
    }

    /// Calculate spectral power density
    pub fn calculate_power_spectral_density(&self, state: &QuantumState) -> Vec<f64> {
        let probabilities = state.probabilities();
        let n = probabilities.len();
        
        // Calculate power spectral density
        probabilities.iter()
            .enumerate()
            .map(|(k, &prob)| {
                if k == 0 || k == n / 2 {
                    prob // DC and Nyquist components
                } else {
                    2.0 * prob // Double-sided spectrum
                }
            })
            .collect()
    }

    /// Detect periodic components in the signal
    pub fn detect_periodicities(&self, state: &QuantumState, threshold: f64) -> Vec<(f64, f64)> {
        let power_spectrum = self.calculate_power_spectral_density(state);
        let n = power_spectrum.len();
        let mut periodicities = Vec::new();
        
        for (k, &power) in power_spectrum.iter().enumerate().skip(1) {
            if power > threshold {
                let frequency = k as f64 / n as f64;
                let period = if frequency > 0.0 { 1.0 / frequency } else { f64::INFINITY };
                periodicities.push((frequency, period));
            }
        }
        
        // Sort by power (strongest first)
        periodicities.sort_by(|a, b| {
            let power_a = power_spectrum[(a.0 * n as f64) as usize];
            let power_b = power_spectrum[(b.0 * n as f64) as usize];
            power_b.partial_cmp(&power_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        periodicities
    }
}

/// Windowed Quantum Fourier Transform for large systems
#[derive(Debug, Clone)]
pub struct WindowedQFT {
    /// Total number of qubits
    total_qubits: usize,
    /// Window size
    window_size: usize,
    /// Overlap between windows
    overlap: usize,
}

impl WindowedQFT {
    /// Create a new windowed QFT
    pub fn new(total_qubits: usize, window_size: usize, overlap: usize) -> QarResult<Self> {
        if window_size >= total_qubits {
            return Err(QarError::InvalidInput(
                "Window size must be smaller than total qubits".to_string()
            ));
        }
        
        if overlap >= window_size {
            return Err(QarError::InvalidInput(
                "Overlap must be smaller than window size".to_string()
            ));
        }

        Ok(Self {
            total_qubits,
            window_size,
            overlap,
        })
    }

    /// Apply windowed QFT
    pub fn apply_windowed_qft(&self, state: &mut QuantumState) -> QarResult<Vec<Vec<f64>>> {
        if state.num_qubits != self.total_qubits {
            return Err(QarError::QuantumError(
                "State size doesn't match total qubits".to_string()
            ));
        }

        let mut results = Vec::new();
        let step = self.window_size - self.overlap;
        let mut start = 0;

        while start + self.window_size <= self.total_qubits {
            // Extract window
            let window_state = self.extract_window(state, start, self.window_size)?;
            
            // Apply QFT to window
            let mut windowed_state = window_state;
            let qft = QuantumFourierTransform::new(self.window_size);
            qft.apply_qft(&mut windowed_state)?;
            
            // Extract frequency information
            let frequencies = qft.extract_frequencies(&windowed_state);
            results.push(frequencies);
            
            start += step;
        }

        Ok(results)
    }

    /// Extract a window from the quantum state
    fn extract_window(&self, state: &QuantumState, start: usize, size: usize) -> QarResult<QuantumState> {
        // For simplicity, we'll create a new state with the window qubits
        // In a full implementation, this would involve partial tracing
        let mut window_state = QuantumState::new(size);
        
        // Copy relevant amplitudes (simplified approach)
        let window_size = 1 << size;
        for i in 0..window_size {
            if i < state.amplitudes.len() {
                window_state.amplitudes[i] = state.amplitudes[i];
            }
        }
        
        window_state.normalize();
        Ok(window_state)
    }

    /// Reconstruct spectrum from windowed results
    pub fn reconstruct_spectrum(&self, windowed_results: &[Vec<f64>]) -> Vec<f64> {
        if windowed_results.is_empty() {
            return Vec::new();
        }

        let window_freq_size = windowed_results[0].len();
        let total_freq_size = self.total_qubits * window_freq_size / self.window_size;
        let mut spectrum = vec![0.0; total_freq_size];
        let mut counts = vec![0; total_freq_size];

        let step = self.window_size - self.overlap;
        
        for (window_idx, window_freqs) in windowed_results.iter().enumerate() {
            let start_freq = window_idx * step * window_freq_size / self.window_size;
            
            for (i, &freq_val) in window_freqs.iter().enumerate() {
                let global_freq_idx = start_freq + i;
                if global_freq_idx < spectrum.len() {
                    spectrum[global_freq_idx] += freq_val;
                    counts[global_freq_idx] += 1;
                }
            }
        }

        // Average overlapping regions
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                spectrum[i] /= *count as f64;
            }
        }

        spectrum
    }
}

/// Quantum Fast Fourier Transform optimizations
#[derive(Debug)]
pub struct OptimizedQFT {
    /// Number of qubits
    num_qubits: usize,
    /// Use parallel gate execution
    parallel_execution: bool,
    /// Cache rotation gates
    cached_gates: std::collections::HashMap<String, Gate>,
}

impl OptimizedQFT {
    /// Create a new optimized QFT
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            parallel_execution: false,
            cached_gates: std::collections::HashMap::new(),
        }
    }

    /// Enable parallel gate execution
    pub fn with_parallel_execution(mut self) -> Self {
        self.parallel_execution = true;
        self
    }

    /// Get or create cached rotation gate
    fn get_rotation_gate(&mut self, angle: f64) -> Gate {
        let key = format!("phase_{:.10}", angle);
        
        if let Some(gate) = self.cached_gates.get(&key) {
            gate.clone()
        } else {
            let gate = StandardGates::cphase(angle);
            self.cached_gates.insert(key, gate.clone());
            gate
        }
    }

    /// Apply optimized QFT
    pub fn apply_optimized_qft(&mut self, state: &mut QuantumState) -> QarResult<()> {
        if state.num_qubits != self.num_qubits {
            return Err(QarError::QuantumError(
                "State and QFT qubit count mismatch".to_string()
            ));
        }

        // Pre-compute all rotation angles
        let mut rotation_angles = Vec::new();
        for j in 0..self.num_qubits {
            for k in (j + 1)..self.num_qubits {
                let phase_power = k - j;
                let angle = 2.0 * PI / (2.0_f64.powi(phase_power as i32));
                rotation_angles.push((j, k, angle));
            }
        }

        // Apply QFT with optimizations
        for j in 0..self.num_qubits {
            // Apply Hadamard gate
            let h_gate = StandardGates::hadamard();
            state.apply_single_qubit_gate(j, &h_gate)?;

            // Apply controlled phase gates
            for k in (j + 1)..self.num_qubits {
                let phase_power = k - j;
                let angle = 2.0 * PI / (2.0_f64.powi(phase_power as i32));
                
                let cp_gate = self.get_rotation_gate(angle);
                state.apply_two_qubit_gate(k, j, &cp_gate)?;
            }
        }

        // Apply SWAP gates to reverse qubit order
        for i in 0..(self.num_qubits / 2) {
            let swap_gate = StandardGates::swap();
            state.apply_two_qubit_gate(i, self.num_qubits - 1 - i, &swap_gate)?;
        }

        Ok(())
    }

    /// Clear gate cache
    pub fn clear_cache(&mut self) {
        self.cached_gates.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cached_gates.len(), self.cached_gates.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_qft_creation() {
        let qft = QuantumFourierTransform::new(3);
        assert_eq!(qft.num_qubits, 3);
        assert!(!qft.approximate);

        let approx_qft = QuantumFourierTransform::new(4).with_approximation(1e-6);
        assert!(approx_qft.approximate);
        assert_eq!(approx_qft.angle_threshold, 1e-6);
    }

    #[test]
    fn test_qft_application() {
        let qft = QuantumFourierTransform::new(2);
        let mut state = QuantumState::new(2);
        
        // Apply QFT
        let result = qft.apply_qft(&mut state);
        assert!(result.is_ok());
        
        // State should still be normalized
        assert!(state.is_normalized());
    }

    #[test]
    fn test_inverse_qft() {
        let qft = QuantumFourierTransform::new(2);
        let mut state = QuantumState::new(2);
        let original_state = state.clone();
        
        // Apply QFT then inverse QFT
        qft.apply_qft(&mut state).unwrap();
        qft.apply_inverse_qft(&mut state).unwrap();
        
        // Should return to original state (approximately)
        for (i, (orig, final_amp)) in original_state.amplitudes.iter()
            .zip(state.amplitudes.iter()).enumerate() {
            assert_relative_eq!(orig.re, final_amp.re, epsilon = 1e-10, 
                               "Real part mismatch at index {}", i);
            assert_relative_eq!(orig.im, final_amp.im, epsilon = 1e-10,
                               "Imaginary part mismatch at index {}", i);
        }
    }

    #[test]
    fn test_frequency_extraction() {
        let qft = QuantumFourierTransform::new(2);
        let state = QuantumState::superposition(2);
        
        let frequencies = qft.extract_frequencies(&state);
        assert_eq!(frequencies.len(), 4);
        
        // All frequencies should be equal for superposition
        for &freq in &frequencies {
            assert_relative_eq!(freq, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dominant_frequencies() {
        let qft = QuantumFourierTransform::new(2);
        let state = QuantumState::new(2); // |00⟩ state
        
        let dominant = qft.get_dominant_frequencies(&state, 2);
        assert_eq!(dominant.len(), 2);
        
        // First frequency should be dominant (DC component)
        assert_eq!(dominant[0].0, 0);
        assert_eq!(dominant[0].1, 1.0);
    }

    #[test]
    fn test_power_spectral_density() {
        let qft = QuantumFourierTransform::new(2);
        let state = QuantumState::new(2);
        
        let psd = qft.calculate_power_spectral_density(&state);
        assert_eq!(psd.len(), 4);
        
        // Only DC component should have power
        assert_eq!(psd[0], 1.0);
        for i in 1..psd.len() {
            assert_eq!(psd[i], 0.0);
        }
    }

    #[test]
    fn test_windowed_qft_creation() {
        let result = WindowedQFT::new(8, 4, 2);
        assert!(result.is_ok());
        
        let windowed_qft = result.unwrap();
        assert_eq!(windowed_qft.total_qubits, 8);
        assert_eq!(windowed_qft.window_size, 4);
        assert_eq!(windowed_qft.overlap, 2);
    }

    #[test]
    fn test_windowed_qft_invalid_parameters() {
        // Window size >= total qubits
        let result = WindowedQFT::new(4, 4, 1);
        assert!(result.is_err());
        
        // Overlap >= window size
        let result = WindowedQFT::new(8, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimized_qft() {
        let mut opt_qft = OptimizedQFT::new(3).with_parallel_execution();
        let mut state = QuantumState::new(3);
        
        let result = opt_qft.apply_optimized_qft(&mut state);
        assert!(result.is_ok());
        assert!(state.is_normalized());
        
        // Check cache was used
        let (cache_size, _) = opt_qft.cache_stats();
        assert!(cache_size > 0);
    }

    #[test]
    fn test_cache_management() {
        let mut opt_qft = OptimizedQFT::new(3);
        let mut state = QuantumState::new(3);
        
        // Apply QFT to populate cache
        opt_qft.apply_optimized_qft(&mut state).unwrap();
        let (size_before, _) = opt_qft.cache_stats();
        assert!(size_before > 0);
        
        // Clear cache
        opt_qft.clear_cache();
        let (size_after, _) = opt_qft.cache_stats();
        assert_eq!(size_after, 0);
    }

    #[test]
    fn test_periodicity_detection() {
        let qft = QuantumFourierTransform::new(3);
        let state = QuantumState::superposition(3);
        
        let periodicities = qft.detect_periodicities(&state, 0.1);
        
        // Should detect multiple periodicities in superposition
        assert!(!periodicities.is_empty());
        
        // All detected periodicities should have valid frequencies
        for (freq, period) in periodicities {
            assert!(freq >= 0.0 && freq <= 1.0);
            assert!(period > 0.0);
        }
    }
}