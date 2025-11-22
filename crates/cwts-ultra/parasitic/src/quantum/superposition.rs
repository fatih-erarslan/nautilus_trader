//! # Quantum Superposition Operations
//! 
//! Quantum superposition state management and interference operations

use serde::{Serialize, Deserialize};
use num_complex::Complex64;
use std::collections::HashMap;
use crate::quantum::{QuantumMode, QuantumError};
use crate::{quantum_gate, if_quantum};

/// Superposition state manager
#[derive(Debug, Clone)]
pub struct SuperpositionManager {
    states: HashMap<String, SuperpositionState>,
    interference_cache: HashMap<String, InterferencePattern>,
    coherence_time: f64,
}

impl SuperpositionManager {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            interference_cache: HashMap::new(),
            coherence_time: 100.0, // microseconds
        }
    }
    
    /// Create superposition of multiple states
    pub fn create_superposition(&mut self, id: String, basis_states: Vec<BasisState>) -> Result<(), QuantumError> {
        if basis_states.is_empty() {
            return Err(QuantumError::Configuration("No basis states provided".to_string()));
        }
        
        let superposition = quantum_gate!(
            // Classical mode: probabilistic mixture
            SuperpositionState::create_classical(&basis_states)?,
            // Enhanced mode: quantum-inspired coherent superposition
            SuperpositionState::create_enhanced(&basis_states, self.coherence_time)?,
            // Full quantum mode: true quantum superposition
            SuperpositionState::create_quantum(&basis_states, self.coherence_time)?
        );
        
        self.states.insert(id, superposition);
        Ok(())
    }
    
    /// Apply quantum interference between superposition states
    pub fn create_interference(&mut self, state1_id: &str, state2_id: &str, interaction_strength: f64) -> Result<InterferencePattern, QuantumError> {
        let state1 = self.states.get(state1_id)
            .ok_or_else(|| QuantumError::Simulation(format!("State {} not found", state1_id)))?;
        
        let state2 = self.states.get(state2_id)
            .ok_or_else(|| QuantumError::Simulation(format!("State {} not found", state2_id)))?;
        
        let interference = quantum_gate!(
            // Classical: statistical interference
            InterferencePattern::classical_interference(state1, state2, interaction_strength),
            // Enhanced: quantum-inspired interference with coherence effects
            InterferencePattern::enhanced_interference(state1, state2, interaction_strength, self.coherence_time),
            // Full quantum: true quantum interference
            InterferencePattern::quantum_interference(state1, state2, interaction_strength, self.coherence_time)
        );
        
        let cache_key = format!("{}_{}", state1_id, state2_id);
        self.interference_cache.insert(cache_key, interference.clone());
        
        Ok(interference)
    }
    
    /// Measure superposition state
    pub fn measure_state(&self, id: &str) -> Option<MeasurementResult> {
        let state = self.states.get(id)?;
        Some(state.measure())
    }
    
    /// Apply decoherence to all states
    pub fn apply_decoherence(&mut self, time_step: f64) {
        let decoherence_factor = (-time_step / self.coherence_time).exp();
        
        for state in self.states.values_mut() {
            state.apply_decoherence(decoherence_factor);
        }
        
        // Clean up interference patterns that have decohered
        self.interference_cache.retain(|_, pattern| {
            pattern.coherence_factor > 0.01
        });
    }
    
    /// Create multi-state superposition for pattern matching
    pub fn create_pattern_superposition(&mut self, patterns: Vec<Vec<f64>>) -> Result<String, QuantumError> {
        let state_id = format!("pattern_superposition_{}", fastrand::u64(..));
        
        let mut basis_states = Vec::new();
        for (i, pattern) in patterns.iter().enumerate() {
            let amplitude = 1.0 / (patterns.len() as f64).sqrt(); // Equal superposition
            let phase = i as f64 * 2.0 * std::f64::consts::PI / patterns.len() as f64;
            
            basis_states.push(BasisState {
                id: format!("pattern_{}", i),
                amplitude: Complex64::new(amplitude * phase.cos(), amplitude * phase.sin()),
                classical_weight: 1.0 / patterns.len() as f64,
                data: pattern.clone(),
            });
        }
        
        self.create_superposition(state_id.clone(), basis_states)?;
        Ok(state_id)
    }
    
    /// Perform amplitude amplification (quantum search acceleration)
    pub fn amplitude_amplification(&mut self, state_id: &str, target_states: Vec<String>) -> Result<(), QuantumError> {
        let state = self.states.get_mut(state_id)
            .ok_or_else(|| QuantumError::Simulation(format!("State {} not found", state_id)))?;
        
        quantum_gate!(
            // Classical: boost probabilities of target states
            state.classical_amplification(&target_states),
            // Enhanced: quantum-inspired amplitude amplification with interference
            state.enhanced_amplification(&target_states, 1.5),
            // Full quantum: Grover-style amplitude amplification
            state.quantum_amplification(&target_states, self.coherence_time)
        )?;
        
        Ok(())
    }
    
    /// Get interference pattern between states
    pub fn get_interference(&self, state1_id: &str, state2_id: &str) -> Option<&InterferencePattern> {
        let cache_key = format!("{}_{}", state1_id, state2_id);
        self.interference_cache.get(&cache_key)
    }
    
    /// Clone and evolve superposition state
    pub fn evolve_superposition(&mut self, state_id: &str, evolution_time: f64) -> Result<(), QuantumError> {
        let state = self.states.get_mut(state_id)
            .ok_or_else(|| QuantumError::Simulation(format!("State {} not found", state_id)))?;
        
        state.time_evolution(evolution_time);
        Ok(())
    }
    
    /// Get state information
    pub fn get_state_info(&self, state_id: &str) -> Option<SuperpositionInfo> {
        let state = self.states.get(state_id)?;
        Some(SuperpositionInfo {
            num_basis_states: state.get_basis_count(),
            total_amplitude: state.get_total_amplitude(),
            coherence_measure: state.get_coherence(),
            classical_entropy: state.get_classical_entropy(),
            quantum_entropy: state.get_quantum_entropy(),
        })
    }
}

impl Default for SuperpositionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Superposition state representation
#[derive(Debug, Clone)]
pub struct SuperpositionState {
    basis_states: Vec<BasisState>,
    state_type: SuperpositionType,
    creation_time: std::time::Instant,
    coherence_factor: f64,
}

impl SuperpositionState {
    pub fn create_classical(basis_states: &[BasisState]) -> Result<Self, QuantumError> {
        // Normalize classical weights
        let total_weight: f64 = basis_states.iter().map(|s| s.classical_weight).sum();
        if total_weight <= 0.0 {
            return Err(QuantumError::Configuration("Invalid classical weights".to_string()));
        }
        
        let normalized_states: Vec<BasisState> = basis_states.iter()
            .map(|state| BasisState {
                id: state.id.clone(),
                amplitude: Complex64::new((state.classical_weight / total_weight).sqrt(), 0.0),
                classical_weight: state.classical_weight / total_weight,
                data: state.data.clone(),
            })
            .collect();
        
        Ok(Self {
            basis_states: normalized_states,
            state_type: SuperpositionType::Classical,
            creation_time: std::time::Instant::now(),
            coherence_factor: 1.0,
        })
    }
    
    pub fn create_enhanced(basis_states: &[BasisState], coherence_time: f64) -> Result<Self, QuantumError> {
        // Create quantum-inspired superposition with phase relationships
        let n = basis_states.len();
        let base_amplitude = 1.0 / (n as f64).sqrt();
        
        let enhanced_states: Vec<BasisState> = basis_states.iter().enumerate()
            .map(|(i, state)| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                let interference_factor = 1.0 + 0.2 * (phase * 2.0).cos(); // Add interference
                
                BasisState {
                    id: state.id.clone(),
                    amplitude: Complex64::new(
                        base_amplitude * interference_factor * phase.cos(),
                        base_amplitude * interference_factor * phase.sin(),
                    ),
                    classical_weight: state.classical_weight,
                    data: state.data.clone(),
                }
            })
            .collect();
        
        Ok(Self {
            basis_states: enhanced_states,
            state_type: SuperpositionType::Enhanced { coherence_time },
            creation_time: std::time::Instant::now(),
            coherence_factor: 1.0,
        })
    }
    
    pub fn create_quantum(basis_states: &[BasisState], coherence_time: f64) -> Result<Self, QuantumError> {
        // Create true quantum superposition with proper normalization
        let n = basis_states.len();
        
        // Calculate normalization factor
        let total_amplitude_sq: f64 = basis_states.iter()
            .map(|s| s.amplitude.norm_sqr())
            .sum();
        
        let norm_factor = if total_amplitude_sq > 0.0 {
            1.0 / total_amplitude_sq.sqrt()
        } else {
            1.0 / (n as f64).sqrt()
        };
        
        let quantum_states: Vec<BasisState> = basis_states.iter()
            .map(|state| BasisState {
                id: state.id.clone(),
                amplitude: if total_amplitude_sq > 0.0 {
                    state.amplitude * norm_factor
                } else {
                    Complex64::new(1.0 / (n as f64).sqrt(), 0.0)
                },
                classical_weight: state.classical_weight,
                data: state.data.clone(),
            })
            .collect();
        
        Ok(Self {
            basis_states: quantum_states,
            state_type: SuperpositionType::Quantum { 
                coherence_time, 
                entanglement_degree: 0.0 
            },
            creation_time: std::time::Instant::now(),
            coherence_factor: 1.0,
        })
    }
    
    /// Measure the superposition state
    pub fn measure(&self) -> MeasurementResult {
        match &self.state_type {
            SuperpositionType::Classical => self.classical_measurement(),
            SuperpositionType::Enhanced { .. } => self.enhanced_measurement(),
            SuperpositionType::Quantum { .. } => self.quantum_measurement(),
        }
    }
    
    fn classical_measurement(&self) -> MeasurementResult {
        let rand_val = fastrand::f64();
        let mut cumulative_prob = 0.0;
        
        for state in &self.basis_states {
            cumulative_prob += state.classical_weight;
            if rand_val <= cumulative_prob {
                return MeasurementResult {
                    measured_state: state.id.clone(),
                    probability: state.classical_weight,
                    measurement_type: MeasurementType::Classical,
                    collapse_fidelity: 1.0,
                };
            }
        }
        
        // Fallback to last state
        let last_state = &self.basis_states[self.basis_states.len() - 1];
        MeasurementResult {
            measured_state: last_state.id.clone(),
            probability: last_state.classical_weight,
            measurement_type: MeasurementType::Classical,
            collapse_fidelity: 1.0,
        }
    }
    
    fn enhanced_measurement(&self) -> MeasurementResult {
        let probabilities: Vec<f64> = self.basis_states.iter()
            .map(|s| s.amplitude.norm_sqr() * self.coherence_factor)
            .collect();
        
        let total_prob: f64 = probabilities.iter().sum();
        let rand_val = fastrand::f64() * total_prob;
        let mut cumulative_prob = 0.0;
        
        for (i, state) in self.basis_states.iter().enumerate() {
            cumulative_prob += probabilities[i];
            if rand_val <= cumulative_prob {
                return MeasurementResult {
                    measured_state: state.id.clone(),
                    probability: probabilities[i] / total_prob,
                    measurement_type: MeasurementType::Enhanced,
                    collapse_fidelity: self.coherence_factor,
                };
            }
        }
        
        // Fallback
        let last_state = &self.basis_states[self.basis_states.len() - 1];
        MeasurementResult {
            measured_state: last_state.id.clone(),
            probability: probabilities[probabilities.len() - 1] / total_prob,
            measurement_type: MeasurementType::Enhanced,
            collapse_fidelity: self.coherence_factor,
        }
    }
    
    fn quantum_measurement(&self) -> MeasurementResult {
        // Born rule: |amplitude|² gives measurement probability
        let probabilities: Vec<f64> = self.basis_states.iter()
            .map(|s| s.amplitude.norm_sqr())
            .collect();
        
        let rand_val = fastrand::f64();
        let mut cumulative_prob = 0.0;
        
        for (i, state) in self.basis_states.iter().enumerate() {
            cumulative_prob += probabilities[i];
            if rand_val <= cumulative_prob {
                return MeasurementResult {
                    measured_state: state.id.clone(),
                    probability: probabilities[i],
                    measurement_type: MeasurementType::Quantum,
                    collapse_fidelity: self.coherence_factor,
                };
            }
        }
        
        // Fallback
        let last_state = &self.basis_states[self.basis_states.len() - 1];
        MeasurementResult {
            measured_state: last_state.id.clone(),
            probability: probabilities[probabilities.len() - 1],
            measurement_type: MeasurementType::Quantum,
            collapse_fidelity: self.coherence_factor,
        }
    }
    
    /// Apply decoherence to the state
    pub fn apply_decoherence(&mut self, decoherence_factor: f64) {
        self.coherence_factor *= decoherence_factor;
        
        match &self.state_type {
            SuperpositionType::Enhanced { .. } | SuperpositionType::Quantum { .. } => {
                // Reduce off-diagonal coherences
                for state in &mut self.basis_states {
                    state.amplitude *= decoherence_factor;
                    
                    // Add some classical mixture
                    let classical_component = Complex64::new(
                        (state.classical_weight * (1.0 - decoherence_factor)).sqrt(),
                        0.0,
                    );
                    state.amplitude = state.amplitude * decoherence_factor + classical_component;
                }
            }
            SuperpositionType::Classical => {
                // Classical states don't decohere in the same way
            }
        }
    }
    
    /// Time evolution of the superposition
    pub fn time_evolution(&mut self, time: f64) {
        match &self.state_type {
            SuperpositionType::Quantum { .. } | SuperpositionType::Enhanced { .. } => {
                for (i, state) in self.basis_states.iter_mut().enumerate() {
                    // Apply time evolution with individual energy levels
                    let energy = i as f64 + 1.0; // Simple energy assignment
                    let phase_factor = Complex64::new(0.0, -energy * time).exp();
                    state.amplitude *= phase_factor;
                }
            }
            SuperpositionType::Classical => {
                // Classical states don't evolve in phase
            }
        }
    }
    
    /// Amplitude amplification operations
    pub fn classical_amplification(&mut self, target_states: &[String]) -> Result<(), QuantumError> {
        for state in &mut self.basis_states {
            if target_states.contains(&state.id) {
                state.classical_weight *= 2.0; // Boost target states
            }
        }
        
        // Renormalize
        let total_weight: f64 = self.basis_states.iter().map(|s| s.classical_weight).sum();
        for state in &mut self.basis_states {
            state.classical_weight /= total_weight;
        }
        
        Ok(())
    }
    
    pub fn enhanced_amplification(&mut self, target_states: &[String], boost_factor: f64) -> Result<(), QuantumError> {
        for state in &mut self.basis_states {
            if target_states.contains(&state.id) {
                state.amplitude *= boost_factor;
            } else {
                // Slight reduction for non-target states
                state.amplitude *= (1.0 / boost_factor).sqrt();
            }
        }
        
        // Renormalize amplitudes
        let total_amplitude_sq: f64 = self.basis_states.iter()
            .map(|s| s.amplitude.norm_sqr())
            .sum();
        let norm_factor = 1.0 / total_amplitude_sq.sqrt();
        
        for state in &mut self.basis_states {
            state.amplitude *= norm_factor;
        }
        
        Ok(())
    }
    
    pub fn quantum_amplification(&mut self, target_states: &[String], coherence_time: f64) -> Result<(), QuantumError> {
        // Grover-style amplitude amplification
        let n = self.basis_states.len();
        let target_count = target_states.len();
        
        if target_count == 0 || target_count >= n {
            return Ok(()); // No amplification needed
        }
        
        // Optimal number of iterations for Grover's algorithm
        let optimal_iterations = ((std::f64::consts::PI / 4.0) * (n as f64 / target_count as f64).sqrt()) as u32;
        
        for _ in 0..optimal_iterations.min(10) { // Limit iterations to prevent over-rotation
            // Oracle: flip sign of target states
            for state in &mut self.basis_states {
                if target_states.contains(&state.id) {
                    state.amplitude = -state.amplitude;
                }
            }
            
            // Diffusion operator: invert about average
            let average_amplitude = self.basis_states.iter()
                .map(|s| s.amplitude)
                .sum::<Complex64>() / n as f64;
            
            for state in &mut self.basis_states {
                state.amplitude = 2.0 * average_amplitude - state.amplitude;
            }
        }
        
        // Apply decoherence due to operations
        let decoherence_factor = (-optimal_iterations as f64 * 0.01 / coherence_time).exp();
        self.apply_decoherence(decoherence_factor);
        
        Ok(())
    }
    
    // Utility methods
    pub fn get_basis_count(&self) -> usize {
        self.basis_states.len()
    }
    
    pub fn get_total_amplitude(&self) -> Complex64 {
        self.basis_states.iter().map(|s| s.amplitude).sum()
    }
    
    pub fn get_coherence(&self) -> f64 {
        self.coherence_factor
    }
    
    pub fn get_classical_entropy(&self) -> f64 {
        let probabilities: Vec<f64> = self.basis_states.iter()
            .map(|s| s.classical_weight)
            .collect();
        
        -probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }
    
    pub fn get_quantum_entropy(&self) -> f64 {
        let probabilities: Vec<f64> = self.basis_states.iter()
            .map(|s| s.amplitude.norm_sqr())
            .collect();
        
        -probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>()
    }
}

/// Individual basis state in superposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisState {
    pub id: String,
    pub amplitude: Complex64,
    pub classical_weight: f64,
    pub data: Vec<f64>,
}

/// Types of superposition states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuperpositionType {
    Classical,
    Enhanced { coherence_time: f64 },
    Quantum { coherence_time: f64, entanglement_degree: f64 },
}

/// Interference pattern between superposition states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferencePattern {
    pub constructive_regions: Vec<InterferenceRegion>,
    pub destructive_regions: Vec<InterferenceRegion>,
    pub coherence_factor: f64,
    pub pattern_type: InterferenceType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceRegion {
    pub amplitude_boost: f64,
    pub phase_shift: f64,
    pub spatial_extent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterferenceType {
    Classical { correlation_strength: f64 },
    Enhanced { interference_visibility: f64 },
    Quantum { entanglement_contribution: f64 },
}

impl InterferencePattern {
    pub fn classical_interference(state1: &SuperpositionState, state2: &SuperpositionState, strength: f64) -> Self {
        let mut constructive = Vec::new();
        let mut destructive = Vec::new();
        
        // Simple classical correlation-based interference
        for (i, s1) in state1.basis_states.iter().enumerate() {
            for (j, s2) in state2.basis_states.iter().enumerate() {
                let correlation = calculate_data_correlation(&s1.data, &s2.data);
                
                if correlation > 0.5 {
                    constructive.push(InterferenceRegion {
                        amplitude_boost: correlation * strength,
                        phase_shift: 0.0,
                        spatial_extent: 1.0,
                    });
                } else if correlation < -0.5 {
                    destructive.push(InterferenceRegion {
                        amplitude_boost: correlation.abs() * strength,
                        phase_shift: std::f64::consts::PI,
                        spatial_extent: 1.0,
                    });
                }
            }
        }
        
        Self {
            constructive_regions: constructive,
            destructive_regions: destructive,
            coherence_factor: strength,
            pattern_type: InterferenceType::Classical { correlation_strength: strength },
        }
    }
    
    pub fn enhanced_interference(state1: &SuperpositionState, state2: &SuperpositionState, strength: f64, coherence_time: f64) -> Self {
        let mut constructive = Vec::new();
        let mut destructive = Vec::new();
        
        // Quantum-inspired interference with phase relationships
        for s1 in &state1.basis_states {
            for s2 in &state2.basis_states {
                let amplitude_product = s1.amplitude * s2.amplitude.conj();
                let phase_diff = amplitude_product.arg();
                let amplitude_correlation = amplitude_product.norm();
                
                let interference_strength = amplitude_correlation * strength;
                
                if phase_diff.cos() > 0.0 {
                    constructive.push(InterferenceRegion {
                        amplitude_boost: interference_strength * phase_diff.cos(),
                        phase_shift: phase_diff,
                        spatial_extent: coherence_time / 100.0,
                    });
                } else {
                    destructive.push(InterferenceRegion {
                        amplitude_boost: interference_strength * phase_diff.cos().abs(),
                        phase_shift: phase_diff,
                        spatial_extent: coherence_time / 100.0,
                    });
                }
            }
        }
        
        let visibility = if constructive.len() + destructive.len() > 0 {
            constructive.len() as f64 / (constructive.len() + destructive.len()) as f64
        } else {
            0.0
        };
        
        Self {
            constructive_regions: constructive,
            destructive_regions: destructive,
            coherence_factor: strength * (-1.0 / coherence_time).exp(),
            pattern_type: InterferenceType::Enhanced { interference_visibility: visibility },
        }
    }
    
    pub fn quantum_interference(state1: &SuperpositionState, state2: &SuperpositionState, strength: f64, coherence_time: f64) -> Self {
        // True quantum interference based on amplitude overlap
        let mut pattern = Self::enhanced_interference(state1, state2, strength, coherence_time);
        
        // Calculate entanglement contribution
        let entanglement = calculate_entanglement_measure(state1, state2);
        
        // Modify interference based on entanglement
        for region in &mut pattern.constructive_regions {
            region.amplitude_boost *= 1.0 + entanglement * 0.5;
        }
        
        for region in &mut pattern.destructive_regions {
            region.amplitude_boost *= 1.0 + entanglement * 0.3;
        }
        
        pattern.pattern_type = InterferenceType::Quantum { entanglement_contribution: entanglement };
        
        pattern
    }
}

/// Measurement result from superposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    pub measured_state: String,
    pub probability: f64,
    pub measurement_type: MeasurementType,
    pub collapse_fidelity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementType {
    Classical,
    Enhanced,
    Quantum,
}

/// Superposition state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionInfo {
    pub num_basis_states: usize,
    pub total_amplitude: Complex64,
    pub coherence_measure: f64,
    pub classical_entropy: f64,
    pub quantum_entropy: f64,
}

// Helper functions

fn calculate_data_correlation(data1: &[f64], data2: &[f64]) -> f64 {
    if data1.len() != data2.len() || data1.is_empty() {
        return 0.0;
    }
    
    let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
    let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;
    
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    
    for (x, y) in data1.iter().zip(data2.iter()) {
        let diff1 = x - mean1;
        let diff2 = y - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    
    let denominator = (sum_sq1 * sum_sq2).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn calculate_entanglement_measure(state1: &SuperpositionState, state2: &SuperpositionState) -> f64 {
    // Simplified entanglement measure based on amplitude correlations
    let mut total_correlation = 0.0;
    let mut count = 0;
    
    for s1 in &state1.basis_states {
        for s2 in &state2.basis_states {
            let amplitude_product = (s1.amplitude * s2.amplitude.conj()).norm();
            total_correlation += amplitude_product;
            count += 1;
        }
    }
    
    if count > 0 {
        (total_correlation / count as f64).min(1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::QuantumMode;
    
    #[test]
    fn test_superposition_manager_creation() {
        let manager = SuperpositionManager::new();
        assert!(manager.states.is_empty());
        assert!(manager.interference_cache.is_empty());
    }
    
    #[test]
    fn test_create_superposition() {
        let mut manager = SuperpositionManager::new();
        
        let basis_states = vec![
            BasisState {
                id: "state_0".to_string(),
                amplitude: Complex64::new(0.7, 0.0),
                classical_weight: 0.5,
                data: vec![1.0, 0.0],
            },
            BasisState {
                id: "state_1".to_string(),
                amplitude: Complex64::new(0.7, 0.0),
                classical_weight: 0.5,
                data: vec![0.0, 1.0],
            },
        ];
        
        let result = manager.create_superposition("test_state".to_string(), basis_states);
        assert!(result.is_ok());
        assert!(manager.states.contains_key("test_state"));
    }
    
    #[test]
    fn test_measure_superposition() {
        let basis_states = vec![
            BasisState {
                id: "state_0".to_string(),
                amplitude: Complex64::new(1.0, 0.0),
                classical_weight: 1.0,
                data: vec![1.0],
            },
        ];
        
        let state = SuperpositionState::create_classical(&basis_states).unwrap();
        let measurement = state.measure();
        
        assert_eq!(measurement.measured_state, "state_0");
        assert!((measurement.probability - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_quantum_superposition_creation() {
        let basis_states = vec![
            BasisState {
                id: "state_0".to_string(),
                amplitude: Complex64::new(0.7, 0.0),
                classical_weight: 0.5,
                data: vec![1.0, 0.0],
            },
            BasisState {
                id: "state_1".to_string(),
                amplitude: Complex64::new(0.0, 0.7),
                classical_weight: 0.5,
                data: vec![0.0, 1.0],
            },
        ];
        
        let state = SuperpositionState::create_quantum(&basis_states, 100.0).unwrap();
        assert_eq!(state.basis_states.len(), 2);
        
        // Check normalization
        let total_prob: f64 = state.basis_states.iter()
            .map(|s| s.amplitude.norm_sqr())
            .sum();
        assert!((total_prob - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_interference_pattern() {
        let basis_states1 = vec![
            BasisState {
                id: "state_0".to_string(),
                amplitude: Complex64::new(0.7, 0.0),
                classical_weight: 1.0,
                data: vec![1.0, 0.0],
            },
        ];
        
        let basis_states2 = vec![
            BasisState {
                id: "state_1".to_string(),
                amplitude: Complex64::new(0.7, 0.0),
                classical_weight: 1.0,
                data: vec![1.0, 0.0], // Same data should create constructive interference
            },
        ];
        
        let state1 = SuperpositionState::create_enhanced(&basis_states1, 100.0).unwrap();
        let state2 = SuperpositionState::create_enhanced(&basis_states2, 100.0).unwrap();
        
        let interference = InterferencePattern::enhanced_interference(&state1, &state2, 0.8, 100.0);
        
        assert!(interference.coherence_factor > 0.0);
        // Should have some interference patterns due to identical data
        assert!(!interference.constructive_regions.is_empty() || !interference.destructive_regions.is_empty());
    }
    
    #[test]
    fn test_amplitude_amplification() {
        let basis_states = vec![
            BasisState {
                id: "target".to_string(),
                amplitude: Complex64::new(0.5, 0.0),
                classical_weight: 0.25,
                data: vec![1.0],
            },
            BasisState {
                id: "other1".to_string(),
                amplitude: Complex64::new(0.5, 0.0),
                classical_weight: 0.25,
                data: vec![0.0],
            },
            BasisState {
                id: "other2".to_string(),
                amplitude: Complex64::new(0.5, 0.0),
                classical_weight: 0.25,
                data: vec![0.0],
            },
            BasisState {
                id: "other3".to_string(),
                amplitude: Complex64::new(0.5, 0.0),
                classical_weight: 0.25,
                data: vec![0.0],
            },
        ];
        
        let mut state = SuperpositionState::create_quantum(&basis_states, 100.0).unwrap();
        let target_states = vec!["target".to_string()];
        
        let result = state.quantum_amplification(&target_states, 100.0);
        assert!(result.is_ok());
        
        // Target state should have higher amplitude after amplification
        let target_state = state.basis_states.iter()
            .find(|s| s.id == "target")
            .unwrap();
        let other_state = state.basis_states.iter()
            .find(|s| s.id == "other1")
            .unwrap();
        
        // After Grover amplification, target should have higher probability
        assert!(target_state.amplitude.norm_sqr() >= other_state.amplitude.norm_sqr());
    }
    
    #[test]
    fn test_decoherence_effects() {
        let basis_states = vec![
            BasisState {
                id: "state_0".to_string(),
                amplitude: Complex64::new(0.7, 0.7),
                classical_weight: 0.5,
                data: vec![1.0],
            },
            BasisState {
                id: "state_1".to_string(),
                amplitude: Complex64::new(0.0, 0.7),
                classical_weight: 0.5,
                data: vec![0.0],
            },
        ];
        
        let mut state = SuperpositionState::create_quantum(&basis_states, 100.0).unwrap();
        let initial_coherence = state.get_coherence();
        
        state.apply_decoherence(0.9);
        let final_coherence = state.get_coherence();
        
        assert!(final_coherence < initial_coherence);
    }
    
    #[test]
    fn test_pattern_superposition() {
        let mut manager = SuperpositionManager::new();
        let patterns = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let state_id = manager.create_pattern_superposition(patterns).unwrap();
        assert!(manager.states.contains_key(&state_id));
        
        let state_info = manager.get_state_info(&state_id).unwrap();
        assert_eq!(state_info.num_basis_states, 3);
    }
}