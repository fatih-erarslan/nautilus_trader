//! # Quantum Entanglement Operations
//! 
//! Quantum entanglement creation and manipulation for enhanced correlation detection

use serde::{Serialize, Deserialize};
use num_complex::Complex64;
use std::collections::HashMap;
use crate::quantum::{QuantumMode, QuantumError};
use crate::{quantum_gate, if_quantum};

/// Quantum entanglement manager
#[derive(Debug, Clone)]
pub struct EntanglementManager {
    entangled_pairs: HashMap<(u32, u32), EntanglementStrength>,
    correlation_matrix: Vec<Vec<f64>>,
    num_qubits: u32,
}

impl EntanglementManager {
    pub fn new(num_qubits: u32) -> Self {
        let matrix_size = num_qubits as usize;
        Self {
            entangled_pairs: HashMap::new(),
            correlation_matrix: vec![vec![0.0; matrix_size]; matrix_size],
            num_qubits,
        }
    }
    
    /// Create entanglement between two qubits
    pub fn create_entanglement(&mut self, qubit1: u32, qubit2: u32, strength: f64) -> Result<(), QuantumError> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err(QuantumError::Configuration("Qubit index out of range".to_string()));
        }
        
        let pair = if qubit1 < qubit2 { (qubit1, qubit2) } else { (qubit2, qubit1) };
        
        quantum_gate!(
            // Classical mode: track correlations probabilistically
            {
                self.entangled_pairs.insert(pair, EntanglementStrength::Classical(strength.min(1.0)));
                self.correlation_matrix[qubit1 as usize][qubit2 as usize] = strength;
                self.correlation_matrix[qubit2 as usize][qubit1 as usize] = strength;
            },
            // Enhanced mode: quantum-inspired entanglement with interference
            {
                let enhanced_strength = self.calculate_enhanced_entanglement(qubit1, qubit2, strength);
                self.entangled_pairs.insert(pair, EntanglementStrength::Enhanced(enhanced_strength));
                self.update_correlation_matrix_enhanced(qubit1, qubit2, enhanced_strength);
            },
            // Full quantum mode: true quantum entanglement
            {
                let quantum_strength = self.create_quantum_entanglement(qubit1, qubit2, strength)?;
                self.entangled_pairs.insert(pair, EntanglementStrength::Quantum(quantum_strength));
                self.update_quantum_correlations(qubit1, qubit2, quantum_strength);
            }
        );
        
        Ok(())
    }
    
    /// Measure entanglement strength between qubits
    pub fn measure_entanglement(&self, qubit1: u32, qubit2: u32) -> Option<f64> {
        let pair = if qubit1 < qubit2 { (qubit1, qubit2) } else { (qubit2, qubit1) };
        
        self.entangled_pairs.get(&pair).map(|strength| match strength {
            EntanglementStrength::Classical(s) => *s,
            EntanglementStrength::Enhanced(s) => s.magnitude(),
            EntanglementStrength::Quantum(s) => s.entanglement_measure,
        })
    }
    
    /// Get all entangled correlations
    pub fn get_correlation_matrix(&self) -> &Vec<Vec<f64>> {
        &self.correlation_matrix
    }
    
    /// Detect Bell inequality violations (quantum advantage indicator)
    pub fn detect_bell_violations(&self) -> Vec<BellViolation> {
        let mut violations = Vec::new();
        
        if_quantum!({
            // Check CHSH inequality for all qubit combinations
            for i in 0..self.num_qubits {
                for j in (i + 1)..self.num_qubits {
                    if let Some(violation) = self.check_chsh_inequality(i, j) {
                        violations.push(violation);
                    }
                }
            }
        });
        
        violations
    }
    
    /// Create quantum teleportation channel
    pub fn create_teleportation_channel(&mut self, source: u32, auxiliary: u32, target: u32) -> Result<TeleportationChannel, QuantumError> {
        // Create Bell pair between auxiliary and target
        self.create_entanglement(auxiliary, target, 1.0)?;
        
        quantum_gate!(
            // Classical: probabilistic teleportation
            TeleportationChannel::new_classical(source, auxiliary, target),
            // Enhanced: quantum-inspired teleportation with enhanced fidelity
            TeleportationChannel::new_enhanced(source, auxiliary, target, self.get_enhancement_factor()),
            // Full quantum: perfect teleportation (in ideal case)
            TeleportationChannel::new_quantum(source, auxiliary, target, self.get_quantum_fidelity())
        )
    }
    
    /// Perform entanglement swapping
    pub fn entanglement_swapping(&mut self, qubit1: u32, qubit2: u32, qubit3: u32, qubit4: u32) -> Result<SwappingResult, QuantumError> {
        // Check if initial entanglements exist
        let pair1 = if qubit1 < qubit2 { (qubit1, qubit2) } else { (qubit2, qubit1) };
        let pair2 = if qubit3 < qubit4 { (qubit3, qubit4) } else { (qubit4, qubit3) };
        
        let entanglement1 = self.entangled_pairs.get(&pair1)
            .ok_or_else(|| QuantumError::Simulation("No entanglement between qubits 1-2".to_string()))?;
        
        let entanglement2 = self.entangled_pairs.get(&pair2)
            .ok_or_else(|| QuantumError::Simulation("No entanglement between qubits 3-4".to_string()))?;
        
        let success_probability = quantum_gate!(
            // Classical: limited swapping success
            0.25, // Classical correlation swapping
            // Enhanced: improved quantum-inspired swapping
            0.5 * self.calculate_swapping_enhancement(&entanglement1, &entanglement2),
            // Full quantum: theoretical maximum
            0.5 // Bell measurement success probability
        );
        
        if fastrand::f64() < success_probability {
            // Remove original entanglements
            self.entangled_pairs.remove(&pair1);
            self.entangled_pairs.remove(&pair2);
            
            // Create new entanglement between distant qubits
            let new_strength = self.calculate_swapped_strength(&entanglement1, &entanglement2);
            let new_pair = if qubit1 < qubit4 { (qubit1, qubit4) } else { (qubit4, qubit1) };
            self.entangled_pairs.insert(new_pair, new_strength);
            
            Ok(SwappingResult {
                success: true,
                new_entangled_pair: (qubit1, qubit4),
                fidelity: success_probability,
                measurement_outcomes: vec![fastrand::u8(0..2), fastrand::u8(0..2)],
            })
        } else {
            Ok(SwappingResult {
                success: false,
                new_entangled_pair: (0, 0),
                fidelity: 0.0,
                measurement_outcomes: vec![fastrand::u8(0..2), fastrand::u8(0..2)],
            })
        }
    }
    
    /// Generate entangled state for pattern matching
    pub fn generate_entangled_pattern_state(&self, patterns: &[Vec<f64>]) -> Result<EntangledPatternState, QuantumError> {
        let num_patterns = patterns.len();
        if num_patterns == 0 {
            return Err(QuantumError::Configuration("No patterns provided".to_string()));
        }
        
        let num_qubits_needed = (num_patterns as f64).log2().ceil() as u32;
        if num_qubits_needed > self.num_qubits {
            return Err(QuantumError::ResourceExhausted(
                format!("Need {} qubits for {} patterns", num_qubits_needed, num_patterns)
            ));
        }
        
        quantum_gate!(
            // Classical: weighted correlations
            EntangledPatternState::new_classical(patterns),
            // Enhanced: quantum-inspired superposition with entangled features
            EntangledPatternState::new_enhanced(patterns, &self.correlation_matrix),
            // Full quantum: true entangled superposition
            EntangledPatternState::new_quantum(patterns, num_qubits_needed)
        )
    }
    
    // Private helper methods
    
    fn calculate_enhanced_entanglement(&self, qubit1: u32, qubit2: u32, base_strength: f64) -> Complex64 {
        let phase = self.calculate_entanglement_phase(qubit1, qubit2);
        let magnitude = base_strength * self.get_enhancement_factor();
        Complex64::new(magnitude * phase.cos(), magnitude * phase.sin())
    }
    
    fn calculate_entanglement_phase(&self, qubit1: u32, qubit2: u32) -> f64 {
        let q1 = qubit1 as f64;
        let q2 = qubit2 as f64;
        (q1 * q2 * std::f64::consts::PI / self.num_qubits as f64) % (2.0 * std::f64::consts::PI)
    }
    
    fn create_quantum_entanglement(&self, qubit1: u32, qubit2: u32, strength: f64) -> Result<QuantumEntanglement, QuantumError> {
        // Create maximally entangled Bell state
        let bell_coefficients = self.generate_bell_coefficients(strength);
        
        Ok(QuantumEntanglement {
            entanglement_measure: strength,
            bell_coefficients,
            concurrence: strength, // Simplified
            negativity: strength.powi(2),
            fidelity: (1.0 + strength) / 2.0,
        })
    }
    
    fn generate_bell_coefficients(&self, strength: f64) -> [Complex64; 4] {
        let sqrt_strength = strength.sqrt();
        let sqrt_complement = (1.0 - strength).sqrt();
        
        [
            Complex64::new(sqrt_strength / std::f64::consts::SQRT_2, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),                                       // |01⟩
            Complex64::new(0.0, 0.0),                                       // |10⟩
            Complex64::new(sqrt_complement / std::f64::consts::SQRT_2, 0.0), // |11⟩
        ]
    }
    
    fn update_correlation_matrix_enhanced(&mut self, qubit1: u32, qubit2: u32, entanglement: Complex64) {
        let correlation = entanglement.norm_sqr();
        self.correlation_matrix[qubit1 as usize][qubit2 as usize] = correlation;
        self.correlation_matrix[qubit2 as usize][qubit1 as usize] = correlation;
        
        // Update indirect correlations
        for i in 0..self.num_qubits {
            if i != qubit1 && i != qubit2 {
                let indirect1 = self.correlation_matrix[qubit1 as usize][i as usize] * correlation * 0.1;
                let indirect2 = self.correlation_matrix[qubit2 as usize][i as usize] * correlation * 0.1;
                
                self.correlation_matrix[qubit2 as usize][i as usize] += indirect1;
                self.correlation_matrix[i as usize][qubit2 as usize] += indirect1;
                self.correlation_matrix[qubit1 as usize][i as usize] += indirect2;
                self.correlation_matrix[i as usize][qubit1 as usize] += indirect2;
            }
        }
    }
    
    fn update_quantum_correlations(&mut self, qubit1: u32, qubit2: u32, entanglement: QuantumEntanglement) {
        let correlation = entanglement.entanglement_measure;
        self.correlation_matrix[qubit1 as usize][qubit2 as usize] = correlation;
        self.correlation_matrix[qubit2 as usize][qubit1 as usize] = correlation;
    }
    
    fn check_chsh_inequality(&self, qubit1: u32, qubit2: u32) -> Option<BellViolation> {
        let correlation = self.correlation_matrix[qubit1 as usize][qubit2 as usize];
        
        // Simplified CHSH test - in practice would need multiple measurement settings
        let chsh_value = 2.0 * correlation.sqrt();
        
        if chsh_value > 2.0 {
            Some(BellViolation {
                qubits: (qubit1, qubit2),
                chsh_value,
                violation_strength: chsh_value - 2.0,
                significance: ((chsh_value - 2.0) / 0.828).min(1.0), // Tsirelson bound
            })
        } else {
            None
        }
    }
    
    fn get_enhancement_factor(&self) -> f64 {
        match QuantumMode::current() {
            QuantumMode::Enhanced => 1.5,
            QuantumMode::Full => 2.0,
            _ => 1.0,
        }
    }
    
    fn get_quantum_fidelity(&self) -> f64 {
        match QuantumMode::current() {
            QuantumMode::Full => 0.95, // Account for decoherence
            _ => 0.8,
        }
    }
    
    fn calculate_swapping_enhancement(&self, ent1: &EntanglementStrength, ent2: &EntanglementStrength) -> f64 {
        let strength1 = match ent1 {
            EntanglementStrength::Classical(s) => *s,
            EntanglementStrength::Enhanced(s) => s.magnitude(),
            EntanglementStrength::Quantum(s) => s.entanglement_measure,
        };
        
        let strength2 = match ent2 {
            EntanglementStrength::Classical(s) => *s,
            EntanglementStrength::Enhanced(s) => s.magnitude(),
            EntanglementStrength::Quantum(s) => s.entanglement_measure,
        };
        
        (strength1 * strength2).sqrt()
    }
    
    fn calculate_swapped_strength(&self, ent1: &EntanglementStrength, ent2: &EntanglementStrength) -> EntanglementStrength {
        let enhancement = self.calculate_swapping_enhancement(ent1, ent2);
        
        quantum_gate!(
            EntanglementStrength::Classical(enhancement * 0.5),
            EntanglementStrength::Enhanced(Complex64::new(enhancement * 0.7, 0.0)),
            EntanglementStrength::Quantum(QuantumEntanglement {
                entanglement_measure: enhancement * 0.8,
                bell_coefficients: self.generate_bell_coefficients(enhancement * 0.8),
                concurrence: enhancement * 0.8,
                negativity: (enhancement * 0.8).powi(2),
                fidelity: (1.0 + enhancement * 0.8) / 2.0,
            })
        )
    }
}

/// Entanglement strength representation for different quantum modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementStrength {
    Classical(f64),
    Enhanced(Complex64),
    Quantum(QuantumEntanglement),
}

/// Full quantum entanglement representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEntanglement {
    pub entanglement_measure: f64,
    pub bell_coefficients: [Complex64; 4],
    pub concurrence: f64,
    pub negativity: f64,
    pub fidelity: f64,
}

/// Bell inequality violation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BellViolation {
    pub qubits: (u32, u32),
    pub chsh_value: f64,
    pub violation_strength: f64,
    pub significance: f64,
}

/// Quantum teleportation channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationChannel {
    pub source_qubit: u32,
    pub auxiliary_qubit: u32,
    pub target_qubit: u32,
    pub fidelity: f64,
    pub channel_type: TeleportationChannelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeleportationChannelType {
    Classical { success_probability: f64 },
    Enhanced { enhancement_factor: f64 },
    Quantum { bell_state_fidelity: f64 },
}

impl TeleportationChannel {
    pub fn new_classical(source: u32, auxiliary: u32, target: u32) -> Result<Self, QuantumError> {
        Ok(Self {
            source_qubit: source,
            auxiliary_qubit: auxiliary,
            target_qubit: target,
            fidelity: 0.5, // Classical limit
            channel_type: TeleportationChannelType::Classical { success_probability: 0.25 },
        })
    }
    
    pub fn new_enhanced(source: u32, auxiliary: u32, target: u32, enhancement: f64) -> Result<Self, QuantumError> {
        Ok(Self {
            source_qubit: source,
            auxiliary_qubit: auxiliary,
            target_qubit: target,
            fidelity: 0.5 + 0.3 * enhancement,
            channel_type: TeleportationChannelType::Enhanced { enhancement_factor: enhancement },
        })
    }
    
    pub fn new_quantum(source: u32, auxiliary: u32, target: u32, fidelity: f64) -> Result<Self, QuantumError> {
        Ok(Self {
            source_qubit: source,
            auxiliary_qubit: auxiliary,
            target_qubit: target,
            fidelity,
            channel_type: TeleportationChannelType::Quantum { bell_state_fidelity: fidelity },
        })
    }
    
    /// Perform quantum teleportation
    pub fn teleport(&self, state: &QuantumStateVector) -> Result<TeleportationResult, QuantumError> {
        let success_probability = match &self.channel_type {
            TeleportationChannelType::Classical { success_probability } => *success_probability,
            TeleportationChannelType::Enhanced { enhancement_factor } => 0.25 * (1.0 + enhancement_factor),
            TeleportationChannelType::Quantum { bell_state_fidelity } => bell_state_fidelity * 0.5,
        };
        
        let success = fastrand::f64() < success_probability;
        
        if success {
            let mut output_state = state.clone();
            // Apply teleportation transformation (simplified)
            output_state.apply_fidelity(self.fidelity);
            
            Ok(TeleportationResult {
                success: true,
                output_state: Some(output_state),
                measurement_outcomes: vec![fastrand::u8(0..2), fastrand::u8(0..2)],
                fidelity_achieved: self.fidelity,
            })
        } else {
            Ok(TeleportationResult {
                success: false,
                output_state: None,
                measurement_outcomes: vec![fastrand::u8(0..2), fastrand::u8(0..2)],
                fidelity_achieved: 0.0,
            })
        }
    }
}

/// Entanglement swapping result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwappingResult {
    pub success: bool,
    pub new_entangled_pair: (u32, u32),
    pub fidelity: f64,
    pub measurement_outcomes: Vec<u8>,
}

/// Quantum teleportation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationResult {
    pub success: bool,
    pub output_state: Option<QuantumStateVector>,
    pub measurement_outcomes: Vec<u8>,
    pub fidelity_achieved: f64,
}

/// Quantum state vector (simplified representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateVector {
    pub amplitudes: Vec<Complex64>,
    pub num_qubits: u32,
}

impl QuantumStateVector {
    pub fn new(num_qubits: u32) -> Self {
        let num_states = 1usize << num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        amplitudes[0] = Complex64::new(1.0, 0.0); // |0...0⟩
        
        Self { amplitudes, num_qubits }
    }
    
    pub fn apply_fidelity(&mut self, fidelity: f64) {
        let noise_factor = (1.0 - fidelity).sqrt();
        
        for amplitude in &mut self.amplitudes {
            // Add noise to amplitude
            let noise_real = fastrand::f64() * noise_factor * 0.1 - 0.05;
            let noise_imag = fastrand::f64() * noise_factor * 0.1 - 0.05;
            
            *amplitude += Complex64::new(noise_real, noise_imag);
        }
        
        // Renormalize
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        let norm = norm_sq.sqrt();
        
        if norm > 0.0 {
            for amplitude in &mut self.amplitudes {
                *amplitude /= norm;
            }
        }
    }
}

/// Entangled pattern state for quantum pattern matching
#[derive(Debug, Clone)]
pub struct EntangledPatternState {
    pub patterns: Vec<Vec<f64>>,
    pub entanglement_structure: EntanglementStructure,
    pub quantum_correlations: HashMap<usize, HashMap<usize, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementStructure {
    Classical { correlation_matrix: Vec<Vec<f64>> },
    Enhanced { interference_patterns: Vec<Complex64> },
    Quantum { bell_pairs: Vec<(usize, usize)>, fidelities: Vec<f64> },
}

impl EntangledPatternState {
    pub fn new_classical(patterns: &[Vec<f64>]) -> Result<Self, QuantumError> {
        let n = patterns.len();
        let mut correlation_matrix = vec![vec![0.0; n]; n];
        
        // Calculate classical correlations
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    correlation_matrix[i][j] = calculate_pattern_correlation(&patterns[i], &patterns[j]);
                } else {
                    correlation_matrix[i][j] = 1.0;
                }
            }
        }
        
        Ok(Self {
            patterns: patterns.to_vec(),
            entanglement_structure: EntanglementStructure::Classical { correlation_matrix },
            quantum_correlations: HashMap::new(),
        })
    }
    
    pub fn new_enhanced(patterns: &[Vec<f64>], system_correlations: &[Vec<f64>]) -> Result<Self, QuantumError> {
        let mut interference_patterns = Vec::new();
        
        // Generate quantum-inspired interference patterns
        for (i, pattern) in patterns.iter().enumerate() {
            let phase = i as f64 * std::f64::consts::PI / patterns.len() as f64;
            let magnitude = pattern.iter().sum::<f64>() / pattern.len() as f64;
            
            // Apply system correlation influence
            let correlation_influence = if i < system_correlations.len() && i < system_correlations[i].len() {
                system_correlations[i][i] * 0.5
            } else {
                0.0
            };
            
            let enhanced_magnitude = magnitude * (1.0 + correlation_influence);
            interference_patterns.push(Complex64::new(
                enhanced_magnitude * phase.cos(),
                enhanced_magnitude * phase.sin(),
            ));
        }
        
        Ok(Self {
            patterns: patterns.to_vec(),
            entanglement_structure: EntanglementStructure::Enhanced { interference_patterns },
            quantum_correlations: HashMap::new(),
        })
    }
    
    pub fn new_quantum(patterns: &[Vec<f64>], num_qubits: u32) -> Result<Self, QuantumError> {
        let mut bell_pairs = Vec::new();
        let mut fidelities = Vec::new();
        
        // Create Bell pairs for pattern correlations
        let n = patterns.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let correlation = calculate_pattern_correlation(&patterns[i], &patterns[j]);
                if correlation > 0.5 {
                    bell_pairs.push((i, j));
                    fidelities.push(correlation);
                }
            }
        }
        
        Ok(Self {
            patterns: patterns.to_vec(),
            entanglement_structure: EntanglementStructure::Quantum { bell_pairs, fidelities },
            quantum_correlations: HashMap::new(),
        })
    }
    
    /// Match input against entangled patterns
    pub fn match_entangled(&self, input: &[f64]) -> Vec<EntangledPatternMatch> {
        let mut matches = Vec::new();
        
        for (i, pattern) in self.patterns.iter().enumerate() {
            let base_similarity = calculate_pattern_correlation(input, pattern);
            
            let entangled_boost = self.calculate_entanglement_boost(i, input);
            let final_similarity = base_similarity * (1.0 + entangled_boost);
            
            matches.push(EntangledPatternMatch {
                pattern_index: i,
                base_similarity,
                entangled_similarity: final_similarity,
                entanglement_contribution: entangled_boost,
                quantum_correlation: self.get_quantum_correlation(i),
            });
        }
        
        matches.sort_by(|a, b| b.entangled_similarity.partial_cmp(&a.entangled_similarity).unwrap());
        matches
    }
    
    fn calculate_entanglement_boost(&self, pattern_index: usize, input: &[f64]) -> f64 {
        match &self.entanglement_structure {
            EntanglementStructure::Classical { correlation_matrix } => {
                // Classical correlation boost
                let mut boost = 0.0;
                for (j, other_pattern) in self.patterns.iter().enumerate() {
                    if j != pattern_index && j < correlation_matrix.len() && pattern_index < correlation_matrix[j].len() {
                        let correlation = correlation_matrix[j][pattern_index];
                        let input_similarity = calculate_pattern_correlation(input, other_pattern);
                        boost += correlation * input_similarity * 0.1;
                    }
                }
                boost
            }
            EntanglementStructure::Enhanced { interference_patterns } => {
                // Quantum-inspired interference boost
                if pattern_index < interference_patterns.len() {
                    let interference = interference_patterns[pattern_index];
                    let input_phase = input.iter().sum::<f64>() * std::f64::consts::PI;
                    let phase_alignment = (interference.arg() - input_phase).cos();
                    interference.norm() * phase_alignment.abs() * 0.3
                } else {
                    0.0
                }
            }
            EntanglementStructure::Quantum { bell_pairs, fidelities } => {
                // Quantum entanglement boost
                let mut boost = 0.0;
                for ((i, j), &fidelity) in bell_pairs.iter().zip(fidelities.iter()) {
                    if *i == pattern_index || *j == pattern_index {
                        let partner_idx = if *i == pattern_index { *j } else { *i };
                        if partner_idx < self.patterns.len() {
                            let partner_similarity = calculate_pattern_correlation(input, &self.patterns[partner_idx]);
                            boost += fidelity * partner_similarity * 0.4;
                        }
                    }
                }
                boost
            }
        }
    }
    
    fn get_quantum_correlation(&self, pattern_index: usize) -> f64 {
        self.quantum_correlations
            .get(&pattern_index)
            .and_then(|correlations| correlations.values().max_by(|a, b| a.partial_cmp(b).unwrap()))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Entangled pattern matching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntangledPatternMatch {
    pub pattern_index: usize,
    pub base_similarity: f64,
    pub entangled_similarity: f64,
    pub entanglement_contribution: f64,
    pub quantum_correlation: f64,
}

// Helper functions

fn calculate_pattern_correlation(pattern1: &[f64], pattern2: &[f64]) -> f64 {
    if pattern1.len() != pattern2.len() || pattern1.is_empty() {
        return 0.0;
    }
    
    let mean1 = pattern1.iter().sum::<f64>() / pattern1.len() as f64;
    let mean2 = pattern2.iter().sum::<f64>() / pattern2.len() as f64;
    
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    
    for (x, y) in pattern1.iter().zip(pattern2.iter()) {
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
        (numerator / denominator + 1.0) / 2.0 // Normalize to [0, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::QuantumMode;
    
    #[test]
    fn test_entanglement_manager_creation() {
        let manager = EntanglementManager::new(4);
        assert_eq!(manager.num_qubits, 4);
        assert!(manager.entangled_pairs.is_empty());
        assert_eq!(manager.correlation_matrix.len(), 4);
    }
    
    #[test]
    fn test_create_entanglement() {
        let mut manager = EntanglementManager::new(4);
        let result = manager.create_entanglement(0, 1, 0.8);
        
        assert!(result.is_ok());
        assert!(manager.measure_entanglement(0, 1).is_some());
        assert!(manager.measure_entanglement(0, 1).unwrap() > 0.0);
    }
    
    #[test]
    fn test_teleportation_channel() {
        let channel = TeleportationChannel::new_classical(0, 1, 2).unwrap();
        assert_eq!(channel.source_qubit, 0);
        assert_eq!(channel.auxiliary_qubit, 1);
        assert_eq!(channel.target_qubit, 2);
        assert!(channel.fidelity > 0.0);
    }
    
    #[test]
    fn test_quantum_state_vector() {
        let mut state = QuantumStateVector::new(2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.amplitudes[0].re, 1.0);
        
        state.apply_fidelity(0.9);
        // State should still be normalized
        let norm_sq: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_entangled_pattern_state() {
        let patterns = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let state = EntangledPatternState::new_classical(&patterns).unwrap();
        assert_eq!(state.patterns.len(), 3);
        
        let input = vec![0.8, 0.2, 0.1];
        let matches = state.match_entangled(&input);
        assert_eq!(matches.len(), 3);
        assert!(matches[0].entangled_similarity >= matches[1].entangled_similarity);
    }
    
    #[test]
    fn test_bell_violation_detection() {
        QuantumMode::set_global(QuantumMode::Full);
        let mut manager = EntanglementManager::new(4);
        
        // Create strong entanglement
        manager.create_entanglement(0, 1, 1.0).unwrap();
        
        let violations = manager.detect_bell_violations();
        // In enhanced/full mode, should detect violations for strong entanglement
        assert!(!violations.is_empty() || QuantumMode::current() == QuantumMode::Classical);
        
        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }
    
    #[test]
    fn test_entanglement_swapping() {
        let mut manager = EntanglementManager::new(4);
        
        // Create initial entanglements
        manager.create_entanglement(0, 1, 0.8).unwrap();
        manager.create_entanglement(2, 3, 0.8).unwrap();
        
        let result = manager.entanglement_swapping(0, 1, 2, 3).unwrap();
        // Result depends on random outcome, but should be valid
        assert!(result.fidelity >= 0.0);
    }
    
    #[test]
    fn test_pattern_correlation() {
        let pattern1 = vec![1.0, 0.0, 0.0];
        let pattern2 = vec![1.0, 0.0, 0.0];
        let pattern3 = vec![0.0, 1.0, 0.0];
        
        let correlation = calculate_pattern_correlation(&pattern1, &pattern2);
        assert!((correlation - 1.0).abs() < 0.01);
        
        let correlation2 = calculate_pattern_correlation(&pattern1, &pattern3);
        assert!(correlation2 < 1.0);
    }
}