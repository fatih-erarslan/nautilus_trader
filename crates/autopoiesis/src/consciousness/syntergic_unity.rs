//! Syntergic Unity Implementation
//! Implements Grinberg's concept of syntergic unity in consciousness
//! Represents the unified experience emerging from neuronal coherence

use crate::prelude::*;
use crate::consciousness::neuronal_field::{NeuronalField, FieldState};
use nalgebra::{Matrix3, Vector3, Complex};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Minimum syntergic coherence for unified experience
const UNITY_THRESHOLD: f64 = 0.9;

/// Integration time constant for consciousness consolidation
const INTEGRATION_TAU: f64 = 0.05;

/// Syntergic unity representing unified conscious experience
#[derive(Debug, Clone)]
pub struct SyntergicUnity {
    /// Primary neuronal field
    neuronal_field: NeuronalField,
    
    /// Unity coherence matrix
    unity_matrix: Matrix3<f64>,
    
    /// Experiential quality dimensions
    qualia_dimensions: HashMap<String, f64>,
    
    /// Conscious moment buffer
    moment_buffer: Vec<ConsciousMoment>,
    
    /// Integration strength
    integration_strength: f64,
    
    /// Temporal continuity measure
    continuity_measure: f64,
    
    /// Current unity level
    unity_level: f64,
}

/// Represents a discrete moment of conscious experience
#[derive(Debug, Clone)]
pub struct ConsciousMoment {
    /// Timestamp of the moment
    pub timestamp: f64,
    
    /// Field state at this moment
    pub field_state: FieldState,
    
    /// Qualia intensities
    pub qualia: HashMap<String, f64>,
    
    /// Unity coherence
    pub coherence: f64,
    
    /// Experiential content
    pub content: ExperientialContent,
}

/// Content of conscious experience
#[derive(Debug, Clone)]
pub struct ExperientialContent {
    /// Sensory components
    pub sensory: HashMap<String, f64>,
    
    /// Cognitive components
    pub cognitive: HashMap<String, f64>,
    
    /// Emotional valence
    pub emotional_valence: f64,
    
    /// Attention focus
    pub attention_vector: Vector3<f64>,
}

impl SyntergicUnity {
    pub fn new(field_dimensions: (usize, usize, usize)) -> Self {
        let mut qualia_dimensions = HashMap::new();
        
        // Initialize basic qualia dimensions
        qualia_dimensions.insert("brightness".to_string(), 0.0);
        qualia_dimensions.insert("color_saturation".to_string(), 0.0);
        qualia_dimensions.insert("texture".to_string(), 0.0);
        qualia_dimensions.insert("emotional_tone".to_string(), 0.0);
        qualia_dimensions.insert("temporal_flow".to_string(), 0.0);
        
        Self {
            neuronal_field: NeuronalField::new(field_dimensions),
            unity_matrix: Matrix3::identity(),
            qualia_dimensions,
            moment_buffer: Vec::new(),
            integration_strength: 1.0,
            continuity_measure: 0.0,
            unity_level: 0.0,
        }
    }
    
    /// Initialize unity with coherent field state
    pub fn initialize_coherent_unity(&mut self, base_frequency: f64) {
        self.neuronal_field.initialize_coherent_state(base_frequency);
        self.update_unity_measures();
    }
    
    /// Process a time step of unified consciousness
    pub fn process_conscious_moment(&mut self, dt: f64, external_input: Option<ExternalInput>) {
        // Evolve neuronal field
        self.neuronal_field.evolve(dt);
        
        // Apply external input if present
        if let Some(input) = external_input {
            self.apply_external_input(input);
        }
        
        // Update unity measures
        self.update_unity_measures();
        
        // Create conscious moment
        let moment = self.create_conscious_moment();
        
        // Update continuity
        self.update_continuity(&moment);
        
        // Store moment in buffer
        self.moment_buffer.push(moment);
        
        // Maintain buffer size
        if self.moment_buffer.len() > 100 {
            self.moment_buffer.remove(0);
        }
    }
    
    /// Update all unity measures
    fn update_unity_measures(&mut self) {
        let field_state = self.neuronal_field.get_field_state();
        
        // Update unity matrix based on field coherence
        self.unity_matrix = field_state.coherence * self.integration_strength;
        
        // Calculate overall unity level
        self.unity_level = self.compute_unity_level(&field_state);
        
        // Update qualia dimensions
        self.update_qualia(&field_state);
    }
    
    /// Compute overall unity level from field state
    fn compute_unity_level(&self, field_state: &FieldState) -> f64 {
        let coherence_factor = field_state.coherence.trace() / 3.0;
        let integration_factor = self.compute_integration_factor(field_state);
        let continuity_factor = self.continuity_measure;
        
        // Weighted combination with nonlinear enhancement
        let raw_unity = 0.4 * coherence_factor + 0.4 * integration_factor + 0.2 * continuity_factor;
        
        // Apply sigmoid for threshold behavior
        1.0 / (1.0 + (-10.0 * (raw_unity - 0.5)).exp())
    }
    
    /// Compute integration factor from syntergic synthesis
    fn compute_integration_factor(&self, field_state: &FieldState) -> f64 {
        let mut integration = 0.0;
        let dimensions = field_state.amplitude.len();
        
        // Compute cross-regional integration
        for i in 0..dimensions {
            for j in 0..dimensions {
                for k in 0..dimensions {
                    if i < field_state.amplitude.len() && 
                       j < field_state.amplitude[0].len() &&
                       k < field_state.amplitude[0][0].len() {
                        
                        let amplitude = field_state.amplitude[i][j][k];
                        let phase = field_state.phase[i][j][k];
                        
                        // Calculate local integration contribution
                        integration += amplitude * phase.cos() * self.integration_strength;
                    }
                }
            }
        }
        
        integration / (dimensions.pow(3) as f64)
    }
    
    /// Update qualia dimensions based on field state
    fn update_qualia(&mut self, field_state: &FieldState) {
        let avg_amplitude = self.compute_average_amplitude(&field_state.amplitude);
        let phase_variance = self.compute_phase_variance(&field_state.phase);
        let coherence_strength = field_state.coherence.determinant();
        
        // Map field properties to qualia
        *self.qualia_dimensions.get_mut("brightness").unwrap() = avg_amplitude;
        *self.qualia_dimensions.get_mut("color_saturation").unwrap() = coherence_strength.abs();
        *self.qualia_dimensions.get_mut("texture").unwrap() = phase_variance;
        *self.qualia_dimensions.get_mut("emotional_tone").unwrap() = 
            (field_state.consciousness_level - 0.5) * 2.0;
        *self.qualia_dimensions.get_mut("temporal_flow").unwrap() = self.continuity_measure;
    }
    
    /// Compute average amplitude across field
    fn compute_average_amplitude(&self, amplitude_field: &[Vec<Vec<f64>>]) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..amplitude_field.len() {
            for j in 0..amplitude_field[0].len() {
                for k in 0..amplitude_field[0][0].len() {
                    sum += amplitude_field[i][j][k];
                    count += 1;
                }
            }
        }
        
        if count > 0 { sum / count as f64 } else { 0.0 }
    }
    
    /// Compute phase variance across field
    fn compute_phase_variance(&self, phase_field: &[Vec<Vec<f64>>]) -> f64 {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;
        
        for i in 0..phase_field.len() {
            for j in 0..phase_field[0].len() {
                for k in 0..phase_field[0][0].len() {
                    let phase = phase_field[i][j][k];
                    sum += phase;
                    sum_sq += phase * phase;
                    count += 1;
                }
            }
        }
        
        if count > 1 {
            let mean = sum / count as f64;
            (sum_sq / count as f64) - mean * mean
        } else {
            0.0
        }
    }
    
    /// Create a conscious moment from current state
    fn create_conscious_moment(&self) -> ConsciousMoment {
        let field_state = self.neuronal_field.get_field_state();
        
        let content = ExperientialContent {
            sensory: self.extract_sensory_content(&field_state),
            cognitive: self.extract_cognitive_content(&field_state),
            emotional_valence: self.qualia_dimensions["emotional_tone"],
            attention_vector: self.compute_attention_vector(&field_state),
        };
        
        ConsciousMoment {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            field_state,
            qualia: self.qualia_dimensions.clone(),
            coherence: self.unity_level,
            content,
        }
    }
    
    /// Extract sensory content from field state
    fn extract_sensory_content(&self, field_state: &FieldState) -> HashMap<String, f64> {
        let mut sensory = HashMap::new();
        
        // Map field regions to sensory modalities
        sensory.insert("visual".to_string(), self.qualia_dimensions["brightness"]);
        sensory.insert("auditory".to_string(), self.qualia_dimensions["temporal_flow"]);
        sensory.insert("tactile".to_string(), self.qualia_dimensions["texture"]);
        sensory.insert("proprioceptive".to_string(), field_state.consciousness_level);
        
        sensory
    }
    
    /// Extract cognitive content from field state
    fn extract_cognitive_content(&self, field_state: &FieldState) -> HashMap<String, f64> {
        let mut cognitive = HashMap::new();
        
        cognitive.insert("attention".to_string(), self.unity_level);
        cognitive.insert("memory_activation".to_string(), self.continuity_measure);
        cognitive.insert("conceptual_binding".to_string(), field_state.coherence.trace() / 3.0);
        cognitive.insert("executive_control".to_string(), self.integration_strength);
        
        cognitive
    }
    
    /// Compute attention vector from field gradients
    fn compute_attention_vector(&self, field_state: &FieldState) -> Vector3<f64> {
        let dimensions = field_state.amplitude.len();
        let mut gradient = Vector3::zeros();
        
        // Compute center of mass of high-amplitude regions
        let mut weighted_sum = Vector3::zeros();
        let mut total_weight = 0.0;
        
        for i in 0..dimensions {
            for j in 0..dimensions {
                for k in 0..dimensions {
                    if i < field_state.amplitude.len() &&
                       j < field_state.amplitude[0].len() &&
                       k < field_state.amplitude[0][0].len() {
                        
                        let weight = field_state.amplitude[i][j][k].powi(2);
                        weighted_sum += weight * Vector3::new(i as f64, j as f64, k as f64);
                        total_weight += weight;
                    }
                }
            }
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            Vector3::zeros()
        }
    }
    
    /// Update temporal continuity measure
    fn update_continuity(&mut self, current_moment: &ConsciousMoment) {
        if let Some(previous_moment) = self.moment_buffer.last() {
            // Compute similarity between consecutive moments
            let coherence_similarity = 
                (current_moment.coherence - previous_moment.coherence).abs();
            let qualia_similarity = self.compute_qualia_similarity(
                &current_moment.qualia, &previous_moment.qualia);
            
            let moment_continuity = 1.0 - (coherence_similarity + qualia_similarity) / 2.0;
            
            // Update with exponential smoothing
            self.continuity_measure = INTEGRATION_TAU * moment_continuity + 
                (1.0 - INTEGRATION_TAU) * self.continuity_measure;
        }
    }
    
    /// Compute similarity between qualia states
    fn compute_qualia_similarity(&self, qualia1: &HashMap<String, f64>, 
                                qualia2: &HashMap<String, f64>) -> f64 {
        let mut similarity = 0.0;
        let mut count = 0;
        
        for (key, value1) in qualia1 {
            if let Some(value2) = qualia2.get(key) {
                similarity += (value1 - value2).abs();
                count += 1;
            }
        }
        
        if count > 0 { similarity / count as f64 } else { 0.0 }
    }
    
    /// Apply external input to the unity system
    fn apply_external_input(&mut self, input: ExternalInput) {
        // Apply sensory input to neuronal field
        if let Some(position) = input.spatial_position {
            self.neuronal_field.apply_stimulus(
                position, input.intensity, input.frequency
            );
        }
        
        // Modulate integration strength
        if let Some(attention_modulation) = input.attention_modulation {
            self.integration_strength *= attention_modulation;
        }
    }
    
    /// Check if system has achieved unified consciousness
    pub fn is_unified(&self) -> bool {
        self.unity_level >= UNITY_THRESHOLD
    }
    
    /// Get consciousness quality assessment
    pub fn assess_consciousness_quality(&self) -> ConsciousnessQuality {
        ConsciousnessQuality {
            unity_level: self.unity_level,
            coherence_strength: self.neuronal_field.get_field_state().coherence.trace() / 3.0,
            qualia_richness: self.compute_qualia_richness(),
            temporal_continuity: self.continuity_measure,
            integration_efficiency: self.integration_strength,
            is_unified: self.is_unified(),
            moment_count: self.moment_buffer.len(),
        }
    }
    
    /// Compute richness of qualia experience
    fn compute_qualia_richness(&self) -> f64 {
        let mut richness = 0.0;
        
        for value in self.qualia_dimensions.values() {
            richness += value.abs();
        }
        
        richness / self.qualia_dimensions.len() as f64
    }
    
    /// Get current conscious moment
    pub fn get_current_moment(&self) -> Option<&ConsciousMoment> {
        self.moment_buffer.last()
    }
    
    /// Get moment history
    pub fn get_moment_history(&self) -> &[ConsciousMoment] {
        &self.moment_buffer
    }
}

/// External input to consciousness system
#[derive(Debug, Clone)]
pub struct ExternalInput {
    pub spatial_position: Option<(usize, usize, usize)>,
    pub intensity: f64,
    pub frequency: f64,
    pub attention_modulation: Option<f64>,
    pub emotional_valence: f64,
}

/// Assessment of consciousness quality
#[derive(Debug, Clone)]
pub struct ConsciousnessQuality {
    pub unity_level: f64,
    pub coherence_strength: f64,
    pub qualia_richness: f64,
    pub temporal_continuity: f64,
    pub integration_efficiency: f64,
    pub is_unified: bool,
    pub moment_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_syntergic_unity_creation() {
        let unity = SyntergicUnity::new((10, 10, 10));
        assert!(!unity.is_unified());
        assert_eq!(unity.qualia_dimensions.len(), 5);
    }
    
    #[test]
    fn test_conscious_moment_processing() {
        let mut unity = SyntergicUnity::new((5, 5, 5));
        unity.initialize_coherent_unity(40.0);
        
        unity.process_conscious_moment(0.01, None);
        assert_eq!(unity.moment_buffer.len(), 1);
        
        let moment = unity.get_current_moment().unwrap();
        assert!(moment.coherence >= 0.0);
    }
    
    #[test]
    fn test_consciousness_quality_assessment() {
        let mut unity = SyntergicUnity::new((5, 5, 5));
        unity.initialize_coherent_unity(40.0);
        
        for _ in 0..10 {
            unity.process_conscious_moment(0.01, None);
        }
        
        let quality = unity.assess_consciousness_quality();
        assert!(quality.unity_level >= 0.0 && quality.unity_level <= 1.0);
        assert!(quality.coherence_strength >= 0.0);
    }
    
    #[test]
    fn test_external_input_processing() {
        let mut unity = SyntergicUnity::new((5, 5, 5));
        
        let input = ExternalInput {
            spatial_position: Some((2, 2, 2)),
            intensity: 1.0,
            frequency: 40.0,
            attention_modulation: Some(1.5),
            emotional_valence: 0.5,
        };
        
        unity.process_conscious_moment(0.01, Some(input));
        assert!(unity.integration_strength != 1.0); // Should be modulated
    }
}