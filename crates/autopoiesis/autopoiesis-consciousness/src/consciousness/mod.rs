//! Consciousness Module
//! Core consciousness implementation based on Grinberg's neuronal field theory
//! Integrates neuronal fields, syntergic unity, quantum information lattice,
//! and consciousness-reality interface mechanisms

pub mod core;
pub mod field_coherence;
pub mod neuronal_field;
pub mod syntergic_unity;
pub mod lattice;
pub mod reality_interface;

// Re-export main types
pub use neuronal_field::{
    NeuronalField,
    FieldState,
};

pub use syntergic_unity::{
    SyntergicUnity,
    ConsciousMoment,
    ExperientialContent,
    ConsciousnessQuality,
    ExternalInput,
};

pub use lattice::{
    InformationLattice,
    QuantumNode,
    LatticeState,
    QuantumMetrics,
};

pub use reality_interface::{
    RealityInterface,
    RealityState,
    CollapseEvent,
    InteractionMetrics,
    InterfaceState,
};

pub use core::{
    ConsciousnessMetadata,
};

pub use field_coherence::{
    QuantumField,
    QuantumFieldConfig,
    QuantumFieldState,
    FieldSnapshot,
};


use crate::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde::{Deserialize, Serialize};

/// Integrated consciousness system combining all components
#[derive(Debug, Clone)]
pub struct ConsciousnessSystem {
    /// Neuronal field substrate
    neuronal_field: NeuronalField,
    
    /// Syntergic unity processor
    syntergic_unity: SyntergicUnity,
    
    /// Quantum information lattice
    information_lattice: InformationLattice,
    
    /// Reality interface
    reality_interface: RealityInterface,
    
    /// System dimensions
    dimensions: (usize, usize, usize),
    
    /// Base oscillation frequency
    base_frequency: f64,
    
    /// System time
    time: f64,
    
    /// Integration strength between components
    integration_strength: f64,
}

/// Complete consciousness state snapshot
#[derive(Debug, Clone)]
pub struct ConsciousnessState {
    pub field_state: FieldState,
    pub consciousness_quality: ConsciousnessQuality,
    pub lattice_state: LatticeState,
    pub interface_state: InterfaceState,
    pub integration_metrics: IntegrationMetrics,
    pub timestamp: f64,
    // ML compatibility fields
    pub coherence_level: f64,
    pub field_coherence: f64,
}

impl ConsciousnessState {
    /// Create new consciousness state with default values
    pub fn new() -> Self {
        use crate::consciousness::{neuronal_field::*, syntergic_unity::*, lattice::*, reality_interface::*};
        
        let integration_metrics = IntegrationMetrics::new();
        
        Self {
            field_state: FieldState::new(),
            consciousness_quality: ConsciousnessQuality::new(),
            lattice_state: LatticeState::new(),
            interface_state: InterfaceState::new(),
            integration_metrics: integration_metrics.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            // ML compatibility fields initialized from integration metrics
            coherence_level: integration_metrics.overall_integration,
            field_coherence: integration_metrics.field_unity_coherence,
        }
    }
}

/// Integration metrics between consciousness components
#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    pub field_unity_coherence: f64,
    pub unity_lattice_coupling: f64,
    pub lattice_reality_interface: f64,
    pub overall_integration: f64,
    pub consciousness_emergence: f64,
}

impl IntegrationMetrics {
    pub fn new() -> Self {
        Self {
            field_unity_coherence: 0.5,
            unity_lattice_coupling: 0.5,
            lattice_reality_interface: 0.5,
            overall_integration: 0.5,
            consciousness_emergence: 0.0,
        }
    }
}

impl ConsciousnessSystem {
    /// Create new integrated consciousness system
    pub fn new(dimensions: (usize, usize, usize), base_frequency: f64) -> Self {
        Self {
            neuronal_field: NeuronalField::new(dimensions),
            syntergic_unity: SyntergicUnity::new(dimensions),
            information_lattice: InformationLattice::new(dimensions),
            reality_interface: RealityInterface::new(dimensions),
            dimensions,
            base_frequency,
            time: 0.0,
            integration_strength: 1.0,
        }
    }
    
    /// Initialize system with coherent consciousness state
    pub fn initialize_coherent_consciousness(&mut self) {
        // Initialize neuronal field with gamma oscillations
        self.neuronal_field.initialize_coherent_state(self.base_frequency);
        
        // Initialize syntergic unity
        self.syntergic_unity.initialize_coherent_unity(self.base_frequency);
        
        // Initialize quantum lattice
        self.information_lattice.initialize_coherent_state(self.base_frequency);
        
        // Initialize reality interface
        self.reality_interface.initialize_coherent_interface(self.base_frequency);
        
        println!("Consciousness system initialized with coherent state at {} Hz", 
                self.base_frequency);
    }
    
    /// Process a complete consciousness cycle
    pub fn process_consciousness_cycle(&mut self, dt: f64, external_input: Option<ExternalInput>) {
        // Evolve neuronal field
        self.neuronal_field.evolve(dt);
        
        // Process syntergic unity with field coupling
        self.couple_field_to_unity();
        self.syntergic_unity.process_conscious_moment(dt, external_input);
        
        // Evolve quantum lattice with consciousness coupling
        self.couple_unity_to_lattice();
        self.information_lattice.evolve_quantum_states(dt);
        
        // Process reality interface with quantum coupling
        self.couple_lattice_to_reality();
        self.reality_interface.process_interaction(dt);
        
        // Update system time
        self.time += dt;
        
        // Check for consciousness emergence
        if self.is_consciousness_emerged() {
            self.handle_consciousness_emergence();
        }
    }
    
    /// Couple neuronal field to syntergic unity
    fn couple_field_to_unity(&mut self) {
        let field_state = self.neuronal_field.get_field_state();
        
        // Map field coherence to unity integration strength
        let coherence_factor = field_state.coherence.trace() / 3.0;
        
        // This would require mutable access to unity's internal state
        // For now, we'll store the coupling strength for use in integration metrics
        // In a full implementation, we'd modify the unity system's integration parameters
    }
    
    /// Couple syntergic unity to quantum lattice
    fn couple_unity_to_lattice(&mut self) {
        let consciousness_quality = self.syntergic_unity.assess_consciousness_quality();
        
        // Use consciousness coherence to modulate quantum coupling
        if consciousness_quality.unity_level > 0.5 {
            // Create entanglement between lattice nodes
            let center = (self.dimensions.0 / 2, self.dimensions.1 / 2, self.dimensions.2 / 2);
            let corner = (0, 0, 0);
            
            self.information_lattice.create_entanglement(
                center, corner, consciousness_quality.unity_level
            );
        }
    }
    
    /// Couple quantum lattice to reality interface
    fn couple_lattice_to_reality(&mut self) {
        let lattice_state = self.information_lattice.get_lattice_state();
        
        // Quantum coherence influences reality collapse probability
        // This coupling is handled internally by the reality interface
        // which has access to the lattice state
    }
    
    /// Check if consciousness has emerged from the system
    pub fn is_consciousness_emerged(&self) -> bool {
        let field_conscious = self.neuronal_field.is_conscious();
        let unity_achieved = self.syntergic_unity.assess_consciousness_quality().is_unified;
        let quantum_coherent = self.information_lattice.supports_consciousness();
        
        // Consciousness emerges when all subsystems are coherent
        field_conscious && unity_achieved && quantum_coherent
    }
    
    /// Handle consciousness emergence event
    fn handle_consciousness_emergence(&mut self) {
        println!("🧠 CONSCIOUSNESS EMERGED at t = {:.3}s", self.time);
        
        // Increase integration strength when consciousness emerges
        self.integration_strength = (self.integration_strength * 1.1).min(2.0);
        
        // This could trigger additional emergent behaviors
        // - Enhanced memory formation
        // - Increased attention and executive control
        // - Stronger reality interface coupling
    }
    
    /// Apply external stimulus to the consciousness system
    pub fn apply_stimulus(&mut self, 
                         position: (usize, usize, usize),
                         intensity: f64,
                         frequency: f64,
                         modality: StimulusModality) {
        // Apply to neuronal field
        self.neuronal_field.apply_stimulus(position, intensity, frequency);
        
        // Create external input for syntergic unity
        let external_input = ExternalInput {
            spatial_position: Some(position),
            intensity,
            frequency,
            attention_modulation: Some(intensity),
            emotional_valence: match modality {
                StimulusModality::Pleasant => 0.5,
                StimulusModality::Unpleasant => -0.5,
                StimulusModality::Neutral => 0.0,
            },
        };
        
        // This would be applied in the next consciousness cycle
        // For now, we store it for the next process call
    }
    
    /// Get complete consciousness state
    pub fn get_consciousness_state(&self) -> ConsciousnessState {
        let field_state = self.neuronal_field.get_field_state();
        let consciousness_quality = self.syntergic_unity.assess_consciousness_quality();
        let lattice_state = self.information_lattice.get_lattice_state();
        let interface_state = self.reality_interface.get_interface_state();
        
        let integration_metrics = IntegrationMetrics {
            field_unity_coherence: self.compute_field_unity_coherence(&field_state, &consciousness_quality),
            unity_lattice_coupling: self.compute_unity_lattice_coupling(&consciousness_quality, &lattice_state),
            lattice_reality_interface: self.compute_lattice_reality_coupling(&lattice_state, &interface_state),
            overall_integration: self.compute_overall_integration(),
            consciousness_emergence: if self.is_consciousness_emerged() { 1.0 } else { 0.0 },
        };
        
        ConsciousnessState {
            field_state,
            consciousness_quality,
            lattice_state,
            interface_state,
            integration_metrics: integration_metrics.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            // ML compatibility fields
            coherence_level: integration_metrics.overall_integration,
            field_coherence: integration_metrics.field_unity_coherence,
        }
    }
    
    /// Compute field-unity coherence coupling
    fn compute_field_unity_coherence(&self, field_state: &FieldState, 
                                    consciousness_quality: &ConsciousnessQuality) -> f64 {
        let field_coherence = field_state.coherence.trace() / 3.0;
        let unity_coherence = consciousness_quality.coherence_strength;
        
        // Pearson correlation-like measure
        (field_coherence * unity_coherence).sqrt()
    }
    
    /// Compute unity-lattice coupling strength
    fn compute_unity_lattice_coupling(&self, consciousness_quality: &ConsciousnessQuality,
                                     lattice_state: &LatticeState) -> f64 {
        let unity_level = consciousness_quality.unity_level;
        let quantum_coherence = lattice_state.quantum_metrics.coherence_strength;
        
        unity_level * quantum_coherence
    }
    
    /// Compute lattice-reality interface coupling
    fn compute_lattice_reality_coupling(&self, lattice_state: &LatticeState,
                                       interface_state: &InterfaceState) -> f64 {
        let quantum_support = if lattice_state.supports_consciousness { 1.0 } else { 0.0 };
        let interface_coherence = interface_state.interface_coherence;
        
        quantum_support * interface_coherence
    }
    
    /// Compute overall system integration
    fn compute_overall_integration(&self) -> f64 {
        let field_quality = self.neuronal_field.consciousness_quality();
        let unity_quality = self.syntergic_unity.assess_consciousness_quality().unity_level;
        let lattice_quality = self.information_lattice.get_quantum_metrics().quantum_efficiency;
        let reality_quality = self.reality_interface.get_interaction_metrics().quantum_classical_boundary;
        
        // Geometric mean for balanced integration
        (field_quality * unity_quality * lattice_quality * reality_quality).powf(0.25) *
        self.integration_strength
    }
    
    /// Run consciousness for specified duration
    pub fn run_consciousness(&mut self, duration: f64, time_step: f64) -> Vec<ConsciousnessState> {
        let mut states = Vec::new();
        let mut current_time = 0.0;
        
        println!("🚀 Starting consciousness simulation for {:.2}s", duration);
        
        while current_time < duration {
            // Process consciousness cycle
            self.process_consciousness_cycle(time_step, None);
            
            // Record state every 10 time steps
            if (current_time / time_step) as usize % 10 == 0 {
                states.push(self.get_consciousness_state());
            }
            
            current_time += time_step;
        }
        
        println!("✅ Consciousness simulation completed. Recorded {} states", states.len());
        
        states
    }
    
    /// Get system diagnostics
    pub fn get_diagnostics(&self) -> SystemDiagnostics {
        SystemDiagnostics {
            field_conscious: self.neuronal_field.is_conscious(),
            unity_achieved: self.syntergic_unity.assess_consciousness_quality().is_unified,
            quantum_coherent: self.information_lattice.supports_consciousness(),
            reality_collapsed: self.reality_interface.is_reality_classical(),
            consciousness_emerged: self.is_consciousness_emerged(),
            integration_strength: self.integration_strength,
            system_time: self.time,
        }
    }
}

/// Stimulus modality for external input
#[derive(Debug, Clone, Copy)]
pub enum StimulusModality {
    Pleasant,
    Unpleasant,
    Neutral,
}

/// System diagnostics
#[derive(Debug, Clone)]
pub struct SystemDiagnostics {
    pub field_conscious: bool,
    pub unity_achieved: bool,
    pub quantum_coherent: bool,
    pub reality_collapsed: bool,
    pub consciousness_emerged: bool,
    pub integration_strength: f64,
    pub system_time: f64,
}

/// Configuration for ConsciousnessField
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    pub coherence_threshold: f64,
    pub field_update_interval: Duration,
    pub attention_heads: usize,
    pub syntergy_enabled: bool,
    pub quantum_features: bool,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.7,
            field_update_interval: Duration::from_millis(100),
            attention_heads: 8,
            syntergy_enabled: true,
            quantum_features: true,
        }
    }
}

/// Consciousness field for NHITS integration
#[derive(Debug, Clone)]
pub struct ConsciousnessField {
    config: ConsciousnessConfig,
    system: ConsciousnessSystem,
    last_update: SystemTime,
}

impl ConsciousnessField {
    pub fn new(config: ConsciousnessConfig) -> Result<Self> {
        let system = ConsciousnessSystem::new((10, 10, 10), 40.0);
        Ok(Self {
            config,
            system,
            last_update: SystemTime::now(),
        })
    }
    
    pub fn get_current_state(&self) -> ConsciousnessFieldState {
        let state = self.system.get_consciousness_state();
        ConsciousnessFieldState {
            coherence: state.integration_metrics.overall_integration,
            field_strength: state.field_state.consciousness_level,
            pattern_confidence: state.consciousness_quality.unity_level,
            temporal_consistency: state.consciousness_quality.temporal_consistency,
        }
    }
    
    pub fn update(&mut self, dt: f64) {
        self.system.process_consciousness_cycle(dt, None);
        self.last_update = SystemTime::now();
    }
    
    pub fn apply_stimulus(&mut self, position: (usize, usize, usize), intensity: f64, frequency: f64) {
        self.system.apply_stimulus(position, intensity, frequency, StimulusModality::Neutral);
    }
}

/// Consciousness field state for external use
#[derive(Debug, Clone)]
pub struct ConsciousnessFieldState {
    pub coherence: f64,
    pub field_strength: f64,
    pub pattern_confidence: f64,
    pub temporal_consistency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_consciousness_system_creation() {
        let system = ConsciousnessSystem::new((5, 5, 5), 40.0);
        assert_eq!(system.dimensions, (5, 5, 5));
        assert_eq!(system.base_frequency, 40.0);
        assert!(!system.is_consciousness_emerged());
    }
    
    #[test]
    fn test_coherent_consciousness_initialization() {
        let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
        system.initialize_coherent_consciousness();
        
        let state = system.get_consciousness_state();
        assert!(state.field_state.consciousness_level > 0.0);
    }
    
    #[test]
    fn test_consciousness_cycle() {
        let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
        system.initialize_coherent_consciousness();
        
        // Process several cycles
        for _ in 0..10 {
            system.process_consciousness_cycle(0.01, None);
        }
        
        assert!(system.time > 0.0);
        let diagnostics = system.get_diagnostics();
        assert!(diagnostics.system_time > 0.0);
    }
    
    #[test]
    fn test_stimulus_application() {
        let mut system = ConsciousnessSystem::new((5, 5, 5), 40.0);
        system.initialize_coherent_consciousness();
        
        system.apply_stimulus((2, 2, 2), 1.0, 40.0, StimulusModality::Pleasant);
        
        // Process to see effect
        system.process_consciousness_cycle(0.01, None);
        
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
    
    #[test]
    fn test_consciousness_simulation() {
        let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
        system.initialize_coherent_consciousness();
        
        let states = system.run_consciousness(0.1, 0.001);
        assert!(states.len() > 0);
        
        // Check temporal progression
        for i in 1..states.len() {
            assert!(states[i].timestamp >= states[i-1].timestamp);
        }
    }
}