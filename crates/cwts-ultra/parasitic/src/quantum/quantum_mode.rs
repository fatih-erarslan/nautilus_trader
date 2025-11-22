//! # Quantum Mode Management
//! 
//! Core quantum mode definitions and runtime switching logic.

use serde::{Serialize, Deserialize};
use std::fmt;

/// Runtime quantum processing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumMode {
    /// Pure classical algorithms with performance optimizations
    Classical,
    /// Hybrid quantum-inspired classical algorithms 
    Enhanced,
    /// Full quantum simulation with qubits and quantum gates
    Full,
}

impl Default for QuantumMode {
    fn default() -> Self {
        QuantumMode::Classical
    }
}

impl fmt::Display for QuantumMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumMode::Classical => write!(f, "classical"),
            QuantumMode::Enhanced => write!(f, "enhanced"),
            QuantumMode::Full => write!(f, "full-quantum"),
        }
    }
}

impl QuantumMode {
    /// Check if currently in full quantum mode
    pub fn is_full_quantum(&self) -> bool {
        matches!(self, QuantumMode::Full)
    }
    
    /// Get the current runtime mode
    pub fn current() -> Self {
        // Check environment variable or config
        std::env::var("QUANTUM_MODE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default()
    }
}

impl std::str::FromStr for QuantumMode {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "classical" | "classic" | "c" => Ok(QuantumMode::Classical),
            "enhanced" | "hybrid" | "e" | "h" => Ok(QuantumMode::Enhanced),
            "full" | "quantum" | "full-quantum" | "f" | "q" => Ok(QuantumMode::Full),
            _ => Err(format!("Invalid quantum mode: '{}'. Valid options: classical, enhanced, full", s)),
        }
    }
}

impl QuantumMode {
    /// Get the current global quantum mode
    pub fn current() -> QuantumMode {
        use std::sync::atomic::{AtomicU8, Ordering};
        static QUANTUM_MODE: AtomicU8 = AtomicU8::new(0);
        match QUANTUM_MODE.load(Ordering::Relaxed) {
            1 => QuantumMode::Enhanced,
            2 => QuantumMode::Full,
            _ => QuantumMode::Classical,
        }
    }
    
    /// Set the global quantum mode
    pub fn set_global(mode: QuantumMode) {
        use std::sync::atomic::{AtomicU8, Ordering};
        static QUANTUM_MODE: AtomicU8 = AtomicU8::new(0);
        let value = match mode {
            QuantumMode::Classical => 0,
            QuantumMode::Enhanced => 1,
            QuantumMode::Full => 2,
        };
        QUANTUM_MODE.store(value, Ordering::Relaxed);
    }
    
    /// Check if quantum features are enabled
    pub fn is_quantum_enabled(self) -> bool {
        self.uses_quantum()
    }
    
    /// Get all available modes
    pub const fn all_modes() -> &'static [QuantumMode] {
        &[QuantumMode::Classical, QuantumMode::Enhanced, QuantumMode::Full]
    }
    
    /// Check if this mode uses any quantum features
    #[inline]
    pub const fn uses_quantum(&self) -> bool {
        matches!(self, QuantumMode::Enhanced | QuantumMode::Full)
    }
    
    /// Check if this mode uses full quantum simulation
    #[inline] 
    pub const fn is_full_simulation(&self) -> bool {
        matches!(self, QuantumMode::Full)
    }
    
    /// Get computational complexity scaling for this mode
    pub fn complexity_scaling(&self) -> ComplexityScaling {
        match self {
            QuantumMode::Classical => ComplexityScaling {
                time_complexity: "O(n²)",
                space_complexity: "O(n)",
                parallelization_factor: 1.0,
                quantum_advantage: None,
            },
            QuantumMode::Enhanced => ComplexityScaling {
                time_complexity: "O(n log n)",
                space_complexity: "O(n)",
                parallelization_factor: 4.0,
                quantum_advantage: Some(2.5),
            },
            QuantumMode::Full => ComplexityScaling {
                time_complexity: "O(√n)",
                space_complexity: "O(2ⁿ)",
                parallelization_factor: 16.0,
                quantum_advantage: Some(100.0),
            },
        }
    }
    
    /// Get resource requirements for this mode
    pub fn resource_requirements(&self) -> ResourceRequirements {
        match self {
            QuantumMode::Classical => ResourceRequirements {
                cpu_cores: 1,
                memory_mb: 100,
                gpu_required: false,
                quantum_volume: 0,
                coherence_time_us: None,
            },
            QuantumMode::Enhanced => ResourceRequirements {
                cpu_cores: 4,
                memory_mb: 500,
                gpu_required: false,
                quantum_volume: 0,
                coherence_time_us: None,
            },
            QuantumMode::Full => ResourceRequirements {
                cpu_cores: 8,
                memory_mb: 2048,
                gpu_required: true,
                quantum_volume: 64,
                coherence_time_us: Some(100),
            },
        }
    }
    
    /// Get mode-specific configuration
    pub fn get_config(&self) -> ModeConfig {
        match self {
            QuantumMode::Classical => ModeConfig {
                enable_simd: true,
                enable_gpu: false,
                enable_quantum_circuits: false,
                enable_entanglement: false,
                enable_superposition: false,
                max_qubits: 0,
                noise_modeling: false,
                error_correction: false,
            },
            QuantumMode::Enhanced => ModeConfig {
                enable_simd: true,
                enable_gpu: true,
                enable_quantum_circuits: false,
                enable_entanglement: false,
                enable_superposition: false,
                max_qubits: 0,
                noise_modeling: false,
                error_correction: false,
            },
            QuantumMode::Full => ModeConfig {
                enable_simd: true,
                enable_gpu: true,
                enable_quantum_circuits: true,
                enable_entanglement: true,
                enable_superposition: true,
                max_qubits: 20,
                noise_modeling: true,
                error_correction: false,
            },
        }
    }
}

/// Computational complexity characteristics for quantum modes
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityScaling {
    pub time_complexity: &'static str,
    pub space_complexity: &'static str,
    pub parallelization_factor: f64,
    pub quantum_advantage: Option<f64>,
}

/// Resource requirements for quantum modes
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_required: bool,
    pub quantum_volume: u32,
    pub coherence_time_us: Option<u64>,
}

/// Mode-specific configuration settings
#[derive(Debug, Clone, PartialEq)]
pub struct ModeConfig {
    pub enable_simd: bool,
    pub enable_gpu: bool,
    pub enable_quantum_circuits: bool,
    pub enable_entanglement: bool,
    pub enable_superposition: bool,
    pub max_qubits: u32,
    pub noise_modeling: bool,
    pub error_correction: bool,
}

/// Quantum mode transition validator
pub struct ModeTransitionValidator;

impl ModeTransitionValidator {
    /// Validate if a mode transition is allowed
    pub fn validate_transition(
        from: QuantumMode, 
        to: QuantumMode,
        system_capabilities: &SystemCapabilities
    ) -> Result<TransitionPlan, TransitionError> {
        // Check system compatibility
        let to_requirements = to.resource_requirements();
        
        if to_requirements.cpu_cores > system_capabilities.available_cpu_cores {
            return Err(TransitionError::InsufficientResources {
                required: to_requirements.cpu_cores,
                available: system_capabilities.available_cpu_cores,
                resource_type: "CPU cores".to_string(),
            });
        }
        
        if to_requirements.memory_mb > system_capabilities.available_memory_mb {
            return Err(TransitionError::InsufficientResources {
                required: to_requirements.memory_mb,
                available: system_capabilities.available_memory_mb,
                resource_type: "Memory MB".to_string(),
            });
        }
        
        if to_requirements.gpu_required && !system_capabilities.gpu_available {
            return Err(TransitionError::MissingHardware("GPU required but not available".to_string()));
        }
        
        // Check if transition requires special handling
        let transition_type = match (from, to) {
            (QuantumMode::Classical, QuantumMode::Enhanced) => TransitionType::ClassicalToEnhanced,
            (QuantumMode::Classical, QuantumMode::Full) => TransitionType::ClassicalToQuantum,
            (QuantumMode::Enhanced, QuantumMode::Classical) => TransitionType::EnhancedToClassical,
            (QuantumMode::Enhanced, QuantumMode::Full) => TransitionType::EnhancedToQuantum,
            (QuantumMode::Full, QuantumMode::Enhanced) => TransitionType::QuantumToEnhanced,
            (QuantumMode::Full, QuantumMode::Classical) => TransitionType::QuantumToClassical,
            _ => TransitionType::NoChange,
        };
        
        Ok(TransitionPlan {
            from,
            to,
            transition_type,
            estimated_duration_ms: estimate_transition_time(&transition_type),
            required_resources: to_requirements,
            validation_checks: generate_validation_checks(&transition_type),
        })
    }
}

/// System capabilities for mode validation
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub available_cpu_cores: u32,
    pub available_memory_mb: u64,
    pub gpu_available: bool,
    pub quantum_simulator_available: bool,
    pub max_supported_qubits: u32,
}

impl SystemCapabilities {
    /// Detect system capabilities automatically
    pub fn detect() -> Self {
        Self {
            available_cpu_cores: num_cpus::get() as u32,
            available_memory_mb: 8192, // Simplified - would use actual system info
            gpu_available: false, // Would detect actual GPU availability
            quantum_simulator_available: true, // Software simulator always available
            max_supported_qubits: 20,
        }
    }
}

/// Transition plan for mode changes
#[derive(Debug, Clone)]
pub struct TransitionPlan {
    pub from: QuantumMode,
    pub to: QuantumMode,
    pub transition_type: TransitionType,
    pub estimated_duration_ms: u64,
    pub required_resources: ResourceRequirements,
    pub validation_checks: Vec<ValidationCheck>,
}

/// Types of mode transitions
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    NoChange,
    ClassicalToEnhanced,
    ClassicalToQuantum,
    EnhancedToClassical,
    EnhancedToQuantum,
    QuantumToEnhanced,
    QuantumToClassical,
}

/// Validation checks for transitions
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// Transition errors
#[derive(Debug, thiserror::Error)]
pub enum TransitionError {
    #[error("Insufficient resources: need {required} {resource_type}, have {available}")]
    InsufficientResources {
        required: u64,
        available: u64,
        resource_type: String,
    },
    #[error("Missing required hardware: {0}")]
    MissingHardware(String),
    #[error("Invalid transition: {0}")]
    InvalidTransition(String),
    #[error("System not ready: {0}")]
    SystemNotReady(String),
}

fn estimate_transition_time(transition_type: &TransitionType) -> u64 {
    match transition_type {
        TransitionType::NoChange => 0,
        TransitionType::ClassicalToEnhanced => 100,
        TransitionType::EnhancedToClassical => 50,
        TransitionType::ClassicalToQuantum => 1000,
        TransitionType::EnhancedToQuantum => 500,
        TransitionType::QuantumToEnhanced => 200,
        TransitionType::QuantumToClassical => 100,
    }
}

fn generate_validation_checks(transition_type: &TransitionType) -> Vec<ValidationCheck> {
    match transition_type {
        TransitionType::NoChange => vec![],
        TransitionType::ClassicalToEnhanced => vec![
            ValidationCheck {
                name: "SIMD Support".to_string(),
                description: "Verify SIMD instruction support".to_string(),
                required: false,
            },
        ],
        TransitionType::ClassicalToQuantum | TransitionType::EnhancedToQuantum => vec![
            ValidationCheck {
                name: "Quantum Simulator".to_string(),
                description: "Initialize quantum circuit simulator".to_string(),
                required: true,
            },
            ValidationCheck {
                name: "Memory Allocation".to_string(),
                description: "Allocate quantum state memory".to_string(),
                required: true,
            },
        ],
        _ => vec![
            ValidationCheck {
                name: "Cleanup".to_string(),
                description: "Clean up previous mode resources".to_string(),
                required: true,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_mode_string_conversion() {
        assert_eq!("classical".parse::<QuantumMode>().unwrap(), QuantumMode::Classical);
        assert_eq!("enhanced".parse::<QuantumMode>().unwrap(), QuantumMode::Enhanced);
        assert_eq!("full".parse::<QuantumMode>().unwrap(), QuantumMode::Full);
        assert_eq!("quantum".parse::<QuantumMode>().unwrap(), QuantumMode::Full);
        
        assert!("invalid".parse::<QuantumMode>().is_err());
    }
    
    #[test]
    fn test_mode_properties() {
        assert!(!QuantumMode::Classical.uses_quantum());
        assert!(QuantumMode::Enhanced.uses_quantum());
        assert!(QuantumMode::Full.uses_quantum());
        
        assert!(!QuantumMode::Classical.is_full_simulation());
        assert!(!QuantumMode::Enhanced.is_full_simulation());
        assert!(QuantumMode::Full.is_full_simulation());
    }
    
    #[test]
    fn test_resource_requirements() {
        let classical_req = QuantumMode::Classical.resource_requirements();
        let full_req = QuantumMode::Full.resource_requirements();
        
        assert!(classical_req.cpu_cores <= full_req.cpu_cores);
        assert!(classical_req.memory_mb <= full_req.memory_mb);
        assert!(!classical_req.gpu_required);
        assert!(full_req.gpu_required);
    }
    
    #[test]
    fn test_transition_validation() {
        let capabilities = SystemCapabilities {
            available_cpu_cores: 8,
            available_memory_mb: 4096,
            gpu_available: true,
            quantum_simulator_available: true,
            max_supported_qubits: 20,
        };
        
        let plan = ModeTransitionValidator::validate_transition(
            QuantumMode::Classical,
            QuantumMode::Full,
            &capabilities
        ).unwrap();
        
        assert_eq!(plan.from, QuantumMode::Classical);
        assert_eq!(plan.to, QuantumMode::Full);
        assert_eq!(plan.transition_type, TransitionType::ClassicalToQuantum);
        assert!(plan.estimated_duration_ms > 0);
    }
    
    #[test] 
    fn test_insufficient_resources() {
        let limited_capabilities = SystemCapabilities {
            available_cpu_cores: 1,
            available_memory_mb: 100,
            gpu_available: false,
            quantum_simulator_available: false,
            max_supported_qubits: 0,
        };
        
        let result = ModeTransitionValidator::validate_transition(
            QuantumMode::Classical,
            QuantumMode::Full,
            &limited_capabilities
        );
        
        assert!(result.is_err());
        match result {
            Err(TransitionError::InsufficientResources { .. }) => {},
            Err(TransitionError::MissingHardware(_)) => {},
            _ => panic!("Expected resource or hardware error"),
        }
    }
}