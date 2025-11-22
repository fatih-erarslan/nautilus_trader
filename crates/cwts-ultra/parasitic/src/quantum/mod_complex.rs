//! # Quantum/Classical Switching System
//! 
//! Provides seamless runtime switching between classical quantum-enhanced algorithms
//! and full quantum computing features with zero overhead when quantum is disabled.

pub mod quantum_mode;
pub mod classical_enhanced;
pub mod quantum_simulators;
pub mod quantum_gates;
pub mod entanglement;
pub mod superposition;
pub mod quantum_tunneling;
pub mod benchmarks;
pub mod memory;

pub use quantum_mode::*;
pub use classical_enhanced::*;
pub use quantum_simulators::*;
pub use quantum_gates::*;
pub use entanglement::*;
pub use superposition::*;
pub use quantum_tunneling::*;
pub use memory::{QuantumTradingMemory, BiologicalMemorySystem};

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use serde::{Serialize, Deserialize};


/// Global quantum runtime mode state
static QUANTUM_MODE: AtomicU8 = AtomicU8::new(0); // 0 = Classical, 1 = Enhanced, 2 = Full

/// Quantum computing mode for the trading system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum QuantumMode {
    /// Classical algorithms with quantum-inspired optimizations
    Classical = 0,
    /// Quantum-enhanced classical algorithms (hybrid approach)
    Enhanced = 1,
    /// Full quantum computing simulation
    Full = 2,
}

impl Default for QuantumMode {
    fn default() -> Self {
        QuantumMode::Classical
    }
}

impl From<u8> for QuantumMode {
    fn from(value: u8) -> Self {
        match value {
            0 => QuantumMode::Classical,
            1 => QuantumMode::Enhanced,
            2 => QuantumMode::Full,
            _ => QuantumMode::Classical,
        }
    }
}

impl QuantumMode {
    /// Get the current global quantum mode
    #[inline]
    pub fn current() -> Self {
        QUANTUM_MODE.load(Ordering::Relaxed).into()
    }
    
    /// Set the global quantum mode
    #[inline]
    pub fn set_global(mode: Self) {
        QUANTUM_MODE.store(mode as u8, Ordering::Relaxed);
    }
    
    /// Check if quantum features are enabled
    #[inline]
    pub fn is_quantum_enabled(&self) -> bool {
        matches!(self, QuantumMode::Enhanced | QuantumMode::Full)
    }
    
    /// Check if full quantum simulation is enabled
    #[inline]
    pub fn is_full_quantum(&self) -> bool {
        matches!(self, QuantumMode::Full)
    }
    
    /// Get mode description
    pub fn description(&self) -> &'static str {
        match self {
            QuantumMode::Classical => "Classical quantum-enhanced algorithms",
            QuantumMode::Enhanced => "Hybrid quantum-classical processing",
            QuantumMode::Full => "Full quantum computing simulation",
        }
    }
    
    /// Get expected performance characteristics
    pub fn performance_info(&self) -> QuantumPerformanceInfo {
        match self {
            QuantumMode::Classical => QuantumPerformanceInfo {
                cpu_overhead: 0.0,
                memory_overhead: 1.0,
                quantum_advantage: 1.0,
                parallelization_factor: 1.0,
            },
            QuantumMode::Enhanced => QuantumPerformanceInfo {
                cpu_overhead: 0.15,
                memory_overhead: 1.2,
                quantum_advantage: 2.5,
                parallelization_factor: 4.0,
            },
            QuantumMode::Full => QuantumPerformanceInfo {
                cpu_overhead: 0.4,
                memory_overhead: 2.0,
                quantum_advantage: 10.0,
                parallelization_factor: 16.0,
            },
        }
    }
}

/// Performance characteristics for different quantum modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceInfo {
    /// Additional CPU overhead as a multiplier (0.0 = no overhead, 1.0 = 100% overhead)
    pub cpu_overhead: f64,
    /// Memory overhead multiplier
    pub memory_overhead: f64,
    /// Expected quantum advantage multiplier for compatible algorithms
    pub quantum_advantage: f64,
    /// Parallelization improvement factor
    pub parallelization_factor: f64,
}

/// Quantum runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Default quantum mode
    pub default_mode: QuantumMode,
    /// Enable automatic mode switching based on problem size
    pub auto_switch: bool,
    /// Threshold for switching to Enhanced mode (problem complexity score)
    pub enhanced_threshold: f64,
    /// Threshold for switching to Full quantum mode
    pub full_quantum_threshold: f64,
    /// Maximum qubits for full quantum simulation
    pub max_qubits: u32,
    /// Quantum circuit depth limit
    pub max_circuit_depth: u32,
    /// Enable quantum error correction
    pub error_correction: bool,
    /// Quantum noise model settings
    pub noise_model: QuantumNoiseConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNoiseConfig {
    /// Enable decoherence modeling
    pub enable_decoherence: bool,
    /// T1 relaxation time (microseconds)
    pub t1_relaxation: f64,
    /// T2 dephasing time (microseconds) 
    pub t2_dephasing: f64,
    /// Gate error probability
    pub gate_error_rate: f64,
    /// Measurement error probability
    pub measurement_error_rate: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            default_mode: QuantumMode::Classical,
            auto_switch: true,
            enhanced_threshold: 100.0,
            full_quantum_threshold: 1000.0,
            max_qubits: 20,
            max_circuit_depth: 100,
            error_correction: false,
            noise_model: QuantumNoiseConfig {
                enable_decoherence: true,
                t1_relaxation: 100.0,
                t2_dephasing: 50.0,
                gate_error_rate: 0.001,
                measurement_error_rate: 0.01,
            },
        }
    }
}

/// Global quantum runtime manager
#[derive(Debug)]
pub struct QuantumRuntime {
    config: QuantumConfig,
    simulators: QuantumSimulators,
    performance_tracker: Arc<QuantumPerformanceTracker>,
}

impl QuantumRuntime {
    /// Create a new quantum runtime with configuration
    pub fn new(config: QuantumConfig) -> Self {
        // Set initial global mode
        QuantumMode::set_global(config.default_mode);
        
        Self {
            simulators: QuantumSimulators::new(&config),
            performance_tracker: Arc::new(QuantumPerformanceTracker::new()),
            config,
        }
    }
    
    /// Initialize quantum runtime from command line arguments
    pub fn from_args(args: &[String]) -> Self {
        let mut config = QuantumConfig::default();
        
        // Parse quantum-related command line arguments
        for (i, arg) in args.iter().enumerate() {
            match arg.as_str() {
                "--quantum" => {
                    config.default_mode = QuantumMode::Full;
                }
                "--quantum-enhanced" => {
                    config.default_mode = QuantumMode::Enhanced;
                }
                "--quantum-classical" => {
                    config.default_mode = QuantumMode::Classical;
                }
                "--max-qubits" => {
                    if let Some(value) = args.get(i + 1) {
                        if let Ok(qubits) = value.parse::<u32>() {
                            config.max_qubits = qubits;
                        }
                    }
                }
                "--no-auto-switch" => {
                    config.auto_switch = false;
                }
                "--quantum-noise" => {
                    config.noise_model.enable_decoherence = true;
                }
                _ => {}
            }
        }
        
        Self::new(config)
    }
    
    /// Get current quantum mode
    #[inline]
    pub fn current_mode(&self) -> QuantumMode {
        QuantumMode::current()
    }
    
    /// Switch quantum mode at runtime
    pub async fn switch_mode(&mut self, new_mode: QuantumMode) -> Result<(), QuantumError> {
        let old_mode = QuantumMode::current();
        
        if old_mode == new_mode {
            return Ok(());
        }
        
        // Validate mode switch
        self.validate_mode_switch(new_mode)?;
        
        // Perform mode switch
        QuantumMode::set_global(new_mode);
        
        // Update simulators if needed
        if new_mode.is_quantum_enabled() && !old_mode.is_quantum_enabled() {
            self.simulators.initialize_quantum().await?;
        } else if !new_mode.is_quantum_enabled() && old_mode.is_quantum_enabled() {
            self.simulators.shutdown_quantum().await?;
        }
        
        // Log the switch
        tracing::info!(
            "Quantum mode switched: {} -> {}",
            old_mode.description(),
            new_mode.description()
        );
        
        Ok(())
    }
    
    /// Automatically determine optimal quantum mode for a given problem
    pub fn auto_determine_mode(&self, problem_complexity: f64) -> QuantumMode {
        if !self.config.auto_switch {
            return self.config.default_mode;
        }
        
        if problem_complexity >= self.config.full_quantum_threshold {
            QuantumMode::Full
        } else if problem_complexity >= self.config.enhanced_threshold {
            QuantumMode::Enhanced
        } else {
            QuantumMode::Classical
        }
    }
    
    /// Get quantum simulators
    pub fn simulators(&self) -> &QuantumSimulators {
        &self.simulators
    }
    
    /// Get performance tracker
    pub fn performance_tracker(&self) -> Arc<QuantumPerformanceTracker> {
        self.performance_tracker.clone()
    }
    
    /// Get runtime configuration
    pub fn config(&self) -> &QuantumConfig {
        &self.config
    }
    
    /// Validate if mode switch is possible
    fn validate_mode_switch(&self, new_mode: QuantumMode) -> Result<(), QuantumError> {
        match new_mode {
            QuantumMode::Full => {
                if self.config.max_qubits < 1 {
                    return Err(QuantumError::Configuration(
                        "Full quantum mode requires at least 1 qubit".to_string()
                    ));
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Get system quantum capabilities
    pub fn get_capabilities(&self) -> QuantumCapabilities {
        QuantumCapabilities {
            max_qubits: self.config.max_qubits,
            max_circuit_depth: self.config.max_circuit_depth,
            supported_modes: vec![
                QuantumMode::Classical,
                QuantumMode::Enhanced,
                QuantumMode::Full,
            ],
            error_correction_available: self.config.error_correction,
            noise_modeling_available: self.config.noise_model.enable_decoherence,
            auto_switching_available: self.config.auto_switch,
        }
    }
}

/// System quantum capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCapabilities {
    pub max_qubits: u32,
    pub max_circuit_depth: u32,
    pub supported_modes: Vec<QuantumMode>,
    pub error_correction_available: bool,
    pub noise_modeling_available: bool,
    pub auto_switching_available: bool,
}

/// Quantum runtime errors
#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Simulation error: {0}")]
    Simulation(String),
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("Mode switch error: {0}")]
    ModeSwitch(String),
}

/// Performance tracking for quantum operations
#[derive(Debug)]
pub struct QuantumPerformanceTracker {
    classical_stats: parking_lot::RwLock<OperationStats>,
    enhanced_stats: parking_lot::RwLock<OperationStats>,
    quantum_stats: parking_lot::RwLock<OperationStats>,
}

#[derive(Debug, Default, Clone)]
pub struct OperationStats {
    pub total_operations: u64,
    pub total_time_ns: u64,
    pub average_time_ns: f64,
    pub success_count: u64,
    pub error_count: u64,
    pub quantum_advantage_factor: f64,
}

impl QuantumPerformanceTracker {
    pub fn new() -> Self {
        Self {
            classical_stats: parking_lot::RwLock::new(OperationStats::default()),
            enhanced_stats: parking_lot::RwLock::new(OperationStats::default()),
            quantum_stats: parking_lot::RwLock::new(OperationStats::default()),
        }
    }
    
    /// Record an operation's performance
    pub fn record_operation(&self, mode: QuantumMode, duration_ns: u64, success: bool) {
        let stats_lock = match mode {
            QuantumMode::Classical => &self.classical_stats,
            QuantumMode::Enhanced => &self.enhanced_stats,
            QuantumMode::Full => &self.quantum_stats,
        };
        
        let mut stats = stats_lock.write();
        stats.total_operations += 1;
        stats.total_time_ns += duration_ns;
        stats.average_time_ns = stats.total_time_ns as f64 / stats.total_operations as f64;
        
        if success {
            stats.success_count += 1;
        } else {
            stats.error_count += 1;
        }
        
        // Calculate quantum advantage relative to classical
        if mode != QuantumMode::Classical && stats.total_operations > 0 {
            let classical_stats = self.classical_stats.read();
            if classical_stats.average_time_ns > 0.0 {
                stats.quantum_advantage_factor = classical_stats.average_time_ns / stats.average_time_ns;
            }
        }
    }
    
    /// Get statistics for a specific mode
    pub fn get_stats(&self, mode: QuantumMode) -> OperationStats {
        let stats_lock = match mode {
            QuantumMode::Classical => &self.classical_stats,
            QuantumMode::Enhanced => &self.enhanced_stats,
            QuantumMode::Full => &self.quantum_stats,
        };
        
        stats_lock.read().clone()
    }
    
    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> QuantumPerformanceReport {
        QuantumPerformanceReport {
            classical: self.get_stats(QuantumMode::Classical),
            enhanced: self.get_stats(QuantumMode::Enhanced),
            quantum: self.get_stats(QuantumMode::Full),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantumPerformanceReport {
    pub classical: OperationStats,
    pub enhanced: OperationStats,
    pub quantum: OperationStats,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Convenience macros for quantum feature gating
#[macro_export]
macro_rules! quantum_gate {
    ($classical:expr, $enhanced:expr, $full:expr) => {
        match $crate::quantum::QuantumMode::current() {
            $crate::quantum::QuantumMode::Classical => $classical,
            $crate::quantum::QuantumMode::Enhanced => $enhanced,
            $crate::quantum::QuantumMode::Full => $full,
        }
    };
}

#[macro_export]
macro_rules! if_quantum {
    ($quantum_code:expr) => {
        if $crate::quantum::QuantumMode::current().is_quantum_enabled() {
            Some($quantum_code)
        } else {
            None
        }
    };
}

#[macro_export]
macro_rules! if_full_quantum {
    ($quantum_code:expr) => {
        if $crate::quantum::QuantumMode::current().is_full_quantum() {
            Some($quantum_code)
        } else {
            None
        }
    };
}

/// Initialize quantum runtime from environment
pub fn init_quantum_runtime() -> QuantumRuntime {
    let args: Vec<String> = std::env::args().collect();
    QuantumRuntime::from_args(&args)
}

/// Get or initialize global quantum runtime
pub fn quantum_runtime() -> &'static QuantumRuntime {
    use std::sync::OnceLock;
    static QUANTUM_RUNTIME: OnceLock<QuantumRuntime> = OnceLock::new();
    QUANTUM_RUNTIME.get_or_init(init_quantum_runtime)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_mode_switching() {
        assert_eq!(QuantumMode::current(), QuantumMode::Classical);
        
        QuantumMode::set_global(QuantumMode::Enhanced);
        assert_eq!(QuantumMode::current(), QuantumMode::Enhanced);
        
        QuantumMode::set_global(QuantumMode::Full);
        assert_eq!(QuantumMode::current(), QuantumMode::Full);
        
        // Reset to classical for other tests
        QuantumMode::set_global(QuantumMode::Classical);
    }
    
    #[test]
    fn test_quantum_gate_macro() {
        QuantumMode::set_global(QuantumMode::Classical);
        let result = quantum_gate!("classical", "enhanced", "quantum");
        assert_eq!(result, "classical");
        
        QuantumMode::set_global(QuantumMode::Enhanced);
        let result = quantum_gate!("classical", "enhanced", "quantum");
        assert_eq!(result, "enhanced");
        
        QuantumMode::set_global(QuantumMode::Full);
        let result = quantum_gate!("classical", "enhanced", "quantum");
        assert_eq!(result, "quantum");
        
        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }
    
    #[test]
    fn test_runtime_from_args() {
        let args = vec![
            "program".to_string(),
            "--quantum".to_string(),
            "--max-qubits".to_string(),
            "16".to_string(),
        ];
        
        let runtime = QuantumRuntime::from_args(&args);
        assert_eq!(runtime.config.default_mode, QuantumMode::Full);
        assert_eq!(runtime.config.max_qubits, 16);
    }
    
    #[test]
    fn test_performance_tracking() {
        let tracker = QuantumPerformanceTracker::new();
        
        tracker.record_operation(QuantumMode::Classical, 1000, true);
        tracker.record_operation(QuantumMode::Enhanced, 500, true);
        
        let classical_stats = tracker.get_stats(QuantumMode::Classical);
        let enhanced_stats = tracker.get_stats(QuantumMode::Enhanced);
        
        assert_eq!(classical_stats.total_operations, 1);
        assert_eq!(classical_stats.average_time_ns, 1000.0);
        
        assert_eq!(enhanced_stats.total_operations, 1);
        assert_eq!(enhanced_stats.average_time_ns, 500.0);
        assert!(enhanced_stats.quantum_advantage_factor > 1.0);
    }
}