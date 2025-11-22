//! # Parasitic Trading System
//! 
//! Advanced parasitic trading system with quantum capabilities and biomimetic organisms.
//! This is a minimal working version focused on core pairlist functionality.

pub mod quantum;
pub mod pairlist;

// Re-export core functionality
pub use pairlist::{
    TradingPair,
    SelectedPair,
    ParasiticPattern,
    HostType,
    ExploitationStrategy,
    CQGSComplianceMetrics,
    WhaleNestDetector,
    WhaleNest,
    ZombiePairDetector,
    ZombiePair,
    MycelialNetworkAnalyzer,
    CorrelationNetwork,
    ParasiticResourceManager,
};

pub use quantum::{
    QuantumMode,
    QuantumConfig,
    init_quantum_runtime,
};

use serde::{Serialize, Deserialize};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Basic parasitic system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticConfig {
    pub quantum_enabled: bool,
    pub simd_enabled: bool,
    pub max_pairs: usize,
    pub sensitivity: f64,
}

impl Default for ParasiticConfig {
    fn default() -> Self {
        Self {
            quantum_enabled: false,
            simd_enabled: true,
            max_pairs: 100,
            sensitivity: 0.8,
        }
    }
}

/// Initialize the parasitic system with quantum runtime
pub fn init_parasitic_system() -> Result<ParasiticConfig, Box<dyn std::error::Error>> {
    // Initialize quantum runtime
    let _runtime = init_quantum_runtime();
    
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    let config = ParasiticConfig::default();
    tracing::info!("Parasitic trading system v{} initialized", VERSION);
    Ok(config)
}

/// Create a whale nest detector with default configuration
pub fn create_whale_detector(sensitivity: f64, min_whale_size: f64) -> WhaleNestDetector {
    WhaleNestDetector::new(sensitivity, min_whale_size)
}

/// Create a zombie pair detector with default configuration
pub fn create_zombie_detector(sensitivity: f64, min_manipulation: f64) -> ZombiePairDetector {
    ZombiePairDetector::new(sensitivity, min_manipulation)
}

/// Create a mycelial network analyzer with default configuration
pub fn create_mycelial_analyzer(sensitivity: f64, min_correlation: f64) -> MycelialNetworkAnalyzer {
    MycelialNetworkAnalyzer::new(sensitivity, min_correlation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_initialization() {
        let result = init_parasitic_system();
        assert!(result.is_ok());
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_whale_detector_creation() {
        let detector = create_whale_detector(0.8, 1000000.0);
        assert_eq!(detector.sensitivity, 0.8);
        assert_eq!(detector.min_whale_size, 1000000.0);
    }

    #[test]
    fn test_zombie_detector_creation() {
        let detector = create_zombie_detector(0.9, 0.7);
        assert_eq!(detector.sensitivity, 0.9);
        assert_eq!(detector.min_manipulation_score, 0.7);
    }

    #[test]
    fn test_mycelial_analyzer_creation() {
        let analyzer = create_mycelial_analyzer(0.8, 0.3);
        assert_eq!(analyzer.sensitivity, 0.8);
        assert_eq!(analyzer.min_correlation, 0.3);
    }
}