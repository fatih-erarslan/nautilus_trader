//! CDFA Integration Module
//!
//! Seamless integration with the CDFA (Cognitive Decision Fusion Architecture)
//! parallel infrastructure for enhanced swarm intelligence capabilities.

pub mod cdfa_integration;
pub mod quantum_bridge;

pub use cdfa_integration::*;
pub use quantum_bridge::*;

/// Integration configuration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub enable_cdfa_fusion: bool,
    pub quantum_parallel_threads: Option<usize>,
    pub memory_sharing: bool,
    pub cross_algorithm_communication: bool,
}