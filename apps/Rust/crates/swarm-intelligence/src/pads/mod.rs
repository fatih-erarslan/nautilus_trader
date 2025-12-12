//! # Panarchy Adaptive Decision System (PADS)
//!
//! Enterprise-grade adaptive decision system based on panarchy theory and
//! hierarchical decision-making frameworks. Integrates with swarm intelligence
//! algorithms to provide resilient, adaptive, and emergent decision capabilities.
//!
//! ## Core Components
//!
//! - **Panarchy Framework**: Multi-scale adaptive cycle modeling
//! - **Decision Engine**: Multi-criteria adaptive decision analysis  
//! - **Integration Layer**: Swarm algorithm coordination
//! - **Governance**: Autonomous system governance and control
//! - **Monitoring**: Real-time performance and adaptation tracking
//!
//! ## Architecture
//!
//! The PADS system operates across four hierarchical decision layers:
//! 1. **Tactical**: Immediate response and execution
//! 2. **Operational**: Short-term optimization and coordination  
//! 3. **Strategic**: Medium-term planning and resource allocation
//! 4. **Meta-Strategic**: Long-term adaptation and transformation
//!
//! Each layer implements adaptive cycles with four phases:
//! - Growth (r): Rapid expansion and exploitation
//! - Conservation (K): Optimization and consolidation
//! - Release (Ω): Creative destruction and innovation
//! - Reorganization (α): Renewal and transformation

pub mod core;
pub mod panarchy;
pub mod decision_engine;
pub mod integration;
pub mod governance;
pub mod monitoring;

// Panarchy-specific modules
pub mod adaptive_cycles;
pub mod resilience;
pub mod emergence;
pub mod transformation;

// Re-exports for public API
pub use core::{
    PadsSystem, PadsConfig, DecisionLayer, DecisionContext,
    PadsError, PadsResult
};

pub use panarchy::{
    PanarchyFramework, AdaptiveCycle, CrossScaleInteraction,
    PanarchyState, AdaptiveCyclePhase
};

pub use decision_engine::{
    AdaptiveDecisionEngine, DecisionTree, MultiCriteriaAnalysis,
    DecisionOptimizer, UncertaintyQuantifier
};

pub use integration::{
    SwarmIntegration, QuantumAgentBridge, CdfaCoordinator,
    PerformanceFeedback, SystemCoordinator
};

pub use governance::{
    AutonomousGovernance, PolicyEngine, ComplianceMonitor,
    SecurityManager, GovernanceFramework
};

pub use monitoring::{
    RealTimeMonitor, PerformanceAnalyzer, AdaptationTracker,
    SystemHealthMonitor, MetricsCollector
};

// Adaptive cycle components
pub use adaptive_cycles::{
    GrowthPhase, ConservationPhase, ReleasePhase, ReorganizationPhase,
    CycleTransition, CycleMetrics
};

pub use resilience::{
    ResilienceEngine, SystemResilience, ResilienceMetrics,
    FailureRecovery, AdaptiveCapacity
};

pub use emergence::{
    EmergenceDetector, EmergentBehavior, EmergenceAnalyzer,
    ComplexityMeasure, EmergentPatterns
};

pub use transformation::{
    TransformationEngine, TransformationPathway, AdaptiveTransformation,
    SystemEvolution, TransformationMetrics
};

/// Version and metadata
pub const PADS_VERSION: &str = "1.0.0";
pub const PADS_BUILD: &str = env!("CARGO_PKG_VERSION");

/// Initialize the PADS system with default configuration
pub async fn init_pads() -> PadsResult<PadsSystem> {
    let config = PadsConfig::default();
    PadsSystem::new(config).await
}

/// Initialize PADS with custom configuration
pub async fn init_pads_with_config(config: PadsConfig) -> PadsResult<PadsSystem> {
    PadsSystem::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pads_initialization() {
        let pads = init_pads().await.unwrap();
        assert!(pads.is_healthy());
    }
    
    #[tokio::test]
    async fn test_custom_config_initialization() {
        let config = PadsConfig::builder()
            .with_decision_layers(4)
            .with_adaptive_cycles(true)
            .with_real_time_monitoring(true)
            .build();
            
        let pads = init_pads_with_config(config).await.unwrap();
        assert!(pads.is_healthy());
    }
}