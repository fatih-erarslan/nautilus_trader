//! Factory functions for creating PADS system components

use crate::core::config::PadsConfig;
use crate::core::pads::PanarchyAdaptiveDecisionSystem;
use crate::error::PadsResult;
use std::sync::Arc;

/// Factory for creating PADS system components
pub struct PadsFactory;

impl PadsFactory {
    /// Create a new PADS system with default configuration
    pub async fn create_default() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let config = PadsConfig::default();
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a new PADS system with custom configuration
    pub async fn create_with_config(config: PadsConfig) -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        config.validate()?;
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a high-performance PADS system
    pub async fn create_high_performance() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.performance.decision_latency_ns = 5_000; // 5 microseconds
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.performance.cache_size = 50_000;
        config.agents.count = 16; // More agents for better performance
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a low-latency PADS system
    pub async fn create_low_latency() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.performance.decision_latency_ns = 1_000; // 1 microsecond
        config.performance.analysis_latency_ns = 500;   // 500 nanoseconds
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.agents.count = 8; // Fewer agents for lower latency
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a conservative PADS system with enhanced risk management
    pub async fn create_conservative() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.risk.max_risk_tolerance = 0.02; // 2% maximum risk
        config.risk.enable_black_swan_detection = true;
        config.risk.assessment_window_seconds = 600; // 10 minutes
        config.enable_risk_management = true;
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create an aggressive PADS system
    pub async fn create_aggressive() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.risk.max_risk_tolerance = 0.15; // 15% maximum risk
        config.performance.decision_latency_ns = 2_000; // 2 microseconds
        config.agents.count = 20; // More agents for complex decisions
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a research-oriented PADS system
    pub async fn create_research() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.enable_pattern_analysis = true;
        config.enable_panarchy_system = true;
        config.analysis.enable_antifragility = true;
        config.analysis.enable_panarchy = true;
        config.analysis.enable_narrative_forecasting = true;
        config.analysis.window_size = 1000;
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
    
    /// Create a minimal PADS system for testing
    pub async fn create_minimal() -> PadsResult<PanarchyAdaptiveDecisionSystem> {
        let mut config = PadsConfig::default();
        config.agents.count = 4;
        config.enable_quantum_agents = false;
        config.enable_pattern_analysis = false;
        config.enable_panarchy_system = false;
        config.enable_board_system = false;
        
        PanarchyAdaptiveDecisionSystem::new_with_config(config).await
    }
}

/// Create a PADS system optimized for a specific use case
pub async fn create_optimized_for_use_case(use_case: &str) -> PadsResult<PanarchyAdaptiveDecisionSystem> {
    match use_case {
        "high_frequency" => PadsFactory::create_low_latency().await,
        "algorithmic_trading" => PadsFactory::create_high_performance().await,
        "risk_management" => PadsFactory::create_conservative().await,
        "research" => PadsFactory::create_research().await,
        "testing" => PadsFactory::create_minimal().await,
        _ => PadsFactory::create_default().await,
    }
}

/// Create a PADS system from configuration file
pub async fn create_from_config_file(config_path: &str) -> PadsResult<PanarchyAdaptiveDecisionSystem> {
    let config = PadsConfig::from_file(config_path)?;
    PadsFactory::create_with_config(config).await
}