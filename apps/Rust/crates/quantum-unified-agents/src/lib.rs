//! Unified Quantum Agents for Nautilus Trader
//!
//! This crate provides unified implementations of all quantum agents that implement
//! the QuantumAgent trait, enabling seamless PADS integration and orchestration.

pub mod quantum_agentic_reasoning_agent;
pub mod quantum_hedge_agent;
pub mod quantum_lmsr_agent;
pub mod quantum_prospect_agent;
pub mod quantum_nqo_agent;
pub mod pads_integration;
pub mod unified_registry;
pub mod swarm_enhanced_coordination;

// Re-export all agents
pub use quantum_agentic_reasoning_agent::UnifiedQARAgent;
pub use quantum_hedge_agent::UnifiedQuantumHedgeAgent;
// pub use quantum_lmsr_agent::UnifiedQuantumLMSRAgent;
// pub use quantum_prospect_agent::UnifiedQuantumProspectAgent;
// pub use quantum_nqo_agent::UnifiedNQOAgent;
pub use pads_integration::{PADSSignalAggregator, PADSIntegrationManager, AggregationStrategy};
pub use unified_registry::UnifiedQuantumAgentRegistry;
pub use swarm_enhanced_coordination::{
    SwarmEnhancedQuantumCoordinator, OptimizationObjective, SwarmIndividual,
    CoordinationMetrics, PerformanceMetrics
};

// Re-export core types
pub use quantum_core::{
    QuantumAgent, QuantumSignal, PADSSignal, PADSAction, MarketData, LatticeState,
    QuantumConfig, AgentHealth, AgentMetrics, QuantumResult, QuantumError,
    HealthStatus, DecoherenceEvent
};