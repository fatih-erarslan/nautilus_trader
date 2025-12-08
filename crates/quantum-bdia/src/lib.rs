//! Quantum Belief-Desire-Intention-Action (BDIA) Framework
//! 
//! A sophisticated multi-agent decision-making system that combines:
//! - CADM (Comprehensive Action Determination Model) for belief formation
//! - BDIA framework for intention mapping
//! - Prospect Theory for risk-adjusted decisions
//! - Quantum-inspired fusion via integration with quantum-hive
//! - Reinforcement learning through cognitive reappraisal
//! - Multi-agent consensus mechanisms

#![feature(portable_simd)]

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;
use tracing::{info, debug, warn};

pub mod factors;
pub mod prospect;
pub mod agent;
pub mod network;
pub mod quantum_fusion;
pub mod cognitive;
pub mod market_phase;
pub mod hive_integration;
pub mod pbit_fusion;

// Re-exports
pub use factors::{StandardFactors, MarketData};
pub use prospect::{ProspectTheory, ProspectValue};
pub use agent::{BDIAAgent, Belief, Desire, Intention, AgentConfig};
pub use network::{BDIANetwork, ConsensusDecision, NetworkConfig};
pub use quantum_fusion::{QuantumFusion, FusionResult};
pub use cognitive::{CognitiveReappraisal, LearningRate};
pub use market_phase::{MarketPhase, PhaseDetector};
pub use hive_integration::{BDIAHiveIntegration, EnhancedDecision, StrategicAdvice};
pub use pbit_fusion::{PBitFusion, PBitFusionConfig, PBitFusionResult};

use quantum_hive::{AutopoieticHive, QuantumQueen};

/// Main BDIA system that coordinates all components
pub struct QuantumBDIASystem {
    /// Multi-agent network
    network: Arc<RwLock<BDIANetwork>>,
    
    /// Quantum fusion engine
    quantum_fusion: Arc<QuantumFusion>,
    
    /// Market phase detector
    phase_detector: Arc<PhaseDetector>,
    
    /// Cognitive reappraisal engine
    cognitive_engine: Arc<CognitiveReappraisal>,
    
    /// Integration with quantum hive
    hive_integration: Option<Arc<BDIAHiveIntegration>>,
    
    /// System metrics
    metrics: Arc<SystemMetrics>,
}

#[derive(Debug, Default)]
pub struct SystemMetrics {
    pub total_decisions: u64,
    pub avg_confidence: f64,
    pub quantum_advantage: f64,
    pub learning_progress: f64,
}

/// Trading decision types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionType {
    Buy,
    Sell,
    Hold,
    Exit,
    Hedge,
    Increase(u32),
    Decrease(u32),
}

impl QuantumBDIASystem {
    /// Create new BDIA system with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(SystemConfig::default())
    }
    
    /// Create new BDIA system with custom configuration
    pub fn with_config(config: SystemConfig) -> Result<Self> {
        info!("🚀 Initializing Quantum BDIA System");
        
        // Create multi-agent network
        let network = Arc::new(RwLock::new(
            BDIANetwork::new(config.network_config)?
        ));
        
        // Initialize quantum fusion
        let quantum_fusion = Arc::new(QuantumFusion::new(config.quantum_config)?);
        
        // Create phase detector
        let phase_detector = Arc::new(PhaseDetector::new());
        
        // Initialize cognitive engine
        let cognitive_engine = Arc::new(CognitiveReappraisal::new(
            config.learning_config
        ));
        
        // No hive integration initially
        let hive_integration = None;
        
        // Initialize metrics
        let metrics = Arc::new(SystemMetrics::default());
        
        info!("✅ Quantum BDIA System initialized successfully");
        
        Ok(Self {
            network,
            quantum_fusion,
            phase_detector,
            cognitive_engine,
            hive_integration,
            metrics,
        })
    }
    
    /// Connect to quantum hive for enhanced decision making
    pub async fn connect_to_hive(&mut self, hive: Arc<AutopoieticHive>) -> Result<()> {
        info!("🔗 Connecting BDIA system to Quantum Hive");
        
        let integration = BDIAHiveIntegration::new(
            BDIAConfig::default(),
            hive,
            hive_integration::IntegrationConfig::default(),
        ).await?;
        
        self.hive_integration = Some(Arc::new(integration));
        
        info!("✅ Successfully connected to Quantum Hive");
        Ok(())
    }
    
    /// Make a trading decision based on market data
    pub async fn decide(&self, market_data: &MarketData) -> Result<Decision> {
        let start = std::time::Instant::now();
        
        // Detect market phase
        let phase = self.phase_detector.detect(market_data);
        debug!("Market phase: {:?}", phase);
        
        // Get network consensus decision
        let network = self.network.read();
        let consensus = network.consensus_decision(market_data, phase).await?;
        
        // Apply quantum fusion if connected to hive
        let final_decision = if let Some(integration) = &self.hive_integration {
            let enhanced = integration.process_with_hive(market_data).await?;
            Decision::from_enhanced(enhanced)
        } else {
            // Use local quantum fusion
            let fusion_result = self.quantum_fusion.fuse(consensus).await?;
            Decision::from_fusion(fusion_result)
        };
        
        // Update metrics
        self.update_metrics(&final_decision, start.elapsed());
        
        Ok(final_decision)
    }
    
    /// Apply cognitive reappraisal based on feedback
    pub async fn learn_from_feedback(
        &self,
        market_data: &MarketData,
        predicted_return: f64,
        actual_return: f64,
    ) -> Result<()> {
        info!("📚 Learning from feedback: predicted={:.3}, actual={:.3}", 
              predicted_return, actual_return);
        
        // Update all agents in the network
        let mut network = self.network.write();
        network.update_all_agents(
            market_data,
            predicted_return,
            actual_return,
            &self.cognitive_engine,
        ).await?;
        
        // Update phase detector with new information
        self.phase_detector.update(market_data, actual_return);
        
        // If connected to hive, learning is automatically shared through integration
        // (The integration handles feedback through quantum state updates)
        
        Ok(())
    }
    
    /// Get current system state for monitoring
    pub fn get_system_state(&self) -> SystemState {
        let network = self.network.read();
        
        SystemState {
            agent_count: network.agent_count(),
            current_phase: self.phase_detector.current_phase(),
            quantum_coherence: self.quantum_fusion.coherence(),
            learning_progress: self.cognitive_engine.learning_progress(),
            hive_connected: self.hive_integration.is_some(),
            metrics: self.get_metrics(),
        }
    }
    
    /// Update system metrics
    fn update_metrics(&self, decision: &Decision, latency: std::time::Duration) {
        // Implementation would update metrics atomically
    }
    
    /// Get current metrics
    fn get_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            total_decisions: 0, // Would read from atomic counters
            avg_confidence: 0.0,
            quantum_advantage: 0.0,
            learning_progress: 0.0,
        }
    }
    
    /// Process market data and return enhanced decision
    pub async fn process(&self, market_data: &MarketData) -> Result<FusionResult> {
        // Detect market phase
        let phase = self.phase_detector.detect(market_data);
        
        // Get network consensus decision
        let network = self.network.read();
        let consensus = network.consensus_decision(market_data, phase).await?;
        
        // Apply quantum fusion
        let fusion_result = self.quantum_fusion.fuse(consensus).await?;
        
        Ok(fusion_result)
    }
    
    /// Get consensus decision from network
    pub async fn get_consensus_decision(&self, market_data: &MarketData) -> Result<ConsensusDecision> {
        let phase = self.phase_detector.detect(market_data);
        let network = self.network.read();
        network.consensus_decision(market_data, phase).await
    }
    
    /// Get current market phase
    pub async fn get_market_phase(&self) -> MarketPhase {
        self.phase_detector.current_phase()
    }
    
    /// Get last market data (placeholder for now)
    pub async fn get_last_market_data(&self) -> Result<MarketData> {
        // This would store the last processed market data
        Ok(MarketData::random())
    }
    
    /// Update with feedback
    pub async fn update(
        &self,
        market_data: &MarketData,
        predicted_return: f64,
        actual_return: f64,
    ) -> Result<()> {
        self.learn_from_feedback(market_data, predicted_return, actual_return).await
    }
    
    /// Get system statistics
    pub async fn get_statistics(&self) -> SystemStatistics {
        SystemStatistics {
            total_decisions: 0,
            average_confidence: 0.0,
            decisions_with_quantum: 0,
            average_latency_us: 0.0,
        }
    }
}

/// System configuration
#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub network_config: NetworkConfig,
    pub quantum_config: QuantumConfig,
    pub learning_config: LearningConfig,
}

/// BDIA configuration for hive integration
#[derive(Debug, Clone)]
pub struct BDIAConfig {
    pub network_config: NetworkConfig,
    pub quantum_config: QuantumConfig,
    pub learning_config: LearningConfig,
}

impl Default for BDIAConfig {
    fn default() -> Self {
        Self {
            network_config: NetworkConfig::default(),
            quantum_config: QuantumConfig::default(),
            learning_config: LearningConfig::default(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            network_config: NetworkConfig::default(),
            quantum_config: QuantumConfig::default(),
            learning_config: LearningConfig::default(),
        }
    }
}

/// Quantum fusion configuration
#[derive(Debug, Clone)]
pub struct QuantumConfig {
    pub qubits: usize,
    pub shots: usize,
    pub use_hardware_acceleration: bool,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            qubits: 5,
            shots: 1024,
            use_hardware_acceleration: true,
        }
    }
}

/// Learning configuration
#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub base_learning_rate: f64,
    pub adaptive_learning: bool,
    pub momentum: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.01,
            adaptive_learning: true,
            momentum: 0.9,
        }
    }
}

/// Trading decision with metadata
#[derive(Debug, Clone)]
pub struct Decision {
    pub decision_type: DecisionType,
    pub confidence: f64,
    pub quantum_confidence: f64,
    pub intention_signal: f64,
    pub reasoning: Vec<String>,
}

impl Decision {
    /// Create decision from quantum fusion result
    fn from_fusion(fusion: FusionResult) -> Self {
        Self {
            decision_type: fusion.decision,
            confidence: fusion.classical_confidence,
            quantum_confidence: fusion.quantum_confidence,
            intention_signal: fusion.intention_signal,
            reasoning: fusion.reasoning,
        }
    }
    
    /// Create decision from enhanced result (with hive coordination)
    fn from_enhanced(enhanced: EnhancedDecision) -> Self {
        Self {
            decision_type: enhanced.decision,
            confidence: enhanced.classical_confidence,
            quantum_confidence: enhanced.quantum_confidence,
            intention_signal: 0.0, // Not provided in enhanced decision
            reasoning: enhanced.reasoning,
        }
    }
}

/// System state for monitoring
#[derive(Debug, Clone)]
pub struct SystemState {
    pub agent_count: usize,
    pub current_phase: MarketPhase,
    pub quantum_coherence: f64,
    pub learning_progress: f64,
    pub hive_connected: bool,
    pub metrics: SystemMetrics,
}

/// System statistics
#[derive(Debug, Clone)]
pub struct SystemStatistics {
    pub total_decisions: u64,
    pub average_confidence: f64,
    pub decisions_with_quantum: u64,
    pub average_latency_us: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_creation() {
        let system = QuantumBDIASystem::new().unwrap();
        let state = system.get_system_state();
        
        assert!(state.agent_count > 0);
        assert!(!state.hive_connected);
    }
    
    #[tokio::test]
    async fn test_decision_making() {
        let system = QuantumBDIASystem::new().unwrap();
        
        let market_data = MarketData::random();
        
        let decision = system.decide(&market_data).await.unwrap();
        
        assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
        assert!(decision.quantum_confidence > 0.0 && decision.quantum_confidence <= 1.0);
    }
}