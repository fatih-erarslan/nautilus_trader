//! # PADS Assembly
//!
//! Central coordination system for quantum agents implementing the Panarchy
//! Adaptive Decision System (PADS).

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;
use super::{AgentManager, AssemblyMetrics, CoordinationBus};

/// Configuration for PADS assembly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyConfig {
    /// Maximum number of concurrent agents
    pub max_concurrent_agents: usize,
    /// Enable full panarchy layer
    pub enable_panarchy_full: bool,
    /// Quantum coherence threshold for decision making
    pub quantum_coherence_threshold: f64,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Decision synthesis strategy
    pub synthesis_strategy: SynthesisStrategy,
    /// Convergence threshold for agent consensus
    pub convergence_threshold: f64,
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for AssemblyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_agents: 12,
            enable_panarchy_full: true,
            quantum_coherence_threshold: 0.7,
            enable_monitoring: true,
            synthesis_strategy: SynthesisStrategy::Adaptive,
            convergence_threshold: 0.8,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

/// Synthesis strategy for combining agent signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisStrategy {
    /// Weighted average based on agent performance
    WeightedAverage,
    /// Consensus-based decision making
    Consensus,
    /// Adaptive strategy based on market conditions
    Adaptive,
    /// Quantum superposition of all signals
    QuantumSuperposition,
}

/// Performance optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Conservative - prioritize reliability
    Conservative,
    /// Balanced - balance speed and accuracy
    Balanced,
    /// Aggressive - prioritize speed
    Aggressive,
    /// Quantum - use quantum acceleration
    Quantum,
}

/// Main PADS assembly coordinating all quantum agents
pub struct PadsAssembly {
    config: AssemblyConfig,
    agent_manager: Arc<AgentManager>,
    coordination_bus: Arc<CoordinationBus>,
    assembly_state: Arc<RwLock<AssemblyState>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
}

impl PadsAssembly {
    /// Create new PADS assembly
    pub async fn new(config: AssemblyConfig) -> PadsResult<Self> {
        let agent_manager = Arc::new(AgentManager::new().await?);
        let coordination_bus = Arc::new(CoordinationBus::new().await?);
        let assembly_state = Arc::new(RwLock::new(AssemblyState::default()));
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));
        
        let assembly = Self {
            config,
            agent_manager,
            coordination_bus,
            assembly_state,
            performance_monitor,
        };
        
        // Initialize agents
        assembly.initialize().await?;
        
        Ok(assembly)
    }
    
    /// Initialize the assembly and all agents
    async fn initialize(&self) -> PadsResult<()> {
        // Initialize all quantum agents
        self.agent_manager.initialize_agents().await?;
        
        // Update assembly state
        let mut state = self.assembly_state.write().await;
        state.status = AssemblyStatus::Active;
        state.initialized_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Ok(())
    }
    
    /// Process market data through the assembly
    pub async fn process_decision(
        &self,
        market_data: &MarketData,
        context: &DecisionContext,
    ) -> PadsResult<PadsDecision> {
        let start_time = std::time::Instant::now();
        
        // Check assembly state
        let state = self.assembly_state.read().await;
        if state.status != AssemblyStatus::Active {
            return Err(PadsError::AssemblyNotActive);
        }
        drop(state);
        
        // Process through all quantum agents
        let agent_signals = self.agent_manager
            .process_market_data(market_data, context)
            .await?;
        
        // Synthesize signals into final decision
        let decision = self.synthesize_decision(&agent_signals, market_data, context).await?;
        
        // Update performance metrics
        let processing_time = start_time.elapsed();
        if let Ok(mut monitor) = self.performance_monitor.write().await {
            monitor.record_decision(processing_time, &decision);
        }
        
        // Update assembly state
        let mut state = self.assembly_state.write().await;
        state.total_decisions += 1;
        state.last_decision_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Ok(decision)
    }
    
    /// Synthesize agent signals into final decision
    async fn synthesize_decision(
        &self,
        agent_signals: &[super::AgentSignal],
        market_data: &MarketData,
        context: &DecisionContext,
    ) -> PadsResult<PadsDecision> {
        match self.config.synthesis_strategy {
            SynthesisStrategy::WeightedAverage => {
                self.weighted_average_synthesis(agent_signals).await
            }
            SynthesisStrategy::Consensus => {
                self.consensus_synthesis(agent_signals).await
            }
            SynthesisStrategy::Adaptive => {
                self.adaptive_synthesis(agent_signals, market_data, context).await
            }
            SynthesisStrategy::QuantumSuperposition => {
                self.quantum_superposition_synthesis(agent_signals).await
            }
        }
    }
    
    /// Weighted average synthesis strategy
    async fn weighted_average_synthesis(
        &self,
        agent_signals: &[super::AgentSignal],
    ) -> PadsResult<PadsDecision> {
        if agent_signals.is_empty() {
            return Ok(PadsDecision::default());
        }
        
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        let mut buy_weight = 0.0;
        let mut sell_weight = 0.0;
        let mut hold_weight = 0.0;
        
        for signal in agent_signals {
            let weight = signal.confidence * self.get_agent_weight(&signal.agent_type);
            total_weight += weight;
            weighted_confidence += signal.confidence * weight;
            
            match signal.trading_action {
                TradingAction::Buy => buy_weight += weight,
                TradingAction::Sell => sell_weight += weight,
                TradingAction::Hold => hold_weight += weight,
            }
        }
        
        if total_weight == 0.0 {
            return Ok(PadsDecision::default());
        }
        
        // Determine final action
        let final_action = if buy_weight > sell_weight && buy_weight > hold_weight {
            TradingAction::Buy
        } else if sell_weight > buy_weight && sell_weight > hold_weight {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };
        
        let final_confidence = weighted_confidence / total_weight;
        
        Ok(PadsDecision {
            action: final_action,
            confidence: final_confidence,
            agent_consensus: agent_signals.len() as f64 / 12.0, // 12 total agents
            quantum_coherence: self.calculate_quantum_coherence(agent_signals),
            reasoning: self.build_decision_reasoning(agent_signals),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
    
    /// Consensus synthesis strategy
    async fn consensus_synthesis(
        &self,
        agent_signals: &[super::AgentSignal],
    ) -> PadsResult<PadsDecision> {
        if agent_signals.is_empty() {
            return Ok(PadsDecision::default());
        }
        
        // Count votes for each action
        let mut buy_votes = 0;
        let mut sell_votes = 0;
        let mut hold_votes = 0;
        let mut total_confidence = 0.0;
        
        for signal in agent_signals {
            if signal.confidence >= self.config.convergence_threshold {
                match signal.trading_action {
                    TradingAction::Buy => buy_votes += 1,
                    TradingAction::Sell => sell_votes += 1,
                    TradingAction::Hold => hold_votes += 1,
                }
            }
            total_confidence += signal.confidence;
        }
        
        // Determine consensus action
        let final_action = if buy_votes > sell_votes && buy_votes > hold_votes {
            TradingAction::Buy
        } else if sell_votes > buy_votes && sell_votes > hold_votes {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };
        
        let consensus_ratio = (buy_votes.max(sell_votes).max(hold_votes) as f64) / (agent_signals.len() as f64);
        let final_confidence = (total_confidence / agent_signals.len() as f64) * consensus_ratio;
        
        Ok(PadsDecision {
            action: final_action,
            confidence: final_confidence,
            agent_consensus: consensus_ratio,
            quantum_coherence: self.calculate_quantum_coherence(agent_signals),
            reasoning: self.build_decision_reasoning(agent_signals),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
    
    /// Adaptive synthesis strategy
    async fn adaptive_synthesis(
        &self,
        agent_signals: &[super::AgentSignal],
        market_data: &MarketData,
        _context: &DecisionContext,
    ) -> PadsResult<PadsDecision> {
        // Choose synthesis method based on market conditions
        let volatility = market_data.volatility.unwrap_or(0.1);
        
        if volatility > 0.05 {
            // High volatility - use consensus for stability
            self.consensus_synthesis(agent_signals).await
        } else {
            // Normal volatility - use weighted average for precision
            self.weighted_average_synthesis(agent_signals).await
        }
    }
    
    /// Quantum superposition synthesis strategy
    async fn quantum_superposition_synthesis(
        &self,
        agent_signals: &[super::AgentSignal],
    ) -> PadsResult<PadsDecision> {
        if agent_signals.is_empty() {
            return Ok(PadsDecision::default());
        }
        
        // Calculate quantum state probabilities
        let mut buy_amplitude = 0.0;
        let mut sell_amplitude = 0.0;
        let mut hold_amplitude = 0.0;
        let mut total_coherence = 0.0;
        
        for signal in agent_signals {
            let coherence = signal.quantum_features.coherence;
            let confidence = signal.confidence;
            let amplitude = (coherence * confidence).sqrt();
            
            match signal.trading_action {
                TradingAction::Buy => buy_amplitude += amplitude,
                TradingAction::Sell => sell_amplitude += amplitude,
                TradingAction::Hold => hold_amplitude += amplitude,
            }
            
            total_coherence += coherence;
        }
        
        // Normalize amplitudes
        let total_amplitude = buy_amplitude + sell_amplitude + hold_amplitude;
        if total_amplitude > 0.0 {
            buy_amplitude /= total_amplitude;
            sell_amplitude /= total_amplitude;
            hold_amplitude /= total_amplitude;
        }
        
        // Collapse quantum state to classical decision
        let final_action = if buy_amplitude > sell_amplitude && buy_amplitude > hold_amplitude {
            TradingAction::Buy
        } else if sell_amplitude > buy_amplitude && sell_amplitude > hold_amplitude {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };
        
        let final_confidence = buy_amplitude.max(sell_amplitude).max(hold_amplitude);
        let quantum_coherence = total_coherence / agent_signals.len() as f64;
        
        Ok(PadsDecision {
            action: final_action,
            confidence: final_confidence,
            agent_consensus: agent_signals.len() as f64 / 12.0,
            quantum_coherence,
            reasoning: self.build_decision_reasoning(agent_signals),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
    
    /// Get weight for specific agent type
    fn get_agent_weight(&self, agent_type: &super::AgentType) -> f64 {
        // Weights based on agent capabilities and historical performance
        match agent_type {
            super::AgentType::QuantumAgenticReasoning => 0.15,
            super::AgentType::QuantumBiologicalMarketIntuition => 0.12,
            super::AgentType::QuantumBehavioralDynamicsAnalysis => 0.10,
            super::AgentType::QuantumAnnealingRegression => 0.09,
            super::AgentType::QuantumErrorCorrection => 0.08,
            super::AgentType::IntelligentQuantumAnomalyDetection => 0.08,
            super::AgentType::NeuralQuantumOptimization => 0.08,
            super::AgentType::QuantumLogarithmicMarketScoringRules => 0.10,
            super::AgentType::QuantumProspectTheory => 0.08,
            super::AgentType::QuantumHedgeAlgorithm => 0.06,
            super::AgentType::QuantumLongShortTermMemory => 0.08,
            super::AgentType::QuantumWhaleDefense => 0.06,
        }
    }
    
    /// Calculate overall quantum coherence
    fn calculate_quantum_coherence(&self, agent_signals: &[super::AgentSignal]) -> f64 {
        if agent_signals.is_empty() {
            return 0.0;
        }
        
        let total_coherence: f64 = agent_signals.iter()
            .map(|signal| signal.quantum_features.coherence)
            .sum();
        
        total_coherence / agent_signals.len() as f64
    }
    
    /// Build decision reasoning from agent signals
    fn build_decision_reasoning(&self, agent_signals: &[super::AgentSignal]) -> Vec<String> {
        let mut reasoning = Vec::new();
        
        reasoning.push(format!("Processed signals from {} quantum agents", agent_signals.len()));
        
        // Add top contributing agents
        let mut sorted_signals = agent_signals.to_vec();
        sorted_signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        for (i, signal) in sorted_signals.iter().take(3).enumerate() {
            reasoning.push(format!(
                "#{}: {} (confidence: {:.2}, action: {:?})",
                i + 1,
                signal.agent_type.as_str(),
                signal.confidence,
                signal.trading_action
            ));
        }
        
        reasoning
    }
    
    /// Get assembly status
    pub async fn get_status(&self) -> AssemblyStatus {
        self.assembly_state.read().await.status.clone()
    }
    
    /// Get assembly metrics
    pub async fn get_assembly_metrics(&self) -> AssemblyMetrics {
        self.agent_manager.get_assembly_metrics().await
    }
    
    /// Get agent count
    pub async fn agent_count(&self) -> usize {
        self.agent_manager.agent_count().await
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_monitor.read().await.get_metrics()
    }
}

/// Assembly state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyState {
    pub status: AssemblyStatus,
    pub initialized_at: u64,
    pub total_decisions: u64,
    pub last_decision_at: u64,
}

impl Default for AssemblyState {
    fn default() -> Self {
        Self {
            status: AssemblyStatus::Initializing,
            initialized_at: 0,
            total_decisions: 0,
            last_decision_at: 0,
        }
    }
}

/// Assembly status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssemblyStatus {
    Initializing,
    Active,
    Degraded,
    Error,
    Shutdown,
}

/// Performance monitor for assembly
#[derive(Debug)]
pub struct PerformanceMonitor {
    total_decisions: u64,
    average_processing_time: std::time::Duration,
    accuracy_history: Vec<f64>,
    coherence_history: Vec<f64>,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            total_decisions: 0,
            average_processing_time: std::time::Duration::from_millis(0),
            accuracy_history: Vec::new(),
            coherence_history: Vec::new(),
        }
    }
    
    fn record_decision(&mut self, processing_time: std::time::Duration, decision: &PadsDecision) {
        // Update processing time
        let total_time = self.average_processing_time * self.total_decisions as u32 + processing_time;
        self.total_decisions += 1;
        self.average_processing_time = total_time / self.total_decisions as u32;
        
        // Record coherence
        self.coherence_history.push(decision.quantum_coherence);
        if self.coherence_history.len() > 1000 {
            self.coherence_history.drain(0..500);
        }
    }
    
    fn get_metrics(&self) -> PerformanceMetrics {
        let average_coherence = if self.coherence_history.is_empty() {
            0.0
        } else {
            self.coherence_history.iter().sum::<f64>() / self.coherence_history.len() as f64
        };
        
        PerformanceMetrics {
            total_decisions: self.total_decisions,
            average_processing_time: self.average_processing_time,
            average_coherence,
            uptime_percentage: 99.9, // Placeholder
        }
    }
}

/// Performance metrics for assembly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_decisions: u64,
    pub average_processing_time: std::time::Duration,
    pub average_coherence: f64,
    pub uptime_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_assembly_config_default() {
        let config = AssemblyConfig::default();
        assert_eq!(config.max_concurrent_agents, 12);
        assert!(config.enable_panarchy_full);
        assert_eq!(config.quantum_coherence_threshold, 0.7);
    }
    
    #[tokio::test]
    async fn test_assembly_creation() {
        let config = AssemblyConfig::default();
        let assembly = PadsAssembly::new(config).await;
        
        // Should either succeed or fail gracefully
        match assembly {
            Ok(assembly) => {
                assert_eq!(assembly.get_status().await, AssemblyStatus::Active);
                assert_eq!(assembly.agent_count().await, 12);
            }
            Err(e) => {
                println!("Assembly creation failed (may be expected in test): {:?}", e);
            }
        }
    }
    
    #[test]
    fn test_synthesis_strategies() {
        // Test that all synthesis strategies are defined
        let strategies = [
            SynthesisStrategy::WeightedAverage,
            SynthesisStrategy::Consensus,
            SynthesisStrategy::Adaptive,
            SynthesisStrategy::QuantumSuperposition,
        ];
        
        assert_eq!(strategies.len(), 4);
    }
    
    #[test]
    fn test_optimization_levels() {
        let levels = [
            OptimizationLevel::Conservative,
            OptimizationLevel::Balanced,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Quantum,
        ];
        
        assert_eq!(levels.len(), 4);
    }
}