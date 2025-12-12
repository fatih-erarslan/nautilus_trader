//! Integration with quantum-hive for BDIA system

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{info, debug, warn};

use crate::{
    QuantumBDIASystem, DecisionType, BDIAConfig,
    factors::MarketData,
    network::ConsensusDecision,
    quantum_fusion::FusionResult,
};

use quantum_hive::{
    QuantumQueen, AutopoieticHive, HiveMessage, HiveResponse,
    QuantumState as HiveQuantumState, MarketState as HiveMarketState,
    Decision as HiveDecision, DecisionType as HiveDecisionType,
};

/// Bridge between BDIA and quantum-hive
pub struct BDIAHiveIntegration {
    /// BDIA system
    bdia_system: Arc<QuantumBDIASystem>,
    
    /// Channel to send decisions to hive
    hive_tx: mpsc::Sender<HiveMessage>,
    
    /// Channel to receive feedback from hive
    feedback_rx: mpsc::Receiver<HiveResponse>,
    
    /// Integration configuration
    config: IntegrationConfig,
}

/// Integration configuration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable bidirectional communication
    pub bidirectional: bool,
    /// Decision confidence threshold
    pub confidence_threshold: f64,
    /// Enable quantum fusion with hive
    pub quantum_fusion: bool,
    /// Maximum latency for decisions (microseconds)
    pub max_latency_us: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            bidirectional: true,
            confidence_threshold: 0.7,
            quantum_fusion: true,
            max_latency_us: 10_000, // 10ms
        }
    }
}

impl BDIAHiveIntegration {
    /// Create new integration bridge
    pub async fn new(
        bdia_config: BDIAConfig,
        hive: Arc<AutopoieticHive>,
        config: IntegrationConfig,
    ) -> Result<Self> {
        info!("Creating BDIA-Hive integration bridge");
        
        // Create BDIA system
        let bdia_system = Arc::new(QuantumBDIASystem::new(bdia_config).await?);
        
        // Create communication channels
        let (hive_tx, mut hive_rx) = mpsc::channel(100);
        let (feedback_tx, feedback_rx) = mpsc::channel(100);
        
        // Register BDIA as a hive component
        hive.register_component("quantum-bdia", hive_tx.clone()).await?;
        
        // Spawn hive message handler
        let bdia = bdia_system.clone();
        let fb_tx = feedback_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = hive_rx.recv().await {
                match msg {
                    HiveMessage::MarketUpdate(market_state) => {
                        // Convert hive market state to BDIA format
                        let market_data = Self::convert_market_state(&market_state);
                        
                        // Process through BDIA
                        match bdia.process(&market_data).await {
                            Ok(decision) => {
                                let response = HiveResponse::Decision(
                                    Self::convert_to_hive_decision(decision)
                                );
                                let _ = fb_tx.send(response).await;
                            }
                            Err(e) => {
                                warn!("BDIA processing error: {}", e);
                            }
                        }
                    }
                    HiveMessage::QuantumState(quantum_state) => {
                        // Use quantum state for enhanced processing
                        debug!("Received quantum state from hive");
                        // Could enhance BDIA quantum fusion with hive quantum state
                    }
                    HiveMessage::Feedback { predicted, actual } => {
                        // Update BDIA with performance feedback
                        if let Ok(market_data) = bdia.get_last_market_data().await {
                            let _ = bdia.update(&market_data, predicted, actual).await;
                        }
                    }
                    _ => {}
                }
            }
        });
        
        Ok(Self {
            bdia_system,
            hive_tx,
            feedback_rx,
            config,
        })
    }
    
    /// Process market data through BDIA with hive coordination
    pub async fn process_with_hive(
        &self,
        market_data: &MarketData,
    ) -> Result<EnhancedDecision> {
        let start = std::time::Instant::now();
        
        // Get BDIA decision
        let bdia_result = self.bdia_system.process(market_data).await?;
        
        // If quantum fusion is enabled and we have high confidence, 
        // coordinate with hive quantum state
        let enhanced_confidence = if self.config.quantum_fusion 
            && bdia_result.quantum_confidence > self.config.confidence_threshold {
            
            // Request quantum state from hive
            self.hive_tx.send(HiveMessage::RequestQuantumState).await?;
            
            // Wait for quantum state (with timeout)
            match tokio::time::timeout(
                tokio::time::Duration::from_micros(self.config.max_latency_us / 2),
                self.feedback_rx.try_recv()
            ).await {
                Ok(Some(HiveResponse::QuantumState(hive_quantum))) => {
                    // Merge quantum confidences
                    let merged = Self::merge_quantum_states(
                        &bdia_result.quantum_state,
                        &hive_quantum,
                    );
                    merged
                }
                _ => bdia_result.quantum_confidence,
            }
        } else {
            bdia_result.quantum_confidence
        };
        
        let latency_us = start.elapsed().as_micros() as u64;
        
        // Send decision to hive if confidence is high enough
        if enhanced_confidence > self.config.confidence_threshold {
            let hive_decision = Self::convert_to_hive_decision(bdia_result.clone());
            self.hive_tx.send(HiveMessage::ComponentDecision {
                component: "quantum-bdia".to_string(),
                decision: hive_decision,
            }).await?;
        }
        
        Ok(EnhancedDecision {
            decision: bdia_result.decision,
            classical_confidence: bdia_result.classical_confidence,
            quantum_confidence: enhanced_confidence,
            hive_coordinated: enhanced_confidence > bdia_result.quantum_confidence,
            latency_us,
            reasoning: bdia_result.reasoning,
        })
    }
    
    /// Enable BDIA to act as strategic advisor to QuantumQueen
    pub async fn advise_queen(
        &self,
        queen: &QuantumQueen,
        market_data: &MarketData,
    ) -> Result<StrategicAdvice> {
        debug!("BDIA advising QuantumQueen");
        
        // Get BDIA's multi-agent consensus
        let consensus = self.bdia_system.get_consensus_decision(market_data).await?;
        
        // Analyze agent agreement patterns
        let agreement_analysis = Self::analyze_agent_agreement(&consensus);
        
        // Generate strategic advice based on BDIA's cognitive patterns
        let advice = StrategicAdvice {
            primary_recommendation: consensus.decision,
            confidence: consensus.confidence,
            market_phase: self.bdia_system.get_market_phase().await,
            agent_consensus: agreement_analysis,
            risk_assessment: Self::assess_risk(&consensus),
            opportunity_score: Self::calculate_opportunity(&consensus),
            reasoning: vec![
                format!("BDIA Network Consensus: {} agents", 
                    consensus.agent_decisions.len()),
                format!("Weighted Intention: {:.3}", consensus.weighted_intention),
                format!("Market Phase: {:?}", self.bdia_system.get_market_phase().await),
                format!("Confidence: {:.2}%", consensus.confidence * 100.0),
            ],
        };
        
        // Send strategic advice to queen
        queen.receive_strategic_input("bdia", advice.clone()).await?;
        
        Ok(advice)
    }
    
    /// Convert market state formats
    fn convert_market_state(hive_state: &HiveMarketState) -> MarketData {
        MarketData {
            sentiment: hive_state.sentiment,
            risk_appetite: hive_state.risk_level,
            correlation: hive_state.correlation.unwrap_or(0.0),
            microstructure: hive_state.microstructure.unwrap_or(0.0),
            volume_profile: hive_state.volume_trend,
            flow_imbalance: hive_state.flow_imbalance.unwrap_or(0.0),
            momentum: hive_state.momentum,
            trend: hive_state.trend,
            volatility: hive_state.volatility,
            cycle: hive_state.market_regime as f64 * 0.25 - 0.5, // Map regime to cycle
            fractal: 0.0, // Not available in hive state
            entropy: 0.0, // Calculate if needed
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Convert BDIA decision to hive format
    fn convert_to_hive_decision(result: FusionResult) -> HiveDecision {
        let action = match result.decision {
            DecisionType::Buy => HiveDecisionType::Buy,
            DecisionType::Sell => HiveDecisionType::Sell,
            DecisionType::Hold => HiveDecisionType::Hold,
            DecisionType::Increase(pct) => HiveDecisionType::ScaleIn(pct as f64 / 100.0),
            DecisionType::Decrease(pct) => HiveDecisionType::ScaleOut(pct as f64 / 100.0),
            DecisionType::Hedge => HiveDecisionType::Hedge,
            DecisionType::Exit => HiveDecisionType::Exit,
        };
        
        HiveDecision {
            action,
            confidence: result.quantum_confidence,
            quantum_confidence: Some(result.quantum_confidence),
            reasoning: result.reasoning.join("; "),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Merge quantum states from BDIA and hive
    fn merge_quantum_states(
        bdia_quantum: &crate::quantum_fusion::QuantumState,
        hive_quantum: &HiveQuantumState,
    ) -> f64 {
        // Weighted average based on entanglement and coherence
        let bdia_weight = bdia_quantum.entanglement;
        let hive_weight = hive_quantum.coherence;
        
        let total_weight = bdia_weight + hive_weight;
        if total_weight > 0.0 {
            // Average the probability distributions somehow
            let bdia_conf = bdia_quantum.probabilities.iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(0.5);
            
            let hive_conf = hive_quantum.fidelity; // Use fidelity as confidence proxy
            
            (bdia_conf * bdia_weight + hive_conf * hive_weight) / total_weight
        } else {
            0.5
        }
    }
    
    /// Analyze agent agreement patterns
    fn analyze_agent_agreement(consensus: &ConsensusDecision) -> AgentConsensus {
        use std::collections::HashMap;
        
        let mut decision_counts: HashMap<DecisionType, usize> = HashMap::new();
        let mut intention_sum = 0.0;
        let mut intention_squared_sum = 0.0;
        
        for (_, decision, intention) in &consensus.agent_decisions {
            *decision_counts.entry(*decision).or_insert(0) += 1;
            intention_sum += intention;
            intention_squared_sum += intention * intention;
        }
        
        let n = consensus.agent_decisions.len() as f64;
        let mean_intention = intention_sum / n;
        let variance = (intention_squared_sum / n) - (mean_intention * mean_intention);
        let std_dev = variance.sqrt();
        
        let majority_decision = decision_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(decision, _)| *decision)
            .unwrap_or(DecisionType::Hold);
        
        let majority_percentage = decision_counts.get(&majority_decision)
            .copied()
            .unwrap_or(0) as f64 / n;
        
        AgentConsensus {
            majority_decision,
            majority_percentage,
            intention_mean: mean_intention,
            intention_std: std_dev,
            decision_distribution: decision_counts,
        }
    }
    
    /// Assess risk based on consensus
    fn assess_risk(consensus: &ConsensusDecision) -> RiskAssessment {
        let divergence = consensus.agent_decisions.iter()
            .map(|(_, _, intention)| (intention - consensus.weighted_intention).abs())
            .sum::<f64>() / consensus.agent_decisions.len() as f64;
        
        let risk_level = if divergence > 0.5 {
            RiskLevel::High
        } else if divergence > 0.3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        RiskAssessment {
            level: risk_level,
            divergence_score: divergence,
            confidence_adjusted_risk: divergence / consensus.confidence.max(0.1),
        }
    }
    
    /// Calculate opportunity score
    fn calculate_opportunity(consensus: &ConsensusDecision) -> f64 {
        // High confidence + strong signal = high opportunity
        let signal_strength = consensus.weighted_intention.abs();
        let opportunity = signal_strength * consensus.confidence;
        
        // Adjust for market phase
        let phase_multiplier = match consensus.market_phase {
            crate::market_phase::MarketPhase::Growth => 1.2,
            crate::market_phase::MarketPhase::Conservation => 0.8,
            crate::market_phase::MarketPhase::Release => 0.6,
            crate::market_phase::MarketPhase::Reorganization => 1.0,
        };
        
        (opportunity * phase_multiplier).clamp(0.0, 1.0)
    }
    
    /// Get integration statistics
    pub async fn get_statistics(&self) -> IntegrationStats {
        let bdia_stats = self.bdia_system.get_statistics().await;
        
        IntegrationStats {
            total_decisions: bdia_stats.total_decisions,
            average_confidence: bdia_stats.average_confidence,
            quantum_usage_rate: bdia_stats.decisions_with_quantum as f64 
                / bdia_stats.total_decisions.max(1) as f64,
            hive_coordination_rate: 0.0, // Track in production
            average_latency_us: bdia_stats.average_latency_us,
        }
    }
}

/// Enhanced decision with hive coordination
#[derive(Debug, Clone)]
pub struct EnhancedDecision {
    pub decision: DecisionType,
    pub classical_confidence: f64,
    pub quantum_confidence: f64,
    pub hive_coordinated: bool,
    pub latency_us: u64,
    pub reasoning: Vec<String>,
}

/// Strategic advice for QuantumQueen
#[derive(Debug, Clone)]
pub struct StrategicAdvice {
    pub primary_recommendation: DecisionType,
    pub confidence: f64,
    pub market_phase: crate::market_phase::MarketPhase,
    pub agent_consensus: AgentConsensus,
    pub risk_assessment: RiskAssessment,
    pub opportunity_score: f64,
    pub reasoning: Vec<String>,
}

/// Agent consensus analysis
#[derive(Debug, Clone)]
pub struct AgentConsensus {
    pub majority_decision: DecisionType,
    pub majority_percentage: f64,
    pub intention_mean: f64,
    pub intention_std: f64,
    pub decision_distribution: std::collections::HashMap<DecisionType, usize>,
}

/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub divergence_score: f64,
    pub confidence_adjusted_risk: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Integration statistics
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    pub total_decisions: u64,
    pub average_confidence: f64,
    pub quantum_usage_rate: f64,
    pub hive_coordination_rate: f64,
    pub average_latency_us: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_state_conversion() {
        let hive_state = HiveMarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.2,
            trend: 0.5,
            momentum: 0.3,
            sentiment: 0.6,
            risk_level: 0.4,
            correlation: Some(0.7),
            microstructure: Some(0.1),
            volume_trend: 0.2,
            price_level: 0.0,
            flow_imbalance: Some(0.3),
            market_regime: 2,
        };
        
        let market_data = BDIAHiveIntegration::convert_market_state(&hive_state);
        
        assert_eq!(market_data.sentiment, 0.6);
        assert_eq!(market_data.risk_appetite, 0.4);
        assert_eq!(market_data.momentum, 0.3);
    }
    
    #[test]
    fn test_decision_conversion() {
        use crate::quantum_fusion::{FusionResult, QuantumState};
        
        let result = FusionResult {
            decision: DecisionType::Buy,
            quantum_confidence: 0.85,
            classical_confidence: 0.80,
            intention_signal: 0.7,
            quantum_state: QuantumState {
                amplitudes: vec![],
                probabilities: vec![0.9, 0.1],
                entanglement: 0.5,
                phases: vec![],
            },
            reasoning: vec!["Test reason".to_string()],
        };
        
        let hive_decision = BDIAHiveIntegration::convert_to_hive_decision(result);
        
        assert_eq!(hive_decision.action, HiveDecisionType::Buy);
        assert_eq!(hive_decision.confidence, 0.85);
        assert_eq!(hive_decision.quantum_confidence, Some(0.85));
    }
    
    #[test]
    fn test_risk_assessment() {
        let consensus = ConsensusDecision {
            decision: DecisionType::Buy,
            weighted_intention: 0.6,
            agent_decisions: vec![
                ("Agent1".to_string(), DecisionType::Buy, 0.8),
                ("Agent2".to_string(), DecisionType::Buy, 0.7),
                ("Agent3".to_string(), DecisionType::Hold, 0.2),
                ("Agent4".to_string(), DecisionType::Sell, -0.3),
                ("Agent5".to_string(), DecisionType::Buy, 0.9),
            ],
            confidence: 0.7,
            market_phase: crate::market_phase::MarketPhase::Growth,
            timestamp: chrono::Utc::now(),
        };
        
        let risk = BDIAHiveIntegration::assess_risk(&consensus);
        
        assert!(risk.divergence_score > 0.0);
        assert!(risk.confidence_adjusted_risk > 0.0);
    }
}