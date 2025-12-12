//! PADS Integration Layer for Unified Quantum Agents
//!
//! This module provides the integration layer between unified quantum agents
//! and the Portfolio Autonomous Decision System (PADS).

use quantum_core::{
    PADSSignal, PADSAction, QuantumSignal, MarketData, LatticeState,
    QuantumAgent, QuantumResult, QuantumError
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// PADS Signal Aggregator for combining signals from multiple quantum agents
#[derive(Debug, Clone)]
pub struct PADSSignalAggregator {
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum signal age in seconds
    pub max_signal_age_s: u64,
    /// Signal weights by agent type
    pub agent_weights: HashMap<String, f64>,
}

/// Signal aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple weighted average
    WeightedAverage,
    /// Consensus-based (majority vote)
    Consensus,
    /// Best performing agent
    BestPerformer,
    /// Quantum coherence weighted
    CoherenceWeighted,
    /// Risk-adjusted weighted
    RiskAdjusted,
    /// Ensemble learning
    Ensemble,
}

impl Default for PADSSignalAggregator {
    fn default() -> Self {
        let mut agent_weights = HashMap::new();
        agent_weights.insert("QuantumAgenticReasoning".to_string(), 0.25);
        agent_weights.insert("QuantumHedge".to_string(), 0.20);
        agent_weights.insert("QuantumLMSR".to_string(), 0.15);
        agent_weights.insert("QuantumProspect".to_string(), 0.20);
        agent_weights.insert("NQO".to_string(), 0.20);
        
        Self {
            strategy: AggregationStrategy::CoherenceWeighted,
            min_confidence: 0.3,
            max_signal_age_s: 60,
            agent_weights,
        }
    }
}

impl PADSSignalAggregator {
    /// Create new signal aggregator
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }
    
    /// Aggregate multiple PADS signals into a unified signal
    pub fn aggregate_signals(&self, signals: &[PADSSignal]) -> QuantumResult<PADSSignal> {
        if signals.is_empty() {
            return Err(QuantumError::ProcessingError {
                message: "No signals to aggregate".to_string(),
            });
        }
        
        // Filter signals by age and confidence
        let valid_signals: Vec<&PADSSignal> = signals.iter()
            .filter(|signal| {
                let age = Utc::now().signed_duration_since(signal.quantum_signal.timestamp);
                signal.confidence >= self.min_confidence && 
                age.num_seconds() <= self.max_signal_age_s as i64
            })
            .collect();
        
        if valid_signals.is_empty() {
            return Err(QuantumError::ProcessingError {
                message: "No valid signals after filtering".to_string(),
            });
        }
        
        match self.strategy {
            AggregationStrategy::WeightedAverage => self.weighted_average_aggregation(&valid_signals),
            AggregationStrategy::Consensus => self.consensus_aggregation(&valid_signals),
            AggregationStrategy::BestPerformer => self.best_performer_aggregation(&valid_signals),
            AggregationStrategy::CoherenceWeighted => self.coherence_weighted_aggregation(&valid_signals),
            AggregationStrategy::RiskAdjusted => self.risk_adjusted_aggregation(&valid_signals),
            AggregationStrategy::Ensemble => self.ensemble_aggregation(&valid_signals),
        }
    }
    
    /// Weighted average aggregation
    fn weighted_average_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        let mut weighted_expected_return = 0.0;
        let mut weighted_position_size = 0.0;
        let mut weighted_risk_level = 0.0;
        
        // Calculate action scores
        let mut action_scores: HashMap<PADSAction, f64> = HashMap::new();
        
        for signal in signals {
            let agent_type = signal.quantum_signal.metadata
                .get("agent_type")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            
            let weight = self.agent_weights.get(agent_type).copied().unwrap_or(0.1);
            total_weight += weight;
            
            weighted_confidence += signal.confidence * weight;
            weighted_expected_return += signal.expected_return * weight;
            weighted_position_size += signal.position_size * weight;
            weighted_risk_level += signal.risk_level * weight;
            
            // Accumulate action scores
            *action_scores.entry(signal.action).or_insert(0.0) += weight;
        }
        
        if total_weight == 0.0 {
            return Err(QuantumError::ProcessingError {
                message: "Total weight is zero".to_string(),
            });
        }
        
        // Normalize weighted values
        weighted_confidence /= total_weight;
        weighted_expected_return /= total_weight;
        weighted_position_size /= total_weight;
        weighted_risk_level /= total_weight;
        
        // Select action with highest score
        let best_action = action_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| *action)
            .unwrap_or(PADSAction::Hold);
        
        Ok(PADSSignal {
            quantum_signal: self.create_aggregated_quantum_signal(signals)?,
            action: best_action,
            confidence: weighted_confidence,
            risk_level: weighted_risk_level,
            expected_return: weighted_expected_return,
            position_size: weighted_position_size,
            metadata: [
                ("aggregation_strategy".to_string(), "weighted_average".to_string()),
                ("signal_count".to_string(), signals.len().to_string()),
            ].iter().cloned().collect(),
        })
    }
    
    /// Consensus-based aggregation
    fn consensus_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        // Count votes for each action
        let mut action_votes: HashMap<PADSAction, usize> = HashMap::new();
        
        for signal in signals {
            *action_votes.entry(signal.action).or_insert(0) += 1;
        }
        
        // Find consensus action (majority vote)
        let total_votes = signals.len();
        let consensus_action = action_votes.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(action, _)| *action)
            .unwrap_or(PADSAction::Hold);
        
        // Calculate consensus strength
        let consensus_votes = action_votes.get(&consensus_action).unwrap_or(&0);
        let consensus_strength = *consensus_votes as f64 / total_votes as f64;
        
        // Average other metrics from consensus signals
        let consensus_signals: Vec<&PADSSignal> = signals.iter()
            .filter(|signal| signal.action == consensus_action)
            .cloned()
            .collect();
        
        let avg_confidence = consensus_signals.iter()
            .map(|s| s.confidence)
            .sum::<f64>() / consensus_signals.len() as f64;
        
        let avg_expected_return = consensus_signals.iter()
            .map(|s| s.expected_return)
            .sum::<f64>() / consensus_signals.len() as f64;
        
        let avg_position_size = consensus_signals.iter()
            .map(|s| s.position_size)
            .sum::<f64>() / consensus_signals.len() as f64;
        
        let avg_risk_level = consensus_signals.iter()
            .map(|s| s.risk_level)
            .sum::<f64>() / consensus_signals.len() as f64;
        
        Ok(PADSSignal {
            quantum_signal: self.create_aggregated_quantum_signal(signals)?,
            action: consensus_action,
            confidence: avg_confidence * consensus_strength,
            risk_level: avg_risk_level,
            expected_return: avg_expected_return,
            position_size: avg_position_size,
            metadata: [
                ("aggregation_strategy".to_string(), "consensus".to_string()),
                ("consensus_strength".to_string(), consensus_strength.to_string()),
                ("consensus_votes".to_string(), consensus_votes.to_string()),
            ].iter().cloned().collect(),
        })
    }
    
    /// Best performer aggregation
    fn best_performer_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        // Find signal with highest confidence
        let best_signal = signals.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or_else(|| QuantumError::ProcessingError {
                message: "No best signal found".to_string(),
            })?;
        
        let mut aggregated_signal = (*best_signal).clone();
        aggregated_signal.metadata.insert("aggregation_strategy".to_string(), "best_performer".to_string());
        aggregated_signal.metadata.insert("best_confidence".to_string(), best_signal.confidence.to_string());
        
        Ok(aggregated_signal)
    }
    
    /// Coherence-weighted aggregation
    fn coherence_weighted_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        let mut total_coherence = 0.0;
        let mut coherence_weighted_confidence = 0.0;
        let mut coherence_weighted_expected_return = 0.0;
        let mut coherence_weighted_position_size = 0.0;
        let mut coherence_weighted_risk_level = 0.0;
        
        // Calculate action scores weighted by coherence
        let mut action_scores: HashMap<PADSAction, f64> = HashMap::new();
        
        for signal in signals {
            let coherence = signal.quantum_signal.coherence;
            total_coherence += coherence;
            
            coherence_weighted_confidence += signal.confidence * coherence;
            coherence_weighted_expected_return += signal.expected_return * coherence;
            coherence_weighted_position_size += signal.position_size * coherence;
            coherence_weighted_risk_level += signal.risk_level * coherence;
            
            // Accumulate action scores
            *action_scores.entry(signal.action).or_insert(0.0) += coherence;
        }
        
        if total_coherence == 0.0 {
            return Err(QuantumError::ProcessingError {
                message: "Total coherence is zero".to_string(),
            });
        }
        
        // Normalize coherence-weighted values
        coherence_weighted_confidence /= total_coherence;
        coherence_weighted_expected_return /= total_coherence;
        coherence_weighted_position_size /= total_coherence;
        coherence_weighted_risk_level /= total_coherence;
        
        // Select action with highest coherence score
        let best_action = action_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| *action)
            .unwrap_or(PADSAction::Hold);
        
        Ok(PADSSignal {
            quantum_signal: self.create_aggregated_quantum_signal(signals)?,
            action: best_action,
            confidence: coherence_weighted_confidence,
            risk_level: coherence_weighted_risk_level,
            expected_return: coherence_weighted_expected_return,
            position_size: coherence_weighted_position_size,
            metadata: [
                ("aggregation_strategy".to_string(), "coherence_weighted".to_string()),
                ("total_coherence".to_string(), total_coherence.to_string()),
                ("avg_coherence".to_string(), (total_coherence / signals.len() as f64).to_string()),
            ].iter().cloned().collect(),
        })
    }
    
    /// Risk-adjusted aggregation
    fn risk_adjusted_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        // Weight signals inversely by risk level
        let mut total_risk_weight = 0.0;
        let mut risk_weighted_confidence = 0.0;
        let mut risk_weighted_expected_return = 0.0;
        let mut risk_weighted_position_size = 0.0;
        let mut risk_weighted_risk_level = 0.0;
        
        let mut action_scores: HashMap<PADSAction, f64> = HashMap::new();
        
        for signal in signals {
            // Inverse risk weighting (lower risk = higher weight)
            let risk_weight = 1.0 / (signal.risk_level + 0.01); // Add small epsilon to avoid division by zero
            total_risk_weight += risk_weight;
            
            risk_weighted_confidence += signal.confidence * risk_weight;
            risk_weighted_expected_return += signal.expected_return * risk_weight;
            risk_weighted_position_size += signal.position_size * risk_weight;
            risk_weighted_risk_level += signal.risk_level * risk_weight;
            
            *action_scores.entry(signal.action).or_insert(0.0) += risk_weight;
        }
        
        if total_risk_weight == 0.0 {
            return Err(QuantumError::ProcessingError {
                message: "Total risk weight is zero".to_string(),
            });
        }
        
        // Normalize risk-weighted values
        risk_weighted_confidence /= total_risk_weight;
        risk_weighted_expected_return /= total_risk_weight;
        risk_weighted_position_size /= total_risk_weight;
        risk_weighted_risk_level /= total_risk_weight;
        
        let best_action = action_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| *action)
            .unwrap_or(PADSAction::Hold);
        
        Ok(PADSSignal {
            quantum_signal: self.create_aggregated_quantum_signal(signals)?,
            action: best_action,
            confidence: risk_weighted_confidence,
            risk_level: risk_weighted_risk_level,
            expected_return: risk_weighted_expected_return,
            position_size: risk_weighted_position_size,
            metadata: [
                ("aggregation_strategy".to_string(), "risk_adjusted".to_string()),
                ("total_risk_weight".to_string(), total_risk_weight.to_string()),
            ].iter().cloned().collect(),
        })
    }
    
    /// Ensemble aggregation using multiple strategies
    fn ensemble_aggregation(&self, signals: &[&PADSSignal]) -> QuantumResult<PADSSignal> {
        // Apply multiple strategies and combine results
        let strategies = vec![
            AggregationStrategy::WeightedAverage,
            AggregationStrategy::Consensus,
            AggregationStrategy::CoherenceWeighted,
            AggregationStrategy::RiskAdjusted,
        ];
        
        let mut ensemble_results = Vec::new();
        
        for strategy in strategies {
            let mut temp_aggregator = self.clone();
            temp_aggregator.strategy = strategy;
            
            if let Ok(result) = temp_aggregator.aggregate_signals(signals) {
                ensemble_results.push(result);
            }
        }
        
        if ensemble_results.is_empty() {
            return Err(QuantumError::ProcessingError {
                message: "No ensemble results generated".to_string(),
            });
        }
        
        // Meta-aggregation of ensemble results
        let ensemble_signals: Vec<&PADSSignal> = ensemble_results.iter().collect();
        
        // Use coherence-weighted strategy for meta-aggregation
        let mut meta_aggregator = self.clone();
        meta_aggregator.strategy = AggregationStrategy::CoherenceWeighted;
        
        let mut final_result = meta_aggregator.coherence_weighted_aggregation(&ensemble_signals)?;
        final_result.metadata.insert("aggregation_strategy".to_string(), "ensemble".to_string());
        final_result.metadata.insert("ensemble_size".to_string(), ensemble_results.len().to_string());
        
        Ok(final_result)
    }
    
    /// Create aggregated quantum signal from multiple signals
    fn create_aggregated_quantum_signal(&self, signals: &[&PADSSignal]) -> QuantumResult<QuantumSignal> {
        let avg_amplitude = signals.iter().map(|s| s.quantum_signal.amplitude).sum::<f64>() / signals.len() as f64;
        let avg_phase = signals.iter().map(|s| s.quantum_signal.phase).sum::<f64>() / signals.len() as f64;
        let avg_coherence = signals.iter().map(|s| s.quantum_signal.coherence).sum::<f64>() / signals.len() as f64;
        let avg_strength = signals.iter().map(|s| s.quantum_signal.strength).sum::<f64>() / signals.len() as f64;
        
        // Combine entanglement information
        let mut combined_entanglement = HashMap::new();
        for signal in signals {
            for (agent_id, entanglement) in &signal.quantum_signal.entanglement {
                let current_value = combined_entanglement.get(agent_id).copied().unwrap_or(0.0);
                combined_entanglement.insert(agent_id.clone(), current_value + entanglement);
            }
        }
        
        // Normalize entanglement values
        for value in combined_entanglement.values_mut() {
            *value /= signals.len() as f64;
        }
        
        Ok(QuantumSignal {
            id: Uuid::new_v4().to_string(),
            agent_id: "aggregated".to_string(),
            signal_type: quantum_core::QuantumSignalType::Portfolio, // Aggregated signals are portfolio-level
            strength: avg_strength,
            amplitude: avg_amplitude,
            phase: avg_phase,
            coherence: avg_coherence,
            entanglement: combined_entanglement,
            data: [
                ("aggregated_signal_count".to_string(), signals.len() as f64),
                ("avg_amplitude".to_string(), avg_amplitude),
                ("avg_phase".to_string(), avg_phase),
                ("avg_coherence".to_string(), avg_coherence),
            ].iter().cloned().collect(),
            metadata: [
                ("signal_type".to_string(), "aggregated".to_string()),
            ].iter().cloned().collect(),
            timestamp: Utc::now(),
        })
    }
}

/// PADS Integration Manager
#[derive(Debug)]
pub struct PADSIntegrationManager {
    /// Signal aggregator
    pub aggregator: PADSSignalAggregator,
    /// Historical signals for performance tracking
    pub signal_history: Vec<PADSSignal>,
    /// Performance metrics
    pub performance_metrics: PADSPerformanceMetrics,
    /// Maximum history size
    pub max_history_size: usize,
}

impl Default for PADSIntegrationManager {
    fn default() -> Self {
        Self {
            aggregator: PADSSignalAggregator::default(),
            signal_history: Vec::new(),
            performance_metrics: PADSPerformanceMetrics::default(),
            max_history_size: 1000,
        }
    }
}

impl PADSIntegrationManager {
    /// Create new PADS integration manager
    pub fn new(aggregation_strategy: AggregationStrategy) -> Self {
        Self {
            aggregator: PADSSignalAggregator::new(aggregation_strategy),
            ..Default::default()
        }
    }
    
    /// Process signals from quantum agents and generate PADS decision
    pub async fn process_quantum_signals(
        &mut self,
        agent_signals: Vec<PADSSignal>,
    ) -> QuantumResult<PADSSignal> {
        // Aggregate signals
        let aggregated_signal = self.aggregator.aggregate_signals(&agent_signals)?;
        
        // Store in history
        self.signal_history.push(aggregated_signal.clone());
        
        // Maintain history size
        if self.signal_history.len() > self.max_history_size {
            self.signal_history.drain(0..self.signal_history.len() - self.max_history_size);
        }
        
        // Update performance metrics
        self.update_performance_metrics(&aggregated_signal, &agent_signals);
        
        Ok(aggregated_signal)
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self, aggregated_signal: &PADSSignal, agent_signals: &[PADSSignal]) {
        self.performance_metrics.total_signals_processed += 1;
        self.performance_metrics.total_agents_contributed += agent_signals.len() as u64;
        
        // Update confidence distribution
        self.performance_metrics.avg_confidence = 
            (self.performance_metrics.avg_confidence * (self.performance_metrics.total_signals_processed - 1) as f64
            + aggregated_signal.confidence) / self.performance_metrics.total_signals_processed as f64;
        
        // Update coherence metrics
        let avg_coherence = agent_signals.iter()
            .map(|s| s.quantum_signal.coherence)
            .sum::<f64>() / agent_signals.len() as f64;
            
        self.performance_metrics.avg_coherence = 
            (self.performance_metrics.avg_coherence * (self.performance_metrics.total_signals_processed - 1) as f64
            + avg_coherence) / self.performance_metrics.total_signals_processed as f64;
        
        // Track action distribution
        *self.performance_metrics.action_distribution.entry(aggregated_signal.action).or_insert(0) += 1;
        
        self.performance_metrics.last_update = Utc::now();
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PADSPerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Get recent signal history
    pub fn get_recent_signals(&self, count: usize) -> Vec<&PADSSignal> {
        self.signal_history.iter()
            .rev()
            .take(count)
            .collect()
    }
}

/// PADS performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PADSPerformanceMetrics {
    /// Total signals processed
    pub total_signals_processed: u64,
    /// Total agents that have contributed
    pub total_agents_contributed: u64,
    /// Average confidence level
    pub avg_confidence: f64,
    /// Average coherence level
    pub avg_coherence: f64,
    /// Action distribution
    pub action_distribution: HashMap<PADSAction, u64>,
    /// Signal success rate (if tracking outcomes)
    pub signal_success_rate: f64,
    /// Last metrics update
    pub last_update: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantum_core::{QuantumSignalType};
    
    fn create_test_signal(agent_id: &str, action: PADSAction, confidence: f64, coherence: f64) -> PADSSignal {
        PADSSignal {
            quantum_signal: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: agent_id.to_string(),
                signal_type: QuantumSignalType::Trading,
                strength: confidence,
                amplitude: 0.5,
                phase: 0.0,
                coherence,
                entanglement: HashMap::new(),
                data: HashMap::new(),
                metadata: [("agent_type".to_string(), agent_id.to_string())].iter().cloned().collect(),
                timestamp: Utc::now(),
            },
            action,
            confidence,
            risk_level: 0.3,
            expected_return: 0.05,
            position_size: 0.1,
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_weighted_average_aggregation() {
        let aggregator = PADSSignalAggregator::new(AggregationStrategy::WeightedAverage);
        
        let signals = vec![
            create_test_signal("QAR", PADSAction::Buy, 0.8, 0.9),
            create_test_signal("Hedge", PADSAction::Buy, 0.7, 0.8),
            create_test_signal("LMSR", PADSAction::Hold, 0.6, 0.7),
        ];
        
        let signal_refs: Vec<&PADSSignal> = signals.iter().collect();
        let result = aggregator.weighted_average_aggregation(&signal_refs);
        
        assert!(result.is_ok());
        let aggregated = result.unwrap();
        assert_eq!(aggregated.action, PADSAction::Buy); // Majority action
        assert!(aggregated.confidence > 0.0);
    }
    
    #[test]
    fn test_consensus_aggregation() {
        let aggregator = PADSSignalAggregator::new(AggregationStrategy::Consensus);
        
        let signals = vec![
            create_test_signal("QAR", PADSAction::Buy, 0.8, 0.9),
            create_test_signal("Hedge", PADSAction::Buy, 0.7, 0.8),
            create_test_signal("LMSR", PADSAction::Sell, 0.6, 0.7),
        ];
        
        let signal_refs: Vec<&PADSSignal> = signals.iter().collect();
        let result = aggregator.consensus_aggregation(&signal_refs);
        
        assert!(result.is_ok());
        let aggregated = result.unwrap();
        assert_eq!(aggregated.action, PADSAction::Buy); // Consensus action
    }
    
    #[tokio::test]
    async fn test_pads_integration_manager() {
        let mut manager = PADSIntegrationManager::new(AggregationStrategy::CoherenceWeighted);
        
        let signals = vec![
            create_test_signal("QAR", PADSAction::Buy, 0.8, 0.9),
            create_test_signal("Hedge", PADSAction::Buy, 0.7, 0.8),
        ];
        
        let result = manager.process_quantum_signals(signals).await;
        assert!(result.is_ok());
        
        let metrics = manager.get_performance_metrics();
        assert_eq!(metrics.total_signals_processed, 1);
        assert_eq!(metrics.total_agents_contributed, 2);
    }
}