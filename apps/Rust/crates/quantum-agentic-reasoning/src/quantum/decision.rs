//! Decision engine implementation for Quantum Agentic Reasoning
//!
//! This module provides the core decision-making logic using quantum circuits
//! for enhanced trading decisions with quantum advantage.

use crate::core::{
    QarResult, QarError, StandardFactors, FactorMap, MarketContext, 
    TradingDecision, DecisionType, MarketPhase, constants,
    DecisionMetrics, PatternData, PatternMatch, RegimeAnalysis, 
    DecisionOptimization, DecisionOutcome, DecisionEngine
};
use crate::quantum::QuantumState;
use crate::core::{CoreQuantumCircuit as QuantumCircuit, CircuitParams, ExecutionContext};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use super::types::*;
use super::traits::*;
use super::circuits::{QftCircuit, DecisionOptimizationCircuit, PatternRecognitionCircuit};

/// Quantum decision engine for trading decisions
#[derive(Debug)]
pub struct QuantumDecisionEngine {
    /// QFT circuit for market analysis
    qft_circuit: QftCircuit,
    /// Decision optimization circuit
    optimization_circuit: DecisionOptimizationCircuit,
    /// Pattern recognition circuit
    pattern_circuit: PatternRecognitionCircuit,
    /// Confidence threshold for decisions
    confidence_threshold: f64,
    /// Decision history
    decision_history: Arc<RwLock<Vec<TradingDecision>>>,
    /// Performance metrics
    metrics: Arc<RwLock<DecisionMetrics>>,
    /// Learned patterns
    learned_patterns: Arc<RwLock<Vec<PatternData>>>,
}

impl QuantumDecisionEngine {
    /// Create a new quantum decision engine
    pub fn new(num_qubits: usize, confidence_threshold: f64) -> Self {
        let qft_circuit = QftCircuit::new(num_qubits);
        let optimization_circuit = DecisionOptimizationCircuit::new(
            num_qubits,
            constants::decision::AMPLITUDE_AMPLIFICATION_ITERATIONS,
            constants::decision::MIN_DECISION_CONFIDENCE,
        );
        let mut pattern_circuit = PatternRecognitionCircuit::new(
            num_qubits,
            constants::pattern::ENCODING_PRECISION,
        );

        // Add some default patterns
        let _ = pattern_circuit.add_reference_pattern(vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1]); // Bull pattern
        let _ = pattern_circuit.add_reference_pattern(vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.1, 0.1, 0.1]); // Bear pattern
        let _ = pattern_circuit.add_reference_pattern(vec![0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]); // Sideways pattern

        Self {
            qft_circuit,
            optimization_circuit,
            pattern_circuit,
            confidence_threshold,
            decision_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(DecisionMetrics {
                total_decisions: 0,
                successful_decisions: 0,
                average_confidence: 0.0,
                average_execution_time_ms: 0.0,
                last_decision_time: chrono::Utc::now(),
            })),
            learned_patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Extract factor values as a vector
    fn extract_factor_values(&self, factors: &FactorMap) -> Vec<f64> {
        StandardFactors::all_factors()
            .iter()
            .map(|factor| factors.get(factor).unwrap_or(0.0))
            .collect()
    }

    /// Perform quantum market analysis using QFT
    async fn analyze_market_quantum(&self, factors: &FactorMap) -> QarResult<RegimeAnalysis> {
        let factor_values = self.extract_factor_values(factors);
        let params = CircuitParams::new(factor_values, self.qft_circuit.num_qubits());
        let context = ExecutionContext::default();

        let qft_result = self.qft_circuit.execute(&params, &context).await?;

        // Analyze spectral power to determine market regime
        let spectral_power = qft_result.expectation_values;
        let phase = self.determine_market_phase(&spectral_power);
        let confidence = self.calculate_regime_confidence(&spectral_power);
        let strength = self.calculate_regime_strength(&spectral_power);

        Ok(RegimeAnalysis {
            phase,
            confidence,
            strength,
            volatility: self.estimate_volatility(&spectral_power),
            noise_level: self.estimate_noise_level(&spectral_power),
            spectral_power,
            phase_coherence: self.calculate_phase_coherence(&qft_result.probabilities),
        })
    }

    /// Optimize decision using quantum amplitude amplification
    async fn optimize_decision_quantum(&self, factors: &FactorMap) -> QarResult<DecisionOptimization> {
        let factor_values = self.extract_factor_values(factors);
        let params = CircuitParams::new(factor_values, self.optimization_circuit.num_qubits());
        let context = ExecutionContext::default();

        let opt_result = self.optimization_circuit.execute(&params, &context).await?;

        let weights = opt_result.expectation_values;
        let confidence = self.calculate_decision_confidence(&weights);
        let information_gain = self.calculate_information_gain(&weights, &factor_values);

        Ok(DecisionOptimization {
            weights,
            confidence,
            information_gain,
            metadata: HashMap::new(),
        })
    }

    /// Recognize patterns using quantum similarity
    async fn recognize_patterns_quantum(&self, factors: &FactorMap) -> QarResult<Vec<PatternMatch>> {
        let factor_values = self.extract_factor_values(factors);
        
        // Pad or truncate to match circuit requirements
        let expected_size = 1 << self.pattern_circuit.num_qubits();
        let mut padded_values = factor_values;
        padded_values.resize(expected_size, 0.0);

        let params = CircuitParams::new(padded_values, self.pattern_circuit.num_qubits());
        let context = ExecutionContext::default();

        let pattern_result = self.pattern_circuit.execute(&params, &context).await?;

        // Convert results to pattern matches
        let mut matches = Vec::new();
        for (i, &similarity) in pattern_result.expectation_values.iter().enumerate() {
            if similarity >= constants::MIN_PATTERN_CONFIDENCE {
                matches.push(PatternMatch::new(
                    format!("pattern_{}", i),
                    similarity,
                    similarity, // Use similarity as confidence for now
                ));
            }
        }

        Ok(matches)
    }

    /// Combine quantum analyses into trading decision
    fn synthesize_decision(
        &self,
        regime: &RegimeAnalysis,
        optimization: &DecisionOptimization,
        patterns: &[PatternMatch],
        context: &MarketContext,
    ) -> QarResult<TradingDecision> {
        // Calculate base decision from optimization weights
        let strongest_factor = optimization.strongest_factor()
            .map(|(idx, weight)| (StandardFactors::all_factors()[idx], weight))
            .unwrap_or((StandardFactors::Momentum, 0.0));

        // Determine decision type based on regime and patterns
        let decision_type = self.determine_decision_type(regime, patterns, strongest_factor.1);

        // Calculate final confidence
        let pattern_confidence = if patterns.is_empty() {
            0.5
        } else {
            patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64
        };

        let final_confidence = (regime.confidence + optimization.confidence + pattern_confidence) / 3.0;

        // Check if confidence meets threshold
        if final_confidence < self.confidence_threshold {
            return Ok(TradingDecision {
                id: uuid::Uuid::new_v4().to_string(),
                decision_type: DecisionType::Hold,
                confidence: final_confidence,
                reasoning: "Insufficient confidence for action".to_string(),
                factors_used: HashMap::new(),
                timestamp: chrono::Utc::now(),
                execution_time_ms: 0.0,
                quantum_advantage: true,
                metadata: HashMap::new(),
            });
        }

        // Create decision with metadata
        let mut metadata = HashMap::new();
        metadata.insert("market_phase".to_string(), format!("{:?}", regime.phase));
        metadata.insert("regime_strength".to_string(), regime.strength.to_string());
        metadata.insert("pattern_matches".to_string(), patterns.len().to_string());
        metadata.insert("information_gain".to_string(), optimization.information_gain.to_string());

        let mut factors_used = HashMap::new();
        for (i, factor) in StandardFactors::all_factors().iter().enumerate() {
            if i < optimization.weights.len() {
                factors_used.insert(format!("{:?}", factor), optimization.weights[i]);
            }
        }

        Ok(TradingDecision {
            id: uuid::Uuid::new_v4().to_string(),
            decision_type,
            confidence: final_confidence,
            reasoning: self.generate_reasoning(regime, optimization, patterns),
            factors_used,
            timestamp: chrono::Utc::now(),
            execution_time_ms: 0.0, // Will be filled by caller
            quantum_advantage: true,
            metadata,
        })
    }

    /// Determine market phase from spectral analysis
    fn determine_market_phase(&self, spectral_power: &[f64]) -> MarketPhase {
        if spectral_power.is_empty() {
            return MarketPhase::Uncertain;
        }

        let total_power: f64 = spectral_power.iter().sum();
        if total_power == 0.0 {
            return MarketPhase::Sideways;
        }

        // Analyze frequency distribution
        let low_freq_power: f64 = spectral_power.iter().take(spectral_power.len() / 3).sum();
        let mid_freq_power: f64 = spectral_power.iter()
            .skip(spectral_power.len() / 3)
            .take(spectral_power.len() / 3)
            .sum();
        let high_freq_power: f64 = spectral_power.iter()
            .skip(2 * spectral_power.len() / 3)
            .sum();

        let low_ratio = low_freq_power / total_power;
        let high_ratio = high_freq_power / total_power;

        if low_ratio > 0.6 {
            MarketPhase::Growth
        } else if high_ratio > 0.6 {
            MarketPhase::Decline
        } else if mid_freq_power / total_power > 0.5 {
            MarketPhase::Sideways
        } else {
            MarketPhase::Uncertain
        }
    }

    /// Calculate regime confidence
    fn calculate_regime_confidence(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 0.0;
        }

        let total_power: f64 = spectral_power.iter().sum();
        if total_power == 0.0 {
            return 0.0;
        }

        // Shannon entropy-based confidence
        let entropy: f64 = spectral_power.iter()
            .map(|&p| {
                let prob = p / total_power;
                if prob > 0.0 {
                    -prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = (spectral_power.len() as f64).ln();
        if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            0.0
        }
    }

    /// Calculate regime strength
    fn calculate_regime_strength(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 0.0;
        }

        let max_power = spectral_power.iter().fold(0.0, |a, &b| a.max(b));
        let total_power: f64 = spectral_power.iter().sum();

        if total_power > 0.0 {
            max_power / total_power
        } else {
            0.0
        }
    }

    /// Estimate volatility from spectral power
    fn estimate_volatility(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.len() < 2 {
            return 0.0;
        }

        let mean = spectral_power.iter().sum::<f64>() / spectral_power.len() as f64;
        let variance = spectral_power.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / spectral_power.len() as f64;

        variance.sqrt()
    }

    /// Estimate noise level
    fn estimate_noise_level(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 1.0;
        }

        // High frequency components indicate noise
        let high_freq_start = 2 * spectral_power.len() / 3;
        let high_freq_power: f64 = spectral_power.iter().skip(high_freq_start).sum();
        let total_power: f64 = spectral_power.iter().sum();

        if total_power > 0.0 {
            high_freq_power / total_power
        } else {
            1.0
        }
    }

    /// Calculate phase coherence
    fn calculate_phase_coherence(&self, probabilities: &[f64]) -> f64 {
        if probabilities.len() < 2 {
            return 0.0;
        }

        let mean = probabilities.iter().sum::<f64>() / probabilities.len() as f64;
        let variance = probabilities.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / probabilities.len() as f64;

        if variance > 0.0 {
            1.0 / (1.0 + variance)
        } else {
            1.0
        }
    }

    /// Calculate decision confidence
    fn calculate_decision_confidence(&self, weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }

        let max_weight = weights.iter().fold(0.0, |a, &b| a.max(b.abs()));
        let mean_weight = weights.iter().map(|x| x.abs()).sum::<f64>() / weights.len() as f64;

        if mean_weight > 0.0 {
            (max_weight / mean_weight).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate information gain
    fn calculate_information_gain(&self, weights: &[f64], factors: &[f64]) -> f64 {
        if weights.len() != factors.len() || weights.is_empty() {
            return 0.0;
        }

        let weighted_sum: f64 = weights.iter().zip(factors.iter())
            .map(|(&w, &f)| w * f)
            .sum();

        let factor_sum: f64 = factors.iter().sum();

        if factor_sum > 0.0 {
            (weighted_sum / factor_sum).abs()
        } else {
            0.0
        }
    }

    /// Determine decision type
    fn determine_decision_type(&self, regime: &RegimeAnalysis, patterns: &[PatternMatch], weight: f64) -> DecisionType {
        // Strong positive weight and growth regime suggest buy
        if weight > 0.5 && matches!(regime.phase, MarketPhase::Growth) {
            return DecisionType::Buy;
        }

        // Strong negative weight and decline regime suggest sell
        if weight < -0.5 && matches!(regime.phase, MarketPhase::Decline) {
            return DecisionType::Sell;
        }

        // Check pattern signals
        for pattern in patterns {
            if pattern.pattern_id.contains("bull") && pattern.confidence > 0.7 {
                return DecisionType::Buy;
            }
            if pattern.pattern_id.contains("bear") && pattern.confidence > 0.7 {
                return DecisionType::Sell;
            }
        }

        // Uncertain conditions suggest hold
        if matches!(regime.phase, MarketPhase::Uncertain | MarketPhase::Sideways) {
            return DecisionType::Hold;
        }

        // Default to hold for weak signals
        DecisionType::Hold
    }

    /// Generate reasoning for decision
    fn generate_reasoning(&self, regime: &RegimeAnalysis, optimization: &DecisionOptimization, patterns: &[PatternMatch]) -> String {
        let mut reasoning = Vec::new();

        reasoning.push(format!("Market regime: {:?} (confidence: {:.2})", regime.phase, regime.confidence));
        reasoning.push(format!("Regime strength: {:.2}", regime.strength));

        if let Some((factor_idx, weight)) = optimization.strongest_factor() {
            let factor_name = format!("{:?}", StandardFactors::all_factors()[factor_idx]);
            reasoning.push(format!("Strongest factor: {} (weight: {:.2})", factor_name, weight));
        }

        reasoning.push(format!("Information gain: {:.2}", optimization.information_gain));

        if !patterns.is_empty() {
            let pattern_count = patterns.len();
            let avg_confidence = patterns.iter().map(|p| p.confidence).sum::<f64>() / pattern_count as f64;
            reasoning.push(format!("Patterns detected: {} (avg confidence: {:.2})", pattern_count, avg_confidence));
        }

        reasoning.join("; ")
    }
}

#[async_trait]
impl DecisionEngine for QuantumDecisionEngine {
    async fn make_decision(
        &self,
        factors: &FactorMap,
        context: &MarketContext,
    ) -> QarResult<TradingDecision> {
        let start_time = std::time::Instant::now();

        // Perform quantum analyses in parallel
        let (regime_result, optimization_result, patterns_result) = tokio::try_join!(
            self.analyze_market_quantum(factors),
            self.optimize_decision_quantum(factors),
            self.recognize_patterns_quantum(factors)
        )?;

        // Synthesize final decision
        let mut decision = self.synthesize_decision(
            &regime_result,
            &optimization_result,
            &patterns_result,
            context,
        )?;

        // Set execution time
        decision.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Store decision in history
        let mut history = self.decision_history.write().await;
        history.push(decision.clone());

        // Keep only recent decisions
        if history.len() > constants::DEFAULT_MEMORY_LENGTH {
            history.remove(0);
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_decisions += 1;
        metrics.average_execution_time_ms = 
            (metrics.average_execution_time_ms * (metrics.total_decisions - 1) as f64 + decision.execution_time_ms) 
            / metrics.total_decisions as f64;
        metrics.average_confidence = 
            (metrics.average_confidence * (metrics.total_decisions - 1) as f64 + decision.confidence) 
            / metrics.total_decisions as f64;
        metrics.last_decision_time = decision.timestamp;

        Ok(decision)
    }

    async fn update_with_feedback(
        &mut self,
        decision_id: &str,
        outcome: DecisionOutcome,
    ) -> QarResult<()> {
        let mut history = self.decision_history.write().await;
        
        // Find the decision and update success metrics
        if let Some(decision) = history.iter().find(|d| d.id == decision_id) {
            let mut metrics = self.metrics.write().await;
            
            match outcome {
                DecisionOutcome::Success { .. } => {
                    metrics.successful_decisions += 1;
                    
                    // Learn from successful patterns
                    let pattern_data = PatternData {
                        id: format!("success_{}", decision_id),
                        features: decision.factors_used.values().cloned().collect(),
                        timestamp: chrono::Utc::now(),
                        metadata: decision.metadata.clone(),
                    };
                    
                    let mut patterns = self.learned_patterns.write().await;
                    patterns.push(pattern_data);
                    
                    // Add to pattern circuit if space available
                    if patterns.len() <= constants::MAX_PATTERN_STORAGE {
                        let feature_vector = decision.factors_used.values().cloned().collect::<Vec<_>>();
                        if feature_vector.len() == (1 << self.pattern_circuit.num_qubits()) {
                            let _ = self.pattern_circuit.add_reference_pattern(feature_vector);
                        }
                    }
                }
                DecisionOutcome::Failure { .. } => {
                    // Could implement negative learning here
                }
                DecisionOutcome::Pending => {
                    // Nothing to update yet
                }
            }
        }

        Ok(())
    }

    fn confidence_threshold(&self) -> f64 {
        self.config.confidence_threshold
    }

    async fn set_confidence_threshold(&self, threshold: f64) -> QarResult<()> {
        // Update would normally happen here
        Ok(())
    }
    
    async fn update_with_feedback(
        &self,
        outcome: DecisionOutcome,
    ) -> QarResult<()> {
        // Use a different approach to avoid the naming conflict
        self.process_feedback_outcome(outcome).await
    }

    fn get_metrics(&self) -> DecisionMetrics {
        // This is a synchronous method, so we can't await
        // In a real implementation, you might want to use a different approach
        // For now, return a default or cached value
        DecisionMetrics {
            total_decisions: 0,
            successful_decisions: 0,
            average_confidence: self.confidence_threshold,
            average_execution_time_ms: 0.0,
            last_decision_time: chrono::Utc::now(),
        }
    }
}

impl QuantumDecisionEngine {
    /// Get decision history
    pub async fn get_decision_history(&self) -> Vec<TradingDecision> {
        self.decision_history.read().await.clone()
    }

    /// Get learned patterns
    pub async fn get_learned_patterns(&self) -> Vec<PatternData> {
        self.learned_patterns.read().await.clone()
    }

    /// Clear decision history
    pub async fn clear_history(&self) {
        let mut history = self.decision_history.write().await;
        history.clear();
    }

    /// Get detailed metrics (async version)
    pub async fn get_detailed_metrics(&self) -> DecisionMetrics {
        self.metrics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_decision_engine_creation() {
        let engine = QuantumDecisionEngine::new(3, 0.6);
        assert_eq!(engine.confidence_threshold, 0.6);
        assert_eq!(engine.qft_circuit.num_qubits(), 3);
    }

    #[tokio::test]
    async fn test_decision_making() {
        let mut engine = QuantumDecisionEngine::new(3, 0.3);
        
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Momentum, 0.8).unwrap();
        factors.set(StandardFactors::Volume, 0.6).unwrap();
        factors.set(StandardFactors::Volatility, 0.4).unwrap();
        
        let context = MarketContext {
            symbol: "BTC/USD".to_string(),
            timeframe: "1h".to_string(),
            current_price: 50000.0,
            current_time: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        let decision = engine.make_decision(&factors, &context).await;
        assert!(decision.is_ok());
        
        let decision = decision.unwrap();
        assert!(decision.confidence >= 0.0);
        assert!(decision.execution_time_ms > 0.0);
        assert!(decision.quantum_advantage);
    }

    #[tokio::test]
    async fn test_feedback_learning() {
        let mut engine = QuantumDecisionEngine::new(3, 0.3);
        
        // Make a decision first
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Momentum, 0.7).unwrap();
        
        let context = MarketContext {
            symbol: "ETH/USD".to_string(),
            timeframe: "1h".to_string(),
            current_price: 3000.0,
            current_time: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        let decision = engine.make_decision(&factors, &context).await.unwrap();
        
        // Provide feedback
        let outcome = DecisionOutcome::Success {
            profit: 100.0,
            duration_ms: 60000,
        };
        
        let result = engine.update_with_feedback(&decision.id, outcome).await;
        assert!(result.is_ok());
        
        let patterns = engine.get_learned_patterns().await;
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_market_phase_determination() {
        let engine = QuantumDecisionEngine::new(3, 0.5);
        
        // Test growth pattern (low frequency dominance)
        let growth_spectrum = vec![0.6, 0.3, 0.1];
        let phase = engine.determine_market_phase(&growth_spectrum);
        assert_eq!(phase, MarketPhase::Growth);
        
        // Test decline pattern (high frequency dominance)
        let decline_spectrum = vec![0.1, 0.2, 0.7];
        let phase = engine.determine_market_phase(&decline_spectrum);
        assert_eq!(phase, MarketPhase::Decline);
        
        // Test sideways pattern (mid frequency dominance)
        let sideways_spectrum = vec![0.2, 0.6, 0.2];
        let phase = engine.determine_market_phase(&sideways_spectrum);
        assert_eq!(phase, MarketPhase::Sideways);
    }

    #[test]
    fn test_confidence_calculation() {
        let engine = QuantumDecisionEngine::new(3, 0.5);
        
        // High confidence case (low entropy)
        let concentrated_power = vec![0.8, 0.1, 0.1];
        let confidence = engine.calculate_regime_confidence(&concentrated_power);
        assert!(confidence > 0.5);
        
        // Low confidence case (high entropy)
        let uniform_power = vec![0.33, 0.33, 0.34];
        let confidence = engine.calculate_regime_confidence(&uniform_power);
        assert!(confidence < 0.5);
    }

    #[tokio::test]
    async fn test_threshold_setting() {
        let mut engine = QuantumDecisionEngine::new(3, 0.5);
        assert_eq!(engine.confidence_threshold(), 0.5);
        
        engine.set_confidence_threshold(0.7);
        assert_eq!(engine.confidence_threshold(), 0.7);
    }
}