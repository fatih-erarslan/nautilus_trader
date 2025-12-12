//! Decision Engine Implementation
//!
//! Core decision-making engine that combines quantum analysis with classical trading logic.

use crate::core::{QarResult, DecisionType, FactorMap, TradingDecision};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use super::{DecisionConfig, EnhancedTradingDecision, RiskAssessment, ExecutionPlan, QuantumInsights};
use prospect_theory::{
    QuantumProspectTheory, QuantumProspectTheoryConfig, DecisionContext as PTDecisionContext,
    ProspectDecision, FramingContext, FrameType, TradingAction, BehavioralFactors
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core decision engine implementation
pub struct DecisionEngineImpl {
    config: DecisionConfig,
    decision_history: Vec<EnhancedTradingDecision>,
    performance_metrics: PerformanceMetrics,
    decision_algorithms: DecisionAlgorithms,
    prospect_theory: QuantumProspectTheory,
}

/// Performance metrics for decision tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total decisions made
    pub total_decisions: usize,
    /// Successful decisions
    pub successful_decisions: usize,
    /// Average confidence
    pub average_confidence: f64,
    /// Average execution time
    pub average_execution_time: f64,
    /// Decision accuracy by type
    pub accuracy_by_type: HashMap<DecisionType, f64>,
}

/// Decision algorithms configuration
#[derive(Debug)]
pub struct DecisionAlgorithms {
    /// Quantum decision weight
    pub quantum_weight: f64,
    /// Classical decision weight
    pub classical_weight: f64,
    /// Risk adjustment factor
    pub risk_adjustment: f64,
    /// Momentum consideration
    pub momentum_factor: f64,
    /// Volatility consideration
    pub volatility_factor: f64,
}

/// Decision context for enhanced decision making
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// Current market factors
    pub factors: FactorMap,
    /// Analysis results
    pub analysis: AnalysisResult,
    /// Historical performance
    pub historical_performance: Option<PerformanceMetrics>,
    /// Risk constraints
    pub risk_constraints: RiskConstraints,
}

/// Risk constraints for decision making
#[derive(Debug, Clone)]
pub struct RiskConstraints {
    /// Maximum position size
    pub max_position_size: f64,
    /// Maximum risk per trade
    pub max_risk_per_trade: f64,
    /// Maximum portfolio risk
    pub max_portfolio_risk: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_decisions: 0,
            successful_decisions: 0,
            average_confidence: 0.0,
            average_execution_time: 0.0,
            accuracy_by_type: HashMap::new(),
        }
    }
}

impl Default for DecisionAlgorithms {
    fn default() -> Self {
        Self {
            quantum_weight: 0.7,
            classical_weight: 0.3,
            risk_adjustment: 0.2,
            momentum_factor: 0.15,
            volatility_factor: 0.1,
        }
    }
}

impl DecisionEngineImpl {
    /// Create a new decision engine
    pub fn new(config: DecisionConfig) -> QarResult<Self> {
        let pt_config = QuantumProspectTheoryConfig {
            alpha: 0.88,
            beta: 0.88,
            lambda: 2.25,
            tversky_kahneman_gamma: 0.61,
            prelec_alpha: 0.69,
            prelec_beta: 1.0,
            quantum_enhancement: true,
            cache_size: 50000,
            target_latency_ns: 500,
        };
        
        let prospect_theory = QuantumProspectTheory::new(pt_config)
            .map_err(|e| QarError::DecisionEngine(format!("Prospect Theory init failed: {}", e)))?;
        
        Ok(Self {
            config,
            decision_history: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
            decision_algorithms: DecisionAlgorithms::default(),
            prospect_theory,
        })
    }

    /// Make an enhanced trading decision
    pub async fn make_enhanced_decision(
        &mut self,
        context: DecisionContext,
    ) -> QarResult<EnhancedTradingDecision> {
        let start_time = std::time::Instant::now();

        // Generate base decision using quantum or classical approach
        let base_decision = if self.config.use_quantum {
            self.make_quantum_enhanced_decision(&context).await?
        } else {
            self.make_classical_decision(&context).await?
        };

        // Validate decision against constraints
        self.validate_decision(&base_decision, &context.risk_constraints)?;

        // Assess risk
        let risk_assessment = self.assess_decision_risk(&base_decision, &context).await?;

        // Create execution plan
        let execution_plan = self.create_execution_plan(&base_decision, &risk_assessment, &context).await?;

        // Generate quantum insights if available
        let quantum_insights = if self.config.use_quantum {
            Some(self.generate_quantum_insights(&context).await?)
        } else {
            None
        };

        // Create enhanced decision
        let enhanced_decision = EnhancedTradingDecision {
            decision: base_decision,
            quantum_insights,
            risk_assessment,
            execution_plan,
            metadata: self.create_decision_metadata(&context),
        };

        // Update performance metrics
        let execution_time = start_time.elapsed().as_secs_f64();
        self.update_performance_metrics(&enhanced_decision, execution_time);

        // Store in history
        self.decision_history.push(enhanced_decision.clone());

        Ok(enhanced_decision)
    }

    /// Make quantum-enhanced decision
    async fn make_quantum_enhanced_decision(&self, context: &DecisionContext) -> QarResult<TradingDecision> {
        // Use Quantum Prospect Theory for behavioral decision making
        let pt_decision = self.make_prospect_theory_decision(context)?;
        
        // Extract quantum-relevant factors
        let quantum_factors = self.extract_quantum_factors(&context.factors)?;
        
        // Calculate quantum decision weights enhanced by prospect theory
        let quantum_weights = self.calculate_quantum_weights(&quantum_factors, &context.analysis)?;
        
        // Generate quantum decision
        let quantum_decision = self.apply_quantum_algorithm(&quantum_weights, context)?;
        
        // Combine with classical logic
        let classical_component = self.make_classical_decision(context).await?;
        
        // Blend all three approaches: Prospect Theory, Quantum, and Classical
        let blended_decision = self.blend_three_decisions(&pt_decision, &quantum_decision, &classical_component)?;
        
        Ok(blended_decision)
    }
    
    /// Make decision using Quantum Prospect Theory
    fn make_prospect_theory_decision(&self, context: &DecisionContext) -> QarResult<TradingDecision> {
        // Extract market factors for prospect theory
        let trend = context.factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let volatility = context.factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let momentum = context.factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let sentiment = context.factors.get_factor(&crate::core::StandardFactors::Sentiment)?;
        
        // Create market data for prospect theory
        let current_price = 100.0; // Normalized price
        let volatility_range = volatility * 10.0; // Scale volatility
        
        let market_data = prospect_theory::MarketData {
            symbol: "TRADING_PAIR".to_string(),
            current_price,
            possible_outcomes: vec![
                current_price + volatility_range * 2.0,
                current_price + volatility_range,
                current_price,
                current_price - volatility_range,
                current_price - volatility_range * 2.0,
            ],
            buy_probabilities: vec![
                trend * 0.4,
                trend * 0.3,
                0.2,
                (1.0 - trend) * 0.2,
                (1.0 - trend) * 0.1,
            ],
            sell_probabilities: vec![
                (1.0 - trend) * 0.1,
                (1.0 - trend) * 0.2,
                0.2,
                trend * 0.3,
                trend * 0.4,
            ],
            hold_probabilities: vec![0.2, 0.2, 0.2, 0.2, 0.2],
            frame: FramingContext {
                frame_type: if sentiment > 0.6 { FrameType::Gain } 
                           else if sentiment < 0.4 { FrameType::Loss } 
                           else { FrameType::Neutral },
                emphasis: sentiment,
            },
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        
        // Make prospect theory decision
        let pt_decision = self.prospect_theory.make_trading_decision(&market_data, None)
            .map_err(|e| QarError::DecisionEngine(format!("Prospect Theory decision failed: {}", e)))?;
        
        // Convert to our TradingDecision format
        let decision_type = match pt_decision.action {
            TradingAction::Buy => DecisionType::Buy,
            TradingAction::Sell => DecisionType::Sell,
            TradingAction::Hold => DecisionType::Hold,
        };
        
        Ok(TradingDecision {
            decision_type,
            confidence: pt_decision.confidence,
            expected_return: Some(pt_decision.expected_value),
            risk_assessment: Some(pt_decision.risk_metric),
            urgency_score: Some(momentum),
            reasoning: format!("Prospect Theory: value={:.3}, loss_aversion={:.3}, behavioral_bias={:.3}",
                             pt_decision.prospect_value,
                             pt_decision.behavioral_factors.loss_aversion_impact,
                             pt_decision.behavioral_factors.probability_weighting_bias),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Extract quantum-relevant factors
    fn extract_quantum_factors(&self, factors: &FactorMap) -> QarResult<QuantumFactors> {
        Ok(QuantumFactors {
            superposition_factor: self.calculate_superposition_factor(factors)?,
            entanglement_factor: self.calculate_entanglement_factor(factors)?,
            interference_factor: self.calculate_interference_factor(factors)?,
            coherence_factor: self.calculate_coherence_factor(factors)?,
        })
    }

    /// Calculate superposition factor
    fn calculate_superposition_factor(&self, factors: &FactorMap) -> QarResult<f64> {
        // Superposition represents multiple market states simultaneously
        let trend = factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let momentum = factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        // Higher volatility and uncertainty increase superposition
        let uncertainty = (0.5 - (trend - 0.5).abs()) * 2.0; // Maximum when trend is neutral
        let superposition = (volatility + uncertainty + (1.0 - momentum.abs())) / 3.0;
        
        Ok(superposition.max(0.0).min(1.0))
    }

    /// Calculate entanglement factor
    fn calculate_entanglement_factor(&self, factors: &FactorMap) -> QarResult<f64> {
        // Entanglement represents correlation between different market factors
        let volume = factors.get_factor(&crate::core::StandardFactors::Volume)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let sentiment = factors.get_factor(&crate::core::StandardFactors::Sentiment)?;
        
        // Calculate correlations between factors
        let volume_liquidity_corr = (volume - liquidity).abs();
        let sentiment_volume_corr = (sentiment - volume).abs();
        
        // Higher correlation (lower difference) means higher entanglement
        let entanglement = 1.0 - (volume_liquidity_corr + sentiment_volume_corr) / 2.0;
        
        Ok(entanglement.max(0.0).min(1.0))
    }

    /// Calculate interference factor
    fn calculate_interference_factor(&self, factors: &FactorMap) -> QarResult<f64> {
        // Interference represents constructive/destructive market forces
        let trend = factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let momentum = factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let sentiment = factors.get_factor(&crate::core::StandardFactors::Sentiment)?;
        
        // Calculate alignment between forces
        let trend_momentum_alignment = 1.0 - (trend - momentum).abs();
        let trend_sentiment_alignment = 1.0 - (trend - sentiment).abs();
        
        // Higher alignment means constructive interference
        let interference = (trend_momentum_alignment + trend_sentiment_alignment) / 2.0;
        
        Ok(interference.max(0.0).min(1.0))
    }

    /// Calculate coherence factor
    fn calculate_coherence_factor(&self, factors: &FactorMap) -> QarResult<f64> {
        // Coherence represents market state stability
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        let risk = factors.get_factor(&crate::core::StandardFactors::Risk)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        // Higher efficiency and lower risk/volatility increase coherence
        let coherence = (efficiency + (1.0 - risk) + (1.0 - volatility)) / 3.0;
        
        Ok(coherence.max(0.0).min(1.0))
    }

    /// Calculate quantum decision weights
    fn calculate_quantum_weights(&self, quantum_factors: &QuantumFactors, analysis: &AnalysisResult) -> QarResult<QuantumWeights> {
        Ok(QuantumWeights {
            amplitude_weights: self.calculate_amplitude_weights(quantum_factors, analysis)?,
            phase_weights: self.calculate_phase_weights(quantum_factors, analysis)?,
            probability_weights: self.calculate_probability_weights(quantum_factors, analysis)?,
        })
    }

    /// Calculate amplitude weights
    fn calculate_amplitude_weights(&self, quantum_factors: &QuantumFactors, analysis: &AnalysisResult) -> QarResult<Vec<f64>> {
        let mut weights = Vec::new();
        
        // Buy amplitude
        let buy_amplitude = (quantum_factors.superposition_factor * analysis.trend_strength +
                           quantum_factors.interference_factor * analysis.confidence) / 2.0;
        weights.push(buy_amplitude);
        
        // Sell amplitude
        let sell_amplitude = (quantum_factors.superposition_factor * (1.0 - analysis.trend_strength) +
                            quantum_factors.interference_factor * analysis.confidence) / 2.0;
        weights.push(sell_amplitude);
        
        // Hold amplitude
        let hold_amplitude = quantum_factors.coherence_factor * (1.0 - analysis.confidence);
        weights.push(hold_amplitude);
        
        Ok(weights)
    }

    /// Calculate phase weights
    fn calculate_phase_weights(&self, quantum_factors: &QuantumFactors, analysis: &AnalysisResult) -> QarResult<Vec<f64>> {
        let mut weights = Vec::new();
        
        // Phase represents timing/momentum component
        let base_phase = quantum_factors.entanglement_factor * std::f64::consts::PI;
        
        // Buy phase
        weights.push(base_phase * analysis.trend_strength);
        
        // Sell phase
        weights.push(base_phase * (1.0 - analysis.trend_strength));
        
        // Hold phase
        weights.push(base_phase * 0.5); // Neutral phase
        
        Ok(weights)
    }

    /// Calculate probability weights
    fn calculate_probability_weights(&self, quantum_factors: &QuantumFactors, analysis: &AnalysisResult) -> QarResult<Vec<f64>> {
        let amplitude_weights = self.calculate_amplitude_weights(quantum_factors, analysis)?;
        
        // Probabilities are squares of amplitudes (Born rule)
        let mut probabilities: Vec<f64> = amplitude_weights.iter().map(|a| a.powi(2)).collect();
        
        // Normalize probabilities
        let total: f64 = probabilities.iter().sum();
        if total > 0.0 {
            probabilities.iter_mut().for_each(|p| *p /= total);
        } else {
            // Equal probabilities if total is zero
            probabilities = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        }
        
        Ok(probabilities)
    }

    /// Apply quantum algorithm
    fn apply_quantum_algorithm(&self, weights: &QuantumWeights, context: &DecisionContext) -> QarResult<TradingDecision> {
        // Select decision type based on highest probability
        let decision_type = if weights.probability_weights[0] > weights.probability_weights[1] && 
                             weights.probability_weights[0] > weights.probability_weights[2] {
            DecisionType::Buy
        } else if weights.probability_weights[1] > weights.probability_weights[2] {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };

        // Calculate confidence based on quantum coherence
        let max_probability = weights.probability_weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let confidence = max_probability * context.analysis.confidence;

        // Calculate expected return using quantum interference
        let trend_factor = context.factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let quantum_enhancement = weights.probability_weights[0] - weights.probability_weights[1]; // Buy prob - Sell prob
        let expected_return = Some((trend_factor - 0.5) * 0.1 * quantum_enhancement);

        // Risk assessment based on quantum uncertainty
        let uncertainty = 1.0 - weights.probability_weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let risk_assessment = Some(uncertainty * context.analysis.confidence);

        // Urgency based on momentum and quantum phase
        let momentum = context.factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let phase_urgency = weights.phase_weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0) / std::f64::consts::PI;
        let urgency_score = Some(momentum * phase_urgency);

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return,
            risk_assessment,
            urgency_score,
            reasoning: format!("Quantum decision: probabilities=[{:.3}, {:.3}, {:.3}], coherence={:.3}", 
                             weights.probability_weights[0], weights.probability_weights[1], weights.probability_weights[2],
                             context.analysis.confidence),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Make classical decision
    async fn make_classical_decision(&self, context: &DecisionContext) -> QarResult<TradingDecision> {
        let trend = context.factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let momentum = context.factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let volatility = context.factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let sentiment = context.factors.get_factor(&crate::core::StandardFactors::Sentiment)?;

        // Classical decision logic
        let buy_score = trend * 0.4 + momentum * 0.3 + sentiment * 0.2 + (1.0 - volatility) * 0.1;
        let sell_score = (1.0 - trend) * 0.4 + (1.0 - momentum) * 0.3 + (1.0 - sentiment) * 0.2 + volatility * 0.1;
        let hold_score = 1.0 - (buy_score - sell_score).abs();

        let decision_type = if buy_score > sell_score && buy_score > hold_score {
            DecisionType::Buy
        } else if sell_score > hold_score {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };

        let confidence = match decision_type {
            DecisionType::Buy => buy_score,
            DecisionType::Sell => sell_score,
            DecisionType::Hold => hold_score,
        };

        let expected_return = Some((trend - 0.5) * 0.08);
        let risk_assessment = Some(volatility);
        let urgency_score = Some(momentum);

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return,
            risk_assessment,
            urgency_score,
            reasoning: format!("Classical analysis: buy={:.2}, sell={:.2}, hold={:.2}", buy_score, sell_score, hold_score),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Blend prospect theory, quantum, and classical decisions
    fn blend_three_decisions(&self, 
                           prospect: &TradingDecision, 
                           quantum: &TradingDecision, 
                           classical: &TradingDecision) -> QarResult<TradingDecision> {
        // Enhanced weighting that considers prospect theory
        let pt_weight = 0.4;  // Prospect Theory gets high weight for behavioral insights
        let quantum_weight = self.decision_algorithms.quantum_weight * 0.6;
        let classical_weight = self.decision_algorithms.classical_weight * 0.6;
        
        // Normalize weights
        let total_weight = pt_weight + quantum_weight + classical_weight;
        let pt_w = pt_weight / total_weight;
        let q_w = quantum_weight / total_weight;
        let c_w = classical_weight / total_weight;

        // Weighted confidence
        let confidence = prospect.confidence * pt_w + quantum.confidence * q_w + classical.confidence * c_w;

        // Decision type based on highest weighted confidence
        let pt_score = prospect.confidence * pt_w;
        let q_score = quantum.confidence * q_w;
        let c_score = classical.confidence * c_w;
        
        let decision_type = if pt_score >= q_score && pt_score >= c_score {
            prospect.decision_type.clone()
        } else if q_score >= c_score {
            quantum.decision_type.clone()
        } else {
            classical.decision_type.clone()
        };

        // Blend other metrics
        let expected_return = match (prospect.expected_return, quantum.expected_return, classical.expected_return) {
            (Some(p), Some(q), Some(c)) => Some(p * pt_w + q * q_w + c * c_w),
            (Some(p), Some(q), None) => Some(p * (pt_w / (pt_w + q_w)) + q * (q_w / (pt_w + q_w))),
            (Some(p), None, Some(c)) => Some(p * (pt_w / (pt_w + c_w)) + c * (c_w / (pt_w + c_w))),
            (None, Some(q), Some(c)) => Some(q * (q_w / (q_w + c_w)) + c * (c_w / (q_w + c_w))),
            (Some(p), None, None) => Some(p),
            (None, Some(q), None) => Some(q),
            (None, None, Some(c)) => Some(c),
            (None, None, None) => None,
        };

        let risk_assessment = match (prospect.risk_assessment, quantum.risk_assessment, classical.risk_assessment) {
            (Some(p), Some(q), Some(c)) => Some(p * pt_w + q * q_w + c * c_w),
            (Some(p), Some(q), None) => Some(p * (pt_w / (pt_w + q_w)) + q * (q_w / (pt_w + q_w))),
            (Some(p), None, Some(c)) => Some(p * (pt_w / (pt_w + c_w)) + c * (c_w / (pt_w + c_w))),
            (None, Some(q), Some(c)) => Some(q * (q_w / (q_w + c_w)) + c * (c_w / (q_w + c_w))),
            (Some(p), None, None) => Some(p),
            (None, Some(q), None) => Some(q),
            (None, None, Some(c)) => Some(c),
            (None, None, None) => None,
        };

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return,
            risk_assessment,
            urgency_score: quantum.urgency_score, // Use quantum urgency as it's most sophisticated
            reasoning: format!("Integrated decision (PT:{:.1}, Q:{:.1}, C:{:.1}): PT={}, Q={}, C={}", 
                             pt_w, q_w, c_w, prospect.reasoning, quantum.reasoning, classical.reasoning),
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Blend quantum and classical decisions (legacy method)
    fn blend_decisions(&self, quantum: &TradingDecision, classical: &TradingDecision) -> QarResult<TradingDecision> {
        let quantum_weight = self.decision_algorithms.quantum_weight;
        let classical_weight = self.decision_algorithms.classical_weight;

        // Weighted confidence
        let confidence = quantum.confidence * quantum_weight + classical.confidence * classical_weight;

        // Decision type based on weighted confidence
        let decision_type = if quantum_weight > classical_weight {
            quantum.decision_type.clone()
        } else {
            classical.decision_type.clone()
        };

        // Blend other metrics
        let expected_return = match (quantum.expected_return, classical.expected_return) {
            (Some(q), Some(c)) => Some(q * quantum_weight + c * classical_weight),
            (Some(q), None) => Some(q),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        let risk_assessment = match (quantum.risk_assessment, classical.risk_assessment) {
            (Some(q), Some(c)) => Some(q * quantum_weight + c * classical_weight),
            (Some(q), None) => Some(q),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        let urgency_score = match (quantum.urgency_score, classical.urgency_score) {
            (Some(q), Some(c)) => Some(q * quantum_weight + c * classical_weight),
            (Some(q), None) => Some(q),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return,
            risk_assessment,
            urgency_score,
            reasoning: format!("Blended decision (Q:{:.1}, C:{:.1}): {}", 
                             quantum_weight, classical_weight, quantum.reasoning),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Validate decision against constraints
    fn validate_decision(&self, decision: &TradingDecision, constraints: &RiskConstraints) -> QarResult<()> {
        if decision.confidence < constraints.min_confidence {
            return Err(QarError::DecisionEngine(
                format!("Decision confidence {:.2} below minimum {:.2}", 
                       decision.confidence, constraints.min_confidence)
            ));
        }

        if let Some(risk) = decision.risk_assessment {
            if risk > constraints.max_risk_per_trade {
                return Err(QarError::DecisionEngine(
                    format!("Decision risk {:.2} exceeds maximum {:.2}", 
                           risk, constraints.max_risk_per_trade)
                ));
            }
        }

        Ok(())
    }

    /// Assess decision risk
    async fn assess_decision_risk(&self, decision: &TradingDecision, context: &DecisionContext) -> QarResult<RiskAssessment> {
        let base_risk = decision.risk_assessment.unwrap_or(0.5);
        let volatility = context.factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = context.factors.get_factor(&crate::core::StandardFactors::Liquidity)?;

        let risk_score = (base_risk + volatility + (1.0 - liquidity)) / 3.0;
        
        // Calculate VaR and other risk metrics
        let var_95 = risk_score * 0.1; // Simplified VaR calculation
        let expected_shortfall = var_95 * 1.3;
        let max_drawdown_risk = risk_score * 0.15;
        let liquidity_risk = 1.0 - liquidity;
        let risk_adjusted_return = decision.expected_return.unwrap_or(0.0) / risk_score.max(0.01);

        Ok(RiskAssessment {
            risk_score,
            var_95,
            expected_shortfall,
            max_drawdown_risk,
            liquidity_risk,
            risk_adjusted_return,
        })
    }

    /// Create execution plan
    async fn create_execution_plan(&self, decision: &TradingDecision, risk: &RiskAssessment, context: &DecisionContext) -> QarResult<ExecutionPlan> {
        let strategy = match decision.decision_type {
            DecisionType::Buy | DecisionType::Sell if decision.urgency_score.unwrap_or(0.0) > 0.7 => {
                super::ExecutionStrategy::Market
            },
            DecisionType::Buy | DecisionType::Sell => super::ExecutionStrategy::Limit,
            DecisionType::Hold => super::ExecutionStrategy::TWAP,
        };

        let position_size = context.risk_constraints.max_position_size * decision.confidence;
        
        let stop_loss = if matches!(decision.decision_type, DecisionType::Buy) {
            Some(0.95) // 5% stop loss for buy
        } else if matches!(decision.decision_type, DecisionType::Sell) {
            Some(1.05) // 5% stop loss for sell
        } else {
            None
        };

        let take_profit = decision.expected_return.map(|ret| {
            if ret > 0.0 { 1.0 + ret * 2.0 } else { 1.0 + ret.abs() }
        });

        let time_horizon = match decision.urgency_score.unwrap_or(0.0) {
            x if x > 0.8 => std::time::Duration::from_secs(300),  // 5 minutes
            x if x > 0.5 => std::time::Duration::from_secs(1800), // 30 minutes
            _ => std::time::Duration::from_secs(3600), // 1 hour
        };

        let priority = if risk.risk_score > 0.7 {
            super::ExecutionPriority::High
        } else if decision.urgency_score.unwrap_or(0.0) > 0.6 {
            super::ExecutionPriority::Medium
        } else {
            super::ExecutionPriority::Low
        };

        Ok(ExecutionPlan {
            strategy,
            position_size,
            stop_loss,
            take_profit,
            time_horizon,
            priority,
        })
    }

    /// Generate quantum insights
    async fn generate_quantum_insights(&self, context: &DecisionContext) -> QarResult<QuantumInsights> {
        let quantum_factors = self.extract_quantum_factors(&context.factors)?;
        
        // Generate superposition analysis
        let superposition_analysis = vec![
            quantum_factors.superposition_factor,
            1.0 - quantum_factors.superposition_factor,
        ];

        // Generate entanglement correlations
        let mut entanglement_correlations = HashMap::new();
        entanglement_correlations.insert("volume_liquidity".to_string(), quantum_factors.entanglement_factor);
        entanglement_correlations.insert("trend_momentum".to_string(), quantum_factors.interference_factor);
        entanglement_correlations.insert("sentiment_risk".to_string(), quantum_factors.coherence_factor);

        // Generate interference patterns
        let interference_patterns = vec![
            quantum_factors.interference_factor,
            quantum_factors.interference_factor * 0.8,
            quantum_factors.interference_factor * 0.6,
        ];

        let measurement_uncertainty = 1.0 - quantum_factors.coherence_factor;

        Ok(QuantumInsights {
            superposition_analysis,
            entanglement_correlations,
            interference_patterns,
            measurement_uncertainty,
        })
    }

    /// Create decision metadata
    fn create_decision_metadata(&self, context: &DecisionContext) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("engine_version".to_string(), "2.0.0".to_string());
        metadata.insert("quantum_enabled".to_string(), self.config.use_quantum.to_string());
        metadata.insert("analysis_confidence".to_string(), context.analysis.confidence.to_string());
        metadata.insert("total_decisions".to_string(), self.performance_metrics.total_decisions.to_string());
        metadata.insert("decision_timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, decision: &EnhancedTradingDecision, execution_time: f64) {
        self.performance_metrics.total_decisions += 1;
        
        // Update average confidence
        let new_confidence = decision.decision.confidence;
        let total = self.performance_metrics.total_decisions as f64;
        self.performance_metrics.average_confidence = 
            (self.performance_metrics.average_confidence * (total - 1.0) + new_confidence) / total;

        // Update average execution time
        self.performance_metrics.average_execution_time = 
            (self.performance_metrics.average_execution_time * (total - 1.0) + execution_time) / total;

        // Update accuracy by type (simplified - in real implementation would track actual outcomes)
        let accuracy = decision.decision.confidence; // Use confidence as proxy for accuracy
        self.performance_metrics.accuracy_by_type
            .entry(decision.decision.decision_type.clone())
            .and_modify(|e| *e = (*e + accuracy) / 2.0)
            .or_insert(accuracy);
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get decision history
    pub fn get_decision_history(&self) -> &[EnhancedTradingDecision] {
        &self.decision_history
    }

    /// Get latest decision
    pub fn get_latest_decision(&self) -> Option<&EnhancedTradingDecision> {
        self.decision_history.last()
    }

    /// Update decision algorithms
    pub fn update_algorithms(&mut self, algorithms: DecisionAlgorithms) {
        self.decision_algorithms = algorithms;
    }
}

/// Quantum factors structure
#[derive(Debug, Clone)]
pub struct QuantumFactors {
    pub superposition_factor: f64,
    pub entanglement_factor: f64,
    pub interference_factor: f64,
    pub coherence_factor: f64,
}

/// Quantum weights for decision making
#[derive(Debug, Clone)]
pub struct QuantumWeights {
    pub amplitude_weights: Vec<f64>,
    pub phase_weights: Vec<f64>,
    pub probability_weights: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    fn create_test_context() -> DecisionContext {
        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volatility.to_string(), 0.4);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.7);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.6);
        factors.insert(StandardFactors::Risk.to_string(), 0.3);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.8);
        
        let factor_map = FactorMap::new(factors).unwrap();
        
        let analysis = AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        };

        let risk_constraints = RiskConstraints {
            max_position_size: 0.1,
            max_risk_per_trade: 0.05,
            max_portfolio_risk: 0.2,
            min_confidence: 0.6,
        };

        DecisionContext {
            factors: factor_map,
            analysis,
            historical_performance: None,
            risk_constraints,
        }
    }

    #[tokio::test]
    async fn test_decision_engine_creation() {
        let config = DecisionConfig::default();
        let engine = DecisionEngineImpl::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_decision_making() {
        let config = DecisionConfig {
            use_quantum: true,
            ..Default::default()
        };
        let mut engine = DecisionEngineImpl::new(config).unwrap();
        let context = create_test_context();

        let decision = engine.make_enhanced_decision(context).await;
        assert!(decision.is_ok());

        let decision = decision.unwrap();
        assert!(decision.quantum_insights.is_some());
        assert!(decision.decision.confidence > 0.0);
        assert!(decision.decision.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_classical_decision_making() {
        let config = DecisionConfig {
            use_quantum: false,
            ..Default::default()
        };
        let mut engine = DecisionEngineImpl::new(config).unwrap();
        let context = create_test_context();

        let decision = engine.make_enhanced_decision(context).await;
        assert!(decision.is_ok());

        let decision = decision.unwrap();
        assert!(decision.quantum_insights.is_none());
        assert!(decision.decision.confidence > 0.0);
    }

    #[test]
    fn test_quantum_factors_calculation() {
        let config = DecisionConfig::default();
        let engine = DecisionEngineImpl::new(config).unwrap();
        let context = create_test_context();

        let quantum_factors = engine.extract_quantum_factors(&context.factors).unwrap();
        
        assert!(quantum_factors.superposition_factor >= 0.0 && quantum_factors.superposition_factor <= 1.0);
        assert!(quantum_factors.entanglement_factor >= 0.0 && quantum_factors.entanglement_factor <= 1.0);
        assert!(quantum_factors.interference_factor >= 0.0 && quantum_factors.interference_factor <= 1.0);
        assert!(quantum_factors.coherence_factor >= 0.0 && quantum_factors.coherence_factor <= 1.0);
    }

    #[test]
    fn test_decision_validation() {
        let config = DecisionConfig::default();
        let engine = DecisionEngineImpl::new(config).unwrap();
        
        let constraints = RiskConstraints {
            max_position_size: 0.1,
            max_risk_per_trade: 0.05,
            max_portfolio_risk: 0.2,
            min_confidence: 0.6,
        };

        // Valid decision
        let valid_decision = TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            expected_return: Some(0.03),
            risk_assessment: Some(0.04),
            urgency_score: Some(0.5),
            reasoning: "Test decision".to_string(),
            timestamp: chrono::Utc::now(),
        };

        assert!(engine.validate_decision(&valid_decision, &constraints).is_ok());

        // Invalid decision - low confidence
        let invalid_decision = TradingDecision {
            confidence: 0.5, // Below minimum
            ..valid_decision.clone()
        };

        assert!(engine.validate_decision(&invalid_decision, &constraints).is_err());

        // Invalid decision - high risk
        let high_risk_decision = TradingDecision {
            risk_assessment: Some(0.08), // Above maximum
            ..valid_decision
        };

        assert!(engine.validate_decision(&high_risk_decision, &constraints).is_err());
    }

    #[test]
    fn test_probability_normalization() {
        let config = DecisionConfig::default();
        let engine = DecisionEngineImpl::new(config).unwrap();
        
        let quantum_factors = QuantumFactors {
            superposition_factor: 0.7,
            entanglement_factor: 0.6,
            interference_factor: 0.8,
            coherence_factor: 0.9,
        };

        let analysis = AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        };

        let probabilities = engine.calculate_probability_weights(&quantum_factors, &analysis).unwrap();
        
        // Probabilities should sum to 1.0
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        
        // All probabilities should be non-negative
        assert!(probabilities.iter().all(|&p| p >= 0.0));
    }

    #[tokio::test]
    async fn test_performance_metrics_update() {
        let config = DecisionConfig::default();
        let mut engine = DecisionEngineImpl::new(config).unwrap();
        let context = create_test_context();

        // Make several decisions
        for _ in 0..5 {
            let _ = engine.make_enhanced_decision(context.clone()).await.unwrap();
        }

        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_decisions, 5);
        assert!(metrics.average_confidence > 0.0);
        assert!(metrics.average_execution_time >= 0.0);
        assert!(!metrics.accuracy_by_type.is_empty());
    }

    #[test]
    fn test_decision_blending() {
        let config = DecisionConfig::default();
        let engine = DecisionEngineImpl::new(config).unwrap();

        let quantum_decision = TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            expected_return: Some(0.05),
            risk_assessment: Some(0.3),
            urgency_score: Some(0.7),
            reasoning: "Quantum decision".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let classical_decision = TradingDecision {
            decision_type: DecisionType::Hold,
            confidence: 0.6,
            expected_return: Some(0.02),
            risk_assessment: Some(0.4),
            urgency_score: Some(0.5),
            reasoning: "Classical decision".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let blended = engine.blend_decisions(&quantum_decision, &classical_decision).unwrap();
        
        // Should favor quantum decision due to higher weight
        assert_eq!(blended.decision_type, DecisionType::Buy);
        
        // Confidence should be weighted average
        let expected_confidence = 0.8 * 0.7 + 0.6 * 0.3;
        assert!((blended.confidence - expected_confidence).abs() < 0.01);
    }
}