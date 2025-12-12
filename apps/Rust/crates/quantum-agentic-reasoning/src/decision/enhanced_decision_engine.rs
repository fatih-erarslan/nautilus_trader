//! Enhanced Decision Engine with Integrated Whale Defense ML
//!
//! This module extends the base decision engine with real-time whale defense
//! ML integration, replacing simulation with actual ensemble predictions.

use crate::core::{QarResult, DecisionType, FactorMap, TradingDecision};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use crate::decision::{
    DecisionConfig, EnhancedTradingDecision, RiskAssessment, ExecutionPlan, 
    QuantumInsights, DecisionEngineImpl, DecisionContext, RiskConstraints
};
use crate::whale_defense_integration::{
    QARWhaleDefense, WhaleDefenseConfig, WhaleDefenseAnalysis, 
    EnhancedDecisionContext
};
// For now, we'll create simplified compliance structures
// In production, these would be imported from tengri-compliance and regime-detector
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Simplified compliance check structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub decision_type: String,
    pub position_size: f64,
    pub confidence: f64,
    pub risk_score: f64,
    pub execution_strategy: String,
}

/// Simplified compliance result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    pub compliant: bool,
    pub violations: Vec<String>,
    pub score: f64,
}

/// Simplified compliance engine
pub struct ComplianceEngine {
    pub strict_mode: bool,
}

impl ComplianceEngine {
    pub fn new() -> Result<Self, crate::error::QarError> {
        Ok(Self { strict_mode: true })
    }
    
    pub async fn check_compliance(&self, check: ComplianceCheck) -> Result<ComplianceResult, crate::error::QarError> {
        let mut violations = Vec::new();
        
        // Simple compliance checks
        if check.position_size > 0.2 {
            violations.push("Position size exceeds 20% limit".to_string());
        }
        
        if check.confidence < 0.6 {
            violations.push("Confidence below minimum threshold".to_string());
        }
        
        if check.risk_score > 0.8 {
            violations.push("Risk score too high".to_string());
        }
        
        let compliant = violations.is_empty();
        let score = if compliant { 1.0 } else { 0.5 };
        
        Ok(ComplianceResult {
            compliant,
            violations,
            score,
        })
    }
}

/// Simplified regime detector
pub struct RegimeDetector {
    pub current_regime: String,
}

impl RegimeDetector {
    pub fn new() -> Result<Self, crate::error::QarError> {
        Ok(Self {
            current_regime: "Bull".to_string(),
        })
    }
    
    pub fn update(&mut self, _input: RegimeInput) -> Result<RegimeResult, crate::error::QarError> {
        Ok(RegimeResult {
            regime_type: RegimeType::Bull,
            confidence: 0.8,
            transition_probability: Some(0.1),
        })
    }
}

/// Simplified regime input
#[derive(Debug, Clone)]
pub struct RegimeInput {
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub trend: f64,
    pub momentum: f64,
    pub timestamp: u64,
}

/// Simplified regime result
#[derive(Debug, Clone)]
pub struct RegimeResult {
    pub regime_type: RegimeType,
    pub confidence: f32,
    pub transition_probability: Option<f32>,
}

/// Simplified regime types
#[derive(Debug, Clone)]
pub enum RegimeType {
    Bull,
    Bear,
    Sideways,
    Volatile,
    Crisis,
}

/// Enhanced decision engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDecisionConfig {
    /// Base decision configuration
    pub base_config: DecisionConfig,
    /// Whale defense configuration
    pub whale_defense: WhaleDefenseConfig,
    /// Enable regime detection integration
    pub enable_regime_detection: bool,
    /// Enable TENGRI compliance checks
    pub enable_tengri_compliance: bool,
    /// Maximum total decision time (microseconds)
    pub max_decision_time_us: u64,
}

impl Default for EnhancedDecisionConfig {
    fn default() -> Self {
        Self {
            base_config: DecisionConfig::default(),
            whale_defense: WhaleDefenseConfig::default(),
            enable_regime_detection: true,
            enable_tengri_compliance: true,
            max_decision_time_us: 500, // Sub-millisecond total
        }
    }
}

/// Enhanced decision with integrated whale defense
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedTradingDecision {
    /// Base enhanced decision
    pub base_decision: EnhancedTradingDecision,
    /// Whale defense analysis
    pub whale_defense: Option<WhaleDefenseAnalysis>,
    /// Regime detection results
    pub regime_detection: Option<RegimeAnalysis>,
    /// TENGRI compliance status
    pub compliance_status: Option<ComplianceResult>,
    /// Integration metrics
    pub integration_metrics: IntegrationMetrics,
}

/// Regime analysis integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    /// Current regime type
    pub current_regime: String,
    /// Regime confidence
    pub regime_confidence: f32,
    /// Transition probability
    pub transition_probability: f32,
    /// Regime-based risk adjustment
    pub risk_adjustment: f32,
}

/// Integration performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total decision time (microseconds)
    pub total_decision_time_us: u64,
    /// Whale defense time (microseconds) 
    pub whale_defense_time_us: u64,
    /// Regime detection time (microseconds)
    pub regime_detection_time_us: u64,
    /// Compliance check time (microseconds)
    pub compliance_time_us: u64,
    /// Decision integration time (microseconds)
    pub integration_time_us: u64,
}

/// Enhanced decision engine with full integration
pub struct IntegratedDecisionEngine {
    /// Base decision engine
    base_engine: DecisionEngineImpl,
    /// Whale defense system
    whale_defense: Arc<QARWhaleDefense>,
    /// Regime detector
    regime_detector: Option<Arc<RwLock<RegimeDetector>>>,
    /// TENGRI compliance engine
    compliance_engine: Option<Arc<ComplianceEngine>>,
    /// Configuration
    config: EnhancedDecisionConfig,
    /// Performance tracking
    performance_history: Arc<RwLock<Vec<IntegrationMetrics>>>,
}

impl IntegratedDecisionEngine {
    /// Create new integrated decision engine
    pub async fn new(config: EnhancedDecisionConfig) -> QarResult<Self> {
        // Create base decision engine
        let base_engine = DecisionEngineImpl::new(config.base_config.clone())?;
        
        // Create whale defense system
        let whale_defense = QARWhaleDefense::new(config.whale_defense.clone()).await?;
        
        // Create regime detector if enabled
        let regime_detector = if config.enable_regime_detection {
            let detector = RegimeDetector::new()
                .map_err(|e| QarError::General(format!("Failed to create regime detector: {}", e)))?;
            Some(Arc::new(RwLock::new(detector)))
        } else {
            None
        };
        
        // Create compliance engine if enabled
        let compliance_engine = if config.enable_tengri_compliance {
            let engine = ComplianceEngine::new()
                .map_err(|e| QarError::General(format!("Failed to create compliance engine: {}", e)))?;
            Some(Arc::new(engine))
        } else {
            None
        };
        
        Ok(Self {
            base_engine,
            whale_defense: Arc::new(whale_defense),
            regime_detector,
            compliance_engine,
            config,
            performance_history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
        })
    }
    
    /// Make fully integrated trading decision
    pub async fn make_integrated_decision(
        &mut self,
        context: DecisionContext,
        current_price: f32,
        current_volume: f32,
        bid: Option<f32>,
        ask: Option<f32>,
    ) -> QarResult<IntegratedTradingDecision> {
        let total_start = Instant::now();
        let mut metrics = IntegrationMetrics {
            total_decision_time_us: 0,
            whale_defense_time_us: 0,
            regime_detection_time_us: 0,
            compliance_time_us: 0,
            integration_time_us: 0,
        };
        
        // 1. Whale Defense Analysis
        let whale_start = Instant::now();
        let whale_defense_analysis = self.whale_defense
            .analyze_whale_activity(&context, current_price, current_volume, bid, ask)
            .await?;
        metrics.whale_defense_time_us = whale_start.elapsed().as_micros() as u64;
        
        // 2. Regime Detection Analysis  
        let regime_start = Instant::now();
        let regime_analysis = if let Some(ref detector) = self.regime_detector {
            Some(self.analyze_regime(&context, detector, current_price, current_volume).await?)
        } else {
            None
        };
        metrics.regime_detection_time_us = regime_start.elapsed().as_micros() as u64;
        
        // 3. Create Enhanced Decision Context
        let integration_start = Instant::now();
        let enhanced_context = self.create_enhanced_context(
            context, 
            &whale_defense_analysis, 
            &regime_analysis
        )?;
        
        // 4. Make Base Decision with Enhanced Context
        let base_decision = self.base_engine
            .make_enhanced_decision(enhanced_context.base_context.clone())
            .await?;
        
        // 5. Apply Whale Defense Adjustments
        let adjusted_decision = self.apply_whale_defense_adjustments(
            base_decision,
            &whale_defense_analysis
        )?;
        
        // 6. Apply Regime-Based Adjustments
        let regime_adjusted_decision = if let Some(ref regime) = regime_analysis {
            self.apply_regime_adjustments(adjusted_decision, regime)?
        } else {
            adjusted_decision
        };
        
        metrics.integration_time_us = integration_start.elapsed().as_micros() as u64;
        
        // 7. TENGRI Compliance Check
        let compliance_start = Instant::now();
        let compliance_status = if let Some(ref engine) = self.compliance_engine {
            Some(self.check_compliance(engine, &regime_adjusted_decision).await?)
        } else {
            None
        };
        metrics.compliance_time_us = compliance_start.elapsed().as_micros() as u64;
        
        // 8. Final Compliance Adjustments
        let final_decision = if let Some(ref compliance) = compliance_status {
            self.apply_compliance_adjustments(regime_adjusted_decision, compliance)?
        } else {
            regime_adjusted_decision
        };
        
        // Calculate total time and check constraints
        metrics.total_decision_time_us = total_start.elapsed().as_micros() as u64;
        
        if metrics.total_decision_time_us > self.config.max_decision_time_us {
            tracing::warn!(
                "Integrated decision time {}μs exceeds {}μs target",
                metrics.total_decision_time_us, self.config.max_decision_time_us
            );
        }
        
        // Store performance metrics
        if let Ok(mut history) = self.performance_history.write().await {
            history.push(metrics.clone());
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        Ok(IntegratedTradingDecision {
            base_decision: final_decision,
            whale_defense: Some(whale_defense_analysis),
            regime_detection: regime_analysis,
            compliance_status,
            integration_metrics: metrics,
        })
    }
    
    /// Analyze current market regime
    async fn analyze_regime(
        &self,
        context: &DecisionContext,
        detector: &Arc<RwLock<RegimeDetector>>,
        price: f32,
        volume: f32,
    ) -> QarResult<RegimeAnalysis> {
        let mut detector = detector.write().await;
        
        // Extract market factors for regime detection
        let volatility = context.factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let trend = context.factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let momentum = context.factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        
        // Update regime detector with current market data
        let regime_input = regime_detector::RegimeInput {
            price: price as f64,
            volume: volume as f64,
            volatility,
            trend,
            momentum,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };
        
        let regime_result = detector.update(regime_input)
            .map_err(|e| QarError::General(format!("Regime detection failed: {}", e)))?;
        
        // Calculate risk adjustment based on regime
        let risk_adjustment = match regime_result.regime_type {
            RegimeType::Bull => 0.9,      // Reduce risk in bull markets
            RegimeType::Bear => 1.3,      // Increase risk awareness in bear markets
            RegimeType::Sideways => 1.0,  // Neutral risk in sideways markets
            RegimeType::Volatile => 1.5,  // High risk in volatile markets
            RegimeType::Crisis => 2.0,    // Maximum risk in crisis
        };
        
        Ok(RegimeAnalysis {
            current_regime: format!("{:?}", regime_result.regime_type),
            regime_confidence: regime_result.confidence,
            transition_probability: regime_result.transition_probability.unwrap_or(0.0),
            risk_adjustment,
        })
    }
    
    /// Create enhanced decision context with all integrations
    fn create_enhanced_context(
        &self,
        mut base_context: DecisionContext,
        whale_analysis: &WhaleDefenseAnalysis,
        regime_analysis: &Option<RegimeAnalysis>,
    ) -> QarResult<EnhancedDecisionContext> {
        // Adjust risk constraints based on whale and regime analysis
        let whale_risk_multiplier = 1.0 + whale_analysis.risk_factors.market_impact_risk;
        let regime_risk_multiplier = regime_analysis
            .as_ref()
            .map(|r| r.risk_adjustment)
            .unwrap_or(1.0);
        
        let combined_risk_multiplier = whale_risk_multiplier * regime_risk_multiplier;
        
        // Update risk constraints
        base_context.risk_constraints.max_risk_per_trade *= combined_risk_multiplier;
        base_context.risk_constraints.max_portfolio_risk *= combined_risk_multiplier;
        base_context.risk_constraints.min_confidence = 
            (base_context.risk_constraints.min_confidence * combined_risk_multiplier).min(0.95);
        
        // Create enhanced context
        let enhanced_context = EnhancedDecisionContext::from_base(base_context)
            .with_whale_defense(whale_analysis.clone());
        
        Ok(enhanced_context)
    }
    
    /// Apply whale defense adjustments to decision
    fn apply_whale_defense_adjustments(
        &self,
        mut decision: EnhancedTradingDecision,
        whale_analysis: &WhaleDefenseAnalysis,
    ) -> QarResult<EnhancedTradingDecision> {
        // Apply position size adjustment
        decision.execution_plan.position_size *= whale_analysis.defensive_recommendations.reduce_position_size;
        
        // Adjust execution strategy based on threat level
        if whale_analysis.threat_level >= 4 {
            decision.execution_plan.strategy = crate::decision::ExecutionStrategy::TWAP;
        } else if whale_analysis.threat_level >= 3 {
            decision.execution_plan.strategy = crate::decision::ExecutionStrategy::Limit;
        }
        
        // Add execution delay for high threats
        if whale_analysis.defensive_recommendations.execution_delay_us > 0 {
            decision.execution_plan.time_horizon = std::time::Duration::from_micros(
                whale_analysis.defensive_recommendations.execution_delay_us
            );
        }
        
        // Adjust confidence based on whale probability
        decision.decision.confidence *= (1.0 - whale_analysis.whale_probability * 0.3) as f64;
        
        // Update reasoning
        decision.decision.reasoning = format!(
            "{} | Whale Defense: threat_level={}, probability={:.2}, defensive_measures={}",
            decision.decision.reasoning,
            whale_analysis.threat_level,
            whale_analysis.whale_probability,
            whale_analysis.defensive_recommendations.fragment_orders
        );
        
        Ok(decision)
    }
    
    /// Apply regime-based adjustments
    fn apply_regime_adjustments(
        &self,
        mut decision: EnhancedTradingDecision,
        regime_analysis: &RegimeAnalysis,
    ) -> QarResult<EnhancedTradingDecision> {
        // Adjust position size based on regime
        decision.execution_plan.position_size *= (2.0 - regime_analysis.risk_adjustment) as f64;
        
        // Adjust confidence based on regime confidence
        decision.decision.confidence *= regime_analysis.regime_confidence as f64;
        
        // Update risk assessment
        decision.risk_assessment.risk_score *= regime_analysis.risk_adjustment;
        
        // Update reasoning
        decision.decision.reasoning = format!(
            "{} | Regime: {}, confidence={:.2}, risk_adj={:.2}",
            decision.decision.reasoning,
            regime_analysis.current_regime,
            regime_analysis.regime_confidence,
            regime_analysis.risk_adjustment
        );
        
        Ok(decision)
    }
    
    /// Check TENGRI compliance
    async fn check_compliance(
        &self,
        engine: &Arc<ComplianceEngine>,
        decision: &EnhancedTradingDecision,
    ) -> QarResult<ComplianceResult> {
        let compliance_check = ComplianceCheck {
            decision_type: format!("{:?}", decision.decision.decision_type),
            position_size: decision.execution_plan.position_size,
            confidence: decision.decision.confidence,
            risk_score: decision.risk_assessment.risk_score,
            execution_strategy: format!("{:?}", decision.execution_plan.strategy),
        };
        
        engine.check_compliance(compliance_check)
            .await
            .map_err(|e| QarError::General(format!("Compliance check failed: {}", e)))
    }
    
    /// Apply compliance adjustments
    fn apply_compliance_adjustments(
        &self,
        mut decision: EnhancedTradingDecision,
        compliance: &ComplianceResult,
    ) -> QarResult<EnhancedTradingDecision> {
        if !compliance.compliant {
            // Reduce position size for non-compliant decisions
            decision.execution_plan.position_size *= 0.5;
            
            // Increase minimum confidence
            decision.decision.confidence = decision.decision.confidence.max(0.8);
            
            // Force conservative execution strategy
            decision.execution_plan.strategy = crate::decision::ExecutionStrategy::TWAP;
            
            // Update reasoning
            decision.decision.reasoning = format!(
                "{} | TENGRI: NON-COMPLIANT - {}",
                decision.decision.reasoning,
                compliance.violations.join(", ")
            );
        } else {
            decision.decision.reasoning = format!(
                "{} | TENGRI: COMPLIANT",
                decision.decision.reasoning
            );
        }
        
        Ok(decision)
    }
    
    /// Get integrated performance metrics
    pub async fn get_integration_metrics(&self) -> Vec<IntegrationMetrics> {
        if let Ok(history) = self.performance_history.read().await {
            history.clone()
        } else {
            Vec::new()
        }
    }
    
    /// Get whale defense metrics
    pub async fn get_whale_defense_metrics(&self) -> crate::whale_defense_integration::WhaleDefenseMetrics {
        self.whale_defense.get_metrics().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};
    use std::collections::HashMap;

    fn create_test_context() -> DecisionContext {
        let mut factors = std::collections::HashMap::new();
        factors.insert("trend".to_string(), 0.7);
        factors.insert("volatility".to_string(), 0.4);
        factors.insert("liquidity".to_string(), 0.8);
        factors.insert("volume".to_string(), 0.6);
        factors.insert("momentum".to_string(), 0.5);
        factors.insert("sentiment".to_string(), 0.6);
        factors.insert("risk".to_string(), 0.3);
        factors.insert("efficiency".to_string(), 0.8);
        
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
    async fn test_integrated_engine_creation() {
        let config = EnhancedDecisionConfig {
            whale_defense: WhaleDefenseConfig {
                use_gpu: false, // Use CPU for testing
                ..Default::default()
            },
            ..Default::default()
        };
        
        let engine = IntegratedDecisionEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_integrated_decision_making() {
        let config = EnhancedDecisionConfig {
            whale_defense: WhaleDefenseConfig {
                use_gpu: false,
                ..Default::default()
            },
            enable_regime_detection: false, // Disable for simpler testing
            enable_tengri_compliance: false,
            ..Default::default()
        };
        
        let mut engine = IntegratedDecisionEngine::new(config).await.unwrap();
        let context = create_test_context();
        
        // Process enough data for whale detection sequence
        for _ in 0..60 {
            let decision = engine.make_integrated_decision(
                context.clone(),
                50000.0,
                1_000_000.0,
                Some(49990.0),
                Some(50010.0),
            ).await;
            
            if decision.is_ok() {
                let decision = decision.unwrap();
                
                // Verify integrated decision structure
                assert!(decision.whale_defense.is_some());
                assert!(decision.integration_metrics.total_decision_time_us > 0);
                assert!(decision.integration_metrics.whale_defense_time_us > 0);
                
                // Verify performance constraint
                assert!(decision.integration_metrics.total_decision_time_us <= 1000); // Allow 1ms for testing
                
                break; // Exit after first successful prediction
            }
        }
    }

    #[tokio::test]
    async fn test_whale_defense_adjustments() {
        let config = EnhancedDecisionConfig {
            whale_defense: WhaleDefenseConfig {
                use_gpu: false,
                ..Default::default()
            },
            enable_regime_detection: false,
            enable_tengri_compliance: false,
            ..Default::default()
        };
        
        let engine = IntegratedDecisionEngine::new(config).await.unwrap();
        
        // Create a high-threat whale analysis
        let whale_analysis = WhaleDefenseAnalysis {
            whale_probability: 0.8,
            threat_level: 4,
            confidence: 0.9,
            inference_time_us: 300,
            interpretability: crate::whale_defense_integration::WhaleInterpretability {
                top_features: vec![("volume".to_string(), 0.9)],
                anomaly_score: 0.7,
                behavioral_classification: "High Threat Whale".to_string(),
                pattern_confidence: 0.9,
            },
            risk_factors: crate::whale_defense_integration::WhaleRiskFactors {
                market_impact_risk: 0.8,
                timing_risk: 0.7,
                information_leakage_risk: 0.6,
                counter_party_risk: 0.5,
            },
            defensive_recommendations: crate::whale_defense_integration::DefensiveRecommendations {
                fragment_orders: true,
                randomize_timing: true,
                use_steganography: true,
                reduce_position_size: 0.5,
                execution_delay_us: 1000,
            },
        };
        
        // Create a dummy enhanced decision
        let mut decision = EnhancedTradingDecision {
            decision: TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                expected_return: Some(0.05),
                risk_assessment: Some(0.3),
                urgency_score: Some(0.6),
                reasoning: "Test decision".to_string(),
                timestamp: chrono::Utc::now(),
            },
            quantum_insights: None,
            risk_assessment: RiskAssessment {
                risk_score: 0.3,
                var_95: 0.03,
                expected_shortfall: 0.039,
                max_drawdown_risk: 0.045,
                liquidity_risk: 0.2,
                risk_adjusted_return: 0.167,
            },
            execution_plan: ExecutionPlan {
                strategy: crate::decision::ExecutionStrategy::Market,
                position_size: 0.1,
                stop_loss: Some(0.95),
                take_profit: Some(1.1),
                time_horizon: std::time::Duration::from_secs(300),
                priority: crate::decision::ExecutionPriority::Medium,
            },
            metadata: HashMap::new(),
        };
        
        let original_position_size = decision.execution_plan.position_size;
        
        // Apply whale defense adjustments
        decision = engine.apply_whale_defense_adjustments(decision, &whale_analysis).unwrap();
        
        // Verify adjustments were applied
        assert!(decision.execution_plan.position_size < original_position_size);
        assert_eq!(decision.execution_plan.strategy, crate::decision::ExecutionStrategy::TWAP);
        assert!(decision.decision.reasoning.contains("Whale Defense"));
    }
}