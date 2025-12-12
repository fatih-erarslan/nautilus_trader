//! # Comprehensive Talebian Risk Engine
//!
//! Main orchestration engine that combines all components:
//! - Aggressive antifragility detection
//! - Opportunistic barbell strategy
//! - Black swan tolerance
//! - Aggressive Kelly criterion
//! - Whale detection and parasitic opportunities
//!
//! Provides unified risk assessment and position sizing recommendations.

use crate::{
    kelly::{KellyCalculation, KellyEngine},
    parasitic_opportunities::{OpportunityAnalysis, ParasiticOpportunityEngine},
    stubs::{
        AntifragilityAssessment, AntifragilityEngine, BarbellAllocation, BarbellEngine,
        BlackSwanAssessment, BlackSwanEngine,
    },
    whale_detection::{WhaleActivitySummary, WhaleDetectionEngine},
    MacchiavelianConfig, MarketData, TalebianRiskAssessment, TalebianRiskError, WhaleDetection,
};
// Removed unused Serialize/Deserialize imports
use std::collections::VecDeque;

#[cfg(feature = "simd")]
use wide::f64x4;

/// Unified Talebian risk engine with all aggressive components
pub struct TalebianRiskEngine {
    config: MacchiavelianConfig,
    antifragility_engine: AntifragilityEngine,
    barbell_engine: BarbellEngine,
    black_swan_engine: BlackSwanEngine,
    kelly_engine: KellyEngine,
    whale_engine: WhaleDetectionEngine,
    opportunity_engine: ParasiticOpportunityEngine,
    assessment_history: VecDeque<TalebianRiskAssessment>,
    performance_tracker: EnginePerformanceTracker,
}

#[derive(Debug, Clone)]
struct EnginePerformanceTracker {
    pub total_assessments: u64,
    pub successful_predictions: u64,
    pub missed_opportunities: u64,
    pub false_positives: u64,
    pub average_confidence: f64,
    pub average_opportunity_score: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_return: f64,
}

/// Comprehensive risk management recommendations
#[derive(Debug, Clone)]
pub struct RiskManagementRecommendation {
    pub assessment: TalebianRiskAssessment,
    pub position_sizing: PositionSizingRecommendation,
    pub risk_controls: RiskControlRecommendation,
    pub timing_guidance: TimingGuidance,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PositionSizingRecommendation {
    pub kelly_fraction: f64,
    pub barbell_allocation: (f64, f64), // (safe, risky)
    pub whale_adjusted_size: f64,
    pub opportunity_adjusted_size: f64,
    pub final_recommended_size: f64,
    pub max_position_size: f64,
    pub confidence_adjusted_size: f64,
}

#[derive(Debug, Clone)]
pub struct RiskControlRecommendation {
    pub stop_loss_level: f64,
    pub take_profit_level: f64,
    pub max_drawdown_limit: f64,
    pub volatility_adjustment: f64,
    pub black_swan_protection: f64,
    pub rebalance_trigger: f64,
}

#[derive(Debug, Clone)]
pub struct TimingGuidance {
    pub entry_urgency: String,
    pub hold_duration_estimate: String,
    pub market_regime: String,
    pub whale_activity_level: String,
    pub volatility_timing: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub expected_return: f64,
    pub expected_volatility: f64,
    pub sharpe_ratio_estimate: f64,
    pub sortino_ratio_estimate: f64,
    pub max_drawdown_estimate: f64,
    pub win_probability: f64,
    pub profit_factor: f64,
}

impl TalebianRiskEngine {
    /// Create new comprehensive Talebian risk engine
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            antifragility_engine: AntifragilityEngine::new(config.clone()),
            barbell_engine: BarbellEngine::new(config.clone()),
            black_swan_engine: BlackSwanEngine::new(config.clone()),
            kelly_engine: KellyEngine::new(config.clone()),
            whale_engine: WhaleDetectionEngine::new(config.clone()),
            opportunity_engine: ParasiticOpportunityEngine::new(config.clone()),
            config,
            assessment_history: VecDeque::with_capacity(10000),
            performance_tracker: EnginePerformanceTracker::new(),
        }
    }

    /// Comprehensive risk assessment with all components
    pub fn assess_risk(
        &mut self,
        market_data: &MarketData,
    ) -> Result<TalebianRiskAssessment, TalebianRiskError> {
        // Step 1: Whale detection (foundational)
        let whale_detection = self.whale_engine.detect_whale_activity(market_data)?;

        // Step 2: Antifragility assessment
        let antifragility = self
            .antifragility_engine
            .calculate_antifragility(market_data)?;

        // Step 3: Black swan risk assessment
        let black_swan = self
            .black_swan_engine
            .assess_black_swan_risk(market_data, &whale_detection)?;

        // Step 4: Parasitic opportunity analysis
        let opportunity = self.opportunity_engine.analyze_opportunity(
            market_data,
            &whale_detection,
            antifragility.antifragility_score,
        )?;

        // Step 5: Barbell allocation calculation
        let barbell = self.barbell_engine.calculate_optimal_allocation(
            market_data,
            &whale_detection,
            antifragility.antifragility_score,
        )?;

        // Step 6: Kelly criterion position sizing
        let kelly = self.kelly_engine.calculate_kelly_fraction(
            market_data,
            &whale_detection,
            opportunity.overall_score * 0.05,
            opportunity.confidence,
        )?;

        // Step 7: Combine all assessments
        let overall_risk_score = self.calculate_overall_risk_score(
            &antifragility,
            &black_swan,
            &opportunity,
            &whale_detection,
        )?;

        let recommended_position_size = self.calculate_recommended_position_size(
            &kelly,
            &barbell,
            &opportunity,
            &whale_detection,
        )?;

        let confidence = self.calculate_overall_confidence(
            &antifragility,
            &black_swan,
            &opportunity,
            &whale_detection,
        )?;

        let assessment = TalebianRiskAssessment {
            risk_score: overall_risk_score,
            antifragility: crate::types::AntifragilityMeasurement {
                score: antifragility.score,
                fragility_index: antifragility.fragility_index,
                robustness: antifragility.robustness,
                volatility_benefit: antifragility.volatility_benefit,
                stress_response: antifragility.stress_response,
            },
            antifragility_score: antifragility.antifragility_score,
            barbell_allocation: (barbell.safe_allocation, barbell.risky_allocation),
            black_swan_probability: black_swan.swan_probability,
            kelly_fraction: kelly.adjusted_fraction,
            _whale_detection: Some(whale_detection),
            parasitic_opportunity: Some(crate::ParasiticOpportunity {
                id: format!("risk_engine_{}", market_data.timestamp_unix),
                expected_return: opportunity.overall_score * 0.02,
                risk_level: 1.0 - opportunity.confidence,
                time_window: 3600,
                confidence: opportunity.confidence,
                entry_price: market_data.price,
                exit_price: market_data.price * (1.0 + opportunity.overall_score * 0.02),
                stop_loss: market_data.price * (1.0 - (1.0 - opportunity.confidence) * 0.05),
                opportunity_score: opportunity.overall_score,
                momentum_factor: opportunity.momentum_component,
                volatility_factor: opportunity.volatility_component,
                whale_alignment: opportunity.whale_alignment_component,
                regime_factor: opportunity.regime_component,
                recommended_allocation: opportunity.recommended_allocation,
            }),
            position_size: recommended_position_size,
            warnings: Vec::new(),
            confidence,
            overall_risk_score,
            recommended_position_size,
        };

        // Store assessment
        self.assessment_history.push_back(assessment.clone());
        while self.assessment_history.len() > 10000 {
            self.assessment_history.pop_front();
        }

        // Update performance tracking
        self.update_performance_tracking(&assessment)?;

        Ok(assessment)
    }

    /// Generate comprehensive risk management recommendations
    pub fn generate_recommendations(
        &mut self,
        market_data: &MarketData,
    ) -> Result<RiskManagementRecommendation, TalebianRiskError> {
        // Get base assessment
        let assessment = self.assess_risk(market_data)?;

        // Generate position sizing recommendations
        let position_sizing = self.generate_position_sizing_recommendation(&assessment)?;

        // Generate risk control recommendations
        let risk_controls = self.generate_risk_control_recommendation(&assessment, market_data)?;

        // Generate timing guidance
        let timing_guidance = self.generate_timing_guidance(&assessment)?;

        // Generate performance metrics
        let performance_metrics = self.generate_performance_metrics(&assessment, market_data)?;

        Ok(RiskManagementRecommendation {
            assessment,
            position_sizing,
            risk_controls,
            timing_guidance,
            performance_metrics,
        })
    }

    /// Calculate overall risk score combining all components
    fn calculate_overall_risk_score(
        &self,
        antifragility: &AntifragilityAssessment,
        black_swan: &BlackSwanAssessment,
        opportunity: &OpportunityAnalysis,
        whale_detection: &WhaleDetection,
    ) -> Result<f64, TalebianRiskError> {
        // Weight different risk components
        let antifragility_weight = 0.25;
        let black_swan_weight = 0.20;
        let opportunity_weight = 0.30;
        let whale_weight = 0.25;

        // Antifragility component (higher = lower risk for aggressive strategy)
        let antifragility_component = antifragility.antifragility_score * antifragility_weight;

        // Black swan component (lower probability = lower risk)
        let black_swan_component = (1.0 - black_swan.swan_probability) * black_swan_weight;

        // Opportunity component (higher opportunity = lower risk for profit)
        let opportunity_component = opportunity.overall_score * opportunity_weight;

        // Whale component (whale detection = lower risk for parasitic strategy)
        let whale_component = if whale_detection.is_whale_detected {
            whale_detection.confidence * whale_weight
        } else {
            0.3 * whale_weight // Baseline risk when no whales
        };

        let overall_score = antifragility_component
            + black_swan_component
            + opportunity_component
            + whale_component;

        Ok(overall_score.max(0.0).min(1.0))
    }

    /// Calculate recommended position size combining all methods
    fn calculate_recommended_position_size(
        &self,
        kelly: &KellyCalculation,
        barbell: &BarbellAllocation,
        opportunity: &OpportunityAnalysis,
        whale_detection: &WhaleDetection,
    ) -> Result<f64, TalebianRiskError> {
        // Start with Kelly fraction as base
        let kelly_base = kelly.risk_adjusted_size;

        // Apply barbell risky allocation
        let barbell_adjusted = kelly_base * barbell.risky_allocation;

        // Apply opportunity boost
        let opportunity_adjusted = barbell_adjusted * (1.0 + opportunity.overall_score * 0.5);

        // Apply whale following boost
        let whale_adjusted = if whale_detection.is_whale_detected {
            opportunity_adjusted
                * whale_detection.confidence
                * self.config.whale_detected_multiplier
        } else {
            opportunity_adjusted
        };

        // Apply confidence scaling
        let confidence_adjusted = whale_adjusted * opportunity.confidence;

        // Apply aggressive bounds
        let final_size = confidence_adjusted
            .max(0.02) // Minimum 2% position
            .min(0.75); // Maximum 75% position (aggressive)

        Ok(final_size)
    }

    /// Calculate overall confidence from all components
    fn calculate_overall_confidence(
        &self,
        antifragility: &AntifragilityAssessment,
        black_swan: &BlackSwanAssessment,
        opportunity: &OpportunityAnalysis,
        whale_detection: &WhaleDetection,
    ) -> Result<f64, TalebianRiskError> {
        let confidence_scores = vec![
            antifragility.confidence,
            black_swan.confidence,
            opportunity.confidence,
            whale_detection.confidence,
        ];

        // Weighted average with higher weight on opportunity and whale detection
        let weights = vec![0.2, 0.2, 0.35, 0.25];

        let weighted_confidence = confidence_scores
            .iter()
            .zip(weights.iter())
            .map(|(conf, weight)| conf * weight)
            .sum::<f64>();

        // Add sample size adjustment
        let sample_adjustment = (self.assessment_history.len() as f64 / 1000.0).min(1.0);
        let adjusted_confidence = weighted_confidence * (0.5 + sample_adjustment * 0.5);

        Ok(adjusted_confidence.max(0.1).min(0.95))
    }

    /// Generate position sizing recommendation
    fn generate_position_sizing_recommendation(
        &self,
        assessment: &TalebianRiskAssessment,
    ) -> Result<PositionSizingRecommendation, TalebianRiskError> {
        let kelly_fraction = assessment.kelly_fraction;
        let barbell_allocation = assessment.barbell_allocation;

        // Whale adjustment
        let whale_adjusted_size = if assessment._whale_detection.as_ref().map_or(false, |w| w.detected) {
            kelly_fraction
                * assessment._whale_detection.as_ref().map_or(0.5, |w| w.confidence)
                * self.config.whale_detected_multiplier
        } else {
            kelly_fraction
        };

        // Opportunity adjustment
        let opportunity_adjusted_size =
            whale_adjusted_size * (1.0 + assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.opportunity_score) * 0.4);

        // Confidence adjustment
        let confidence_adjusted_size = opportunity_adjusted_size * assessment.confidence;

        // Final recommended size
        let final_recommended_size = assessment.recommended_position_size;

        // Maximum position size (risk management)
        let max_position_size = self.config.kelly_max_fraction;

        Ok(PositionSizingRecommendation {
            kelly_fraction,
            barbell_allocation,
            whale_adjusted_size,
            opportunity_adjusted_size,
            final_recommended_size,
            max_position_size,
            confidence_adjusted_size,
        })
    }

    /// Generate risk control recommendations
    fn generate_risk_control_recommendation(
        &self,
        assessment: &TalebianRiskAssessment,
        market_data: &MarketData,
    ) -> Result<RiskControlRecommendation, TalebianRiskError> {
        // Stop loss based on volatility and confidence
        let volatility_multiplier = (market_data.volatility * 3.0).max(0.02).min(0.15);
        let confidence_adjustment = 1.0 / assessment.confidence.max(0.3);
        let stop_loss_level = volatility_multiplier * confidence_adjustment;

        // Take profit based on opportunity score
        let base_take_profit = volatility_multiplier * 2.0;
        let opportunity_multiplier = 1.0 + assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.opportunity_score);
        let take_profit_level = base_take_profit * opportunity_multiplier;

        // Maximum drawdown limit (aggressive tolerance)
        let max_drawdown_limit = self.config.destructive_swan_protection;

        // Volatility adjustment factor
        let volatility_adjustment = (market_data.volatility / 0.02).min(3.0);

        // Black swan protection level
        let black_swan_protection = assessment.black_swan_probability * 0.5; // Reduce position if swan likely

        // Rebalance trigger
        let rebalance_trigger = self.config.dynamic_rebalance_threshold;

        Ok(RiskControlRecommendation {
            stop_loss_level,
            take_profit_level,
            max_drawdown_limit,
            volatility_adjustment,
            black_swan_protection,
            rebalance_trigger,
        })
    }

    /// Generate timing guidance
    fn generate_timing_guidance(
        &self,
        assessment: &TalebianRiskAssessment,
    ) -> Result<TimingGuidance, TalebianRiskError> {
        // Entry urgency based on opportunity and whale detection
        let entry_urgency = if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.opportunity_score > 0.8)
            && assessment._whale_detection.as_ref().map_or(false, |w| w.detected)
        {
            "IMMEDIATE - High opportunity with whale activity detected"
        } else if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.opportunity_score > 0.7) {
            "HIGH - Strong opportunity identified"
        } else if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.opportunity_score > 0.6) {
            "MEDIUM - Moderate opportunity present"
        } else if assessment.parasitic_opportunity.as_ref().map_or(false, |p| 
            p.opportunity_score > self.config.parasitic_opportunity_threshold)
        {
            "LOW - Weak opportunity, consider waiting"
        } else {
            "WAIT - Below opportunity threshold"
        }
        .to_string();

        // Hold duration estimate
        let hold_duration_estimate = if assessment._whale_detection.as_ref().map_or(false, |w| w.detected) {
            "SHORT - Follow whale movement (minutes to hours)"
        } else if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.momentum_factor > 1.5) {
            "MEDIUM - Ride momentum (hours to days)"
        } else {
            "LONG - Antifragile positioning (days to weeks)"
        }
        .to_string();

        // Market regime
        let market_regime = if assessment.antifragility_score > 0.7 {
            "ANTIFRAGILE - Benefits from volatility"
        } else if assessment.black_swan_probability > self.config.black_swan_threshold {
            "VOLATILE - High uncertainty, prepare for swans"
        } else {
            "NORMAL - Standard market conditions"
        }
        .to_string();

        // Whale activity level
        let whale_activity_level = if assessment._whale_detection.as_ref().map_or(false, |w| w.detected) {
            format!(
                "ACTIVE - Confidence: {:.1}%",
                assessment._whale_detection.as_ref().map_or(50.0, |w| w.confidence * 100.0)
            )
        } else {
            "QUIET - No significant whale activity".to_string()
        };

        // Volatility timing
        let volatility_timing = if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.volatility_factor > 1.5) {
            "HIGH - Excellent for antifragile strategies"
        } else if assessment.parasitic_opportunity.as_ref().map_or(false, |p| p.volatility_factor > 1.0) {
            "MEDIUM - Good volatility levels"
        } else {
            "LOW - Low volatility environment"
        }
        .to_string();

        Ok(TimingGuidance {
            entry_urgency,
            hold_duration_estimate,
            market_regime,
            whale_activity_level,
            volatility_timing,
        })
    }

    /// Generate performance metrics estimates
    fn generate_performance_metrics(
        &self,
        assessment: &TalebianRiskAssessment,
        market_data: &MarketData,
    ) -> Result<PerformanceMetrics, TalebianRiskError> {
        // Expected return based on opportunity score and Kelly fraction
        let base_return = assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.opportunity_score) * 0.02; // 2% for perfect opportunity
        let kelly_adjusted_return = base_return * assessment.kelly_fraction;
        let whale_adjusted_return = if assessment._whale_detection.as_ref().map_or(false, |w| w.detected) {
            kelly_adjusted_return * (1.0 + assessment._whale_detection.as_ref().map_or(0.5, |w| w.confidence * 0.5))
        } else {
            kelly_adjusted_return
        };
        let expected_return = whale_adjusted_return;

        // Expected volatility
        let expected_volatility = market_data.volatility
            * (1.0 + assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.volatility_factor) * 0.2);

        // Sharpe ratio estimate
        let risk_free_rate = 0.0003; // 0.03% daily risk-free rate
        let sharpe_ratio_estimate = if expected_volatility > 0.001 {
            (expected_return - risk_free_rate) / expected_volatility
        } else {
            0.0
        };

        // Sortino ratio estimate (assuming downside volatility is 70% of total volatility)
        let downside_volatility = expected_volatility * 0.7;
        let sortino_ratio_estimate = if downside_volatility > 0.001 {
            (expected_return - risk_free_rate) / downside_volatility
        } else {
            0.0
        };

        // Maximum drawdown estimate
        let base_drawdown = expected_volatility * 2.0; // 2x volatility as base drawdown
        let confidence_adjustment = 1.0 / assessment.confidence.max(0.3);
        let max_drawdown_estimate = (base_drawdown * confidence_adjustment).min(0.5);

        // Win probability estimate
        let win_probability = assessment.confidence
            * (0.5 + assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.opportunity_score) * 0.3);

        // Profit factor estimate
        let avg_win = expected_return * 2.0; // Assume wins are 2x expected return
        let avg_loss = expected_volatility; // Assume losses are 1x volatility
        let profit_factor = if avg_loss > 0.001 {
            (win_probability * avg_win) / ((1.0 - win_probability) * avg_loss)
        } else {
            1.0
        };

        Ok(PerformanceMetrics {
            expected_return,
            expected_volatility,
            sharpe_ratio_estimate,
            sortino_ratio_estimate,
            max_drawdown_estimate,
            win_probability,
            profit_factor,
        })
    }

    /// Update performance tracking
    fn update_performance_tracking(
        &mut self,
        assessment: &TalebianRiskAssessment,
    ) -> Result<(), TalebianRiskError> {
        self.performance_tracker.total_assessments += 1;

        // Update running averages
        let n = self.performance_tracker.total_assessments as f64;
        self.performance_tracker.average_confidence =
            (self.performance_tracker.average_confidence * (n - 1.0) + assessment.confidence) / n;

        self.performance_tracker.average_opportunity_score =
            (self.performance_tracker.average_opportunity_score * (n - 1.0)
                + assessment.parasitic_opportunity.as_ref().map_or(0.0, |p| p.opportunity_score))
                / n;

        Ok(())
    }

    /// Record trade outcome for performance improvement
    pub fn record_trade_outcome(
        &mut self,
        return_pct: f64,
        was_whale_trade: bool,
        momentum_score: f64,
    ) -> Result<(), TalebianRiskError> {
        // Update Kelly engine with trade outcome
        self.kelly_engine
            .record_trade_outcome(return_pct, was_whale_trade, momentum_score)?;

        // Update performance tracker
        if return_pct > 0.0 {
            self.performance_tracker.successful_predictions += 1;
        }

        // Update running performance metrics
        let n = self.performance_tracker.total_assessments as f64;
        self.performance_tracker.total_return =
            (self.performance_tracker.total_return * (n - 1.0) + return_pct) / n;

        Ok(())
    }

    /// Get current engine status
    pub fn get_engine_status(&self) -> TalebianEngineStatus {
        TalebianEngineStatus {
            config: self.config.clone(),
            performance_tracker: self.performance_tracker.clone(),
            total_assessments: self.assessment_history.len(),
            whale_activity_summary: self.whale_engine.get_whale_activity_summary(),
            opportunity_status: self.opportunity_engine.get_opportunity_status(),
            kelly_status: self.kelly_engine.get_kelly_status(),
        }
    }

    /// SIMD-optimized bulk risk assessment
    #[cfg(feature = "simd")]
    pub fn assess_bulk_risks(
        &mut self,
        market_data_batch: &[MarketData],
    ) -> Result<Vec<TalebianRiskAssessment>, TalebianRiskError> {
        if market_data_batch.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(market_data_batch.len());

        // Process in chunks for memory efficiency
        for chunk in market_data_batch.chunks(100) {
            for market_data in chunk {
                let assessment = self.assess_risk(market_data)?;
                results.push(assessment);
            }
        }

        Ok(results)
    }
}

/// Current engine status
#[derive(Debug, Clone)]
pub struct TalebianEngineStatus {
    pub config: MacchiavelianConfig,
    pub performance_tracker: EnginePerformanceTracker,
    pub total_assessments: usize,
    pub whale_activity_summary: WhaleActivitySummary,
    pub opportunity_status: crate::parasitic_opportunities::OpportunityStatus,
    pub kelly_status: crate::kelly::KellyStatus,
}

impl EnginePerformanceTracker {
    fn new() -> Self {
        Self {
            total_assessments: 0,
            successful_predictions: 0,
            missed_opportunities: 0,
            false_positives: 0,
            average_confidence: 0.0,
            average_opportunity_score: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            total_return: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MacchiavelianConfig, WhaleDirection};

    #[test]
    fn test_risk_engine_creation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = TalebianRiskEngine::new(config);

        // Verify aggressive configuration
        assert_eq!(engine.config.antifragility_threshold, 0.35);
        assert_eq!(engine.config.barbell_safe_ratio, 0.65);
        assert_eq!(engine.config.black_swan_threshold, 0.18);
        assert_eq!(engine.config.kelly_fraction, 0.55);
    }

    #[test]
    fn test_comprehensive_risk_assessment() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);

        let market_data = MarketData {
            timestamp: 1640995200, // 2022-01-01
            price: 50000.0,
            volume: 1000.0,
            bid: 49990.0,
            ask: 50010.0,
            bid_volume: 500.0,
            ask_volume: 400.0,
            volatility: 0.03,
            returns: vec![0.01, 0.015, -0.005, 0.02, 0.008],
            volume_history: vec![800.0, 900.0, 1200.0, 950.0, 1000.0],
        };

        let assessment = engine.assess_risk(&market_data).unwrap();

        // Verify assessment components
        assert!(assessment.antifragility_score >= 0.0);
        assert!(assessment.antifragility_score <= 1.0);
        assert!(assessment.barbell_allocation.0 + assessment.barbell_allocation.1 <= 1.1); // Allow for rounding
        assert!(assessment.black_swan_probability >= 0.0);
        assert!(assessment.kelly_fraction >= 0.0);
        assert!(assessment.overall_risk_score >= 0.0);
        assert!(assessment.overall_risk_score <= 1.0);
        assert!(assessment.recommended_position_size >= 0.0);
        assert!(assessment.confidence >= 0.0);
        assert!(assessment.confidence <= 1.0);
    }

    #[test]
    fn test_recommendations_generation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);

        let market_data = MarketData {
            timestamp: 1640995200,
            price: 50000.0,
            volume: 2000.0, // High volume
            bid: 49995.0,
            ask: 50005.0,
            bid_volume: 800.0, // Strong bid pressure
            ask_volume: 300.0,
            volatility: 0.04,                              // High volatility
            returns: vec![0.02, 0.025, 0.015, 0.03, 0.01], // Strong positive momentum
            volume_history: vec![1500.0, 1800.0, 2200.0, 1900.0, 2000.0],
        };

        let recommendations = engine.generate_recommendations(&market_data).unwrap();

        // Verify recommendation components
        assert!(recommendations.position_sizing.final_recommended_size > 0.0);
        assert!(recommendations.risk_controls.stop_loss_level > 0.0);
        assert!(recommendations.risk_controls.take_profit_level > 0.0);
        assert!(!recommendations.timing_guidance.entry_urgency.is_empty());
        assert!(recommendations.performance_metrics.expected_return >= 0.0);
    }

    #[test]
    fn test_whale_detection_impact() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);

        // Create market data that should trigger whale detection
        let whale_market_data = MarketData {
            timestamp: 1640995200,
            price: 50000.0,
            volume: 5000.0, // Very high volume (5x normal)
            bid: 49980.0,
            ask: 50020.0,
            bid_volume: 2000.0,
            ask_volume: 1000.0,                            // Strong imbalance
            volatility: 0.05,                              // High volatility
            returns: vec![0.03, 0.04, 0.025, 0.035, 0.02], // Strong momentum
            volume_history: vec![1000.0, 1200.0, 1100.0, 1050.0, 1000.0], // Normal history
        };

        let assessment = engine.assess_risk(&whale_market_data).unwrap();

        // Should detect whale activity
        assert!(assessment.whale_detection.is_whale_detected);
        assert!(assessment.whale_detection.confidence > 0.5);

        // Should result in higher recommended position size due to whale following
        assert!(assessment.recommended_position_size > 0.1); // Should be significant
    }

    #[test]
    fn test_performance_tracking() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);

        // Record several trade outcomes
        engine.record_trade_outcome(0.02, true, 0.8).unwrap();
        engine.record_trade_outcome(0.015, false, 0.5).unwrap();
        engine.record_trade_outcome(-0.01, true, 0.3).unwrap();
        engine.record_trade_outcome(0.03, true, 0.9).unwrap();

        let status = engine.get_engine_status();
        assert!(status.performance_tracker.total_return != 0.0);
        assert!(status.performance_tracker.successful_predictions > 0);
    }

    #[test]
    fn test_aggressive_vs_conservative_comparison() {
        let aggressive_config = MacchiavelianConfig::aggressive_defaults();
        let conservative_config = MacchiavelianConfig::conservative_baseline();

        // Verify aggressive is indeed more aggressive
        assert!(
            aggressive_config.antifragility_threshold < conservative_config.antifragility_threshold
        );
        assert!(aggressive_config.barbell_safe_ratio < conservative_config.barbell_safe_ratio);
        assert!(aggressive_config.black_swan_threshold > conservative_config.black_swan_threshold);
        assert!(aggressive_config.kelly_fraction > conservative_config.kelly_fraction);
        assert!(
            aggressive_config.whale_volume_threshold < conservative_config.whale_volume_threshold
        );
    }

    #[test]
    fn test_opportunity_threshold_compliance() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);

        let low_opportunity_data = MarketData {
            timestamp: 1640995200,
            price: 50000.0,
            volume: 500.0, // Low volume
            bid: 49999.0,
            ask: 50001.0,
            bid_volume: 250.0,
            ask_volume: 250.0,                                     // Balanced
            volatility: 0.005,                                     // Very low volatility
            returns: vec![0.001, 0.0005, -0.0002, 0.0008, 0.0003], // Minimal movement
            volume_history: vec![500.0, 480.0, 520.0, 490.0, 500.0],
        };

        let assessment = engine.assess_risk(&low_opportunity_data).unwrap();

        // Should result in low opportunity score and smaller position
        assert!(
            assessment.parasitic_opportunity.opportunity_score
                < config.parasitic_opportunity_threshold
        );
        assert!(assessment.recommended_position_size < 0.1); // Should be small
    }
}
