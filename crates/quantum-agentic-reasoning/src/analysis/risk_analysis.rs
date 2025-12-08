//! Risk Analysis Module
//!
//! Advanced risk analysis and assessment for quantum trading decisions.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskResult {
    /// Overall risk score (0.0 to 1.0)
    pub score: f64,
    /// Confidence in risk assessment
    pub confidence: f64,
    /// Risk components
    pub risk_components: RiskComponents,
    /// Risk metrics
    pub risk_metrics: RiskMetrics,
    /// Risk recommendations
    pub recommendations: Vec<RiskRecommendation>,
}

/// Risk components breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskComponents {
    /// Market risk
    pub market_risk: f64,
    /// Liquidity risk
    pub liquidity_risk: f64,
    /// Volatility risk
    pub volatility_risk: f64,
    /// Concentration risk
    pub concentration_risk: f64,
    /// Operational risk
    pub operational_risk: f64,
    /// Systematic risk
    pub systematic_risk: f64,
    /// Idiosyncratic risk
    pub idiosyncratic_risk: f64,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Value at Risk (99%)
    pub var_99: f64,
    /// Expected Shortfall
    pub expected_shortfall: f64,
    /// Maximum Drawdown
    pub max_drawdown: f64,
    /// Sharpe Ratio
    pub sharpe_ratio: f64,
    /// Beta
    pub beta: f64,
    /// Tracking Error
    pub tracking_error: f64,
}

/// Risk recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: f64,
}

/// Recommendation type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    PositionSizing,
    Diversification,
    HedgeStrategy,
    StopLoss,
    RiskLimit,
    TimeoutStrategy,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk analyzer
pub struct RiskAnalyzer {
    config: super::AnalysisConfig,
    risk_params: RiskParameters,
    return_history: Vec<f64>,
    drawdown_history: Vec<f64>,
    history: Vec<RiskResult>,
}

/// Risk analysis parameters
#[derive(Debug, Clone)]
pub struct RiskParameters {
    /// VaR confidence level
    pub var_confidence: f64,
    /// Risk-free rate for Sharpe ratio
    pub risk_free_rate: f64,
    /// Maximum acceptable risk level
    pub max_risk_threshold: f64,
    /// Lookback period for risk calculations
    pub lookback_period: usize,
    /// Stress test scenarios
    pub stress_scenarios: Vec<StressScenario>,
}

/// Stress test scenario
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Scenario name
    pub name: String,
    /// Market shock magnitude
    pub market_shock: f64,
    /// Volatility multiplier
    pub volatility_multiplier: f64,
    /// Liquidity impact
    pub liquidity_impact: f64,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            var_confidence: 0.95,
            risk_free_rate: 0.02,
            max_risk_threshold: 0.8,
            lookback_period: 50,
            stress_scenarios: vec![
                StressScenario {
                    name: "Market Crash".to_string(),
                    market_shock: -0.2,
                    volatility_multiplier: 3.0,
                    liquidity_impact: 0.5,
                },
                StressScenario {
                    name: "Liquidity Crisis".to_string(),
                    market_shock: -0.1,
                    volatility_multiplier: 2.0,
                    liquidity_impact: 0.8,
                },
                StressScenario {
                    name: "Flash Crash".to_string(),
                    market_shock: -0.15,
                    volatility_multiplier: 5.0,
                    liquidity_impact: 0.3,
                },
            ],
        }
    }
}

impl RiskAnalyzer {
    /// Create a new risk analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            risk_params: RiskParameters::default(),
            return_history: Vec::new(),
            drawdown_history: Vec::new(),
            history: Vec::new(),
        })
    }

    /// Analyze risk from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<RiskResult> {
        // Extract risk-relevant data
        let risk_data = self.extract_risk_data(factors)?;
        
        // Update return and drawdown history
        self.update_risk_history(&risk_data);

        // Calculate risk components
        let risk_components = self.calculate_risk_components(&risk_data)?;
        
        // Calculate risk metrics
        let risk_metrics = self.calculate_risk_metrics(&risk_data)?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&risk_components, &risk_metrics)?;
        
        // Calculate overall score and confidence
        let score = self.calculate_overall_risk_score(&risk_components);
        let confidence = self.calculate_risk_confidence(&risk_components, &risk_metrics);

        let result = RiskResult {
            score,
            confidence,
            risk_components,
            risk_metrics,
            recommendations,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract risk-relevant data from factors
    fn extract_risk_data(&self, factors: &FactorMap) -> QarResult<RiskData> {
        Ok(RiskData {
            volatility: factors.get_factor(&StandardFactors::Volatility)?,
            liquidity: factors.get_factor(&StandardFactors::Liquidity)?,
            trend: factors.get_factor(&StandardFactors::Trend)?,
            momentum: factors.get_factor(&StandardFactors::Momentum)?,
            volume: factors.get_factor(&StandardFactors::Volume)?,
            sentiment: factors.get_factor(&StandardFactors::Sentiment)?,
            efficiency: factors.get_factor(&StandardFactors::Efficiency)?,
            risk_factor: factors.get_factor(&StandardFactors::Risk)?,
        })
    }

    /// Update risk history
    fn update_risk_history(&mut self, risk_data: &RiskData) {
        // Calculate return from trend and momentum
        let estimated_return = risk_data.trend * 0.1 + risk_data.momentum * 0.05;
        self.return_history.push(estimated_return);
        
        // Calculate estimated drawdown from volatility and risk
        let estimated_drawdown = -(risk_data.volatility * 0.2 + risk_data.risk_factor * 0.15);
        self.drawdown_history.push(estimated_drawdown);

        // Maintain history size
        let max_history = self.risk_params.lookback_period * 2;
        if self.return_history.len() > max_history {
            self.return_history.remove(0);
        }
        if self.drawdown_history.len() > max_history {
            self.drawdown_history.remove(0);
        }
    }

    /// Calculate risk components
    fn calculate_risk_components(&self, risk_data: &RiskData) -> QarResult<RiskComponents> {
        Ok(RiskComponents {
            market_risk: self.calculate_market_risk(risk_data),
            liquidity_risk: self.calculate_liquidity_risk(risk_data),
            volatility_risk: self.calculate_volatility_risk(risk_data),
            concentration_risk: self.calculate_concentration_risk(risk_data),
            operational_risk: self.calculate_operational_risk(risk_data),
            systematic_risk: self.calculate_systematic_risk(risk_data),
            idiosyncratic_risk: self.calculate_idiosyncratic_risk(risk_data),
        })
    }

    /// Calculate market risk
    fn calculate_market_risk(&self, risk_data: &RiskData) -> f64 {
        // Market risk based on volatility and trend instability
        let trend_instability = 1.0 - risk_data.efficiency;
        let base_market_risk = risk_data.volatility * 0.7 + trend_instability * 0.3;
        
        // Adjust for sentiment (poor sentiment increases market risk)
        let sentiment_adjustment = (1.0 - risk_data.sentiment) * 0.2;
        
        (base_market_risk + sentiment_adjustment).min(1.0).max(0.0)
    }

    /// Calculate liquidity risk
    fn calculate_liquidity_risk(&self, risk_data: &RiskData) -> f64 {
        // Liquidity risk inversely related to liquidity factor
        let base_liquidity_risk = 1.0 - risk_data.liquidity;
        
        // Adjust for volume (low volume increases liquidity risk)
        let volume_adjustment = (1.0 - risk_data.volume) * 0.3;
        
        (base_liquidity_risk + volume_adjustment).min(1.0).max(0.0)
    }

    /// Calculate volatility risk
    fn calculate_volatility_risk(&self, risk_data: &RiskData) -> f64 {
        // Direct mapping from volatility factor
        let base_vol_risk = risk_data.volatility;
        
        // Amplify if trend is uncertain
        let trend_uncertainty = if risk_data.trend > 0.3 && risk_data.trend < 0.7 { 0.2 } else { 0.0 };
        
        (base_vol_risk + trend_uncertainty).min(1.0).max(0.0)
    }

    /// Calculate concentration risk
    fn calculate_concentration_risk(&self, risk_data: &RiskData) -> f64 {
        // Simplified concentration risk based on efficiency
        // Low efficiency suggests concentrated positions
        let concentration_risk = 1.0 - risk_data.efficiency;
        
        // Adjust for momentum (high momentum in one direction suggests concentration)
        let momentum_concentration = if risk_data.momentum > 0.8 || risk_data.momentum < 0.2 {
            0.3
        } else {
            0.0
        };
        
        (concentration_risk + momentum_concentration).min(1.0).max(0.0)
    }

    /// Calculate operational risk
    fn calculate_operational_risk(&self, risk_data: &RiskData) -> f64 {
        // Operational risk based on system efficiency and volume
        let efficiency_risk = (1.0 - risk_data.efficiency) * 0.5;
        let volume_risk = if risk_data.volume < 0.3 { 0.3 } else { 0.0 };
        
        (efficiency_risk + volume_risk).min(1.0).max(0.0)
    }

    /// Calculate systematic risk
    fn calculate_systematic_risk(&self, risk_data: &RiskData) -> f64 {
        // Systematic risk based on general market conditions
        let market_stress = risk_data.volatility * 0.4 + (1.0 - risk_data.sentiment) * 0.3;
        let liquidity_stress = (1.0 - risk_data.liquidity) * 0.3;
        
        (market_stress + liquidity_stress).min(1.0).max(0.0)
    }

    /// Calculate idiosyncratic risk
    fn calculate_idiosyncratic_risk(&self, risk_data: &RiskData) -> f64 {
        // Idiosyncratic risk - specific to individual positions
        let specific_risk = risk_data.risk_factor * 0.6;
        let momentum_specific = (risk_data.momentum - 0.5).abs() * 0.4;
        
        (specific_risk + momentum_specific).min(1.0).max(0.0)
    }

    /// Calculate risk metrics
    fn calculate_risk_metrics(&self, risk_data: &RiskData) -> QarResult<RiskMetrics> {
        let var_95 = self.calculate_var(0.95)?;
        let var_99 = self.calculate_var(0.99)?;
        let expected_shortfall = self.calculate_expected_shortfall(0.95)?;
        let max_drawdown = self.calculate_max_drawdown();
        let sharpe_ratio = self.calculate_sharpe_ratio();
        let beta = self.calculate_beta();
        let tracking_error = self.calculate_tracking_error();

        Ok(RiskMetrics {
            var_95,
            var_99,
            expected_shortfall,
            max_drawdown,
            sharpe_ratio,
            beta,
            tracking_error,
        })
    }

    /// Calculate Value at Risk
    fn calculate_var(&self, confidence: f64) -> QarResult<f64> {
        if self.return_history.len() < 10 {
            return Ok(0.05); // Default 5% VaR
        }

        let mut sorted_returns = self.return_history.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let var = -sorted_returns.get(index).unwrap_or(&0.05);
        
        Ok(var.max(0.0))
    }

    /// Calculate Expected Shortfall (Conditional VaR)
    fn calculate_expected_shortfall(&self, confidence: f64) -> QarResult<f64> {
        if self.return_history.len() < 10 {
            return Ok(0.08); // Default 8% ES
        }

        let mut sorted_returns = self.return_history.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let cutoff_index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let tail_returns = &sorted_returns[..cutoff_index.max(1)];
        
        if tail_returns.is_empty() {
            return Ok(0.08);
        }

        let expected_shortfall = -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        Ok(expected_shortfall.max(0.0))
    }

    /// Calculate Maximum Drawdown
    fn calculate_max_drawdown(&self) -> f64 {
        if self.drawdown_history.is_empty() {
            return 0.05; // Default 5% max drawdown
        }

        self.drawdown_history.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&-0.05)
            .abs()
    }

    /// Calculate Sharpe Ratio
    fn calculate_sharpe_ratio(&self) -> f64 {
        if self.return_history.len() < 2 {
            return 0.0;
        }

        let mean_return = self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;
        let excess_return = mean_return - self.risk_params.risk_free_rate / 252.0; // Daily risk-free rate
        
        let variance = self.return_history.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (self.return_history.len() - 1) as f64;
        
        let volatility = variance.sqrt();
        
        if volatility > 0.0 {
            excess_return / volatility
        } else {
            0.0
        }
    }

    /// Calculate Beta
    fn calculate_beta(&self) -> f64 {
        if self.return_history.len() < 10 {
            return 1.0; // Default beta
        }

        // Simplified beta calculation assuming market return of 0.0005 (0.05% daily)
        let market_return = 0.0005;
        let mean_return = self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;
        
        let covariance = self.return_history.iter()
            .map(|r| (r - mean_return) * (market_return - market_return))
            .sum::<f64>() / (self.return_history.len() - 1) as f64;
        
        let market_variance = 0.0001; // Assumed market variance
        
        if market_variance > 0.0 {
            covariance / market_variance
        } else {
            1.0
        }
    }

    /// Calculate Tracking Error
    fn calculate_tracking_error(&self) -> f64 {
        if self.return_history.len() < 2 {
            return 0.02; // Default 2% tracking error
        }

        // Simplified tracking error vs benchmark (assumed 0 return)
        let benchmark_return = 0.0;
        let tracking_differences: Vec<f64> = self.return_history.iter()
            .map(|r| r - benchmark_return)
            .collect();
        
        let mean_diff = tracking_differences.iter().sum::<f64>() / tracking_differences.len() as f64;
        let variance = tracking_differences.iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>() / (tracking_differences.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Generate risk recommendations
    fn generate_recommendations(&self, components: &RiskComponents, metrics: &RiskMetrics) -> QarResult<Vec<RiskRecommendation>> {
        let mut recommendations = Vec::new();

        // High volatility risk recommendation
        if components.volatility_risk > 0.7 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::PositionSizing,
                priority: RecommendationPriority::High,
                description: "Reduce position sizes due to high volatility risk".to_string(),
                expected_impact: 0.3,
            });
        }

        // High liquidity risk recommendation
        if components.liquidity_risk > 0.6 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::TimeoutStrategy,
                priority: RecommendationPriority::Medium,
                description: "Implement timeout strategies for illiquid positions".to_string(),
                expected_impact: 0.2,
            });
        }

        // High concentration risk recommendation
        if components.concentration_risk > 0.8 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::Diversification,
                priority: RecommendationPriority::High,
                description: "Diversify positions to reduce concentration risk".to_string(),
                expected_impact: 0.4,
            });
        }

        // High VaR recommendation
        if metrics.var_95 > 0.1 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::RiskLimit,
                priority: RecommendationPriority::Critical,
                description: "Implement stricter risk limits due to high VaR".to_string(),
                expected_impact: 0.5,
            });
        }

        // Large drawdown recommendation
        if metrics.max_drawdown > 0.15 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::StopLoss,
                priority: RecommendationPriority::High,
                description: "Tighten stop-loss levels to limit drawdowns".to_string(),
                expected_impact: 0.3,
            });
        }

        // Low Sharpe ratio recommendation
        if metrics.sharpe_ratio < 0.5 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::HedgeStrategy,
                priority: RecommendationPriority::Medium,
                description: "Consider hedging strategies to improve risk-adjusted returns".to_string(),
                expected_impact: 0.25,
            });
        }

        Ok(recommendations)
    }

    /// Calculate overall risk score
    fn calculate_overall_risk_score(&self, components: &RiskComponents) -> f64 {
        let risk_weights = [
            (components.market_risk, 0.25),
            (components.liquidity_risk, 0.15),
            (components.volatility_risk, 0.20),
            (components.concentration_risk, 0.15),
            (components.operational_risk, 0.10),
            (components.systematic_risk, 0.10),
            (components.idiosyncratic_risk, 0.05),
        ];

        risk_weights.iter().map(|(risk, weight)| risk * weight).sum()
    }

    /// Calculate confidence in risk assessment
    fn calculate_risk_confidence(&self, components: &RiskComponents, metrics: &RiskMetrics) -> f64 {
        let mut confidence_factors = Vec::new();

        // Data availability confidence
        let data_confidence = if self.return_history.len() >= 30 {
            0.9
        } else if self.return_history.len() >= 10 {
            0.7
        } else {
            0.4
        };
        confidence_factors.push(data_confidence);

        // Risk consistency confidence
        let risk_values = vec![
            components.market_risk,
            components.liquidity_risk,
            components.volatility_risk,
            components.concentration_risk,
        ];
        
        let mean_risk = risk_values.iter().sum::<f64>() / risk_values.len() as f64;
        let risk_variance = risk_values.iter()
            .map(|r| (r - mean_risk).powi(2))
            .sum::<f64>() / risk_values.len() as f64;
        
        let consistency_confidence = 1.0 - risk_variance.sqrt();
        confidence_factors.push(consistency_confidence.max(0.0).min(1.0));

        // Metrics reliability confidence
        let metrics_confidence = if metrics.sharpe_ratio.abs() < 10.0 && 
                                   metrics.max_drawdown < 1.0 &&
                                   metrics.var_95 < 1.0 {
            0.8
        } else {
            0.5
        };
        confidence_factors.push(metrics_confidence);

        // Calculate overall confidence
        confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
    }

    /// Run stress tests
    pub fn run_stress_tests(&self, components: &RiskComponents) -> Vec<StressTestResult> {
        let mut results = Vec::new();
        
        for scenario in &self.risk_params.stress_scenarios {
            let stressed_components = self.apply_stress_scenario(components, scenario);
            let stressed_score = self.calculate_overall_risk_score(&stressed_components);
            
            results.push(StressTestResult {
                scenario_name: scenario.name.clone(),
                original_risk: self.calculate_overall_risk_score(components),
                stressed_risk: stressed_score,
                risk_increase: stressed_score - self.calculate_overall_risk_score(components),
                components: stressed_components,
            });
        }
        
        results
    }

    /// Apply stress scenario to risk components
    fn apply_stress_scenario(&self, components: &RiskComponents, scenario: &StressScenario) -> RiskComponents {
        RiskComponents {
            market_risk: (components.market_risk * (1.0 + scenario.market_shock.abs())).min(1.0),
            liquidity_risk: (components.liquidity_risk * (1.0 + scenario.liquidity_impact)).min(1.0),
            volatility_risk: (components.volatility_risk * scenario.volatility_multiplier).min(1.0),
            concentration_risk: components.concentration_risk,
            operational_risk: (components.operational_risk * 1.2).min(1.0),
            systematic_risk: (components.systematic_risk * (1.0 + scenario.market_shock.abs())).min(1.0),
            idiosyncratic_risk: components.idiosyncratic_risk,
        }
    }

    fn add_to_history(&mut self, result: RiskResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[RiskResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&RiskResult> {
        self.history.last()
    }

    /// Get risk parameters
    pub fn get_parameters(&self) -> &RiskParameters {
        &self.risk_params
    }

    /// Update risk parameters
    pub fn update_parameters(&mut self, params: RiskParameters) {
        self.risk_params = params;
    }
}

/// Risk data structure
#[derive(Debug, Clone)]
pub struct RiskData {
    pub volatility: f64,
    pub liquidity: f64,
    pub trend: f64,
    pub momentum: f64,
    pub volume: f64,
    pub sentiment: f64,
    pub efficiency: f64,
    pub risk_factor: f64,
}

/// Stress test result
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub original_risk: f64,
    pub stressed_risk: f64,
    pub risk_increase: f64,
    pub components: RiskComponents,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_risk_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = RiskAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Volatility.to_string(), 0.6);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.4);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.5);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.3);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        factors.insert(StandardFactors::Risk.to_string(), 0.7);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let risk_result = result.unwrap();
        assert!(risk_result.score >= 0.0 && risk_result.score <= 1.0);
        assert!(risk_result.confidence >= 0.0 && risk_result.confidence <= 1.0);
        assert!(!risk_result.recommendations.is_empty());
    }

    #[test]
    fn test_var_calculation() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = RiskAnalyzer::new(config).unwrap();
        
        // Set up return history
        analyzer.return_history = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08];
        
        let var_95 = analyzer.calculate_var(0.95).unwrap();
        assert!(var_95 > 0.0);
        assert!(var_95 <= 0.1);
        
        let var_99 = analyzer.calculate_var(0.99).unwrap();
        assert!(var_99 >= var_95); // VaR 99% should be higher than VaR 95%
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = RiskAnalyzer::new(config).unwrap();
        
        // Set up positive return history
        analyzer.return_history = vec![0.01, 0.02, 0.015, 0.025, 0.012, 0.018, 0.022, 0.008, 0.035, 0.014];
        
        let sharpe = analyzer.calculate_sharpe_ratio();
        assert!(sharpe > 0.0); // Should be positive for positive returns
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = RiskAnalyzer::new(config).unwrap();
        
        analyzer.drawdown_history = vec![-0.01, -0.02, -0.05, -0.03, -0.01, -0.08, -0.02];
        
        let max_dd = analyzer.calculate_max_drawdown();
        assert_eq!(max_dd, 0.08); // Should find the maximum absolute drawdown
    }

    #[test]
    fn test_risk_component_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = RiskAnalyzer::new(config).unwrap();
        
        let risk_data = RiskData {
            volatility: 0.8,
            liquidity: 0.3,
            trend: 0.5,
            momentum: 0.6,
            volume: 0.4,
            sentiment: 0.2,
            efficiency: 0.5,
            risk_factor: 0.7,
        };
        
        let components = analyzer.calculate_risk_components(&risk_data).unwrap();
        
        assert!(components.market_risk > 0.5); // High volatility should increase market risk
        assert!(components.liquidity_risk > 0.5); // Low liquidity should increase liquidity risk
        assert!(components.volatility_risk > 0.7); // Should reflect high volatility
        
        // All components should be in valid range
        assert!(components.market_risk >= 0.0 && components.market_risk <= 1.0);
        assert!(components.liquidity_risk >= 0.0 && components.liquidity_risk <= 1.0);
        assert!(components.volatility_risk >= 0.0 && components.volatility_risk <= 1.0);
    }

    #[test]
    fn test_recommendation_generation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = RiskAnalyzer::new(config).unwrap();
        
        let high_risk_components = RiskComponents {
            market_risk: 0.8,
            liquidity_risk: 0.7,
            volatility_risk: 0.9,
            concentration_risk: 0.85,
            operational_risk: 0.5,
            systematic_risk: 0.6,
            idiosyncratic_risk: 0.4,
        };
        
        let high_risk_metrics = RiskMetrics {
            var_95: 0.12,
            var_99: 0.18,
            expected_shortfall: 0.15,
            max_drawdown: 0.20,
            sharpe_ratio: 0.3,
            beta: 1.5,
            tracking_error: 0.08,
        };
        
        let recommendations = analyzer.generate_recommendations(&high_risk_components, &high_risk_metrics).unwrap();
        
        assert!(!recommendations.is_empty());
        
        // Should have critical recommendations for high VaR and max drawdown
        let has_critical = recommendations.iter().any(|r| matches!(r.priority, RecommendationPriority::Critical));
        assert!(has_critical);
        
        // Should have diversification recommendation for high concentration risk
        let has_diversification = recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::Diversification));
        assert!(has_diversification);
    }

    #[test]
    fn test_stress_test() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = RiskAnalyzer::new(config).unwrap();
        
        let base_components = RiskComponents {
            market_risk: 0.5,
            liquidity_risk: 0.4,
            volatility_risk: 0.3,
            concentration_risk: 0.4,
            operational_risk: 0.2,
            systematic_risk: 0.3,
            idiosyncratic_risk: 0.2,
        };
        
        let stress_results = analyzer.run_stress_tests(&base_components);
        
        assert!(!stress_results.is_empty());
        
        for result in &stress_results {
            assert!(result.stressed_risk >= result.original_risk); // Stress should increase risk
            assert!(result.risk_increase >= 0.0); // Risk increase should be non-negative
        }
    }

    #[test]
    fn test_overall_risk_score() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = RiskAnalyzer::new(config).unwrap();
        
        let low_risk_components = RiskComponents {
            market_risk: 0.2,
            liquidity_risk: 0.1,
            volatility_risk: 0.2,
            concentration_risk: 0.1,
            operational_risk: 0.1,
            systematic_risk: 0.15,
            idiosyncratic_risk: 0.1,
        };
        
        let high_risk_components = RiskComponents {
            market_risk: 0.9,
            liquidity_risk: 0.8,
            volatility_risk: 0.9,
            concentration_risk: 0.8,
            operational_risk: 0.7,
            systematic_risk: 0.8,
            idiosyncratic_risk: 0.6,
        };
        
        let low_score = analyzer.calculate_overall_risk_score(&low_risk_components);
        let high_score = analyzer.calculate_overall_risk_score(&high_risk_components);
        
        assert!(low_score < high_score);
        assert!(low_score >= 0.0 && low_score <= 1.0);
        assert!(high_score >= 0.0 && high_score <= 1.0);
    }
}