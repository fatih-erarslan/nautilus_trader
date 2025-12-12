//! Risk Assessment Module
//!
//! Advanced risk assessment for quantum trading decisions with real-time monitoring.

use crate::core::{QarResult, FactorMap, TradingDecision};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use super::{RiskAssessment, DecisionConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk assessor for trading decisions
pub struct RiskAssessor {
    config: RiskAssessorConfig,
    risk_models: RiskModels,
    risk_history: Vec<RiskAssessment>,
    risk_limits: RiskLimits,
    stress_scenarios: Vec<StressScenario>,
}

/// Risk assessor configuration
#[derive(Debug, Clone)]
pub struct RiskAssessorConfig {
    /// VaR confidence level
    pub var_confidence: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Maximum portfolio risk
    pub max_portfolio_risk: f64,
    /// Risk measurement frequency
    pub measurement_frequency: std::time::Duration,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Risk tolerance level
    pub risk_tolerance: f64,
}

/// Risk model implementations
#[derive(Debug)]
pub struct RiskModels {
    /// Parametric VaR model
    pub parametric_var: ParametricVarModel,
    /// Monte Carlo simulation model
    pub monte_carlo: MonteCarloModel,
    /// Historical simulation model
    pub historical_sim: HistoricalSimulationModel,
    /// Quantum risk model
    pub quantum_model: QuantumRiskModel,
}

/// Parametric VaR model
#[derive(Debug)]
pub struct ParametricVarModel {
    /// Mean return
    pub mean_return: f64,
    /// Return volatility
    pub volatility: f64,
    /// Distribution type
    pub distribution: DistributionType,
}

/// Distribution type for risk modeling
#[derive(Debug)]
pub enum DistributionType {
    Normal,
    StudentT { degrees_freedom: f64 },
    Skewed { skewness: f64, kurtosis: f64 },
}

/// Monte Carlo simulation model
#[derive(Debug)]
pub struct MonteCarloModel {
    /// Number of simulations
    pub num_simulations: usize,
    /// Time horizon
    pub time_horizon: std::time::Duration,
    /// Random seed
    pub random_seed: Option<u64>,
}

/// Historical simulation model
#[derive(Debug)]
pub struct HistoricalSimulationModel {
    /// Historical returns
    pub historical_returns: Vec<f64>,
    /// Lookback window
    pub lookback_window: usize,
    /// Decay factor for weighting
    pub decay_factor: f64,
}

/// Quantum risk model
#[derive(Debug)]
pub struct QuantumRiskModel {
    /// Quantum correlation matrix
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Quantum volatility factors
    pub volatility_factors: Vec<f64>,
    /// Coherence time
    pub coherence_time: f64,
    /// Entanglement strength
    pub entanglement_strength: f64,
}

/// Risk limits configuration
#[derive(Debug, Clone)]
pub struct RiskLimits {
    /// Maximum single position VaR
    pub max_position_var: f64,
    /// Maximum portfolio VaR
    pub max_portfolio_var: f64,
    /// Maximum concentration
    pub max_concentration: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Maximum correlation
    pub max_correlation: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Minimum liquidity
    pub min_liquidity: f64,
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
    /// Correlation shock
    pub correlation_shock: f64,
    /// Liquidity impact
    pub liquidity_impact: f64,
    /// Scenario probability
    pub probability: f64,
}

/// Enhanced risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRiskAssessment {
    /// Basic risk assessment
    pub basic_assessment: RiskAssessment,
    /// Risk breakdown by source
    pub risk_breakdown: RiskBreakdown,
    /// Stress test results
    pub stress_test_results: Vec<StressTestResult>,
    /// Risk attribution
    pub risk_attribution: RiskAttribution,
    /// Risk recommendations
    pub recommendations: Vec<RiskRecommendation>,
    /// Risk monitoring alerts
    pub alerts: Vec<RiskAlert>,
    /// Quantum risk factors
    pub quantum_factors: QuantumRiskFactors,
}

/// Risk breakdown by source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskBreakdown {
    /// Market risk
    pub market_risk: f64,
    /// Liquidity risk
    pub liquidity_risk: f64,
    /// Operational risk
    pub operational_risk: f64,
    /// Model risk
    pub model_risk: f64,
    /// Counterparty risk
    pub counterparty_risk: f64,
    /// Concentration risk
    pub concentration_risk: f64,
    /// Quantum decoherence risk
    pub quantum_decoherence_risk: f64,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario name
    pub scenario_name: String,
    /// Projected loss
    pub projected_loss: f64,
    /// Loss probability
    pub loss_probability: f64,
    /// Time to recovery
    pub recovery_time: std::time::Duration,
    /// Risk mitigation impact
    pub mitigation_impact: f64,
}

/// Risk attribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAttribution {
    /// Factor contributions
    pub factor_contributions: HashMap<String, f64>,
    /// Asset contributions
    pub asset_contributions: HashMap<String, f64>,
    /// Time contributions
    pub time_contributions: HashMap<String, f64>,
    /// Marginal risk contributions
    pub marginal_contributions: HashMap<String, f64>,
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
    /// Implementation cost
    pub implementation_cost: f64,
    /// Time sensitivity
    pub time_sensitivity: TimeSensitivity,
}

/// Recommendation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ReducePosition,
    IncreaseHedging,
    DiversifyPortfolio,
    AdjustLeverage,
    ModifyStrategy,
    AddStopLoss,
    IncreaseReserves,
    QuantumRecalibration,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Time sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSensitivity {
    Immediate,
    WithinHour,
    WithinDay,
    WithinWeek,
    Flexible,
}

/// Risk alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Alert level
    pub level: AlertLevel,
    /// Message
    pub message: String,
    /// Threshold breached
    pub threshold_breached: f64,
    /// Current value
    pub current_value: f64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Alert type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    VarBreach,
    DrawdownExcess,
    ConcentrationLimit,
    LiquidityShortage,
    VolatilitySpike,
    CorrelationBreakdown,
    QuantumDecoherence,
}

/// Alert level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Quantum risk factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRiskFactors {
    /// Decoherence risk
    pub decoherence_risk: f64,
    /// Measurement uncertainty
    pub measurement_uncertainty: f64,
    /// Quantum correlation breakdown
    pub correlation_breakdown_risk: f64,
    /// Entanglement decay
    pub entanglement_decay: f64,
    /// Quantum noise impact
    pub quantum_noise_impact: f64,
}

impl Default for RiskAssessorConfig {
    fn default() -> Self {
        Self {
            var_confidence: 0.95,
            max_position_size: 0.1,
            max_portfolio_risk: 0.05,
            measurement_frequency: std::time::Duration::from_secs(60),
            real_time_monitoring: true,
            risk_tolerance: 0.6,
        }
    }
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_var: 0.02,
            max_portfolio_var: 0.05,
            max_concentration: 0.3,
            max_leverage: 3.0,
            max_correlation: 0.8,
            max_drawdown: 0.15,
            min_liquidity: 0.3,
        }
    }
}

impl RiskAssessor {
    /// Create a new risk assessor
    pub fn new(config: RiskAssessorConfig) -> QarResult<Self> {
        let risk_models = RiskModels {
            parametric_var: ParametricVarModel {
                mean_return: 0.0,
                volatility: 0.2,
                distribution: DistributionType::Normal,
            },
            monte_carlo: MonteCarloModel {
                num_simulations: 10000,
                time_horizon: std::time::Duration::from_secs(86400), // 1 day
                random_seed: None,
            },
            historical_sim: HistoricalSimulationModel {
                historical_returns: Vec::new(),
                lookback_window: 250,
                decay_factor: 0.94,
            },
            quantum_model: QuantumRiskModel {
                correlation_matrix: vec![vec![1.0]],
                volatility_factors: vec![0.2],
                coherence_time: 100.0,
                entanglement_strength: 0.8,
            },
        };

        let stress_scenarios = Self::initialize_stress_scenarios();

        Ok(Self {
            config,
            risk_models,
            risk_history: Vec::new(),
            risk_limits: RiskLimits::default(),
            stress_scenarios,
        })
    }

    /// Assess risk for a trading decision
    pub async fn assess_risk(
        &mut self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<EnhancedRiskAssessment> {
        // Calculate basic risk assessment
        let basic_assessment = self.calculate_basic_risk(decision, factors, analysis)?;

        // Calculate detailed risk breakdown
        let risk_breakdown = self.calculate_risk_breakdown(decision, factors)?;

        // Run stress tests
        let stress_test_results = self.run_stress_tests(decision, factors)?;

        // Calculate risk attribution
        let risk_attribution = self.calculate_risk_attribution(decision, factors)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&basic_assessment, &risk_breakdown)?;

        // Check for alerts
        let alerts = self.check_risk_alerts(&basic_assessment, &risk_breakdown)?;

        // Calculate quantum risk factors
        let quantum_factors = self.calculate_quantum_risk_factors(decision, factors)?;

        let enhanced_assessment = EnhancedRiskAssessment {
            basic_assessment,
            risk_breakdown,
            stress_test_results,
            risk_attribution,
            recommendations,
            alerts,
            quantum_factors,
        };

        // Store in history
        self.risk_history.push(enhanced_assessment.basic_assessment.clone());
        if self.risk_history.len() > 1000 {
            self.risk_history.remove(0);
        }

        Ok(enhanced_assessment)
    }

    /// Calculate basic risk assessment
    fn calculate_basic_risk(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<RiskAssessment> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let risk_factor = factors.get_factor(&crate::core::StandardFactors::Risk)?;

        // Calculate VaR using multiple models
        let var_95 = self.calculate_var(decision, factors, 0.95)?;

        // Calculate Expected Shortfall
        let expected_shortfall = var_95 * 1.3; // Simplified calculation

        // Calculate maximum drawdown risk
        let max_drawdown_risk = volatility * 0.5 + risk_factor * 0.3;

        // Calculate liquidity risk
        let liquidity_risk = 1.0 - liquidity;

        // Calculate risk-adjusted return
        let expected_return = decision.expected_return.unwrap_or(0.0);
        let risk_adjusted_return = if var_95 > 0.0 {
            expected_return / var_95
        } else {
            expected_return
        };

        // Calculate overall risk score
        let risk_score = (var_95 * 0.3 + max_drawdown_risk * 0.25 + 
                         liquidity_risk * 0.2 + risk_factor * 0.25).min(1.0);

        Ok(RiskAssessment {
            risk_score,
            var_95,
            expected_shortfall,
            max_drawdown_risk,
            liquidity_risk,
            risk_adjusted_return,
        })
    }

    /// Calculate VaR using multiple models
    fn calculate_var(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        confidence: f64,
    ) -> QarResult<f64> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let position_size = 0.1; // Simplified position size

        // Parametric VaR
        let parametric_var = self.calculate_parametric_var(volatility, position_size, confidence);

        // Monte Carlo VaR
        let monte_carlo_var = self.calculate_monte_carlo_var(volatility, position_size, confidence)?;

        // Historical simulation VaR
        let historical_var = self.calculate_historical_var(position_size, confidence);

        // Quantum VaR
        let quantum_var = self.calculate_quantum_var(decision, factors, confidence)?;

        // Combine VaR estimates with weights
        let combined_var = parametric_var * 0.3 + monte_carlo_var * 0.3 + 
                          historical_var * 0.2 + quantum_var * 0.2;

        Ok(combined_var)
    }

    /// Calculate parametric VaR
    fn calculate_parametric_var(&self, volatility: f64, position_size: f64, confidence: f64) -> f64 {
        // Z-score for given confidence level
        let z_score = match confidence {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.645, // Default to 95%
        };

        volatility * z_score * position_size
    }

    /// Calculate Monte Carlo VaR
    fn calculate_monte_carlo_var(
        &self,
        volatility: f64,
        position_size: f64,
        confidence: f64,
    ) -> QarResult<f64> {
        let num_sims = self.risk_models.monte_carlo.num_simulations;
        let mut losses = Vec::with_capacity(num_sims);

        // Simple Monte Carlo simulation
        for _ in 0..num_sims {
            let random_return = self.generate_random_return(volatility);
            let loss = -random_return * position_size;
            losses.push(loss);
        }

        // Sort losses and find VaR
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - confidence) * losses.len() as f64) as usize;
        
        Ok(losses.get(var_index).cloned().unwrap_or(0.0).max(0.0))
    }

    /// Generate random return for Monte Carlo simulation
    fn generate_random_return(&self, volatility: f64) -> f64 {
        // Simplified normal distribution using Box-Muller transform
        use std::f64::consts::PI;
        
        let u1: f64 = rand::random();
        let u2: f64 = rand::random();
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        z * volatility
    }

    /// Calculate historical simulation VaR
    fn calculate_historical_var(&self, position_size: f64, confidence: f64) -> f64 {
        if self.risk_models.historical_sim.historical_returns.is_empty() {
            return 0.02; // Default VaR
        }

        let returns = &self.risk_models.historical_sim.historical_returns;
        let mut losses: Vec<f64> = returns.iter()
            .map(|&r| -r * position_size)
            .collect();

        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - confidence) * losses.len() as f64) as usize;
        
        losses.get(var_index).cloned().unwrap_or(0.0).max(0.0)
    }

    /// Calculate quantum VaR
    fn calculate_quantum_var(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        confidence: f64,
    ) -> QarResult<f64> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        
        // Quantum correction based on decoherence and entanglement
        let decoherence_factor = 1.0 - (efficiency * self.risk_models.quantum_model.coherence_time / 1000.0).min(1.0);
        let entanglement_correction = self.risk_models.quantum_model.entanglement_strength;
        
        // Base quantum volatility
        let quantum_volatility = volatility * (1.0 + decoherence_factor * 0.5);
        
        // Quantum uncertainty principle adjustment
        let uncertainty_adjustment = 1.0 + (1.0 - entanglement_correction) * 0.3;
        
        let position_size = 0.1; // Simplified
        let z_score = match confidence {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.645,
        };

        Ok(quantum_volatility * z_score * position_size * uncertainty_adjustment)
    }

    /// Calculate detailed risk breakdown
    fn calculate_risk_breakdown(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<RiskBreakdown> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let risk_factor = factors.get_factor(&crate::core::StandardFactors::Risk)?;
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;

        Ok(RiskBreakdown {
            market_risk: volatility * 0.8 + risk_factor * 0.2,
            liquidity_risk: 1.0 - liquidity,
            operational_risk: (1.0 - efficiency) * 0.5,
            model_risk: 0.05, // Fixed for now
            counterparty_risk: risk_factor * 0.3,
            concentration_risk: decision.confidence.map(|c| 1.0 - c).unwrap_or(0.5) * 0.4,
            quantum_decoherence_risk: self.calculate_decoherence_risk(factors)?,
        })
    }

    /// Calculate quantum decoherence risk
    fn calculate_decoherence_risk(&self, factors: &FactorMap) -> QarResult<f64> {
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        // Decoherence increases with market inefficiency and volatility
        let base_decoherence = 1.0 - efficiency;
        let volatility_impact = volatility * 0.5;
        let coherence_time_factor = (100.0 / self.risk_models.quantum_model.coherence_time).min(1.0);
        
        Ok((base_decoherence + volatility_impact + coherence_time_factor) / 3.0)
    }

    /// Run stress tests
    fn run_stress_tests(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<Vec<StressTestResult>> {
        let mut results = Vec::new();

        for scenario in &self.stress_scenarios {
            let stressed_factors = self.apply_stress_scenario(factors, scenario)?;
            let stressed_loss = self.calculate_stressed_loss(decision, &stressed_factors, scenario)?;
            
            results.push(StressTestResult {
                scenario_name: scenario.name.clone(),
                projected_loss: stressed_loss,
                loss_probability: scenario.probability,
                recovery_time: std::time::Duration::from_secs(3600 * 24 * 30), // 30 days simplified
                mitigation_impact: 0.3, // 30% risk reduction with mitigation
            });
        }

        Ok(results)
    }

    /// Apply stress scenario to factors
    fn apply_stress_scenario(
        &self,
        factors: &FactorMap,
        scenario: &StressScenario,
    ) -> QarResult<FactorMap> {
        let mut stressed_factors = HashMap::new();

        for (factor_name, &value) in factors.get_all_factors().iter() {
            let stressed_value = match factor_name.as_str() {
                "Volatility" => (value * scenario.volatility_multiplier).min(1.0),
                "Trend" => value + scenario.market_shock,
                "Liquidity" => (value * (1.0 - scenario.liquidity_impact)).max(0.0),
                _ => value,
            };
            stressed_factors.insert(factor_name.clone(), stressed_value);
        }

        FactorMap::new(stressed_factors)
    }

    /// Calculate loss under stress scenario
    fn calculate_stressed_loss(
        &self,
        decision: &TradingDecision,
        stressed_factors: &FactorMap,
        scenario: &StressScenario,
    ) -> QarResult<f64> {
        let stressed_volatility = stressed_factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let position_size = 0.1; // Simplified
        
        // Calculate loss based on scenario severity
        let base_loss = stressed_volatility * position_size * scenario.volatility_multiplier;
        let market_shock_loss = scenario.market_shock.abs() * position_size;
        
        Ok(base_loss + market_shock_loss)
    }

    /// Calculate risk attribution
    fn calculate_risk_attribution(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<RiskAttribution> {
        let mut factor_contributions = HashMap::new();
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let risk_factor = factors.get_factor(&crate::core::StandardFactors::Risk)?;

        factor_contributions.insert("Volatility".to_string(), volatility * 0.4);
        factor_contributions.insert("Liquidity".to_string(), (1.0 - liquidity) * 0.2);
        factor_contributions.insert("Risk".to_string(), risk_factor * 0.3);
        factor_contributions.insert("Quantum".to_string(), 0.1);

        let mut asset_contributions = HashMap::new();
        asset_contributions.insert("Primary_Asset".to_string(), 0.8);
        asset_contributions.insert("Correlated_Assets".to_string(), 0.2);

        let mut time_contributions = HashMap::new();
        time_contributions.insert("Intraday".to_string(), 0.6);
        time_contributions.insert("Overnight".to_string(), 0.4);

        let mut marginal_contributions = HashMap::new();
        marginal_contributions.insert("Position_Size".to_string(), 0.5);
        marginal_contributions.insert("Leverage".to_string(), 0.3);
        marginal_contributions.insert("Correlation".to_string(), 0.2);

        Ok(RiskAttribution {
            factor_contributions,
            asset_contributions,
            time_contributions,
            marginal_contributions,
        })
    }

    /// Generate risk recommendations
    fn generate_recommendations(
        &self,
        basic_assessment: &RiskAssessment,
        risk_breakdown: &RiskBreakdown,
    ) -> QarResult<Vec<RiskRecommendation>> {
        let mut recommendations = Vec::new();

        // High VaR recommendation
        if basic_assessment.var_95 > self.risk_limits.max_position_var {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::ReducePosition,
                priority: RecommendationPriority::High,
                description: format!("VaR ({:.3}) exceeds limit ({:.3}). Reduce position size by 30%.", 
                    basic_assessment.var_95, self.risk_limits.max_position_var),
                expected_impact: 0.3,
                implementation_cost: 0.05,
                time_sensitivity: TimeSensitivity::WithinHour,
            });
        }

        // High liquidity risk
        if risk_breakdown.liquidity_risk > 0.7 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::IncreaseReserves,
                priority: RecommendationPriority::Medium,
                description: "High liquidity risk detected. Increase cash reserves.".to_string(),
                expected_impact: 0.2,
                implementation_cost: 0.02,
                time_sensitivity: TimeSensitivity::WithinDay,
            });
        }

        // Quantum decoherence risk
        if risk_breakdown.quantum_decoherence_risk > 0.6 {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::QuantumRecalibration,
                priority: RecommendationPriority::Medium,
                description: "Quantum decoherence risk elevated. Recalibrate quantum models.".to_string(),
                expected_impact: 0.25,
                implementation_cost: 0.1,
                time_sensitivity: TimeSensitivity::WithinHour,
            });
        }

        // High drawdown risk
        if basic_assessment.max_drawdown_risk > self.risk_limits.max_drawdown {
            recommendations.push(RiskRecommendation {
                recommendation_type: RecommendationType::AddStopLoss,
                priority: RecommendationPriority::High,
                description: "Maximum drawdown risk exceeds limit. Implement tighter stop-loss.".to_string(),
                expected_impact: 0.4,
                implementation_cost: 0.01,
                time_sensitivity: TimeSensitivity::Immediate,
            });
        }

        Ok(recommendations)
    }

    /// Check for risk alerts
    fn check_risk_alerts(
        &self,
        basic_assessment: &RiskAssessment,
        risk_breakdown: &RiskBreakdown,
    ) -> QarResult<Vec<RiskAlert>> {
        let mut alerts = Vec::new();
        let now = chrono::Utc::now();

        // VaR breach alert
        if basic_assessment.var_95 > self.risk_limits.max_position_var {
            alerts.push(RiskAlert {
                alert_type: AlertType::VarBreach,
                level: AlertLevel::Critical,
                message: "Position VaR exceeds limit".to_string(),
                threshold_breached: self.risk_limits.max_position_var,
                current_value: basic_assessment.var_95,
                timestamp: now,
            });
        }

        // Drawdown alert
        if basic_assessment.max_drawdown_risk > self.risk_limits.max_drawdown {
            alerts.push(RiskAlert {
                alert_type: AlertType::DrawdownExcess,
                level: AlertLevel::Warning,
                message: "Maximum drawdown risk elevated".to_string(),
                threshold_breached: self.risk_limits.max_drawdown,
                current_value: basic_assessment.max_drawdown_risk,
                timestamp: now,
            });
        }

        // Liquidity alert
        if risk_breakdown.liquidity_risk > 0.8 {
            alerts.push(RiskAlert {
                alert_type: AlertType::LiquidityShortage,
                level: AlertLevel::Warning,
                message: "Liquidity risk elevated".to_string(),
                threshold_breached: 0.8,
                current_value: risk_breakdown.liquidity_risk,
                timestamp: now,
            });
        }

        // Quantum decoherence alert
        if risk_breakdown.quantum_decoherence_risk > 0.7 {
            alerts.push(RiskAlert {
                alert_type: AlertType::QuantumDecoherence,
                level: AlertLevel::Info,
                message: "Quantum coherence degrading".to_string(),
                threshold_breached: 0.7,
                current_value: risk_breakdown.quantum_decoherence_risk,
                timestamp: now,
            });
        }

        Ok(alerts)
    }

    /// Calculate quantum-specific risk factors
    fn calculate_quantum_risk_factors(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<QuantumRiskFactors> {
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        let decoherence_risk = self.calculate_decoherence_risk(factors)?;
        
        // Measurement uncertainty increases with quantum state complexity
        let measurement_uncertainty = (1.0 - efficiency) * 0.5 + volatility * 0.3;
        
        // Correlation breakdown risk
        let correlation_breakdown_risk = decoherence_risk * 0.8;
        
        // Entanglement decay
        let entanglement_decay = (decoherence_risk + measurement_uncertainty) / 2.0;
        
        // Quantum noise impact
        let quantum_noise_impact = volatility * 0.6 + (1.0 - efficiency) * 0.4;

        Ok(QuantumRiskFactors {
            decoherence_risk,
            measurement_uncertainty,
            correlation_breakdown_risk,
            entanglement_decay,
            quantum_noise_impact,
        })
    }

    /// Initialize default stress scenarios
    fn initialize_stress_scenarios() -> Vec<StressScenario> {
        vec![
            StressScenario {
                name: "Market Crash".to_string(),
                market_shock: -0.2,
                volatility_multiplier: 3.0,
                correlation_shock: 0.5,
                liquidity_impact: 0.6,
                probability: 0.05,
            },
            StressScenario {
                name: "Flash Crash".to_string(),
                market_shock: -0.15,
                volatility_multiplier: 5.0,
                correlation_shock: 0.8,
                liquidity_impact: 0.8,
                probability: 0.02,
            },
            StressScenario {
                name: "Liquidity Crisis".to_string(),
                market_shock: -0.1,
                volatility_multiplier: 2.0,
                correlation_shock: 0.3,
                liquidity_impact: 0.9,
                probability: 0.08,
            },
            StressScenario {
                name: "Quantum Decoherence".to_string(),
                market_shock: -0.05,
                volatility_multiplier: 1.5,
                correlation_shock: 0.9,
                liquidity_impact: 0.3,
                probability: 0.1,
            },
        ]
    }

    /// Update risk models with new data
    pub fn update_risk_models(&mut self, returns: Vec<f64>) {
        // Update historical simulation model
        self.risk_models.historical_sim.historical_returns.extend(returns.clone());
        
        // Maintain window size
        let max_size = self.risk_models.historical_sim.lookback_window;
        if self.risk_models.historical_sim.historical_returns.len() > max_size {
            let excess = self.risk_models.historical_sim.historical_returns.len() - max_size;
            self.risk_models.historical_sim.historical_returns.drain(..excess);
        }

        // Update parametric model
        if !returns.is_empty() {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (returns.len() - 1).max(1) as f64;
            
            self.risk_models.parametric_var.mean_return = mean;
            self.risk_models.parametric_var.volatility = variance.sqrt();
        }
    }

    /// Get risk history
    pub fn get_risk_history(&self) -> &[RiskAssessment] {
        &self.risk_history
    }

    /// Get latest risk assessment
    pub fn get_latest_assessment(&self) -> Option<&RiskAssessment> {
        self.risk_history.last()
    }

    /// Update risk limits
    pub fn update_risk_limits(&mut self, limits: RiskLimits) {
        self.risk_limits = limits;
    }

    /// Get current risk limits
    pub fn get_risk_limits(&self) -> &RiskLimits {
        &self.risk_limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{StandardFactors, DecisionType};
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    fn create_test_decision() -> TradingDecision {
        TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            expected_return: Some(0.05),
            risk_assessment: Some(0.3),
            urgency_score: Some(0.6),
            reasoning: "Test decision".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    fn create_test_factors() -> FactorMap {
        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Volatility.to_string(), 0.2);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.7);
        factors.insert(StandardFactors::Risk.to_string(), 0.4);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.7);
        FactorMap::new(factors).unwrap()
    }

    fn create_test_analysis() -> AnalysisResult {
        AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_risk_assessor_creation() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config);
        assert!(assessor.is_ok());
    }

    #[tokio::test]
    async fn test_basic_risk_assessment() {
        let config = RiskAssessorConfig::default();
        let mut assessor = RiskAssessor::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let assessment = assessor.assess_risk(&decision, &factors, &analysis).await;
        assert!(assessment.is_ok());

        let assessment = assessment.unwrap();
        assert!(assessment.basic_assessment.risk_score >= 0.0 && assessment.basic_assessment.risk_score <= 1.0);
        assert!(assessment.basic_assessment.var_95 >= 0.0);
        assert!(assessment.basic_assessment.expected_shortfall >= assessment.basic_assessment.var_95);
    }

    #[test]
    fn test_parametric_var_calculation() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let var = assessor.calculate_parametric_var(0.2, 0.1, 0.95);
        assert!(var > 0.0);
        assert!(var < 1.0);
        
        // Higher volatility should result in higher VaR
        let high_vol_var = assessor.calculate_parametric_var(0.4, 0.1, 0.95);
        assert!(high_vol_var > var);
    }

    #[tokio::test]
    async fn test_monte_carlo_var() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let var = assessor.calculate_monte_carlo_var(0.2, 0.1, 0.95).await;
        assert!(var.is_ok());
        
        let var_value = var.unwrap();
        assert!(var_value >= 0.0);
    }

    #[tokio::test]
    async fn test_stress_testing() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        
        let stress_results = assessor.run_stress_tests(&decision, &factors);
        assert!(stress_results.is_ok());
        
        let results = stress_results.unwrap();
        assert!(!results.is_empty());
        
        for result in &results {
            assert!(result.projected_loss >= 0.0);
            assert!(result.loss_probability >= 0.0 && result.loss_probability <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_quantum_risk_factors() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        
        let quantum_factors = assessor.calculate_quantum_risk_factors(&decision, &factors);
        assert!(quantum_factors.is_ok());
        
        let factors = quantum_factors.unwrap();
        assert!(factors.decoherence_risk >= 0.0 && factors.decoherence_risk <= 1.0);
        assert!(factors.measurement_uncertainty >= 0.0 && factors.measurement_uncertainty <= 1.0);
        assert!(factors.entanglement_decay >= 0.0 && factors.entanglement_decay <= 1.0);
    }

    #[test]
    fn test_risk_breakdown_calculation() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        
        let breakdown = assessor.calculate_risk_breakdown(&decision, &factors);
        assert!(breakdown.is_ok());
        
        let breakdown = breakdown.unwrap();
        assert!(breakdown.market_risk >= 0.0 && breakdown.market_risk <= 1.0);
        assert!(breakdown.liquidity_risk >= 0.0 && breakdown.liquidity_risk <= 1.0);
        assert!(breakdown.operational_risk >= 0.0 && breakdown.operational_risk <= 1.0);
    }

    #[test]
    fn test_alert_generation() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        // Create high-risk assessment
        let high_risk_assessment = RiskAssessment {
            risk_score: 0.9,
            var_95: 0.08,  // Above default limit of 0.02
            expected_shortfall: 0.12,
            max_drawdown_risk: 0.25,  // Above default limit of 0.15
            liquidity_risk: 0.9,
            risk_adjusted_return: 0.1,
        };
        
        let risk_breakdown = RiskBreakdown {
            market_risk: 0.8,
            liquidity_risk: 0.9,  // High liquidity risk
            operational_risk: 0.3,
            model_risk: 0.1,
            counterparty_risk: 0.2,
            concentration_risk: 0.4,
            quantum_decoherence_risk: 0.8,  // High decoherence risk
        };
        
        let alerts = assessor.check_risk_alerts(&high_risk_assessment, &risk_breakdown);
        assert!(alerts.is_ok());
        
        let alerts = alerts.unwrap();
        assert!(!alerts.is_empty());
        
        // Should have alerts for VaR breach, drawdown, liquidity, and quantum decoherence
        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::VarBreach)));
        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::DrawdownExcess)));
    }

    #[test]
    fn test_recommendation_generation() {
        let config = RiskAssessorConfig::default();
        let assessor = RiskAssessor::new(config).unwrap();
        
        let high_risk_assessment = RiskAssessment {
            risk_score: 0.9,
            var_95: 0.08,
            expected_shortfall: 0.12,
            max_drawdown_risk: 0.25,
            liquidity_risk: 0.8,
            risk_adjusted_return: 0.1,
        };
        
        let risk_breakdown = RiskBreakdown {
            market_risk: 0.8,
            liquidity_risk: 0.8,
            operational_risk: 0.3,
            model_risk: 0.1,
            counterparty_risk: 0.2,
            concentration_risk: 0.4,
            quantum_decoherence_risk: 0.7,
        };
        
        let recommendations = assessor.generate_recommendations(&high_risk_assessment, &risk_breakdown);
        assert!(recommendations.is_ok());
        
        let recommendations = recommendations.unwrap();
        assert!(!recommendations.is_empty());
        
        // Should have recommendations for position reduction, reserves, and quantum recalibration
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::ReducePosition)));
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, RecommendationType::AddStopLoss)));
    }

    #[test]
    fn test_risk_model_updates() {
        let config = RiskAssessorConfig::default();
        let mut assessor = RiskAssessor::new(config).unwrap();
        
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
        assessor.update_risk_models(returns.clone());
        
        // Check that historical returns were updated
        assert_eq!(assessor.risk_models.historical_sim.historical_returns.len(), returns.len());
        
        // Check that parametric model was updated
        assert!(assessor.risk_models.parametric_var.volatility > 0.0);
    }
}