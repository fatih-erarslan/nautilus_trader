//! Decision Engine Module
//!
//! This module contains the quantum-enhanced decision engine for trading decisions,
//! including strategy selection, risk assessment, and execution planning.

pub mod decision_engine;
pub mod enhanced_decision_engine;
pub mod strategy_selector;
pub mod risk_assessor;
pub mod execution_planner;
pub mod quantum_decision;
pub mod decision_validator;
pub mod portfolio_manager;
pub mod order_manager;

// Re-export the quantum decision engine as DecisionEngineImpl
pub use quantum_decision::QuantumDecisionEngine as DecisionEngineImpl;

use crate::core::{QarResult, DecisionType, StandardFactors, FactorMap, TradingDecision};
use crate::analysis::AnalysisResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Decision engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfig {
    /// Risk tolerance (0.0 to 1.0)
    pub risk_tolerance: f64,
    /// Minimum confidence for decisions
    pub min_confidence: f64,
    /// Maximum position size
    pub max_position_size: f64,
    /// Use quantum decision making
    pub use_quantum: bool,
    /// Decision timeout in seconds
    pub decision_timeout: u64,
    /// Enable portfolio rebalancing
    pub enable_rebalancing: bool,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            min_confidence: 0.6,
            max_position_size: 0.1,
            use_quantum: true,
            decision_timeout: 30,
            enable_rebalancing: true,
        }
    }
}

/// Enhanced trading decision with quantum insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTradingDecision {
    /// Base trading decision
    pub decision: TradingDecision,
    /// Quantum analysis insights
    pub quantum_insights: Option<QuantumInsights>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Execution plan
    pub execution_plan: ExecutionPlan,
    /// Decision metadata
    pub metadata: HashMap<String, String>,
}

/// Quantum decision insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumInsights {
    /// Quantum superposition analysis
    pub superposition_analysis: Vec<f64>,
    /// Entanglement correlations
    pub entanglement_correlations: HashMap<String, f64>,
    /// Quantum interference patterns
    pub interference_patterns: Vec<f64>,
    /// Measurement uncertainty
    pub measurement_uncertainty: f64,
}

/// Risk assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score (0.0 to 1.0)
    pub risk_score: f64,
    /// Value at Risk (VaR)
    pub var_95: f64,
    /// Expected Shortfall
    pub expected_shortfall: f64,
    /// Maximum drawdown risk
    pub max_drawdown_risk: f64,
    /// Liquidity risk
    pub liquidity_risk: f64,
    /// Risk-adjusted return expectation
    pub risk_adjusted_return: f64,
}

/// Execution plan for trading decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Execution strategy
    pub strategy: ExecutionStrategy,
    /// Target position size
    pub position_size: f64,
    /// Stop loss level
    pub stop_loss: Option<f64>,
    /// Take profit level
    pub take_profit: Option<f64>,
    /// Time horizon
    pub time_horizon: std::time::Duration,
    /// Execution priority
    pub priority: ExecutionPriority,
}

/// Execution strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TWAP,
    VWAP,
    QuantumOptimized,
}

/// Execution priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPriority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Main decision engine
#[derive(Debug)]
pub struct DecisionEngine {
    config: DecisionConfig,
    strategy_selector: strategy_selector::StrategySelector,
    risk_assessor: risk_assessor::RiskAssessor,
    execution_planner: execution_planner::ExecutionPlanner,
    quantum_decision: quantum_decision::QuantumDecisionMaker,
    decision_validator: decision_validator::DecisionValidator,
    portfolio_manager: portfolio_manager::PortfolioManager,
    order_manager: order_manager::OrderManager,
    decision_history: Vec<EnhancedTradingDecision>,
}

impl DecisionEngine {
    /// Create a new decision engine
    pub fn new(config: DecisionConfig) -> QarResult<Self> {
        Ok(Self {
            strategy_selector: strategy_selector::StrategySelector::new(config.clone())?,
            risk_assessor: risk_assessor::RiskAssessor::new(config.clone())?,
            execution_planner: execution_planner::ExecutionPlanner::new(config.clone())?,
            quantum_decision: quantum_decision::QuantumDecisionMaker::new(config.clone())?,
            decision_validator: decision_validator::DecisionValidator::new(config.clone())?,
            portfolio_manager: portfolio_manager::PortfolioManager::new(config.clone())?,
            order_manager: order_manager::OrderManager::new(config.clone())?,
            config,
            decision_history: Vec::new(),
        })
    }

    /// Make a comprehensive trading decision
    pub async fn make_decision(
        &mut self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<EnhancedTradingDecision> {
        // Select trading strategy
        let strategy = self.strategy_selector.select_strategy(factors, analysis).await?;

        // Make base decision
        let base_decision = if self.config.use_quantum {
            self.quantum_decision.make_quantum_decision(factors, analysis, &strategy).await?
        } else {
            self.make_classical_decision(factors, analysis, &strategy).await?
        };

        // Assess risk
        let risk_assessment = self.risk_assessor.assess_risk(&base_decision, factors, analysis).await?;

        // Validate decision
        self.decision_validator.validate_decision(&base_decision, &risk_assessment).await?;

        // Create execution plan
        let execution_plan = self.execution_planner.create_plan(&base_decision, &risk_assessment).await?;

        // Get quantum insights if enabled
        let quantum_insights = if self.config.use_quantum {
            Some(self.quantum_decision.get_insights().await?)
        } else {
            None
        };

        // Create enhanced decision
        let enhanced_decision = EnhancedTradingDecision {
            decision: base_decision,
            quantum_insights,
            risk_assessment,
            execution_plan,
            metadata: self.create_decision_metadata(factors, analysis),
        };

        // Store in history
        self.decision_history.push(enhanced_decision.clone());

        // Update portfolio manager
        self.portfolio_manager.update_with_decision(&enhanced_decision).await?;

        Ok(enhanced_decision)
    }

    /// Make classical (non-quantum) decision
    async fn make_classical_decision(
        &self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
        strategy: &strategy_selector::StrategySelector,
    ) -> QarResult<TradingDecision> {
        // Simple rule-based decision making
        let trend_factor = factors.get_factor(&StandardFactors::Trend)?;
        let volatility_factor = factors.get_factor(&StandardFactors::Volatility)?;
        let momentum_factor = factors.get_factor(&StandardFactors::Momentum)?;

        let decision_type = match analysis.trend {
            crate::analysis::TrendDirection::Bullish if trend_factor > 0.6 => DecisionType::Buy,
            crate::analysis::TrendDirection::Bearish if trend_factor < 0.4 => DecisionType::Sell,
            _ => DecisionType::Hold,
        };

        let confidence = (trend_factor + momentum_factor + (1.0 - volatility_factor)) / 3.0;
        let urgency_score = volatility_factor;

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return: Some(trend_factor * 0.1),
            risk_assessment: Some(volatility_factor),
            urgency_score: Some(urgency_score),
            reasoning: format!("Classical analysis: trend={:.2}, volatility={:.2}, momentum={:.2}", 
                             trend_factor, volatility_factor, momentum_factor),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Create decision metadata
    fn create_decision_metadata(&self, factors: &FactorMap, analysis: &AnalysisResult) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("engine_version".to_string(), "1.0.0".to_string());
        metadata.insert("use_quantum".to_string(), self.config.use_quantum.to_string());
        metadata.insert("risk_tolerance".to_string(), self.config.risk_tolerance.to_string());
        metadata.insert("analysis_confidence".to_string(), analysis.confidence.to_string());
        metadata.insert("factor_count".to_string(), factors.factor_count().to_string());
        metadata
    }

    /// Get decision history
    pub fn get_decision_history(&self) -> &[EnhancedTradingDecision] {
        &self.decision_history
    }

    /// Get latest decision
    pub fn get_latest_decision(&self) -> Option<&EnhancedTradingDecision> {
        self.decision_history.last()
    }

    /// Clear decision history
    pub fn clear_history(&mut self) {
        self.decision_history.clear();
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if self.decision_history.is_empty() {
            return metrics;
        }

        // Calculate basic performance metrics
        let total_decisions = self.decision_history.len() as f64;
        let buy_decisions = self.decision_history.iter()
            .filter(|d| matches!(d.decision.decision_type, DecisionType::Buy))
            .count() as f64;
        let sell_decisions = self.decision_history.iter()
            .filter(|d| matches!(d.decision.decision_type, DecisionType::Sell))
            .count() as f64;
        let hold_decisions = self.decision_history.iter()
            .filter(|d| matches!(d.decision.decision_type, DecisionType::Hold))
            .count() as f64;

        let avg_confidence = self.decision_history.iter()
            .map(|d| d.decision.confidence)
            .sum::<f64>() / total_decisions;

        let avg_risk_score = self.decision_history.iter()
            .map(|d| d.risk_assessment.risk_score)
            .sum::<f64>() / total_decisions;

        metrics.insert("total_decisions".to_string(), total_decisions);
        metrics.insert("buy_ratio".to_string(), buy_decisions / total_decisions);
        metrics.insert("sell_ratio".to_string(), sell_decisions / total_decisions);
        metrics.insert("hold_ratio".to_string(), hold_decisions / total_decisions);
        metrics.insert("average_confidence".to_string(), avg_confidence);
        metrics.insert("average_risk_score".to_string(), avg_risk_score);

        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    #[tokio::test]
    async fn test_decision_engine_creation() {
        let config = DecisionConfig::default();
        let engine = DecisionEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_classical_decision_making() {
        let config = DecisionConfig {
            use_quantum: false,
            ..Default::default()
        };
        let mut engine = DecisionEngine::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.8);
        factors.insert(StandardFactors::Volatility.to_string(), 0.3);
        factors.insert(StandardFactors::Momentum.to_string(), 0.7);
        
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

        let decision = engine.make_decision(&factor_map, &analysis).await;
        assert!(decision.is_ok());
        
        let decision = decision.unwrap();
        assert!(matches!(decision.decision.decision_type, DecisionType::Buy));
        assert!(decision.decision.confidence > 0.0);
    }

    #[test]
    fn test_performance_metrics() {
        let config = DecisionConfig::default();
        let mut engine = DecisionEngine::new(config).unwrap();

        // Add some mock decisions
        let decision = EnhancedTradingDecision {
            decision: TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                expected_return: Some(0.05),
                risk_assessment: Some(0.3),
                urgency_score: Some(0.2),
                reasoning: "Test decision".to_string(),
                timestamp: chrono::Utc::now(),
            },
            quantum_insights: None,
            risk_assessment: RiskAssessment {
                risk_score: 0.3,
                var_95: 0.05,
                expected_shortfall: 0.07,
                max_drawdown_risk: 0.1,
                liquidity_risk: 0.2,
                risk_adjusted_return: 0.04,
            },
            execution_plan: ExecutionPlan {
                strategy: ExecutionStrategy::Market,
                position_size: 0.05,
                stop_loss: Some(0.95),
                take_profit: Some(1.1),
                time_horizon: std::time::Duration::from_secs(3600),
                priority: ExecutionPriority::Medium,
            },
            metadata: HashMap::new(),
        };

        engine.decision_history.push(decision);

        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.get("total_decisions"), Some(&1.0));
        assert_eq!(metrics.get("buy_ratio"), Some(&1.0));
        assert_eq!(metrics.get("average_confidence"), Some(&0.8));
    }
}