//! Strategy Selection Module
//!
//! Intelligent strategy selection using quantum-enhanced decision making and market analysis.

use crate::core::{QarResult, DecisionType, FactorMap, TradingDecision};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use super::{DecisionConfig, RiskAssessment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy selector for quantum trading decisions
pub struct StrategySelector {
    config: StrategySelectorConfig,
    strategy_registry: StrategyRegistry,
    performance_tracker: StrategyPerformanceTracker,
    selection_history: Vec<StrategySelection>,
}

/// Strategy selector configuration
#[derive(Debug, Clone)]
pub struct StrategySelectorConfig {
    /// Maximum number of strategies to consider
    pub max_strategies: usize,
    /// Performance lookback window
    pub performance_window: usize,
    /// Minimum confidence threshold for strategy selection
    pub min_confidence: f64,
    /// Risk tolerance level
    pub risk_tolerance: f64,
    /// Enable adaptive strategy weighting
    pub adaptive_weighting: bool,
}

/// Registry of available trading strategies
#[derive(Debug)]
pub struct StrategyRegistry {
    /// Available strategies
    pub strategies: HashMap<StrategyType, StrategyDefinition>,
    /// Strategy combinations
    pub combinations: Vec<StrategyCombination>,
}

/// Individual strategy definition
#[derive(Debug, Clone)]
pub struct StrategyDefinition {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Expected performance characteristics
    pub characteristics: StrategyCharacteristics,
    /// Execution parameters
    pub parameters: StrategyParameters,
    /// Market conditions where strategy performs best
    pub optimal_conditions: MarketConditions,
}

/// Strategy type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StrategyType {
    /// Quantum momentum strategy
    QuantumMomentum,
    /// Quantum mean reversion
    QuantumMeanReversion,
    /// Quantum arbitrage
    QuantumArbitrage,
    /// Quantum trend following
    QuantumTrendFollowing,
    /// Quantum volatility trading
    QuantumVolatilityTrading,
    /// Quantum pairs trading
    QuantumPairsTrading,
    /// Quantum regime switching
    QuantumRegimeSwitching,
    /// Classical momentum
    ClassicalMomentum,
    /// Classical mean reversion
    ClassicalMeanReversion,
    /// Hybrid quantum-classical
    HybridQuantumClassical,
    /// Risk parity
    RiskParity,
    /// Market neutral
    MarketNeutral,
}

/// Strategy characteristics
#[derive(Debug, Clone)]
pub struct StrategyCharacteristics {
    /// Expected return
    pub expected_return: f64,
    /// Volatility
    pub volatility: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Market correlation
    pub market_correlation: f64,
}

/// Strategy execution parameters
#[derive(Debug, Clone)]
pub struct StrategyParameters {
    /// Position sizing method
    pub position_sizing: PositionSizingMethod,
    /// Entry conditions
    pub entry_conditions: Vec<EntryCondition>,
    /// Exit conditions
    pub exit_conditions: Vec<ExitCondition>,
    /// Risk management rules
    pub risk_management: RiskManagementRules,
    /// Quantum parameters if applicable
    pub quantum_params: Option<QuantumStrategyParams>,
}

/// Position sizing method
#[derive(Debug, Clone)]
pub enum PositionSizingMethod {
    Fixed(f64),
    PercentageOfCapital(f64),
    VolatilityAdjusted(f64),
    KellyOptimal,
    RiskParity,
    QuantumOptimal,
}

/// Entry condition
#[derive(Debug, Clone)]
pub struct EntryCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Weight in final decision
    pub weight: f64,
}

/// Exit condition
#[derive(Debug, Clone)]
pub struct ExitCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Priority level
    pub priority: ExitPriority,
}

/// Condition type enumeration
#[derive(Debug, Clone)]
pub enum ConditionType {
    PriceThreshold,
    VolatilityThreshold,
    MomentumSignal,
    TrendReversal,
    QuantumCoherence,
    QuantumEntanglement,
    TechnicalIndicator(String),
    TimeBasedExit,
}

/// Exit priority
#[derive(Debug, Clone)]
pub enum ExitPriority {
    Low,
    Medium,
    High,
    Emergency,
}

/// Risk management rules
#[derive(Debug, Clone)]
pub struct RiskManagementRules {
    /// Stop loss percentage
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
    /// Maximum position size
    pub max_position_size: f64,
    /// Maximum correlation
    pub max_correlation: f64,
    /// VaR limit
    pub var_limit: Option<f64>,
}

/// Quantum strategy parameters
#[derive(Debug, Clone)]
pub struct QuantumStrategyParams {
    /// Number of qubits
    pub num_qubits: usize,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Coherence time
    pub coherence_time: f64,
    /// Entanglement strength
    pub entanglement_strength: f64,
    /// Measurement frequency
    pub measurement_frequency: f64,
}

/// Market conditions specification
#[derive(Debug, Clone)]
pub struct MarketConditions {
    /// Volatility range
    pub volatility_range: (f64, f64),
    /// Trend strength range
    pub trend_range: (f64, f64),
    /// Liquidity requirements
    pub liquidity_min: f64,
    /// Market regime preferences
    pub preferred_regimes: Vec<crate::analysis::MarketRegime>,
}

/// Strategy combination
#[derive(Debug, Clone)]
pub struct StrategyCombination {
    /// Component strategies
    pub strategies: Vec<StrategyType>,
    /// Strategy weights
    pub weights: Vec<f64>,
    /// Combination method
    pub combination_method: CombinationMethod,
    /// Expected performance
    pub expected_characteristics: StrategyCharacteristics,
}

/// Strategy combination method
#[derive(Debug, Clone)]
pub enum CombinationMethod {
    EqualWeight,
    VolatilityWeighted,
    PerformanceWeighted,
    RiskAdjusted,
    QuantumOptimal,
}

/// Strategy selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelection {
    /// Selected strategy
    pub strategy: StrategyType,
    /// Selection confidence
    pub confidence: f64,
    /// Strategy combination if applicable
    pub combination: Option<StrategyCombination>,
    /// Expected performance
    pub expected_performance: StrategyCharacteristics,
    /// Market conditions assessment
    pub market_conditions: MarketConditionAssessment,
    /// Selection reasoning
    pub reasoning: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Market condition assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditionAssessment {
    /// Current volatility regime
    pub volatility_regime: VolatilityRegime,
    /// Trend strength
    pub trend_strength: f64,
    /// Market efficiency
    pub market_efficiency: f64,
    /// Liquidity condition
    pub liquidity_condition: LiquidityCondition,
    /// Risk environment
    pub risk_environment: RiskEnvironment,
}

/// Volatility regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

/// Liquidity condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityCondition {
    Poor,
    Adequate,
    Good,
    Excellent,
}

/// Risk environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEnvironment {
    Low,
    Moderate,
    Elevated,
    High,
    Crisis,
}

/// Strategy performance tracker
#[derive(Debug)]
pub struct StrategyPerformanceTracker {
    /// Performance records
    pub performance_records: HashMap<StrategyType, Vec<PerformanceRecord>>,
    /// Current metrics
    pub current_metrics: HashMap<StrategyType, StrategyMetrics>,
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Strategy used
    pub strategy: StrategyType,
    /// Return achieved
    pub return_achieved: f64,
    /// Risk taken
    pub risk_taken: f64,
    /// Market conditions
    pub market_conditions: MarketConditionAssessment,
    /// Success indicator
    pub success: bool,
}

/// Strategy metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Volatility
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade duration
    pub avg_trade_duration: std::time::Duration,
}

impl Default for StrategySelectorConfig {
    fn default() -> Self {
        Self {
            max_strategies: 5,
            performance_window: 100,
            min_confidence: 0.6,
            risk_tolerance: 0.5,
            adaptive_weighting: true,
        }
    }
}

impl StrategySelector {
    /// Create a new strategy selector
    pub fn new(config: StrategySelectorConfig) -> QarResult<Self> {
        let strategy_registry = Self::initialize_strategy_registry();
        let performance_tracker = StrategyPerformanceTracker {
            performance_records: HashMap::new(),
            current_metrics: HashMap::new(),
        };

        Ok(Self {
            config,
            strategy_registry,
            performance_tracker,
            selection_history: Vec::new(),
        })
    }

    /// Select optimal strategy for current market conditions
    pub async fn select_strategy(
        &mut self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<StrategySelection> {
        // Assess current market conditions
        let market_conditions = self.assess_market_conditions(factors, analysis)?;

        // Evaluate all available strategies
        let strategy_scores = self.evaluate_strategies(&market_conditions, risk_assessment)?;

        // Select best strategy based on scores
        let selected_strategy = self.select_best_strategy(&strategy_scores)?;

        // Calculate confidence
        let confidence = self.calculate_selection_confidence(&strategy_scores, &selected_strategy);

        // Check if combination strategy is better
        let combination = self.evaluate_strategy_combinations(&market_conditions, &strategy_scores)?;

        // Get expected performance
        let expected_performance = self.get_expected_performance(&selected_strategy, &market_conditions)?;

        // Generate selection reasoning
        let reasoning = self.generate_selection_reasoning(&selected_strategy, &market_conditions, &strategy_scores);

        let selection = StrategySelection {
            strategy: selected_strategy,
            confidence,
            combination,
            expected_performance,
            market_conditions,
            reasoning,
            timestamp: chrono::Utc::now(),
        };

        // Store in history
        self.selection_history.push(selection.clone());
        if self.selection_history.len() > self.config.performance_window {
            self.selection_history.remove(0);
        }

        Ok(selection)
    }

    /// Initialize strategy registry with available strategies
    fn initialize_strategy_registry() -> StrategyRegistry {
        let mut strategies = HashMap::new();

        // Quantum Momentum Strategy
        strategies.insert(
            StrategyType::QuantumMomentum,
            StrategyDefinition {
                strategy_type: StrategyType::QuantumMomentum,
                name: "Quantum Momentum".to_string(),
                description: "Quantum-enhanced momentum strategy using superposition and entanglement".to_string(),
                characteristics: StrategyCharacteristics {
                    expected_return: 0.12,
                    volatility: 0.15,
                    max_drawdown: 0.08,
                    sharpe_ratio: 0.8,
                    win_rate: 0.65,
                    profit_factor: 1.8,
                    market_correlation: 0.6,
                },
                parameters: StrategyParameters {
                    position_sizing: PositionSizingMethod::QuantumOptimal,
                    entry_conditions: vec![
                        EntryCondition {
                            condition_type: ConditionType::QuantumCoherence,
                            parameters: HashMap::from([("threshold".to_string(), 0.7)]),
                            weight: 0.4,
                        },
                        EntryCondition {
                            condition_type: ConditionType::MomentumSignal,
                            parameters: HashMap::from([("lookback".to_string(), 20.0), ("threshold".to_string(), 0.02)]),
                            weight: 0.6,
                        },
                    ],
                    exit_conditions: vec![
                        ExitCondition {
                            condition_type: ConditionType::TrendReversal,
                            parameters: HashMap::from([("sensitivity".to_string(), 0.8)]),
                            priority: ExitPriority::High,
                        },
                    ],
                    risk_management: RiskManagementRules {
                        stop_loss: Some(0.05),
                        take_profit: Some(0.15),
                        max_position_size: 0.2,
                        max_correlation: 0.8,
                        var_limit: Some(0.03),
                    },
                    quantum_params: Some(QuantumStrategyParams {
                        num_qubits: 8,
                        circuit_depth: 10,
                        coherence_time: 100.0,
                        entanglement_strength: 0.8,
                        measurement_frequency: 0.1,
                    }),
                },
                optimal_conditions: MarketConditions {
                    volatility_range: (0.1, 0.3),
                    trend_range: (0.6, 1.0),
                    liquidity_min: 0.7,
                    preferred_regimes: vec![
                        crate::analysis::MarketRegime::Bull,
                        crate::analysis::MarketRegime::Transition,
                    ],
                },
            },
        );

        // Quantum Mean Reversion Strategy
        strategies.insert(
            StrategyType::QuantumMeanReversion,
            StrategyDefinition {
                strategy_type: StrategyType::QuantumMeanReversion,
                name: "Quantum Mean Reversion".to_string(),
                description: "Quantum-enhanced mean reversion using interference patterns".to_string(),
                characteristics: StrategyCharacteristics {
                    expected_return: 0.10,
                    volatility: 0.12,
                    max_drawdown: 0.06,
                    sharpe_ratio: 0.83,
                    win_rate: 0.68,
                    profit_factor: 1.9,
                    market_correlation: -0.2,
                },
                parameters: StrategyParameters {
                    position_sizing: PositionSizingMethod::VolatilityAdjusted(0.15),
                    entry_conditions: vec![
                        EntryCondition {
                            condition_type: ConditionType::QuantumEntanglement,
                            parameters: HashMap::from([("correlation_threshold".to_string(), 0.8)]),
                            weight: 0.5,
                        },
                        EntryCondition {
                            condition_type: ConditionType::PriceThreshold,
                            parameters: HashMap::from([("deviation".to_string(), 2.0)]),
                            weight: 0.5,
                        },
                    ],
                    exit_conditions: vec![
                        ExitCondition {
                            condition_type: ConditionType::PriceThreshold,
                            parameters: HashMap::from([("target".to_string(), 0.5)]),
                            priority: ExitPriority::Medium,
                        },
                    ],
                    risk_management: RiskManagementRules {
                        stop_loss: Some(0.04),
                        take_profit: Some(0.08),
                        max_position_size: 0.25,
                        max_correlation: 0.7,
                        var_limit: Some(0.025),
                    },
                    quantum_params: Some(QuantumStrategyParams {
                        num_qubits: 6,
                        circuit_depth: 8,
                        coherence_time: 80.0,
                        entanglement_strength: 0.9,
                        measurement_frequency: 0.15,
                    }),
                },
                optimal_conditions: MarketConditions {
                    volatility_range: (0.05, 0.25),
                    trend_range: (0.3, 0.7),
                    liquidity_min: 0.6,
                    preferred_regimes: vec![
                        crate::analysis::MarketRegime::Consolidation,
                        crate::analysis::MarketRegime::Bear,
                    ],
                },
            },
        );

        // Add more strategies...
        let combinations = vec![
            StrategyCombination {
                strategies: vec![StrategyType::QuantumMomentum, StrategyType::QuantumMeanReversion],
                weights: vec![0.6, 0.4],
                combination_method: CombinationMethod::QuantumOptimal,
                expected_characteristics: StrategyCharacteristics {
                    expected_return: 0.11,
                    volatility: 0.13,
                    max_drawdown: 0.07,
                    sharpe_ratio: 0.85,
                    win_rate: 0.67,
                    profit_factor: 1.85,
                    market_correlation: 0.3,
                },
            },
        ];

        StrategyRegistry {
            strategies,
            combinations,
        }
    }

    /// Assess current market conditions
    fn assess_market_conditions(
        &self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<MarketConditionAssessment> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let trend = factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let risk = factors.get_factor(&crate::core::StandardFactors::Risk)?;
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;

        let volatility_regime = match volatility {
            v if v < 0.1 => VolatilityRegime::Low,
            v if v < 0.2 => VolatilityRegime::Medium,
            v if v < 0.4 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        };

        let liquidity_condition = match liquidity {
            l if l < 0.3 => LiquidityCondition::Poor,
            l if l < 0.6 => LiquidityCondition::Adequate,
            l if l < 0.8 => LiquidityCondition::Good,
            _ => LiquidityCondition::Excellent,
        };

        let risk_environment = match risk {
            r if r < 0.2 => RiskEnvironment::Low,
            r if r < 0.4 => RiskEnvironment::Moderate,
            r if r < 0.6 => RiskEnvironment::Elevated,
            r if r < 0.8 => RiskEnvironment::High,
            _ => RiskEnvironment::Crisis,
        };

        Ok(MarketConditionAssessment {
            volatility_regime,
            trend_strength: analysis.trend_strength,
            market_efficiency: efficiency,
            liquidity_condition,
            risk_environment,
        })
    }

    /// Evaluate all strategies against current conditions
    fn evaluate_strategies(
        &self,
        conditions: &MarketConditionAssessment,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<HashMap<StrategyType, f64>> {
        let mut scores = HashMap::new();

        for (strategy_type, strategy_def) in &self.strategy_registry.strategies {
            let score = self.calculate_strategy_score(strategy_def, conditions, risk_assessment)?;
            scores.insert(strategy_type.clone(), score);
        }

        Ok(scores)
    }

    /// Calculate score for individual strategy
    fn calculate_strategy_score(
        &self,
        strategy: &StrategyDefinition,
        conditions: &MarketConditionAssessment,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<f64> {
        let mut score = 0.0;

        // Volatility suitability score
        let vol_score = match (&strategy.optimal_conditions.volatility_range, &conditions.volatility_regime) {
            ((min_vol, max_vol), VolatilityRegime::Low) => {
                let vol_val = 0.05;
                if vol_val >= *min_vol && vol_val <= *max_vol { 1.0 } else { 0.3 }
            }
            ((min_vol, max_vol), VolatilityRegime::Medium) => {
                let vol_val = 0.15;
                if vol_val >= *min_vol && vol_val <= *max_vol { 1.0 } else { 0.5 }
            }
            ((min_vol, max_vol), VolatilityRegime::High) => {
                let vol_val = 0.3;
                if vol_val >= *min_vol && vol_val <= *max_vol { 1.0 } else { 0.4 }
            }
            (_, VolatilityRegime::Extreme) => 0.2, // Most strategies perform poorly in extreme volatility
        };

        // Trend suitability score
        let trend_score = {
            let (min_trend, max_trend) = strategy.optimal_conditions.trend_range;
            if conditions.trend_strength >= min_trend && conditions.trend_strength <= max_trend {
                1.0
            } else {
                1.0 - (conditions.trend_strength - (min_trend + max_trend) / 2.0).abs()
            }
        }.max(0.0).min(1.0);

        // Liquidity suitability score
        let liquidity_score = match (&strategy.optimal_conditions.liquidity_min, &conditions.liquidity_condition) {
            (min_liq, LiquidityCondition::Poor) => if *min_liq <= 0.3 { 0.8 } else { 0.2 },
            (min_liq, LiquidityCondition::Adequate) => if *min_liq <= 0.6 { 0.9 } else { 0.4 },
            (min_liq, LiquidityCondition::Good) => if *min_liq <= 0.8 { 1.0 } else { 0.7 },
            (_, LiquidityCondition::Excellent) => 1.0,
        };

        // Risk-adjusted score
        let risk_score = if risk_assessment.risk_score <= self.config.risk_tolerance {
            1.0 - (risk_assessment.risk_score / self.config.risk_tolerance).min(1.0)
        } else {
            0.3 // Penalty for high risk
        };

        // Historical performance score
        let performance_score = self.get_historical_performance_score(strategy.strategy_type.clone());

        // Combine scores with weights
        score = vol_score * 0.25 + trend_score * 0.25 + liquidity_score * 0.15 + 
                risk_score * 0.20 + performance_score * 0.15;

        // Bonus for quantum strategies in suitable conditions
        if matches!(strategy.strategy_type, 
                   StrategyType::QuantumMomentum | 
                   StrategyType::QuantumMeanReversion | 
                   StrategyType::QuantumArbitrage |
                   StrategyType::QuantumTrendFollowing |
                   StrategyType::QuantumVolatilityTrading |
                   StrategyType::QuantumPairsTrading |
                   StrategyType::QuantumRegimeSwitching) {
            if conditions.market_efficiency > 0.7 {
                score *= 1.1; // 10% bonus for quantum strategies in efficient markets
            }
        }

        Ok(score.max(0.0).min(1.0))
    }

    /// Get historical performance score for strategy
    fn get_historical_performance_score(&self, strategy_type: StrategyType) -> f64 {
        if let Some(metrics) = self.performance_tracker.current_metrics.get(&strategy_type) {
            // Combine various performance metrics
            let sharpe_component = (metrics.sharpe_ratio / 2.0).min(1.0).max(0.0);
            let return_component = (metrics.annualized_return / 0.2).min(1.0).max(0.0);
            let drawdown_component = (1.0 - metrics.max_drawdown / 0.2).min(1.0).max(0.0);
            let win_rate_component = metrics.win_rate;

            (sharpe_component * 0.3 + return_component * 0.3 + 
             drawdown_component * 0.2 + win_rate_component * 0.2)
        } else {
            0.5 // Neutral score for strategies without history
        }
    }

    /// Select best strategy from scores
    fn select_best_strategy(&self, scores: &HashMap<StrategyType, f64>) -> QarResult<StrategyType> {
        scores.iter()
            .filter(|(_, &score)| score >= self.config.min_confidence)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(strategy, _)| strategy.clone())
            .ok_or_else(|| QarError::StrategySelection("No strategy meets minimum confidence threshold".to_string()))
    }

    /// Calculate selection confidence
    fn calculate_selection_confidence(
        &self,
        scores: &HashMap<StrategyType, f64>,
        selected: &StrategyType,
    ) -> f64 {
        if let Some(&best_score) = scores.get(selected) {
            // Calculate confidence based on separation from second-best
            let mut sorted_scores: Vec<f64> = scores.values().cloned().collect();
            sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

            if sorted_scores.len() >= 2 {
                let second_best = sorted_scores[1];
                let separation = best_score - second_best;
                (best_score + separation).min(1.0)
            } else {
                best_score
            }
        } else {
            0.5
        }
    }

    /// Evaluate strategy combinations
    fn evaluate_strategy_combinations(
        &self,
        conditions: &MarketConditionAssessment,
        strategy_scores: &HashMap<StrategyType, f64>,
    ) -> QarResult<Option<StrategyCombination>> {
        let mut best_combination: Option<StrategyCombination> = None;
        let mut best_score = 0.0;

        for combination in &self.strategy_registry.combinations {
            let mut combination_score = 0.0;
            let mut valid_combination = true;

            for (strategy, weight) in combination.strategies.iter().zip(&combination.weights) {
                if let Some(&individual_score) = strategy_scores.get(strategy) {
                    combination_score += individual_score * weight;
                } else {
                    valid_combination = false;
                    break;
                }
            }

            if valid_combination && combination_score > best_score {
                best_score = combination_score;
                best_combination = Some(combination.clone());
            }
        }

        // Only return combination if it's significantly better than individual strategies
        if let Some(combination) = &best_combination {
            let max_individual_score = strategy_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            if best_score > max_individual_score * 1.05 { // 5% improvement threshold
                Ok(Some(combination.clone()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get expected performance for selected strategy
    fn get_expected_performance(
        &self,
        strategy_type: &StrategyType,
        conditions: &MarketConditionAssessment,
    ) -> QarResult<StrategyCharacteristics> {
        if let Some(strategy_def) = self.strategy_registry.strategies.get(strategy_type) {
            let mut characteristics = strategy_def.characteristics.clone();

            // Adjust for current market conditions
            match conditions.volatility_regime {
                VolatilityRegime::Low => {
                    characteristics.volatility *= 0.8;
                    characteristics.expected_return *= 0.9;
                }
                VolatilityRegime::High => {
                    characteristics.volatility *= 1.3;
                    characteristics.expected_return *= 1.1;
                }
                VolatilityRegime::Extreme => {
                    characteristics.volatility *= 1.8;
                    characteristics.expected_return *= 0.7;
                    characteristics.max_drawdown *= 1.5;
                }
                _ => {}
            }

            // Adjust for risk environment
            match conditions.risk_environment {
                RiskEnvironment::Low => {
                    characteristics.expected_return *= 1.05;
                    characteristics.sharpe_ratio *= 1.1;
                }
                RiskEnvironment::High | RiskEnvironment::Crisis => {
                    characteristics.expected_return *= 0.8;
                    characteristics.max_drawdown *= 1.3;
                    characteristics.sharpe_ratio *= 0.7;
                }
                _ => {}
            }

            Ok(characteristics)
        } else {
            Err(QarError::StrategySelection(format!("Strategy {:?} not found in registry", strategy_type)))
        }
    }

    /// Generate selection reasoning
    fn generate_selection_reasoning(
        &self,
        strategy: &StrategyType,
        conditions: &MarketConditionAssessment,
        scores: &HashMap<StrategyType, f64>,
    ) -> String {
        let score = scores.get(strategy).unwrap_or(&0.0);
        
        format!(
            "Selected {:?} (score: {:.2}) based on current market conditions: {:?} volatility, {:.1}% trend strength, {:?} liquidity, {:?} risk environment. Strategy optimally suited for these conditions with strong historical performance.",
            strategy,
            score,
            conditions.volatility_regime,
            conditions.trend_strength * 100.0,
            conditions.liquidity_condition,
            conditions.risk_environment
        )
    }

    /// Update strategy performance
    pub fn update_performance(
        &mut self,
        strategy: StrategyType,
        return_achieved: f64,
        risk_taken: f64,
        market_conditions: MarketConditionAssessment,
        success: bool,
    ) {
        let record = PerformanceRecord {
            timestamp: chrono::Utc::now(),
            strategy: strategy.clone(),
            return_achieved,
            risk_taken,
            market_conditions,
            success,
        };

        self.performance_tracker.performance_records
            .entry(strategy.clone())
            .or_insert_with(Vec::new)
            .push(record);

        // Update current metrics
        self.update_strategy_metrics(strategy);
    }

    /// Update strategy metrics based on performance history
    fn update_strategy_metrics(&mut self, strategy: StrategyType) {
        if let Some(records) = self.performance_tracker.performance_records.get(&strategy) {
            if records.is_empty() {
                return;
            }

            let recent_records = if records.len() > self.config.performance_window {
                &records[records.len() - self.config.performance_window..]
            } else {
                records
            };

            let total_return: f64 = recent_records.iter().map(|r| r.return_achieved).sum();
            let returns: Vec<f64> = recent_records.iter().map(|r| r.return_achieved).collect();
            let mean_return = total_return / returns.len() as f64;
            
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1).max(1) as f64;
            let volatility = variance.sqrt();

            let wins = recent_records.iter().filter(|r| r.success).count();
            let win_rate = wins as f64 / recent_records.len() as f64;

            let positive_returns: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
            let negative_returns: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
            let profit_factor = if negative_returns > 0.0 {
                positive_returns / negative_returns
            } else {
                if positive_returns > 0.0 { f64::INFINITY } else { 1.0 }
            };

            // Calculate max drawdown
            let mut peak = 0.0;
            let mut max_drawdown = 0.0;
            let mut cumulative_return = 0.0;

            for record in recent_records {
                cumulative_return += record.return_achieved;
                if cumulative_return > peak {
                    peak = cumulative_return;
                }
                let drawdown = peak - cumulative_return;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }

            let sharpe_ratio = if volatility > 0.0 {
                (mean_return - 0.02 / 252.0) / volatility // Assuming 2% risk-free rate
            } else {
                0.0
            };

            let avg_duration_secs = recent_records.len() as u64 * 3600; // Simplified: 1 hour per record
            let avg_trade_duration = std::time::Duration::from_secs(avg_duration_secs);

            let metrics = StrategyMetrics {
                total_return,
                annualized_return: mean_return * 252.0, // Assuming daily returns
                volatility: volatility * (252.0_f64).sqrt(),
                sharpe_ratio,
                max_drawdown,
                win_rate,
                profit_factor,
                num_trades: recent_records.len(),
                avg_trade_duration,
            };

            self.performance_tracker.current_metrics.insert(strategy, metrics);
        }
    }

    /// Get strategy selection history
    pub fn get_selection_history(&self) -> &[StrategySelection] {
        &self.selection_history
    }

    /// Get latest selection
    pub fn get_latest_selection(&self) -> Option<&StrategySelection> {
        self.selection_history.last()
    }

    /// Get strategy performance metrics
    pub fn get_strategy_metrics(&self, strategy: &StrategyType) -> Option<&StrategyMetrics> {
        self.performance_tracker.current_metrics.get(strategy)
    }

    /// Get available strategies
    pub fn get_available_strategies(&self) -> Vec<StrategyType> {
        self.strategy_registry.strategies.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    fn create_test_factors() -> FactorMap {
        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Volatility.to_string(), 0.15);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.8);
        factors.insert(StandardFactors::Risk.to_string(), 0.3);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volume.to_string(), 0.7);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.8);
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

    fn create_test_risk_assessment() -> RiskAssessment {
        RiskAssessment {
            risk_score: 0.4,
            var_95: 0.03,
            expected_shortfall: 0.045,
            max_drawdown_risk: 0.06,
            liquidity_risk: 0.2,
            risk_adjusted_return: 0.12,
        }
    }

    #[tokio::test]
    async fn test_strategy_selector_creation() {
        let config = StrategySelectorConfig::default();
        let selector = StrategySelector::new(config);
        assert!(selector.is_ok());
    }

    #[tokio::test]
    async fn test_market_condition_assessment() {
        let config = StrategySelectorConfig::default();
        let selector = StrategySelector::new(config).unwrap();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let conditions = selector.assess_market_conditions(&factors, &analysis).unwrap();
        
        assert!(matches!(conditions.volatility_regime, VolatilityRegime::Medium));
        assert!(matches!(conditions.liquidity_condition, LiquidityCondition::Good));
        assert!(matches!(conditions.risk_environment, RiskEnvironment::Moderate));
        assert_eq!(conditions.trend_strength, 0.8);
    }

    #[tokio::test]
    async fn test_strategy_selection() {
        let config = StrategySelectorConfig::default();
        let mut selector = StrategySelector::new(config).unwrap();
        let factors = create_test_factors();
        let analysis = create_test_analysis();
        let risk_assessment = create_test_risk_assessment();

        let selection = selector.select_strategy(&factors, &analysis, &risk_assessment).await;
        assert!(selection.is_ok());

        let selection = selection.unwrap();
        assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);
        assert!(!selection.reasoning.is_empty());
    }

    #[test]
    fn test_strategy_score_calculation() {
        let config = StrategySelectorConfig::default();
        let selector = StrategySelector::new(config).unwrap();
        
        let strategy = selector.strategy_registry.strategies
            .get(&StrategyType::QuantumMomentum)
            .unwrap();
        
        let conditions = MarketConditionAssessment {
            volatility_regime: VolatilityRegime::Medium,
            trend_strength: 0.8,
            market_efficiency: 0.7,
            liquidity_condition: LiquidityCondition::Good,
            risk_environment: RiskEnvironment::Moderate,
        };
        
        let risk_assessment = create_test_risk_assessment();
        
        let score = selector.calculate_strategy_score(strategy, &conditions, &risk_assessment).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_performance_tracking() {
        let config = StrategySelectorConfig::default();
        let mut selector = StrategySelector::new(config).unwrap();
        
        let conditions = MarketConditionAssessment {
            volatility_regime: VolatilityRegime::Medium,
            trend_strength: 0.8,
            market_efficiency: 0.7,
            liquidity_condition: LiquidityCondition::Good,
            risk_environment: RiskEnvironment::Moderate,
        };

        // Add some performance records
        selector.update_performance(
            StrategyType::QuantumMomentum,
            0.05,
            0.03,
            conditions.clone(),
            true,
        );

        selector.update_performance(
            StrategyType::QuantumMomentum,
            -0.02,
            0.04,
            conditions,
            false,
        );

        let metrics = selector.get_strategy_metrics(&StrategyType::QuantumMomentum);
        assert!(metrics.is_some());
        
        let metrics = metrics.unwrap();
        assert_eq!(metrics.num_trades, 2);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_strategy_registry_initialization() {
        let registry = StrategySelector::initialize_strategy_registry();
        assert!(!registry.strategies.is_empty());
        assert!(registry.strategies.contains_key(&StrategyType::QuantumMomentum));
        assert!(registry.strategies.contains_key(&StrategyType::QuantumMeanReversion));
    }

    #[test]
    fn test_confidence_calculation() {
        let config = StrategySelectorConfig::default();
        let selector = StrategySelector::new(config).unwrap();
        
        let mut scores = HashMap::new();
        scores.insert(StrategyType::QuantumMomentum, 0.9);
        scores.insert(StrategyType::QuantumMeanReversion, 0.7);
        scores.insert(StrategyType::ClassicalMomentum, 0.6);
        
        let confidence = selector.calculate_selection_confidence(&scores, &StrategyType::QuantumMomentum);
        assert!(confidence > 0.9); // Should have high confidence due to clear winner
        
        // Test with closer scores
        scores.insert(StrategyType::QuantumMeanReversion, 0.85);
        let confidence = selector.calculate_selection_confidence(&scores, &StrategyType::QuantumMomentum);
        assert!(confidence < 1.0); // Should have lower confidence due to closer competition
    }

    #[test]
    fn test_strategy_combination_evaluation() {
        let config = StrategySelectorConfig::default();
        let selector = StrategySelector::new(config).unwrap();
        
        let mut strategy_scores = HashMap::new();
        strategy_scores.insert(StrategyType::QuantumMomentum, 0.8);
        strategy_scores.insert(StrategyType::QuantumMeanReversion, 0.7);
        
        let conditions = MarketConditionAssessment {
            volatility_regime: VolatilityRegime::Medium,
            trend_strength: 0.8,
            market_efficiency: 0.7,
            liquidity_condition: LiquidityCondition::Good,
            risk_environment: RiskEnvironment::Moderate,
        };
        
        let combination = selector.evaluate_strategy_combinations(&conditions, &strategy_scores).unwrap();
        // Since we have a combination that should score higher, it might return Some
        // The exact result depends on the scoring logic
    }
}

// Type alias for backward compatibility
pub type TradingStrategy = StrategySelector;