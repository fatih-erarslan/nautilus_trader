//! Execution Planning Module
//!
//! Advanced execution planning for quantum trading decisions with order optimization.

use crate::core::{QarResult, TradingDecision, DecisionType, FactorMap};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use super::{ExecutionPlan, ExecutionStrategy, ExecutionPriority, RiskAssessment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Execution planner for trading decisions
pub struct ExecutionPlanner {
    config: ExecutionPlannerConfig,
    market_microstructure: MarketMicrostructure,
    execution_algorithms: ExecutionAlgorithms,
    execution_history: Vec<ExecutionRecord>,
    performance_tracker: ExecutionPerformanceTracker,
}

/// Execution planner configuration
#[derive(Debug, Clone)]
pub struct ExecutionPlannerConfig {
    /// Default execution strategy
    pub default_strategy: ExecutionStrategy,
    /// Maximum slippage tolerance
    pub max_slippage: f64,
    /// Minimum order size
    pub min_order_size: f64,
    /// Maximum order size
    pub max_order_size: f64,
    /// Enable smart order routing
    pub smart_routing: bool,
    /// Enable dark pool access
    pub dark_pools: bool,
    /// Execution time limit
    pub execution_timeout: std::time::Duration,
}

/// Market microstructure information
#[derive(Debug)]
pub struct MarketMicrostructure {
    /// Current bid-ask spread
    pub bid_ask_spread: f64,
    /// Market depth information
    pub market_depth: MarketDepth,
    /// Order book imbalance
    pub order_book_imbalance: f64,
    /// Average trade size
    pub average_trade_size: f64,
    /// Trading volume profile
    pub volume_profile: VolumeProfile,
    /// Market impact model
    pub market_impact_model: MarketImpactModel,
}

/// Market depth information
#[derive(Debug)]
pub struct MarketDepth {
    /// Bid depth at various levels
    pub bid_depths: Vec<(f64, f64)>, // (price, quantity)
    /// Ask depth at various levels
    pub ask_depths: Vec<(f64, f64)>, // (price, quantity)
    /// Total bid volume
    pub total_bid_volume: f64,
    /// Total ask volume
    pub total_ask_volume: f64,
}

/// Volume profile analysis
#[derive(Debug)]
pub struct VolumeProfile {
    /// Volume Weighted Average Price
    pub vwap: f64,
    /// Time Weighted Average Price
    pub twap: f64,
    /// Participation rate
    pub participation_rate: f64,
    /// Volume distribution by time
    pub time_distribution: HashMap<String, f64>,
    /// Volume concentration
    pub volume_concentration: f64,
}

/// Market impact model
#[derive(Debug)]
pub struct MarketImpactModel {
    /// Temporary impact coefficient
    pub temporary_impact: f64,
    /// Permanent impact coefficient
    pub permanent_impact: f64,
    /// Impact decay rate
    pub decay_rate: f64,
    /// Liquidity parameter
    pub liquidity_parameter: f64,
}

/// Execution algorithms available
#[derive(Debug)]
pub struct ExecutionAlgorithms {
    /// TWAP algorithm
    pub twap: TwapAlgorithm,
    /// VWAP algorithm
    pub vwap: VwapAlgorithm,
    /// Implementation Shortfall algorithm
    pub implementation_shortfall: ImplementationShortfallAlgorithm,
    /// Participation of Volume algorithm
    pub pov: ParticipationOfVolumeAlgorithm,
    /// Quantum execution algorithm
    pub quantum_algo: QuantumExecutionAlgorithm,
}

/// TWAP (Time Weighted Average Price) algorithm
#[derive(Debug)]
pub struct TwapAlgorithm {
    /// Time slices for execution
    pub time_slices: usize,
    /// Execution interval
    pub interval: std::time::Duration,
    /// Price improvement tolerance
    pub price_tolerance: f64,
}

/// VWAP (Volume Weighted Average Price) algorithm
#[derive(Debug)]
pub struct VwapAlgorithm {
    /// Volume participation rate
    pub participation_rate: f64,
    /// Volume profile to match
    pub target_profile: Vec<f64>,
    /// Aggressiveness factor
    pub aggressiveness: f64,
}

/// Implementation Shortfall algorithm
#[derive(Debug)]
pub struct ImplementationShortfallAlgorithm {
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Urgency factor
    pub urgency: f64,
    /// Market impact sensitivity
    pub impact_sensitivity: f64,
}

/// Participation of Volume algorithm
#[derive(Debug)]
pub struct ParticipationOfVolumeAlgorithm {
    /// Target participation rate
    pub target_participation: f64,
    /// Maximum participation rate
    pub max_participation: f64,
    /// Volume forecast accuracy
    pub forecast_accuracy: f64,
}

/// Quantum execution algorithm
#[derive(Debug)]
pub struct QuantumExecutionAlgorithm {
    /// Quantum superposition orders
    pub superposition_orders: bool,
    /// Quantum entangled routing
    pub entangled_routing: bool,
    /// Coherence preservation time
    pub coherence_time: std::time::Duration,
    /// Quantum advantage threshold
    pub advantage_threshold: f64,
}

/// Enhanced execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedExecutionPlan {
    /// Basic execution plan
    pub basic_plan: ExecutionPlan,
    /// Detailed execution schedule
    pub execution_schedule: ExecutionSchedule,
    /// Order splitting strategy
    pub order_splitting: OrderSplittingStrategy,
    /// Market timing analysis
    pub market_timing: MarketTimingAnalysis,
    /// Cost analysis
    pub cost_analysis: ExecutionCostAnalysis,
    /// Risk considerations
    pub execution_risks: ExecutionRiskAnalysis,
    /// Performance expectations
    pub performance_expectations: PerformanceExpectations,
}

/// Execution schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSchedule {
    /// Execution phases
    pub phases: Vec<ExecutionPhase>,
    /// Total estimated duration
    pub estimated_duration: std::time::Duration,
    /// Earliest start time
    pub earliest_start: chrono::DateTime<chrono::Utc>,
    /// Latest completion time
    pub latest_completion: chrono::DateTime<chrono::Utc>,
}

/// Individual execution phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPhase {
    /// Phase identifier
    pub phase_id: String,
    /// Order quantity for this phase
    pub quantity: f64,
    /// Target execution time
    pub target_time: chrono::DateTime<chrono::Utc>,
    /// Execution strategy for this phase
    pub strategy: ExecutionStrategy,
    /// Expected market conditions
    pub expected_conditions: MarketConditionExpectation,
}

/// Market condition expectation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditionExpectation {
    /// Expected volatility
    pub volatility: f64,
    /// Expected liquidity
    pub liquidity: f64,
    /// Expected spread
    pub spread: f64,
    /// Expected volume
    pub volume: f64,
}

/// Order splitting strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSplittingStrategy {
    /// Splitting method
    pub method: SplittingMethod,
    /// Number of child orders
    pub num_orders: usize,
    /// Size distribution
    pub size_distribution: Vec<f64>,
    /// Timing distribution
    pub timing_distribution: Vec<std::time::Duration>,
    /// Randomization factor
    pub randomization: f64,
}

/// Splitting method enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplittingMethod {
    EqualSize,
    VolumeProportional,
    LiquidityAdaptive,
    QuantumOptimal,
    TimeWeighted,
    VolatilityAdjusted,
}

/// Market timing analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTimingAnalysis {
    /// Optimal execution windows
    pub optimal_windows: Vec<TimeWindow>,
    /// Liquidity forecast
    pub liquidity_forecast: LiquidityForecast,
    /// Volatility forecast
    pub volatility_forecast: VolatilityForecast,
    /// News impact assessment
    pub news_impact: NewsImpactAssessment,
}

/// Time window for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Window start time
    pub start: chrono::DateTime<chrono::Utc>,
    /// Window end time
    pub end: chrono::DateTime<chrono::Utc>,
    /// Expected market quality
    pub market_quality: f64,
    /// Execution priority for this window
    pub priority: WindowPriority,
}

/// Window priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowPriority {
    Optimal,
    Good,
    Acceptable,
    Poor,
    Avoid,
}

/// Liquidity forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityForecast {
    /// Hourly liquidity predictions
    pub hourly_predictions: Vec<f64>,
    /// Peak liquidity times
    pub peak_times: Vec<chrono::DateTime<chrono::Utc>>,
    /// Minimum liquidity times
    pub low_times: Vec<chrono::DateTime<chrono::Utc>>,
    /// Confidence in forecast
    pub confidence: f64,
}

/// Volatility forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityForecast {
    /// Hourly volatility predictions
    pub hourly_predictions: Vec<f64>,
    /// Volatility regime
    pub regime: VolatilityRegime,
    /// Expected volatility spikes
    pub spike_times: Vec<chrono::DateTime<chrono::Utc>>,
    /// Forecast accuracy
    pub accuracy: f64,
}

/// Volatility regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

/// News impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsImpactAssessment {
    /// Scheduled announcements
    pub scheduled_events: Vec<ScheduledEvent>,
    /// Expected impact magnitude
    pub impact_magnitude: f64,
    /// Impact direction
    pub impact_direction: ImpactDirection,
    /// Time to impact
    pub time_to_impact: Option<std::time::Duration>,
}

/// Scheduled market event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// Event description
    pub description: String,
    /// Event time
    pub time: chrono::DateTime<chrono::Utc>,
    /// Expected impact level
    pub impact_level: ImpactLevel,
    /// Market sectors affected
    pub affected_sectors: Vec<String>,
}

/// Impact level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Impact direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactDirection {
    Positive,
    Negative,
    Neutral,
    Unknown,
}

/// Execution cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCostAnalysis {
    /// Expected market impact cost
    pub market_impact_cost: f64,
    /// Expected timing cost
    pub timing_cost: f64,
    /// Expected opportunity cost
    pub opportunity_cost: f64,
    /// Commission and fees
    pub commission_fees: f64,
    /// Total execution cost
    pub total_cost: f64,
    /// Cost uncertainty
    pub cost_uncertainty: f64,
}

/// Execution risk analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRiskAnalysis {
    /// Slippage risk
    pub slippage_risk: f64,
    /// Timing risk
    pub timing_risk: f64,
    /// Liquidity risk
    pub liquidity_risk: f64,
    /// Market impact risk
    pub market_impact_risk: f64,
    /// Operational risk
    pub operational_risk: f64,
    /// Overall execution risk
    pub overall_risk: f64,
}

/// Performance expectations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    /// Expected fill rate
    pub expected_fill_rate: f64,
    /// Expected execution time
    pub expected_duration: std::time::Duration,
    /// Expected slippage
    pub expected_slippage: f64,
    /// Expected market impact
    pub expected_impact: f64,
    /// Probability of completion
    pub completion_probability: f64,
}

/// Execution record
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Original decision
    pub decision: TradingDecision,
    /// Execution plan used
    pub plan: ExecutionPlan,
    /// Actual performance
    pub performance: ExecutionPerformance,
    /// Market conditions during execution
    pub market_conditions: MarketConditionsSnapshot,
}

/// Execution performance metrics
#[derive(Debug, Clone)]
pub struct ExecutionPerformance {
    /// Actual fill rate
    pub fill_rate: f64,
    /// Actual execution time
    pub execution_time: std::time::Duration,
    /// Actual slippage
    pub slippage: f64,
    /// Actual market impact
    pub market_impact: f64,
    /// Total execution cost
    pub total_cost: f64,
    /// Success indicator
    pub success: bool,
}

/// Market conditions snapshot
#[derive(Debug, Clone)]
pub struct MarketConditionsSnapshot {
    /// Volatility during execution
    pub volatility: f64,
    /// Liquidity during execution
    pub liquidity: f64,
    /// Average spread
    pub average_spread: f64,
    /// Volume during execution
    pub volume: f64,
    /// Price movement
    pub price_movement: f64,
}

/// Execution performance tracker
#[derive(Debug)]
pub struct ExecutionPerformanceTracker {
    /// Performance by strategy
    pub strategy_performance: HashMap<ExecutionStrategy, Vec<ExecutionPerformance>>,
    /// Performance by market conditions
    pub condition_performance: HashMap<String, Vec<ExecutionPerformance>>,
    /// Overall metrics
    pub overall_metrics: OverallExecutionMetrics,
}

/// Overall execution metrics
#[derive(Debug)]
pub struct OverallExecutionMetrics {
    /// Average fill rate
    pub avg_fill_rate: f64,
    /// Average slippage
    pub avg_slippage: f64,
    /// Average execution time
    pub avg_execution_time: std::time::Duration,
    /// Success rate
    pub success_rate: f64,
    /// Total cost savings
    pub cost_savings: f64,
}

impl Default for ExecutionPlannerConfig {
    fn default() -> Self {
        Self {
            default_strategy: ExecutionStrategy::TWAP,
            max_slippage: 0.01,
            min_order_size: 100.0,
            max_order_size: 1_000_000.0,
            smart_routing: true,
            dark_pools: true,
            execution_timeout: std::time::Duration::from_secs(3600),
        }
    }
}

impl ExecutionPlanner {
    /// Create a new execution planner
    pub fn new(config: ExecutionPlannerConfig) -> QarResult<Self> {
        let market_microstructure = MarketMicrostructure {
            bid_ask_spread: 0.001,
            market_depth: MarketDepth {
                bid_depths: Vec::new(),
                ask_depths: Vec::new(),
                total_bid_volume: 0.0,
                total_ask_volume: 0.0,
            },
            order_book_imbalance: 0.0,
            average_trade_size: 1000.0,
            volume_profile: VolumeProfile {
                vwap: 0.0,
                twap: 0.0,
                participation_rate: 0.1,
                time_distribution: HashMap::new(),
                volume_concentration: 0.5,
            },
            market_impact_model: MarketImpactModel {
                temporary_impact: 0.001,
                permanent_impact: 0.0005,
                decay_rate: 0.1,
                liquidity_parameter: 1000000.0,
            },
        };

        let execution_algorithms = ExecutionAlgorithms {
            twap: TwapAlgorithm {
                time_slices: 10,
                interval: std::time::Duration::from_secs(60),
                price_tolerance: 0.001,
            },
            vwap: VwapAlgorithm {
                participation_rate: 0.1,
                target_profile: vec![0.1; 10],
                aggressiveness: 0.5,
            },
            implementation_shortfall: ImplementationShortfallAlgorithm {
                risk_aversion: 0.5,
                urgency: 0.3,
                impact_sensitivity: 1.0,
            },
            pov: ParticipationOfVolumeAlgorithm {
                target_participation: 0.1,
                max_participation: 0.2,
                forecast_accuracy: 0.8,
            },
            quantum_algo: QuantumExecutionAlgorithm {
                superposition_orders: true,
                entangled_routing: true,
                coherence_time: std::time::Duration::from_secs(300),
                advantage_threshold: 0.05,
            },
        };

        let performance_tracker = ExecutionPerformanceTracker {
            strategy_performance: HashMap::new(),
            condition_performance: HashMap::new(),
            overall_metrics: OverallExecutionMetrics {
                avg_fill_rate: 0.95,
                avg_slippage: 0.001,
                avg_execution_time: std::time::Duration::from_secs(300),
                success_rate: 0.9,
                cost_savings: 0.0,
            },
        };

        Ok(Self {
            config,
            market_microstructure,
            execution_algorithms,
            execution_history: Vec::new(),
            performance_tracker,
        })
    }

    /// Create enhanced execution plan
    pub async fn create_execution_plan(
        &mut self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<EnhancedExecutionPlan> {
        // Create basic execution plan
        let basic_plan = self.create_basic_plan(decision, risk_assessment, factors)?;

        // Analyze market microstructure
        self.update_market_microstructure(factors).await?;

        // Create detailed execution schedule
        let execution_schedule = self.create_execution_schedule(decision, &basic_plan, factors)?;

        // Determine order splitting strategy
        let order_splitting = self.determine_order_splitting(decision, factors)?;

        // Analyze market timing
        let market_timing = self.analyze_market_timing(decision, factors).await?;

        // Calculate execution costs
        let cost_analysis = self.calculate_execution_costs(decision, &basic_plan, factors)?;

        // Assess execution risks
        let execution_risks = self.assess_execution_risks(decision, &basic_plan, factors)?;

        // Set performance expectations
        let performance_expectations = self.set_performance_expectations(decision, &basic_plan)?;

        Ok(EnhancedExecutionPlan {
            basic_plan,
            execution_schedule,
            order_splitting,
            market_timing,
            cost_analysis,
            execution_risks,
            performance_expectations,
        })
    }

    /// Create basic execution plan
    fn create_basic_plan(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
        factors: &FactorMap,
    ) -> QarResult<ExecutionPlan> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let urgency = decision.urgency_score.unwrap_or(0.5);

        // Select execution strategy based on conditions
        let strategy = self.select_execution_strategy(decision, factors, risk_assessment)?;

        // Calculate position size
        let position_size = self.calculate_position_size(decision, risk_assessment)?;

        // Determine stop loss and take profit
        let (stop_loss, take_profit) = self.calculate_stop_take_levels(decision, risk_assessment)?;

        // Set time horizon based on urgency and market conditions
        let time_horizon = self.calculate_time_horizon(urgency, volatility, liquidity);

        // Set execution priority
        let priority = self.determine_execution_priority(decision, risk_assessment);

        Ok(ExecutionPlan {
            strategy,
            position_size,
            stop_loss,
            take_profit,
            time_horizon,
            priority,
        })
    }

    /// Select optimal execution strategy
    fn select_execution_strategy(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<ExecutionStrategy> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let urgency = decision.urgency_score.unwrap_or(0.5);

        // Strategy selection logic
        let strategy = if urgency > 0.8 || risk_assessment.risk_score > 0.7 {
            ExecutionStrategy::Market // Immediate execution for urgent/risky decisions
        } else if volatility > 0.6 {
            ExecutionStrategy::ImplementationShortfall // Optimal for volatile markets
        } else if liquidity < 0.4 {
            ExecutionStrategy::TWAP // Conservative for illiquid markets
        } else if self.should_use_quantum_strategy(factors)? {
            ExecutionStrategy::Quantum // Use quantum advantages
        } else {
            ExecutionStrategy::VWAP // Default for normal conditions
        };

        Ok(strategy)
    }

    /// Determine if quantum execution strategy should be used
    fn should_use_quantum_strategy(&self, factors: &FactorMap) -> QarResult<bool> {
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        // Use quantum strategy when market is efficient but has moderate volatility
        Ok(efficiency > 0.7 && volatility > 0.2 && volatility < 0.5)
    }

    /// Calculate optimal position size
    fn calculate_position_size(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<f64> {
        let base_size = self.config.max_order_size * 0.1; // Start with 10% of max
        let confidence_adjustment = decision.confidence;
        let risk_adjustment = 1.0 - risk_assessment.risk_score;

        let adjusted_size = base_size * confidence_adjustment * risk_adjustment;
        
        Ok(adjusted_size.max(self.config.min_order_size).min(self.config.max_order_size))
    }

    /// Calculate stop loss and take profit levels
    fn calculate_stop_take_levels(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> QarResult<(Option<f64>, Option<f64>)> {
        let base_stop = 0.02; // 2% default stop loss
        let base_take = 0.06; // 6% default take profit

        // Adjust based on risk assessment
        let risk_multiplier = 1.0 + risk_assessment.risk_score;
        let stop_loss = base_stop * risk_multiplier;
        
        // Adjust take profit based on expected return
        let take_profit = if let Some(expected_return) = decision.expected_return {
            (expected_return * 2.0).max(base_take) // At least 2x expected return
        } else {
            base_take
        };

        Ok((Some(stop_loss), Some(take_profit)))
    }

    /// Calculate execution time horizon
    fn calculate_time_horizon(
        &self,
        urgency: f64,
        volatility: f64,
        liquidity: f64,
    ) -> std::time::Duration {
        let base_time = 3600; // 1 hour base
        
        // Reduce time for high urgency
        let urgency_factor = 1.0 - urgency * 0.8;
        
        // Increase time for high volatility (need more careful execution)
        let volatility_factor = 1.0 + volatility * 0.5;
        
        // Increase time for low liquidity
        let liquidity_factor = 2.0 - liquidity;
        
        let adjusted_time = (base_time as f64 * urgency_factor * volatility_factor * liquidity_factor) as u64;
        
        std::time::Duration::from_secs(adjusted_time.max(300).min(14400)) // 5 min to 4 hours
    }

    /// Determine execution priority
    fn determine_execution_priority(
        &self,
        decision: &TradingDecision,
        risk_assessment: &RiskAssessment,
    ) -> ExecutionPriority {
        let urgency = decision.urgency_score.unwrap_or(0.5);
        let risk = risk_assessment.risk_score;

        match (urgency, risk) {
            (u, r) if u > 0.8 || r > 0.8 => ExecutionPriority::High,
            (u, r) if u > 0.6 || r > 0.6 => ExecutionPriority::Medium,
            _ => ExecutionPriority::Low,
        }
    }

    /// Update market microstructure data
    async fn update_market_microstructure(&mut self, factors: &FactorMap) -> QarResult<()> {
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let volume = factors.get_factor(&crate::core::StandardFactors::Volume)?;

        // Update spread based on liquidity
        self.market_microstructure.bid_ask_spread = (1.0 - liquidity) * 0.01;

        // Update market depth (simplified)
        self.market_microstructure.market_depth.total_bid_volume = volume * 10000.0;
        self.market_microstructure.market_depth.total_ask_volume = volume * 10000.0;

        // Update order book imbalance
        self.market_microstructure.order_book_imbalance = (0.5 - liquidity) * 2.0;

        // Update impact model
        self.market_microstructure.market_impact_model.temporary_impact = volatility * 0.002;
        self.market_microstructure.market_impact_model.permanent_impact = volatility * 0.001;

        Ok(())
    }

    /// Create detailed execution schedule
    fn create_execution_schedule(
        &self,
        decision: &TradingDecision,
        plan: &ExecutionPlan,
        factors: &FactorMap,
    ) -> QarResult<ExecutionSchedule> {
        let total_duration = plan.time_horizon;
        let num_phases = match plan.strategy {
            ExecutionStrategy::Market => 1,
            ExecutionStrategy::TWAP => self.execution_algorithms.twap.time_slices,
            ExecutionStrategy::VWAP => 10,
            ExecutionStrategy::Limit => 5,
            ExecutionStrategy::ImplementationShortfall => 8,
            ExecutionStrategy::Quantum => 6,
        };

        let phase_duration = total_duration / num_phases as u32;
        let mut phases = Vec::new();
        let now = chrono::Utc::now();

        for i in 0..num_phases {
            let phase_start = now + phase_duration * i as u32;
            let phase_quantity = plan.position_size / num_phases as f64;

            phases.push(ExecutionPhase {
                phase_id: format!("phase_{}", i + 1),
                quantity: phase_quantity,
                target_time: phase_start,
                strategy: plan.strategy.clone(),
                expected_conditions: self.forecast_market_conditions(phase_start, factors)?,
            });
        }

        Ok(ExecutionSchedule {
            phases,
            estimated_duration: total_duration,
            earliest_start: now,
            latest_completion: now + total_duration,
        })
    }

    /// Forecast market conditions for a specific time
    fn forecast_market_conditions(
        &self,
        target_time: chrono::DateTime<chrono::Utc>,
        factors: &FactorMap,
    ) -> QarResult<MarketConditionExpectation> {
        let base_volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let base_liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let base_volume = factors.get_factor(&crate::core::StandardFactors::Volume)?;

        // Simple time-based adjustments (in real implementation, would use sophisticated models)
        let hour = target_time.hour();
        let time_factor = match hour {
            9..=11 => 1.2,  // Market open, higher activity
            12..=13 => 0.8, // Lunch, lower activity
            14..=16 => 1.1, // Afternoon, moderate activity
            _ => 0.6,       // After hours, much lower activity
        };

        Ok(MarketConditionExpectation {
            volatility: base_volatility * time_factor,
            liquidity: base_liquidity * time_factor,
            spread: self.market_microstructure.bid_ask_spread / time_factor,
            volume: base_volume * time_factor,
        })
    }

    /// Determine order splitting strategy
    fn determine_order_splitting(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<OrderSplittingStrategy> {
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;

        let method = if liquidity < 0.3 {
            SplittingMethod::LiquidityAdaptive
        } else if volatility > 0.6 {
            SplittingMethod::VolatilityAdjusted
        } else if self.should_use_quantum_strategy(factors)? {
            SplittingMethod::QuantumOptimal
        } else {
            SplittingMethod::VolumeProportional
        };

        let num_orders = match liquidity {
            l if l < 0.3 => 15, // More orders for illiquid markets
            l if l < 0.6 => 10,
            _ => 5,
        };

        // Create size distribution based on method
        let size_distribution = self.create_size_distribution(&method, num_orders);
        let timing_distribution = self.create_timing_distribution(num_orders);

        Ok(OrderSplittingStrategy {
            method,
            num_orders,
            size_distribution,
            timing_distribution,
            randomization: 0.1, // 10% randomization to avoid detection
        })
    }

    /// Create size distribution for order splitting
    fn create_size_distribution(&self, method: &SplittingMethod, num_orders: usize) -> Vec<f64> {
        match method {
            SplittingMethod::EqualSize => vec![1.0 / num_orders as f64; num_orders],
            SplittingMethod::VolumeProportional => {
                // Front-loaded distribution
                let mut distribution = Vec::new();
                for i in 0..num_orders {
                    let weight = 2.0 * (num_orders - i) as f64 / (num_orders * (num_orders + 1)) as f64;
                    distribution.push(weight);
                }
                distribution
            }
            SplittingMethod::LiquidityAdaptive => {
                // Smaller orders initially, larger as liquidity improves
                let mut distribution = Vec::new();
                for i in 0..num_orders {
                    let weight = (i + 1) as f64 / (num_orders * (num_orders + 1) / 2) as f64;
                    distribution.push(weight);
                }
                distribution
            }
            _ => vec![1.0 / num_orders as f64; num_orders], // Default to equal
        }
    }

    /// Create timing distribution for order splitting
    fn create_timing_distribution(&self, num_orders: usize) -> Vec<std::time::Duration> {
        let total_time = 3600; // 1 hour
        let interval = total_time / num_orders;
        
        (0..num_orders)
            .map(|i| std::time::Duration::from_secs((i * interval) as u64))
            .collect()
    }

    /// Analyze market timing opportunities
    async fn analyze_market_timing(
        &self,
        decision: &TradingDecision,
        factors: &FactorMap,
    ) -> QarResult<MarketTimingAnalysis> {
        let optimal_windows = self.identify_optimal_windows(factors).await?;
        let liquidity_forecast = self.forecast_liquidity(factors)?;
        let volatility_forecast = self.forecast_volatility(factors)?;
        let news_impact = self.assess_news_impact(factors)?;

        Ok(MarketTimingAnalysis {
            optimal_windows,
            liquidity_forecast,
            volatility_forecast,
            news_impact,
        })
    }

    /// Identify optimal execution windows
    async fn identify_optimal_windows(&self, factors: &FactorMap) -> QarResult<Vec<TimeWindow>> {
        let mut windows = Vec::new();
        let now = chrono::Utc::now();
        
        // Generate windows for next 24 hours
        for hour in 0..24 {
            let window_start = now + chrono::Duration::hours(hour);
            let window_end = window_start + chrono::Duration::hours(1);
            
            let market_quality = self.calculate_market_quality_for_time(window_start, factors)?;
            let priority = match market_quality {
                q if q > 0.8 => WindowPriority::Optimal,
                q if q > 0.6 => WindowPriority::Good,
                q if q > 0.4 => WindowPriority::Acceptable,
                q if q > 0.2 => WindowPriority::Poor,
                _ => WindowPriority::Avoid,
            };

            windows.push(TimeWindow {
                start: window_start,
                end: window_end,
                market_quality,
                priority,
            });
        }

        Ok(windows)
    }

    /// Calculate market quality for specific time
    fn calculate_market_quality_for_time(
        &self,
        time: chrono::DateTime<chrono::Utc>,
        factors: &FactorMap,
    ) -> QarResult<f64> {
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;

        // Time-based adjustments
        let hour = time.hour();
        let time_factor = match hour {
            9..=11 => 0.9,  // Market open, good but volatile
            11..=12 => 1.0, // Pre-lunch, optimal
            12..=13 => 0.6, // Lunch, poor
            13..=15 => 0.9, // Afternoon, good
            15..=16 => 0.8, // Near close, moderate
            _ => 0.3,       // After hours, poor
        };

        let quality = (liquidity * 0.4 + (1.0 - volatility) * 0.3 + efficiency * 0.3) * time_factor;
        Ok(quality.max(0.0).min(1.0))
    }

    /// Forecast liquidity
    fn forecast_liquidity(&self, factors: &FactorMap) -> QarResult<LiquidityForecast> {
        let base_liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        
        let mut hourly_predictions = Vec::new();
        let mut peak_times = Vec::new();
        let mut low_times = Vec::new();

        let now = chrono::Utc::now();
        
        for hour in 0..24 {
            let time = now + chrono::Duration::hours(hour);
            let hour_of_day = time.hour();
            
            let time_multiplier = match hour_of_day {
                9..=11 => 1.2,
                11..=12 => 1.3,
                12..=13 => 0.6,
                13..=15 => 1.1,
                15..=16 => 1.0,
                _ => 0.4,
            };
            
            let predicted_liquidity = base_liquidity * time_multiplier;
            hourly_predictions.push(predicted_liquidity);
            
            if predicted_liquidity > base_liquidity * 1.2 {
                peak_times.push(time);
            } else if predicted_liquidity < base_liquidity * 0.7 {
                low_times.push(time);
            }
        }

        Ok(LiquidityForecast {
            hourly_predictions,
            peak_times,
            low_times,
            confidence: 0.8,
        })
    }

    /// Forecast volatility
    fn forecast_volatility(&self, factors: &FactorMap) -> QarResult<VolatilityForecast> {
        let base_volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        
        let regime = match base_volatility {
            v if v < 0.1 => VolatilityRegime::Low,
            v if v < 0.2 => VolatilityRegime::Normal,
            v if v < 0.4 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        };

        let mut hourly_predictions = Vec::new();
        let mut spike_times = Vec::new();
        let now = chrono::Utc::now();

        for hour in 0..24 {
            let time = now + chrono::Duration::hours(hour);
            let hour_of_day = time.hour();
            
            let time_multiplier = match hour_of_day {
                9..=10 => 1.5,  // Market open volatility
                15..=16 => 1.3, // Near close volatility
                12..=13 => 0.8, // Lunch calm
                _ => 1.0,
            };
            
            let predicted_volatility = base_volatility * time_multiplier;
            hourly_predictions.push(predicted_volatility);
            
            if predicted_volatility > base_volatility * 1.4 {
                spike_times.push(time);
            }
        }

        Ok(VolatilityForecast {
            hourly_predictions,
            regime,
            spike_times,
            accuracy: 0.7,
        })
    }

    /// Assess news impact
    fn assess_news_impact(&self, factors: &FactorMap) -> QarResult<NewsImpactAssessment> {
        // Simplified news impact assessment
        let sentiment = factors.get_factor(&crate::core::StandardFactors::Sentiment)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;

        let impact_magnitude = if volatility > 0.5 { 0.8 } else { 0.3 };
        let impact_direction = if sentiment > 0.6 {
            ImpactDirection::Positive
        } else if sentiment < 0.4 {
            ImpactDirection::Negative
        } else {
            ImpactDirection::Neutral
        };

        // Mock scheduled events (in real implementation, would fetch from news feeds)
        let scheduled_events = vec![
            ScheduledEvent {
                description: "Economic Data Release".to_string(),
                time: chrono::Utc::now() + chrono::Duration::hours(2),
                impact_level: ImpactLevel::Medium,
                affected_sectors: vec!["Financial".to_string()],
            },
        ];

        Ok(NewsImpactAssessment {
            scheduled_events,
            impact_magnitude,
            impact_direction,
            time_to_impact: Some(std::time::Duration::from_secs(7200)), // 2 hours
        })
    }

    /// Calculate execution costs
    fn calculate_execution_costs(
        &self,
        decision: &TradingDecision,
        plan: &ExecutionPlan,
        factors: &FactorMap,
    ) -> QarResult<ExecutionCostAnalysis> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;

        // Market impact cost
        let market_impact_cost = self.calculate_market_impact_cost(plan.position_size, volatility, liquidity);

        // Timing cost based on urgency
        let urgency = decision.urgency_score.unwrap_or(0.5);
        let timing_cost = urgency * 0.005; // Higher urgency = higher timing cost

        // Opportunity cost
        let expected_return = decision.expected_return.unwrap_or(0.0);
        let opportunity_cost = expected_return * 0.1; // 10% of expected return

        // Commission and fees (simplified)
        let commission_fees = plan.position_size * 0.0001; // 1 basis point

        let total_cost = market_impact_cost + timing_cost + opportunity_cost + commission_fees;
        let cost_uncertainty = volatility * 0.5; // Higher volatility = higher uncertainty

        Ok(ExecutionCostAnalysis {
            market_impact_cost,
            timing_cost,
            opportunity_cost,
            commission_fees,
            total_cost,
            cost_uncertainty,
        })
    }

    /// Calculate market impact cost
    fn calculate_market_impact_cost(&self, position_size: f64, volatility: f64, liquidity: f64) -> f64 {
        let impact_model = &self.market_microstructure.market_impact_model;
        
        // Simplified market impact calculation
        let temporary_impact = impact_model.temporary_impact * (position_size / 1000000.0).sqrt() * volatility;
        let permanent_impact = impact_model.permanent_impact * (position_size / 1000000.0) * (1.0 - liquidity);
        
        temporary_impact + permanent_impact
    }

    /// Assess execution risks
    fn assess_execution_risks(
        &self,
        decision: &TradingDecision,
        plan: &ExecutionPlan,
        factors: &FactorMap,
    ) -> QarResult<ExecutionRiskAnalysis> {
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;

        let slippage_risk = volatility * 0.5 + (1.0 - liquidity) * 0.3;
        let timing_risk = decision.urgency_score.unwrap_or(0.5) * 0.4;
        let liquidity_risk = 1.0 - liquidity;
        let market_impact_risk = (plan.position_size / 1000000.0) * volatility;
        let operational_risk = 0.1; // Fixed operational risk

        let overall_risk = (slippage_risk + timing_risk + liquidity_risk + market_impact_risk + operational_risk) / 5.0;

        Ok(ExecutionRiskAnalysis {
            slippage_risk,
            timing_risk,
            liquidity_risk,
            market_impact_risk,
            operational_risk,
            overall_risk,
        })
    }

    /// Set performance expectations
    fn set_performance_expectations(
        &self,
        decision: &TradingDecision,
        plan: &ExecutionPlan,
    ) -> QarResult<PerformanceExpectations> {
        let base_fill_rate = match plan.strategy {
            ExecutionStrategy::Market => 0.99,
            ExecutionStrategy::Limit => 0.85,
            ExecutionStrategy::TWAP => 0.95,
            ExecutionStrategy::VWAP => 0.93,
            ExecutionStrategy::ImplementationShortfall => 0.92,
            ExecutionStrategy::Quantum => 0.96,
        };

        let expected_fill_rate = base_fill_rate * decision.confidence;
        let expected_duration = plan.time_horizon;
        let expected_slippage = 0.001 * (1.0 + decision.urgency_score.unwrap_or(0.5));
        let expected_impact = 0.0005; // 0.5 basis points
        let completion_probability = expected_fill_rate * 0.95;

        Ok(PerformanceExpectations {
            expected_fill_rate,
            expected_duration,
            expected_slippage,
            expected_impact,
            completion_probability,
        })
    }

    /// Record execution performance
    pub fn record_execution(
        &mut self,
        decision: TradingDecision,
        plan: ExecutionPlan,
        performance: ExecutionPerformance,
        market_conditions: MarketConditionsSnapshot,
    ) {
        let record = ExecutionRecord {
            timestamp: chrono::Utc::now(),
            decision,
            plan: plan.clone(),
            performance: performance.clone(),
            market_conditions,
        };

        self.execution_history.push(record);

        // Update performance tracking
        self.performance_tracker.strategy_performance
            .entry(plan.strategy)
            .or_insert_with(Vec::new)
            .push(performance);

        // Update overall metrics
        self.update_overall_metrics();

        // Maintain history size
        if self.execution_history.len() > 1000 {
            self.execution_history.remove(0);
        }
    }

    /// Update overall performance metrics
    fn update_overall_metrics(&mut self) {
        let all_performances: Vec<&ExecutionPerformance> = self.performance_tracker
            .strategy_performance
            .values()
            .flatten()
            .collect();

        if all_performances.is_empty() {
            return;
        }

        let avg_fill_rate = all_performances.iter().map(|p| p.fill_rate).sum::<f64>() / all_performances.len() as f64;
        let avg_slippage = all_performances.iter().map(|p| p.slippage).sum::<f64>() / all_performances.len() as f64;
        
        let avg_execution_time_secs = all_performances.iter()
            .map(|p| p.execution_time.as_secs())
            .sum::<u64>() / all_performances.len() as u64;
        let avg_execution_time = std::time::Duration::from_secs(avg_execution_time_secs);

        let success_rate = all_performances.iter().filter(|p| p.success).count() as f64 / all_performances.len() as f64;

        self.performance_tracker.overall_metrics = OverallExecutionMetrics {
            avg_fill_rate,
            avg_slippage,
            avg_execution_time,
            success_rate,
            cost_savings: 0.0, // Would be calculated based on benchmark comparison
        };
    }

    /// Get execution history
    pub fn get_execution_history(&self) -> &[ExecutionRecord] {
        &self.execution_history
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &OverallExecutionMetrics {
        &self.performance_tracker.overall_metrics
    }

    /// Get strategy performance
    pub fn get_strategy_performance(&self, strategy: &ExecutionStrategy) -> Option<&Vec<ExecutionPerformance>> {
        self.performance_tracker.strategy_performance.get(strategy)
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
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        factors.insert(StandardFactors::Risk.to_string(), 0.4);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.7);
        FactorMap::new(factors).unwrap()
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
    async fn test_execution_planner_creation() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config);
        assert!(planner.is_ok());
    }

    #[tokio::test]
    async fn test_execution_plan_creation() {
        let config = ExecutionPlannerConfig::default();
        let mut planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let risk_assessment = create_test_risk_assessment();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let plan = planner.create_execution_plan(&decision, &risk_assessment, &factors, &analysis).await;
        assert!(plan.is_ok());

        let plan = plan.unwrap();
        assert!(plan.basic_plan.position_size > 0.0);
        assert!(!plan.execution_schedule.phases.is_empty());
        assert!(plan.performance_expectations.expected_fill_rate > 0.0);
    }

    #[test]
    fn test_strategy_selection() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();
        let risk_assessment = create_test_risk_assessment();

        let strategy = planner.select_execution_strategy(&decision, &factors, &risk_assessment);
        assert!(strategy.is_ok());
    }

    #[test]
    fn test_position_size_calculation() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let risk_assessment = create_test_risk_assessment();

        let position_size = planner.calculate_position_size(&decision, &risk_assessment);
        assert!(position_size.is_ok());
        
        let size = position_size.unwrap();
        assert!(size >= planner.config.min_order_size);
        assert!(size <= planner.config.max_order_size);
    }

    #[test]
    fn test_market_impact_calculation() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let impact = planner.calculate_market_impact_cost(100000.0, 0.2, 0.7);
        assert!(impact >= 0.0);
        
        // Larger order should have higher impact
        let large_impact = planner.calculate_market_impact_cost(1000000.0, 0.2, 0.7);
        assert!(large_impact > impact);
    }

    #[test]
    fn test_order_splitting() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();

        let splitting = planner.determine_order_splitting(&decision, &factors);
        assert!(splitting.is_ok());
        
        let splitting = splitting.unwrap();
        assert!(splitting.num_orders > 0);
        assert_eq!(splitting.size_distribution.len(), splitting.num_orders);
        assert_eq!(splitting.timing_distribution.len(), splitting.num_orders);
        
        // Size distribution should sum to approximately 1.0
        let total_size: f64 = splitting.size_distribution.iter().sum();
        assert!((total_size - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_time_horizon_calculation() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        // High urgency should result in shorter time horizon
        let urgent_time = planner.calculate_time_horizon(0.9, 0.2, 0.7);
        let normal_time = planner.calculate_time_horizon(0.5, 0.2, 0.7);
        assert!(urgent_time < normal_time);
        
        // High volatility should result in longer time horizon
        let volatile_time = planner.calculate_time_horizon(0.5, 0.8, 0.7);
        assert!(volatile_time > normal_time);
    }

    #[tokio::test]
    async fn test_market_timing_analysis() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let factors = create_test_factors();

        let timing = planner.analyze_market_timing(&decision, &factors).await;
        assert!(timing.is_ok());
        
        let timing = timing.unwrap();
        assert!(!timing.optimal_windows.is_empty());
        assert!(!timing.liquidity_forecast.hourly_predictions.is_empty());
        assert!(!timing.volatility_forecast.hourly_predictions.is_empty());
    }

    #[test]
    fn test_execution_cost_calculation() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let plan = ExecutionPlan {
            strategy: ExecutionStrategy::VWAP,
            position_size: 100000.0,
            stop_loss: Some(0.02),
            take_profit: Some(0.06),
            time_horizon: std::time::Duration::from_secs(3600),
            priority: ExecutionPriority::Medium,
        };
        let factors = create_test_factors();

        let costs = planner.calculate_execution_costs(&decision, &plan, &factors);
        assert!(costs.is_ok());
        
        let costs = costs.unwrap();
        assert!(costs.total_cost >= 0.0);
        assert!(costs.market_impact_cost >= 0.0);
        assert!(costs.commission_fees >= 0.0);
    }

    #[test]
    fn test_execution_risk_assessment() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let plan = ExecutionPlan {
            strategy: ExecutionStrategy::VWAP,
            position_size: 100000.0,
            stop_loss: Some(0.02),
            take_profit: Some(0.06),
            time_horizon: std::time::Duration::from_secs(3600),
            priority: ExecutionPriority::Medium,
        };
        let factors = create_test_factors();

        let risks = planner.assess_execution_risks(&decision, &plan, &factors);
        assert!(risks.is_ok());
        
        let risks = risks.unwrap();
        assert!(risks.overall_risk >= 0.0 && risks.overall_risk <= 1.0);
        assert!(risks.slippage_risk >= 0.0 && risks.slippage_risk <= 1.0);
        assert!(risks.liquidity_risk >= 0.0 && risks.liquidity_risk <= 1.0);
    }

    #[test]
    fn test_performance_expectations() {
        let config = ExecutionPlannerConfig::default();
        let planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let plan = ExecutionPlan {
            strategy: ExecutionStrategy::VWAP,
            position_size: 100000.0,
            stop_loss: Some(0.02),
            take_profit: Some(0.06),
            time_horizon: std::time::Duration::from_secs(3600),
            priority: ExecutionPriority::Medium,
        };

        let expectations = planner.set_performance_expectations(&decision, &plan);
        assert!(expectations.is_ok());
        
        let expectations = expectations.unwrap();
        assert!(expectations.expected_fill_rate >= 0.0 && expectations.expected_fill_rate <= 1.0);
        assert!(expectations.completion_probability >= 0.0 && expectations.completion_probability <= 1.0);
        assert!(expectations.expected_slippage >= 0.0);
    }

    #[test]
    fn test_performance_tracking() {
        let config = ExecutionPlannerConfig::default();
        let mut planner = ExecutionPlanner::new(config).unwrap();
        
        let decision = create_test_decision();
        let plan = ExecutionPlan {
            strategy: ExecutionStrategy::VWAP,
            position_size: 100000.0,
            stop_loss: Some(0.02),
            take_profit: Some(0.06),
            time_horizon: std::time::Duration::from_secs(3600),
            priority: ExecutionPriority::Medium,
        };
        
        let performance = ExecutionPerformance {
            fill_rate: 0.95,
            execution_time: std::time::Duration::from_secs(300),
            slippage: 0.001,
            market_impact: 0.0005,
            total_cost: 0.002,
            success: true,
        };
        
        let market_conditions = MarketConditionsSnapshot {
            volatility: 0.2,
            liquidity: 0.7,
            average_spread: 0.001,
            volume: 1000000.0,
            price_movement: 0.005,
        };

        planner.record_execution(decision, plan.clone(), performance.clone(), market_conditions);
        
        assert_eq!(planner.execution_history.len(), 1);
        
        let strategy_performance = planner.get_strategy_performance(&plan.strategy);
        assert!(strategy_performance.is_some());
        assert_eq!(strategy_performance.unwrap().len(), 1);
    }
}