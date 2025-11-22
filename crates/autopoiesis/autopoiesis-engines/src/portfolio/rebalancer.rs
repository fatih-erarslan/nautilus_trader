//! Portfolio rebalancing implementation

use crate::prelude::*;
use crate::models::{Position, Order, OrderSide, MarketData};
use chrono::{DateTime, Utc, Duration, Datelike, Timelike};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Portfolio rebalancer for automated portfolio rebalancing
#[derive(Debug, Clone)]
pub struct PortfolioRebalancer {
    /// Rebalancer configuration
    config: PortfolioRebalancerConfig,
    
    /// Target allocations
    target_allocations: HashMap<String, f64>,
    
    /// Rebalancing history
    rebalancing_history: Vec<RebalancingRecord>,
    
    /// Last rebalancing timestamp
    last_rebalancing_time: Option<DateTime<Utc>>,
    
    /// Drift tracking
    drift_tracker: DriftTracker,
}

#[derive(Debug, Clone)]
pub struct PortfolioRebalancerConfig {
    /// Rebalancing frequency
    pub rebalancing_frequency: RebalancingFrequency,
    
    /// Drift threshold for triggering rebalancing
    pub drift_threshold_pct: f64,
    
    /// Minimum allocation change to execute
    pub min_allocation_change_pct: f64,
    
    /// Transaction cost threshold
    pub transaction_cost_threshold_pct: f64,
    
    /// Rebalancing method
    pub rebalancing_method: RebalancingMethod,
    
    /// Constraints
    pub constraints: RebalancingConstraints,
    
    /// Advanced settings
    pub advanced_settings: AdvancedSettings,
}

#[derive(Debug, Clone)]
pub enum RebalancingFrequency {
    /// Continuous monitoring and rebalancing
    Continuous,
    
    /// Daily rebalancing at specific time
    Daily { hour: u32, minute: u32 },
    
    /// Weekly rebalancing on specific day
    Weekly { day: chrono::Weekday, hour: u32, minute: u32 },
    
    /// Monthly rebalancing on specific day
    Monthly { day: u32, hour: u32, minute: u32 },
    
    /// Quarterly rebalancing
    Quarterly { month_offset: u32, day: u32, hour: u32, minute: u32 },
    
    /// Threshold-based rebalancing
    ThresholdBased { check_frequency_minutes: u32 },
}

#[derive(Debug, Clone)]
pub enum RebalancingMethod {
    /// Full rebalancing to target allocations
    FullRebalancing,
    
    /// Partial rebalancing with dampening factor
    PartialRebalancing { dampening_factor: f64 },
    
    /// Cash-efficient rebalancing (using new cash flows)
    CashEfficient,
    
    /// Threshold band rebalancing
    ThresholdBand { lower_band: f64, upper_band: f64 },
    
    /// Time-weighted rebalancing
    TimeWeighted { time_decay_factor: f64 },
    
    /// Volatility-adjusted rebalancing
    VolatilityAdjusted,
}

#[derive(Debug, Clone)]
pub struct RebalancingConstraints {
    /// Minimum trade size
    pub min_trade_size: Decimal,
    
    /// Maximum trade size per asset
    pub max_trade_size_per_asset: Option<Decimal>,
    
    /// Maximum number of trades per rebalancing
    pub max_trades_per_rebalancing: Option<u32>,
    
    /// Blacklisted assets for rebalancing
    pub blacklisted_assets: Vec<String>,
    
    /// Trading hour restrictions
    pub trading_hours: Option<TradingHours>,
    
    /// Tax considerations
    pub tax_optimization: TaxOptimization,
}

#[derive(Debug, Clone)]
pub struct TradingHours {
    pub start_hour: u32,
    pub start_minute: u32,
    pub end_hour: u32,
    pub end_minute: u32,
    pub timezone: String,
}

#[derive(Debug, Clone)]
pub struct TaxOptimization {
    /// Enable tax-loss harvesting
    pub enable_tax_loss_harvesting: bool,
    
    /// Minimum holding period for long-term gains
    pub min_holding_period_days: u32,
    
    /// Tax rate for short-term gains
    pub short_term_tax_rate: f64,
    
    /// Tax rate for long-term gains
    pub long_term_tax_rate: f64,
}

#[derive(Debug, Clone)]
pub struct AdvancedSettings {
    /// Enable smart order routing
    pub enable_smart_order_routing: bool,
    
    /// Market impact consideration
    pub consider_market_impact: bool,
    
    /// Liquidity filtering
    pub liquidity_threshold: Option<Decimal>,
    
    /// Correlation-based adjustments
    pub correlation_adjustments: bool,
    
    /// Dynamic threshold adjustment
    pub dynamic_threshold_adjustment: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DriftTracker {
    current_drifts: HashMap<String, f64>,
    drift_history: Vec<DriftSnapshot>,
    max_observed_drift: f64,
    last_update: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct DriftSnapshot {
    timestamp: DateTime<Utc>,
    portfolio_drift: f64,
    asset_drifts: HashMap<String, f64>,
    trigger_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct RebalancingRecord {
    timestamp: DateTime<Utc>,
    trigger_reason: RebalancingTrigger,
    pre_rebalancing_allocations: HashMap<String, f64>,
    target_allocations: HashMap<String, f64>,
    post_rebalancing_allocations: HashMap<String, f64>,
    generated_orders: Vec<Order>,
    estimated_transaction_costs: Decimal,
    execution_summary: ExecutionSummary,
}

#[derive(Debug, Clone)]
pub enum RebalancingTrigger {
    /// Scheduled rebalancing
    Scheduled,
    
    /// Drift threshold exceeded
    DriftThreshold { max_drift: f64 },
    
    /// Manual trigger
    Manual,
    
    /// New cash inflow
    CashInflow { amount: Decimal },
    
    /// Market volatility spike
    VolatilitySpike { volatility_level: f64 },
    
    /// Correlation breakdown
    CorrelationBreakdown,
}

#[derive(Debug, Clone, Default)]
struct ExecutionSummary {
    total_orders_generated: u32,
    total_trade_value: Decimal,
    estimated_slippage: f64,
    estimated_execution_time_minutes: u32,
    tax_implications: TaxImplications,
}

#[derive(Debug, Clone, Default)]
struct TaxImplications {
    short_term_gains: Decimal,
    long_term_gains: Decimal,
    realized_losses: Decimal,
    tax_efficiency_score: f64,
}

#[derive(Debug, Clone)]
pub struct RebalancingResult {
    pub trigger_reason: RebalancingTrigger,
    pub rebalancing_needed: bool,
    pub generated_orders: Vec<Order>,
    pub allocation_changes: HashMap<String, AllocationChange>,
    pub execution_summary: ExecutionSummary,
    pub drift_analysis: DriftAnalysis,
    pub cost_benefit_analysis: CostBenefitAnalysis,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AllocationChange {
    pub symbol: String,
    pub current_allocation: f64,
    pub target_allocation: f64,
    pub allocation_change: f64,
    pub trade_value: Decimal,
    pub trade_direction: String,
}

#[derive(Debug, Clone)]
pub struct DriftAnalysis {
    pub portfolio_level_drift: f64,
    pub max_asset_drift: f64,
    pub average_asset_drift: f64,
    pub drift_trend: DriftTrend,
    pub time_since_last_rebalancing: Duration,
}

#[derive(Debug, Clone)]
pub enum DriftTrend {
    Increasing,
    Stable,
    Decreasing,
}

#[derive(Debug, Clone)]
pub struct CostBenefitAnalysis {
    pub estimated_transaction_costs: Decimal,
    pub estimated_market_impact: f64,
    pub expected_benefit_from_rebalancing: f64,
    pub cost_benefit_ratio: f64,
    pub recommendation: RebalancingRecommendation,
}

#[derive(Debug, Clone)]
pub enum RebalancingRecommendation {
    ProceedWithRebalancing,
    DelayRebalancing { reason: String },
    ModifyTargetAllocations { suggestions: Vec<String> },
    PartialRebalancing { subset: Vec<String> },
}

impl Default for PortfolioRebalancerConfig {
    fn default() -> Self {
        Self {
            rebalancing_frequency: RebalancingFrequency::ThresholdBased {
                check_frequency_minutes: 60,
            },
            drift_threshold_pct: 0.05, // 5% drift threshold
            min_allocation_change_pct: 0.01, // 1% minimum change
            transaction_cost_threshold_pct: 0.005, // 0.5% transaction cost threshold
            rebalancing_method: RebalancingMethod::FullRebalancing,
            constraints: RebalancingConstraints {
                min_trade_size: Decimal::from(100),
                max_trade_size_per_asset: Some(Decimal::from(50000)),
                max_trades_per_rebalancing: Some(20),
                blacklisted_assets: Vec::new(),
                trading_hours: None,
                tax_optimization: TaxOptimization {
                    enable_tax_loss_harvesting: false,
                    min_holding_period_days: 365,
                    short_term_tax_rate: 0.37,
                    long_term_tax_rate: 0.20,
                },
            },
            advanced_settings: AdvancedSettings {
                enable_smart_order_routing: false,
                consider_market_impact: true,
                liquidity_threshold: Some(Decimal::from(10000)),
                correlation_adjustments: false,
                dynamic_threshold_adjustment: false,
            },
        }
    }
}

impl PortfolioRebalancer {
    /// Create a new portfolio rebalancer
    pub fn new(config: PortfolioRebalancerConfig) -> Self {
        Self {
            config,
            target_allocations: HashMap::new(),
            rebalancing_history: Vec::new(),
            last_rebalancing_time: None,
            drift_tracker: DriftTracker::default(),
        }
    }

    /// Set target allocations
    pub fn set_target_allocations(&mut self, targets: HashMap<String, f64>) -> Result<()> {
        // Validate that allocations sum to 1.0
        let total_allocation: f64 = targets.values().sum();
        if (total_allocation - 1.0).abs() > 1e-6 {
            return Err(Error::Config(format!(
                "Target allocations sum to {:.6}, expected 1.0", 
                total_allocation
            )));
        }

        // Validate individual allocations
        for (symbol, allocation) in &targets {
            if *allocation < 0.0 || *allocation > 1.0 {
                return Err(Error::Config(format!(
                    "Invalid allocation for {}: {:.6}. Must be between 0.0 and 1.0",
                    symbol, allocation
                )));
            }
        }

        self.target_allocations = targets;
        info!("Updated target allocations for {} assets", self.target_allocations.len());
        Ok(())
    }

    /// Check if rebalancing is needed and generate orders
    pub async fn check_rebalancing(
        &mut self,
        current_positions: &[Position],
        market_data: &HashMap<String, MarketData>,
        cash_available: Decimal
    ) -> Result<RebalancingResult> {
        // Calculate current allocations
        let current_allocations = self.calculate_current_allocations(current_positions, cash_available)?;
        
        // Update drift tracking
        self.update_drift_tracking(&current_allocations).await?;
        
        // Check various rebalancing triggers
        let trigger_reason = self.check_rebalancing_triggers(&current_allocations).await?;
        
        let rebalancing_needed = matches!(trigger_reason, Some(_));
        
        if !rebalancing_needed {
            return Ok(RebalancingResult {
                trigger_reason: RebalancingTrigger::Manual, // Default for no trigger
                rebalancing_needed: false,
                generated_orders: Vec::new(),
                allocation_changes: HashMap::new(),
                execution_summary: ExecutionSummary::default(),
                drift_analysis: self.create_drift_analysis(&current_allocations),
                cost_benefit_analysis: CostBenefitAnalysis {
                    estimated_transaction_costs: Decimal::ZERO,
                    estimated_market_impact: 0.0,
                    expected_benefit_from_rebalancing: 0.0,
                    cost_benefit_ratio: 0.0,
                    recommendation: RebalancingRecommendation::DelayRebalancing {
                        reason: "No rebalancing trigger met".to_string(),
                    },
                },
                generated_at: Utc::now(),
            });
        }

        let trigger = trigger_reason.unwrap();
        
        // Calculate target changes
        let allocation_changes = self.calculate_allocation_changes(&current_allocations, current_positions)?;
        
        // Perform cost-benefit analysis
        let cost_benefit_analysis = self.perform_cost_benefit_analysis(&allocation_changes, market_data).await?;
        
        // Check if rebalancing is still beneficial after cost analysis
        if !matches!(cost_benefit_analysis.recommendation, RebalancingRecommendation::ProceedWithRebalancing) {
            return Ok(RebalancingResult {
                trigger_reason: trigger,
                rebalancing_needed: false,
                generated_orders: Vec::new(),
                allocation_changes,
                execution_summary: ExecutionSummary::default(),
                drift_analysis: self.create_drift_analysis(&current_allocations),
                cost_benefit_analysis,
                generated_at: Utc::now(),
            });
        }

        // Generate rebalancing orders
        let (orders, execution_summary) = self.generate_rebalancing_orders(&allocation_changes, current_positions, market_data).await?;
        
        // Record rebalancing
        let rebalancing_record = RebalancingRecord {
            timestamp: Utc::now(),
            trigger_reason: trigger.clone(),
            pre_rebalancing_allocations: current_allocations.clone(),
            target_allocations: self.target_allocations.clone(),
            post_rebalancing_allocations: HashMap::new(), // Would be updated after execution
            generated_orders: orders.clone(),
            estimated_transaction_costs: execution_summary.total_trade_value * Decimal::new(1, 3), // 0.1% est cost
            execution_summary: execution_summary.clone(),
        };

        self.rebalancing_history.push(rebalancing_record);
        self.last_rebalancing_time = Some(Utc::now());

        // Maintain history size
        if self.rebalancing_history.len() > 1000 {
            self.rebalancing_history.drain(0..100);
        }

        Ok(RebalancingResult {
            trigger_reason: trigger,
            rebalancing_needed: true,
            generated_orders: orders,
            allocation_changes,
            execution_summary,
            drift_analysis: self.create_drift_analysis(&current_allocations),
            cost_benefit_analysis,
            generated_at: Utc::now(),
        })
    }

    /// Force rebalancing regardless of triggers
    pub async fn force_rebalancing(
        &mut self,
        current_positions: &[Position],
        market_data: &HashMap<String, MarketData>,
        cash_available: Decimal
    ) -> Result<RebalancingResult> {
        let current_allocations = self.calculate_current_allocations(current_positions, cash_available)?;
        let allocation_changes = self.calculate_allocation_changes(&current_allocations, current_positions)?;
        let cost_benefit_analysis = self.perform_cost_benefit_analysis(&allocation_changes, market_data).await?;
        let (orders, execution_summary) = self.generate_rebalancing_orders(&allocation_changes, current_positions, market_data).await?;

        Ok(RebalancingResult {
            trigger_reason: RebalancingTrigger::Manual,
            rebalancing_needed: true,
            generated_orders: orders,
            allocation_changes,
            execution_summary,
            drift_analysis: self.create_drift_analysis(&current_allocations),
            cost_benefit_analysis,
            generated_at: Utc::now(),
        })
    }

    fn calculate_current_allocations(&self, positions: &[Position], cash_available: Decimal) -> Result<HashMap<String, f64>> {
        let mut current_allocations = HashMap::new();
        
        // Calculate total portfolio value
        let position_value: Decimal = positions.iter()
            .map(|p| p.quantity * p.mark_price)
            .sum();
        let total_value = position_value + cash_available;

        if total_value <= Decimal::ZERO {
            return Ok(current_allocations);
        }

        // Calculate position allocations
        for position in positions {
            if position.quantity > Decimal::ZERO {
                let position_value = position.quantity * position.mark_price;
                let allocation = (position_value / total_value).to_f64().unwrap_or(0.0);
                current_allocations.insert(position.symbol.clone(), allocation);
            }
        }

        // Add cash allocation if significant
        let cash_allocation = (cash_available / total_value).to_f64().unwrap_or(0.0);
        if cash_allocation > 0.001 { // More than 0.1%
            current_allocations.insert("CASH".to_string(), cash_allocation);
        }

        Ok(current_allocations)
    }

    async fn update_drift_tracking(&mut self, current_allocations: &HashMap<String, f64>) -> Result<()> {
        let mut portfolio_drift = 0.0;
        let mut asset_drifts = HashMap::new();

        for (symbol, target_allocation) in &self.target_allocations {
            let current_allocation = current_allocations.get(symbol).copied().unwrap_or(0.0);
            let drift = (current_allocation - target_allocation).abs();
            
            asset_drifts.insert(symbol.clone(), drift);
            portfolio_drift += drift;
        }

        // Calculate portfolio-level drift
        portfolio_drift /= self.target_allocations.len() as f64;

        // Update drift tracker
        self.drift_tracker.current_drifts = asset_drifts.clone();
        self.drift_tracker.max_observed_drift = f64::max(self.drift_tracker.max_observed_drift, portfolio_drift);
        self.drift_tracker.last_update = Some(Utc::now());

        // Add snapshot to history
        let snapshot = DriftSnapshot {
            timestamp: Utc::now(),
            portfolio_drift,
            asset_drifts,
            trigger_reason: None,
        };

        self.drift_tracker.drift_history.push(snapshot);

        // Maintain drift history (keep last 1000 snapshots)
        if self.drift_tracker.drift_history.len() > 1000 {
            self.drift_tracker.drift_history.drain(0..100);
        }

        Ok(())
    }

    async fn check_rebalancing_triggers(&self, _current_allocations: &HashMap<String, f64>) -> Result<Option<RebalancingTrigger>> {
        // Check drift threshold trigger
        let max_drift = self.drift_tracker.current_drifts.values()
            .fold(0.0f64, |max, &drift| f64::max(max, drift));
        
        if max_drift > self.config.drift_threshold_pct {
            return Ok(Some(RebalancingTrigger::DriftThreshold { max_drift }));
        }

        // Check scheduled rebalancing
        if let Some(trigger) = self.check_scheduled_rebalancing().await? {
            return Ok(Some(trigger));
        }

        // Check volatility-based triggers (simplified)
        if self.config.advanced_settings.dynamic_threshold_adjustment {
            let current_volatility = 0.25; // Would calculate from market data
            if current_volatility > 0.30 {
                return Ok(Some(RebalancingTrigger::VolatilitySpike { 
                    volatility_level: current_volatility 
                }));
            }
        }

        Ok(None)
    }

    async fn check_scheduled_rebalancing(&self) -> Result<Option<RebalancingTrigger>> {
        let now = Utc::now();
        
        match &self.config.rebalancing_frequency {
            RebalancingFrequency::Daily { hour, minute } => {
                if let Some(last_rebalancing) = self.last_rebalancing_time {
                    let today_target = now.date_naive().and_hms_opt(*hour, *minute, 0).unwrap();
                    let today_target_utc: DateTime<Utc> = DateTime::from_naive_utc_and_offset(today_target, Utc);
                    
                    if now >= today_target_utc && last_rebalancing < today_target_utc {
                        return Ok(Some(RebalancingTrigger::Scheduled));
                    }
                }
            },
            RebalancingFrequency::Weekly { day, hour, minute } => {
                // Check if it's the target day and time
                if now.weekday() == *day && 
                   now.hour() >= *hour && 
                   now.minute() >= *minute {
                    if let Some(last_rebalancing) = self.last_rebalancing_time {
                        if (now - last_rebalancing).num_days() >= 7 {
                            return Ok(Some(RebalancingTrigger::Scheduled));
                        }
                    } else {
                        return Ok(Some(RebalancingTrigger::Scheduled));
                    }
                }
            },
            RebalancingFrequency::ThresholdBased { check_frequency_minutes } => {
                if let Some(last_rebalancing) = self.last_rebalancing_time {
                    if (now - last_rebalancing).num_minutes() >= *check_frequency_minutes as i64 {
                        // This trigger is handled by drift checking, not scheduling
                        return Ok(None);
                    }
                }
            },
            _ => {
                // Other frequency types would be implemented similarly
            }
        }

        Ok(None)
    }

    fn calculate_allocation_changes(&self, current_allocations: &HashMap<String, f64>, positions: &[Position]) -> Result<HashMap<String, AllocationChange>> {
        let mut changes = HashMap::new();
        
        // Calculate total portfolio value
        let total_value: Decimal = positions.iter()
            .map(|p| p.quantity * p.mark_price)
            .sum();

        for (symbol, target_allocation) in &self.target_allocations {
            let current_allocation = current_allocations.get(symbol).copied().unwrap_or(0.0);
            let allocation_change = target_allocation - current_allocation;
            
            // Skip if change is below minimum threshold
            if allocation_change.abs() < self.config.min_allocation_change_pct {
                continue;
            }

            let trade_value = total_value * Decimal::from_f64_retain(allocation_change).unwrap_or_default();
            let trade_direction = if allocation_change > 0.0 { "buy" } else { "sell" };

            changes.insert(symbol.clone(), AllocationChange {
                symbol: symbol.clone(),
                current_allocation,
                target_allocation: *target_allocation,
                allocation_change,
                trade_value: trade_value.abs(),
                trade_direction: trade_direction.to_string(),
            });
        }

        Ok(changes)
    }

    async fn perform_cost_benefit_analysis(&self, allocation_changes: &HashMap<String, AllocationChange>, _market_data: &HashMap<String, MarketData>) -> Result<CostBenefitAnalysis> {
        let total_trade_value: Decimal = allocation_changes.values()
            .map(|change| change.trade_value)
            .sum();

        // Estimate transaction costs
        let estimated_transaction_costs = total_trade_value * Decimal::new(1, 3); // 0.1% estimated cost

        // Estimate market impact (simplified)
        let estimated_market_impact = if total_trade_value > Decimal::from(100000) {
            0.002 // 0.2% impact for large trades
        } else {
            0.0005 // 0.05% impact for smaller trades
        };

        // Calculate expected benefit from rebalancing (simplified)
        let max_drift = allocation_changes.values()
            .map(|change| change.allocation_change.abs())
            .fold(0.0f64, |a, b| f64::max(a, b));
        
        let expected_benefit = max_drift * 0.5; // Simplified benefit calculation

        // Calculate cost-benefit ratio
        let transaction_cost_pct = estimated_transaction_costs.to_f64().unwrap_or(0.0) / 
                                 total_trade_value.to_f64().unwrap_or(1.0);
        let total_cost = transaction_cost_pct + estimated_market_impact;
        
        let cost_benefit_ratio = if total_cost > 0.0 {
            expected_benefit / total_cost
        } else {
            f64::INFINITY
        };

        // Make recommendation
        let recommendation = if cost_benefit_ratio > 2.0 {
            RebalancingRecommendation::ProceedWithRebalancing
        } else if cost_benefit_ratio > 1.0 {
            RebalancingRecommendation::PartialRebalancing {
                subset: allocation_changes.keys()
                    .filter(|symbol| {
                        let change = allocation_changes.get(*symbol).unwrap();
                        change.allocation_change.abs() > self.config.drift_threshold_pct * 1.5
                    })
                    .cloned()
                    .collect(),
            }
        } else {
            RebalancingRecommendation::DelayRebalancing {
                reason: format!("Cost-benefit ratio too low: {:.2}", cost_benefit_ratio),
            }
        };

        Ok(CostBenefitAnalysis {
            estimated_transaction_costs,
            estimated_market_impact,
            expected_benefit_from_rebalancing: expected_benefit,
            cost_benefit_ratio,
            recommendation,
        })
    }

    async fn generate_rebalancing_orders(&self, allocation_changes: &HashMap<String, AllocationChange>, positions: &[Position], market_data: &HashMap<String, MarketData>) -> Result<(Vec<Order>, ExecutionSummary)> {
        let mut orders = Vec::new();
        let mut total_trade_value = Decimal::ZERO;
        let mut estimated_slippage = 0.0;

        for (symbol, change) in allocation_changes {
            // Skip blacklisted assets
            if self.config.constraints.blacklisted_assets.contains(symbol) {
                continue;
            }

            // Get current market price
            let current_price = market_data.get(symbol)
                .map(|md| md.mid)
                .or_else(|| {
                    positions.iter()
                        .find(|p| p.symbol == *symbol)
                        .map(|p| p.mark_price)
                })
                .unwrap_or(Decimal::from(100)); // Fallback price

            // Calculate order quantity
            let order_quantity = change.trade_value / current_price;

            // Check minimum trade size
            if order_quantity < self.config.constraints.min_trade_size {
                continue;
            }

            // Check maximum trade size
            if let Some(max_size) = self.config.constraints.max_trade_size_per_asset {
                if order_quantity > max_size {
                    continue;
                }
            }

            // Create order
            let order = Order {
                id: uuid::Uuid::new_v4(),
                symbol: symbol.clone(),
                side: if change.trade_direction == "buy" { OrderSide::Buy } else { OrderSide::Sell },
                order_type: crate::models::OrderType::Market,
                quantity: order_quantity,
                price: Some(current_price),
                time_in_force: crate::models::TimeInForce::IOC,
                status: crate::models::OrderStatus::Pending,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };

            orders.push(order);
            total_trade_value += change.trade_value;
            estimated_slippage += 0.0005; // 0.05% per trade
        }

        // Check maximum trades constraint
        if let Some(max_trades) = self.config.constraints.max_trades_per_rebalancing {
            if orders.len() > max_trades as usize {
                orders.truncate(max_trades as usize);
            }
        }

        let execution_summary = ExecutionSummary {
            total_orders_generated: orders.len() as u32,
            total_trade_value,
            estimated_slippage: estimated_slippage / orders.len().max(1) as f64,
            estimated_execution_time_minutes: (orders.len() as u32 * 2).min(60), // 2 minutes per order, max 1 hour
            tax_implications: TaxImplications::default(), // Would calculate based on positions
        };

        Ok((orders, execution_summary))
    }

    fn create_drift_analysis(&self, _current_allocations: &HashMap<String, f64>) -> DriftAnalysis {
        let portfolio_drift = self.drift_tracker.current_drifts.values()
            .sum::<f64>() / self.drift_tracker.current_drifts.len().max(1) as f64;

        let max_asset_drift = self.drift_tracker.current_drifts.values()
            .fold(0.0f64, |max, &drift| f64::max(max, drift));

        let average_asset_drift = portfolio_drift;

        // Determine drift trend
        let drift_trend = if self.drift_tracker.drift_history.len() >= 3 {
            let recent_drifts: Vec<f64> = self.drift_tracker.drift_history
                .iter()
                .rev()
                .take(3)
                .map(|snapshot| snapshot.portfolio_drift)
                .collect();

            if recent_drifts[0] > recent_drifts[2] * 1.1 {
                DriftTrend::Increasing
            } else if recent_drifts[0] < recent_drifts[2] * 0.9 {
                DriftTrend::Decreasing
            } else {
                DriftTrend::Stable
            }
        } else {
            DriftTrend::Stable
        };

        let time_since_last_rebalancing = if let Some(last_time) = self.last_rebalancing_time {
            Utc::now() - last_time
        } else {
            Duration::days(365) // Default to 1 year if no previous rebalancing
        };

        DriftAnalysis {
            portfolio_level_drift: portfolio_drift,
            max_asset_drift,
            average_asset_drift,
            drift_trend,
            time_since_last_rebalancing,
        }
    }

    /// Get rebalancing history
    pub fn get_rebalancing_history(&self, limit: Option<usize>) -> Vec<RebalancingRecord> {
        let limit = limit.unwrap_or(100);
        self.rebalancing_history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get current drift status
    pub fn get_drift_status(&self) -> &DriftTracker {
        &self.drift_tracker
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: PortfolioRebalancerConfig) {
        self.config = new_config;
        info!("Updated portfolio rebalancer configuration");
    }
}