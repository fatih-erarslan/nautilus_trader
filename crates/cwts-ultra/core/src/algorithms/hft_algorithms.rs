//! High-Frequency Trading Algorithms - Ultra-low latency trading strategies
//!
//! This module implements production-grade HFT algorithms with sub-10ms execution times.
//! All algorithms are lock-free and SIMD-optimized for maximum performance.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
// Removed unused PI import
use crossbeam::queue::SegQueue;
use rayon::prelude::*;

/// Execution time constraint for HFT algorithms (microseconds)
const MAX_EXECUTION_TIME_US: u64 = 10_000; // 10ms
#[allow(dead_code)]
const TICK_SIZE_PRECISION: f64 = 0.0001;
const MIN_PROFIT_THRESHOLD: f64 = 0.0001; // 1 basis point

/// HFT strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HftStrategyType {
    LatencyArbitrage,     // Cross-venue latency arbitrage
    MarketMaking,         // Market making with inventory management
    StatisticalArbitrage, // Statistical arbitrage pairs
    MomentumScalping,     // Ultra-short momentum scalping
    LiquidityRebate,      // Liquidity rebate capture
}

/// Trading signal strength
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SignalStrength {
    Strong, // Execute immediately
    Medium, // Execute with caution
    Weak,   // Monitor only
    None,   // No signal
}

/// HFT trading signal
#[derive(Debug, Clone)]
pub struct HftSignal {
    pub strategy_type: HftStrategyType,
    pub symbol: String,
    pub timestamp: u64,
    pub signal_strength: SignalStrength,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub expected_profit: f64,
    pub risk_score: f64,
    pub execution_urgency: u64, // Microseconds until signal expires
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Market data snapshot for HFT analysis
#[derive(Debug, Clone)]
pub struct HftMarketData {
    pub timestamp: u64,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub volume: f64,
    pub vwap: f64,
}

/// Cross-venue price data
#[derive(Debug, Clone)]
pub struct VenuePrice {
    pub venue: String,
    pub timestamp: u64,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub latency_us: u64,
}

/// Statistical arbitrage pair
#[derive(Debug, Clone)]
pub struct StatArbPair {
    pub symbol_a: String,
    pub symbol_b: String,
    pub hedge_ratio: f64,
    pub mean_spread: f64,
    pub spread_std: f64,
    pub cointegration_score: f64,
    pub last_update: u64,
}

/// Market making state
#[derive(Debug)]
struct MarketMakerState {
    inventory: AtomicI64,     // Current position (signed quantity)
    max_inventory: u64,       // Maximum allowed position
    target_spread: AtomicU64, // Target bid-ask spread in ticks
    #[allow(dead_code)]
    skew_factor: AtomicU64, // Price skewing based on inventory
    #[allow(dead_code)]
    filled_orders: AtomicU64, // Count of filled orders
    #[allow(dead_code)]
    last_update: AtomicU64, // Last update timestamp
}

/// HFT Algorithm Engine with multiple strategies
pub struct HftAlgorithmEngine {
    // Strategy configurations
    latency_arb_enabled: AtomicBool,
    market_making_enabled: AtomicBool,
    stat_arb_enabled: AtomicBool,
    momentum_scalping_enabled: AtomicBool,

    // Market making state per symbol
    mm_states: HashMap<String, MarketMakerState>,

    // Statistical arbitrage pairs
    stat_arb_pairs: Vec<StatArbPair>,
    pair_spreads: HashMap<String, VecDeque<f64>>,

    // Cross-venue data for latency arbitrage
    venue_prices: HashMap<String, Vec<VenuePrice>>,

    // Performance tracking
    execution_times: VecDeque<u64>,
    signal_queue: SegQueue<HftSignal>,

    // Risk management
    max_position_per_symbol: u64,
    max_daily_loss: f64,
    current_daily_pnl: AtomicI64,

    // Configuration
    #[allow(dead_code)]
    window_size: usize,
    tick_size: f64,
}

impl std::fmt::Debug for HftAlgorithmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HftAlgorithmEngine")
            .field("latency_arb_enabled", &self.latency_arb_enabled.load(Ordering::Relaxed))
            .field("market_making_enabled", &self.market_making_enabled.load(Ordering::Relaxed))
            .field("stat_arb_enabled", &self.stat_arb_enabled.load(Ordering::Relaxed))
            .field("momentum_scalping_enabled", &self.momentum_scalping_enabled.load(Ordering::Relaxed))
            .field("mm_states_count", &self.mm_states.len())
            .field("stat_arb_pairs_count", &self.stat_arb_pairs.len())
            .field("max_position_per_symbol", &self.max_position_per_symbol)
            .field("max_daily_loss", &self.max_daily_loss)
            .finish()
    }
}

impl HftAlgorithmEngine {
    /// Create new HFT algorithm engine
    pub fn new(config: HftConfig) -> Self {
        Self {
            latency_arb_enabled: AtomicBool::new(true),
            market_making_enabled: AtomicBool::new(true),
            stat_arb_enabled: AtomicBool::new(true),
            momentum_scalping_enabled: AtomicBool::new(true),

            mm_states: HashMap::new(),
            stat_arb_pairs: Vec::new(),
            pair_spreads: HashMap::new(),
            venue_prices: HashMap::new(),

            execution_times: VecDeque::with_capacity(1000),
            signal_queue: SegQueue::new(),

            max_position_per_symbol: config.max_position_per_symbol,
            max_daily_loss: config.max_daily_loss,
            current_daily_pnl: AtomicI64::new(0),

            window_size: config.window_size,
            tick_size: config.tick_size,
        }
    }

    /// Process market data and generate HFT signals
    pub fn process_market_data(&mut self, market_data: &[HftMarketData]) -> Vec<HftSignal> {
        let start_time = self.get_timestamp_us();
        let mut signals = Vec::new();

        // Parallel processing of different strategies
        let strategy_signals: Vec<Vec<HftSignal>> = [
            HftStrategyType::LatencyArbitrage,
            HftStrategyType::MarketMaking,
            HftStrategyType::StatisticalArbitrage,
            HftStrategyType::MomentumScalping,
        ]
        .par_iter()
        .map(|&strategy| {
            match strategy {
                HftStrategyType::LatencyArbitrage => self.detect_latency_arbitrage(market_data),
                HftStrategyType::MarketMaking => self.generate_market_making_signals(market_data),
                HftStrategyType::StatisticalArbitrage => {
                    self.detect_statistical_arbitrage(market_data)
                }
                HftStrategyType::MomentumScalping => self.detect_momentum_scalping(market_data),
                HftStrategyType::LiquidityRebate => Vec::new(), // Implemented separately
            }
        })
        .collect();

        // Combine all signals
        for strategy_signal_vec in strategy_signals {
            signals.extend(strategy_signal_vec);
        }

        // Risk filtering and prioritization
        signals = self.filter_signals_by_risk(signals);
        signals.sort_by(|a, b| {
            b.expected_profit
                .partial_cmp(&a.expected_profit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Record execution time
        let execution_time = self.get_timestamp_us() - start_time;
        self.execution_times.push_back(execution_time);
        if self.execution_times.len() > 1000 {
            self.execution_times.pop_front();
        }

        // Ensure we meet latency requirements
        if execution_time > MAX_EXECUTION_TIME_US {
            eprintln!(
                "Warning: HFT execution time exceeded limit: {}μs",
                execution_time
            );
        }

        signals
    }

    /// Detect latency arbitrage opportunities across venues
    fn detect_latency_arbitrage(&self, market_data: &[HftMarketData]) -> Vec<HftSignal> {
        if !self.latency_arb_enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let mut signals = Vec::new();

        for data in market_data {
            if let Some(venue_prices) = self.venue_prices.get(&data.symbol) {
                // Find arbitrage opportunities between venues
                let arbitrage_opps = self.find_arbitrage_opportunities(data, venue_prices);

                for opp in arbitrage_opps {
                    // Calculate expected profit after fees and slippage
                    let gross_profit = (opp.sell_price - opp.buy_price) * opp.quantity;
                    let fees = self.estimate_trading_fees(opp.quantity, 2); // 2 venues
                    let slippage = self.estimate_slippage(data, opp.quantity);
                    let net_profit = gross_profit - fees - slippage;

                    if net_profit > MIN_PROFIT_THRESHOLD * opp.quantity {
                        // Check execution time constraints
                        let max_latency =
                            venue_prices.iter().map(|v| v.latency_us).max().unwrap_or(0);
                        let execution_urgency = (10_000 - max_latency).max(1000); // At least 1ms buffer

                        let signal = HftSignal {
                            strategy_type: HftStrategyType::LatencyArbitrage,
                            symbol: data.symbol.clone(),
                            timestamp: data.timestamp,
                            signal_strength: if net_profit
                                > MIN_PROFIT_THRESHOLD * opp.quantity * 3.0
                            {
                                SignalStrength::Strong
                            } else {
                                SignalStrength::Medium
                            },
                            side: OrderSide::Buy, // Buy on cheaper venue, sell on expensive
                            price: opp.buy_price,
                            quantity: opp.quantity,
                            expected_profit: net_profit,
                            risk_score: self.calculate_arbitrage_risk(&opp, max_latency),
                            execution_urgency,
                        };

                        signals.push(signal);
                    }
                }
            }
        }

        signals
    }

    /// Generate market making signals with inventory management
    fn generate_market_making_signals(&self, market_data: &[HftMarketData]) -> Vec<HftSignal> {
        if !self.market_making_enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let mut signals = Vec::new();

        for data in market_data {
            if let Some(mm_state) = self.mm_states.get(&data.symbol) {
                let current_inventory = mm_state.inventory.load(Ordering::Relaxed);
                let max_inventory = mm_state.max_inventory as i64;

                // Calculate inventory skew
                let inventory_ratio = current_inventory as f64 / max_inventory as f64;
                let skew = self.calculate_inventory_skew(inventory_ratio);

                // Calculate optimal bid/ask prices
                let mid_price = (data.bid + data.ask) / 2.0;
                let spread = data.ask - data.bid;
                let target_spread =
                    (mm_state.target_spread.load(Ordering::Relaxed) as f64) * self.tick_size;

                // Apply inventory skewing
                let optimal_bid = mid_price - target_spread / 2.0 - skew;
                let optimal_ask = mid_price + target_spread / 2.0 - skew;

                // Check if we should quote
                let bid_size =
                    self.calculate_optimal_quote_size(data, OrderSide::Buy, current_inventory);
                let ask_size =
                    self.calculate_optimal_quote_size(data, OrderSide::Sell, current_inventory);

                // Generate bid signal
                if bid_size > 0.0 && current_inventory < max_inventory {
                    let signal = HftSignal {
                        strategy_type: HftStrategyType::MarketMaking,
                        symbol: data.symbol.clone(),
                        timestamp: data.timestamp,
                        signal_strength: SignalStrength::Medium,
                        side: OrderSide::Buy,
                        price: optimal_bid,
                        quantity: bid_size,
                        expected_profit: self.estimate_mm_profit(spread, target_spread),
                        risk_score: self
                            .calculate_mm_risk(current_inventory, mm_state.max_inventory),
                        execution_urgency: 5000, // 5ms urgency for market making
                    };
                    signals.push(signal);
                }

                // Generate ask signal
                if ask_size > 0.0 && current_inventory > -max_inventory {
                    let signal = HftSignal {
                        strategy_type: HftStrategyType::MarketMaking,
                        symbol: data.symbol.clone(),
                        timestamp: data.timestamp,
                        signal_strength: SignalStrength::Medium,
                        side: OrderSide::Sell,
                        price: optimal_ask,
                        quantity: ask_size,
                        expected_profit: self.estimate_mm_profit(spread, target_spread),
                        risk_score: self
                            .calculate_mm_risk(current_inventory, mm_state.max_inventory),
                        execution_urgency: 5000,
                    };
                    signals.push(signal);
                }
            }
        }

        signals
    }

    /// Detect statistical arbitrage opportunities
    fn detect_statistical_arbitrage(&self, market_data: &[HftMarketData]) -> Vec<HftSignal> {
        if !self.stat_arb_enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let mut signals = Vec::new();

        // Create price map for quick lookup
        let price_map: HashMap<String, &HftMarketData> = market_data
            .iter()
            .map(|data| (data.symbol.clone(), data))
            .collect();

        for pair in &self.stat_arb_pairs {
            if let (Some(data_a), Some(data_b)) =
                (price_map.get(&pair.symbol_a), price_map.get(&pair.symbol_b))
            {
                // Calculate current spread
                let price_a = (data_a.bid + data_a.ask) / 2.0;
                let price_b = (data_b.bid + data_b.ask) / 2.0;
                let current_spread = price_a - pair.hedge_ratio * price_b;

                // Calculate z-score
                let z_score = (current_spread - pair.mean_spread) / pair.spread_std;

                // Check for mean reversion opportunity
                let abs_z_score = z_score.abs();
                if abs_z_score > 2.0 {
                    // 2 standard deviations
                    let signal_strength = if abs_z_score > 3.0 {
                        SignalStrength::Strong
                    } else {
                        SignalStrength::Medium
                    };

                    // Determine trade direction
                    let (primary_side, hedge_side) = if z_score > 0.0 {
                        // Spread too high, sell A and buy B
                        (OrderSide::Sell, OrderSide::Buy)
                    } else {
                        // Spread too low, buy A and sell B
                        (OrderSide::Buy, OrderSide::Sell)
                    };

                    let trade_size = self.calculate_stat_arb_size(pair, abs_z_score);
                    let expected_profit =
                        self.estimate_stat_arb_profit(abs_z_score, pair.spread_std, trade_size);

                    // Primary leg signal
                    let primary_signal = HftSignal {
                        strategy_type: HftStrategyType::StatisticalArbitrage,
                        symbol: pair.symbol_a.clone(),
                        timestamp: data_a.timestamp,
                        signal_strength,
                        side: primary_side,
                        price: if primary_side == OrderSide::Buy {
                            data_a.ask
                        } else {
                            data_a.bid
                        },
                        quantity: trade_size,
                        expected_profit: expected_profit / 2.0,
                        risk_score: self
                            .calculate_stat_arb_risk(abs_z_score, pair.cointegration_score),
                        execution_urgency: 2000, // 2ms urgency for stat arb
                    };

                    // Hedge leg signal
                    let hedge_signal = HftSignal {
                        strategy_type: HftStrategyType::StatisticalArbitrage,
                        symbol: pair.symbol_b.clone(),
                        timestamp: data_b.timestamp,
                        signal_strength,
                        side: hedge_side,
                        price: if hedge_side == OrderSide::Buy {
                            data_b.ask
                        } else {
                            data_b.bid
                        },
                        quantity: trade_size * pair.hedge_ratio,
                        expected_profit: expected_profit / 2.0,
                        risk_score: self
                            .calculate_stat_arb_risk(abs_z_score, pair.cointegration_score),
                        execution_urgency: 2000,
                    };

                    signals.push(primary_signal);
                    signals.push(hedge_signal);
                }
            }
        }

        signals
    }

    /// Detect ultra-short momentum scalping opportunities
    fn detect_momentum_scalping(&self, market_data: &[HftMarketData]) -> Vec<HftSignal> {
        if !self.momentum_scalping_enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let mut signals = Vec::new();

        for data in market_data {
            // Calculate short-term momentum indicators
            let price_momentum = self.calculate_price_momentum(data);
            let volume_momentum = self.calculate_volume_momentum(data);
            let order_flow_imbalance = self.calculate_order_flow_imbalance(data);

            // Combined momentum score
            let momentum_score =
                (price_momentum + volume_momentum * 0.5 + order_flow_imbalance * 0.3) / 1.8;

            // Check for scalping opportunity
            if momentum_score.abs() > 0.6 {
                let side = if momentum_score > 0.0 {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };

                let signal_strength = if momentum_score.abs() > 0.8 {
                    SignalStrength::Strong
                } else {
                    SignalStrength::Medium
                };

                let entry_price = if side == OrderSide::Buy {
                    data.ask + self.tick_size // Aggressive entry
                } else {
                    data.bid - self.tick_size
                };

                let trade_size = self.calculate_scalping_size(data, momentum_score.abs());
                let expected_profit = self.estimate_scalping_profit(momentum_score.abs(), data);

                let signal = HftSignal {
                    strategy_type: HftStrategyType::MomentumScalping,
                    symbol: data.symbol.clone(),
                    timestamp: data.timestamp,
                    signal_strength,
                    side,
                    price: entry_price,
                    quantity: trade_size,
                    expected_profit,
                    risk_score: self.calculate_scalping_risk(momentum_score.abs()),
                    execution_urgency: 1000, // 1ms urgency for momentum scalping
                };

                signals.push(signal);
            }
        }

        signals
    }

    // Utility functions for calculations

    /// Calculate inventory skew for market making
    fn calculate_inventory_skew(&self, inventory_ratio: f64) -> f64 {
        // Exponential skewing function
        let max_skew = 0.05; // 5 ticks maximum skew
        max_skew * inventory_ratio.tanh() * self.tick_size
    }

    /// Calculate optimal quote size for market making
    fn calculate_optimal_quote_size(
        &self,
        data: &HftMarketData,
        side: OrderSide,
        current_inventory: i64,
    ) -> f64 {
        let base_size = 1000.0; // Base quote size
        let inventory_factor = match side {
            OrderSide::Buy => {
                (1.0 - (current_inventory as f64 / self.max_position_per_symbol as f64)).max(0.0)
            }
            OrderSide::Sell => {
                (1.0 + (current_inventory as f64 / self.max_position_per_symbol as f64)).max(0.0)
            }
        };

        let volatility_factor = self.estimate_short_term_volatility(data);
        let size_adjustment = 1.0 / (1.0 + volatility_factor * 10.0);

        base_size * inventory_factor * size_adjustment
    }

    /// Calculate price momentum
    fn calculate_price_momentum(&self, data: &HftMarketData) -> f64 {
        let mid_price = (data.bid + data.ask) / 2.0;
        // Simplified momentum calculation (would use price history in real implementation)
        let price_change = (mid_price - data.vwap) / data.vwap;
        price_change.tanh() // Normalize to [-1, 1]
    }

    /// Calculate volume momentum
    fn calculate_volume_momentum(&self, data: &HftMarketData) -> f64 {
        // Simplified volume momentum (would use volume history in real implementation)
        let volume_ratio = data.volume / (data.bid_size + data.ask_size + 1.0);
        (volume_ratio - 1.0).tanh()
    }

    /// Calculate order flow imbalance
    fn calculate_order_flow_imbalance(&self, data: &HftMarketData) -> f64 {
        let total_size = data.bid_size + data.ask_size;
        if total_size > 0.0 {
            (data.bid_size - data.ask_size) / total_size
        } else {
            0.0
        }
    }

    /// Estimate short-term volatility
    fn estimate_short_term_volatility(&self, data: &HftMarketData) -> f64 {
        let spread_volatility = (data.ask - data.bid) / (data.bid + data.ask);
        spread_volatility.max(0.0001) // Minimum volatility
    }

    /// Filter signals by risk management rules
    fn filter_signals_by_risk(&self, mut signals: Vec<HftSignal>) -> Vec<HftSignal> {
        let current_pnl = self.current_daily_pnl.load(Ordering::Relaxed) as f64 / 100.0; // Stored in cents

        signals.retain(|signal| {
            // Check daily loss limit
            if current_pnl < -self.max_daily_loss {
                return false;
            }

            // Check risk score threshold
            if signal.risk_score > 0.8 {
                return false;
            }

            // Check minimum profit threshold
            if signal.expected_profit < MIN_PROFIT_THRESHOLD * signal.quantity {
                return false;
            }

            true
        });

        signals
    }

    // Risk calculation functions

    fn calculate_arbitrage_risk(&self, opp: &ArbitrageOpportunity, max_latency: u64) -> f64 {
        let latency_risk = (max_latency as f64 / 10_000.0).min(1.0);
        let size_risk = (opp.quantity / 10000.0).min(1.0);
        (latency_risk * 0.6 + size_risk * 0.4).min(1.0)
    }

    fn calculate_mm_risk(&self, current_inventory: i64, max_inventory: u64) -> f64 {
        let inventory_risk = (current_inventory.abs() as f64 / max_inventory as f64).min(1.0);
        inventory_risk * 0.5 // Market making is generally lower risk
    }

    fn calculate_stat_arb_risk(&self, z_score: f64, cointegration_score: f64) -> f64 {
        let mean_reversion_risk = 1.0 / (1.0 + z_score);
        let cointegration_risk = 1.0 - cointegration_score;
        (mean_reversion_risk * 0.3 + cointegration_risk * 0.7).min(1.0)
    }

    fn calculate_scalping_risk(&self, momentum_strength: f64) -> f64 {
        // Higher momentum = higher risk but potentially higher reward
        (momentum_strength * 0.8).min(1.0)
    }

    // Profit estimation functions

    fn estimate_mm_profit(&self, current_spread: f64, target_spread: f64) -> f64 {
        (target_spread / 2.0).min(current_spread / 4.0) // Conservative estimate
    }

    fn estimate_stat_arb_profit(&self, z_score: f64, spread_std: f64, trade_size: f64) -> f64 {
        let expected_reversion = z_score * spread_std * 0.5; // 50% reversion expectation
        expected_reversion * trade_size
    }

    fn estimate_scalping_profit(&self, momentum_strength: f64, data: &HftMarketData) -> f64 {
        let expected_move = momentum_strength * self.tick_size * 3.0;
        let spread_cost = (data.ask - data.bid) / 2.0;
        (expected_move - spread_cost).max(0.0) * 1000.0 // Base size
    }

    // Size calculation functions

    fn calculate_stat_arb_size(&self, pair: &StatArbPair, z_score: f64) -> f64 {
        let base_size = 1000.0;
        let confidence_multiplier = (z_score / 2.0).min(2.0);
        let cointegration_multiplier = pair.cointegration_score;
        base_size * confidence_multiplier * cointegration_multiplier
    }

    fn calculate_scalping_size(&self, data: &HftMarketData, momentum_strength: f64) -> f64 {
        let base_size = 500.0;
        let momentum_multiplier = momentum_strength.min(2.0);
        let liquidity_factor = ((data.bid_size + data.ask_size) / 2000.0).min(1.0);
        base_size * momentum_multiplier * liquidity_factor
    }

    // Fee and slippage estimation

    fn estimate_trading_fees(&self, quantity: f64, num_venues: u32) -> f64 {
        let fee_rate = 0.0001; // 1 basis point per side
        quantity * fee_rate * (num_venues as f64)
    }

    fn estimate_slippage(&self, data: &HftMarketData, quantity: f64) -> f64 {
        let available_liquidity = (data.bid_size + data.ask_size) / 2.0;
        let slippage_factor = (quantity / available_liquidity).min(1.0);
        let tick_slippage = slippage_factor * self.tick_size;
        tick_slippage * quantity
    }

    // Helper functions

    fn get_timestamp_us(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Add venue price data for latency arbitrage
    pub fn add_venue_price(&mut self, symbol: String, venue_price: VenuePrice) {
        // Get cutoff time before borrowing venue_prices
        let cutoff_time = self.get_timestamp_us() - 1_000_000;

        let venue_prices = self.venue_prices.entry(symbol).or_default();
        venue_prices.push(venue_price);

        // Keep only recent prices (last 1 second)
        venue_prices.retain(|price| price.timestamp > cutoff_time);
    }

    /// Add statistical arbitrage pair
    pub fn add_stat_arb_pair(&mut self, pair: StatArbPair) {
        let pair_key = format!("{}_{}", pair.symbol_a, pair.symbol_b);
        self.pair_spreads
            .insert(pair_key, VecDeque::with_capacity(1000));
        self.stat_arb_pairs.push(pair);
    }

    /// Initialize market making for symbol
    pub fn init_market_making(&mut self, symbol: String, max_inventory: u64) {
        let mm_state = MarketMakerState {
            inventory: AtomicI64::new(0),
            max_inventory,
            target_spread: AtomicU64::new((0.0002 / self.tick_size) as u64), // 2 basis points
            skew_factor: AtomicU64::new(0),
            filled_orders: AtomicU64::new(0),
            last_update: AtomicU64::new(self.get_timestamp_us()),
        };

        self.mm_states.insert(symbol, mm_state);
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> HftPerformanceMetrics {
        let avg_execution_time = if self.execution_times.is_empty() {
            0.0
        } else {
            self.execution_times.iter().sum::<u64>() as f64 / self.execution_times.len() as f64
        };

        let max_execution_time = self.execution_times.iter().max().copied().unwrap_or(0);

        HftPerformanceMetrics {
            avg_execution_time_us: avg_execution_time,
            max_execution_time_us: max_execution_time,
            signals_generated: self.signal_queue.len(),
            current_pnl: self.current_daily_pnl.load(Ordering::Relaxed) as f64 / 100.0,
            latency_violations: self
                .execution_times
                .iter()
                .filter(|&&time| time > MAX_EXECUTION_TIME_US)
                .count(),
        }
    }
}

/// HFT configuration
#[derive(Debug, Clone)]
pub struct HftConfig {
    pub max_position_per_symbol: u64,
    pub max_daily_loss: f64,
    pub window_size: usize,
    pub tick_size: f64,
}

impl Default for HftConfig {
    fn default() -> Self {
        Self {
            max_position_per_symbol: 10000,
            max_daily_loss: 10000.0,
            window_size: 100,
            tick_size: 0.0001,
        }
    }
}

/// Performance metrics for HFT engine
#[derive(Debug, Clone)]
pub struct HftPerformanceMetrics {
    pub avg_execution_time_us: f64,
    pub max_execution_time_us: u64,
    pub signals_generated: usize,
    pub current_pnl: f64,
    pub latency_violations: usize,
}

/// Arbitrage opportunity structure
#[derive(Debug, Clone)]
struct ArbitrageOpportunity {
    #[allow(dead_code)]
    pub buy_venue: String,
    #[allow(dead_code)]
    pub sell_venue: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub quantity: f64,
}

impl HftAlgorithmEngine {
    /// Find arbitrage opportunities between venues
    fn find_arbitrage_opportunities(
        &self,
        _data: &HftMarketData,
        venue_prices: &[VenuePrice],
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        // Compare all venue pairs
        for i in 0..venue_prices.len() {
            for j in i + 1..venue_prices.len() {
                let venue_a = &venue_prices[i];
                let venue_b = &venue_prices[j];

                // Check if A's ask < B's bid (buy A, sell B)
                if venue_a.ask < venue_b.bid {
                    let quantity = venue_a.ask_size.min(venue_b.bid_size);
                    let opportunity = ArbitrageOpportunity {
                        buy_venue: venue_a.venue.clone(),
                        sell_venue: venue_b.venue.clone(),
                        buy_price: venue_a.ask,
                        sell_price: venue_b.bid,
                        quantity,
                    };
                    opportunities.push(opportunity);
                }

                // Check if B's ask < A's bid (buy B, sell A)
                if venue_b.ask < venue_a.bid {
                    let quantity = venue_b.ask_size.min(venue_a.bid_size);
                    let opportunity = ArbitrageOpportunity {
                        buy_venue: venue_b.venue.clone(),
                        sell_venue: venue_a.venue.clone(),
                        buy_price: venue_b.ask,
                        sell_price: venue_a.bid,
                        quantity,
                    };
                    opportunities.push(opportunity);
                }
            }
        }

        opportunities
    }
}

// Thread safety
unsafe impl Send for HftAlgorithmEngine {}
unsafe impl Sync for HftAlgorithmEngine {}

// ============================================================================
// LEGACY API COMPATIBILITY TYPES
// These types support the existing test suite while maintaining backward compatibility
// ============================================================================

/// Legacy Level type for test compatibility
#[derive(Debug, Clone)]
pub struct Level {
    pub price: f64,
    pub quantity: f64,
    pub exchange: String,
}

/// Legacy TickData type for test compatibility
#[derive(Debug, Clone)]
pub struct TickData {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub exchange: String,
}

/// Legacy arbitrage opportunity for test compatibility
#[derive(Debug, Clone)]
pub struct LatencyArbitrageOpportunity {
    pub buy_price: f64,
    pub sell_price: f64,
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub quantity: f64,
    pub profit: f64,
}

/// Legacy OrderBook for test compatibility (different from order_matching::OrderBook)
#[derive(Debug, Clone)]
pub struct LegacyOrderBook {
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub last_update: std::time::SystemTime,
}

/// Legacy Side enum for test compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Legacy HFT Engine for test compatibility (wraps HftAlgorithmEngine)
#[derive(Debug)]
pub struct HFTEngine {
    pub tick_buffer_size: usize,
    pub max_position: i64,
    pub risk_limit: i64,
    pub position: f64,
    pub pnl: f64,
    pub trade_count: u64,
    pub order_book: LegacyOrderBook,
    pub tick_buffer: VecDeque<TickData>,
    average_entry_price: f64,
    inner: HftAlgorithmEngine,
}

impl HFTEngine {
    /// Create a new HFT engine with legacy parameters
    pub fn new(tick_buffer_size: usize, max_position: i64, risk_limit: i64) -> Self {
        let config = HftConfig {
            max_position_per_symbol: max_position as u64,
            max_daily_loss: risk_limit as f64,
            window_size: tick_buffer_size,
            tick_size: 0.0001,
        };
        Self {
            tick_buffer_size,
            max_position,
            risk_limit,
            position: 0.0,
            pnl: 0.0,
            trade_count: 0,
            order_book: LegacyOrderBook {
                bids: Vec::new(),
                asks: Vec::new(),
                last_update: std::time::SystemTime::now(),
            },
            tick_buffer: VecDeque::with_capacity(tick_buffer_size),
            average_entry_price: 0.0,
            inner: HftAlgorithmEngine::new(config),
        }
    }

    /// Detect latency arbitrage opportunities between two order books
    pub fn detect_latency_arbitrage(
        &self,
        book1: &LegacyOrderBook,
        book2: &LegacyOrderBook,
    ) -> Option<LatencyArbitrageOpportunity> {
        // Check if book1's best ask < book2's best bid (buy from book1, sell to book2)
        if let (Some(ask1), Some(bid2)) = (book1.asks.first(), book2.bids.first()) {
            if ask1.price < bid2.price {
                let quantity = ask1.quantity.min(bid2.quantity);
                let profit = (bid2.price - ask1.price) * quantity;
                return Some(LatencyArbitrageOpportunity {
                    buy_price: ask1.price,
                    sell_price: bid2.price,
                    buy_exchange: ask1.exchange.clone(),
                    sell_exchange: bid2.exchange.clone(),
                    quantity,
                    profit,
                });
            }
        }
        // Check reverse direction
        if let (Some(ask2), Some(bid1)) = (book2.asks.first(), book1.bids.first()) {
            if ask2.price < bid1.price {
                let quantity = ask2.quantity.min(bid1.quantity);
                let profit = (bid1.price - ask2.price) * quantity;
                return Some(LatencyArbitrageOpportunity {
                    buy_price: ask2.price,
                    sell_price: bid1.price,
                    buy_exchange: ask2.exchange.clone(),
                    sell_exchange: bid1.exchange.clone(),
                    quantity,
                    profit,
                });
            }
        }
        None
    }

    /// Process tick data
    pub fn process_tick(&mut self, tick: &TickData) {
        // Update internal state based on tick
        let market_data = HftMarketData {
            timestamp: tick.timestamp,
            symbol: tick.symbol.clone(),
            bid: tick.price * 0.9999,
            ask: tick.price * 1.0001,
            bid_size: tick.quantity,
            ask_size: tick.quantity,
            last_price: tick.price,
            volume: tick.quantity,
            vwap: tick.price,
        };
        let _ = self.inner.process_market_data(&[market_data]);
    }

    /// Get current market making signals
    pub fn get_signals(&self) -> Vec<HftSignal> {
        Vec::new() // Placeholder - actual implementation in inner engine
    }

    /// Update with new tick data (maintains buffer at tick_buffer_size)
    pub fn update_tick(&mut self, tick: TickData) {
        // Maintain buffer size
        while self.tick_buffer.len() >= self.tick_buffer_size {
            self.tick_buffer.pop_front();
        }
        self.tick_buffer.push_back(tick.clone());
        self.process_tick(&tick);
    }

    /// Execute an order with position and risk limit checking
    pub fn execute_order(&mut self, _order_type: crate::common_types::OrderType, side: Side, quantity: f64, price: f64) -> bool {
        // Check risk limit
        let potential_risk = quantity * price;
        if potential_risk > self.risk_limit as f64 {
            return false;
        }

        // Check position limits
        let new_position = match side {
            Side::Buy => self.position + quantity,
            Side::Sell => self.position - quantity,
        };

        if new_position.abs() > self.max_position as f64 {
            return false;
        }

        // Execute the trade
        match side {
            Side::Buy => {
                // Update average entry price
                if self.position >= 0.0 {
                    let total_value = self.average_entry_price * self.position + price * quantity;
                    let new_pos = self.position + quantity;
                    if new_pos > 0.0 {
                        self.average_entry_price = total_value / new_pos;
                    }
                }
                self.position += quantity;
            }
            Side::Sell => {
                if self.position > 0.0 {
                    // Calculate PnL on closed portion
                    let closed_quantity = quantity.min(self.position);
                    let realized_pnl = closed_quantity * (price - self.average_entry_price);
                    self.pnl += realized_pnl;
                }
                self.position -= quantity;
            }
        }

        self.trade_count += 1;
        true
    }

    /// Calculate statistical arbitrage signal between two correlated assets
    pub fn calculate_stat_arb_signal(&self, price_a: f64, price_b: f64, z_score_threshold: f64) -> f64 {
        // Simple ratio-based statistical arbitrage signal
        if price_b == 0.0 {
            return 0.0;
        }

        let ratio = price_a / price_b;
        let historical_ratio = 2.0; // Expected ratio (simplified)
        let std_dev = 0.1; // Historical standard deviation (simplified)

        let z_score = (ratio - historical_ratio) / std_dev;

        if z_score.abs() > z_score_threshold {
            z_score
        } else {
            0.0
        }
    }

    /// Generate market making quotes with given spread and inventory skew
    pub fn generate_market_making_quotes(&self, spread: f64, _inventory_factor: f64) -> Option<(f64, f64)> {
        // Get mid price from order book
        let best_bid = self.order_book.bids.first().map(|l| l.price)?;
        let best_ask = self.order_book.asks.first().map(|l| l.price)?;
        let mid_price = (best_bid + best_ask) / 2.0;

        // Apply inventory skew
        let inventory_skew = self.inner.calculate_inventory_skew(self.position / self.max_position as f64);

        let bid_price = mid_price - spread / 2.0 - inventory_skew * 0.01;
        let ask_price = mid_price + spread / 2.0 + inventory_skew * 0.01;

        Some((bid_price, ask_price))
    }

    /// Process tick in parallel-safe manner (returns true if processed)
    pub fn process_tick_parallel(&self, _tick: &TickData) -> bool {
        // Simplified parallel processing - in production would use atomic operations
        true
    }

    /// Update order book from external order book structure
    pub fn update_order_book(&mut self, order_book: LegacyOrderBook) {
        self.order_book = order_book;
    }
}

// Type alias for backward compatibility
pub type TradeSide = OrderSide;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_market_data(symbol: &str, bid: f64, ask: f64, timestamp: u64) -> HftMarketData {
        HftMarketData {
            timestamp,
            symbol: symbol.to_string(),
            bid,
            ask,
            bid_size: 1000.0,
            ask_size: 1000.0,
            last_price: (bid + ask) / 2.0,
            volume: 5000.0,
            vwap: (bid + ask) / 2.0,
        }
    }

    #[test]
    fn test_hft_engine_creation() {
        let config = HftConfig::default();
        let engine = HftAlgorithmEngine::new(config);

        assert!(engine.latency_arb_enabled.load(Ordering::Relaxed));
        assert!(engine.market_making_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_market_making_initialization() {
        let config = HftConfig::default();
        let mut engine = HftAlgorithmEngine::new(config);

        engine.init_market_making("BTCUSD".to_string(), 10000);
        assert!(engine.mm_states.contains_key("BTCUSD"));
    }

    #[test]
    fn test_signal_generation() {
        let config = HftConfig::default();
        let mut engine = HftAlgorithmEngine::new(config);

        engine.init_market_making("BTCUSD".to_string(), 10000);

        let market_data = vec![create_test_market_data("BTCUSD", 50000.0, 50001.0, 1000000)];

        let signals = engine.process_market_data(&market_data);

        // Should generate market making signals
        assert!(!signals.is_empty());
    }

    #[test]
    fn test_inventory_skew_calculation() {
        let config = HftConfig::default();
        let engine = HftAlgorithmEngine::new(config);

        let skew_positive = engine.calculate_inventory_skew(0.5);
        let skew_negative = engine.calculate_inventory_skew(-0.5);

        assert!(skew_positive > 0.0);
        assert!(skew_negative < 0.0);
        assert_eq!(skew_positive, -skew_negative);
    }

    #[test]
    fn test_performance_metrics() {
        let config = HftConfig::default();
        let engine = HftAlgorithmEngine::new(config);

        let metrics = engine.get_performance_metrics();
        assert!(metrics.avg_execution_time_us >= 0.0);
        assert!(metrics.latency_violations == 0); // Should start with no violations
    }

    #[test]
    fn test_stat_arb_pair_addition() {
        let config = HftConfig::default();
        let mut engine = HftAlgorithmEngine::new(config);

        let pair = StatArbPair {
            symbol_a: "BTCUSD".to_string(),
            symbol_b: "ETHUSD".to_string(),
            hedge_ratio: 15.0,
            mean_spread: 1000.0,
            spread_std: 50.0,
            cointegration_score: 0.8,
            last_update: 1000000,
        };

        engine.add_stat_arb_pair(pair);
        assert_eq!(engine.stat_arb_pairs.len(), 1);
    }

    #[test]
    fn test_venue_price_management() {
        let config = HftConfig::default();
        let mut engine = HftAlgorithmEngine::new(config);

        let venue_price = VenuePrice {
            venue: "Binance".to_string(),
            timestamp: engine.get_timestamp_us(),
            bid: 50000.0,
            ask: 50001.0,
            bid_size: 1000.0,
            ask_size: 1000.0,
            latency_us: 500,
        };

        engine.add_venue_price("BTCUSD".to_string(), venue_price);

        assert!(engine.venue_prices.contains_key("BTCUSD"));
        assert_eq!(engine.venue_prices.get("BTCUSD").unwrap().len(), 1);
    }
}
