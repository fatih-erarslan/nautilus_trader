use crate::common_types::TradeSide;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SlippageError {
    #[error("Invalid order size: {0}")]
    InvalidOrderSize(f64),
    #[error("Insufficient liquidity: requested {requested}, available {available}")]
    InsufficientLiquidity { requested: f64, available: f64 },
    #[error("Invalid price level: {0}")]
    InvalidPriceLevel(f64),
    #[error("Market impact calculation failed: {0}")]
    MarketImpactFailed(String),
    #[error("Historical data insufficient for analysis")]
    InsufficientData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub timestamp: u64,
}

// TradeSide now imported from common_types module

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageAnalysis {
    pub expected_price: f64,
    pub estimated_fill_price: f64,
    pub slippage_bps: f64,
    pub slippage_percentage: f64,
    pub slippage_amount: f64,
    pub market_impact: f64,
    pub price_improvement: f64,
    pub confidence_interval: (f64, f64),
    pub liquidity_score: f64,
    pub execution_cost: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct MarketImpactModel {
    pub temporary_impact_coeff: f64,
    pub permanent_impact_coeff: f64,
    pub volatility_factor: f64,
    pub liquidity_factor: f64,
    pub volume_decay_factor: f64,
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        Self {
            temporary_impact_coeff: 0.5,
            permanent_impact_coeff: 0.1,
            volatility_factor: 0.3,
            liquidity_factor: 0.2,
            volume_decay_factor: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlippageParameters {
    pub historical_window: usize,
    pub confidence_level: f64,
    pub min_liquidity_threshold: f64,
    pub max_market_impact: f64,
    pub volume_participation_limit: f64,
    pub tick_size: f64,
    pub model: MarketImpactModel,
}

impl Default for SlippageParameters {
    fn default() -> Self {
        Self {
            historical_window: 1000,
            confidence_level: 0.95,
            min_liquidity_threshold: 10000.0,
            max_market_impact: 0.05,         // 5%
            volume_participation_limit: 0.1, // 10% of volume
            tick_size: 0.01,
            model: MarketImpactModel::default(),
        }
    }
}

pub struct SlippageCalculator {
    parameters: SlippageParameters,
    order_books: HashMap<String, OrderBook>,
    trade_history: HashMap<String, VecDeque<Trade>>,
    volume_profile: HashMap<String, VecDeque<f64>>,
    volatility_cache: HashMap<String, f64>,
}

impl SlippageCalculator {
    pub fn new(parameters: SlippageParameters) -> Self {
        Self {
            parameters,
            order_books: HashMap::new(),
            trade_history: HashMap::new(),
            volume_profile: HashMap::new(),
            volatility_cache: HashMap::new(),
        }
    }

    /// Update order book data
    pub fn update_order_book(&mut self, order_book: OrderBook) {
        self.order_books
            .insert(order_book.symbol.clone(), order_book);
    }

    /// Add trade to historical data
    pub fn add_trade(&mut self, trade: Trade) {
        let trades = self.trade_history.entry(trade.symbol.clone()).or_default();

        trades.push_back(trade.clone());

        // Maintain window size
        while trades.len() > self.parameters.historical_window {
            trades.pop_front();
        }

        // Update volume profile
        let volumes = self.volume_profile.entry(trade.symbol.clone()).or_default();

        volumes.push_back(trade.quantity);

        while volumes.len() > self.parameters.historical_window {
            volumes.pop_front();
        }

        // Update volatility cache
        self.update_volatility(&trade.symbol);
    }

    /// Calculate slippage for a given order
    pub fn calculate_slippage(
        &self,
        symbol: &str,
        order_size: f64,
        side: &TradeSide,
        expected_price: Option<f64>,
    ) -> Result<SlippageAnalysis, SlippageError> {
        if order_size <= 0.0 {
            return Err(SlippageError::InvalidOrderSize(order_size));
        }

        let order_book = self
            .order_books
            .get(symbol)
            .ok_or(SlippageError::InsufficientData)?;

        let levels = match *side {
            TradeSide::Buy => &order_book.asks,
            TradeSide::Sell => &order_book.bids,
        };

        if levels.is_empty() {
            return Err(SlippageError::InsufficientLiquidity {
                requested: order_size,
                available: 0.0,
            });
        }

        // Calculate volume-weighted average price (VWAP) for execution
        let (estimated_fill_price, total_available) =
            self.calculate_vwap_execution(levels, order_size)?;

        if total_available < order_size {
            return Err(SlippageError::InsufficientLiquidity {
                requested: order_size,
                available: total_available,
            });
        }

        // Use expected price or mid-price
        let reference_price = expected_price.unwrap_or_else(|| {
            let best_bid = order_book.bids.first().map(|l| l.price).unwrap_or(0.0);
            let best_ask = order_book.asks.first().map(|l| l.price).unwrap_or(0.0);
            (best_bid + best_ask) / 2.0
        });

        // Calculate basic slippage metrics
        let slippage_amount = match side {
            TradeSide::Buy => estimated_fill_price - reference_price,
            TradeSide::Sell => reference_price - estimated_fill_price,
        };

        let slippage_percentage = if reference_price != 0.0 {
            (slippage_amount / reference_price) * 100.0
        } else {
            0.0
        };

        let slippage_bps = slippage_percentage * 100.0; // Convert to basis points

        // Calculate market impact
        let market_impact =
            self.calculate_market_impact(symbol, order_size, side, reference_price)?;

        // Calculate price improvement (negative slippage is improvement)
        let price_improvement = if slippage_amount < 0.0 {
            slippage_amount.abs()
        } else {
            0.0
        };

        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(
            symbol,
            slippage_percentage,
            self.parameters.confidence_level,
        )?;

        // Calculate liquidity score
        let liquidity_score = self.calculate_liquidity_score(levels, order_size);

        // Calculate total execution cost
        let execution_cost = slippage_amount + market_impact;

        Ok(SlippageAnalysis {
            expected_price: reference_price,
            estimated_fill_price,
            slippage_bps,
            slippage_percentage,
            slippage_amount,
            market_impact,
            price_improvement,
            confidence_interval,
            liquidity_score,
            execution_cost,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Calculate volume-weighted average price for order execution
    fn calculate_vwap_execution(
        &self,
        levels: &[OrderBookLevel],
        order_size: f64,
    ) -> Result<(f64, f64), SlippageError> {
        let mut remaining_size = order_size;
        let mut total_cost = 0.0;
        let mut total_filled = 0.0;
        let mut total_available = 0.0;

        for level in levels {
            total_available += level.quantity;

            if remaining_size <= 0.0 {
                break;
            }

            let fill_quantity = remaining_size.min(level.quantity);
            total_cost += fill_quantity * level.price;
            total_filled += fill_quantity;
            remaining_size -= fill_quantity;
        }

        if total_filled == 0.0 {
            return Err(SlippageError::InsufficientLiquidity {
                requested: order_size,
                available: total_available,
            });
        }

        let vwap = total_cost / total_filled;
        Ok((vwap, total_available))
    }

    /// Calculate market impact using advanced models
    fn calculate_market_impact(
        &self,
        symbol: &str,
        order_size: f64,
        _side: &TradeSide,
        reference_price: f64,
    ) -> Result<f64, SlippageError> {
        // Get average daily volume
        let avg_volume = self.calculate_average_volume(symbol)?;

        // Get volatility
        let volatility = self.volatility_cache.get(symbol).copied().unwrap_or(0.02);

        // Calculate participation rate
        let participation_rate = order_size / avg_volume;

        // Square root model for temporary impact
        let temporary_impact =
            self.parameters.model.temporary_impact_coeff * volatility * participation_rate.sqrt();

        // Linear model for permanent impact
        let permanent_impact = self.parameters.model.permanent_impact_coeff * participation_rate;

        // Total market impact
        let total_impact_percentage = temporary_impact + permanent_impact;

        // Apply volatility and liquidity adjustments
        let volatility_adjustment = volatility * self.parameters.model.volatility_factor;
        let liquidity_adjustment = self.calculate_liquidity_adjustment(symbol)?;

        let adjusted_impact =
            total_impact_percentage * (1.0 + volatility_adjustment) * (1.0 + liquidity_adjustment);

        // Convert to price impact
        let price_impact = reference_price * adjusted_impact;

        Ok(price_impact.min(reference_price * self.parameters.max_market_impact))
    }

    /// Calculate confidence interval for slippage estimate
    fn calculate_confidence_interval(
        &self,
        symbol: &str,
        slippage_percentage: f64,
        confidence_level: f64,
    ) -> Result<(f64, f64), SlippageError> {
        let trades = self
            .trade_history
            .get(symbol)
            .ok_or(SlippageError::InsufficientData)?;

        if trades.len() < 30 {
            return Err(SlippageError::InsufficientData);
        }

        // Calculate historical slippage distribution
        let mut slippage_samples = Vec::new();
        let window_size = 10;

        for window in trades.iter().collect::<Vec<_>>().windows(window_size) {
            if let (Some(first), Some(last)) = (window.first(), window.last()) {
                let price_change = (last.price - first.price) / first.price * 100.0;
                slippage_samples.push(price_change);
            }
        }

        if slippage_samples.is_empty() {
            return Ok((slippage_percentage, slippage_percentage));
        }

        // Calculate standard deviation
        let mean = slippage_samples.iter().sum::<f64>() / slippage_samples.len() as f64;
        let variance = slippage_samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / slippage_samples.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate confidence interval using t-distribution approximation
        let z_score = self.get_z_score(confidence_level);
        let margin_of_error = z_score * std_dev;

        let lower_bound = slippage_percentage - margin_of_error;
        let upper_bound = slippage_percentage + margin_of_error;

        Ok((lower_bound, upper_bound))
    }

    /// Calculate liquidity score based on order book depth
    fn calculate_liquidity_score(&self, levels: &[OrderBookLevel], order_size: f64) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }

        let total_liquidity: f64 = levels.iter().map(|l| l.quantity).sum();
        let depth_at_levels = levels.len() as f64;

        // Score based on total liquidity vs order size
        let liquidity_ratio = total_liquidity / order_size;

        // Score based on depth (number of price levels)
        let depth_score = (depth_at_levels / 20.0).min(1.0); // Normalize to max 20 levels

        // Combined score (0-1)
        let base_score = (liquidity_ratio / 10.0).min(1.0); // Good if 10x the order size

        (base_score * 0.7 + depth_score * 0.3).min(1.0)
    }

    /// Estimate dynamic slippage based on market conditions
    pub fn estimate_dynamic_slippage(
        &self,
        symbol: &str,
        order_size: f64,
        side: TradeSide,
        time_horizon_ms: u64,
    ) -> Result<SlippageAnalysis, SlippageError> {
        let base_analysis = self.calculate_slippage(symbol, order_size, &side, None)?;

        // Adjust for time horizon - longer execution time = more slippage
        let time_factor = (time_horizon_ms as f64 / 1000.0).sqrt(); // Square root of seconds
        let time_adjusted_slippage = base_analysis.slippage_percentage * (1.0 + time_factor * 0.1);

        // Adjust for market volatility
        let volatility = self.volatility_cache.get(symbol).copied().unwrap_or(0.02);
        let volatility_adjustment = volatility * 10.0; // Scale volatility impact
        let volatility_adjusted_slippage = time_adjusted_slippage * (1.0 + volatility_adjustment);

        // Calculate adjusted fill price
        let price_adjustment =
            base_analysis.expected_price * (volatility_adjusted_slippage / 100.0);
        let adjusted_fill_price = match side {
            TradeSide::Buy => base_analysis.estimated_fill_price + price_adjustment,
            TradeSide::Sell => base_analysis.estimated_fill_price - price_adjustment,
        };

        Ok(SlippageAnalysis {
            expected_price: base_analysis.expected_price,
            estimated_fill_price: adjusted_fill_price,
            slippage_bps: volatility_adjusted_slippage * 100.0,
            slippage_percentage: volatility_adjusted_slippage,
            slippage_amount: price_adjustment,
            market_impact: base_analysis.market_impact * (1.0 + time_factor * 0.2),
            price_improvement: base_analysis.price_improvement,
            confidence_interval: (
                base_analysis.confidence_interval.0 * (1.0 + volatility_adjustment),
                base_analysis.confidence_interval.1 * (1.0 + volatility_adjustment),
            ),
            liquidity_score: base_analysis.liquidity_score * (1.0 - time_factor * 0.1).max(0.1),
            execution_cost: base_analysis.execution_cost * (1.0 + time_factor * 0.15),
            timestamp: base_analysis.timestamp,
        })
    }

    /// Calculate optimal order splitting to minimize slippage
    pub fn calculate_optimal_order_splits(
        &self,
        symbol: &str,
        total_order_size: f64,
        _side: TradeSide,
        max_participation_rate: f64,
    ) -> Result<Vec<f64>, SlippageError> {
        let avg_volume = self.calculate_average_volume(symbol)?;
        let max_order_size = avg_volume * max_participation_rate;

        if total_order_size <= max_order_size {
            return Ok(vec![total_order_size]);
        }

        let num_splits = (total_order_size / max_order_size).ceil() as usize;
        let base_split_size = total_order_size / num_splits as f64;

        let mut splits = vec![base_split_size; num_splits];

        // Adjust last split for remainder
        let remainder = total_order_size - (base_split_size * (num_splits - 1) as f64);
        if let Some(last) = splits.last_mut() {
            *last = remainder;
        }

        Ok(splits)
    }

    // Helper methods

    fn update_volatility(&mut self, symbol: &str) {
        if let Some(trades) = self.trade_history.get(symbol) {
            if trades.len() < 2 {
                return;
            }

            let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

            if !returns.is_empty() {
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns
                    .iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                let volatility = variance.sqrt();

                self.volatility_cache.insert(symbol.to_string(), volatility);
            }
        }
    }

    fn calculate_average_volume(&self, symbol: &str) -> Result<f64, SlippageError> {
        let volumes = self
            .volume_profile
            .get(symbol)
            .ok_or(SlippageError::InsufficientData)?;

        if volumes.is_empty() {
            return Err(SlippageError::InsufficientData);
        }

        Ok(volumes.iter().sum::<f64>() / volumes.len() as f64)
    }

    fn calculate_liquidity_adjustment(&self, symbol: &str) -> Result<f64, SlippageError> {
        let order_book = self
            .order_books
            .get(symbol)
            .ok_or(SlippageError::InsufficientData)?;

        let total_bid_liquidity: f64 = order_book.bids.iter().map(|l| l.quantity).sum();
        let total_ask_liquidity: f64 = order_book.asks.iter().map(|l| l.quantity).sum();
        let total_liquidity = total_bid_liquidity + total_ask_liquidity;

        // Lower liquidity = higher adjustment (more impact)
        let liquidity_adjustment = if total_liquidity < self.parameters.min_liquidity_threshold {
            (self.parameters.min_liquidity_threshold - total_liquidity)
                / self.parameters.min_liquidity_threshold
        } else {
            0.0
        };

        Ok(liquidity_adjustment)
    }

    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Approximate z-scores for common confidence levels
        match confidence_level {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.96, // Default to 95%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_order_book() -> OrderBook {
        OrderBook {
            symbol: "BTC".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 49950.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 49940.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 49930.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 50050.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 50060.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 50070.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        }
    }

    #[test]
    fn test_vwap_calculation() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_sample_order_book();

        let (vwap, available) = calculator
            .calculate_vwap_execution(&order_book.asks, 2.0)
            .unwrap();

        // Should execute 1.0 at 50050.0 and 1.0 at 50060.0
        // VWAP = (1.0 * 50050.0 + 1.0 * 50060.0) / 2.0 = 50055.0
        assert_eq!(vwap, 50055.0);
        assert_eq!(available, 8.0); // Total ask liquidity
    }

    #[test]
    fn test_slippage_calculation() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_sample_order_book();
        calculator.update_order_book(order_book);

        let analysis = calculator
            .calculate_slippage("BTC", 1.5, TradeSide::Buy, Some(50000.0))
            .unwrap();

        // Should have positive slippage (paying more than expected)
        assert!(analysis.slippage_amount > 0.0);
        assert!(analysis.estimated_fill_price > 50000.0);
    }

    #[test]
    fn test_liquidity_score() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());
        let levels = vec![
            OrderBookLevel {
                price: 100.0,
                quantity: 10.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 101.0,
                quantity: 20.0,
                timestamp: 0,
            },
        ];

        let score = calculator.calculate_liquidity_score(&levels, 5.0);

        // Should have high liquidity score (30 units available vs 5 needed)
        assert!(score > 0.5);
    }

    #[test]
    fn test_order_splitting() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Add volume data
        let volumes = vec![100.0, 120.0, 110.0, 130.0, 90.0];
        for volume in volumes {
            calculator
                .volume_profile
                .entry("BTC".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(volume);
        }

        let splits = calculator
            .calculate_optimal_order_splits(
                "BTC",
                50.0, // Total order size
                TradeSide::Buy,
                0.1, // 10% participation rate
            )
            .unwrap();

        // Average volume = 110, max order size = 11, so should split 50 into multiple orders
        assert!(splits.len() > 1);
        assert_eq!(splits.iter().sum::<f64>(), 50.0);
    }
}
