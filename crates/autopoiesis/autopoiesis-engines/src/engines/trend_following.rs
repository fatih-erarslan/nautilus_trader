//! Trend following trading engine implementation

use crate::prelude::*;
use crate::models::{MarketData, Order, OrderSide, OrderType, TimeInForce, OrderStatus};
use async_trait::async_trait;
use chrono::Utc;
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::VecDeque;
use uuid::Uuid;

/// Trend following engine that identifies and follows market trends
#[derive(Debug, Clone)]
pub struct TrendFollowingEngine {
    /// Engine configuration
    config: TrendFollowingConfig,
    
    /// Price history for trend calculation
    price_history: VecDeque<PricePoint>,
    
    /// Current trend state
    trend_state: TrendState,
    
    /// Current position
    position: Option<Position>,
    
    /// Risk parameters
    risk_params: RiskParams,
}

#[derive(Debug, Clone)]
pub struct TrendFollowingConfig {
    /// Short-term moving average period
    pub short_ma_period: usize,
    
    /// Long-term moving average period  
    pub long_ma_period: usize,
    
    /// Minimum trend strength threshold
    pub trend_threshold: f64,
    
    /// Maximum position size
    pub max_position_size: Decimal,
    
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    
    /// Take profit percentage
    pub take_profit_pct: f64,
}

#[derive(Debug, Clone)]
struct PricePoint {
    timestamp: chrono::DateTime<Utc>,
    price: Decimal,
    volume: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
enum TrendState {
    Uptrend,
    Downtrend,
    Sideways,
    Uncertain,
}

#[derive(Debug, Clone)]
struct RiskParams {
    max_risk_per_trade: f64,
    max_daily_loss: f64,
    position_sizing_method: PositionSizingMethod,
}

#[derive(Debug, Clone)]
enum PositionSizingMethod {
    FixedAmount(Decimal),
    PercentOfEquity(f64),
    VolatilityAdjusted,
}

impl Default for TrendFollowingConfig {
    fn default() -> Self {
        Self {
            short_ma_period: 10,
            long_ma_period: 30,
            trend_threshold: 0.01,
            max_position_size: Decimal::from(10000),
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
        }
    }
}

impl Default for RiskParams {
    fn default() -> Self {
        Self {
            max_risk_per_trade: 0.02,
            max_daily_loss: 0.05,
            position_sizing_method: PositionSizingMethod::PercentOfEquity(0.1),
        }
    }
}

impl TrendFollowingEngine {
    /// Create a new trend following engine
    pub fn new(config: TrendFollowingConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            trend_state: TrendState::Uncertain,
            position: None,
            risk_params: RiskParams::default(),
        }
    }

    /// Process new market data and generate trading signals
    pub async fn process_market_data(&mut self, market_data: &MarketData) -> Result<Vec<Order>> {
        // Update price history
        self.update_price_history(market_data);
        
        // Calculate trend indicators
        let trend_signal = self.calculate_trend_signal()?;
        
        // Update trend state
        self.update_trend_state(trend_signal);
        
        // Generate trading signals
        self.generate_trading_signals(market_data).await
    }

    fn update_price_history(&mut self, market_data: &MarketData) {
        let price_point = PricePoint {
            timestamp: market_data.timestamp,
            price: market_data.mid,
            volume: market_data.volume_24h,
        };

        self.price_history.push_back(price_point);

        // Keep only necessary history
        let max_history = self.config.long_ma_period * 2;
        while self.price_history.len() > max_history {
            self.price_history.pop_front();
        }
    }

    fn calculate_trend_signal(&self) -> Result<f64> {
        if self.price_history.len() < self.config.long_ma_period {
            return Ok(0.0); // Not enough data
        }

        let short_ma = self.calculate_moving_average(self.config.short_ma_period)?;
        let long_ma = self.calculate_moving_average(self.config.long_ma_period)?;

        // Calculate trend strength as percentage difference
        let trend_strength = (short_ma - long_ma) / long_ma;
        
        Ok(trend_strength.to_f64().unwrap_or(0.0))
    }

    fn calculate_moving_average(&self, period: usize) -> Result<Decimal> {
        if self.price_history.len() < period {
            return Err(Error::Analysis("Insufficient data for moving average".to_string()));
        }

        let sum: Decimal = self.price_history
            .iter()
            .rev()
            .take(period)
            .map(|p| p.price)
            .sum();

        Ok(sum / Decimal::from(period))
    }

    fn update_trend_state(&mut self, trend_signal: f64) {
        self.trend_state = if trend_signal > self.config.trend_threshold {
            TrendState::Uptrend
        } else if trend_signal < -self.config.trend_threshold {
            TrendState::Downtrend
        } else if trend_signal.abs() < self.config.trend_threshold / 2.0 {
            TrendState::Sideways
        } else {
            TrendState::Uncertain
        };
    }

    async fn generate_trading_signals(&mut self, market_data: &MarketData) -> Result<Vec<Order>> {
        let mut orders = Vec::new();

        match (&self.trend_state, &self.position) {
            // Enter long position on uptrend
            (TrendState::Uptrend, None) => {
                let order = self.create_entry_order(market_data, OrderSide::Buy)?;
                orders.push(order);
            }
            
            // Enter short position on downtrend
            (TrendState::Downtrend, None) => {
                let order = self.create_entry_order(market_data, OrderSide::Sell)?;
                orders.push(order);
            }
            
            // Exit position on trend reversal
            (TrendState::Downtrend, Some(pos)) if pos.side == crate::models::PositionSide::Long => {
                let order = self.create_exit_order(market_data, pos)?;
                orders.push(order);
            }
            
            (TrendState::Uptrend, Some(pos)) if pos.side == crate::models::PositionSide::Short => {
                let order = self.create_exit_order(market_data, pos)?;
                orders.push(order);
            }
            
            // Check stop loss and take profit
            (_, Some(pos)) => {
                if let Some(order) = self.check_exit_conditions(market_data, pos)? {
                    orders.push(order);
                }
            }
            
            _ => {} // No action needed
        }

        Ok(orders)
    }

    fn create_entry_order(&self, market_data: &MarketData, side: OrderSide) -> Result<Order> {
        let quantity = self.calculate_position_size(market_data)?;
        
        Ok(Order {
            id: Uuid::new_v4(),
            symbol: market_data.symbol.clone(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }

    fn create_exit_order(&self, market_data: &MarketData, position: &Position) -> Result<Order> {
        let side = match position.side {
            crate::models::PositionSide::Long => OrderSide::Sell,
            crate::models::PositionSide::Short => OrderSide::Buy,
        };

        Ok(Order {
            id: Uuid::new_v4(),
            symbol: market_data.symbol.clone(),
            side,
            order_type: OrderType::Market,
            quantity: position.quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }

    fn calculate_position_size(&self, market_data: &MarketData) -> Result<Decimal> {
        match &self.risk_params.position_sizing_method {
            PositionSizingMethod::FixedAmount(amount) => Ok(*amount),
            PositionSizingMethod::PercentOfEquity(pct) => {
                // Simplified - would need account equity in real implementation
                let estimated_equity = Decimal::from(100000); // $100k default
                Ok(estimated_equity * Decimal::from_f64_retain(*pct).unwrap_or_default())
            }
            PositionSizingMethod::VolatilityAdjusted => {
                let volatility = self.calculate_volatility()?;
                let base_size = Decimal::from(10000);
                let vol_factor = Decimal::from_f64_retain(1.0 / (volatility + 0.01)).unwrap_or(Decimal::ONE);
                Ok(base_size * vol_factor)
            }
        }
    }

    fn calculate_volatility(&self) -> Result<f64> {
        if self.price_history.len() < 20 {
            return Ok(0.02); // Default volatility
        }

        let returns: Vec<f64> = self.price_history
            .iter()
            .zip(self.price_history.iter().skip(1))
            .map(|(prev, curr)| {
                let ret = (curr.price - prev.price) / prev.price;
                ret.to_f64().unwrap_or(0.0)
            })
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    fn check_exit_conditions(&self, market_data: &MarketData, position: &Position) -> Result<Option<Order>> {
        let current_price = market_data.mid;
        let entry_price = position.entry_price;
        
        let price_change_pct = match position.side {
            crate::models::PositionSide::Long => (current_price - entry_price) / entry_price,
            crate::models::PositionSide::Short => (entry_price - current_price) / entry_price,
        };

        let price_change_f64 = price_change_pct.to_f64().unwrap_or(0.0);

        // Check stop loss
        if price_change_f64 < -self.config.stop_loss_pct {
            return Ok(Some(self.create_exit_order(market_data, position)?));
        }

        // Check take profit
        if price_change_f64 > self.config.take_profit_pct {
            return Ok(Some(self.create_exit_order(market_data, position)?));
        }

        Ok(None)
    }

    /// Get current trend state
    pub fn trend_state(&self) -> &TrendState {
        &self.trend_state
    }

    /// Get current position
    pub fn position(&self) -> Option<&Position> {
        self.position.as_ref()
    }

    /// Update position after trade execution
    pub fn update_position(&mut self, trade: &crate::models::Trade) {
        // Implementation would update internal position tracking
        // This is simplified for compilation
    }

    /// Get engine configuration
    pub fn config(&self) -> &TrendFollowingConfig {
        &self.config
    }

    /// Update engine configuration
    pub fn update_config(&mut self, config: TrendFollowingConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_trend_following_engine_creation() {
        let config = TrendFollowingConfig::default();
        let engine = TrendFollowingEngine::new(config);
        
        assert_eq!(engine.trend_state, TrendState::Uncertain);
        assert!(engine.position.is_none());
    }

    #[tokio::test]
    async fn test_market_data_processing() {
        let mut engine = TrendFollowingEngine::new(TrendFollowingConfig::default());
        
        let market_data = MarketData {
            symbol: "BTC/USD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(50000),
            ask: dec!(50001),
            mid: dec!(50000.5),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = engine.process_market_data(&market_data).await;
        assert!(result.is_ok());
    }
}