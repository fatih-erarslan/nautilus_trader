//! Core types and data structures for Tengri trading strategy
//! 
//! Provides common data types used across all components including
//! market data, trading signals, portfolio metrics, and system events.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use uuid::Uuid;

/// Trading instrument identifier
pub type InstrumentId = String;

/// Order identifier  
pub type OrderId = String;

/// Trade identifier
pub type TradeId = String;

/// Market price data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    
    /// Current price
    pub price: f64,
    
    /// Timestamp of price update
    pub timestamp: DateTime<Utc>,
    
    /// Data source (e.g., "binance", "databento", "tardis")
    pub source: String,
    
    /// 24-hour trading volume
    pub volume_24h: Option<f64>,
    
    /// Current bid price
    pub bid: Option<f64>,
    
    /// Current ask price
    pub ask: Option<f64>,
}

/// Trade execution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    /// Trading symbol
    pub symbol: String,
    
    /// Trade execution price
    pub price: f64,
    
    /// Trade quantity
    pub quantity: f64,
    
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Trade side ("buy" or "sell")
    pub side: String,
    
    /// Unique trade identifier
    pub trade_id: Option<String>,
    
    /// Data source
    pub source: String,
}

/// Order book price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price at this level
    pub price: f64,
    
    /// Quantity available at this price
    pub quantity: f64,
}

/// Complete order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookData {
    /// Trading symbol
    pub symbol: String,
    
    /// Bid price levels (buyers)
    pub bids: Vec<PriceLevel>,
    
    /// Ask price levels (sellers)
    pub asks: Vec<PriceLevel>,
    
    /// Timestamp of order book snapshot
    pub timestamp: DateTime<Utc>,
    
    /// Data source
    pub source: String,
}

/// Polymarket prediction market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketData {
    /// Unique market identifier
    pub market_id: String,
    
    /// Market question/description
    pub question: String,
    
    /// Price for "Yes" outcome
    pub yes_price: f64,
    
    /// Price for "No" outcome
    pub no_price: f64,
    
    /// Total trading volume
    pub volume: f64,
    
    /// Available liquidity
    pub liquidity: f64,
    
    /// Data timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Market category
    pub category: String,
}

/// Technical analysis signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal identifier
    pub id: Uuid,
    
    /// Trading symbol
    pub symbol: String,
    
    /// Signal type ("BUY", "SELL", "HOLD")
    pub signal_type: SignalType,
    
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Signal generation timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Signal source/strategy
    pub source: String,
    
    /// Additional signal metadata
    pub metadata: HashMap<String, f64>,
    
    /// Signal expiry time
    pub expires_at: Option<DateTime<Utc>>,
}

/// Signal type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
}

/// Position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position identifier
    pub id: Uuid,
    
    /// Trading symbol
    pub symbol: String,
    
    /// Position side ("LONG" or "SHORT")
    pub side: PositionSide,
    
    /// Position size
    pub size: f64,
    
    /// Entry price
    pub entry_price: f64,
    
    /// Current market price
    pub current_price: f64,
    
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    
    /// Position opening timestamp
    pub opened_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Stop loss price
    pub stop_loss: Option<f64>,
    
    /// Take profit price
    pub take_profit: Option<f64>,
}

/// Position side enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Order information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Order identifier
    pub id: OrderId,
    
    /// Trading symbol
    pub symbol: String,
    
    /// Order type
    pub order_type: OrderType,
    
    /// Order side
    pub side: OrderSide,
    
    /// Order quantity
    pub quantity: f64,
    
    /// Order price (for limit orders)
    pub price: Option<f64>,
    
    /// Order status
    pub status: OrderStatus,
    
    /// Filled quantity
    pub filled_quantity: f64,
    
    /// Average fill price
    pub average_price: Option<f64>,
    
    /// Order creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Order update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Time in force
    pub time_in_force: TimeInForce,
}

/// Order type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLossLimit,
    TakeProfit,
    TakeProfitLimit,
}

/// Order side enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
}

/// Time in force enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeInForce {
    GTC, // Good Till Canceled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD, // Good Till Date
}

/// Portfolio metrics and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    /// Total portfolio value
    pub total_value: f64,
    
    /// Available cash balance
    pub cash_balance: f64,
    
    /// Total unrealized P&L
    pub unrealized_pnl: f64,
    
    /// Total realized P&L (today)
    pub realized_pnl: f64,
    
    /// Portfolio return (percentage)
    pub return_percentage: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Sharpe ratio
    pub sharpe_ratio: Option<f64>,
    
    /// Value at Risk (95%)
    pub var_95: f64,
    
    /// Number of open positions
    pub open_positions: usize,
    
    /// Portfolio volatility
    pub volatility: f64,
    
    /// Beta to market
    pub beta: Option<f64>,
    
    /// Metrics calculation timestamp
    pub timestamp: DateTime<Utc>,
}

/// Risk metrics for portfolio management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Current leverage ratio
    pub leverage: f64,
    
    /// Margin utilization percentage
    pub margin_usage: f64,
    
    /// Position concentration (largest position %)
    pub concentration: f64,
    
    /// Correlation with market
    pub market_correlation: f64,
    
    /// Risk-adjusted return
    pub risk_adjusted_return: f64,
    
    /// Maximum allowed loss
    pub max_loss_limit: f64,
    
    /// Current daily loss
    pub current_daily_loss: f64,
    
    /// Risk score (0-100)
    pub risk_score: f64,
    
    /// Timestamp of risk calculation
    pub timestamp: DateTime<Utc>,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    /// Data source name
    pub source: String,
    
    /// Data freshness (seconds since last update)
    pub freshness: u64,
    
    /// Data completeness (0.0 to 1.0)
    pub completeness: f64,
    
    /// Data accuracy score (0.0 to 1.0)
    pub accuracy: f64,
    
    /// Average latency (milliseconds)
    pub latency_ms: u64,
    
    /// Number of data points received
    pub data_points: u64,
    
    /// Number of errors encountered
    pub error_count: u64,
    
    /// Quality assessment timestamp
    pub timestamp: DateTime<Utc>,
}

/// Market volatility metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityMetrics {
    /// Trading symbol
    pub symbol: String,
    
    /// Current volatility (annualized)
    pub current_volatility: f64,
    
    /// 30-day historical volatility
    pub volatility_30d: f64,
    
    /// Volatility percentile (0-100)
    pub volatility_percentile: f64,
    
    /// Implied volatility (if available)
    pub implied_volatility: Option<f64>,
    
    /// VIX correlation (if applicable)
    pub vix_correlation: Option<f64>,
    
    /// Calculation timestamp
    pub timestamp: DateTime<Utc>,
}

/// Performance attribution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAttribution {
    /// Strategy component name
    pub component: String,
    
    /// P&L contribution
    pub pnl_contribution: f64,
    
    /// Return contribution (%)
    pub return_contribution: f64,
    
    /// Risk contribution (%)
    pub risk_contribution: f64,
    
    /// Number of trades
    pub trade_count: u64,
    
    /// Win rate (%)
    pub win_rate: f64,
    
    /// Average trade P&L
    pub avg_trade_pnl: f64,
    
    /// Attribution period start
    pub period_start: DateTime<Utc>,
    
    /// Attribution period end
    pub period_end: DateTime<Utc>,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    
    /// Memory usage (MB)
    pub memory_usage: f64,
    
    /// GPU memory usage (MB, if available)
    pub gpu_memory_usage: Option<f64>,
    
    /// Network latency (ms)
    pub network_latency: f64,
    
    /// Processing throughput (events/second)
    pub throughput: f64,
    
    /// Error rate (errors/minute)
    pub error_rate: f64,
    
    /// System uptime (seconds)
    pub uptime: u64,
    
    /// Metrics timestamp
    pub timestamp: DateTime<Utc>,
}

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    Trending,
    Sideways,
    Volatile,
    LowVolatility,
    BullMarket,
    BearMarket,
    Unknown,
}

/// Market state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Current market regime
    pub regime: MarketRegime,
    
    /// Market trend direction (-1.0 to 1.0)
    pub trend_strength: f64,
    
    /// Volatility regime (low/medium/high)
    pub volatility_regime: String,
    
    /// Market sentiment score (-1.0 to 1.0)
    pub sentiment: f64,
    
    /// Fear & Greed index (0-100)
    pub fear_greed_index: Option<f64>,
    
    /// Market liquidity score (0.0 to 1.0)
    pub liquidity_score: f64,
    
    /// State determination timestamp
    pub timestamp: DateTime<Utc>,
}

/// Trading session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSession {
    /// Session identifier
    pub id: Uuid,
    
    /// Session start time
    pub start_time: DateTime<Utc>,
    
    /// Session end time (if completed)
    pub end_time: Option<DateTime<Utc>>,
    
    /// Session P&L
    pub session_pnl: f64,
    
    /// Number of trades in session
    pub trade_count: u64,
    
    /// Win rate for session
    pub win_rate: f64,
    
    /// Maximum drawdown in session
    pub max_drawdown: f64,
    
    /// Session notes
    pub notes: Option<String>,
}

impl Default for TradingSignal {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            symbol: String::new(),
            signal_type: SignalType::Hold,
            strength: 0.0,
            confidence: 0.0,
            timestamp: Utc::now(),
            source: String::new(),
            metadata: HashMap::new(),
            expires_at: None,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            symbol: String::new(),
            side: PositionSide::Long,
            size: 0.0,
            entry_price: 0.0,
            current_price: 0.0,
            unrealized_pnl: 0.0,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
            stop_loss: None,
            take_profit: None,
        }
    }
}

impl Default for Order {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            symbol: String::new(),
            order_type: OrderType::Market,
            side: OrderSide::Buy,
            quantity: 0.0,
            price: None,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            average_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            time_in_force: TimeInForce::GTC,
        }
    }
}

impl Default for PortfolioMetrics {
    fn default() -> Self {
        Self {
            total_value: 0.0,
            cash_balance: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            return_percentage: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: None,
            var_95: 0.0,
            open_positions: 0,
            volatility: 0.0,
            beta: None,
            timestamp: Utc::now(),
        }
    }
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            leverage: 1.0,
            margin_usage: 0.0,
            concentration: 0.0,
            market_correlation: 0.0,
            risk_adjusted_return: 0.0,
            max_loss_limit: 0.05,
            current_daily_loss: 0.0,
            risk_score: 0.0,
            timestamp: Utc::now(),
        }
    }
}

impl Default for DataQuality {
    fn default() -> Self {
        Self {
            source: String::new(),
            freshness: 0,
            completeness: 1.0,
            accuracy: 1.0,
            latency_ms: 0,
            data_points: 0,
            error_count: 0,
            timestamp: Utc::now(),
        }
    }
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            regime: MarketRegime::Unknown,
            trend_strength: 0.0,
            volatility_regime: "medium".to_string(),
            sentiment: 0.0,
            fear_greed_index: None,
            liquidity_score: 0.5,
            timestamp: Utc::now(),
        }
    }
}

// Utility functions for type conversions and validations

impl TradingSignal {
    /// Check if signal is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        match self.expires_at {
            Some(expiry) => Utc::now() < expiry,
            None => true,
        }
    }
    
    /// Get signal age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.timestamp).num_seconds()
    }
}

impl Position {
    /// Calculate current P&L for the position
    pub fn calculate_pnl(&self) -> f64 {
        match self.side {
            PositionSide::Long => (self.current_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - self.current_price) * self.size,
        }
    }
    
    /// Calculate position value
    pub fn value(&self) -> f64 {
        self.current_price * self.size
    }
    
    /// Check if position should be closed based on stop loss/take profit
    pub fn should_close(&self) -> Option<String> {
        let _current_pnl = self.calculate_pnl();
        
        if let Some(stop_loss) = self.stop_loss {
            if (self.side == PositionSide::Long && self.current_price <= stop_loss) ||
               (self.side == PositionSide::Short && self.current_price >= stop_loss) {
                return Some("stop_loss".to_string());
            }
        }
        
        if let Some(take_profit) = self.take_profit {
            if (self.side == PositionSide::Long && self.current_price >= take_profit) ||
               (self.side == PositionSide::Short && self.current_price <= take_profit) {
                return Some("take_profit".to_string());
            }
        }
        
        None
    }
}

impl Order {
    /// Check if order is active (can be filled)
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::New | OrderStatus::PartiallyFilled)
    }
    
    /// Get remaining quantity to be filled
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }
    
    /// Calculate fill percentage
    pub fn fill_percentage(&self) -> f64 {
        if self.quantity > 0.0 {
            (self.filled_quantity / self.quantity) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal_validity() {
        let mut signal = TradingSignal::default();
        assert!(signal.is_valid());
        
        signal.expires_at = Some(Utc::now() - chrono::Duration::minutes(1));
        assert!(!signal.is_valid());
    }

    #[test]
    fn test_position_pnl_calculation() {
        let mut position = Position::default();
        position.side = PositionSide::Long;
        position.entry_price = 100.0;
        position.current_price = 110.0;
        position.size = 1.0;
        
        assert_eq!(position.calculate_pnl(), 10.0);
        
        position.side = PositionSide::Short;
        assert_eq!(position.calculate_pnl(), -10.0);
    }

    #[test]
    fn test_order_remaining_quantity() {
        let mut order = Order::default();
        order.quantity = 100.0;
        order.filled_quantity = 30.0;
        
        assert_eq!(order.remaining_quantity(), 70.0);
        assert_eq!(order.fill_percentage(), 30.0);
    }
}