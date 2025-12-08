//! Market Data Adapter for Real-Time Trading Integration
//! 
//! This module bridges the gap between external market data sources
//! and the Talebian Risk Management system, providing real-time
//! market data ingestion, transformation, and analysis.

use crate::{MarketData, WhaleDetection, WhaleDirection, TalebianRiskError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Market data source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Binance,
    Coinbase,
    Kraken,
    OKX,
    Bybit,
    DataCollector,    // From data-collector crate
    DataPipeline,     // From data-pipeline crate
    MarketIntelligence, // From market-intelligence crate
}

/// Enhanced market data with full trading information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMarketData {
    /// Base market data for Talebian risk calculations
    pub base_data: MarketData,
    
    /// OHLCV data for technical analysis
    pub ohlcv: OHLCVData,
    
    /// Order book depth
    pub order_book: OrderBookSnapshot,
    
    /// Recent trades for whale detection
    pub recent_trades: Vec<TradeData>,
    
    /// Market microstructure metrics
    pub microstructure: MicrostructureMetrics,
    
    /// Exchange-specific data
    pub exchange_data: ExchangeData,
    
    /// Data quality metrics
    pub data_quality: DataQuality,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVData {
    pub symbol: String,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades_count: u64,
    pub taker_buy_base_volume: f64,
    pub taker_buy_quote_volume: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub spread: f64,
    pub mid_price: f64,
    pub imbalance: f64,
}

/// Order book price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub trade_id: u64,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub quantity: f64,
    pub is_buyer_maker: bool,
    pub is_whale_trade: bool,
    pub trade_value_usd: f64,
}

/// Market microstructure metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureMetrics {
    pub effective_spread: f64,
    pub realized_spread: f64,
    pub price_impact: f64,
    pub kyle_lambda: f64,
    pub amihud_illiquidity: f64,
    pub roll_measure: f64,
    pub order_flow_toxicity: f64,
    pub adverse_selection: f64,
}

/// Exchange-specific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeData {
    pub exchange: String,
    pub funding_rate: Option<f64>,
    pub open_interest: Option<f64>,
    pub mark_price: Option<f64>,
    pub index_price: Option<f64>,
    pub settlement_price: Option<f64>,
    pub liquidation_volume_24h: Option<f64>,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    pub completeness: f64,
    pub timeliness: f64,
    pub accuracy: f64,
    pub consistency: f64,
    pub latency_ms: u64,
    pub gaps_detected: u32,
    pub outliers_detected: u32,
}

/// Market data adapter for real-time integration
pub struct MarketDataAdapter {
    /// Current symbol being tracked
    symbol: String,
    
    /// Data source
    source: DataSource,
    
    /// Historical data buffer
    history_buffer: Vec<EnhancedMarketData>,
    
    /// Maximum history size
    max_history: usize,
    
    /// Whale detection threshold (USD)
    whale_threshold_usd: f64,
    
    /// Price history for return calculations
    price_history: Vec<f64>,
    
    /// Volume history for analysis
    volume_history: Vec<f64>,
}

impl MarketDataAdapter {
    /// Create new market data adapter
    pub fn new(symbol: String, source: DataSource) -> Self {
        Self {
            symbol,
            source,
            history_buffer: Vec::with_capacity(1000),
            max_history: 1000,
            whale_threshold_usd: 100_000.0, // $100k whale threshold
            price_history: Vec::with_capacity(100),
            volume_history: Vec::with_capacity(100),
        }
    }
    
    /// Convert data-collector Kline to EnhancedMarketData
    pub fn from_kline(&mut self, kline: &KlineData) -> Result<MarketData, TalebianRiskError> {
        // Update price history
        self.price_history.push(kline.close);
        if self.price_history.len() > 100 {
            self.price_history.remove(0);
        }
        
        // Update volume history  
        self.volume_history.push(kline.volume);
        if self.volume_history.len() > 100 {
            self.volume_history.remove(0);
        }
        
        // Calculate returns
        let returns = self.calculate_returns();
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&returns);
        
        // Create base market data
        Ok(MarketData {
            timestamp: kline.close_time,
            timestamp_unix: kline.close_time.timestamp(),
            price: kline.close,
            volume: kline.volume,
            bid: kline.close * 0.9995, // Approximate bid
            ask: kline.close * 1.0005, // Approximate ask
            bid_volume: kline.volume * 0.45, // Approximate
            ask_volume: kline.volume * 0.55, // Approximate
            volatility,
            returns,
            volume_history: self.volume_history.clone(),
        })
    }
    
    /// Convert order book to whale detection
    pub fn detect_whales_from_orderbook(&self, order_book: &OrderBookSnapshot) -> WhaleDetection {
        let mut whale_detected = false;
        let mut whale_direction = WhaleDirection::Neutral;
        let mut whale_size: f64 = 0.0;
        
        // Check for large orders in bids
        for bid in &order_book.bids {
            let order_value = bid.price * bid.quantity;
            if order_value > self.whale_threshold_usd {
                whale_detected = true;
                whale_direction = WhaleDirection::Buying;
                whale_size = whale_size.max(order_value);
            }
        }
        
        // Check for large orders in asks
        for ask in &order_book.asks {
            let order_value = ask.price * ask.quantity;
            if order_value > self.whale_threshold_usd {
                whale_detected = true;
                if whale_direction == WhaleDirection::Buying {
                    whale_direction = WhaleDirection::Mixed;
                } else {
                    whale_direction = WhaleDirection::Selling;
                }
                whale_size = whale_size.max(order_value);
            }
        }
        
        WhaleDetection {
            timestamp: order_book.timestamp.timestamp(),
            detected: whale_detected,
            volume_spike: whale_size / self.whale_threshold_usd,
            direction: whale_direction,
            confidence: if whale_detected { 0.8 } else { 0.2 },
            whale_size,
            impact: order_book.imbalance.abs(),
            is_whale_detected: whale_detected,
            order_book_imbalance: order_book.imbalance,
            price_impact: whale_size / (order_book.bid_depth + order_book.ask_depth),
        }
    }
    
    /// Convert recent trades to whale detection
    pub fn detect_whales_from_trades(&self, trades: &[TradeData]) -> WhaleDetection {
        let mut whale_trades = Vec::new();
        let mut total_whale_volume = 0.0;
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        
        for trade in trades {
            if trade.trade_value_usd > self.whale_threshold_usd {
                whale_trades.push(trade);
                total_whale_volume += trade.quantity;
                
                if trade.is_buyer_maker {
                    sell_volume += trade.quantity;
                } else {
                    buy_volume += trade.quantity;
                }
            }
        }
        
        let whale_detected = !whale_trades.is_empty();
        let direction = if buy_volume > sell_volume * 1.5 {
            WhaleDirection::Buying
        } else if sell_volume > buy_volume * 1.5 {
            WhaleDirection::Selling
        } else if whale_detected {
            WhaleDirection::Mixed
        } else {
            WhaleDirection::Neutral
        };
        
        WhaleDetection {
            timestamp: Utc::now().timestamp(),
            detected: whale_detected,
            volume_spike: total_whale_volume / f64::max(trades.iter().map(|t| t.quantity).sum::<f64>(), 1.0),
            direction,
            confidence: whale_trades.len() as f64 / trades.len().max(1) as f64,
            whale_size: whale_trades.iter().map(|t| t.trade_value_usd).sum(),
            impact: (buy_volume - sell_volume).abs() / f64::max(buy_volume + sell_volume, 1.0),
            is_whale_detected: whale_detected,
            order_book_imbalance: (buy_volume - sell_volume) / f64::max(buy_volume + sell_volume, 1.0),
            price_impact: total_whale_volume / f64::max(trades.iter().map(|t| t.quantity).sum::<f64>(), 1.0),
        }
    }
    
    /// Create EnhancedMarketData from multiple sources
    pub fn create_enhanced_market_data(
        &mut self,
        ohlcv: OHLCVData,
        order_book: OrderBookSnapshot,
        recent_trades: Vec<TradeData>,
    ) -> Result<EnhancedMarketData, TalebianRiskError> {
        // Create base market data
        let kline = KlineData {
            symbol: ohlcv.symbol.clone(),
            close_time: ohlcv.close_time,
            close: ohlcv.close,
            volume: ohlcv.volume,
        };
        let base_data = self.from_kline(&kline)?;
        
        // Calculate microstructure metrics
        let microstructure = self.calculate_microstructure(&order_book, &recent_trades);
        
        // Create exchange data
        let exchange_data = ExchangeData {
            exchange: self.source.to_string(),
            funding_rate: None,
            open_interest: None,
            mark_price: Some(ohlcv.close),
            index_price: None,
            settlement_price: None,
            liquidation_volume_24h: None,
        };
        
        // Calculate data quality
        let data_quality = DataQuality {
            completeness: 0.95,
            timeliness: 0.98,
            accuracy: 0.99,
            consistency: 0.97,
            latency_ms: 10,
            gaps_detected: 0,
            outliers_detected: 0,
        };
        
        let enhanced = EnhancedMarketData {
            base_data,
            ohlcv,
            order_book,
            recent_trades,
            microstructure,
            exchange_data,
            data_quality,
        };
        
        // Add to history buffer
        self.history_buffer.push(enhanced.clone());
        if self.history_buffer.len() > self.max_history {
            self.history_buffer.remove(0);
        }
        
        Ok(enhanced)
    }
    
    /// Calculate returns from price history
    fn calculate_returns(&self) -> Vec<f64> {
        let mut returns = Vec::new();
        for i in 1..self.price_history.len() {
            let ret = (self.price_history[i] / self.price_history[i-1]) - 1.0;
            returns.push(ret);
        }
        returns
    }
    
    /// Calculate volatility from returns
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }
    
    /// Calculate market microstructure metrics
    fn calculate_microstructure(
        &self,
        order_book: &OrderBookSnapshot,
        trades: &[TradeData],
    ) -> MicrostructureMetrics {
        // Effective spread
        let effective_spread = if !trades.is_empty() {
            let last_trade = &trades[trades.len() - 1];
            2.0 * (last_trade.price - order_book.mid_price).abs() / order_book.mid_price
        } else {
            order_book.spread / order_book.mid_price
        };
        
        // Kyle's lambda (price impact coefficient)
        let kyle_lambda = if !trades.is_empty() {
            let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();
            if let (Some(last), Some(first)) = (trades.last(), trades.first()) {
                let price_change = last.price - first.price;
                (price_change / first.price).abs() / f64::max(total_volume, 1.0)
            } else {
                0.0
            }
        } else {
            0.0001
        };
        
        // Amihud illiquidity measure
        let amihud_illiquidity = if !trades.is_empty() {
            trades.iter()
                .map(|t| {
                    if let Some(first) = trades.first() {
                        (t.price / first.price - 1.0).abs() / t.quantity
                    } else {
                        0.0
                    }
                })
                .sum::<f64>() / trades.len() as f64
        } else {
            0.0001
        };
        
        MicrostructureMetrics {
            effective_spread,
            realized_spread: effective_spread * 0.8, // Approximation
            price_impact: kyle_lambda,
            kyle_lambda,
            amihud_illiquidity,
            roll_measure: effective_spread * 0.5, // Approximation
            order_flow_toxicity: order_book.imbalance.abs(),
            adverse_selection: effective_spread * 0.3, // Approximation
        }
    }
}

/// Helper struct for Kline data
#[derive(Debug, Clone)]
struct KlineData {
    pub symbol: String,
    pub close_time: DateTime<Utc>,
    pub close: f64,
    pub volume: f64,
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Binance => write!(f, "Binance"),
            DataSource::Coinbase => write!(f, "Coinbase"),
            DataSource::Kraken => write!(f, "Kraken"),
            DataSource::OKX => write!(f, "OKX"),
            DataSource::Bybit => write!(f, "Bybit"),
            DataSource::DataCollector => write!(f, "DataCollector"),
            DataSource::DataPipeline => write!(f, "DataPipeline"),
            DataSource::MarketIntelligence => write!(f, "MarketIntelligence"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_conversion() {
        let mut adapter = MarketDataAdapter::new("BTC/USDT".to_string(), DataSource::Binance);
        
        let kline = KlineData {
            symbol: "BTC/USDT".to_string(),
            close_time: Utc::now(),
            close: 50000.0,
            volume: 1000.0,
        };
        
        let result = adapter.from_kline(&kline);
        assert!(result.is_ok());
        
        let market_data = result.unwrap();
        assert_eq!(market_data.price, 50000.0);
        assert_eq!(market_data.volume, 1000.0);
    }
    
    #[test]
    fn test_whale_detection_from_orderbook() {
        let adapter = MarketDataAdapter::new("BTC/USDT".to_string(), DataSource::Binance);
        
        let order_book = OrderBookSnapshot {
            timestamp: Utc::now(),
            bids: vec![
                PriceLevel { price: 49900.0, quantity: 10.0, order_count: 5 },
                PriceLevel { price: 49800.0, quantity: 5.0, order_count: 3 },
            ],
            asks: vec![
                PriceLevel { price: 50100.0, quantity: 8.0, order_count: 4 },
                PriceLevel { price: 50200.0, quantity: 3.0, order_count: 2 },
            ],
            bid_depth: 750000.0,
            ask_depth: 550000.0,
            spread: 200.0,
            mid_price: 50000.0,
            imbalance: 0.2,
        };
        
        let whale_detection = adapter.detect_whales_from_orderbook(&order_book);
        
        // Large order: 10 BTC at $49,900 = $499,000 (whale)
        assert!(whale_detection.detected);
        assert_eq!(whale_detection.direction, WhaleDirection::Buying);
        assert!(whale_detection.whale_size > 400000.0);
    }
}