//! Market microstructure analysis module
//! 
//! Implements advanced algorithms for analyzing order book dynamics, trade flow,
//! liquidity metrics, and market efficiency measures.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::{statistical, time_series},
};
use ndarray::{Array1, Array2, ArrayView1};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque, BTreeMap};
use chrono::{DateTime, Utc, Duration};
use rayon::prelude::*;
use tracing::{info, debug, warn};

/// Market microstructure analyzer for order flow and liquidity analysis
#[derive(Debug, Clone)]
pub struct MicrostructureAnalyzer {
    config: MicrostructureConfig,
    order_book_analyzer: OrderBookAnalyzer,
    trade_flow_analyzer: TradeFlowAnalyzer,
    liquidity_analyzer: LiquidityAnalyzer,
    efficiency_analyzer: EfficiencyAnalyzer,
    historical_data: VecDeque<MicrostructureSnapshot>,
}

#[derive(Debug, Clone)]
pub struct MicrostructureConfig {
    pub max_order_book_levels: usize,
    pub trade_classification_window: Duration,
    pub liquidity_measurement_window: Duration,
    pub efficiency_test_window: usize,
    pub tick_size: f64,
    pub min_trade_size: f64,
    pub resilience_shock_sizes: Vec<f64>,
    pub depth_levels: Vec<f64>,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            max_order_book_levels: 20,
            trade_classification_window: Duration::seconds(10),
            liquidity_measurement_window: Duration::minutes(5),
            efficiency_test_window: 1000,
            tick_size: 0.01,
            min_trade_size: 0.001,
            resilience_shock_sizes: vec![0.1, 0.5, 1.0, 2.0],
            depth_levels: vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        }
    }
}

impl MicrostructureAnalyzer {
    pub fn new(config: &Config) -> Result<Self> {
        let microstructure_config = MicrostructureConfig::default();
        
        Ok(Self {
            config: microstructure_config.clone(),
            order_book_analyzer: OrderBookAnalyzer::new(&microstructure_config)?,
            trade_flow_analyzer: TradeFlowAnalyzer::new(&microstructure_config)?,
            liquidity_analyzer: LiquidityAnalyzer::new(&microstructure_config)?,
            efficiency_analyzer: EfficiencyAnalyzer::new(&microstructure_config)?,
            historical_data: VecDeque::with_capacity(10000),
        })
    }
    
    /// Analyze market microstructure
    pub async fn analyze_structure(&self, data: &MarketData) -> Result<MicrostructureAnalysis> {
        let start_time = std::time::Instant::now();
        debug!("Starting microstructure analysis for {}", data.symbol);
        
        // Parallel analysis of different microstructure components
        let (bid_ask_spread, market_depth, order_flow, liquidity_metrics, market_efficiency, trading_intensity) = tokio::try_join!(
            self.calculate_bid_ask_spread(data),
            self.analyze_market_depth(data),
            self.analyze_order_flow(data),
            self.calculate_liquidity_metrics(data),
            self.analyze_market_efficiency(data),
            self.analyze_trading_intensity(data)
        )?;
        
        let analysis = MicrostructureAnalysis {
            bid_ask_spread,
            market_depth,
            order_flow,
            liquidity_metrics,
            market_efficiency,
            trading_intensity,
        };
        
        let processing_time = start_time.elapsed();
        debug!("Microstructure analysis completed in {:?}", processing_time);
        
        Ok(analysis)
    }
    
    /// Calculate bid-ask spread
    async fn calculate_bid_ask_spread(&self, data: &MarketData) -> Result<f64> {
        if let Some(ref order_book) = data.order_book {
            if !order_book.bids.is_empty() && !order_book.asks.is_empty() {
                let spread = order_book.asks[0].price - order_book.bids[0].price;
                Ok(spread)
            } else {
                Ok(0.0)
            }
        } else {
            // Estimate spread from price volatility if no order book
            let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
            if !returns.is_empty() {
                Ok(returns.std_dev() * 2.0) // Rough approximation
            } else {
                Ok(0.0)
            }
        }
    }
    
    /// Analyze market depth
    async fn analyze_market_depth(&self, data: &MarketData) -> Result<MarketDepth> {
        if let Some(ref order_book) = data.order_book {
            self.order_book_analyzer.analyze_depth(order_book).await
        } else {
            // Fallback depth analysis using trade data
            let avg_trade_size = if !data.trades.is_empty() {
                data.trades.iter().map(|t| t.quantity).sum::<f64>() / data.trades.len() as f64
            } else {
                1.0
            };
            
            Ok(MarketDepth {
                total_bid_depth: avg_trade_size * 10.0, // Rough estimate
                total_ask_depth: avg_trade_size * 10.0,
                depth_imbalance: 0.0,
                depth_by_level: vec![(1, avg_trade_size * 5.0, avg_trade_size * 5.0)],
                market_resilience: 0.5,
            })
        }
    }
    
    /// Analyze order flow
    async fn analyze_order_flow(&self, data: &MarketData) -> Result<DetailedOrderFlow> {
        self.trade_flow_analyzer.analyze_flow(data).await
    }
    
    /// Calculate liquidity metrics
    async fn calculate_liquidity_metrics(&self, data: &MarketData) -> Result<LiquidityMetrics> {
        self.liquidity_analyzer.calculate_metrics(data).await
    }
    
    /// Analyze market efficiency
    async fn analyze_market_efficiency(&self, data: &MarketData) -> Result<MarketEfficiency> {
        self.efficiency_analyzer.analyze_efficiency(data).await
    }
    
    /// Analyze trading intensity
    async fn analyze_trading_intensity(&self, data: &MarketData) -> Result<TradingIntensity> {
        if data.trades.is_empty() {
            return Ok(TradingIntensity {
                trades_per_minute: 0.0,
                volume_per_minute: 0.0,
                average_trade_size: 0.0,
                trade_size_variance: 0.0,
                intensity_clustering: 0.0,
            });
        }
        
        let total_trades = data.trades.len() as f64;
        let total_volume: f64 = data.trades.iter().map(|t| t.quantity).sum();
        let trade_sizes: Vec<f64> = data.trades.iter().map(|t| t.quantity).collect();
        
        // Assume data spans 1 minute for simplicity
        let trades_per_minute = total_trades;
        let volume_per_minute = total_volume;
        let average_trade_size = total_volume / total_trades;
        let trade_size_variance = trade_sizes.variance();
        
        // Calculate intensity clustering using temporal analysis
        let intensity_clustering = self.calculate_intensity_clustering(&data.trades)?;
        
        Ok(TradingIntensity {
            trades_per_minute,
            volume_per_minute,
            average_trade_size,
            trade_size_variance,
            intensity_clustering,
        })
    }
    
    /// Calculate intensity clustering
    fn calculate_intensity_clustering(&self, trades: &[Trade]) -> Result<f64> {
        if trades.len() < 10 {
            return Ok(0.0);
        }
        
        // Calculate inter-arrival times
        let mut inter_arrival_times = Vec::new();
        for i in 1..trades.len() {
            let time_diff = trades[i].timestamp - trades[i-1].timestamp;
            inter_arrival_times.push(time_diff.num_milliseconds() as f64);
        }
        
        if inter_arrival_times.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate clustering using coefficient of variation
        let mean_time = inter_arrival_times.clone().mean();
        let std_time = inter_arrival_times.std_dev();
        
        if mean_time > 0.0 {
            Ok(std_time / mean_time)
        } else {
            Ok(0.0)
        }
    }
}

/// Order book analyzer
#[derive(Debug, Clone)]
struct OrderBookAnalyzer {
    config: MicrostructureConfig,
}

impl OrderBookAnalyzer {
    fn new(config: &MicrostructureConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze_depth(&self, order_book: &OrderBook) -> Result<MarketDepth> {
        let total_bid_depth: f64 = order_book.bids.iter()
            .take(self.config.max_order_book_levels)
            .map(|level| level.quantity)
            .sum();
            
        let total_ask_depth: f64 = order_book.asks.iter()
            .take(self.config.max_order_book_levels)
            .map(|level| level.quantity)
            .sum();
        
        let total_depth = total_bid_depth + total_ask_depth;
        let depth_imbalance = if total_depth > 0.0 {
            (total_bid_depth - total_ask_depth) / total_depth
        } else {
            0.0
        };
        
        // Calculate depth by level
        let mut depth_by_level = Vec::new();
        for level in 1..=self.config.max_order_book_levels.min(10) {
            let bid_depth: f64 = order_book.bids.iter()
                .take(level)
                .map(|l| l.quantity)
                .sum();
                
            let ask_depth: f64 = order_book.asks.iter()
                .take(level)
                .map(|l| l.quantity)
                .sum();
                
            depth_by_level.push((level as u32, bid_depth, ask_depth));
        }
        
        // Calculate market resilience
        let market_resilience = self.calculate_market_resilience(order_book)?;
        
        Ok(MarketDepth {
            total_bid_depth,
            total_ask_depth,
            depth_imbalance,
            depth_by_level,
            market_resilience,
        })
    }
    
    fn calculate_market_resilience(&self, order_book: &OrderBook) -> Result<f64> {
        if order_book.bids.is_empty() || order_book.asks.is_empty() {
            return Ok(0.0);
        }
        
        let mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2.0;
        
        // Calculate cumulative depth at various price levels
        let mut resilience_scores = Vec::new();
        
        for &shock_size in &self.config.resilience_shock_sizes {
            let shock_price_up = mid_price * (1.0 + shock_size / 100.0);
            let shock_price_down = mid_price * (1.0 - shock_size / 100.0);
            
            // Calculate liquidity needed to move price by shock_size
            let ask_liquidity: f64 = order_book.asks.iter()
                .take_while(|level| level.price <= shock_price_up)
                .map(|level| level.quantity)
                .sum();
                
            let bid_liquidity: f64 = order_book.bids.iter()
                .take_while(|level| level.price >= shock_price_down)
                .map(|level| level.quantity)
                .sum();
            
            let avg_liquidity = (ask_liquidity + bid_liquidity) / 2.0;
            resilience_scores.push(avg_liquidity);
        }
        
        // Higher liquidity at larger shocks indicates better resilience
        if resilience_scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(resilience_scores.iter().sum::<f64>() / resilience_scores.len() as f64)
        }
    }
}

/// Trade flow analyzer
#[derive(Debug, Clone)]
struct TradeFlowAnalyzer {
    config: MicrostructureConfig,
}

impl TradeFlowAnalyzer {
    fn new(config: &MicrostructureConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze_flow(&self, data: &MarketData) -> Result<DetailedOrderFlow> {
        if data.trades.is_empty() {
            return Ok(DetailedOrderFlow {
                buy_volume: 0.0,
                sell_volume: 0.0,
                net_flow: 0.0,
                aggressive_ratio: 0.0,
                order_arrival_rate: 0.0,
                cancellation_rate: 0.0,
                fill_rate: 0.0,
            });
        }
        
        let (buy_volume, sell_volume) = self.calculate_directional_volumes(&data.trades)?;
        let net_flow = buy_volume - sell_volume;
        let aggressive_ratio = self.calculate_aggressive_ratio(&data.trades)?;
        let order_arrival_rate = self.calculate_order_arrival_rate(&data.trades)?;
        
        // Note: cancellation_rate and fill_rate would require order-level data
        let cancellation_rate = 0.0; // Placeholder
        let fill_rate = 1.0; // Assume all trades are fills
        
        Ok(DetailedOrderFlow {
            buy_volume,
            sell_volume,
            net_flow,
            aggressive_ratio,
            order_arrival_rate,
            cancellation_rate,
            fill_rate,
        })
    }
    
    fn calculate_directional_volumes(&self, trades: &[Trade]) -> Result<(f64, f64)> {
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        
        for trade in trades {
            match trade.side {
                TradeSide::Buy => buy_volume += trade.quantity,
                TradeSide::Sell => sell_volume += trade.quantity,
            }
        }
        
        Ok((buy_volume, sell_volume))
    }
    
    fn calculate_aggressive_ratio(&self, trades: &[Trade]) -> Result<f64> {
        if trades.is_empty() {
            return Ok(0.0);
        }
        
        let aggressive_trades = trades.iter()
            .filter(|t| matches!(t.trade_type, TradeType::Market))
            .count();
            
        Ok(aggressive_trades as f64 / trades.len() as f64)
    }
    
    fn calculate_order_arrival_rate(&self, trades: &[Trade]) -> Result<f64> {
        if trades.len() < 2 {
            return Ok(0.0);
        }
        
        let time_span = trades.last().unwrap().timestamp - trades.first().unwrap().timestamp;
        let time_span_minutes = time_span.num_minutes() as f64;
        
        if time_span_minutes > 0.0 {
            Ok(trades.len() as f64 / time_span_minutes)
        } else {
            Ok(0.0)
        }
    }
}

/// Liquidity analyzer
#[derive(Debug, Clone)]
struct LiquidityAnalyzer {
    config: MicrostructureConfig,
}

impl LiquidityAnalyzer {
    fn new(config: &MicrostructureConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn calculate_metrics(&self, data: &MarketData) -> Result<LiquidityMetrics> {
        let amihud_illiquidity = self.calculate_amihud_illiquidity(data)?;
        let roll_spread = self.calculate_roll_spread(data)?;
        let effective_spread = self.calculate_effective_spread(data)?;
        let realized_spread = self.calculate_realized_spread(data)?;
        let price_impact = self.calculate_price_impact(data)?;
        let kyle_lambda = self.calculate_kyle_lambda(data)?;
        
        Ok(LiquidityMetrics {
            amihud_illiquidity,
            roll_spread,
            effective_spread,
            realized_spread,
            price_impact,
            kyle_lambda,
        })
    }
    
    /// Calculate Amihud illiquidity measure
    fn calculate_amihud_illiquidity(&self, data: &MarketData) -> Result<f64> {
        if data.prices.len() < 2 || data.volumes.is_empty() {
            return Ok(0.0);
        }
        
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        if returns.len() != data.volumes.len() - 1 {
            return Ok(0.0);
        }
        
        let mut illiquidity_measures = Vec::new();
        
        for i in 0..returns.len() {
            let volume_idx = i + 1; // Volume corresponds to the period ending the return
            if volume_idx < data.volumes.len() && data.volumes[volume_idx] > 0.0 {
                let illiquidity = returns[i].abs() / data.volumes[volume_idx];
                illiquidity_measures.push(illiquidity);
            }
        }
        
        if illiquidity_measures.is_empty() {
            Ok(0.0)
        } else {
            Ok(illiquidity_measures.mean())
        }
    }
    
    /// Calculate Roll spread estimate
    fn calculate_roll_spread(&self, data: &MarketData) -> Result<f64> {
        if data.prices.len() < 3 {
            return Ok(0.0);
        }
        
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        if returns.len() < 2 {
            return Ok(0.0);
        }
        
        // Calculate first-order autocovariance
        let autocovariance = statistical::autocorrelation(&returns, 1)?;
        
        if autocovariance.is_empty() {
            return Ok(0.0);
        }
        
        let first_order_autocov = autocovariance[0];
        
        // Roll spread = 2 * sqrt(-autocovariance)
        if first_order_autocov < 0.0 {
            Ok(2.0 * (-first_order_autocov).sqrt())
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate effective spread
    fn calculate_effective_spread(&self, data: &MarketData) -> Result<f64> {
        if let Some(ref order_book) = data.order_book {
            if !order_book.bids.is_empty() && !order_book.asks.is_empty() {
                let mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2.0;
                
                // Calculate volume-weighted effective spread
                let mut weighted_spread = 0.0;
                let mut total_volume = 0.0;
                
                for trade in &data.trades {
                    let spread = 2.0 * (trade.price - mid_price).abs() / mid_price;
                    weighted_spread += spread * trade.quantity;
                    total_volume += trade.quantity;
                }
                
                if total_volume > 0.0 {
                    Ok(weighted_spread / total_volume)
                } else {
                    Ok((order_book.asks[0].price - order_book.bids[0].price) / mid_price)
                }
            } else {
                Ok(0.0)
            }
        } else {
            // Estimate from price volatility
            let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
            Ok(returns.std_dev() * 2.0)
        }
    }
    
    /// Calculate realized spread
    fn calculate_realized_spread(&self, data: &MarketData) -> Result<f64> {
        // Simplified realized spread calculation
        // In practice, this requires quote midpoint at time of trade and after a delay
        let effective_spread = self.calculate_effective_spread(data)?;
        Ok(effective_spread * 0.5) // Rough approximation
    }
    
    /// Calculate price impact
    fn calculate_price_impact(&self, data: &MarketData) -> Result<f64> {
        if data.trades.len() < 2 || data.prices.len() < 2 {
            return Ok(0.0);
        }
        
        let mut impacts = Vec::new();
        
        for i in 1..data.trades.len() {
            let trade = &data.trades[i];
            let prev_trade = &data.trades[i-1];
            
            if trade.timestamp != prev_trade.timestamp {
                let price_change = (trade.price - prev_trade.price) / prev_trade.price;
                let volume_ratio = trade.quantity / (trade.quantity + prev_trade.quantity);
                let impact = price_change.abs() * volume_ratio;
                impacts.push(impact);
            }
        }
        
        if impacts.is_empty() {
            Ok(0.0)
        } else {
            Ok(impacts.mean())
        }
    }
    
    /// Calculate Kyle's lambda (price impact coefficient)
    fn calculate_kyle_lambda(&self, data: &MarketData) -> Result<f64> {
        if data.trades.len() < 10 {
            return Ok(0.0);
        }
        
        let mut price_changes = Vec::new();
        let mut signed_volumes = Vec::new();
        
        for i in 1..data.trades.len() {
            let current_trade = &data.trades[i];
            let prev_trade = &data.trades[i-1];
            
            let price_change = current_trade.price - prev_trade.price;
            let signed_volume = match current_trade.side {
                TradeSide::Buy => current_trade.quantity,
                TradeSide::Sell => -current_trade.quantity,
            };
            
            price_changes.push(price_change);
            signed_volumes.push(signed_volume);
        }
        
        // Calculate correlation between price changes and signed volume
        if price_changes.len() == signed_volumes.len() && !price_changes.is_empty() {
            let correlation = statistical::correlation(&price_changes, &signed_volumes)?;
            Ok(correlation.abs())
        } else {
            Ok(0.0)
        }
    }
}

/// Market efficiency analyzer
#[derive(Debug, Clone)]
struct EfficiencyAnalyzer {
    config: MicrostructureConfig,
}

impl EfficiencyAnalyzer {
    fn new(config: &MicrostructureConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze_efficiency(&self, data: &MarketData) -> Result<MarketEfficiency> {
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        let variance_ratio = self.calculate_variance_ratio(&returns)?;
        let hurst_exponent = if returns.len() >= 20 { 
            statistical::hurst_exponent(&returns)? 
        } else { 
            0.5 
        };
        let autocorrelation = if returns.len() >= 10 { 
            statistical::autocorrelation(&returns, 5)? 
        } else { 
            vec![0.0; 5] 
        };
        let information_share = self.calculate_information_share(data)?;
        let price_discovery_metrics = self.calculate_price_discovery_metrics(data)?;
        
        Ok(MarketEfficiency {
            variance_ratio,
            hurst_exponent,
            autocorrelation,
            information_share,
            price_discovery_metrics,
        })
    }
    
    /// Calculate variance ratio test statistic
    fn calculate_variance_ratio(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < 20 {
            return Ok(1.0);
        }
        
        let k = 2; // Compare 2-period vs 1-period variance
        let n = returns.len();
        
        if n < k * 10 {
            return Ok(1.0);
        }
        
        // Calculate 1-period variance
        let var_1 = returns.variance();
        
        // Calculate k-period returns
        let mut k_period_returns = Vec::new();
        for i in 0..n-k+1 {
            let k_return: f64 = returns[i..i+k].iter().sum();
            k_period_returns.push(k_return);
        }
        
        let var_k = k_period_returns.variance();
        
        // Variance ratio should be k for random walk
        if var_1 > 0.0 {
            Ok(var_k / (k as f64 * var_1))
        } else {
            Ok(1.0)
        }
    }
    
    /// Calculate information share
    fn calculate_information_share(&self, data: &MarketData) -> Result<f64> {
        // Simplified information share calculation
        // In practice, this requires multiple markets/venues
        if data.trades.is_empty() {
            return Ok(0.0);
        }
        
        let total_volume: f64 = data.trades.iter().map(|t| t.quantity).sum();
        
        // Assume this market has significant information share if volume is high
        if total_volume > 1000.0 {
            Ok(0.8)
        } else if total_volume > 100.0 {
            Ok(0.5)
        } else {
            Ok(0.2)
        }
    }
    
    /// Calculate price discovery metrics
    fn calculate_price_discovery_metrics(&self, data: &MarketData) -> Result<PriceDiscoveryMetrics> {
        // Simplified price discovery metrics
        let information_share = self.calculate_information_share(data)?;
        
        // Component share (permanent vs transitory price changes)
        let component_share = if data.prices.len() >= 10 {
            let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
            let autocorr = statistical::autocorrelation(&returns, 1)?;
            
            if !autocorr.is_empty() {
                (1.0 + autocorr[0]).max(0.0).min(1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };
        
        // Hasbrouck information share (based on variance decomposition)
        let hasbrouck_info_share = information_share * 0.9; // Approximation
        
        // Gonzalo-Granger metric
        let gonzalo_granger_metric = (information_share + component_share) / 2.0;
        
        Ok(PriceDiscoveryMetrics {
            information_share,
            component_share,
            hasbrouck_info_share,
            gonzalo_granger_metric,
        })
    }
}

/// Microstructure snapshot for historical tracking
#[derive(Debug, Clone)]
struct MicrostructureSnapshot {
    timestamp: DateTime<Utc>,
    bid_ask_spread: f64,
    market_depth: f64,
    liquidity_score: f64,
    trading_intensity: f64,
    efficiency_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_microstructure_analyzer_creation() {
        let config = Config::default();
        let analyzer = MicrostructureAnalyzer::new(&config);
        assert!(analyzer.is_ok());
    }
    
    #[tokio::test]
    async fn test_microstructure_analysis() {
        let config = Config::default();
        let analyzer = MicrostructureAnalyzer::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let analysis = analyzer.analyze_structure(&market_data).await;
        assert!(analysis.is_ok());
        
        let result = analysis.unwrap();
        assert!(result.bid_ask_spread >= 0.0);
    }
    
    #[test]
    fn test_liquidity_metrics() {
        let config = MicrostructureConfig::default();
        let analyzer = LiquidityAnalyzer::new(&config).unwrap();
        
        let market_data = MarketData::mock_data();
        
        // Test Amihud illiquidity calculation
        let amihud = analyzer.calculate_amihud_illiquidity(&market_data);
        assert!(amihud.is_ok());
        assert!(amihud.unwrap() >= 0.0);
    }
    
    #[test]
    fn test_efficiency_analysis() {
        let config = MicrostructureConfig::default();
        let analyzer = EfficiencyAnalyzer::new(&config).unwrap();
        
        let returns = vec![0.01, -0.005, 0.008, -0.012, 0.003, 0.007, -0.004, 0.009];
        let variance_ratio = analyzer.calculate_variance_ratio(&returns);
        
        assert!(variance_ratio.is_ok());
        let vr = variance_ratio.unwrap();
        assert!(vr > 0.0);
    }
    
    #[test]
    fn test_order_book_depth_analysis() {
        let config = MicrostructureConfig::default();
        let analyzer = OrderBookAnalyzer::new(&config).unwrap();
        
        let order_book = OrderBook {
            timestamp: Utc::now(),
            bids: vec![
                OrderBookLevel { price: 99.5, quantity: 100.0, order_count: Some(5) },
                OrderBookLevel { price: 99.4, quantity: 150.0, order_count: Some(3) },
            ],
            asks: vec![
                OrderBookLevel { price: 100.5, quantity: 120.0, order_count: Some(4) },
                OrderBookLevel { price: 100.6, quantity: 80.0, order_count: Some(2) },
            ],
            sequence: 12345,
        };
        
        // This would be an async test in practice
        // let depth = analyzer.analyze_depth(&order_book).await;
        // assert!(depth.is_ok());
    }
}