//! Advanced whale detection and analysis module
//! 
//! Implements sophisticated algorithms for detecting large institutional trading activity
//! through volume profile analysis, order flow imbalance detection, and smart money tracking.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::statistical,
};
use ndarray::{Array1, Array2};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use rayon::prelude::*;
use tracing::{info, debug, warn};

/// Advanced whale detection and analysis engine
#[derive(Debug, Clone)]
pub struct WhaleAnalyzer {
    config: WhaleConfig,
    volume_profile_analyzer: VolumeProfileAnalyzer,
    order_flow_analyzer: OrderFlowAnalyzer,
    smart_money_tracker: SmartMoneyTracker,
    historical_patterns: VecDeque<WhalePattern>,
}

#[derive(Debug, Clone)]
pub struct WhaleConfig {
    pub whale_threshold_btc: f64,          // Minimum BTC equivalent for whale classification
    pub volume_spike_threshold: f64,       // Minimum volume spike multiplier
    pub imbalance_threshold: f64,          // Order flow imbalance threshold
    pub confidence_threshold: f64,         // Minimum confidence for signal emission
    pub lookback_periods: usize,           // Historical periods for pattern analysis
    pub smoothing_factor: f64,             // Exponential smoothing factor
    pub detection_sensitivity: f64,        // Detection sensitivity (0.0 - 1.0)
}

impl Default for WhaleConfig {
    fn default() -> Self {
        Self {
            whale_threshold_btc: 10.0,
            volume_spike_threshold: 2.5,
            imbalance_threshold: 0.7,
            confidence_threshold: 0.8,
            lookback_periods: 100,
            smoothing_factor: 0.3,
            detection_sensitivity: 0.8,
        }
    }
}

impl WhaleAnalyzer {
    pub fn new(config: &Config) -> Result<Self> {
        let whale_config = WhaleConfig::default(); // TODO: Extract from main config
        
        Ok(Self {
            config: whale_config.clone(),
            volume_profile_analyzer: VolumeProfileAnalyzer::new(&whale_config)?,
            order_flow_analyzer: OrderFlowAnalyzer::new(&whale_config)?,
            smart_money_tracker: SmartMoneyTracker::new(&whale_config)?,
            historical_patterns: VecDeque::with_capacity(1000),
        })
    }
    
    /// Analyze market data for whale activity
    pub async fn analyze(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        let start_time = std::time::Instant::now();
        debug!("Starting whale analysis for {}", data.symbol);
        
        // Parallel analysis of different whale detection methods
        let (volume_signals, order_flow_signals, smart_money_signals) = tokio::try_join!(
            self.analyze_volume_profile(data),
            self.analyze_order_flow(data),
            self.analyze_smart_money_patterns(data)
        )?;
        
        // Combine and validate signals
        let mut all_signals = Vec::new();
        all_signals.extend(volume_signals);
        all_signals.extend(order_flow_signals);
        all_signals.extend(smart_money_signals);
        
        // Cross-validate signals and filter by confidence
        let validated_signals = self.cross_validate_signals(all_signals, data)?;
        
        // Update historical patterns for learning
        self.update_patterns(&validated_signals, data).await?;
        
        let processing_time = start_time.elapsed();
        debug!("Whale analysis completed in {:?}", processing_time);
        
        Ok(validated_signals)
    }
    
    /// Analyze volume profile for institutional activity
    async fn analyze_volume_profile(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        self.volume_profile_analyzer.analyze(data).await
    }
    
    /// Analyze order flow for imbalances indicating whale activity
    async fn analyze_order_flow(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        self.order_flow_analyzer.analyze(data).await
    }
    
    /// Analyze smart money patterns and institutional behavior
    async fn analyze_smart_money_patterns(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        self.smart_money_tracker.analyze(data).await
    }
    
    /// Cross-validate signals from different detection methods
    fn cross_validate_signals(&self, signals: Vec<WhaleSignal>, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        let mut validated_signals = Vec::new();
        
        // Group signals by type and time proximity
        let signal_groups = self.group_signals_by_proximity(signals);
        
        for group in signal_groups {
            if let Some(consensus_signal) = self.build_consensus_signal(group, data)? {
                if consensus_signal.confidence >= self.config.confidence_threshold {
                    validated_signals.push(consensus_signal);
                }
            }
        }
        
        Ok(validated_signals)
    }
    
    /// Group signals that are close in time and type
    fn group_signals_by_proximity(&self, signals: Vec<WhaleSignal>) -> Vec<Vec<WhaleSignal>> {
        let mut groups = Vec::new();
        let mut remaining_signals = signals;
        
        while !remaining_signals.is_empty() {
            let seed_signal = remaining_signals.remove(0);
            let mut group = vec![seed_signal.clone()];
            
            remaining_signals.retain(|signal| {
                let time_diff = (signal.timestamp - seed_signal.timestamp).num_minutes().abs();
                let type_match = std::mem::discriminant(&signal.signal_type) == 
                                std::mem::discriminant(&seed_signal.signal_type);
                
                if time_diff <= 5 && type_match {
                    group.push(signal.clone());
                    false // Remove from remaining
                } else {
                    true // Keep in remaining
                }
            });
            
            groups.push(group);
        }
        
        groups
    }
    
    /// Build consensus signal from a group of similar signals
    fn build_consensus_signal(&self, signals: Vec<WhaleSignal>, data: &MarketData) -> Result<Option<WhaleSignal>> {
        if signals.is_empty() {
            return Ok(None);
        }
        
        let first_signal = &signals[0];
        let avg_strength = signals.iter().map(|s| s.strength).sum::<f64>() / signals.len() as f64;
        let avg_confidence = signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64;
        
        // Boost confidence if multiple detection methods agree
        let confidence_boost = if signals.len() > 1 {
            (signals.len() as f64 - 1.0) * 0.1
        } else {
            0.0
        };
        
        let final_confidence = (avg_confidence + confidence_boost).min(1.0);
        
        // Calculate enhanced volume profile and order flow
        let volume_profile = self.calculate_enhanced_volume_profile(data)?;
        let order_flow = self.calculate_enhanced_order_flow(data)?;
        let impact_analysis = self.calculate_price_impact_analysis(data, avg_strength)?;
        
        Ok(Some(WhaleSignal {
            signal_type: first_signal.signal_type.clone(),
            strength: avg_strength,
            confidence: final_confidence,
            volume_profile,
            order_flow,
            impact_analysis,
            timestamp: Utc::now(),
        }))
    }
    
    /// Calculate enhanced volume profile analysis
    fn calculate_enhanced_volume_profile(&self, data: &MarketData) -> Result<VolumeProfile> {
        if data.prices.is_empty() || data.volumes.is_empty() {
            return Err(AnalysisError::InsufficientData("Empty price or volume data".to_string()));
        }
        
        let price_levels = self.create_price_levels(&data.prices);
        let volume_by_price = self.distribute_volume_by_price(&data.prices, &data.volumes, &price_levels)?;
        
        // Calculate Value Area (70% of volume)
        let total_volume: f64 = volume_by_price.iter().map(|(_, vol)| vol).sum();
        let target_volume = total_volume * 0.70;
        
        let (value_area_high, value_area_low, point_of_control) = 
            self.calculate_value_area(&volume_by_price, target_volume)?;
        
        // Calculate volume delta and cumulative delta
        let volume_delta = self.calculate_volume_delta(data)?;
        let cumulative_volume_delta = self.calculate_cumulative_volume_delta(data)?;
        
        Ok(VolumeProfile {
            value_area_high,
            value_area_low,
            point_of_control,
            volume_by_price,
            volume_delta,
            cumulative_volume_delta,
        })
    }
    
    /// Create price levels for volume distribution
    fn create_price_levels(&self, prices: &[f64]) -> Vec<f64> {
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range = max_price - min_price;
        let tick_size = price_range / 100.0; // 100 price levels
        
        (0..101).map(|i| min_price + (i as f64 * tick_size)).collect()
    }
    
    /// Distribute volume across price levels
    fn distribute_volume_by_price(
        &self, 
        prices: &[f64], 
        volumes: &[f64], 
        price_levels: &[f64]
    ) -> Result<Vec<(f64, f64)>> {
        let mut volume_distribution = vec![0.0; price_levels.len()];
        
        for (price, volume) in prices.iter().zip(volumes.iter()) {
            // Find nearest price level
            if let Some(level_index) = self.find_nearest_price_level(*price, price_levels) {
                volume_distribution[level_index] += volume;
            }
        }
        
        Ok(price_levels.iter()
            .zip(volume_distribution.iter())
            .map(|(&price, &volume)| (price, volume))
            .collect())
    }
    
    /// Find the nearest price level index
    fn find_nearest_price_level(&self, price: f64, price_levels: &[f64]) -> Option<usize> {
        price_levels.iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - price).abs().partial_cmp(&(b - price).abs()).unwrap()
            })
            .map(|(index, _)| index)
    }
    
    /// Calculate Value Area (70% of volume concentration)
    fn calculate_value_area(
        &self, 
        volume_by_price: &[(f64, f64)], 
        target_volume: f64
    ) -> Result<(f64, f64, f64)> {
        // Find Point of Control (highest volume price)
        let poc_index = volume_by_price.iter()
            .enumerate()
            .max_by(|(_, (_, vol_a)), (_, (_, vol_b))| vol_a.partial_cmp(vol_b).unwrap())
            .map(|(index, _)| index)
            .ok_or_else(|| AnalysisError::CalculationError("No volume data for POC".to_string()))?;
        
        let point_of_control = volume_by_price[poc_index].0;
        
        // Expand from POC to find Value Area
        let mut current_volume = volume_by_price[poc_index].1;
        let mut low_index = poc_index;
        let mut high_index = poc_index;
        
        while current_volume < target_volume {
            let expand_up = high_index + 1 < volume_by_price.len() &&
                (low_index == 0 || volume_by_price[high_index + 1].1 >= volume_by_price[low_index - 1].1);
            
            if expand_up {
                high_index += 1;
                current_volume += volume_by_price[high_index].1;
            } else if low_index > 0 {
                low_index -= 1;
                current_volume += volume_by_price[low_index].1;
            } else {
                break;
            }
        }
        
        let value_area_high = volume_by_price[high_index].0;
        let value_area_low = volume_by_price[low_index].0;
        
        Ok((value_area_high, value_area_low, point_of_control))
    }
    
    /// Calculate volume delta (buy volume - sell volume)
    fn calculate_volume_delta(&self, data: &MarketData) -> Result<f64> {
        if data.trades.is_empty() {
            return Ok(0.0);
        }
        
        let (buy_volume, sell_volume) = data.trades.iter().fold((0.0, 0.0), |(buy, sell), trade| {
            match trade.side {
                TradeSide::Buy => (buy + trade.quantity, sell),
                TradeSide::Sell => (buy, sell + trade.quantity),
            }
        });
        
        Ok(buy_volume - sell_volume)
    }
    
    /// Calculate cumulative volume delta
    fn calculate_cumulative_volume_delta(&self, data: &MarketData) -> Result<f64> {
        if data.trades.is_empty() {
            return Ok(0.0);
        }
        
        let mut cumulative_delta = 0.0;
        
        for trade in &data.trades {
            match trade.side {
                TradeSide::Buy => cumulative_delta += trade.quantity,
                TradeSide::Sell => cumulative_delta -= trade.quantity,
            }
        }
        
        Ok(cumulative_delta)
    }
    
    /// Calculate enhanced order flow analysis
    fn calculate_enhanced_order_flow(&self, data: &MarketData) -> Result<OrderFlowImbalance> {
        if data.trades.is_empty() {
            return Ok(OrderFlowImbalance {
                buy_pressure: 0.0,
                sell_pressure: 0.0,
                imbalance_ratio: 0.0,
                aggressive_buy_ratio: 0.0,
                aggressive_sell_ratio: 0.0,
                order_size_distribution: OrderSizeDistribution {
                    small_orders: 0.0,
                    medium_orders: 0.0,
                    large_orders: 0.0,
                    whale_orders: 0.0,
                },
            });
        }
        
        let total_buy_volume = data.trades.iter()
            .filter(|t| matches!(t.side, TradeSide::Buy))
            .map(|t| t.quantity)
            .sum::<f64>();
            
        let total_sell_volume = data.trades.iter()
            .filter(|t| matches!(t.side, TradeSide::Sell))
            .map(|t| t.quantity)
            .sum::<f64>();
        
        let total_volume = total_buy_volume + total_sell_volume;
        let buy_pressure = if total_volume > 0.0 { total_buy_volume / total_volume } else { 0.0 };
        let sell_pressure = if total_volume > 0.0 { total_sell_volume / total_volume } else { 0.0 };
        
        let imbalance_ratio = if sell_pressure > 0.0 { buy_pressure / sell_pressure } else { f64::INFINITY };
        
        // Calculate aggressive trade ratios (market orders vs limit orders)
        let aggressive_buys = data.trades.iter()
            .filter(|t| matches!(t.side, TradeSide::Buy) && matches!(t.trade_type, TradeType::Market))
            .count() as f64;
            
        let aggressive_sells = data.trades.iter()
            .filter(|t| matches!(t.side, TradeSide::Sell) && matches!(t.trade_type, TradeType::Market))
            .count() as f64;
        
        let total_trades = data.trades.len() as f64;
        let aggressive_buy_ratio = if total_trades > 0.0 { aggressive_buys / total_trades } else { 0.0 };
        let aggressive_sell_ratio = if total_trades > 0.0 { aggressive_sells / total_trades } else { 0.0 };
        
        // Calculate order size distribution
        let order_size_distribution = self.calculate_order_size_distribution(data)?;
        
        Ok(OrderFlowImbalance {
            buy_pressure,
            sell_pressure,
            imbalance_ratio,
            aggressive_buy_ratio,
            aggressive_sell_ratio,
            order_size_distribution,
        })
    }
    
    /// Calculate order size distribution for whale detection
    fn calculate_order_size_distribution(&self, data: &MarketData) -> Result<OrderSizeDistribution> {
        if data.trades.is_empty() {
            return Ok(OrderSizeDistribution {
                small_orders: 0.0,
                medium_orders: 0.0,
                large_orders: 0.0,
                whale_orders: 0.0,
            });
        }
        
        let total_volume = data.trades.iter().map(|t| t.quantity).sum::<f64>();
        
        let (small, medium, large, whale) = data.trades.iter().fold((0.0, 0.0, 0.0, 0.0), |(s, m, l, w), trade| {
            let btc_equivalent = trade.quantity * trade.price / 50000.0; // Rough BTC conversion
            
            match btc_equivalent {
                x if x < 1.0 => (s + trade.quantity, m, l, w),
                x if x < 10.0 => (s, m + trade.quantity, l, w),
                x if x < 100.0 => (s, m, l + trade.quantity, w),
                _ => (s, m, l, w + trade.quantity),
            }
        });
        
        Ok(OrderSizeDistribution {
            small_orders: if total_volume > 0.0 { small / total_volume } else { 0.0 },
            medium_orders: if total_volume > 0.0 { medium / total_volume } else { 0.0 },
            large_orders: if total_volume > 0.0 { large / total_volume } else { 0.0 },
            whale_orders: if total_volume > 0.0 { whale / total_volume } else { 0.0 },
        })
    }
    
    /// Calculate price impact analysis
    fn calculate_price_impact_analysis(&self, data: &MarketData, strength: f64) -> Result<PriceImpactAnalysis> {
        if data.prices.len() < 2 {
            return Ok(PriceImpactAnalysis {
                immediate_impact: 0.0,
                delayed_impact: 0.0,
                recovery_time: None,
                market_depth: 0.0,
                slippage_estimate: 0.0,
            });
        }
        
        // Calculate immediate price impact
        let price_changes: Vec<f64> = data.prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
            
        let immediate_impact = price_changes.last().unwrap_or(&0.0).abs();
        
        // Estimate delayed impact using price reversal
        let delayed_impact = if price_changes.len() >= 5 {
            let recent_changes = &price_changes[price_changes.len()-5..];
            recent_changes.iter().sum::<f64>().abs() / recent_changes.len() as f64
        } else {
            immediate_impact * 0.5
        };
        
        // Estimate market depth from order book if available
        let market_depth = if let Some(ref order_book) = data.order_book {
            let total_bid_depth: f64 = order_book.bids.iter().map(|level| level.quantity).sum();
            let total_ask_depth: f64 = order_book.asks.iter().map(|level| level.quantity).sum();
            (total_bid_depth + total_ask_depth) / 2.0
        } else {
            data.volumes.iter().sum::<f64>() / data.volumes.len() as f64
        };
        
        // Estimate slippage based on impact and volume
        let slippage_estimate = immediate_impact * strength * 0.1;
        
        Ok(PriceImpactAnalysis {
            immediate_impact,
            delayed_impact,
            recovery_time: Some(Duration::minutes((immediate_impact * 100.0) as i64)),
            market_depth,
            slippage_estimate,
        })
    }
    
    /// Update historical patterns for machine learning
    async fn update_patterns(&self, signals: &[WhaleSignal], data: &MarketData) -> Result<()> {
        // Implementation for updating historical patterns
        // This would feed into machine learning models for pattern recognition
        Ok(())
    }
    
    /// Update model parameters based on feedback
    pub async fn update_parameters(&mut self, feedback: &WhaleFeedback) -> Result<()> {
        info!("Updating whale analyzer parameters with feedback");
        
        // Adjust sensitivity based on false positive rate
        let false_positive_rate = feedback.false_positives as f64 / 
            (feedback.true_positives + feedback.false_positives) as f64;
            
        if false_positive_rate > 0.3 {
            // Too many false positives, increase threshold
            self.config.confidence_threshold = (self.config.confidence_threshold + 0.05).min(0.95);
        } else if false_positive_rate < 0.1 {
            // Very few false positives, can be more aggressive
            self.config.confidence_threshold = (self.config.confidence_threshold - 0.02).max(0.5);
        }
        
        // Apply specific parameter adjustments
        for (param, adjustment) in &feedback.parameter_adjustments {
            match param.as_str() {
                "whale_threshold_btc" => {
                    self.config.whale_threshold_btc = (self.config.whale_threshold_btc + adjustment).max(1.0);
                }
                "volume_spike_threshold" => {
                    self.config.volume_spike_threshold = (self.config.volume_spike_threshold + adjustment).max(1.5);
                }
                "imbalance_threshold" => {
                    self.config.imbalance_threshold = (self.config.imbalance_threshold + adjustment).clamp(0.5, 0.9);
                }
                "detection_sensitivity" => {
                    self.config.detection_sensitivity = (self.config.detection_sensitivity + adjustment).clamp(0.1, 1.0);
                }
                _ => {}
            }
        }
        
        info!("Updated whale analyzer configuration: {:?}", self.config);
        Ok(())
    }
}

/// Volume profile analyzer for institutional activity detection
#[derive(Debug, Clone)]
struct VolumeProfileAnalyzer {
    config: WhaleConfig,
}

impl VolumeProfileAnalyzer {
    fn new(config: &WhaleConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        // Implementation for volume profile analysis
        // This would detect unusual volume concentrations and patterns
        Ok(Vec::new())
    }
}

/// Order flow analyzer for detecting imbalances
#[derive(Debug, Clone)]
struct OrderFlowAnalyzer {
    config: WhaleConfig,
}

impl OrderFlowAnalyzer {
    fn new(config: &WhaleConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        // Implementation for order flow analysis
        // This would detect buy/sell pressure imbalances
        Ok(Vec::new())
    }
}

/// Smart money tracker for institutional behavior patterns
#[derive(Debug, Clone)]
struct SmartMoneyTracker {
    config: WhaleConfig,
}

impl SmartMoneyTracker {
    fn new(config: &WhaleConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn analyze(&self, data: &MarketData) -> Result<Vec<WhaleSignal>> {
        // Implementation for smart money pattern detection
        // This would identify institutional trading patterns
        Ok(Vec::new())
    }
}

/// Historical whale pattern for machine learning
#[derive(Debug, Clone)]
struct WhalePattern {
    signal_type: WhaleSignalType,
    market_conditions: MarketConditions,
    outcome: PatternOutcome,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct MarketConditions {
    volatility: f64,
    volume: f64,
    trend: f64,
    time_of_day: u32,
}

#[derive(Debug, Clone)]
enum PatternOutcome {
    Confirmed,
    FalsePositive,
    Partial,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_whale_analyzer_creation() {
        let config = Config::default();
        let analyzer = WhaleAnalyzer::new(&config);
        assert!(analyzer.is_ok());
    }
    
    #[tokio::test]
    async fn test_whale_analysis() {
        let config = Config::default();
        let analyzer = WhaleAnalyzer::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let signals = analyzer.analyze(&market_data).await;
        assert!(signals.is_ok());
    }
    
    #[test]
    fn test_volume_profile_calculation() {
        let config = Config::default();
        let analyzer = WhaleAnalyzer::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let volume_profile = analyzer.calculate_enhanced_volume_profile(&market_data);
        assert!(volume_profile.is_ok());
        
        let vp = volume_profile.unwrap();
        assert!(vp.value_area_high >= vp.value_area_low);
        assert!(vp.point_of_control >= vp.value_area_low);
        assert!(vp.point_of_control <= vp.value_area_high);
    }
    
    #[test]
    fn test_order_flow_calculation() {
        let config = Config::default();
        let analyzer = WhaleAnalyzer::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let order_flow = analyzer.calculate_enhanced_order_flow(&market_data);
        assert!(order_flow.is_ok());
        
        let of = order_flow.unwrap();
        assert!(of.buy_pressure >= 0.0 && of.buy_pressure <= 1.0);
        assert!(of.sell_pressure >= 0.0 && of.sell_pressure <= 1.0);
    }
}