//! Advanced surveillance system for detecting market manipulation and suspicious patterns

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use rust_decimal::{Decimal, prelude::FromStr};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use statrs::statistics::{Statistics, Data};
use dashmap::DashMap;
use regex::Regex;
use crate::error::{ComplianceError, ComplianceResult};

/// Trade record for surveillance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub trader_id: String,
    pub order_id: Uuid,
    pub venue: String,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Order record for surveillance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub trader_id: String,
    pub status: OrderStatus,
    pub time_in_force: String,
    pub filled_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Suspicious pattern detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousPattern {
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub description: String,
    pub evidence: Vec<String>,
    pub trades_involved: Vec<Uuid>,
    pub risk_score: f64,
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    WashTrading,
    Spoofing,
    Layering,
    Momentum,
    Marking,
    Cornering,
    UnusualVolume,
    AbnormalTiming,
    PriceManipulation,
    CrossTrade,
}

/// Wash trading detector
pub struct WashTradingDetector {
    time_window: Duration,
    price_tolerance: Decimal,
    min_confidence: f64,
}

impl WashTradingDetector {
    pub fn new(time_window: Duration, price_tolerance: Decimal, min_confidence: f64) -> Self {
        Self {
            time_window,
            price_tolerance,
            min_confidence,
        }
    }

    pub fn analyze(&self, trades: &[TradeRecord]) -> Vec<SuspiciousPattern> {
        let mut patterns = Vec::new();
        
        for window_start in 0..trades.len() {
            let window_end = trades.len().min(window_start + 100); // Limit window size
            let window_trades = &trades[window_start..window_end];
            
            if let Some(pattern) = self.detect_wash_trading(window_trades) {
                if pattern.confidence >= self.min_confidence {
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }

    fn detect_wash_trading(&self, trades: &[TradeRecord]) -> Option<SuspiciousPattern> {
        let mut suspicious_pairs = Vec::new();
        
        for i in 0..trades.len() {
            for j in i + 1..trades.len() {
                let trade1 = &trades[i];
                let trade2 = &trades[j];
                
                // Check if trades are from same trader (red flag)
                if trade1.trader_id == trade2.trader_id {
                    continue;
                }
                
                // Check time proximity
                let time_diff = trade2.timestamp.signed_duration_since(trade1.timestamp);
                if time_diff.to_std().unwrap_or(Duration::MAX) > self.time_window {
                    continue;
                }
                
                // Check if trades cancel each other out (buy/sell same symbol)
                if trade1.symbol == trade2.symbol &&
                   matches!((trade1.side.clone(), trade2.side.clone()), 
                           (TradeSide::Buy, TradeSide::Sell) | (TradeSide::Sell, TradeSide::Buy)) {
                    
                    // Check price proximity
                    let price_diff = (trade1.price - trade2.price).abs();
                    if price_diff <= self.price_tolerance {
                        // Check quantity similarity
                        let qty_ratio = if trade2.quantity > Decimal::ZERO {
                            (trade1.quantity / trade2.quantity).abs()
                        } else {
                            Decimal::ZERO
                        };
                        
                        if qty_ratio >= Decimal::from_str("0.8").unwrap() && 
                           qty_ratio <= Decimal::from_str("1.2").unwrap() {
                            suspicious_pairs.push((trade1.id, trade2.id));
                        }
                    }
                }
            }
        }
        
        if !suspicious_pairs.is_empty() {
            let confidence = self.calculate_wash_trading_confidence(&suspicious_pairs, trades);
            
            Some(SuspiciousPattern {
                pattern_type: PatternType::WashTrading,
                confidence,
                description: "Potential wash trading detected - offsetting trades with similar prices and quantities".to_string(),
                evidence: vec![
                    format!("Found {} suspicious trade pairs", suspicious_pairs.len()),
                    format!("Trades occurred within {} seconds", self.time_window.as_secs()),
                ],
                trades_involved: suspicious_pairs.into_iter().flat_map(|(a, b)| vec![a, b]).collect(),
                risk_score: confidence / 100.0,
                detected_at: Utc::now(),
            })
        } else {
            None
        }
    }

    fn calculate_wash_trading_confidence(&self, _pairs: &[(Uuid, Uuid)], _trades: &[TradeRecord]) -> f64 {
        // Simplified confidence calculation
        // In production, this would be much more sophisticated
        75.0
    }
}

/// Spoofing detector
pub struct SpoofingDetector {
    cancel_ratio_threshold: f64,
    time_window: Duration,
    min_orders: usize,
}

impl SpoofingDetector {
    pub fn new(cancel_ratio_threshold: f64, time_window: Duration, min_orders: usize) -> Self {
        Self {
            cancel_ratio_threshold,
            time_window,
            min_orders,
        }
    }

    pub fn analyze(&self, orders: &[OrderRecord]) -> Vec<SuspiciousPattern> {
        let mut patterns = Vec::new();
        
        // Group orders by trader and symbol
        let mut trader_orders: HashMap<String, HashMap<String, Vec<&OrderRecord>>> = HashMap::new();
        
        for order in orders {
            trader_orders
                .entry(order.trader_id.clone())
                .or_default()
                .entry(order.symbol.clone())
                .or_default()
                .push(order);
        }
        
        for (trader_id, symbol_orders) in trader_orders {
            for (symbol, orders) in symbol_orders {
                if let Some(pattern) = self.detect_spoofing(&trader_id, &symbol, &orders) {
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }

    fn detect_spoofing(&self, trader_id: &str, symbol: &str, orders: &[&OrderRecord]) -> Option<SuspiciousPattern> {
        if orders.len() < self.min_orders {
            return None;
        }
        
        let mut cancelled_count = 0;
        let mut total_count = 0;
        let mut large_orders_cancelled = 0;
        let mut order_ids = Vec::new();
        
        for order in orders {
            total_count += 1;
            order_ids.push(order.id);
            
            if matches!(order.status, OrderStatus::Cancelled) {
                cancelled_count += 1;
                
                // Check if it was a large order
                if order.quantity > Decimal::from(1000) { // Configurable threshold
                    large_orders_cancelled += 1;
                }
            }
        }
        
        let cancel_ratio = cancelled_count as f64 / total_count as f64;
        
        if cancel_ratio >= self.cancel_ratio_threshold {
            let confidence = self.calculate_spoofing_confidence(cancel_ratio, large_orders_cancelled, total_count);
            
            Some(SuspiciousPattern {
                pattern_type: PatternType::Spoofing,
                confidence,
                description: format!("Potential spoofing detected for {} in {}", trader_id, symbol),
                evidence: vec![
                    format!("Cancel ratio: {:.2}%", cancel_ratio * 100.0),
                    format!("Large orders cancelled: {}", large_orders_cancelled),
                    format!("Total orders: {}", total_count),
                ],
                trades_involved: order_ids,
                risk_score: confidence / 100.0,
                detected_at: Utc::now(),
            })
        } else {
            None
        }
    }

    fn calculate_spoofing_confidence(&self, cancel_ratio: f64, large_cancelled: usize, total: usize) -> f64 {
        let base_confidence = (cancel_ratio - self.cancel_ratio_threshold) * 100.0;
        let large_order_boost = (large_cancelled as f64 / total as f64) * 20.0;
        
        (base_confidence + large_order_boost).min(95.0)
    }
}

/// Volume anomaly detector
pub struct VolumeAnomalyDetector {
    lookback_periods: usize,
    anomaly_threshold: f64, // Standard deviations
}

impl VolumeAnomalyDetector {
    pub fn new(lookback_periods: usize, anomaly_threshold: f64) -> Self {
        Self {
            lookback_periods,
            anomaly_threshold,
        }
    }

    pub fn analyze(&self, trades: &[TradeRecord]) -> Vec<SuspiciousPattern> {
        let mut patterns = Vec::new();
        
        // Group trades by symbol and time buckets
        let mut symbol_volumes: HashMap<String, VecDeque<Decimal>> = HashMap::new();
        
        for trade in trades {
            let volume_data = symbol_volumes
                .entry(trade.symbol.clone())
                .or_insert_with(VecDeque::new);
            
            volume_data.push_back(trade.quantity);
            
            if volume_data.len() > self.lookback_periods {
                volume_data.pop_front();
            }
            
            if volume_data.len() >= self.lookback_periods {
                if let Some(pattern) = self.check_volume_anomaly(&trade.symbol, volume_data, trade) {
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }

    fn check_volume_anomaly(&self, symbol: &str, volumes: &VecDeque<Decimal>, current_trade: &TradeRecord) -> Option<SuspiciousPattern> {
        let values: Vec<f64> = volumes.iter()
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();
        
        if values.len() < 3 {
            return None;
        }
        
        let data = Data::new(values);
        let mean = data.mean().unwrap_or(0.0);
        let std_dev = data.std_dev().unwrap_or(0.0);
        
        let current_volume = current_trade.quantity.to_f64().unwrap_or(0.0);
        let z_score = if std_dev > 0.0 {
            (current_volume - mean) / std_dev
        } else {
            0.0
        };
        
        if z_score.abs() > self.anomaly_threshold {
            let confidence = (z_score.abs() / self.anomaly_threshold * 50.0).min(90.0);
            
            Some(SuspiciousPattern {
                pattern_type: PatternType::UnusualVolume,
                confidence,
                description: format!("Unusual volume detected for {}", symbol),
                evidence: vec![
                    format!("Current volume: {}", current_volume),
                    format!("Historical mean: {:.2}", mean),
                    format!("Z-score: {:.2}", z_score),
                ],
                trades_involved: vec![current_trade.id],
                risk_score: confidence / 100.0,
                detected_at: Utc::now(),
            })
        } else {
            None
        }
    }
}

/// Main surveillance engine
pub struct SurveillanceEngine {
    wash_trading_detector: WashTradingDetector,
    spoofing_detector: SpoofingDetector,
    volume_anomaly_detector: VolumeAnomalyDetector,
    
    // Data storage
    trade_history: Arc<RwLock<VecDeque<TradeRecord>>>,
    order_history: Arc<RwLock<VecDeque<OrderRecord>>>,
    
    // Pattern cache
    detected_patterns: Arc<DashMap<String, Vec<SuspiciousPattern>>>,
    
    // Configuration
    max_history_size: usize,
    analysis_frequency: Duration,
}

impl SurveillanceEngine {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            wash_trading_detector: WashTradingDetector::new(
                Duration::from_secs(300), // 5 minute window
                Decimal::from_str("0.01").unwrap(), // 1 cent tolerance
                70.0, // 70% confidence threshold
            ),
            spoofing_detector: SpoofingDetector::new(
                0.8, // 80% cancel ratio
                Duration::from_secs(60), // 1 minute window
                10, // minimum 10 orders
            ),
            volume_anomaly_detector: VolumeAnomalyDetector::new(
                20, // 20 period lookback
                3.0, // 3 standard deviations
            ),
            trade_history: Arc::new(RwLock::new(VecDeque::new())),
            order_history: Arc::new(RwLock::new(VecDeque::new())),
            detected_patterns: Arc::new(DashMap::new()),
            max_history_size,
            analysis_frequency: Duration::from_secs(30),
        }
    }

    pub fn record_trade(&self, trade: TradeRecord) {
        let mut history = self.trade_history.write();
        history.push_back(trade);
        
        if history.len() > self.max_history_size {
            history.pop_front();
        }
    }

    pub fn record_order(&self, order: OrderRecord) {
        let mut history = self.order_history.write();
        history.push_back(order);
        
        if history.len() > self.max_history_size {
            history.pop_front();
        }
    }

    pub async fn analyze_patterns(&self) -> ComplianceResult<Vec<SuspiciousPattern>> {
        let mut all_patterns = Vec::new();
        
        // Analyze trades
        let trades: Vec<TradeRecord> = self.trade_history.read().iter().cloned().collect();
        let orders: Vec<OrderRecord> = self.order_history.read().iter().cloned().collect();
        
        // Run wash trading detection
        let wash_patterns = self.wash_trading_detector.analyze(&trades);
        all_patterns.extend(wash_patterns);
        
        // Run spoofing detection
        let order_refs: Vec<&OrderRecord> = orders.iter().collect();
        let spoofing_patterns = self.spoofing_detector.analyze(&order_refs);
        all_patterns.extend(spoofing_patterns);
        
        // Run volume anomaly detection
        let volume_patterns = self.volume_anomaly_detector.analyze(&trades);
        all_patterns.extend(volume_patterns);
        
        // Cache patterns by symbol
        for pattern in &all_patterns {
            if let Some(trade_id) = pattern.trades_involved.first() {
                // Find the symbol for this pattern
                if let Some(trade) = trades.iter().find(|t| t.id == *trade_id) {
                    self.detected_patterns
                        .entry(trade.symbol.clone())
                        .or_insert_with(Vec::new)
                        .push(pattern.clone());
                }
            }
        }
        
        // Check if any critical patterns require immediate action
        for pattern in &all_patterns {
            if pattern.risk_score > 0.8 {
                tracing::error!(
                    "CRITICAL PATTERN DETECTED: {:?} with confidence {:.1}%", 
                    pattern.pattern_type, 
                    pattern.confidence
                );
                
                // For critical patterns, we might want to trigger circuit breakers
                if pattern.confidence > 90.0 {
                    return Err(ComplianceError::MarketManipulation {
                        pattern: format!("{:?}", pattern.pattern_type),
                        confidence: pattern.confidence,
                    });
                }
            }
        }
        
        Ok(all_patterns)
    }

    pub fn get_patterns_for_symbol(&self, symbol: &str) -> Vec<SuspiciousPattern> {
        self.detected_patterns
            .get(symbol)
            .map(|patterns| patterns.clone())
            .unwrap_or_default()
    }

    pub fn get_all_patterns(&self) -> Vec<SuspiciousPattern> {
        self.detected_patterns
            .iter()
            .flat_map(|entry| entry.value().clone())
            .collect()
    }

    pub fn clear_old_patterns(&self, cutoff: DateTime<Utc>) {
        for mut entry in self.detected_patterns.iter_mut() {
            entry.value_mut().retain(|pattern| pattern.detected_at > cutoff);
        }
    }

    pub fn get_surveillance_statistics(&self) -> SurveillanceStatistics {
        let trade_count = self.trade_history.read().len();
        let order_count = self.order_history.read().len();
        let pattern_count = self.detected_patterns.len();
        
        let high_risk_patterns = self.get_all_patterns()
            .into_iter()
            .filter(|p| p.risk_score > 0.7)
            .count();
        
        SurveillanceStatistics {
            trades_monitored: trade_count,
            orders_monitored: order_count,
            patterns_detected: pattern_count,
            high_risk_patterns,
            last_analysis: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SurveillanceStatistics {
    pub trades_monitored: usize,
    pub orders_monitored: usize,
    pub patterns_detected: usize,
    pub high_risk_patterns: usize,
    pub last_analysis: DateTime<Utc>,
}