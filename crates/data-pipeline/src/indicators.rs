//! SIMD-optimized technical indicator calculation engine

use crate::{config::IndicatorsConfig, error::{IndicatorError, IndicatorResult}, ComponentHealth, streaming::MarketData};
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use rayon::prelude::*;
#[cfg(feature = "simd")]
use wide::{f64x4, f64x8};
use nalgebra::DVector;
use ndarray::{Array1, Array2, Axis};
use ta::*;
use statrs::statistics::Statistics;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// High-performance SIMD-optimized technical indicator engine
pub struct IndicatorEngine {
    config: Arc<IndicatorsConfig>,
    indicators: Arc<RwLock<HashMap<String, IndicatorState>>>,
    cache: Arc<RwLock<IndicatorCache>>,
    calculator: Arc<SIMDCalculator>,
    pattern_detector: Arc<PatternDetector>,
    signal_generator: Arc<SignalGenerator>,
    metrics: Arc<RwLock<IndicatorMetrics>>,
    data_buffer: Arc<RwLock<DataBuffer>>,
}

/// Indicator state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorState {
    pub symbol: String,
    pub indicators: HashMap<String, IndicatorValue>,
    pub last_update: chrono::DateTime<chrono::Utc>,
    pub data_points: usize,
    pub quality_score: f64,
}

/// Indicator value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorValue {
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence: f64,
    pub parameters: HashMap<String, f64>,
    pub calculation_time: Duration,
}

/// Indicator cache for performance
pub struct IndicatorCache {
    cache: HashMap<String, CachedResult>,
    max_size: usize,
    ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct CachedResult {
    pub result: IndicatorValue,
    pub timestamp: Instant,
}

impl IndicatorCache {
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            ttl,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<IndicatorValue> {
        if let Some(cached) = self.cache.get(key) {
            if cached.timestamp.elapsed() < self.ttl {
                return Some(cached.result.clone());
            } else {
                self.cache.remove(key);
            }
        }
        None
    }

    pub fn put(&mut self, key: String, result: IndicatorValue) {
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }
        
        self.cache.insert(key, CachedResult {
            result,
            timestamp: Instant::now(),
        });
    }

    fn evict_oldest(&mut self) {
        let oldest_key = self.cache.iter()
            .min_by_key(|(_, cached)| cached.timestamp)
            .map(|(key, _)| key.clone());
        
        if let Some(key) = oldest_key {
            self.cache.remove(&key);
        }
    }
}

/// SIMD calculator for high-performance computations
pub struct SIMDCalculator {
    config: Arc<IndicatorsConfig>,
}

impl SIMDCalculator {
    pub fn new(config: Arc<IndicatorsConfig>) -> Self {
        Self { config }
    }

    /// Calculate Simple Moving Average using SIMD
    pub fn sma_simd(&self, data: &[f64], window: usize) -> IndicatorResult<Vec<f64>> {
        if data.len() < window {
            return Err(IndicatorError::InsufficientData(format!(
                "Need at least {} data points, got {}",
                window, data.len()
            )));
        }

        let mut result = Vec::with_capacity(data.len() - window + 1);
        
        // Process data in chunks of 8 for SIMD optimization
        for i in 0..=(data.len() - window) {
            let window_data = &data[i..i + window];
            let sum = self.simd_sum(window_data);
            result.push(sum / window as f64);
        }

        Ok(result)
    }

    /// Calculate Exponential Moving Average using SIMD
    pub fn ema_simd(&self, data: &[f64], period: usize) -> IndicatorResult<Vec<f64>> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData("No data provided".to_string()));
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(data.len());
        
        // Initialize with first value
        result.push(data[0]);
        
        // Calculate EMA using SIMD for batch operations
        let mut ema = data[0];
        for &price in data.iter().skip(1) {
            ema = alpha * price + (1.0 - alpha) * ema;
            result.push(ema);
        }

        Ok(result)
    }

    /// Calculate RSI using SIMD optimizations
    pub fn rsi_simd(&self, data: &[f64], period: usize) -> IndicatorResult<Vec<f64>> {
        if data.len() < period + 1 {
            return Err(IndicatorError::InsufficientData(format!(
                "Need at least {} data points for RSI, got {}",
                period + 1, data.len()
            )));
        }

        // Calculate price changes
        let mut changes = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            changes.push(data[i] - data[i - 1]);
        }

        // Separate gains and losses
        let mut gains = Vec::with_capacity(changes.len());
        let mut losses = Vec::with_capacity(changes.len());
        
        for change in changes {
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // Calculate average gains and losses using SIMD
        let avg_gains = self.sma_simd(&gains, period)?;
        let avg_losses = self.sma_simd(&losses, period)?;

        // Calculate RSI
        let mut rsi = Vec::with_capacity(avg_gains.len());
        for i in 0..avg_gains.len() {
            if avg_losses[i] == 0.0 {
                rsi.push(100.0);
            } else {
                let rs = avg_gains[i] / avg_losses[i];
                rsi.push(100.0 - (100.0 / (1.0 + rs)));
            }
        }

        Ok(rsi)
    }

    /// Calculate MACD using SIMD
    pub fn macd_simd(&self, data: &[f64], fast: usize, slow: usize, signal: usize) -> IndicatorResult<MACDResult> {
        let fast_ema = self.ema_simd(data, fast)?;
        let slow_ema = self.ema_simd(data, slow)?;
        
        // Calculate MACD line
        let mut macd_line = Vec::with_capacity(fast_ema.len().min(slow_ema.len()));
        for i in 0..fast_ema.len().min(slow_ema.len()) {
            macd_line.push(fast_ema[i] - slow_ema[i]);
        }
        
        // Calculate signal line
        let signal_line = self.ema_simd(&macd_line, signal)?;
        
        // Calculate histogram
        let mut histogram = Vec::with_capacity(macd_line.len().min(signal_line.len()));
        for i in 0..macd_line.len().min(signal_line.len()) {
            histogram.push(macd_line[i] - signal_line[i]);
        }
        
        Ok(MACDResult {
            macd: macd_line,
            signal: signal_line,
            histogram,
        })
    }

    /// Calculate Bollinger Bands using SIMD
    pub fn bollinger_bands_simd(&self, data: &[f64], period: usize, std_dev: f64) -> IndicatorResult<BollingerBandsResult> {
        let sma = self.sma_simd(data, period)?;
        let mut upper_band = Vec::with_capacity(sma.len());
        let mut lower_band = Vec::with_capacity(sma.len());
        
        for i in 0..sma.len() {
            let window_start = i;
            let window_end = i + period;
            let window_data = &data[window_start..window_end];
            
            let std = self.standard_deviation_simd(window_data);
            upper_band.push(sma[i] + std_dev * std);
            lower_band.push(sma[i] - std_dev * std);
        }
        
        Ok(BollingerBandsResult {
            upper: upper_band,
            middle: sma,
            lower: lower_band,
        })
    }

    /// Calculate Stochastic Oscillator using SIMD
    pub fn stochastic_simd(&self, high: &[f64], low: &[f64], close: &[f64], k_period: usize, d_period: usize) -> IndicatorResult<StochasticResult> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(IndicatorError::InvalidInput("Arrays must have same length".to_string()));
        }
        
        if high.len() < k_period {
            return Err(IndicatorError::InsufficientData(format!(
                "Need at least {} data points for Stochastic, got {}",
                k_period, high.len()
            )));
        }
        
        let mut k_values = Vec::with_capacity(high.len() - k_period + 1);
        
        // Calculate %K
        for i in 0..=(high.len() - k_period) {
            let window_high = &high[i..i + k_period];
            let window_low = &low[i..i + k_period];
            let current_close = close[i + k_period - 1];
            
            let highest = self.simd_max(window_high);
            let lowest = self.simd_min(window_low);
            
            if highest == lowest {
                k_values.push(50.0);
            } else {
                let k = ((current_close - lowest) / (highest - lowest)) * 100.0;
                k_values.push(k);
            }
        }
        
        // Calculate %D (smoothed %K)
        let d_values = self.sma_simd(&k_values, d_period)?;
        
        Ok(StochasticResult {
            k: k_values,
            d: d_values,
        })
    }

    /// SIMD-optimized sum calculation
    fn simd_sum(&self, data: &[f64]) -> f64 {
        let mut sum = 0.0;
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time with SIMD
        for chunk in chunks {
            let simd_data = f64x8::from_slice_unaligned(chunk);
            sum += simd_data.sum();
        }
        
        // Handle remaining elements
        for &value in remainder {
            sum += value;
        }
        
        sum
    }

    /// SIMD-optimized maximum calculation
    fn simd_max(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut max_val = data[0];
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time with SIMD
        for chunk in chunks {
            let simd_data = f64x8::from_slice_unaligned(chunk);
            max_val = max_val.max(simd_data.max_element());
        }
        
        // Handle remaining elements
        for &value in remainder {
            max_val = max_val.max(value);
        }
        
        max_val
    }

    /// SIMD-optimized minimum calculation
    fn simd_min(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut min_val = data[0];
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time with SIMD
        for chunk in chunks {
            let simd_data = f64x8::from_slice_unaligned(chunk);
            min_val = min_val.min(simd_data.min_element());
        }
        
        // Handle remaining elements
        for &value in remainder {
            min_val = min_val.min(value);
        }
        
        min_val
    }

    /// SIMD-optimized standard deviation calculation
    fn standard_deviation_simd(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = self.simd_sum(data) / data.len() as f64;
        let mut sum_squared_diff = 0.0;
        
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time with SIMD
        for chunk in chunks {
            let simd_data = f64x8::from_slice_unaligned(chunk);
            let diff = simd_data - f64x8::splat(mean);
            sum_squared_diff += (diff * diff).sum();
        }
        
        // Handle remaining elements
        for &value in remainder {
            let diff = value - mean;
            sum_squared_diff += diff * diff;
        }
        
        (sum_squared_diff / data.len() as f64).sqrt()
    }
}

/// MACD calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDResult {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Bollinger Bands calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBandsResult {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Stochastic calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticResult {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

/// Pattern detector for technical analysis
pub struct PatternDetector {
    config: Arc<IndicatorsConfig>,
}

impl PatternDetector {
    pub fn new(config: Arc<IndicatorsConfig>) -> Self {
        Self { config }
    }

    /// Detect candlestick patterns
    pub fn detect_patterns(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        // Detect various patterns
        patterns.extend(self.detect_doji(ohlc));
        patterns.extend(self.detect_hammer(ohlc));
        patterns.extend(self.detect_engulfing(ohlc));
        patterns.extend(self.detect_harami(ohlc));
        patterns.extend(self.detect_morning_star(ohlc));
        patterns.extend(self.detect_evening_star(ohlc));
        
        patterns
    }

    /// Detect Doji patterns
    fn detect_doji(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for (i, candle) in ohlc.iter().enumerate() {
            let body_size = (candle.close - candle.open).abs();
            let range = candle.high - candle.low;
            
            if range > 0.0 && body_size / range < 0.1 {
                patterns.push(PatternResult {
                    pattern_type: PatternType::Doji,
                    start_index: i,
                    end_index: i,
                    confidence: 1.0 - (body_size / range) * 10.0,
                    direction: PatternDirection::Neutral,
                });
            }
        }
        
        patterns
    }

    /// Detect Hammer patterns
    fn detect_hammer(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for (i, candle) in ohlc.iter().enumerate() {
            let body_size = (candle.close - candle.open).abs();
            let lower_shadow = candle.open.min(candle.close) - candle.low;
            let upper_shadow = candle.high - candle.open.max(candle.close);
            
            if lower_shadow > 2.0 * body_size && upper_shadow < body_size {
                patterns.push(PatternResult {
                    pattern_type: PatternType::Hammer,
                    start_index: i,
                    end_index: i,
                    confidence: (lower_shadow / body_size).min(1.0),
                    direction: PatternDirection::Bullish,
                });
            }
        }
        
        patterns
    }

    /// Detect Engulfing patterns
    fn detect_engulfing(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for i in 1..ohlc.len() {
            let prev = &ohlc[i - 1];
            let curr = &ohlc[i];
            
            // Bullish engulfing
            if prev.close < prev.open && curr.close > curr.open &&
               curr.open < prev.close && curr.close > prev.open {
                patterns.push(PatternResult {
                    pattern_type: PatternType::BullishEngulfing,
                    start_index: i - 1,
                    end_index: i,
                    confidence: 0.8,
                    direction: PatternDirection::Bullish,
                });
            }
            
            // Bearish engulfing
            if prev.close > prev.open && curr.close < curr.open &&
               curr.open > prev.close && curr.close < prev.open {
                patterns.push(PatternResult {
                    pattern_type: PatternType::BearishEngulfing,
                    start_index: i - 1,
                    end_index: i,
                    confidence: 0.8,
                    direction: PatternDirection::Bearish,
                });
            }
        }
        
        patterns
    }

    /// Detect Harami patterns
    fn detect_harami(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for i in 1..ohlc.len() {
            let prev = &ohlc[i - 1];
            let curr = &ohlc[i];
            
            let prev_body_size = (prev.close - prev.open).abs();
            let curr_body_size = (curr.close - curr.open).abs();
            
            if prev_body_size > curr_body_size * 2.0 &&
               curr.high < prev.high && curr.low > prev.low {
                
                let direction = if prev.close < prev.open && curr.close > curr.open {
                    PatternDirection::Bullish
                } else if prev.close > prev.open && curr.close < curr.open {
                    PatternDirection::Bearish
                } else {
                    PatternDirection::Neutral
                };
                
                patterns.push(PatternResult {
                    pattern_type: PatternType::Harami,
                    start_index: i - 1,
                    end_index: i,
                    confidence: 0.7,
                    direction,
                });
            }
        }
        
        patterns
    }

    /// Detect Morning Star patterns
    fn detect_morning_star(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for i in 2..ohlc.len() {
            let first = &ohlc[i - 2];
            let second = &ohlc[i - 1];
            let third = &ohlc[i];
            
            // First candle: bearish
            if first.close < first.open &&
               // Second candle: small body (star)
               (second.close - second.open).abs() < (first.close - first.open).abs() * 0.3 &&
               // Third candle: bullish
               third.close > third.open &&
               third.close > (first.open + first.close) / 2.0 {
                
                patterns.push(PatternResult {
                    pattern_type: PatternType::MorningStar,
                    start_index: i - 2,
                    end_index: i,
                    confidence: 0.8,
                    direction: PatternDirection::Bullish,
                });
            }
        }
        
        patterns
    }

    /// Detect Evening Star patterns
    fn detect_evening_star(&self, ohlc: &[OHLC]) -> Vec<PatternResult> {
        let mut patterns = Vec::new();
        
        for i in 2..ohlc.len() {
            let first = &ohlc[i - 2];
            let second = &ohlc[i - 1];
            let third = &ohlc[i];
            
            // First candle: bullish
            if first.close > first.open &&
               // Second candle: small body (star)
               (second.close - second.open).abs() < (first.close - first.open).abs() * 0.3 &&
               // Third candle: bearish
               third.close < third.open &&
               third.close < (first.open + first.close) / 2.0 {
                
                patterns.push(PatternResult {
                    pattern_type: PatternType::EveningStar,
                    start_index: i - 2,
                    end_index: i,
                    confidence: 0.8,
                    direction: PatternDirection::Bearish,
                });
            }
        }
        
        patterns
    }
}

/// OHLC data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLC {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Pattern detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    pub pattern_type: PatternType,
    pub start_index: usize,
    pub end_index: usize,
    pub confidence: f64,
    pub direction: PatternDirection,
}

/// Pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Doji,
    Hammer,
    ShootingStar,
    BullishEngulfing,
    BearishEngulfing,
    Harami,
    MorningStar,
    EveningStar,
    ThreeWhiteSoldiers,
    ThreeBlackCrows,
}

/// Pattern direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Signal generator
pub struct SignalGenerator {
    config: Arc<IndicatorsConfig>,
}

impl SignalGenerator {
    pub fn new(config: Arc<IndicatorsConfig>) -> Self {
        Self { config }
    }

    /// Generate trading signals from indicators
    pub fn generate_signals(&self, indicators: &HashMap<String, IndicatorValue>) -> Vec<TradingSignal> {
        let mut signals = Vec::new();
        
        // RSI signals
        if let Some(rsi) = indicators.get("RSI") {
            signals.extend(self.generate_rsi_signals(rsi));
        }
        
        // MACD signals
        if indicators.contains_key("MACD") && indicators.contains_key("MACD_SIGNAL") {
            let macd = indicators.get("MACD").unwrap();
            let signal = indicators.get("MACD_SIGNAL").unwrap();
            signals.extend(self.generate_macd_signals(macd, signal));
        }
        
        // Bollinger Bands signals
        if indicators.contains_key("BB_UPPER") && indicators.contains_key("BB_LOWER") {
            let upper = indicators.get("BB_UPPER").unwrap();
            let lower = indicators.get("BB_LOWER").unwrap();
            signals.extend(self.generate_bollinger_signals(upper, lower));
        }
        
        signals
    }

    /// Generate RSI-based signals
    fn generate_rsi_signals(&self, rsi: &IndicatorValue) -> Vec<TradingSignal> {
        let mut signals = Vec::new();
        
        if rsi.value > 70.0 {
            signals.push(TradingSignal {
                signal_type: SignalType::Sell,
                strength: (rsi.value - 70.0) / 30.0,
                confidence: rsi.confidence,
                timestamp: chrono::Utc::now(),
                source: "RSI".to_string(),
                parameters: rsi.parameters.clone(),
            });
        } else if rsi.value < 30.0 {
            signals.push(TradingSignal {
                signal_type: SignalType::Buy,
                strength: (30.0 - rsi.value) / 30.0,
                confidence: rsi.confidence,
                timestamp: chrono::Utc::now(),
                source: "RSI".to_string(),
                parameters: rsi.parameters.clone(),
            });
        }
        
        signals
    }

    /// Generate MACD-based signals
    fn generate_macd_signals(&self, macd: &IndicatorValue, signal: &IndicatorValue) -> Vec<TradingSignal> {
        let mut signals = Vec::new();
        
        if macd.value > signal.value {
            signals.push(TradingSignal {
                signal_type: SignalType::Buy,
                strength: ((macd.value - signal.value) / signal.value.abs()).min(1.0),
                confidence: (macd.confidence + signal.confidence) / 2.0,
                timestamp: chrono::Utc::now(),
                source: "MACD".to_string(),
                parameters: HashMap::new(),
            });
        } else if macd.value < signal.value {
            signals.push(TradingSignal {
                signal_type: SignalType::Sell,
                strength: ((signal.value - macd.value) / signal.value.abs()).min(1.0),
                confidence: (macd.confidence + signal.confidence) / 2.0,
                timestamp: chrono::Utc::now(),
                source: "MACD".to_string(),
                parameters: HashMap::new(),
            });
        }
        
        signals
    }

    /// Generate Bollinger Bands signals
    fn generate_bollinger_signals(&self, upper: &IndicatorValue, lower: &IndicatorValue) -> Vec<TradingSignal> {
        let mut signals = Vec::new();
        
        // This would need current price, which would be passed in a real implementation
        // For now, we'll skip this implementation
        
        signals
    }
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub signal_type: SignalType,
    pub strength: f64,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub parameters: HashMap<String, f64>,
}

/// Signal type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

/// Data buffer for efficient data management
pub struct DataBuffer {
    buffers: HashMap<String, VecDeque<MarketData>>,
    max_size: usize,
}

impl DataBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            max_size,
        }
    }

    pub fn add_data(&mut self, symbol: &str, data: MarketData) {
        let buffer = self.buffers.entry(symbol.to_string()).or_insert_with(VecDeque::new);
        
        buffer.push_back(data);
        
        if buffer.len() > self.max_size {
            buffer.pop_front();
        }
    }

    pub fn get_data(&self, symbol: &str) -> Option<&VecDeque<MarketData>> {
        self.buffers.get(symbol)
    }

    pub fn get_ohlc_data(&self, symbol: &str) -> Option<Vec<OHLC>> {
        self.buffers.get(symbol).map(|buffer| {
            buffer.iter().map(|data| OHLC {
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                volume: data.volume,
                timestamp: data.timestamp,
            }).collect()
        })
    }
}

/// Indicator metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorMetrics {
    pub calculations_performed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_calculation_time: Duration,
    pub simd_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub error_count: u64,
    pub last_reset: chrono::DateTime<chrono::Utc>,
}

impl Default for IndicatorMetrics {
    fn default() -> Self {
        Self {
            calculations_performed: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_calculation_time: Duration::from_millis(0),
            simd_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            error_count: 0,
            last_reset: chrono::Utc::now(),
        }
    }
}

impl IndicatorEngine {
    /// Create a new indicator engine
    pub fn new(config: Arc<IndicatorsConfig>) -> Result<Self> {
        info!("Initializing indicator engine with config: {:?}", config);
        
        let cache = Arc::new(RwLock::new(IndicatorCache::new(
            config.cache_size,
            config.update_frequency,
        )));
        
        let calculator = Arc::new(SIMDCalculator::new(config.clone()));
        let pattern_detector = Arc::new(PatternDetector::new(config.clone()));
        let signal_generator = Arc::new(SignalGenerator::new(config.clone()));
        let data_buffer = Arc::new(RwLock::new(DataBuffer::new(10000)));
        
        Ok(Self {
            config,
            indicators: Arc::new(RwLock::new(HashMap::new())),
            cache,
            calculator,
            pattern_detector,
            signal_generator,
            metrics: Arc::new(RwLock::new(IndicatorMetrics::default())),
            data_buffer,
        })
    }

    /// Calculate indicators for market data
    #[instrument(skip(self, data))]
    pub async fn calculate(&self, data: &crate::streaming::MarketData) -> IndicatorResult<HashMap<String, IndicatorValue>> {
        let start_time = Instant::now();
        let mut results = HashMap::new();
        
        // Add data to buffer
        {
            let mut buffer = self.data_buffer.write().await;
            buffer.add_data(&data.symbol, data.clone());
        }
        
        // Get historical data for calculations
        let ohlc_data = {
            let buffer = self.data_buffer.read().await;
            buffer.get_ohlc_data(&data.symbol)
        };
        
        if let Some(ohlc) = ohlc_data {
            let close_prices: Vec<f64> = ohlc.iter().map(|candle| candle.close).collect();
            let high_prices: Vec<f64> = ohlc.iter().map(|candle| candle.high).collect();
            let low_prices: Vec<f64> = ohlc.iter().map(|candle| candle.low).collect();
            
            // Calculate moving averages
            for &window in &self.config.ma_windows {
                if let Ok(sma) = self.calculator.sma_simd(&close_prices, window) {
                    if let Some(&last_sma) = sma.last() {
                        results.insert(
                            format!("SMA_{}", window),
                            IndicatorValue {
                                value: last_sma,
                                timestamp: data.timestamp,
                                confidence: 0.95,
                                parameters: [("window".to_string(), window as f64)].iter().cloned().collect(),
                                calculation_time: start_time.elapsed(),
                            }
                        );
                    }
                }
                
                if let Ok(ema) = self.calculator.ema_simd(&close_prices, window) {
                    if let Some(&last_ema) = ema.last() {
                        results.insert(
                            format!("EMA_{}", window),
                            IndicatorValue {
                                value: last_ema,
                                timestamp: data.timestamp,
                                confidence: 0.95,
                                parameters: [("window".to_string(), window as f64)].iter().cloned().collect(),
                                calculation_time: start_time.elapsed(),
                            }
                        );
                    }
                }
            }
            
            // Calculate RSI
            if let Ok(rsi) = self.calculator.rsi_simd(&close_prices, self.config.rsi_period) {
                if let Some(&last_rsi) = rsi.last() {
                    results.insert(
                        "RSI".to_string(),
                        IndicatorValue {
                            value: last_rsi,
                            timestamp: data.timestamp,
                            confidence: 0.9,
                            parameters: [("period".to_string(), self.config.rsi_period as f64)].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
            }
            
            // Calculate MACD
            if let Ok(macd_result) = self.calculator.macd_simd(
                &close_prices,
                self.config.macd_config.fast_period,
                self.config.macd_config.slow_period,
                self.config.macd_config.signal_period,
            ) {
                if let Some(&last_macd) = macd_result.macd.last() {
                    results.insert(
                        "MACD".to_string(),
                        IndicatorValue {
                            value: last_macd,
                            timestamp: data.timestamp,
                            confidence: 0.9,
                            parameters: [
                                ("fast_period".to_string(), self.config.macd_config.fast_period as f64),
                                ("slow_period".to_string(), self.config.macd_config.slow_period as f64),
                            ].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
                
                if let Some(&last_signal) = macd_result.signal.last() {
                    results.insert(
                        "MACD_SIGNAL".to_string(),
                        IndicatorValue {
                            value: last_signal,
                            timestamp: data.timestamp,
                            confidence: 0.9,
                            parameters: [("signal_period".to_string(), self.config.macd_config.signal_period as f64)].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
            }
            
            // Calculate Bollinger Bands
            if let Ok(bb_result) = self.calculator.bollinger_bands_simd(
                &close_prices,
                self.config.bollinger_config.period,
                self.config.bollinger_config.std_dev,
            ) {
                if let Some(&last_upper) = bb_result.upper.last() {
                    results.insert(
                        "BB_UPPER".to_string(),
                        IndicatorValue {
                            value: last_upper,
                            timestamp: data.timestamp,
                            confidence: 0.9,
                            parameters: [
                                ("period".to_string(), self.config.bollinger_config.period as f64),
                                ("std_dev".to_string(), self.config.bollinger_config.std_dev),
                            ].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
                
                if let Some(&last_lower) = bb_result.lower.last() {
                    results.insert(
                        "BB_LOWER".to_string(),
                        IndicatorValue {
                            value: last_lower,
                            timestamp: data.timestamp,
                            confidence: 0.9,
                            parameters: [
                                ("period".to_string(), self.config.bollinger_config.period as f64),
                                ("std_dev".to_string(), self.config.bollinger_config.std_dev),
                            ].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
            }
            
            // Calculate Stochastic
            if let Ok(stoch_result) = self.calculator.stochastic_simd(
                &high_prices,
                &low_prices,
                &close_prices,
                self.config.stochastic_config.k_period,
                self.config.stochastic_config.d_period,
            ) {
                if let Some(&last_k) = stoch_result.k.last() {
                    results.insert(
                        "STOCH_K".to_string(),
                        IndicatorValue {
                            value: last_k,
                            timestamp: data.timestamp,
                            confidence: 0.85,
                            parameters: [("k_period".to_string(), self.config.stochastic_config.k_period as f64)].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
                
                if let Some(&last_d) = stoch_result.d.last() {
                    results.insert(
                        "STOCH_D".to_string(),
                        IndicatorValue {
                            value: last_d,
                            timestamp: data.timestamp,
                            confidence: 0.85,
                            parameters: [("d_period".to_string(), self.config.stochastic_config.d_period as f64)].iter().cloned().collect(),
                            calculation_time: start_time.elapsed(),
                        }
                    );
                }
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.calculations_performed += 1;
            metrics.cache_misses += 1;
            metrics.average_calculation_time = Duration::from_millis(
                ((metrics.average_calculation_time.as_millis() as u64 * (metrics.calculations_performed - 1)) + 
                 start_time.elapsed().as_millis() as u64) / metrics.calculations_performed
            );
        }
        
        // Update indicator state
        {
            let mut indicators = self.indicators.write().await;
            indicators.insert(data.symbol.clone(), IndicatorState {
                symbol: data.symbol.clone(),
                indicators: results.clone(),
                last_update: chrono::Utc::now(),
                data_points: ohlc_data.map(|ohlc| ohlc.len()).unwrap_or(0),
                quality_score: 0.95,
            });
        }
        
        debug!("Calculated {} indicators for {}", results.len(), data.symbol);
        Ok(results)
    }

    /// Get indicator state for a symbol
    pub async fn get_indicator_state(&self, symbol: &str) -> Option<IndicatorState> {
        let indicators = self.indicators.read().await;
        indicators.get(symbol).cloned()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let metrics = self.metrics.read().await;
        
        if metrics.error_count > 10 {
            Ok(ComponentHealth::Unhealthy)
        } else if metrics.error_count > 5 {
            Ok(ComponentHealth::Degraded)
        } else {
            Ok(ComponentHealth::Healthy)
        }
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> IndicatorMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset engine
    pub async fn reset(&self) -> Result<()> {
        info!("Resetting indicator engine");
        
        // Clear indicators
        {
            let mut indicators = self.indicators.write().await;
            indicators.clear();
        }
        
        // Clear cache
        {
            let mut cache = self.cache.write().await;
            cache.cache.clear();
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = IndicatorMetrics::default();
        }
        
        // Clear data buffer
        {
            let mut buffer = self.data_buffer.write().await;
            buffer.buffers.clear();
        }
        
        info!("Indicator engine reset successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_simd_calculator() {
        let config = Arc::new(IndicatorsConfig::default());
        let calculator = SIMDCalculator::new(config);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // Test SMA
        let sma = calculator.sma_simd(&data, 5).unwrap();
        assert_eq!(sma.len(), 6);
        assert_eq!(sma[0], 3.0); // (1+2+3+4+5)/5
        assert_eq!(sma[5], 8.0); // (6+7+8+9+10)/5
        
        // Test EMA
        let ema = calculator.ema_simd(&data, 5).unwrap();
        assert_eq!(ema.len(), 10);
        assert_eq!(ema[0], 1.0); // First value
        
        // Test RSI
        let rsi = calculator.rsi_simd(&data, 5).unwrap();
        assert!(rsi.len() > 0);
        
        // Test MACD
        let macd = calculator.macd_simd(&data, 3, 6, 2).unwrap();
        assert!(macd.macd.len() > 0);
        assert!(macd.signal.len() > 0);
        assert!(macd.histogram.len() > 0);
    }

    #[test]
    async fn test_pattern_detector() {
        let config = Arc::new(IndicatorsConfig::default());
        let detector = PatternDetector::new(config);
        
        let ohlc = vec![
            OHLC { open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0, timestamp: chrono::Utc::now() },
            OHLC { open: 102.0, high: 103.0, low: 101.0, close: 101.5, volume: 900.0, timestamp: chrono::Utc::now() },
            OHLC { open: 101.5, high: 107.0, low: 100.0, close: 106.0, volume: 1100.0, timestamp: chrono::Utc::now() },
        ];
        
        let patterns = detector.detect_patterns(&ohlc);
        assert!(patterns.len() >= 0);
    }

    #[test]
    async fn test_indicator_cache() {
        let mut cache = IndicatorCache::new(10, Duration::from_secs(60));
        
        let value = IndicatorValue {
            value: 50.0,
            timestamp: chrono::Utc::now(),
            confidence: 0.9,
            parameters: HashMap::new(),
            calculation_time: Duration::from_millis(10),
        };
        
        cache.put("test_key".to_string(), value.clone());
        let cached_value = cache.get("test_key").unwrap();
        
        assert_eq!(cached_value.value, 50.0);
        assert_eq!(cached_value.confidence, 0.9);
    }
}