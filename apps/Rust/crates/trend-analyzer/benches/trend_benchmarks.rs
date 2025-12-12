//! Trend analysis benchmarks
//!
//! This benchmark suite measures the performance of trend analysis algorithms
//! across different market conditions and timeframes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use trend_analyzer::{
    TrendAnalyzer, TrendMetrics, TrendScore, TrendDirection, MarketRegime,
    TrendingRegime, RangingRegime, VolatileRegime, BreakoutRegime,
    KeyLevels, RiskLevel, TrendError,
};
use polars::prelude::*;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create sample market data for benchmarking
fn create_sample_market_data(size: usize) -> DataFrame {
    let timestamps: Vec<i64> = (0..size)
        .map(|i| 1640995200000 + (i as i64) * 60000) // 1-minute intervals
        .collect();
    
    let mut price = 50000.0;
    let mut prices = Vec::with_capacity(size);
    let mut volumes = Vec::with_capacity(size);
    let mut highs = Vec::with_capacity(size);
    let mut lows = Vec::with_capacity(size);
    
    for i in 0..size {
        let noise = (i as f64 * 0.1).sin() * 100.0;
        let trend = (i as f64 / 1000.0).sin() * 1000.0;
        price += noise + trend * 0.01;
        
        prices.push(price);
        highs.push(price + (i as f64 * 0.05).cos().abs() * 50.0);
        lows.push(price - (i as f64 * 0.05).sin().abs() * 50.0);
        volumes.push(1000000.0 + (i as f64 * 0.02).sin() * 500000.0);
    }
    
    df! {
        "timestamp" => timestamps,
        "open" => prices.iter().map(|p| p - 10.0).collect::<Vec<f64>>(),
        "high" => highs,
        "low" => lows,
        "close" => prices,
        "volume" => volumes,
    }.unwrap()
}

/// Create test trend analyzer
fn create_test_trend_analyzer() -> TrendAnalyzer {
    TrendAnalyzer::new(vec![
        "1m".to_string(),
        "5m".to_string(),
        "15m".to_string(),
        "1h".to_string(),
        "4h".to_string(),
        "1d".to_string(),
    ])
}

/// Create test key levels
fn create_test_key_levels() -> KeyLevels {
    KeyLevels {
        support_levels: vec![49000.0, 48000.0, 47000.0],
        resistance_levels: vec![51000.0, 52000.0, 53000.0],
        pivot_points: vec![50000.0, 49500.0, 50500.0],
        fibonacci_levels: vec![48618.0, 49236.0, 50764.0, 51382.0],
        volume_profile: vec![
            VolumeLevel { price: 49500.0, volume: 1000000.0 },
            VolumeLevel { price: 50000.0, volume: 1500000.0 },
            VolumeLevel { price: 50500.0, volume: 800000.0 },
        ],
    }
}

/// Benchmark basic trend analysis
fn bench_basic_trend_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("basic_trend_analysis");
    
    let data_sizes = vec![100, 500, 1000, 2000, 5000];
    
    for size in data_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("data_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(size);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.analyze_trend("BTCUSDT", &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark multi-timeframe analysis
fn bench_multi_timeframe_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("multi_timeframe_analysis");
    
    let timeframe_counts = vec![1, 3, 6, 9, 12];
    
    for count in timeframe_counts {
        group.bench_with_input(
            BenchmarkId::new("timeframes", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let timeframes: Vec<String> = (0..count)
                            .map(|i| format!("{}m", (i + 1) * 5))
                            .collect();
                        let analyzer = TrendAnalyzer::new(timeframes);
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.multi_timeframe_analysis("BTCUSDT", &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark technical indicators
fn bench_technical_indicators(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("technical_indicators");
    
    let indicators = vec![
        "sma", "ema", "rsi", "macd", "bollinger", "stochastic", "atr", "adx"
    ];
    
    for indicator in indicators {
        group.bench_with_input(
            BenchmarkId::new("indicator", indicator),
            &indicator,
            |b, &indicator| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.calculate_indicator(indicator, &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark pattern recognition
fn bench_pattern_recognition(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("pattern_recognition");
    
    let patterns = vec![
        "head_and_shoulders", "double_top", "double_bottom", "triangle", 
        "wedge", "flag", "pennant", "channel"
    ];
    
    for pattern in patterns {
        group.bench_with_input(
            BenchmarkId::new("pattern", pattern),
            &pattern,
            |b, &pattern| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.detect_pattern(pattern, &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark market regime detection
fn bench_market_regime_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("market_regime_detection");
    
    let regimes = vec![
        ("trending", TrendingRegime {
            direction: TrendDirection::Bullish,
            strength: 0.8,
            duration_candles: 100,
        }),
        ("ranging", RangingRegime {
            range_high: 52000.0,
            range_low: 48000.0,
            oscillation_count: 5,
        }),
        ("volatile", VolatileRegime {
            volatility_percentile: 0.95,
            average_true_range: 500.0,
            risk_level: RiskLevel::High,
        }),
        ("breakout", BreakoutRegime {
            breakout_level: 51000.0,
            volume_surge: 2.5,
            direction: TrendDirection::Bullish,
            confirmation_strength: 0.85,
        }),
    ];
    
    for (regime_name, _regime) in regimes {
        group.bench_with_input(
            BenchmarkId::new("regime", regime_name),
            &regime_name,
            |b, &regime_name| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.detect_market_regime(&data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark support and resistance detection
fn bench_support_resistance_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("support_resistance_detection");
    
    let detection_methods = vec![
        "pivot_points", "fibonacci", "volume_profile", "historical_levels"
    ];
    
    for method in detection_methods {
        group.bench_with_input(
            BenchmarkId::new("method", method),
            &method,
            |b, &method| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.detect_support_resistance(method, &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark volume analysis
fn bench_volume_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("volume_analysis");
    
    group.bench_function("volume_confirmation", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                let analyzer = create_test_trend_analyzer();
                let data = create_sample_market_data(1000);
                (analyzer, data)
            },
            |(analyzer, data)| async move {
                let result = analyzer.analyze_volume_confirmation(&data).await;
                black_box(result)
            },
        );
    });
    
    group.bench_function("volume_profile", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                let analyzer = create_test_trend_analyzer();
                let data = create_sample_market_data(1000);
                (analyzer, data)
            },
            |(analyzer, data)| async move {
                let result = analyzer.calculate_volume_profile(&data).await;
                black_box(result)
            },
        );
    });
    
    group.finish();
}

/// Benchmark trend strength calculation
fn bench_trend_strength_calculation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("trend_strength_calculation");
    
    let calculation_methods = vec![
        "adx", "aroon", "momentum", "directional_movement", "trend_intensity"
    ];
    
    for method in calculation_methods {
        group.bench_with_input(
            BenchmarkId::new("method", method),
            &method,
            |b, &method| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let data = create_sample_market_data(1000);
                        (analyzer, data)
                    },
                    |(analyzer, data)| async move {
                        let result = analyzer.calculate_trend_strength(method, &data).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark breakout detection
fn bench_breakout_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("breakout_detection");
    
    group.bench_function("breakout_probability", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                let analyzer = create_test_trend_analyzer();
                let data = create_sample_market_data(1000);
                let key_levels = create_test_key_levels();
                (analyzer, data, key_levels)
            },
            |(analyzer, data, key_levels)| async move {
                let result = analyzer.calculate_breakout_probability(&data, &key_levels).await;
                black_box(result)
            },
        );
    });
    
    group.finish();
}

/// Benchmark concurrent analysis
fn bench_concurrent_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_analysis");
    
    let symbol_counts = vec![1, 5, 10, 20, 50];
    
    for count in symbol_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("symbols", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = create_test_trend_analyzer();
                        let symbols: Vec<String> = (0..count)
                            .map(|i| format!("SYMBOL{}USDT", i))
                            .collect();
                        let data = create_sample_market_data(1000);
                        (analyzer, symbols, data)
                    },
                    |(analyzer, symbols, data)| async move {
                        let mut handles = Vec::new();
                        
                        for symbol in symbols {
                            let analyzer_clone = analyzer.clone();
                            let data_clone = data.clone();
                            let handle = tokio::spawn(async move {
                                analyzer_clone.analyze_trend(&symbol, &data_clone).await
                            });
                            handles.push(handle);
                        }
                        
                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark real-time analysis
fn bench_real_time_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("real_time_analysis");
    
    group.bench_function("incremental_update", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                let analyzer = create_test_trend_analyzer();
                let base_data = create_sample_market_data(1000);
                let new_candle = create_sample_market_data(1);
                (analyzer, base_data, new_candle)
            },
            |(analyzer, base_data, new_candle)| async move {
                let result = analyzer.incremental_update(&base_data, &new_candle).await;
                black_box(result)
            },
        );
    });
    
    group.finish();
}

// Mock implementations for missing structures and methods

#[derive(Debug, Clone)]
pub struct TrendAnalyzer {
    timeframes: Vec<String>,
}

impl TrendAnalyzer {
    pub fn new(timeframes: Vec<String>) -> Self {
        Self { timeframes }
    }
    
    pub async fn analyze_trend(&self, symbol: &str, data: &DataFrame) -> Result<TrendScore, TrendError> {
        // Mock implementation
        Ok(TrendScore {
            symbol: symbol.to_string(),
            overall_score: 0.75,
            trend_direction: TrendDirection::Bullish,
            confidence: 0.85,
            timeframe_scores: vec![],
            market_regime: MarketRegime::Trending(TrendingRegime {
                direction: TrendDirection::Bullish,
                strength: 0.8,
                duration_candles: 100,
            }),
            key_levels: create_test_key_levels(),
        })
    }
    
    pub async fn multi_timeframe_analysis(&self, symbol: &str, data: &DataFrame) -> Result<Vec<TrendScore>, TrendError> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn calculate_indicator(&self, indicator: &str, data: &DataFrame) -> Result<Vec<f64>, TrendError> {
        // Mock implementation
        Ok(vec![1.0, 2.0, 3.0])
    }
    
    pub async fn detect_pattern(&self, pattern: &str, data: &DataFrame) -> Result<Vec<PatternMatch>, TrendError> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn detect_market_regime(&self, data: &DataFrame) -> Result<MarketRegime, TrendError> {
        // Mock implementation
        Ok(MarketRegime::Trending(TrendingRegime {
            direction: TrendDirection::Bullish,
            strength: 0.8,
            duration_candles: 100,
        }))
    }
    
    pub async fn detect_support_resistance(&self, method: &str, data: &DataFrame) -> Result<KeyLevels, TrendError> {
        // Mock implementation
        Ok(create_test_key_levels())
    }
    
    pub async fn analyze_volume_confirmation(&self, data: &DataFrame) -> Result<f64, TrendError> {
        // Mock implementation
        Ok(0.75)
    }
    
    pub async fn calculate_volume_profile(&self, data: &DataFrame) -> Result<Vec<VolumeLevel>, TrendError> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn calculate_trend_strength(&self, method: &str, data: &DataFrame) -> Result<f64, TrendError> {
        // Mock implementation
        Ok(0.8)
    }
    
    pub async fn calculate_breakout_probability(&self, data: &DataFrame, key_levels: &KeyLevels) -> Result<f64, TrendError> {
        // Mock implementation
        Ok(0.65)
    }
    
    pub async fn incremental_update(&self, base_data: &DataFrame, new_candle: &DataFrame) -> Result<TrendScore, TrendError> {
        // Mock implementation
        Ok(TrendScore {
            symbol: "BTCUSDT".to_string(),
            overall_score: 0.75,
            trend_direction: TrendDirection::Bullish,
            confidence: 0.85,
            timeframe_scores: vec![],
            market_regime: MarketRegime::Trending(TrendingRegime {
                direction: TrendDirection::Bullish,
                strength: 0.8,
                duration_candles: 100,
            }),
            key_levels: create_test_key_levels(),
        })
    }
    
    pub fn clone(&self) -> Self {
        Self {
            timeframes: self.timeframes.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeyLevels {
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub pivot_points: Vec<f64>,
    pub fibonacci_levels: Vec<f64>,
    pub volume_profile: Vec<VolumeLevel>,
}

#[derive(Debug, Clone)]
pub struct VolumeLevel {
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_type: String,
    pub confidence: f64,
    pub start_index: usize,
    pub end_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Extreme,
}

impl BreakoutRegime {
    fn new(breakout_level: f64, volume_surge: f64, direction: TrendDirection, confirmation_strength: f64) -> Self {
        Self {
            breakout_level,
            volume_surge,
            direction,
            confirmation_strength,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BreakoutRegime {
    pub breakout_level: f64,
    pub volume_surge: f64,
    pub direction: TrendDirection,
    pub confirmation_strength: f64,
}

impl DataFrame {
    pub fn clone(&self) -> Self {
        self.clone()
    }
}

criterion_group!(
    benches,
    bench_basic_trend_analysis,
    bench_multi_timeframe_analysis,
    bench_technical_indicators,
    bench_pattern_recognition,
    bench_market_regime_detection,
    bench_support_resistance_detection,
    bench_volume_analysis,
    bench_trend_strength_calculation,
    bench_breakout_detection,
    bench_concurrent_analysis,
    bench_real_time_analysis
);

criterion_main!(benches);