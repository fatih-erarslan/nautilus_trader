//! Data processing benchmarks for the data collector
//!
//! This benchmark suite measures the performance of data collection, processing,
//! and storage operations under different conditions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use data_collector::{
    DataCollector, CollectorConfig, DataCollectorError, Result,
    types::{Kline, Trade, OrderBook, FundingRate, DataPoint, DataType},
    collectors::binance::BinanceCollector,
    storage::{StorageBackend, ParquetStorage, CsvStorage, SqliteStorage},
};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create sample kline data for benchmarking
fn create_sample_klines(count: usize) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(count);
    let start_time = Utc::now() - ChronoDuration::days(count as i64);
    
    for i in 0..count {
        let timestamp = start_time + ChronoDuration::minutes(i as i64);
        let price = 50000.0 + (i as f64 * 0.1).sin() * 1000.0;
        
        klines.push(Kline {
            timestamp,
            open: price - 10.0,
            high: price + 50.0,
            low: price - 50.0,
            close: price,
            volume: 1000.0 + (i as f64 * 0.02).cos() * 500.0,
            quote_volume: price * 1000.0,
            trades: 100 + (i % 50),
            taker_buy_base: 500.0,
            taker_buy_quote: price * 500.0,
        });
    }
    
    klines
}

/// Create sample trade data for benchmarking
fn create_sample_trades(count: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(count);
    let start_time = Utc::now() - ChronoDuration::hours(count as i64);
    
    for i in 0..count {
        let timestamp = start_time + ChronoDuration::seconds(i as i64);
        let price = 50000.0 + (i as f64 * 0.05).sin() * 500.0;
        
        trades.push(Trade {
            timestamp,
            price,
            quantity: 1.0 + (i as f64 * 0.01).cos(),
            is_buyer_maker: i % 2 == 0,
            trade_id: i as u64,
        });
    }
    
    trades
}

/// Create sample order book data for benchmarking
fn create_sample_order_book() -> OrderBook {
    let timestamp = Utc::now();
    let mut bids = Vec::new();
    let mut asks = Vec::new();
    
    for i in 0..100 {
        bids.push((49900.0 - i as f64, 1.0 + i as f64 * 0.1));
        asks.push((50100.0 + i as f64, 1.0 + i as f64 * 0.1));
    }
    
    OrderBook {
        timestamp,
        bids,
        asks,
    }
}

/// Create test configuration
fn create_test_config() -> CollectorConfig {
    CollectorConfig {
        output_dir: "/tmp/data_collector_bench".to_string(),
        storage_format: "parquet".to_string(),
        exchanges: vec!["binance".to_string()],
        symbols: vec!["BTCUSDT".to_string()],
        intervals: vec!["1m".to_string(), "5m".to_string(), "1h".to_string()],
        max_concurrent_requests: 10,
        request_delay_ms: 100,
        retry_attempts: 3,
        chunk_size: 1000,
        enable_validation: true,
        enable_compression: true,
        database_url: None,
    }
}

/// Benchmark data collection from different exchanges
fn bench_data_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("data_collection");
    
    let exchanges = vec!["binance", "coinbase", "kraken", "okx", "bybit"];
    
    for exchange in exchanges {
        group.bench_with_input(
            BenchmarkId::new("exchange", exchange),
            &exchange,
            |b, &exchange| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        (collector, exchange)
                    },
                    |(collector, exchange)| async move {
                        let result = collector.collect_historical_data(
                            exchange,
                            "BTCUSDT",
                            "1h",
                            1000,
                        ).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark kline data processing
fn bench_kline_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("kline_processing");
    
    let data_sizes = vec![100, 500, 1000, 5000, 10000];
    
    for size in data_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(size);
                        (collector, klines)
                    },
                    |(collector, klines)| async move {
                        let result = collector.process_klines(&klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark trade data processing
fn bench_trade_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("trade_processing");
    
    let data_sizes = vec![1000, 5000, 10000, 50000, 100000];
    
    for size in data_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let trades = create_sample_trades(size);
                        (collector, trades)
                    },
                    |(collector, trades)| async move {
                        let result = collector.process_trades(&trades).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark storage backends
fn bench_storage_backends(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("storage_backends");
    
    let backends = vec![
        ("parquet", "parquet"),
        ("csv", "csv"),
        ("sqlite", "sqlite"),
    ];
    
    for (backend_name, backend_type) in backends {
        group.bench_with_input(
            BenchmarkId::new("backend", backend_name),
            &backend_type,
            |b, &backend_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(1000);
                        (collector, klines, backend_type)
                    },
                    |(collector, klines, backend_type)| async move {
                        let result = collector.store_data(backend_type, &klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark data validation
fn bench_data_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("data_validation");
    
    let validation_types = vec![
        ("klines", "klines"),
        ("trades", "trades"),
        ("order_book", "order_book"),
        ("funding_rates", "funding_rates"),
    ];
    
    for (validation_name, validation_type) in validation_types {
        group.bench_with_input(
            BenchmarkId::new("validation", validation_name),
            &validation_type,
            |b, &validation_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(1000);
                        (collector, klines, validation_type)
                    },
                    |(collector, klines, validation_type)| async move {
                        let result = collector.validate_data(validation_type, &klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark data compression
fn bench_data_compression(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("data_compression");
    
    let compression_types = vec![
        ("gzip", "gzip"),
        ("zstd", "zstd"),
        ("snappy", "snappy"),
    ];
    
    for (compression_name, compression_type) in compression_types {
        group.bench_with_input(
            BenchmarkId::new("compression", compression_name),
            &compression_type,
            |b, &compression_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(10000);
                        (collector, klines, compression_type)
                    },
                    |(collector, klines, compression_type)| async move {
                        let result = collector.compress_data(compression_type, &klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent data collection
fn bench_concurrent_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_collection");
    
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrency", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let symbols: Vec<String> = (0..concurrency)
                            .map(|i| format!("SYMBOL{}USDT", i))
                            .collect();
                        (collector, symbols)
                    },
                    |(collector, symbols)| async move {
                        let mut handles = Vec::new();
                        
                        for symbol in symbols {
                            let collector_clone = collector.clone();
                            let handle = tokio::spawn(async move {
                                collector_clone.collect_historical_data(
                                    "binance",
                                    &symbol,
                                    "1h",
                                    1000,
                                ).await
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

/// Benchmark rate limiting
fn bench_rate_limiting(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("rate_limiting");
    
    let rate_limits = vec![10, 50, 100, 500, 1000]; // requests per minute
    
    for rate_limit in rate_limits {
        group.bench_with_input(
            BenchmarkId::new("rate_limit", rate_limit),
            &rate_limit,
            |b, &rate_limit| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let mut config = create_test_config();
                        config.max_concurrent_requests = rate_limit;
                        let collector = DataCollector::new(config);
                        (collector, rate_limit)
                    },
                    |(collector, rate_limit)| async move {
                        let result = collector.rate_limited_collection(rate_limit).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark data aggregation
fn bench_data_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("data_aggregation");
    
    let aggregation_types = vec![
        ("ohlcv", "ohlcv"),
        ("volume_profile", "volume_profile"),
        ("time_weighted", "time_weighted"),
        ("trade_buckets", "trade_buckets"),
    ];
    
    for (aggregation_name, aggregation_type) in aggregation_types {
        group.bench_with_input(
            BenchmarkId::new("aggregation", aggregation_name),
            &aggregation_type,
            |b, &aggregation_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let trades = create_sample_trades(10000);
                        (collector, trades, aggregation_type)
                    },
                    |(collector, trades, aggregation_type)| async move {
                        let result = collector.aggregate_data(aggregation_type, &trades).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark data quality checks
fn bench_data_quality(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("data_quality");
    
    let quality_checks = vec![
        ("duplicate_detection", "duplicates"),
        ("gap_detection", "gaps"),
        ("outlier_detection", "outliers"),
        ("consistency_check", "consistency"),
    ];
    
    for (check_name, check_type) in quality_checks {
        group.bench_with_input(
            BenchmarkId::new("quality_check", check_name),
            &check_type,
            |b, &check_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(5000);
                        (collector, klines, check_type)
                    },
                    |(collector, klines, check_type)| async move {
                        let result = collector.quality_check(check_type, &klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage during processing
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_usage");
    
    let memory_scenarios = vec![
        ("small_batch", 1000),
        ("medium_batch", 10000),
        ("large_batch", 100000),
        ("xl_batch", 1000000),
    ];
    
    for (scenario_name, batch_size) in memory_scenarios {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_scenario", scenario_name),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let collector = DataCollector::new(config);
                        let klines = create_sample_klines(batch_size);
                        (collector, klines)
                    },
                    |(collector, klines)| async move {
                        let result = collector.memory_efficient_processing(&klines).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

// Mock implementations for missing structures and methods

#[derive(Clone)]
pub struct DataCollector {
    config: CollectorConfig,
}

impl DataCollector {
    pub fn new(config: CollectorConfig) -> Self {
        Self { config }
    }
    
    pub async fn collect_historical_data(
        &self,
        exchange: &str,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        // Mock implementation
        Ok(create_sample_klines(limit))
    }
    
    pub async fn process_klines(&self, klines: &[Kline]) -> Result<Vec<DataPoint>> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn process_trades(&self, trades: &[Trade]) -> Result<Vec<DataPoint>> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn store_data(&self, backend_type: &str, klines: &[Kline]) -> Result<()> {
        // Mock implementation
        Ok(())
    }
    
    pub async fn validate_data(&self, validation_type: &str, klines: &[Kline]) -> Result<bool> {
        // Mock implementation
        Ok(true)
    }
    
    pub async fn compress_data(&self, compression_type: &str, klines: &[Kline]) -> Result<Vec<u8>> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub async fn rate_limited_collection(&self, rate_limit: usize) -> Result<Vec<Kline>> {
        // Mock implementation
        Ok(create_sample_klines(rate_limit))
    }
    
    pub async fn aggregate_data(&self, aggregation_type: &str, trades: &[Trade]) -> Result<Vec<Kline>> {
        // Mock implementation
        Ok(create_sample_klines(100))
    }
    
    pub async fn quality_check(&self, check_type: &str, klines: &[Kline]) -> Result<bool> {
        // Mock implementation
        Ok(true)
    }
    
    pub async fn memory_efficient_processing(&self, klines: &[Kline]) -> Result<Vec<DataPoint>> {
        // Mock implementation
        Ok(vec![])
    }
    
    pub fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}

// Mock types module
pub mod types {
    use super::*;
    
    #[derive(Debug, Clone)]
    pub struct Kline {
        pub timestamp: DateTime<Utc>,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
        pub quote_volume: f64,
        pub trades: u64,
        pub taker_buy_base: f64,
        pub taker_buy_quote: f64,
    }
    
    #[derive(Debug, Clone)]
    pub struct Trade {
        pub timestamp: DateTime<Utc>,
        pub price: f64,
        pub quantity: f64,
        pub is_buyer_maker: bool,
        pub trade_id: u64,
    }
    
    #[derive(Debug, Clone)]
    pub struct OrderBook {
        pub timestamp: DateTime<Utc>,
        pub bids: Vec<(f64, f64)>,
        pub asks: Vec<(f64, f64)>,
    }
    
    #[derive(Debug, Clone)]
    pub struct FundingRate {
        pub timestamp: DateTime<Utc>,
        pub rate: f64,
    }
    
    #[derive(Debug, Clone)]
    pub struct DataPoint {
        pub timestamp: DateTime<Utc>,
        pub value: f64,
    }
    
    #[derive(Debug, Clone)]
    pub enum DataType {
        Kline,
        Trade,
        OrderBook,
        FundingRate,
    }
}

// Mock collectors module
pub mod collectors {
    use super::*;
    
    pub struct BinanceCollector;
    
    impl BinanceCollector {
        pub fn new() -> Self {
            Self
        }
    }
}

// Mock storage module
pub mod storage {
    use super::*;
    
    pub trait StorageBackend {
        fn store_data(&self, data: &[Kline]) -> Result<()>;
    }
    
    pub struct ParquetStorage;
    pub struct CsvStorage;
    pub struct SqliteStorage;
    
    impl StorageBackend for ParquetStorage {
        fn store_data(&self, _data: &[Kline]) -> Result<()> {
            Ok(())
        }
    }
    
    impl StorageBackend for CsvStorage {
        fn store_data(&self, _data: &[Kline]) -> Result<()> {
            Ok(())
        }
    }
    
    impl StorageBackend for SqliteStorage {
        fn store_data(&self, _data: &[Kline]) -> Result<()> {
            Ok(())
        }
    }
}

// Mock config module
pub mod config {
    #[derive(Debug, Clone)]
    pub struct CollectorConfig {
        pub output_dir: String,
        pub storage_format: String,
        pub exchanges: Vec<String>,
        pub symbols: Vec<String>,
        pub intervals: Vec<String>,
        pub max_concurrent_requests: usize,
        pub request_delay_ms: u64,
        pub retry_attempts: usize,
        pub chunk_size: usize,
        pub enable_validation: bool,
        pub enable_compression: bool,
        pub database_url: Option<String>,
    }
}

// Mock rate_limiter module
pub mod rate_limiter {
    use super::*;
    
    pub async fn init() -> Result<()> {
        Ok(())
    }
}

use types::*;

criterion_group!(
    benches,
    bench_data_collection,
    bench_kline_processing,
    bench_trade_processing,
    bench_storage_backends,
    bench_data_validation,
    bench_data_compression,
    bench_concurrent_collection,
    bench_rate_limiting,
    bench_data_aggregation,
    bench_data_quality,
    bench_memory_usage
);

criterion_main!(benches);