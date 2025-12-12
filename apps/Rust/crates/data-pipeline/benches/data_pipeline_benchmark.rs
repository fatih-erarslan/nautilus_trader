//! Performance benchmarks for the data pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use data_pipeline::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark data pipeline processing
fn benchmark_pipeline_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pipeline_processing");
    
    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = DataPipelineConfig::default();
                    let pipeline = DataPipeline::new(config).await.unwrap();
                    
                    for i in 0..size {
                        let data = fusion::DataItem {
                            symbol: format!("TEST{}", i % 10),
                            timestamp: chrono::Utc::now(),
                            price: 100.0 + (i as f64 * 0.1),
                            volume: 1000.0 + (i as f64 * 10.0),
                            bid: Some(99.95 + (i as f64 * 0.1)),
                            ask: Some(100.05 + (i as f64 * 0.1)),
                            text: Some(format!("Test news item {}", i)),
                            raw_data: vec![],
                        };
                        
                        black_box(pipeline.process_data(data).await.unwrap());
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD indicator calculations
fn benchmark_simd_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_indicators");
    
    let config = Arc::new(IndicatorsConfig::default());
    let calculator = indicators::SIMDCalculator::new(config);
    
    for size in [100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sma_calculation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(calculator.sma_simd(data, 20).unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ema_calculation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(calculator.ema_simd(data, 20).unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rsi_calculation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(calculator.rsi_simd(data, 14).unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("macd_calculation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(calculator.macd_simd(data, 12, 26, 9).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark streaming data processing
fn benchmark_streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("streaming");
    
    group.bench_function("kafka_message_parsing", |b| {
        let json_data = r#"{"symbol":"BTCUSD","timestamp":"2023-01-01T00:00:00Z","open":100.0,"high":105.0,"low":95.0,"close":102.0,"volume":1000.0,"trades":100,"vwap":101.0,"bid":101.5,"ask":102.5,"bid_size":100.0,"ask_size":100.0}"#;
        
        b.iter(|| {
            let market_data: streaming::MarketData = black_box(
                serde_json::from_str(json_data).unwrap()
            );
            black_box(market_data);
        });
    });
    
    group.bench_function("stream_data_creation", |b| {
        b.iter(|| {
            let stream_data = streaming::StreamData {
                topic: "test-topic".to_string(),
                partition: 0,
                offset: 12345,
                timestamp: chrono::Utc::now(),
                key: Some("test-key".to_string()),
                payload: streaming::StreamPayload::MarketData(streaming::MarketData {
                    symbol: "BTCUSD".to_string(),
                    timestamp: chrono::Utc::now(),
                    open: 100.0,
                    high: 105.0,
                    low: 95.0,
                    close: 102.0,
                    volume: 1000.0,
                    trades: 100,
                    vwap: 101.0,
                    bid: 101.5,
                    ask: 102.5,
                    bid_size: 100.0,
                    ask_size: 100.0,
                }),
                headers: std::collections::HashMap::new(),
                metadata: streaming::StreamMetadata {
                    source: "test".to_string(),
                    version: "1.0".to_string(),
                    schema_version: "1.0".to_string(),
                    compression: None,
                    checksum: None,
                    processing_time: Some(Duration::from_millis(1)),
                    quality_score: Some(0.95),
                },
            };
            black_box(stream_data);
        });
    });
    
    group.finish();
}

/// Benchmark sentiment analysis
fn benchmark_sentiment_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("sentiment_analysis");
    
    let texts = vec![
        "This is great news for the market!",
        "The stock price is falling dramatically.",
        "Neutral market conditions continue.",
        "Excellent quarterly results exceeded expectations.",
        "Concerns about economic downturn persist.",
    ];
    
    group.bench_function("text_preprocessing", |b| {
        let config = Arc::new(SentimentConfig::default());
        let preprocessor = sentiment::TextPreprocessor::new(config);
        
        b.iter(|| {
            for text in &texts {
                black_box(preprocessor.preprocess(text).unwrap());
            }
        });
    });
    
    group.bench_function("sentiment_cache_operations", |b| {
        let mut cache = sentiment::SentimentCache::new(1000, Duration::from_secs(60));
        
        b.iter(|| {
            for (i, text) in texts.iter().enumerate() {
                let result = sentiment::SentimentResult {
                    text: text.to_string(),
                    sentiment: sentiment::SentimentLabel::Positive,
                    confidence: 0.8,
                    scores: sentiment::SentimentScores {
                        positive: 0.8,
                        negative: 0.1,
                        neutral: 0.1,
                        compound: 0.7,
                    },
                    emotions: None,
                    aspects: None,
                    language: None,
                    processing_time: Duration::from_millis(10),
                    model_used: "test".to_string(),
                    metadata: sentiment::SentimentMetadata {
                        text_length: text.len(),
                        token_count: 10,
                        preprocessing_applied: vec![],
                        model_version: "1.0".to_string(),
                        cache_hit: false,
                        batch_size: 1,
                        gpu_used: false,
                    },
                };
                
                cache.put(format!("key_{}", i), result.clone());
                black_box(cache.get(&format!("key_{}", i)));
            }
        });
    });
    
    group.finish();
}

/// Benchmark data fusion algorithms
fn benchmark_data_fusion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("data_fusion");
    
    group.bench_function("kalman_filter_operations", |b| {
        b.iter(|| {
            let mut filter = fusion::KalmanFilter::new(4, 2);
            
            for _ in 0..100 {
                black_box(filter.predict().unwrap());
                
                let measurement = nalgebra::DVector::from_vec(vec![100.0, 0.9]);
                black_box(filter.update(&measurement).unwrap());
            }
        });
    });
    
    group.bench_function("bayesian_fusion", |b| {
        let config = Arc::new(FusionConfig::default());
        let fusioner = fusion::BayesianFusioner::new(config);
        
        let sources = vec![
            fusion::DataSource {
                name: "source1".to_string(),
                value: 100.0,
                uncertainty: 0.1,
                timestamp: chrono::Utc::now(),
                quality: 0.9,
            },
            fusion::DataSource {
                name: "source2".to_string(),
                value: 102.0,
                uncertainty: 0.2,
                timestamp: chrono::Utc::now(),
                quality: 0.8,
            },
            fusion::DataSource {
                name: "source3".to_string(),
                value: 98.0,
                uncertainty: 0.15,
                timestamp: chrono::Utc::now(),
                quality: 0.85,
            },
        ];
        
        b.iter(|| {
            black_box(fusioner.fuse_data(&sources).unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark memory and performance utilities
fn benchmark_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities");
    
    group.bench_function("mathematical_operations", |b| {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        
        b.iter(|| {
            black_box(utils::math::moving_average(&data, 20));
            black_box(utils::math::standard_deviation(&data));
            black_box(utils::math::normalize(&data));
        });
    });
    
    group.bench_function("text_processing", |b| {
        let texts = vec![
            "  This is a sample text with\n\tmultiple    spaces  ",
            "Extract numbers: 123.45, -67.89, 1000",
            "Clean this messy\r\n\ttext with various\n\n\nwhitespace",
        ];
        
        b.iter(|| {
            for text in &texts {
                black_box(utils::string::clean_text(text));
                black_box(utils::string::extract_numbers(text));
                black_box(utils::string::truncate_text(text, 50));
            }
        });
    });
    
    group.bench_function("validation_operations", |b| {
        let prices = vec![100.0, -10.0, f64::NAN, f64::INFINITY, 0.0, 150.5];
        let volumes = vec![1000.0, -100.0, 0.0, f64::NAN, 5000.0];
        
        b.iter(|| {
            for &price in &prices {
                black_box(utils::validation::is_valid_price(price));
            }
            for &volume in &volumes {
                black_box(utils::validation::is_valid_volume(volume));
            }
        });
    });
    
    group.finish();
}

/// Benchmark pattern detection
fn benchmark_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");
    
    // Generate sample OHLC data
    let ohlc_data: Vec<indicators::OHLC> = (0..1000)
        .map(|i| {
            let base_price = 100.0 + (i as f64 * 0.01);
            indicators::OHLC {
                open: base_price,
                high: base_price + 2.0,
                low: base_price - 1.5,
                close: base_price + 0.5,
                volume: 1000.0 + (i as f64 * 10.0),
                timestamp: chrono::Utc::now(),
            }
        })
        .collect();
    
    group.bench_function("candlestick_pattern_detection", |b| {
        let config = Arc::new(IndicatorsConfig::default());
        let detector = indicators::PatternDetector::new(config);
        
        b.iter(|| {
            black_box(detector.detect_patterns(&ohlc_data));
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_pipeline_processing,
    benchmark_simd_indicators,
    benchmark_streaming,
    benchmark_sentiment_analysis,
    benchmark_data_fusion,
    benchmark_utilities,
    benchmark_pattern_detection
);

criterion_main!(benches);