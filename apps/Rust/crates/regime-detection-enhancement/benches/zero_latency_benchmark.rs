use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use regime_detection_enhancement::*;
use std::time::Duration;

fn benchmark_simd_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Feature Extraction");
    
    for size in [16, 64, 256, 1024, 4096].iter() {
        let market_data: Vec<f64> = (0..*size).map(|i| i as f64 * 0.001).collect();
        let config = simd_optimizer::SIMDOptimizerConfig::default();
        let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("vectorized_extraction", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.extract_features_vectorized(black_box(&market_data)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_hardware_acceleration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Hardware Acceleration");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 256, 1024].iter() {
        let features: Vec<f64> = (0..*size).map(|i| i as f64 * 0.001).collect();
        
        let config = hardware_accelerator::HardwareAcceleratorConfig {
            enable_gpu: false,
            enable_fpga: false,
            enable_neural: true,
            prefer_gpu: false,
            prefer_neural: true,
            gpu_config: hardware_accelerator::GPUConfig {
                device_id: 0,
                memory_pool_size: 1024,
                compute_streams: 4,
            },
            fpga_config: hardware_accelerator::FPGAConfig {
                fpga_id: 0,
                pipeline_depth: 8,
                clock_frequency: 200_000_000,
            },
            neural_config: hardware_accelerator::NeuralConfig {
                npu_count: 4,
                model_cache_size: 1024 * 1024,
                precision: "FP16".to_string(),
            },
        };
        
        let accelerator = rt.block_on(async {
            hardware_accelerator::HardwareAccelerator::new(config).await.unwrap()
        });
        
        group.bench_with_input(
            BenchmarkId::new("hardware_classification", size),
            size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    accelerator.classify_regime_accelerated(black_box(&features)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_zero_latency_detection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("Zero Latency Detection");
    group.measurement_time(Duration::from_secs(15));
    
    // Create base detector (simplified for benchmarking)
    let base_config = trading_strategies::agents::market_regime_detection::MarketRegimeDetectionConfig::default();
    let base_detector = rt.block_on(async {
        trading_strategies::agents::market_regime_detection::MarketRegimeDetectionAgent::new(base_config).await.unwrap()
    });
    
    let zero_latency_config = ZeroLatencyConfig {
        simd_config: simd_optimizer::SIMDOptimizerConfig::default(),
        hw_config: hardware_accelerator::HardwareAcceleratorConfig {
            enable_gpu: false,
            enable_fpga: false,
            enable_neural: true,
            prefer_gpu: false,
            prefer_neural: true,
            gpu_config: hardware_accelerator::GPUConfig {
                device_id: 0,
                memory_pool_size: 1024,
                compute_streams: 4,
            },
            fpga_config: hardware_accelerator::FPGAConfig {
                fpga_id: 0,
                pipeline_depth: 8,
                clock_frequency: 200_000_000,
            },
            neural_config: hardware_accelerator::NeuralConfig {
                npu_count: 4,
                model_cache_size: 1024 * 1024,
                precision: "FP16".to_string(),
            },
        },
        cache_size: 1000,
        fallback_threshold: 0.7,
    };
    
    let detector = rt.block_on(async {
        ZeroLatencyRegimeDetector::new(base_detector, zero_latency_config).await.unwrap()
    });
    
    let market_data = trading_strategies::types::MarketData {
        symbol: "BTCUSDT".to_string(),
        timestamp: chrono::Utc::now(),
        open: 50000.0,
        high: 50100.0,
        low: 49900.0,
        close: 50050.0,
        volume: 1000.0,
        market_regime: trading_strategies::types::MarketRegime::Trending,
    };
    
    group.bench_function("full_detection_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            detector.detect_regime_zero_latency(black_box(&market_data)).await.unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_simd_optimization,
    benchmark_hardware_acceleration,
    benchmark_zero_latency_detection
);
criterion_main!(benches);