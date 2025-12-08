use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ml_ensemble::{create_default_ensemble, EnsembleConfig, EnsemblePrediction};
use ats_core::types::MarketData;
use tokio::runtime::Runtime;

fn bench_ensemble_prediction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ensemble = rt.block_on(create_default_ensemble()).unwrap();
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    c.bench_function("ensemble_single_prediction", |b| {
        b.iter(|| {
            rt.block_on(async {
                let pred = ensemble.predict(black_box(&market_data)).await.unwrap();
                black_box(pred);
            })
        })
    });
}

fn bench_ensemble_batch_prediction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ensemble = rt.block_on(create_default_ensemble()).unwrap();
    
    let mut group = c.benchmark_group("ensemble_batch_prediction");
    
    for size in [10, 100, 1000].iter() {
        let market_data_batch: Vec<MarketData> = (0..*size)
            .map(|i| MarketData {
                timestamp: i as u64,
                bid: 100.0 + (i as f64 * 0.01),
                ask: 100.1 + (i as f64 * 0.01),
                bid_size: 1000.0,
                ask_size: 1000.0,
            })
            .collect();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    for data in &market_data_batch {
                        let pred = ensemble.predict(black_box(data)).await.unwrap();
                        black_box(pred);
                    }
                })
            })
        });
    }
    
    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    use ml_ensemble::features::FeatureEngineering;
    use ml_ensemble::FeatureConfig;
    
    let config = FeatureConfig::default();
    let feature_engine = FeatureEngineering::new(config);
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    c.bench_function("feature_extraction", |b| {
        b.iter(|| {
            let features = feature_engine.extract_features(black_box(&market_data)).unwrap();
            black_box(features);
        })
    });
}

fn bench_market_detection(c: &mut Criterion) {
    use ml_ensemble::market_detector::MarketConditionDetector;
    
    let detector = MarketConditionDetector::new();
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    c.bench_function("market_condition_detection", |b| {
        b.iter(|| {
            let condition = detector.detect_condition(black_box(&market_data)).unwrap();
            black_box(condition);
        })
    });
}

fn bench_weight_update(c: &mut Criterion) {
    use ml_ensemble::weights::WeightManager;
    use ml_ensemble::ModelWeightsConfig;
    
    let config = ModelWeightsConfig::default();
    let mut weight_manager = WeightManager::new(config);
    
    c.bench_function("weight_update", |b| {
        b.iter(|| {
            weight_manager.update_performance(black_box(0.001)).unwrap();
        })
    });
}

fn bench_confidence_calibration(c: &mut Criterion) {
    use ml_ensemble::calibration::ConfidenceCalibrator;
    use ml_ensemble::{CalibrationConfig, MarketCondition};
    
    let config = CalibrationConfig::default();
    let calibrator = ConfidenceCalibrator::new(config);
    
    c.bench_function("confidence_calibration", |b| {
        b.iter(|| {
            let calibrated = calibrator.calibrate_confidence(
                black_box(0.75),
                black_box(MarketCondition::Trending),
            ).unwrap();
            black_box(calibrated);
        })
    });
}

criterion_group!(
    benches,
    bench_ensemble_prediction,
    bench_ensemble_batch_prediction,
    bench_feature_extraction,
    bench_market_detection,
    bench_weight_update,
    bench_confidence_calibration
);
criterion_main!(benches);