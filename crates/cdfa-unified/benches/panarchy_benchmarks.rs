//! Benchmarks for Panarchy analyzer performance validation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cdfa_unified::analyzers::panarchy::*;
use cdfa_unified::traits::SystemAnalyzer;
use ndarray::Array2;
use std::time::Duration;

fn generate_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = (0..size)
        .map(|i| 100.0 + 10.0 * (i as f64 / 20.0).sin() + 2.0 * (i as f64 / 5.0).cos())
        .collect();
    let volumes = vec![1000.0; size];
    (prices, volumes)
}

fn bench_full_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_full_analysis");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different data sizes
    for size in [50, 100, 200, 500].iter() {
        let (prices, volumes) = generate_test_data(*size);
        let mut analyzer = PanarchyAnalyzer::new();
        
        group.bench_with_input(
            BenchmarkId::new("full_analysis", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(analyzer.analyze_full(&prices, &volumes).unwrap())
                })
            },
        );
    }
    group.finish();
}

fn bench_pcr_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_pcr_calculation");
    group.measurement_time(Duration::from_secs(5));
    
    let analyzer = PanarchyAnalyzer::new();
    let (prices, volumes) = generate_test_data(100);
    let returns = analyzer.calculate_returns(&prices).unwrap();
    
    group.bench_function("pcr_scalar", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_pcr_scalar(&prices, &returns, &volumes).unwrap())
        })
    });
    
    #[cfg(feature = "simd")]
    group.bench_function("pcr_simd", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_pcr_simd(&prices, &returns, &volumes).unwrap())
        })
    });
    
    #[cfg(feature = "gpu")]
    group.bench_function("pcr_gpu", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_pcr_gpu(&prices, &returns, &volumes).unwrap())
        })
    });
    
    group.finish();
}

fn bench_phase_identification(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_phase_identification");
    
    let analyzer = PanarchyAnalyzer::new();
    let (prices, volumes) = generate_test_data(50);
    let returns = analyzer.calculate_returns(&prices).unwrap();
    let pcr = analyzer.calculate_pcr_scalar(&prices, &returns, &volumes).unwrap();
    
    group.bench_function("identify_phase", |b| {
        b.iter(|| {
            black_box(analyzer.identify_phase_fast(&pcr, &returns).unwrap())
        })
    });
    
    group.finish();
}

fn bench_system_analyzer_trait(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_system_analyzer");
    
    let analyzer = PanarchyAnalyzer::new();
    let data = Array2::from_shape_vec((100, 2), 
        (0..200).map(|i| {
            if i % 2 == 0 { 
                100.0 + (i as f64 / 2.0 * 0.1).sin() 
            } else { 
                1000.0 + (i as f64 * 10.0) 
            }
        }).collect()
    ).unwrap();
    let scores = ndarray::Array1::zeros(100);
    
    group.bench_function("system_analyzer_analyze", |b| {
        b.iter(|| {
            black_box(analyzer.analyze(&data.view(), &scores.view()).unwrap())
        })
    });
    
    group.finish();
}

fn bench_mathematical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_math_operations");
    
    let analyzer = PanarchyAnalyzer::new();
    let (prices, _) = generate_test_data(100);
    let returns = analyzer.calculate_returns(&prices).unwrap();
    
    group.bench_function("calculate_returns", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_returns(&prices).unwrap())
        })
    });
    
    group.bench_function("calculate_autocorrelation", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_autocorrelation(&returns, 1).unwrap())
        })
    });
    
    group.bench_function("calculate_volatility", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_volatility(&returns).unwrap())
        })
    });
    
    group.finish();
}

fn bench_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_performance_targets");
    group.significance_level(0.05);
    group.sample_size(1000);
    
    // Target validation benchmarks
    let mut analyzer = PanarchyAnalyzer::new();
    let (prices, volumes) = generate_test_data(50);
    let returns = analyzer.calculate_returns(&prices).unwrap();
    
    // PCR calculation target: <300ns
    group.bench_function("pcr_target_300ns", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let pcr = black_box(analyzer.calculate_pcr_scalar(&prices, &returns, &volumes).unwrap());
            let elapsed = start.elapsed().as_nanos();
            
            // Log if we exceed target significantly
            if elapsed > performance_targets::PCR_CALCULATION_TARGET_NS as u128 * 10 {
                eprintln!("PCR calculation took {}ns (target: {}ns)", 
                         elapsed, performance_targets::PCR_CALCULATION_TARGET_NS);
            }
            
            pcr
        })
    });
    
    // Phase identification target: <200ns
    let pcr = analyzer.calculate_pcr_scalar(&prices, &returns, &volumes).unwrap();
    group.bench_function("phase_target_200ns", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = black_box(analyzer.identify_phase_fast(&pcr, &returns).unwrap());
            let elapsed = start.elapsed().as_nanos();
            
            if elapsed > performance_targets::PHASE_CLASSIFICATION_TARGET_NS as u128 * 10 {
                eprintln!("Phase identification took {}ns (target: {}ns)", 
                         elapsed, performance_targets::PHASE_CLASSIFICATION_TARGET_NS);
            }
            
            result
        })
    });
    
    // Full analysis target: <800ns
    group.bench_function("full_analysis_target_800ns", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = black_box(analyzer.analyze_full(&prices, &volumes).unwrap());
            let elapsed = start.elapsed().as_nanos();
            
            if elapsed > performance_targets::FULL_ANALYSIS_TARGET_NS as u128 * 10 {
                eprintln!("Full analysis took {}ns (target: {}ns)", 
                         elapsed, performance_targets::FULL_ANALYSIS_TARGET_NS);
            }
            
            result
        })
    });
    
    group.finish();
}

fn bench_configuration_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_configuration_impact");
    
    let (prices, volumes) = generate_test_data(100);
    
    // Default configuration
    let mut default_analyzer = PanarchyAnalyzer::new();
    group.bench_function("config_default", |b| {
        b.iter(|| {
            black_box(default_analyzer.analyze_full(&prices, &volumes).unwrap())
        })
    });
    
    // Large window configuration
    let mut large_config = PanarchyConfig::default();
    large_config.window_size = 50;
    let mut large_analyzer = PanarchyAnalyzer::with_config(large_config);
    group.bench_function("config_large_window", |b| {
        b.iter(|| {
            black_box(large_analyzer.analyze_full(&prices, &volumes).unwrap())
        })
    });
    
    // High precision configuration
    let mut precision_config = PanarchyConfig::default();
    precision_config.min_confidence = 0.9;
    precision_config.hysteresis_threshold = 0.05;
    let mut precision_analyzer = PanarchyAnalyzer::with_config(precision_config);
    group.bench_function("config_high_precision", |b| {
        b.iter(|| {
            black_box(precision_analyzer.analyze_full(&prices, &volumes).unwrap())
        })
    });
    
    group.finish();
}

fn bench_data_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_data_scaling");
    group.sample_size(100);
    
    for size in [25, 50, 100, 200, 500, 1000].iter() {
        let (prices, volumes) = generate_test_data(*size);
        let mut analyzer = PanarchyAnalyzer::new();
        
        group.bench_with_input(
            BenchmarkId::new("scaling", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(analyzer.analyze_full(&prices, &volumes).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_parallel");
    
    let mut parallel_config = PanarchyConfig::default();
    parallel_config.enable_parallel = true;
    let parallel_analyzer = PanarchyAnalyzer::with_config(parallel_config);
    
    let mut sequential_config = PanarchyConfig::default();
    sequential_config.enable_parallel = false;
    let sequential_analyzer = PanarchyAnalyzer::with_config(sequential_config);
    
    let (prices, volumes) = generate_test_data(1000);
    
    group.bench_function("parallel_enabled", |b| {
        b.iter(|| {
            black_box(parallel_analyzer.analyze(&Array2::from_shape_vec((500, 2), 
                prices.iter().chain(volumes.iter()).copied().collect()).unwrap().view(),
                &ndarray::Array1::zeros(500).view()).unwrap())
        })
    });
    
    group.bench_function("sequential", |b| {
        b.iter(|| {
            black_box(sequential_analyzer.analyze(&Array2::from_shape_vec((500, 2), 
                prices.iter().chain(volumes.iter()).copied().collect()).unwrap().view(),
                &ndarray::Array1::zeros(500).view()).unwrap())
        })
    });
    
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_simd_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("panarchy_simd");
    
    let mut simd_config = PanarchyConfig::default();
    simd_config.enable_simd = true;
    let simd_analyzer = PanarchyAnalyzer::with_config(simd_config);
    
    let mut scalar_config = PanarchyConfig::default();
    scalar_config.enable_simd = false;
    let scalar_analyzer = PanarchyAnalyzer::with_config(scalar_config);
    
    let (prices, volumes) = generate_test_data(100);
    let returns = scalar_analyzer.calculate_returns(&prices).unwrap();
    
    group.bench_function("simd_pcr", |b| {
        b.iter(|| {
            black_box(simd_analyzer.calculate_pcr_simd(&prices, &returns, &volumes).unwrap())
        })
    });
    
    group.bench_function("scalar_pcr", |b| {
        b.iter(|| {
            black_box(scalar_analyzer.calculate_pcr_scalar(&prices, &returns, &volumes).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_full_analysis,
    bench_pcr_calculation,
    bench_phase_identification,
    bench_system_analyzer_trait,
    bench_mathematical_operations,
    bench_performance_targets,
    bench_configuration_impact,
    bench_data_size_scaling,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_processing);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_simd_optimization);

criterion_main!(
    benches,
    #[cfg(feature = "parallel")]
    parallel_benches,
    #[cfg(feature = "simd")]
    simd_benches,
);