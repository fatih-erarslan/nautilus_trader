use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cdfa_soc_analyzer::{SOCAnalyzer, SOCParameters};
use ndarray::Array1;
use std::time::Duration;

fn generate_test_data(size: usize, pattern: &str) -> Array1<f64> {
    match pattern {
        "random" => {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            Array1::from_vec((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect())
        }
        "sine" => {
            Array1::from_vec((0..size).map(|x| (x as f64 * 0.1).sin()).collect())
        }
        "noise" => {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            Array1::from_vec((0..size).map(|_| rng.gen::<f64>() * 0.01).collect())
        }
        "trend" => {
            Array1::from_vec((0..size).map(|x| x as f64 * 0.01 + (x as f64 * 0.05).sin()).collect())
        }
        _ => Array1::from_vec((0..size).map(|x| x as f64).collect()),
    }
}

fn benchmark_soc_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("soc_analysis");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 500, 1000, 2000, 5000];
    let patterns = ["random", "sine", "noise", "trend"];
    
    for &size in &sizes {
        for &pattern in &patterns {
            let data = generate_test_data(size, pattern);
            let analyzer = SOCAnalyzer::default();
            
            group.bench_with_input(
                BenchmarkId::new(format!("full_analysis_{}", pattern), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = analyzer.analyze(black_box(&data));
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn benchmark_sample_entropy(c: &mut Criterion) {
    use cdfa_soc_analyzer::sample_entropy;
    
    let mut group = c.benchmark_group("sample_entropy");
    group.measurement_time(Duration::from_secs(5));
    
    let sizes = [100, 500, 1000, 2000];
    
    for &size in &sizes {
        let data = generate_test_data(size, "random");
        
        group.bench_with_input(
            BenchmarkId::new("sample_entropy", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = sample_entropy(black_box(data.view()), 2, 0.2);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_entropy_rate(c: &mut Criterion) {
    use cdfa_soc_analyzer::entropy_rate;
    
    let mut group = c.benchmark_group("entropy_rate");
    group.measurement_time(Duration::from_secs(5));
    
    let sizes = [100, 500, 1000, 2000];
    
    for &size in &sizes {
        let data = generate_test_data(size, "trend");
        
        group.bench_with_input(
            BenchmarkId::new("entropy_rate", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = entropy_rate(black_box(data.view()), 1, 10);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_regime_classification(c: &mut Criterion) {
    use cdfa_soc_analyzer::classify_regime;
    
    let mut group = c.benchmark_group("regime_classification");
    group.measurement_time(Duration::from_secs(3));
    
    let params = SOCParameters::default();
    let test_cases = [
        ("critical", 0.8, 0.2, 0.7, 0.5),
        ("stable", 0.3, 0.8, 0.2, 0.4),
        ("unstable", 0.5, 0.2, 0.8, 0.9),
    ];
    
    for (name, complexity, equilibrium, fragility, entropy) in &test_cases {
        group.bench_function(*name, |b| {
            b.iter(|| {
                let result = classify_regime(
                    black_box(*complexity),
                    black_box(*equilibrium),
                    black_box(*fragility),
                    black_box(*entropy),
                    black_box(&params),
                );
                black_box(result)
            })
        });
    }
    group.finish();
}

fn benchmark_simd_operations(c: &mut Criterion) {
    use cdfa_soc_analyzer::simd::{simd_mean, simd_variance, simd_distance_array};
    
    let mut group = c.benchmark_group("simd_operations");
    group.measurement_time(Duration::from_secs(3));
    
    let sizes = [128, 512, 1024, 4096];
    
    for &size in &sizes {
        let data_f32: Vec<f32> = (0..size).map(|x| x as f32 * 0.01).collect();
        let data2_f32: Vec<f32> = (0..size).map(|x| (x as f32 * 0.01).sin()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("simd_mean", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = simd_mean(black_box(&data_f32));
                    black_box(result)
                })
            },
        );
        
        let mean = simd_mean(&data_f32);
        group.bench_with_input(
            BenchmarkId::new("simd_variance", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = simd_variance(black_box(&data_f32), mean);
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("simd_distance", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = simd_distance_array(black_box(&data_f32), black_box(&data2_f32));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_performance_targets(c: &mut Criterion) {
    use cdfa_soc_analyzer::perf::*;
    
    let mut group = c.benchmark_group("performance_targets");
    group.measurement_time(Duration::from_secs(5));
    
    // Test with data size that should meet sub-microsecond targets
    let data = generate_test_data(200, "random");
    let analyzer = SOCAnalyzer::default();
    
    group.bench_function("sub_microsecond_target", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = analyzer.analyze(black_box(&data));
            let elapsed = start.elapsed().as_nanos() as u64;
            
            // Verify we meet our performance target
            if elapsed > FULL_ANALYSIS_TARGET_NS {
                eprintln!("Performance target missed: {}ns > {}ns", elapsed, FULL_ANALYSIS_TARGET_NS);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_soc_analysis,
    benchmark_sample_entropy,
    benchmark_entropy_rate,
    benchmark_regime_classification,
    benchmark_simd_operations,
    benchmark_performance_targets
);

criterion_main!(benches);