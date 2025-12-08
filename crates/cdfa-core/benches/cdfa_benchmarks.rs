//! Performance benchmarks for cdfa-core
//!
//! Run with: cargo bench --package cdfa-core

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use cdfa_core::prelude::*;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn generate_random_array(size: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array1::from_shape_fn(size, |_| rng.gen::<f64>())
}

fn generate_random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>())
}

fn bench_diversity_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("diversity_metrics");
    
    for size in [10, 100, 1000, 10000].iter() {
        let x = generate_random_array(*size, 42);
        let y = generate_random_array(*size, 43);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("kendall_tau", size), size, |b, _| {
            b.iter(|| {
                kendall_tau(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("kendall_tau_fast", size), size, |b, _| {
            b.iter(|| {
                kendall_tau_fast(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("spearman_correlation", size), size, |b, _| {
            b.iter(|| {
                spearman_correlation(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("spearman_correlation_fast", size), size, |b, _| {
            b.iter(|| {
                spearman_correlation_fast(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("pearson_correlation", size), size, |b, _| {
            b.iter(|| {
                pearson_correlation(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("pearson_correlation_fast", size), size, |b, _| {
            b.iter(|| {
                pearson_correlation_fast(black_box(&x), black_box(&y)).unwrap()
            });
        });
        
        if *size <= 1000 {  // DTW is O(n²) so limit to smaller sizes
            group.bench_with_input(BenchmarkId::new("dynamic_time_warping", size), size, |b, _| {
                b.iter(|| {
                    dynamic_time_warping(black_box(&x), black_box(&y)).unwrap()
                });
            });
        }
    }
    
    group.finish();
}

fn bench_jensen_shannon(c: &mut Criterion) {
    let mut group = c.benchmark_group("jensen_shannon");
    
    for size in [10, 100, 1000, 10000].iter() {
        // Generate normalized probability distributions
        let p = {
            let raw = generate_random_array(*size, 44);
            &raw / raw.sum()
        };
        let q = {
            let raw = generate_random_array(*size, 45);
            &raw / raw.sum()
        };
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("divergence", size), size, |b, _| {
            b.iter(|| {
                jensen_shannon_divergence(black_box(&p), black_box(&q)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("distance", size), size, |b, _| {
            b.iter(|| {
                jensen_shannon_distance(black_box(&p), black_box(&q)).unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");
    
    for &(n_vars, n_samples) in [(10, 100), (50, 500), (100, 1000)].iter() {
        let data = generate_random_matrix(n_vars, n_samples, 46);
        
        group.throughput(Throughput::Elements((n_vars * n_vars) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("pearson_matrix", format!("{}x{}", n_vars, n_samples)), 
            &(n_vars, n_samples), 
            |b, _| {
                b.iter(|| {
                    pearson_correlation_matrix(black_box(&data.view())).unwrap()
                });
            }
        );
    }
    
    group.finish();
}

fn bench_fusion_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_methods");
    
    for &(n_sources, n_items) in [(3, 10), (5, 100), (10, 1000), (20, 100)].iter() {
        let scores = generate_random_matrix(n_sources, n_items, 47);
        
        group.throughput(Throughput::Elements((n_sources * n_items) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("average", format!("{}x{}", n_sources, n_items)), 
            &(n_sources, n_items), 
            |b, _| {
                b.iter(|| {
                    CdfaFusion::fuse(black_box(&scores.view()), FusionMethod::Average, None).unwrap()
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("weighted_average", format!("{}x{}", n_sources, n_items)), 
            &(n_sources, n_items), 
            |b, _| {
                let weights = Array1::ones(n_sources) / n_sources as f64;
                let params = FusionParams {
                    weights: Some(weights),
                    ..Default::default()
                };
                b.iter(|| {
                    CdfaFusion::fuse(black_box(&scores.view()), FusionMethod::WeightedAverage, Some(params.clone())).unwrap()
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("borda_count", format!("{}x{}", n_sources, n_items)), 
            &(n_sources, n_items), 
            |b, _| {
                b.iter(|| {
                    CdfaFusion::fuse(black_box(&scores.view()), FusionMethod::BordaCount, None).unwrap()
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("adaptive", format!("{}x{}", n_sources, n_items)), 
            &(n_sources, n_items), 
            |b, _| {
                let params = FusionParams {
                    diversity_threshold: 0.5,
                    score_weight: 0.7,
                    ..Default::default()
                };
                b.iter(|| {
                    CdfaFusion::fuse(black_box(&scores.view()), FusionMethod::Adaptive, Some(params.clone())).unwrap()
                });
            }
        );
    }
    
    group.finish();
}

fn bench_complete_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_pipeline");
    
    for &(n_sources, signal_len) in [(4, 100), (8, 500), (16, 1000)].iter() {
        let signals = generate_random_matrix(n_sources, signal_len, 48);
        
        group.throughput(Throughput::Elements((n_sources * signal_len) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("full_cdfa", format!("{}x{}", n_sources, signal_len)), 
            &(n_sources, signal_len), 
            |b, _| {
                b.iter(|| {
                    // Calculate diversity matrix
                    let mut diversity_matrix = Array2::<f64>::zeros((n_sources, n_sources));
                    for i in 0..n_sources {
                        for j in i+1..n_sources {
                            let corr = pearson_correlation_fast(&signals.row(i), &signals.row(j)).unwrap();
                            let div = 1.0 - corr.abs();
                            diversity_matrix[[i, j]] = div;
                            diversity_matrix[[j, i]] = div;
                        }
                    }
                    
                    // Calculate weights
                    let avg_diversity = diversity_matrix.mean_axis(ndarray::Axis(1)).unwrap();
                    let weights = &avg_diversity / avg_diversity.sum();
                    
                    // Perform fusion
                    let params = FusionParams {
                        weights: Some(weights),
                        diversity_threshold: 0.5,
                        score_weight: 0.7,
                    };
                    
                    CdfaFusion::fuse(black_box(&signals.view()), FusionMethod::Adaptive, Some(params)).unwrap()
                });
            }
        );
    }
    
    group.finish();
}

fn bench_streaming_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");
    
    let window_size = 20;
    let n_sources = 5;
    let window_signals = generate_random_matrix(n_sources, window_size, 49);
    
    group.throughput(Throughput::Elements((n_sources * window_size) as u64));
    
    group.bench_function("window_fusion", |b| {
        b.iter(|| {
            // Quick diversity assessment
            let mut div_sum = 0.0;
            for i in 0..n_sources {
                for j in i+1..n_sources {
                    let corr = pearson_correlation_fast(&window_signals.row(i), &window_signals.row(j)).unwrap_or(0.0);
                    div_sum += 1.0 - corr.abs();
                }
            }
            let avg_diversity = div_sum / ((n_sources * (n_sources - 1)) as f64 / 2.0);
            
            // Adaptive method selection
            let method = if avg_diversity > 0.5 {
                FusionMethod::Average
            } else {
                FusionMethod::BordaCount
            };
            
            CdfaFusion::fuse(black_box(&window_signals.view()), method, None).unwrap()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_diversity_metrics,
    bench_jensen_shannon,
    bench_correlation_matrix,
    bench_fusion_methods,
    bench_complete_pipeline,
    bench_streaming_scenario
);
criterion_main!(benches);