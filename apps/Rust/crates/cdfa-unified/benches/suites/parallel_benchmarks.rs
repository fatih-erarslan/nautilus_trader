use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use cdfa_unified::{
    types::{CdfaArray, CdfaMatrix, CdfaFloat},
    core::diversity::{PearsonDiversityMeasure, KendallDiversityMeasure},
    algorithms::{
        volatility::VolatilityEstimator,
        entropy::EntropyCalculator,
        statistics::StatisticsCalculator,
    },
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const PARALLEL_TARGET_SPEEDUP: f64 = 1.5; // Minimum 1.5x speedup expected
const CPU_COUNT: usize = num_cpus::get();

fn generate_large_dataset(rows: usize, cols: usize) -> CdfaMatrix {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((i as f64 * 0.1) + (j as f64 * 0.01)) % 100.0
    })
}

fn generate_time_series(size: usize) -> CdfaArray {
    Array1::from_shape_fn(size, |i| {
        (i as f64 * 0.001).sin() + ((i as f64 * 0.01).cos() * 0.5)
    })
}

fn sequential_matrix_row_sum(matrix: &CdfaMatrix) -> Vec<CdfaFloat> {
    matrix.axis_iter(Axis(0))
        .map(|row| row.sum())
        .collect()
}

fn parallel_matrix_row_sum(matrix: &CdfaMatrix) -> Vec<CdfaFloat> {
    matrix.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.sum())
        .collect()
}

fn bench_parallel_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/matrix_operations");
    
    for size in [1000, 5000, 10000].iter() {
        let matrix = generate_large_dataset(*size, 100);
        
        group.throughput(Throughput::Elements((*size * 100) as u64));
        
        // Sequential baseline
        let sequential_time = {
            let start = Instant::now();
            for _ in 0..10 {
                black_box(sequential_matrix_row_sum(&matrix));
            }
            start.elapsed().as_nanos() / 10
        };
        
        group.bench_with_input(
            BenchmarkId::new("sequential_row_sum", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(sequential_matrix_row_sum(black_box(&matrix)))
                })
            },
        );
        
        // Parallel implementation
        group.bench_with_input(
            BenchmarkId::new("parallel_row_sum", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = parallel_matrix_row_sum(black_box(&matrix));
                    let parallel_time = start.elapsed().as_nanos();
                    
                    // Validate parallel speedup for larger datasets
                    if *size >= 5000 && CPU_COUNT > 1 {
                        let speedup = sequential_time as f64 / parallel_time as f64;
                        if speedup > 0.1 { // Only check if measurements are meaningful
                            assert!(
                                speedup >= PARALLEL_TARGET_SPEEDUP,
                                "Parallel speedup {} < target {} (CPUs: {})",
                                speedup,
                                PARALLEL_TARGET_SPEEDUP,
                                CPU_COUNT
                            );
                        }
                    }
                    
                    black_box(result)
                })
            },
        );
        
        // Element-wise parallel operations
        let matrix2 = generate_large_dataset(*size, 100);
        group.bench_with_input(
            BenchmarkId::new("parallel_element_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix * &matrix2;
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_parallel_diversity_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/diversity_calculation");
    
    for size in [100, 500, 1000].iter() {
        let correlation_matrix = Array2::from_shape_fn((*size, *size), |(i, j)| {
            if i == j {
                1.0
            } else {
                0.5 * ((i + j) as f64 / *size as f64)
            }
        });
        
        let pearson_measure = PearsonDiversityMeasure::new();
        let kendall_measure = KendallDiversityMeasure::new();
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("pearson_parallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(pearson_measure.calculate_diversity(
                        black_box(&correlation_matrix)
                    ))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("kendall_parallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(kendall_measure.calculate_diversity(
                        black_box(&correlation_matrix)
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_parallel_time_series_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/time_series");
    
    for size in [10000, 50000, 100000].iter() {
        let time_series = generate_time_series(*size);
        let volatility_estimator = VolatilityEstimator::new(20);
        let entropy_calculator = EntropyCalculator::new(10);
        let stats_calculator = StatisticsCalculator::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("sequential_analysis", size),
            size,
            |b, _| {
                b.iter(|| {
                    let volatility = volatility_estimator.calculate_volatility(black_box(&time_series));
                    let entropy = entropy_calculator.calculate_entropy(black_box(&time_series));
                    let stats = stats_calculator.calculate_all_statistics(black_box(&time_series));
                    black_box((volatility, entropy, stats))
                })
            },
        );
        
        // Parallel processing using rayon
        group.bench_with_input(
            BenchmarkId::new("parallel_analysis", size),
            size,
            |b, _| {
                b.iter(|| {
                    let ts = black_box(&time_series);
                    let ts_arc = Arc::new(ts.clone());
                    
                    let results: Vec<_> = vec![
                        {
                            let ts = Arc::clone(&ts_arc);
                            std::thread::spawn(move || {
                                volatility_estimator.calculate_volatility(&ts)
                            })
                        },
                        {
                            let ts = Arc::clone(&ts_arc);
                            std::thread::spawn(move || {
                                entropy_calculator.calculate_entropy(&ts)
                            })
                        },
                        {
                            let ts = Arc::clone(&ts_arc);
                            std::thread::spawn(move || {
                                stats_calculator.calculate_all_statistics(&ts)
                            })
                        },
                    ].into_iter()
                     .map(|handle| handle.join().unwrap())
                     .collect();
                    
                    black_box(results)
                })
            },
        );
    }
    group.finish();
}

fn bench_parallel_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/batch_processing");
    
    for batch_size in [100, 1000, 5000].iter() {
        let datasets: Vec<CdfaArray> = (0..*batch_size)
            .map(|_| generate_time_series(1000))
            .collect();
        
        let volatility_estimator = VolatilityEstimator::new(20);
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        // Sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = datasets
                        .iter()
                        .map(|data| volatility_estimator.calculate_volatility(data))
                        .collect();
                    black_box(results)
                })
            },
        );
        
        // Parallel batch processing
        group.bench_with_input(
            BenchmarkId::new("parallel_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = datasets
                        .par_iter()
                        .map(|data| volatility_estimator.calculate_volatility(data))
                        .collect();
                    black_box(results)
                })
            },
        );
    }
    group.finish();
}

fn bench_thread_pool_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/thread_pool");
    
    // Test different thread pool configurations
    for thread_count in [1, 2, 4, 8, CPU_COUNT].iter() {
        if *thread_count > CPU_COUNT {
            continue;
        }
        
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(*thread_count)
            .build()
            .unwrap();
        
        let data = generate_large_dataset(10000, 50);
        
        group.bench_with_input(
            BenchmarkId::new("thread_pool_efficiency", thread_count),
            thread_count,
            |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        let result: Vec<_> = data
                            .axis_iter(Axis(0))
                            .into_par_iter()
                            .map(|row| {
                                // Simulate CPU-intensive work
                                row.iter().map(|&x| x.powi(2)).sum::<f64>()
                            })
                            .collect();
                        black_box(result)
                    })
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/memory_contention");
    
    for size in [10000, 50000, 100000].iter() {
        let data = generate_time_series(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Test memory access patterns
        group.bench_with_input(
            BenchmarkId::new("sequential_memory_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let sum: f64 = data.iter().sum();
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel_memory_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let sum: f64 = data.par_iter().sum();
                    black_box(sum)
                })
            },
        );
        
        // Test strided access patterns
        group.bench_with_input(
            BenchmarkId::new("strided_parallel_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let sum: f64 = data
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0)
                        .map(|(_, &x)| x)
                        .collect::<Vec<_>>()
                        .par_iter()
                        .sum();
                    black_box(sum)
                })
            },
        );
    }
    group.finish();
}

fn bench_numa_awareness(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/numa_awareness");
    
    // Large dataset that might span NUMA nodes
    let large_data = generate_large_dataset(20000, 100);
    
    group.bench_function("numa_aware_processing", |b| {
        b.iter(|| {
            // Process data in chunks to maintain NUMA locality
            let chunk_size = large_data.nrows() / CPU_COUNT.max(1);
            let results: Vec<_> = (0..CPU_COUNT)
                .into_par_iter()
                .map(|cpu_id| {
                    let start = cpu_id * chunk_size;
                    let end = ((cpu_id + 1) * chunk_size).min(large_data.nrows());
                    
                    if start < end {
                        let chunk = large_data.slice(ndarray::s![start..end, ..]);
                        chunk.sum()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            
            black_box(results)
        })
    });
    
    group.finish();
}

criterion_group!(
    parallel_benches,
    bench_parallel_matrix_operations,
    bench_parallel_diversity_calculation,
    bench_parallel_time_series_analysis,
    bench_parallel_batch_processing,
    bench_thread_pool_efficiency,
    bench_memory_contention,
    bench_numa_awareness
);

criterion_main!(parallel_benches);