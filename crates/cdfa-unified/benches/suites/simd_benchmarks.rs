use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use cdfa_unified::{
    types::{CdfaArray, CdfaFloat},
    core::diversity::{PearsonDiversityMeasure, KendallDiversityMeasure},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const SIMD_TARGET_SPEEDUP: f64 = 2.0; // Minimum 2x speedup expected

fn generate_aligned_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let data1: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
    let data2: Vec<f64> = (0..size).map(|i| (i as f64 * 0.002) + 1.0).collect();
    (data1, data2)
}

// Scalar implementation for comparison
fn scalar_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// AVX2 implementation
#[cfg(target_arch = "x86_64")]
fn avx2_dot_product(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return scalar_dot_product(a, b);
    }
    
    unsafe {
        let mut sum = _mm256_setzero_pd();
        let chunks = a.len() / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let prod = _mm256_mul_pd(va, vb);
            sum = _mm256_add_pd(sum, prod);
        }
        
        // Extract and sum the four 64-bit floats
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f64>();
        
        // Handle remaining elements
        for i in (chunks * 4)..a.len() {
            total += a[i] * b[i];
        }
        
        total
    }
}

// AVX512 implementation (if available)
#[cfg(target_arch = "x86_64")]
fn avx512_dot_product(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx512f") {
        return avx2_dot_product(a, b);
    }
    
    unsafe {
        let mut sum = _mm512_setzero_pd();
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
            let prod = _mm512_mul_pd(va, vb);
            sum = _mm512_add_pd(sum, prod);
        }
        
        let total = _mm512_reduce_add_pd(sum);
        
        // Handle remaining elements
        let mut remaining_total = 0.0;
        for i in (chunks * 8)..a.len() {
            remaining_total += a[i] * b[i];
        }
        
        total + remaining_total
    }
}

fn bench_simd_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/dot_product");
    
    for size in [1000, 10000, 100000].iter() {
        let (data1, data2) = generate_aligned_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(scalar_dot_product(
                        black_box(&data1),
                        black_box(&data2)
                    ))
                })
            },
        );
        
        // AVX2 SIMD
        #[cfg(target_arch = "x86_64")]
        {
            let scalar_time = {
                let start = Instant::now();
                for _ in 0..100 {
                    black_box(scalar_dot_product(&data1, &data2));
                }
                start.elapsed().as_nanos() / 100
            };
            
            group.bench_with_input(
                BenchmarkId::new("avx2", size),
                size,
                |b, _| {
                    b.iter(|| {
                        let start = Instant::now();
                        let result = avx2_dot_product(
                            black_box(&data1),
                            black_box(&data2)
                        );
                        let simd_time = start.elapsed().as_nanos();
                        
                        // Validate SIMD speedup for smaller sizes
                        if *size == 1000 && is_x86_feature_detected!("avx2") {
                            let speedup = scalar_time as f64 / simd_time as f64;
                            if speedup > 0.1 { // Only check if measurements are meaningful
                                assert!(
                                    speedup >= SIMD_TARGET_SPEEDUP,
                                    "AVX2 speedup {} < target {}",
                                    speedup,
                                    SIMD_TARGET_SPEEDUP
                                );
                            }
                        }
                        
                        black_box(result)
                    })
                },
            );
            
            // AVX512 SIMD
            group.bench_with_input(
                BenchmarkId::new("avx512", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(avx512_dot_product(
                            black_box(&data1),
                            black_box(&data2)
                        ))
                    })
                },
            );
        }
        
        // NDArray SIMD (built-in optimization)
        let array1 = Array1::from_vec(data1.clone());
        let array2 = Array1::from_vec(data2.clone());
        group.bench_with_input(
            BenchmarkId::new("ndarray_simd", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(array1.dot(black_box(&array2)))
                })
            },
        );
    }
    group.finish();
}

fn bench_simd_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/matrix_operations");
    
    for size in [100, 500, 1000].iter() {
        let matrix1 = Array2::<f64>::from_shape_fn((*size, *size), |(i, j)| {
            (i as f64 * 0.01) + (j as f64 * 0.001)
        });
        let matrix2 = Array2::<f64>::from_shape_fn((*size, *size), |(i, j)| {
            (i as f64 * 0.02) + (j as f64 * 0.002) + 1.0
        });
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("element_wise_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix1 * &matrix2;
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = matrix1.dot(&matrix2);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_simd_correlation_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/correlation");
    
    for size in [50, 100, 200].iter() {
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
            BenchmarkId::new("pearson_diversity_simd", size),
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
            BenchmarkId::new("kendall_diversity_simd", size),
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

fn bench_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/feature_detection");
    
    group.bench_function("cpu_features", |b| {
        b.iter(|| {
            #[cfg(target_arch = "x86_64")]
            {
                let features = (
                    is_x86_feature_detected!("sse"),
                    is_x86_feature_detected!("sse2"),
                    is_x86_feature_detected!("sse3"),
                    is_x86_feature_detected!("sse4.1"),
                    is_x86_feature_detected!("sse4.2"),
                    is_x86_feature_detected!("avx"),
                    is_x86_feature_detected!("avx2"),
                    is_x86_feature_detected!("avx512f"),
                    is_x86_feature_detected!("fma"),
                );
                black_box(features)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                black_box(())
            }
        })
    });
    
    group.finish();
}

fn bench_memory_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/memory_alignment");
    
    for size in [1000, 10000].iter() {
        // Aligned allocation
        group.bench_with_input(
            BenchmarkId::new("aligned_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut aligned_vec = Vec::<f64>::with_capacity(size);
                    aligned_vec.resize(size, 0.0);
                    black_box(aligned_vec)
                })
            },
        );
        
        // Check alignment performance impact
        let aligned_data: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("aligned_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let sum: f64 = aligned_data.iter().sum();
                    black_box(sum)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    simd_benches,
    bench_simd_dot_product,
    bench_simd_matrix_operations,
    bench_simd_correlation_calculation,
    bench_feature_detection,
    bench_memory_alignment
);

criterion_main!(simd_benches);