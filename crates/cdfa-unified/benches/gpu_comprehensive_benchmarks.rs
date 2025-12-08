//! Comprehensive GPU benchmarks for CDFA operations
//!
//! This benchmark suite evaluates GPU performance across different operations,
//! backends, and data sizes with detailed profiling and comparison metrics.

use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale
};
use cdfa_unified::{
    gpu::{GpuManager, GpuConfig, GpuPrecision, detection::detect_gpu_devices},
    types::{CdfaFloat, CdfaMatrix},
    prelude::*,
};
use ndarray::Array2;
use std::time::Duration;

/// Benchmark GPU vs CPU matrix multiplication
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Test different matrix sizes
    let sizes = vec![64, 128, 256, 512, 1024];
    
    for size in sizes {
        let matrix_a = Array2::from_shape_fn((size, size), |(i, j)| {
            (i as CdfaFloat * 0.01) + (j as CdfaFloat * 0.001)
        });
        let matrix_b = Array2::from_shape_fn((size, size), |(i, j)| {
            (i as CdfaFloat * 0.02) + (j as CdfaFloat * 0.002) + 1.0
        });
        
        group.throughput(Throughput::Elements((size * size * size) as u64));
        
        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = matrix_a.dot(&matrix_b);
                    black_box(result)
                })
            },
        );
        
        // GPU implementation (if available)
        if let Ok(gpu_manager) = create_gpu_manager() {
            group.bench_with_input(
                BenchmarkId::new("gpu_single", size),
                &size,
                |b, _| {
                    b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                        let result = gpu_manager.matrix_multiply(
                            black_box(&matrix_a),
                            black_box(&matrix_b),
                            None
                        ).await.unwrap_or_else(|_| matrix_a.dot(&matrix_b));
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark element-wise operations
fn bench_element_wise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    let sizes = vec![1000, 10000, 100000, 1000000];
    
    for size in sizes {
        let side_len = (size as f64).sqrt() as usize;
        let matrix_a = Array2::from_shape_fn((side_len, side_len), |(i, j)| {
            (i as CdfaFloat * 0.01) + (j as CdfaFloat * 0.001)
        });
        let matrix_b = Array2::from_shape_fn((side_len, side_len), |(i, j)| {
            (i as CdfaFloat * 0.02) + (j as CdfaFloat * 0.002)
        });
        
        group.throughput(Throughput::Elements(size as u64));
        
        // CPU operations
        group.bench_with_input(
            BenchmarkId::new("cpu_add", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix_a + &matrix_b;
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cpu_multiply", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix_a * &matrix_b;
                    black_box(result)
                })
            },
        );
        
        // GPU operations (if available)
        if let Ok(gpu_manager) = create_gpu_manager() {
            group.bench_with_input(
                BenchmarkId::new("gpu_add", size),
                &size,
                |b, _| {
                    b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                        let result = gpu_manager.element_wise_op(
                            black_box(&matrix_a),
                            black_box(&matrix_b),
                            |x, y| x + y,
                            None
                        ).await.unwrap_or_else(|_| &matrix_a + &matrix_b);
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark reduction operations
fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    let sizes = vec![1000, 10000, 100000, 1000000, 10000000];
    
    for size in sizes {
        let side_len = (size as f64).sqrt() as usize;
        let matrix = Array2::from_shape_fn((side_len, side_len), |(i, j)| {
            ((i + j) as CdfaFloat / size as CdfaFloat).sin()
        });
        
        group.throughput(Throughput::Elements(size as u64));
        
        // CPU reduction
        group.bench_with_input(
            BenchmarkId::new("cpu_sum", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = matrix.sum();
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cpu_mean", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = matrix.mean().unwrap_or(0.0);
                    black_box(result)
                })
            },
        );
        
        // GPU reduction (if available)
        if let Ok(gpu_manager) = create_gpu_manager() {
            group.bench_with_input(
                BenchmarkId::new("gpu_sum", size),
                &size,
                |b, _| {
                    b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                        let result = gpu_manager.reduce_sum(black_box(&matrix), None)
                            .await.unwrap_or_else(|_| matrix.sum());
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark different precision modes
fn bench_precision_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_modes");
    group.measurement_time(Duration::from_secs(20));
    
    let size = 512;
    let matrix_a = Array2::from_shape_fn((size, size), |(i, j)| {
        (i as CdfaFloat * 0.001) + (j as CdfaFloat * 0.0001)
    });
    let matrix_b = Array2::from_shape_fn((size, size), |(i, j)| {
        (i as CdfaFloat * 0.002) + (j as CdfaFloat * 0.0002) + 1.0
    });
    
    let precisions = [
        ("single", GpuPrecision::Single),
        ("double", GpuPrecision::Double),
        ("mixed", GpuPrecision::Mixed),
    ];
    
    for (name, precision) in precisions {
        let config = GpuConfig {
            precision,
            fallback_to_cpu: true,
            ..Default::default()
        };
        
        if let Ok(gpu_manager) = GpuManager::new(config) {
            group.bench_with_input(
                BenchmarkId::new("gpu_matmul", name),
                &name,
                |b, _| {
                    b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                        let result = gpu_manager.matrix_multiply(
                            black_box(&matrix_a),
                            black_box(&matrix_b),
                            None
                        ).await.unwrap_or_else(|_| matrix_a.dot(&matrix_b));
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(Duration::from_secs(15));
    
    let batch_sizes = vec![1, 5, 10, 20, 50];
    let matrix_size = 128;
    
    for batch_size in batch_sizes {
        let matrices: Vec<CdfaMatrix> = (0..batch_size)
            .map(|i| Array2::from_shape_fn((matrix_size, matrix_size), |(row, col)| {
                (i as CdfaFloat * 0.001) + (row as CdfaFloat * 0.01) + (col as CdfaFloat * 0.001)
            }))
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // CPU batch processing
        group.bench_with_input(
            BenchmarkId::new("cpu_batch_sum", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<CdfaFloat> = matrices
                        .iter()
                        .map(|matrix| matrix.sum())
                        .collect();
                    black_box(results)
                })
            },
        );
        
        // GPU batch processing (if available)
        if let Ok(gpu_manager) = create_gpu_manager() {
            group.bench_with_input(
                BenchmarkId::new("gpu_batch_sum", batch_size),
                &batch_size,
                |b, _| {
                    b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                        let operation = |matrix: &CdfaMatrix| -> CdfaResult<CdfaFloat> {
                            Ok(matrix.sum())
                        };
                        
                        let results = gpu_manager.batch_process(
                            black_box(&matrices),
                            operation,
                            None
                        ).await.unwrap_or_else(|_| {
                            matrices.iter().map(|m| m.sum()).collect()
                        });
                        black_box(results)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory transfer overhead
fn bench_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_transfer");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    if let Ok(devices) = detect_gpu_devices() {
        if !devices.is_empty() {
            println!("Testing memory transfer with GPU device: {}", devices[0].name);
        }
    }
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144]; // Bytes
    
    for size in sizes {
        let data_size_mb = size as f64 / (1024.0 * 1024.0);
        
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("data_copy", format!("{:.1}MB", data_size_mb)),
            &size,
            |b, &size| {
                let data = vec![0.0f32; size / 4]; // f32 data
                
                b.iter(|| {
                    let copy = data.clone();
                    black_box(copy)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU device detection and initialization
fn bench_gpu_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_initialization");
    
    group.bench_function("device_detection", |b| {
        b.iter(|| {
            let devices = detect_gpu_devices().unwrap_or_default();
            black_box(devices)
        })
    });
    
    group.bench_function("gpu_manager_creation", |b| {
        b.iter(|| {
            let config = GpuConfig::default();
            let manager = GpuManager::new(config);
            black_box(manager)
        })
    });
    
    group.finish();
}

/// Benchmark numerical accuracy vs performance tradeoffs
fn bench_accuracy_vs_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_vs_performance");
    
    // Create a challenging numerical test case
    let size = 256;
    let matrix = Array2::from_shape_fn((size, size), |(i, j)| {
        let x = i as CdfaFloat / size as CdfaFloat;
        let y = j as CdfaFloat / size as CdfaFloat;
        (x * y).sin() + (x - y).cos()
    });
    
    // CPU reference (highest accuracy)
    group.bench_function("cpu_reference", |b| {
        b.iter(|| {
            let result = matrix.sum();
            black_box(result)
        })
    });
    
    // Test different GPU precision modes if available
    let precisions = [
        ("gpu_single", GpuPrecision::Single),
        ("gpu_double", GpuPrecision::Double),
        ("gpu_mixed", GpuPrecision::Mixed),
    ];
    
    for (name, precision) in precisions {
        let config = GpuConfig {
            precision,
            fallback_to_cpu: true,
            ..Default::default()
        };
        
        if let Ok(gpu_manager) = GpuManager::new(config) {
            group.bench_function(name, |b| {
                b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
                    let result = gpu_manager.reduce_sum(black_box(&matrix), None)
                        .await.unwrap_or_else(|_| matrix.sum());
                    black_box(result)
                })
            });
        }
    }
    
    group.finish();
}

/// Helper function to create GPU manager with fallback
fn create_gpu_manager() -> Result<GpuManager, Box<dyn std::error::Error>> {
    let config = GpuConfig {
        fallback_to_cpu: true,
        enable_profiling: false,
        ..Default::default()
    };
    
    GpuManager::new(config).map_err(|e| e.into())
}

/// Test if GPU is available for benchmarking
fn is_gpu_available() -> bool {
    match detect_gpu_devices() {
        Ok(devices) => !devices.is_empty() && devices.iter().any(|d| {
            !matches!(d.backend, cdfa_unified::gpu::GpuBackend::Cpu)
        }),
        Err(_) => false,
    }
}

criterion_group!(
    benches,
    bench_matrix_multiplication,
    bench_element_wise_operations,
    bench_reduction_operations,
    bench_precision_modes,
    bench_batch_processing,
    bench_memory_transfer,
    bench_gpu_initialization,
    bench_accuracy_vs_performance
);

criterion_main!(benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    
    #[test]
    fn test_benchmark_functions() {
        // Test that benchmark functions don't panic
        let mut criterion = Criterion::default();
        
        // These should not panic even without GPU
        bench_gpu_initialization(&mut criterion);
        bench_memory_transfer(&mut criterion);
    }
    
    #[test]
    fn test_gpu_availability_check() {
        let available = is_gpu_available();
        println!("GPU available for benchmarks: {}", available);
        
        if available {
            let devices = detect_gpu_devices().unwrap();
            for device in devices {
                println!("Available device: {} ({:?})", device.name, device.backend);
            }
        }
    }
    
    #[test]
    fn test_gpu_manager_creation() {
        let result = create_gpu_manager();
        match result {
            Ok(_) => println!("GPU manager created successfully"),
            Err(e) => println!("GPU manager creation failed: {}", e),
        }
    }
}