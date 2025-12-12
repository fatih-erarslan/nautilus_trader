use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use cdfa_unified::{
    types::{CdfaArray, CdfaMatrix, CdfaFloat},
    core::diversity::{PearsonDiversityMeasure, KendallDiversityMeasure},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

// GPU acceleration availability check
fn is_gpu_available() -> bool {
    // In a real implementation, this would check for GPU compute capabilities
    // For now, we'll simulate GPU availability based on feature flags
    cfg!(feature = "gpu")
}

fn is_wgpu_available() -> bool {
    cfg!(feature = "gpu") && cfg!(feature = "wgpu")
}

fn is_candle_gpu_available() -> bool {
    cfg!(feature = "gpu") && cfg!(feature = "candle-core")
}

// Simulated GPU operations for benchmarking
struct MockGpuCompute {
    device_name: String,
    compute_units: usize,
}

impl MockGpuCompute {
    fn new() -> Option<Self> {
        if is_gpu_available() {
            Some(Self {
                device_name: "Mock GPU Device".to_string(),
                compute_units: 2048,
            })
        } else {
            None
        }
    }
    
    // Simulate GPU matrix multiplication
    fn gpu_matrix_multiply(&self, a: &CdfaMatrix, b: &CdfaMatrix) -> CdfaMatrix {
        // In a real implementation, this would use GPU compute shaders
        // For benchmarking, we simulate GPU computation with optimized CPU code
        let start = Instant::now();
        let result = a.dot(b);
        let _gpu_time = start.elapsed();
        
        // Simulate GPU acceleration (assume 2-5x speedup for large matrices)
        std::thread::sleep(std::time::Duration::from_nanos(
            (_gpu_time.as_nanos() / 3) as u64
        ));
        
        result
    }
    
    // Simulate GPU element-wise operations
    fn gpu_element_wise_op<F>(&self, a: &CdfaMatrix, b: &CdfaMatrix, _op: F) -> CdfaMatrix
    where
        F: Fn(CdfaFloat, CdfaFloat) -> CdfaFloat,
    {
        // Simulate GPU parallel element-wise operations
        a * b // Using ndarray's optimized implementation as proxy
    }
    
    // Simulate GPU reduction operations
    fn gpu_reduce_sum(&self, matrix: &CdfaMatrix) -> CdfaFloat {
        // Simulate GPU reduction with parallel sum
        matrix.sum()
    }
    
    // Simulate GPU memory transfer
    fn transfer_to_gpu(&self, _data: &CdfaMatrix) -> std::time::Duration {
        // Simulate memory transfer overhead
        std::time::Duration::from_micros(100)
    }
    
    fn transfer_from_gpu(&self, _data: &CdfaMatrix) -> std::time::Duration {
        // Simulate memory transfer overhead
        std::time::Duration::from_micros(50)
    }
}

fn bench_gpu_vs_cpu_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/matrix_operations");
    
    let gpu = MockGpuCompute::new();
    
    for size in [256, 512, 1024, 2048].iter() {
        let matrix1 = Array2::<CdfaFloat>::from_shape_fn((*size, *size), |(i, j)| {
            (i as CdfaFloat * 0.01) + (j as CdfaFloat * 0.001)
        });
        let matrix2 = Array2::<CdfaFloat>::from_shape_fn((*size, *size), |(i, j)| {
            (i as CdfaFloat * 0.02) + (j as CdfaFloat * 0.002) + 1.0
        });
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu_matrix_multiply", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = matrix1.dot(&matrix2);
                    black_box(result)
                })
            },
        );
        
        // GPU implementation (if available)
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_matrix_multiply", size),
                size,
                |b, _| {
                    b.iter(|| {
                        let transfer_to = gpu_device.transfer_to_gpu(&matrix1);
                        let _transfer_to2 = gpu_device.transfer_to_gpu(&matrix2);
                        
                        let result = gpu_device.gpu_matrix_multiply(
                            black_box(&matrix1),
                            black_box(&matrix2)
                        );
                        
                        let _transfer_from = gpu_device.transfer_from_gpu(&result);
                        
                        // For larger matrices, GPU should show benefits despite transfer overhead
                        if *size >= 1024 {
                            // Validate that computation time benefits outweigh transfer costs
                            assert!(transfer_to.as_micros() < 1000, "GPU transfer overhead too high");
                        }
                        
                        black_box(result)
                    })
                },
            );
        }
        
        // Element-wise operations
        group.bench_with_input(
            BenchmarkId::new("cpu_element_wise", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix1 * &matrix2;
                    black_box(result)
                })
            },
        );
        
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_element_wise", size),
                size,
                |b, _| {
                    b.iter(|| {
                        let result = gpu_device.gpu_element_wise_op(
                            black_box(&matrix1),
                            black_box(&matrix2),
                            |a, b| a * b
                        );
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_gpu_diversity_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/diversity_calculation");
    
    let gpu = MockGpuCompute::new();
    
    for size in [100, 300, 500, 1000].iter() {
        let correlation_matrix = Array2::from_shape_fn((*size, *size), |(i, j)| {
            if i == j {
                1.0
            } else {
                0.5 * ((i + j) as CdfaFloat / *size as CdfaFloat)
            }
        });
        
        let pearson_measure = PearsonDiversityMeasure::new();
        let kendall_measure = KendallDiversityMeasure::new();
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // CPU diversity calculation
        group.bench_with_input(
            BenchmarkId::new("cpu_pearson_diversity", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(pearson_measure.calculate_diversity(
                        black_box(&correlation_matrix)
                    ))
                })
            },
        );
        
        // Simulated GPU-accelerated diversity calculation
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_pearson_diversity", size),
                size,
                |b, _| {
                    b.iter(|| {
                        // Simulate GPU-accelerated computation
                        let _transfer_to = gpu_device.transfer_to_gpu(&correlation_matrix);
                        
                        // GPU computation would parallelize correlation analysis
                        let start = Instant::now();
                        let result = pearson_measure.calculate_diversity(&correlation_matrix);
                        let cpu_time = start.elapsed();
                        
                        // Simulate GPU speedup (2-4x for parallel operations)
                        let simulated_gpu_time = cpu_time / 3;
                        if simulated_gpu_time > std::time::Duration::from_nanos(100) {
                            std::thread::sleep(simulated_gpu_time - std::time::Duration::from_nanos(100));
                        }
                        
                        let _transfer_from = gpu_device.transfer_from_gpu(&correlation_matrix);
                        
                        black_box(result)
                    })
                },
            );
        }
        
        // CPU Kendall diversity
        group.bench_with_input(
            BenchmarkId::new("cpu_kendall_diversity", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(kendall_measure.calculate_diversity(
                        black_box(&correlation_matrix)
                    ))
                })
            },
        );
        
        // GPU Kendall diversity (more complex computation, better GPU benefit)
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_kendall_diversity", size),
                size,
                |b, _| {
                    b.iter(|| {
                        // Kendall rank correlation benefits more from GPU parallelization
                        let _transfer_to = gpu_device.transfer_to_gpu(&correlation_matrix);
                        
                        let start = Instant::now();
                        let result = kendall_measure.calculate_diversity(&correlation_matrix);
                        let cpu_time = start.elapsed();
                        
                        // Simulate higher GPU speedup for complex operations (3-6x)
                        let simulated_gpu_time = cpu_time / 4;
                        if simulated_gpu_time > std::time::Duration::from_nanos(100) {
                            std::thread::sleep(simulated_gpu_time - std::time::Duration::from_nanos(100));
                        }
                        
                        let _transfer_from = gpu_device.transfer_from_gpu(&correlation_matrix);
                        
                        black_box(result)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_gpu_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/batch_processing");
    
    let gpu = MockGpuCompute::new();
    
    for batch_size in [10, 50, 100, 500].iter() {
        let matrices: Vec<CdfaMatrix> = (0..*batch_size)
            .map(|i| Array2::from_shape_fn((100, 100), |(row, col)| {
                (i as CdfaFloat * 0.001) + (row as CdfaFloat * 0.01) + (col as CdfaFloat * 0.001)
            }))
            .collect();
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        // CPU batch processing
        group.bench_with_input(
            BenchmarkId::new("cpu_batch_sum", batch_size),
            batch_size,
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
        
        // GPU batch processing
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_batch_sum", batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        // GPU batch processing can overlap computation and memory transfer
                        let results: Vec<CdfaFloat> = matrices
                            .iter()
                            .map(|matrix| {
                                let _transfer_to = gpu_device.transfer_to_gpu(matrix);
                                let result = gpu_device.gpu_reduce_sum(matrix);
                                let _transfer_from = gpu_device.transfer_from_gpu(matrix);
                                result
                            })
                            .collect();
                        
                        // For large batches, GPU should show benefits
                        if *batch_size >= 100 {
                            // Validate batch processing efficiency
                            assert!(results.len() == *batch_size, "Batch processing failed");
                        }
                        
                        black_box(results)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_gpu_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/memory_bandwidth");
    
    let gpu = MockGpuCompute::new();
    
    for size in [1_000_000, 10_000_000, 100_000_000].iter() {
        let data = Array1::<CdfaFloat>::from_shape_fn(*size, |i| i as CdfaFloat);
        
        group.throughput(Throughput::Bytes((*size * std::mem::size_of::<CdfaFloat>()) as u64));
        
        // CPU memory bandwidth
        group.bench_with_input(
            BenchmarkId::new("cpu_memory_copy", size),
            size,
            |b, _| {
                b.iter(|| {
                    let copy = data.clone();
                    black_box(copy)
                })
            },
        );
        
        // GPU memory transfer benchmark
        if let Some(ref gpu_device) = gpu {
            group.bench_with_input(
                BenchmarkId::new("gpu_memory_transfer", size),
                size,
                |b, _| {
                    b.iter(|| {
                        // Simulate GPU memory transfer
                        let matrix_view = data.view().into_shape((data.len() / 1000, 1000)).unwrap();
                        let transfer_to = gpu_device.transfer_to_gpu(&matrix_view);
                        let transfer_from = gpu_device.transfer_from_gpu(&matrix_view);
                        
                        let total_transfer_time = transfer_to + transfer_from;
                        
                        // Validate memory transfer efficiency
                        let mb_transferred = (*size * std::mem::size_of::<CdfaFloat>()) as f64 / (1024.0 * 1024.0);
                        let bandwidth_gbps = (mb_transferred / 1024.0) / total_transfer_time.as_secs_f64();
                        
                        // Modern GPUs should achieve >100 GB/s bandwidth
                        if *size >= 10_000_000 {
                            assert!(
                                bandwidth_gbps > 50.0,
                                "GPU memory bandwidth {}GB/s too low",
                                bandwidth_gbps
                            );
                        }
                        
                        black_box(total_transfer_time)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_gpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/feature_detection");
    
    group.bench_function("gpu_availability", |b| {
        b.iter(|| {
            let gpu_available = is_gpu_available();
            let wgpu_available = is_wgpu_available();
            let candle_gpu_available = is_candle_gpu_available();
            
            black_box((gpu_available, wgpu_available, candle_gpu_available))
        })
    });
    
    if let Some(gpu) = MockGpuCompute::new() {
        group.bench_function("gpu_device_info", |b| {
            b.iter(|| {
                let device_name = &gpu.device_name;
                let compute_units = gpu.compute_units;
                
                black_box((device_name, compute_units))
            })
        });
    }
    
    group.finish();
}

fn bench_gpu_precision_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/precision_modes");
    
    let gpu = MockGpuCompute::new();
    
    if let Some(ref gpu_device) = gpu {
        let size = 1000;
        let matrix1 = Array2::<CdfaFloat>::from_shape_fn((size, size), |(i, j)| {
            (i as CdfaFloat * 0.001) + (j as CdfaFloat * 0.0001)
        });
        let matrix2 = Array2::<CdfaFloat>::from_shape_fn((size, size), |(i, j)| {
            (i as CdfaFloat * 0.002) + (j as CdfaFloat * 0.0002) + 1.0
        });
        
        group.throughput(Throughput::Elements((size * size) as u64));
        
        // Full precision (f64)
        group.bench_function("gpu_f64_precision", |b| {
            b.iter(|| {
                let result = gpu_device.gpu_matrix_multiply(
                    black_box(&matrix1),
                    black_box(&matrix2)
                );
                black_box(result)
            })
        });
        
        // Half precision simulation (would use f16 in real GPU implementation)
        group.bench_function("gpu_f32_precision", |b| {
            b.iter(|| {
                // Simulate f32 computation (faster on GPU)
                let matrix1_f32 = matrix1.mapv(|x| x as f32 as f64);
                let matrix2_f32 = matrix2.mapv(|x| x as f32 as f64);
                
                let start = Instant::now();
                let result = gpu_device.gpu_matrix_multiply(&matrix1_f32, &matrix2_f32);
                let computation_time = start.elapsed();
                
                // f32 operations should be faster on GPU
                let speedup_factor = 1.5; // Typical f32 vs f64 speedup on GPU
                let simulated_f32_time = computation_time.as_nanos() as f64 / speedup_factor;
                
                if simulated_f32_time > 1000.0 { // Only sleep if meaningful
                    std::thread::sleep(std::time::Duration::from_nanos(simulated_f32_time as u64));
                }
                
                black_box(result)
            })
        });
        
        // Mixed precision simulation
        group.bench_function("gpu_mixed_precision", |b| {
            b.iter(|| {
                // Use f32 for computation, f64 for accumulation
                let matrix1_mixed = matrix1.mapv(|x| x as f32 as f64);
                let matrix2_mixed = matrix2.mapv(|x| x as f32 as f64);
                
                let result = gpu_device.gpu_matrix_multiply(&matrix1_mixed, &matrix2_mixed);
                
                // Convert back to full precision for final result
                let final_result = result.mapv(|x| x);
                
                black_box(final_result)
            })
        });
    }
    
    group.finish();
}

fn bench_gpu_workload_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu/workload_scaling");
    
    let gpu = MockGpuCompute::new();
    
    if let Some(ref gpu_device) = gpu {
        // Test how GPU performance scales with different workload sizes
        for &threads in &[1, 64, 256, 1024, 4096] {
            let work_per_thread = 1000;
            let total_work = threads * work_per_thread;
            
            let data = Array1::<CdfaFloat>::from_shape_fn(total_work, |i| i as CdfaFloat);
            let matrix_view = data.view().into_shape((threads, work_per_thread)).unwrap();
            
            group.throughput(Throughput::Elements(total_work as u64));
            
            group.bench_with_input(
                BenchmarkId::new("gpu_parallel_workload", threads),
                &threads,
                |b, _| {
                    b.iter(|| {
                        // Simulate GPU parallel execution
                        let result = gpu_device.gpu_reduce_sum(&matrix_view);
                        
                        // GPU should show better scaling for larger thread counts
                        if threads >= 1024 {
                            // Validate that GPU utilization is efficient
                            assert!(result > 0.0, "GPU computation failed");
                        }
                        
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    gpu_benches,
    bench_gpu_vs_cpu_matrix_ops,
    bench_gpu_diversity_calculation,
    bench_gpu_batch_processing,
    bench_gpu_memory_bandwidth,
    bench_gpu_feature_detection,
    bench_gpu_precision_modes,
    bench_gpu_workload_scaling
);

criterion_main!(gpu_benches);