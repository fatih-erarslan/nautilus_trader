//! GPU acceleration benchmarks for TorchScript Fusion
//!
//! These benchmarks specifically test GPU acceleration performance,
//! comparing CUDA, Metal, and CPU backends.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_torchscript_fusion::{TorchScriptFusion, FusionType, FusionParams, device_info, DeviceType};
use ndarray::Array2;
use tokio::runtime::Runtime;
use candle_core::Device;

/// GPU benchmark configuration
struct GpuBenchConfig {
    num_signals: usize,
    sequence_length: usize,
    device_type: DeviceType,
    fusion_type: FusionType,
}

impl GpuBenchConfig {
    fn new(num_signals: usize, sequence_length: usize, device_type: DeviceType, fusion_type: FusionType) -> Self {
        Self {
            num_signals,
            sequence_length,
            device_type,
            fusion_type,
        }
    }

    fn id(&self) -> String {
        format!(
            "{:?}_{}_{}sig_{}len",
            self.device_type,
            self.fusion_type.as_str(),
            self.num_signals,
            self.sequence_length
        )
    }
}

/// Create large test datasets for GPU benchmarking
fn create_gpu_test_data(num_signals: usize, sequence_length: usize) -> (Array2<f32>, Array2<f32>) {
    let signals = Array2::from_shape_fn((num_signals, sequence_length), |(i, j)| {
        // Create complex, realistic trading signals
        let base_freq = (i + 1) as f32 * 0.1;
        let time = j as f32 * 0.001;
        
        // Multiple frequency components
        let signal1 = (time * base_freq * std::f32::consts::TAU).sin();
        let signal2 = (time * base_freq * 2.0 * std::f32::consts::TAU).cos() * 0.5;
        let signal3 = (time * base_freq * 4.0 * std::f32::consts::TAU).sin() * 0.25;
        
        // Trend component
        let trend = time * 0.01 * (i as f32 - 2.0);
        
        // Noise component
        let noise = ((i * 1337 + j * 7919) as f32).sin() * 0.1;
        
        signal1 + signal2 + signal3 + trend + noise
    });

    let confidences = Array2::from_shape_fn((num_signals, sequence_length), |(i, j)| {
        // Dynamic confidence based on signal characteristics
        let base_conf = 0.6 + (i as f32 * 0.08).min(0.3);
        let time_var = (j as f32 * 0.01).sin() * 0.1;
        let signal_strength = signals[(i, j)].abs().min(1.0) * 0.2;
        
        (base_conf + time_var + signal_strength).clamp(0.2, 0.95)
    });

    (signals, confidences)
}

/// Get available devices for benchmarking
fn get_available_devices() -> Vec<DeviceType> {
    let mut devices = vec![DeviceType::Cpu];
    
    if let Ok(device_list) = device_info() {
        for device in device_list {
            if device.is_available {
                match device.device_type {
                    DeviceType::Cuda | DeviceType::Metal | DeviceType::Rocm => {
                        if !devices.contains(&device.device_type) {
                            devices.push(device.device_type);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    
    devices
}

/// Create device from device type
fn create_device(device_type: DeviceType, index: usize) -> Option<Device> {
    match device_type {
        DeviceType::Cpu => Some(Device::Cpu),
        DeviceType::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(index).ok()
            }
            #[cfg(not(feature = "cuda"))]
            None
        }
        DeviceType::Metal => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(index).ok()
            }
            #[cfg(not(feature = "metal"))]
            None
        }
        _ => None,
    }
}

/// Benchmark GPU vs CPU performance across different data sizes
fn bench_gpu_vs_cpu_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    // Test increasingly large datasets
    let data_sizes = vec![
        (10, 1000),    // 10K elements
        (20, 2000),    // 40K elements
        (50, 2000),    // 100K elements
        (100, 2000),   // 200K elements
        (200, 2000),   // 400K elements
    ];
    
    let mut group = c.benchmark_group("gpu_cpu_scaling");
    group.measurement_time(std::time::Duration::from_secs(15));
    
    for (num_signals, seq_len) in data_sizes {
        let (signals, confidences) = create_gpu_test_data(num_signals, seq_len);
        let params = FusionParams::default();
        
        group.throughput(Throughput::Elements((num_signals * seq_len) as u64));
        
        for &device_type in &available_devices {
            if let Some(device) = create_device(device_type, 0) {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{:?}", device_type),
                        format!("{}x{}", num_signals, seq_len)
                    ),
                    &(num_signals, seq_len),
                    |b, _| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            
                            let result = fusion.fuse_signals(
                                black_box(&signals),
                                black_box(&confidences),
                                black_box(FusionType::Score),
                                black_box(&params),
                            ).await.unwrap();
                            
                            black_box(result)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark different fusion types on GPU
fn bench_gpu_fusion_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    // Use large dataset to benefit from GPU parallelization
    let (signals, confidences) = create_gpu_test_data(50, 2000);
    let params = FusionParams::default();
    
    let mut group = c.benchmark_group("gpu_fusion_types");
    group.throughput(Throughput::Elements(100_000));
    
    for &device_type in &available_devices {
        if let Some(device) = create_device(device_type, 0) {
            for fusion_type in FusionType::all() {
                group.bench_with_input(
                    BenchmarkId::new(format!("{:?}", device_type), fusion_type.as_str()),
                    &fusion_type,
                    |b, &fusion_type| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            
                            let result = fusion.fuse_signals(
                                black_box(&signals),
                                black_box(&confidences),
                                black_box(fusion_type),
                                black_box(&params),
                            ).await.unwrap();
                            
                            black_box(result)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark memory transfer overhead
fn bench_memory_transfer_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    let data_sizes = vec![
        (5, 100),     // Small - 500 elements
        (10, 500),    // Medium - 5K elements  
        (50, 1000),   // Large - 50K elements
        (100, 2000),  // XLarge - 200K elements
    ];
    
    let mut group = c.benchmark_group("memory_transfer");
    
    for (num_signals, seq_len) in data_sizes {
        let (signals, confidences) = create_gpu_test_data(num_signals, seq_len);
        let params = FusionParams::default();
        
        group.throughput(Throughput::Bytes(
            (num_signals * seq_len * std::mem::size_of::<f32>() * 2) as u64
        ));
        
        for &device_type in &available_devices {
            if device_type == DeviceType::Cpu {
                continue; // Skip CPU for memory transfer test
            }
            
            if let Some(device) = create_device(device_type, 0) {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{:?}_transfer", device_type),
                        format!("{}x{}", num_signals, seq_len)
                    ),
                    &(num_signals, seq_len),
                    |b, _| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            
                            // Measure including memory transfer overhead
                            let result = fusion.fuse_signals(
                                black_box(&signals),
                                black_box(&confidences),
                                black_box(FusionType::Score),
                                black_box(&params),
                            ).await.unwrap();
                            
                            black_box(result)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark batch processing efficiency
fn bench_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    // Test different batch sizes
    let batch_configs = vec![
        (5, 100, 1),    // Single operation
        (5, 100, 10),   // Small batch
        (5, 100, 50),   // Medium batch
        (5, 100, 100),  // Large batch
    ];
    
    let mut group = c.benchmark_group("batch_processing");
    
    for (num_signals, seq_len, batch_size) in batch_configs {
        let params = FusionParams::default();
        
        // Create batch data
        let batch_data: Vec<(Array2<f32>, Array2<f32>)> = (0..batch_size)
            .map(|_| create_gpu_test_data(num_signals, seq_len))
            .collect();
        
        group.throughput(Throughput::Elements(
            (num_signals * seq_len * batch_size) as u64
        ));
        
        for &device_type in &available_devices {
            if let Some(device) = create_device(device_type, 0) {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{:?}_batch", device_type),
                        format!("{}x{}x{}", num_signals, seq_len, batch_size)
                    ),
                    &batch_size,
                    |b, _| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            
                            let mut results = Vec::new();
                            for (signals, confidences) in &batch_data {
                                let result = fusion.fuse_signals(
                                    black_box(signals),
                                    black_box(confidences),
                                    black_box(FusionType::Score),
                                    black_box(&params),
                                ).await.unwrap();
                                results.push(result);
                            }
                            
                            black_box(results)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark GPU utilization patterns
fn bench_gpu_utilization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    // Test different workload patterns
    let workload_patterns = vec![
        ("short_burst", vec![(10, 100); 20]),        // Many small operations
        ("medium_sustained", vec![(50, 500); 10]),   // Medium operations
        ("large_single", vec![(200, 2000)]),         // Single large operation
        ("mixed_pattern", vec![(5, 50), (20, 200), (100, 1000), (10, 100)]), // Mixed sizes
    ];
    
    let mut group = c.benchmark_group("gpu_utilization");
    
    for (pattern_name, workload) in workload_patterns {
        let total_elements: usize = workload.iter()
            .map(|(n, s)| n * s)
            .sum();
        
        group.throughput(Throughput::Elements(total_elements as u64));
        
        for &device_type in &available_devices {
            if device_type == DeviceType::Cpu {
                continue; // Focus on GPU utilization
            }
            
            if let Some(device) = create_device(device_type, 0) {
                group.bench_with_input(
                    BenchmarkId::new(format!("{:?}_util", device_type), pattern_name),
                    &workload,
                    |b, workload| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            let params = FusionParams::default();
                            
                            let mut results = Vec::new();
                            for &(num_signals, seq_len) in workload {
                                let (signals, confidences) = create_gpu_test_data(num_signals, seq_len);
                                
                                let result = fusion.fuse_signals(
                                    black_box(&signals),
                                    black_box(&confidences),
                                    black_box(FusionType::Score),
                                    black_box(&params),
                                ).await.unwrap();
                                results.push(result);
                            }
                            
                            black_box(results)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark precision and numerical stability on GPU
fn bench_gpu_precision(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let available_devices = get_available_devices();
    
    // Create challenging numerical test cases
    let precision_tests = vec![
        ("small_values", 1e-6f32, 1e-8f32),      // Very small numbers
        ("large_values", 1e6f32, 1e4f32),        // Large numbers  
        ("mixed_scale", 1e-3f32, 1e3f32),        // Mixed scales
        ("normal_range", 0.1f32, 10.0f32),       // Normal trading ranges
    ];
    
    let mut group = c.benchmark_group("gpu_precision");
    
    for (test_name, scale1, scale2) in precision_tests {
        // Create signals with specific numerical characteristics
        let signals = Array2::from_shape_fn((10, 1000), |(i, j)| {
            let base = if i % 2 == 0 { scale1 } else { scale2 };
            base * (j as f32 * 0.01).sin()
        });
        
        let confidences = Array2::from_elem((10, 1000), 0.8);
        let params = FusionParams::default();
        
        group.throughput(Throughput::Elements(10_000));
        
        for &device_type in &available_devices {
            if let Some(device) = create_device(device_type, 0) {
                group.bench_with_input(
                    BenchmarkId::new(format!("{:?}_precision", device_type), test_name),
                    test_name,
                    |b, _| {
                        b.to_async(&rt).iter(|| async {
                            let mut fusion = TorchScriptFusion::with_device(device.clone()).await.unwrap();
                            
                            let result = fusion.fuse_signals(
                                black_box(&signals),
                                black_box(&confidences),
                                black_box(FusionType::Weighted), // Most numerically challenging
                                black_box(&params),
                            ).await.unwrap();
                            
                            // Validate result is finite
                            assert!(result.fused_signal.iter().all(|x| x.is_finite()));
                            assert!(result.confidence.iter().all(|x| x.is_finite()));
                            
                            black_box(result)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

criterion_group!(
    gpu_benches,
    bench_gpu_vs_cpu_scaling,
    bench_gpu_fusion_types,
    bench_memory_transfer_overhead,
    bench_batch_processing,
    bench_gpu_utilization,
    bench_gpu_precision
);

criterion_main!(gpu_benches);