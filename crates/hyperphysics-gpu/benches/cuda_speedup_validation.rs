//! CUDA Speedup Validation Benchmark
//!
//! **TARGET: Validate 800× speedup vs CPU baseline**
//!
//! This benchmark measures actual GPU acceleration performance
//! against CPU baseline for HyperPhysics consciousness calculations.
//!
//! NOTE: This benchmark requires the `cuda-backend` feature to be enabled.
//! Run with: cargo bench --features cuda-backend --bench cuda_speedup_validation

#[cfg(feature = "cuda-backend")]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use hyperphysics_gpu::backend::cuda_real::create_cuda_backend;
    use hyperphysics_gpu::backend::cpu::CPUBackend;
    use hyperphysics_gpu::backend::{GPUBackend, BufferUsage};

    /// Test workload sizes
    const WORKLOAD_SIZES: &[usize] = &[
        1024,        // 1K elements
        16384,       // 16K elements
        262144,      // 256K elements
        1048576,     // 1M elements
        16777216,    // 16M elements (target for 800× speedup)
    ];

    /// Simple compute shader for consciousness metric
    const PHI_SHADER: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder for phi calculation
}
"#;

    /// Benchmark CPU baseline
    fn bench_cpu_baseline(c: &mut Criterion) {
        let mut group = c.benchmark_group("CPU Baseline");

        for size in WORKLOAD_SIZES {
            group.bench_with_input(
                BenchmarkId::from_parameter(size),
                size,
                |b, &size| {
                    let backend = CPUBackend::new();
                    let mut buffer = backend.create_buffer(size * 4, BufferUsage::Storage)
                        .expect("Failed to create buffer");

                    let data = vec![1.0f32; size];
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * 4
                        )
                    };

                    b.iter(|| {
                        backend.write_buffer(buffer.as_mut(), bytes)
                            .expect("Write failed");

                        backend.execute_compute(PHI_SHADER, [size as u32, 1, 1])
                            .expect("Execute failed");

                        let result = backend.read_buffer(buffer.as_ref())
                            .expect("Read failed");

                        black_box(result);
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark CUDA GPU acceleration
    fn bench_cuda_gpu(c: &mut Criterion) {
        // Check if CUDA is available
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => {
                eprintln!("CUDA not available, skipping GPU benchmarks");
                return;
            }
        };

        let mut group = c.benchmark_group("CUDA GPU");

        println!("CUDA Device: {}", backend.capabilities().device_name);

        for size in WORKLOAD_SIZES {
            group.bench_with_input(
                BenchmarkId::from_parameter(size),
                size,
                |b, &size| {
                    let mut buffer = backend.create_buffer(size * 4, BufferUsage::Storage)
                        .expect("Failed to create buffer");

                    let data = vec![1.0f32; size];
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * 4
                        )
                    };

                    b.iter(|| {
                        backend.write_buffer(buffer.as_mut(), bytes)
                            .expect("Write failed");

                        backend.execute_compute(PHI_SHADER, [size as u32, 1, 1])
                            .expect("Execute failed");

                        backend.synchronize()
                            .expect("Sync failed");

                        let result = backend.read_buffer(buffer.as_ref())
                            .expect("Read failed");

                        black_box(result);
                    });
                },
            );
        }

        group.finish();

        // Print speedup analysis
        println!("\n=== SPEEDUP ANALYSIS ===");
        println!("Target: 800× speedup for large workloads");
        println!("Run comparison analysis manually by comparing group results");
    }

    /// Compare CPU vs CUDA memory bandwidth
    fn bench_memory_bandwidth(c: &mut Criterion) {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => {
                eprintln!("CUDA not available, skipping memory benchmarks");
                return;
            }
        };

        let mut group = c.benchmark_group("Memory Bandwidth");

        const SIZES: &[usize] = &[1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024];

        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("CUDA_HtoD", size),
                size,
                |b, &size| {
                    let mut buffer = backend.create_buffer(size, BufferUsage::CopyDst)
                        .expect("Failed to create buffer");

                    let data = vec![0u8; size];

                    b.iter(|| {
                        backend.write_buffer(buffer.as_mut(), &data)
                            .expect("Write failed");
                        backend.synchronize().expect("Sync failed");
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("CUDA_DtoH", size),
                size,
                |b, &size| {
                    let buffer = backend.create_buffer(size, BufferUsage::CopySrc)
                        .expect("Failed to create buffer");

                    b.iter(|| {
                        let result = backend.read_buffer(buffer.as_ref())
                            .expect("Read failed");
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group! {
        name = cuda_speedup;
        config = Criterion::default()
            .sample_size(10)
            .measurement_time(std::time::Duration::from_secs(10));
        targets = bench_cpu_baseline, bench_cuda_gpu, bench_memory_bandwidth
    }

    criterion_main!(cuda_speedup);
}

#[cfg(not(feature = "cuda-backend"))]
fn main() {
    println!("CUDA benchmarks require the 'cuda-backend' feature");
    println!("Run with: cargo bench --features cuda-backend --bench cuda_speedup_validation");
}
