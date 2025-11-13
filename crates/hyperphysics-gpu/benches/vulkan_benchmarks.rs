//! Vulkan backend performance benchmarks
//!
//! **SCIENTIFIC METHODOLOGY:**
//! Benchmarks compare Vulkan GPU performance against CPU baseline using:
//! - Criterion.rs for statistical analysis (mean, variance, outliers)
//! - Multiple workload sizes (1K, 10K, 100K, 1M elements)
//! - Various compute patterns (memory-bound, compute-bound, mixed)
//!
//! **EXPECTED PERFORMANCE:**
//! - Memory bandwidth: >100 GB/s (GPU) vs ~50 GB/s (CPU)
//! - Compute throughput: 10-800× speedup for parallel workloads
//! - Latency: <1ms for small dispatches, <10ms for large

#[cfg(feature = "vulkan-backend")]
mod vulkan_benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
    use hyperphysics_gpu::backend::{GPUBackend, BufferUsage};
    use hyperphysics_gpu::backend::vulkan::VulkanBackend;
    use hyperphysics_gpu::backend::cpu::CPUBackend;

    /// Benchmark 1: Buffer allocation overhead
    fn bench_buffer_allocation(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                println!("Vulkan not available, skipping benchmarks");
                return;
            }
        };

        let mut group = c.benchmark_group("buffer_allocation");

        for size in [1024u64, 10_240, 102_400, 1_024_000].iter() {
            group.throughput(Throughput::Bytes(*size));
            group.bench_with_input(BenchmarkId::new("vulkan", size), size, |b, &size| {
                b.iter(|| {
                    let buffer = backend.create_buffer(size, BufferUsage::Storage)
                        .expect("Allocation failed");
                    black_box(buffer);
                });
            });
        }

        group.finish();
    }

    /// Benchmark 2: Buffer write throughput (CPU→GPU transfer)
    fn bench_buffer_write(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let mut group = c.benchmark_group("buffer_write");

        for size in [1024u64, 10_240, 102_400, 1_024_000].iter() {
            let mut buffer = backend.create_buffer(*size, BufferUsage::Storage)
                .expect("Buffer creation failed");
            let data = vec![42u8; *size as usize];

            group.throughput(Throughput::Bytes(*size));
            group.bench_with_input(BenchmarkId::new("vulkan", size), size, |b, _| {
                b.iter(|| {
                    backend.write_buffer(buffer.as_mut(), &data)
                        .expect("Write failed");
                    black_box(&buffer);
                });
            });
        }

        group.finish();
    }

    /// Benchmark 3: Buffer read throughput (GPU→CPU transfer)
    fn bench_buffer_read(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let mut group = c.benchmark_group("buffer_read");

        for size in [1024u64, 10_240, 102_400, 1_024_000].iter() {
            let mut buffer = backend.create_buffer(*size, BufferUsage::Storage)
                .expect("Buffer creation failed");
            let data = vec![42u8; *size as usize];
            backend.write_buffer(buffer.as_mut(), &data).unwrap();

            group.throughput(Throughput::Bytes(*size));
            group.bench_with_input(BenchmarkId::new("vulkan", size), size, |b, _| {
                b.iter(|| {
                    let result = backend.read_buffer(buffer.as_ref())
                        .expect("Read failed");
                    black_box(result);
                });
            });
        }

        group.finish();
    }

    /// Benchmark 4: Compute shader dispatch latency
    fn bench_compute_dispatch(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let shader = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                data[index] = data[index] * 2.0;
            }
        "#;

        let mut group = c.benchmark_group("compute_dispatch");

        for workgroups in [1u32, 4, 16, 64, 256].iter() {
            group.bench_with_input(
                BenchmarkId::new("vulkan", workgroups),
                workgroups,
                |b, &wg| {
                    b.iter(|| {
                        backend.execute_compute(shader, [wg, 1, 1])
                            .expect("Compute failed");
                        black_box(&backend);
                    });
                }
            );
        }

        group.finish();
    }

    /// Benchmark 5: Vector addition (GPU vs CPU)
    fn bench_vector_addition_comparison(c: &mut Criterion) {
        let vulkan_backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        let cpu_backend = CPUBackend::new();

        let shader = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                data[index] = data[index] + 1.0;
            }
        "#;

        let mut group = c.benchmark_group("vector_addition");

        for size in [1024u32, 10_240, 102_400].iter() {
            let workgroups = (size + 255) / 256;

            // GPU benchmark
            group.bench_with_input(
                BenchmarkId::new("gpu", size),
                size,
                |b, _| {
                    b.iter(|| {
                        vulkan_backend.execute_compute(shader, [workgroups, 1, 1])
                            .expect("GPU compute failed");
                        black_box(&vulkan_backend);
                    });
                }
            );

            // CPU benchmark
            group.bench_with_input(
                BenchmarkId::new("cpu", size),
                size,
                |b, &sz| {
                    let mut data = vec![1.0f32; sz as usize];
                    b.iter(|| {
                        for x in data.iter_mut() {
                            *x += 1.0;
                        }
                        black_box(&data);
                    });
                }
            );
        }

        group.finish();
    }

    /// Benchmark 6: WGSL→SPIR-V transpilation overhead
    fn bench_wgsl_transpilation(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let shaders = vec![
            ("simple", r#"
                @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    data[id.x] = data[id.x] * 2.0;
                }
            "#),
            ("complex", r#"
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    let val = input[i];
                    output[i] = sqrt(val * val + 1.0) * sin(val);
                }
            "#),
        ];

        let mut group = c.benchmark_group("wgsl_transpilation");

        for (name, shader) in shaders.iter() {
            group.bench_with_input(
                BenchmarkId::new("vulkan", name),
                shader,
                |b, &shader_src| {
                    b.iter(|| {
                        // Transpilation happens inside execute_compute
                        backend.execute_compute(shader_src, [1, 1, 1])
                            .expect("Transpilation failed");
                        black_box(&backend);
                    });
                }
            );
        }

        group.finish();
    }

    /// Benchmark 7: Synchronization overhead
    fn bench_synchronization(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        c.bench_function("synchronization/vulkan", |b| {
            b.iter(|| {
                backend.synchronize().expect("Sync failed");
                black_box(&backend);
            });
        });
    }

    /// Benchmark 8: Memory bandwidth (theoretical vs achieved)
    fn bench_memory_bandwidth(c: &mut Criterion) {
        let backend = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let size = 100 * 1024 * 1024u64; // 100 MB
        let mut buffer = backend.create_buffer(size, BufferUsage::Storage)
            .expect("Large buffer creation failed");

        let data = vec![0u8; size as usize];

        let mut group = c.benchmark_group("memory_bandwidth");
        group.throughput(Throughput::Bytes(size));

        // Write bandwidth
        group.bench_function("write_100mb", |b| {
            b.iter(|| {
                backend.write_buffer(buffer.as_mut(), &data)
                    .expect("Write failed");
                black_box(&buffer);
            });
        });

        // Read bandwidth
        group.bench_function("read_100mb", |b| {
            b.iter(|| {
                let result = backend.read_buffer(buffer.as_ref())
                    .expect("Read failed");
                black_box(result);
            });
        });

        group.finish();
    }

    criterion_group!(
        benches,
        bench_buffer_allocation,
        bench_buffer_write,
        bench_buffer_read,
        bench_compute_dispatch,
        bench_vector_addition_comparison,
        bench_wgsl_transpilation,
        bench_synchronization,
        bench_memory_bandwidth,
    );

    criterion_main!(benches);
}

#[cfg(not(feature = "vulkan-backend"))]
fn main() {
    println!("Vulkan benchmarks require 'vulkan-backend' feature");
    println!("Run with: cargo bench --features vulkan-backend");
}
