//! Performance benchmarks for Metal backend
//!
//! Compares Metal performance against CPU baseline and measures:
//! - Memory allocation throughput
//! - Kernel compilation time
//! - Compute kernel execution speed
//! - Memory transfer bandwidth

#[cfg(all(feature = "metal-backend", target_os = "macos"))]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use hyperphysics_gpu::backend::{GPUBackend, BufferUsage};
    use hyperphysics_gpu::backend::metal::create_metal_backend;
    use hyperphysics_gpu::backend::cpu::CPUBackend;

    fn bench_buffer_allocation(c: &mut Criterion) {
        let mut group = c.benchmark_group("buffer_allocation");

        if let Ok(Some(metal_backend)) = create_metal_backend() {
            let sizes = vec![1024, 4096, 16384, 65536, 262144, 1048576]; // 1KB to 1MB

            for size in sizes {
                group.bench_with_input(
                    BenchmarkId::new("metal", size),
                    &size,
                    |b, &size| {
                        b.iter(|| {
                            let buffer = metal_backend.create_buffer(size as u64, BufferUsage::Storage);
                            black_box(buffer)
                        })
                    },
                );
            }
        }

        let cpu_backend = CPUBackend::new();
        for size in vec![1024, 4096, 16384, 65536, 262144, 1048576] {
            group.bench_with_input(
                BenchmarkId::new("cpu", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        let buffer = cpu_backend.create_buffer(size as u64, BufferUsage::Storage);
                        black_box(buffer)
                    })
                },
            );
        }

        group.finish();
    }

    fn bench_compute_execution(c: &mut Criterion) {
        let mut group = c.benchmark_group("compute_execution");

        let shader = r#"
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                // Simple compute operation
            }
        "#;

        if let Ok(Some(metal_backend)) = create_metal_backend() {
            let workgroup_sizes = vec![64, 256, 1024, 4096];

            for size in workgroup_sizes {
                group.bench_with_input(
                    BenchmarkId::new("metal", size),
                    &size,
                    |b, &size| {
                        b.iter(|| {
                            let result = metal_backend.execute_compute(shader, [size as u32, 1, 1]);
                            black_box(result)
                        })
                    },
                );
            }
        }

        group.finish();
    }

    fn bench_memory_transfer(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory_transfer");

        if let Ok(Some(metal_backend)) = create_metal_backend() {
            let sizes = vec![1024, 4096, 16384, 65536]; // 1KB to 64KB

            for size in sizes {
                let data = vec![0u8; size];

                group.bench_with_input(
                    BenchmarkId::new("write", size),
                    &size,
                    |b, &_size| {
                        if let Ok(mut buffer) = metal_backend.create_buffer(size as u64, BufferUsage::Storage) {
                            b.iter(|| {
                                let result = metal_backend.write_buffer(buffer.as_mut(), &data);
                                black_box(result)
                            })
                        }
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("read", size),
                    &size,
                    |b, &_size| {
                        if let Ok(buffer) = metal_backend.create_buffer(size as u64, BufferUsage::Storage) {
                            b.iter(|| {
                                let result = metal_backend.read_buffer(buffer.as_ref());
                                black_box(result)
                            })
                        }
                    },
                );
            }
        }

        group.finish();
    }

    fn bench_synchronization(c: &mut Criterion) {
        let mut group = c.benchmark_group("synchronization");

        if let Ok(Some(metal_backend)) = create_metal_backend() {
            group.bench_function("metal_sync", |b| {
                b.iter(|| {
                    let result = metal_backend.synchronize();
                    black_box(result)
                })
            });
        }

        let cpu_backend = CPUBackend::new();
        group.bench_function("cpu_sync", |b| {
            b.iter(|| {
                let result = cpu_backend.synchronize();
                black_box(result)
            })
        });

        group.finish();
    }

    criterion_group!(
        benches,
        bench_buffer_allocation,
        bench_compute_execution,
        bench_memory_transfer,
        bench_synchronization
    );
    criterion_main!(benches);
}

#[cfg(not(all(feature = "metal-backend", target_os = "macos")))]
fn main() {
    println!("Metal benchmarks require macOS and metal-backend feature");
}
