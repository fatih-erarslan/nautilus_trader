//! Vulkan backend performance benchmarks
//!
//! Compares Vulkan compute performance against CPU baseline.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(feature = "vulkan-backend")]
use hyperphysics_gpu::backend::vulkan::create_vulkan_backend;
#[cfg(feature = "vulkan-backend")]
use hyperphysics_gpu::backend::{GPUBackend, BufferUsage};

fn cpu_vector_multiply(data: &mut [f32]) {
    for value in data.iter_mut() {
        *value *= 2.0;
    }
}

#[cfg(feature = "vulkan-backend")]
fn vulkan_buffer_benchmark(c: &mut Criterion) {
    if let Ok(Some(backend)) = create_vulkan_backend() {
        let mut group = c.benchmark_group("vulkan_buffer_operations");

        for size in [1024, 4096, 16384, 65536].iter() {
            let bytes = size * std::mem::size_of::<f32>();

            group.bench_with_input(
                BenchmarkId::new("allocate", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        backend.create_buffer(bytes as u64, BufferUsage::Storage)
                            .expect("Buffer creation failed")
                    })
                },
            );

            // Write benchmark
            let mut buffer = backend.create_buffer(bytes as u64, BufferUsage::Storage)
                .expect("Buffer creation failed");
            let data = vec![1.0f32; *size];
            let data_bytes = bytemuck::cast_slice(&data);

            group.bench_with_input(
                BenchmarkId::new("write", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        backend.write_buffer(buffer.as_mut(), black_box(data_bytes))
                            .expect("Buffer write failed")
                    })
                },
            );

            // Read benchmark
            group.bench_with_input(
                BenchmarkId::new("read", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        backend.read_buffer(buffer.as_ref())
                            .expect("Buffer read failed")
                    })
                },
            );
        }

        group.finish();
    }
}

#[cfg(feature = "vulkan-backend")]
fn vulkan_compute_benchmark(c: &mut Criterion) {
    if let Ok(Some(backend)) = create_vulkan_backend() {
        let mut group = c.benchmark_group("vulkan_compute");

        let shader = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index < arrayLength(&data)) {
                    data[index] = data[index] * 2.0;
                }
            }
        "#;

        for workgroups in [64, 256, 1024, 4096].iter() {
            group.bench_with_input(
                BenchmarkId::new("dispatch", workgroups),
                workgroups,
                |b, &wg| {
                    b.iter(|| {
                        backend.execute_compute(black_box(shader), [wg, 1, 1])
                            .expect("Compute execution failed");
                        backend.synchronize().expect("Sync failed");
                    })
                },
            );
        }

        group.finish();
    }
}

fn cpu_vs_vulkan_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_vulkan");

    for size in [1024, 16384, 65536].iter() {
        // CPU baseline
        let mut cpu_data = vec![1.0f32; *size];
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    cpu_vector_multiply(black_box(&mut cpu_data))
                })
            },
        );

        // Vulkan GPU
        #[cfg(feature = "vulkan-backend")]
        if let Ok(Some(backend)) = create_vulkan_backend() {
            let bytes = size * std::mem::size_of::<f32>();
            let mut buffer = backend.create_buffer(bytes as u64, BufferUsage::Storage)
                .expect("Buffer creation failed");

            let data = vec![1.0f32; *size];
            let data_bytes = bytemuck::cast_slice(&data);
            backend.write_buffer(buffer.as_mut(), data_bytes)
                .expect("Write failed");

            group.bench_with_input(
                BenchmarkId::new("vulkan", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        let shader = r#"
                            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

                            @compute @workgroup_size(256)
                            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                                let index = global_id.x;
                                if (index < arrayLength(&data)) {
                                    data[index] = data[index] * 2.0;
                                }
                            }
                        "#;

                        backend.execute_compute(black_box(shader), [(*size / 256) as u32, 1, 1])
                            .expect("Compute failed");
                        backend.synchronize().expect("Sync failed");
                    })
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "vulkan-backend")]
criterion_group!(
    benches,
    vulkan_buffer_benchmark,
    vulkan_compute_benchmark,
    cpu_vs_vulkan_comparison
);

#[cfg(not(feature = "vulkan-backend"))]
criterion_group!(
    benches,
    cpu_vs_vulkan_comparison
);

criterion_main!(benches);
