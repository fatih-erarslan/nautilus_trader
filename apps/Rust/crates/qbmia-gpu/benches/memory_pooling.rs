//! Memory Pool GPU Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_gpu::{
    memory::{initialize_pool, get_pool, PoolConfig},
    initialize,
};

fn benchmark_memory_allocation(c: &mut Criterion) {
    let _ = initialize();
    let _ = initialize_pool(PoolConfig::default());
    
    let mut group = c.benchmark_group("memory_allocation");
    
    for &size in [1024, 4096, 16384, 65536, 262144].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocate", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    if let Ok(pool) = get_pool() {
                        let result = pool.allocate(0, black_box(size));
                        black_box(result)
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_pool_performance(c: &mut Criterion) {
    let _ = initialize();
    let _ = initialize_pool(PoolConfig::default());
    
    c.bench_function("allocation_deallocation_cycle", |b| {
        b.iter(|| {
            if let Ok(pool) = get_pool() {
                let handles: Vec<_> = (0..black_box(100))
                    .filter_map(|_| pool.allocate(0, 4096).ok())
                    .collect();
                
                // Free all allocations
                for handle in handles {
                    let _ = pool.free(handle);
                }
            }
        });
    });
}

fn benchmark_fragmentation_handling(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("fragmentation");
    
    for &pattern in ["sequential", "random", "mixed"].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocation_pattern", pattern),
            pattern,
            |b, &pattern| {
                b.iter(|| {
                    let config = PoolConfig {
                        initial_size: 1024 * 1024, // 1MB
                        auto_defrag: true,
                        defrag_threshold: 0.3,
                        ..Default::default()
                    };
                    
                    if initialize_pool(config).is_ok() {
                        if let Ok(pool) = get_pool() {
                            let mut handles = Vec::new();
                            
                            match black_box(pattern) {
                                "sequential" => {
                                    for i in 0..50 {
                                        if let Ok(handle) = pool.allocate(0, 1024) {
                                            handles.push(handle);
                                        }
                                    }
                                }
                                "random" => {
                                    for i in 0..50 {
                                        let size = 512 + (i * 37) % 1536; // Pseudo-random sizes
                                        if let Ok(handle) = pool.allocate(0, size) {
                                            handles.push(handle);
                                        }
                                    }
                                }
                                "mixed" => {
                                    for i in 0..50 {
                                        if i % 3 == 0 && !handles.is_empty() {
                                            // Free some allocations
                                            let handle = handles.remove(0);
                                            let _ = pool.free(handle);
                                        } else {
                                            if let Ok(handle) = pool.allocate(0, 1024) {
                                                handles.push(handle);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                            
                            // Clean up
                            for handle in handles {
                                let _ = pool.free(handle);
                            }
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_large_allocations(c: &mut Criterion) {
    let _ = initialize();
    
    let mut group = c.benchmark_group("large_allocations");
    
    for &size in [1024*1024, 4*1024*1024, 16*1024*1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("large_alloc", format!("{}MB", size / (1024*1024))),
            &size,
            |b, &size| {
                b.iter(|| {
                    let config = PoolConfig {
                        initial_size: size * 2, // Ensure pool is large enough
                        max_size: size * 4,
                        ..Default::default()
                    };
                    
                    if initialize_pool(config).is_ok() {
                        if let Ok(pool) = get_pool() {
                            let result = pool.allocate(0, black_box(size));
                            if let Ok(handle) = result {
                                let _ = pool.free(handle);
                            }
                            black_box(result)
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_access(c: &mut Criterion) {
    let _ = initialize();
    let _ = initialize_pool(PoolConfig::default());
    
    c.bench_function("concurrent_allocations", |b| {
        b.iter(|| {
            use std::sync::Arc;
            use std::thread;
            
            if let Ok(pool) = get_pool() {
                let pool = Arc::new(pool);
                let mut handles = Vec::new();
                
                // Simulate concurrent access
                for _ in 0..black_box(4) {
                    let pool_clone = pool.clone();
                    let handle = thread::spawn(move || {
                        let mut local_handles = Vec::new();
                        for _ in 0..10 {
                            if let Ok(handle) = pool_clone.allocate(0, 4096) {
                                local_handles.push(handle);
                            }
                        }
                        local_handles
                    });
                    handles.push(handle);
                }
                
                // Wait for all threads and collect results
                let all_handles: Vec<_> = handles
                    .into_iter()
                    .filter_map(|h| h.join().ok())
                    .flatten()
                    .collect();
                
                // Clean up
                for handle in all_handles {
                    let _ = pool.free(handle);
                }
            }
        });
    });
}

fn benchmark_memory_pressure(c: &mut Criterion) {
    let _ = initialize();
    
    c.bench_function("memory_pressure_handling", |b| {
        b.iter(|| {
            let config = PoolConfig {
                initial_size: 512 * 1024, // Small initial pool
                max_size: 2 * 1024 * 1024, // Limited max size
                auto_defrag: true,
                ..Default::default()
            };
            
            if initialize_pool(config).is_ok() {
                if let Ok(pool) = get_pool() {
                    let mut handles = Vec::new();
                    
                    // Allocate until we hit memory pressure
                    for _ in 0..black_box(200) {
                        match pool.allocate(0, 8192) {
                            Ok(handle) => handles.push(handle),
                            Err(_) => break, // Hit memory limit
                        }
                    }
                    
                    // Free half the allocations
                    for _ in 0..handles.len()/2 {
                        if let Some(handle) = handles.pop() {
                            let _ = pool.free(handle);
                        }
                    }
                    
                    // Try to allocate more
                    for _ in 0..50 {
                        if let Ok(handle) = pool.allocate(0, 8192) {
                            handles.push(handle);
                        }
                    }
                    
                    // Clean up remaining
                    for handle in handles {
                        let _ = pool.free(handle);
                    }
                }
            }
        });
    });
}

fn benchmark_pool_statistics(c: &mut Criterion) {
    let _ = initialize();
    let _ = initialize_pool(PoolConfig::default());
    
    c.bench_function("memory_statistics", |b| {
        b.iter(|| {
            if let Ok(pool) = get_pool() {
                // Allocate some memory
                let handles: Vec<_> = (0..black_box(50))
                    .filter_map(|_| pool.allocate(0, 4096).ok())
                    .collect();
                
                // Get statistics
                let stats = pool.stats();
                black_box(stats);
                
                // Clean up
                for handle in handles {
                    let _ = pool.free(handle);
                }
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_memory_allocation,
    benchmark_memory_pool_performance,
    benchmark_fragmentation_handling,
    benchmark_large_allocations,
    benchmark_concurrent_access,
    benchmark_memory_pressure,
    benchmark_pool_statistics
);
criterion_main!(benches);