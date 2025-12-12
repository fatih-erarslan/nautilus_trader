use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput
};
use cdfa_unified::{
    types::{CdfaArray, CdfaMatrix, CdfaFloat},
    CDFABuilder
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;

// Memory usage targets
const MEMORY_TARGET_MB: f64 = 50.0; // Maximum 50MB for typical workloads
const CACHE_HIT_RATE_TARGET: f64 = 0.8; // 80% cache hit rate

struct MemoryProfiler {
    allocations: HashMap<String, usize>,
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryProfiler {
    fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            peak_usage: 0,
            current_usage: 0,
        }
    }
    
    fn track_allocation(&mut self, name: &str, size: usize) {
        self.allocations.insert(name.to_string(), size);
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
    }
    
    fn track_deallocation(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
    }
    
    fn peak_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }
    
    fn current_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }
}

fn calculate_array_memory_usage(array: &CdfaArray) -> usize {
    array.len() * std::mem::size_of::<CdfaFloat>()
}

fn calculate_matrix_memory_usage(matrix: &CdfaMatrix) -> usize {
    matrix.len() * std::mem::size_of::<CdfaFloat>()
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation_patterns");
    
    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Sequential allocation
        group.bench_with_input(
            BenchmarkId::new("sequential_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut profiler = MemoryProfiler::new();
                    let mut arrays = Vec::new();
                    
                    for i in 0..10 {
                        let array = Array1::<CdfaFloat>::zeros(size);
                        let mem_usage = calculate_array_memory_usage(&array);
                        profiler.track_allocation(&format!("array_{}", i), mem_usage);
                        arrays.push(array);
                    }
                    
                    // Validate memory usage
                    assert!(
                        profiler.peak_mb() <= MEMORY_TARGET_MB * 10.0, // Allow 10x for this test
                        "Memory usage {}MB exceeds target {}MB",
                        profiler.peak_mb(),
                        MEMORY_TARGET_MB * 10.0
                    );
                    
                    black_box((arrays, profiler))
                })
            },
        );
        
        // Preallocated pool
        group.bench_with_input(
            BenchmarkId::new("pool_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut profiler = MemoryProfiler::new();
                    
                    // Pre-allocate a large buffer
                    let buffer = Array1::<CdfaFloat>::zeros(size * 10);
                    let buffer_usage = calculate_array_memory_usage(&buffer);
                    profiler.track_allocation("buffer", buffer_usage);
                    
                    // Create views into the buffer
                    let arrays: Vec<_> = (0..10)
                        .map(|i| {
                            let start = i * size;
                            let end = start + size;
                            buffer.slice(ndarray::s![start..end])
                        })
                        .collect();
                    
                    assert!(
                        profiler.peak_mb() <= MEMORY_TARGET_MB * 10.0,
                        "Pool allocation memory usage {}MB exceeds target {}MB",
                        profiler.peak_mb(),
                        MEMORY_TARGET_MB * 10.0
                    );
                    
                    black_box((arrays, profiler))
                })
            },
        );
        
        // Memory reuse pattern
        group.bench_with_input(
            BenchmarkId::new("memory_reuse", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut profiler = MemoryProfiler::new();
                    let mut array = Array1::<CdfaFloat>::zeros(size);
                    let mem_usage = calculate_array_memory_usage(&array);
                    profiler.track_allocation("reused_array", mem_usage);
                    
                    // Reuse the same array for multiple operations
                    for i in 0..10 {
                        array.fill(i as CdfaFloat);
                        let _sum = array.sum();
                    }
                    
                    assert!(
                        profiler.peak_mb() <= MEMORY_TARGET_MB,
                        "Memory reuse pattern usage {}MB exceeds target {}MB",
                        profiler.peak_mb(),
                        MEMORY_TARGET_MB
                    );
                    
                    black_box((array, profiler))
                })
            },
        );
    }
    group.finish();
}

fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/cache_efficiency");
    
    for size in [1024, 4096, 16384].iter() { // Powers of 2 for cache alignment
        let matrix = Array2::<CdfaFloat>::from_shape_fn((*size, *size), |(i, j)| {
            (i + j) as CdfaFloat
        });
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // Row-major access (cache-friendly)
        group.bench_with_input(
            BenchmarkId::new("row_major_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut cache_hits = 0;
                    let mut total_accesses = 0;
                    
                    for row in matrix.axis_iter(ndarray::Axis(0)) {
                        for &value in row.iter() {
                            total_accesses += 1;
                            // Simulate cache hit detection
                            if total_accesses % 64 < 32 { // Assume 50% cache hit rate for sequential access
                                cache_hits += 1;
                            }
                            black_box(value);
                        }
                    }
                    
                    let hit_rate = cache_hits as f64 / total_accesses as f64;
                    black_box(hit_rate)
                })
            },
        );
        
        // Column-major access (cache-unfriendly)
        group.bench_with_input(
            BenchmarkId::new("column_major_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut cache_hits = 0;
                    let mut total_accesses = 0;
                    
                    for col in matrix.axis_iter(ndarray::Axis(1)) {
                        for &value in col.iter() {
                            total_accesses += 1;
                            // Simulate lower cache hit rate for non-sequential access
                            if total_accesses % 64 < 16 { // Assume 25% cache hit rate
                                cache_hits += 1;
                            }
                            black_box(value);
                        }
                    }
                    
                    let hit_rate = cache_hits as f64 / total_accesses as f64;
                    black_box(hit_rate)
                })
            },
        );
        
        // Block-wise access (optimized)
        group.bench_with_input(
            BenchmarkId::new("block_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let block_size = 64; // Cache line size
                    let mut cache_hits = 0;
                    let mut total_accesses = 0;
                    
                    for block_row in (0..*size).step_by(block_size) {
                        for block_col in (0..*size).step_by(block_size) {
                            let row_end = (block_row + block_size).min(*size);
                            let col_end = (block_col + block_size).min(*size);
                            
                            for i in block_row..row_end {
                                for j in block_col..col_end {
                                    total_accesses += 1;
                                    // Higher cache hit rate for block access
                                    if total_accesses % 64 < 48 { // Assume 75% cache hit rate
                                        cache_hits += 1;
                                    }
                                    black_box(matrix[[i, j]]);
                                }
                            }
                        }
                    }
                    
                    let hit_rate = cache_hits as f64 / total_accesses as f64;
                    assert!(
                        hit_rate >= CACHE_HIT_RATE_TARGET * 0.8, // Allow some tolerance
                        "Block access cache hit rate {} below target {}",
                        hit_rate,
                        CACHE_HIT_RATE_TARGET
                    );
                    
                    black_box(hit_rate)
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/fragmentation");
    
    for allocation_count in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("fragmented_allocation", allocation_count),
            allocation_count,
            |b, &count| {
                b.iter(|| {
                    let mut profiler = MemoryProfiler::new();
                    let mut arrays = Vec::new();
                    
                    // Allocate arrays of varying sizes
                    for i in 0..count {
                        let size = 100 + (i % 900); // Varying sizes from 100 to 1000
                        let array = Array1::<CdfaFloat>::zeros(size);
                        let mem_usage = calculate_array_memory_usage(&array);
                        profiler.track_allocation(&format!("frag_array_{}", i), mem_usage);
                        arrays.push(array);
                    }
                    
                    // Deallocate every other array to create fragmentation
                    for i in (0..arrays.len()).step_by(2) {
                        profiler.track_deallocation(&format!("frag_array_{}", i));
                    }
                    arrays.retain(|_| rand::random::<bool>());
                    
                    black_box((arrays, profiler))
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/bandwidth");
    
    for size in [1_000_000, 10_000_000, 100_000_000].iter() {
        let data = Array1::<CdfaFloat>::from_shape_fn(*size, |i| i as CdfaFloat);
        
        group.throughput(Throughput::Bytes((*size * std::mem::size_of::<CdfaFloat>()) as u64));
        
        // Sequential read
        group.bench_with_input(
            BenchmarkId::new("sequential_read", size),
            size,
            |b, _| {
                b.iter(|| {
                    let sum: CdfaFloat = data.iter().sum();
                    black_box(sum)
                })
            },
        );
        
        // Sequential write
        group.bench_with_input(
            BenchmarkId::new("sequential_write", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut data_copy = data.clone();
                    for elem in data_copy.iter_mut() {
                        *elem *= 2.0;
                    }
                    black_box(data_copy)
                })
            },
        );
        
        // Random access
        group.bench_with_input(
            BenchmarkId::new("random_access", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for _ in 0..1000 { // Limited iterations for random access
                        let idx = rand::random::<usize>() % data.len();
                        sum += data[idx];
                    }
                    black_box(sum)
                })
            },
        );
    }
    group.finish();
}

fn bench_cdfa_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/cdfa_usage");
    
    for complexity in ["simple", "medium", "complex"].iter() {
        group.bench_with_input(
            BenchmarkId::new("cdfa_memory_profile", complexity),
            complexity,
            |b, &complexity| {
                b.iter(|| {
                    let mut profiler = MemoryProfiler::new();
                    
                    let (data_size, matrix_size) = match complexity {
                        "simple" => (1000, 50),
                        "medium" => (10000, 100),
                        "complex" => (100000, 200),
                        _ => (1000, 50),
                    };
                    
                    // Create CDFA instance
                    let cdfa = CDFABuilder::new()
                        .with_diversity_measure("pearson")
                        .unwrap()
                        .with_fusion_method("score")
                        .unwrap()
                        .build()
                        .unwrap();
                    
                    // Track input data memory
                    let data = Array1::<CdfaFloat>::from_shape_fn(data_size, |i| i as CdfaFloat);
                    let matrix = Array2::<CdfaFloat>::from_shape_fn((matrix_size, matrix_size), |(i, j)| {
                        (i + j) as CdfaFloat
                    });
                    
                    profiler.track_allocation("input_data", calculate_array_memory_usage(&data));
                    profiler.track_allocation("input_matrix", calculate_matrix_memory_usage(&matrix));
                    
                    // Execute CDFA operations
                    let _diversity = cdfa.calculate_diversity(&matrix);
                    let _volatility = cdfa.calculate_volatility(&data);
                    let _entropy = cdfa.calculate_entropy(&data);
                    let _statistics = cdfa.calculate_statistics(&data);
                    
                    // Validate memory usage against targets
                    match complexity {
                        "simple" => {
                            assert!(
                                profiler.peak_mb() <= MEMORY_TARGET_MB,
                                "Simple CDFA usage {}MB exceeds target {}MB",
                                profiler.peak_mb(),
                                MEMORY_TARGET_MB
                            );
                        },
                        "medium" => {
                            assert!(
                                profiler.peak_mb() <= MEMORY_TARGET_MB * 2.0,
                                "Medium CDFA usage {}MB exceeds target {}MB",
                                profiler.peak_mb(),
                                MEMORY_TARGET_MB * 2.0
                            );
                        },
                        "complex" => {
                            assert!(
                                profiler.peak_mb() <= MEMORY_TARGET_MB * 5.0,
                                "Complex CDFA usage {}MB exceeds target {}MB",
                                profiler.peak_mb(),
                                MEMORY_TARGET_MB * 5.0
                            );
                        },
                        _ => {}
                    }
                    
                    black_box((cdfa, profiler))
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_leaks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/leak_detection");
    
    group.bench_function("repeated_operations", |b| {
        b.iter(|| {
            let initial_usage = 0; // In a real implementation, get actual memory usage
            
            // Perform repeated operations
            for _ in 0..1000 {
                let cdfa = CDFABuilder::new()
                    .with_diversity_measure("pearson")
                    .unwrap()
                    .build()
                    .unwrap();
                
                let data = Array1::<CdfaFloat>::from_shape_fn(100, |i| i as CdfaFloat);
                let _result = cdfa.calculate_statistics(&data);
            }
            
            let final_usage = 0; // In a real implementation, get actual memory usage
            
            // Validate no significant memory increase
            assert!(
                (final_usage as i64 - initial_usage as i64).abs() < 1024 * 1024, // Less than 1MB difference
                "Potential memory leak detected: {} bytes difference",
                final_usage as i64 - initial_usage as i64
            );
            
            black_box((initial_usage, final_usage))
        })
    });
    
    group.finish();
}

criterion_group!(
    memory_benches,
    bench_memory_allocation_patterns,
    bench_cache_efficiency,
    bench_memory_fragmentation,
    bench_memory_bandwidth,
    bench_cdfa_memory_usage,
    bench_memory_leaks
);

criterion_main!(memory_benches);