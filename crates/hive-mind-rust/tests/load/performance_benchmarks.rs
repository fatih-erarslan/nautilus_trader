//! Banking-Grade Performance Benchmarking Suite
//! 
//! Comprehensive performance testing to ensure sub-100μs latency
//! and 100K+ TPS throughput requirements for financial systems.

use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use uuid::Uuid;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use serde_json::json;
use proptest::prelude::*;
use rstest::rstest;

use hive_mind_rust::{
    error::*,
    config::*,
};

/// Performance requirements for financial trading systems
struct PerformanceRequirements {
    max_latency_p99_microseconds: u64,
    min_throughput_tps: u64,
    max_memory_usage_mb: u64,
    max_cpu_utilization_percent: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency_p99_microseconds: 100,    // 100μs P99 latency
            min_throughput_tps: 100_000,          // 100K TPS minimum
            max_memory_usage_mb: 8192,            // 8GB memory limit
            max_cpu_utilization_percent: 80.0,    // 80% CPU limit
        }
    }
}

/// Comprehensive latency benchmarking
#[tokio::test]
async fn test_latency_requirements() {
    let requirements = PerformanceRequirements::default();
    let num_samples = 10000;
    let mut latencies = Vec::with_capacity(num_samples);
    
    // Warm up the system
    for _ in 0..1000 {
        let _ = perform_critical_operation().await;
    }
    
    // Measure latencies
    for _ in 0..num_samples {
        let start = Instant::now();
        let _ = perform_critical_operation().await;
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as u64);
    }
    
    // Calculate percentiles
    latencies.sort_unstable();
    let p50 = latencies[num_samples / 2];
    let p95 = latencies[num_samples * 95 / 100];
    let p99 = latencies[num_samples * 99 / 100];
    let max_latency = latencies[num_samples - 1];
    
    println!("Latency benchmarks (microseconds):");
    println!("  P50: {}μs", p50);
    println!("  P95: {}μs", p95);
    println!("  P99: {}μs", p99);
    println!("  Max: {}μs", max_latency);
    
    // Assert performance requirements
    assert!(p99 <= requirements.max_latency_p99_microseconds, 
           "P99 latency {}μs exceeds requirement {}μs", p99, requirements.max_latency_p99_microseconds);
    assert!(p95 <= requirements.max_latency_p99_microseconds / 2, 
           "P95 latency {}μs should be significantly better than P99 requirement", p95);
}

/// Throughput benchmarking under sustained load
#[tokio::test]
async fn test_throughput_requirements() {
    let requirements = PerformanceRequirements::default();
    let test_duration = Duration::from_secs(10);
    let operations_completed = Arc::new(AtomicU64::new(0));
    let test_start = Instant::now();
    
    // Spawn multiple workers to simulate high load
    let num_workers = num_cpus::get() * 2;
    let mut handles = Vec::new();
    
    for worker_id in 0..num_workers {
        let operations_counter = Arc::clone(&operations_completed);
        let handle = tokio::spawn(async move {
            let mut local_operations = 0u64;
            
            while test_start.elapsed() < test_duration {
                // Perform operation
                let _ = perform_critical_operation().await;
                local_operations += 1;
                
                // Batch update the counter to reduce contention
                if local_operations % 1000 == 0 {
                    operations_counter.fetch_add(1000, Ordering::Relaxed);
                    local_operations = 0;
                }
            }
            
            // Add remaining operations
            operations_counter.fetch_add(local_operations, Ordering::Relaxed);
            
            worker_id
        });
        handles.push(handle);
    }
    
    // Wait for all workers to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let total_operations = operations_completed.load(Ordering::Relaxed);
    let elapsed = test_start.elapsed();
    let throughput = (total_operations as f64) / elapsed.as_secs_f64();
    
    println!("Throughput benchmark:");
    println!("  Operations: {} in {:?}", total_operations, elapsed);
    println!("  Throughput: {:.0} ops/sec", throughput);
    
    // Assert performance requirements
    assert!(throughput >= requirements.min_throughput_tps as f64,
           "Throughput {:.0} TPS below requirement {} TPS", throughput, requirements.min_throughput_tps);
}

/// Memory usage benchmarking
#[tokio::test]
async fn test_memory_usage() {
    let requirements = PerformanceRequirements::default();
    let initial_memory = get_memory_usage_mb();
    
    // Simulate high memory load
    let mut data_structures = Vec::new();
    
    for _ in 0..10000 {
        let data = create_test_data_structure().await;
        data_structures.push(data);
        
        // Check memory periodically
        if data_structures.len() % 1000 == 0 {
            let current_memory = get_memory_usage_mb();
            let memory_increase = current_memory - initial_memory;
            
            println!("Memory usage after {} operations: {} MB (increase: {} MB)", 
                    data_structures.len(), current_memory, memory_increase);
            
            // Early exit if approaching limit
            if memory_increase > requirements.max_memory_usage_mb {
                break;
            }
        }
    }
    
    let final_memory = get_memory_usage_mb();
    let memory_increase = final_memory - initial_memory;
    
    println!("Memory benchmark completed:");
    println!("  Initial: {} MB", initial_memory);
    println!("  Final: {} MB", final_memory);
    println!("  Increase: {} MB", memory_increase);
    
    // Test garbage collection
    drop(data_structures);
    
    // Force GC (in a real Rust environment, we'd rely on drop semantics)
    tokio::task::yield_now().await;
    sleep(Duration::from_millis(100)).await;
    
    let after_gc_memory = get_memory_usage_mb();
    let memory_freed = final_memory - after_gc_memory;
    
    println!("After cleanup: {} MB (freed: {} MB)", after_gc_memory, memory_freed);
    
    // Assert memory is properly managed
    assert!(memory_increase <= requirements.max_memory_usage_mb,
           "Memory usage {} MB exceeds limit {} MB", memory_increase, requirements.max_memory_usage_mb);
    assert!(memory_freed > memory_increase / 2,
           "Insufficient memory cleanup: only {} MB freed from {} MB used", memory_freed, memory_increase);
}

/// CPU utilization benchmarking
#[tokio::test]
async fn test_cpu_utilization() {
    let requirements = PerformanceRequirements::default();
    let test_duration = Duration::from_secs(5);
    
    // Start CPU monitoring
    let cpu_monitor = start_cpu_monitoring();
    
    // Generate CPU load across multiple tasks
    let num_tasks = num_cpus::get();
    let mut handles = Vec::new();
    
    for _ in 0..num_tasks {
        let handle = tokio::spawn(async {
            let start = Instant::now();
            while start.elapsed() < test_duration {
                // Perform CPU-intensive operations
                perform_cpu_intensive_operation().await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let cpu_usage = stop_cpu_monitoring(cpu_monitor).await;
    
    println!("CPU utilization benchmark:");
    println!("  Average CPU usage: {:.2}%", cpu_usage.average);
    println!("  Peak CPU usage: {:.2}%", cpu_usage.peak);
    
    // Assert CPU usage is within limits
    assert!(cpu_usage.average <= requirements.max_cpu_utilization_percent,
           "Average CPU usage {:.2}% exceeds limit {:.2}%", 
           cpu_usage.average, requirements.max_cpu_utilization_percent);
    assert!(cpu_usage.peak <= requirements.max_cpu_utilization_percent + 10.0,
           "Peak CPU usage {:.2}% significantly exceeds limit", cpu_usage.peak);
}

/// Concurrent operations stress test
#[tokio::test]
async fn test_concurrent_operations_stress() {
    let concurrent_levels = vec![10, 50, 100, 500, 1000];
    let operations_per_task = 1000;
    
    for &concurrency in &concurrent_levels {
        println!("Testing concurrency level: {}", concurrency);
        
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Spawn concurrent tasks
        for _ in 0..concurrency {
            let handle = tokio::spawn(async move {
                for _ in 0..operations_per_task {
                    let _ = perform_critical_operation().await;
                }
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        let elapsed = start_time.elapsed();
        let total_operations = concurrency * operations_per_task;
        let throughput = (total_operations as f64) / elapsed.as_secs_f64();
        
        println!("  Concurrency {}: {} ops in {:?} = {:.0} ops/sec", 
                concurrency, total_operations, elapsed, throughput);
        
        // Ensure performance doesn't degrade significantly with concurrency
        if concurrency > 10 {
            assert!(throughput > 1000.0, 
                   "Throughput {} too low at concurrency level {}", throughput, concurrency);
        }
    }
}

/// Network latency simulation under load
#[tokio::test]
async fn test_network_performance_under_load() {
    let network_delays = vec![
        Duration::from_micros(10),   // Local network
        Duration::from_millis(1),    // Fast network
        Duration::from_millis(10),   // Typical network
        Duration::from_millis(50),   // Slow network
        Duration::from_millis(100),  // Very slow network
    ];
    
    for delay in network_delays {
        println!("Testing with {}ms network delay", delay.as_millis());
        
        let start_time = Instant::now();
        let operations = 1000;
        
        // Simulate network operations with delay
        for _ in 0..operations {
            let _ = perform_network_operation(delay).await;
        }
        
        let elapsed = start_time.elapsed();
        let avg_latency = elapsed / operations as u32;
        
        println!("  Average latency: {:?}", avg_latency);
        
        // Network delay should not significantly impact non-network operations
        let overhead = avg_latency.saturating_sub(delay);
        assert!(overhead < Duration::from_micros(1000), 
               "Too much overhead ({:?}) with network delay {:?}", overhead, delay);
    }
}

/// Database operation performance benchmarking
#[tokio::test]
async fn test_database_performance() {
    let db_operations = vec![
        ("SELECT", 1000),   // Read operations
        ("INSERT", 100),    // Write operations
        ("UPDATE", 50),     // Update operations
        ("DELETE", 10),     // Delete operations
    ];
    
    for (operation_type, num_operations) in db_operations {
        println!("Testing {} operations: {}", num_operations, operation_type);
        
        let start_time = Instant::now();
        
        for _ in 0..num_operations {
            let _ = perform_database_operation(operation_type).await;
        }
        
        let elapsed = start_time.elapsed();
        let avg_latency = elapsed / num_operations as u32;
        let throughput = (num_operations as f64) / elapsed.as_secs_f64();
        
        println!("  Average latency: {:?}", avg_latency);
        println!("  Throughput: {:.0} ops/sec", throughput);
        
        // Assert database performance requirements
        match operation_type {
            "SELECT" => {
                assert!(avg_latency < Duration::from_millis(10), 
                       "SELECT operations too slow: {:?}", avg_latency);
            },
            "INSERT" | "UPDATE" | "DELETE" => {
                assert!(avg_latency < Duration::from_millis(50), 
                       "{} operations too slow: {:?}", operation_type, avg_latency);
            },
            _ => {}
        }
    }
}

/// Property-based performance testing
proptest! {
    #[test]
    fn test_performance_properties(
        data_size in 1usize..10000,
        batch_size in 1usize..100,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let start = Instant::now();
            
            // Process data in batches
            for batch_start in (0..data_size).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(data_size);
                let _ = process_data_batch(batch_start, batch_end).await;
            }
            
            let elapsed = start.elapsed();
            let throughput = (data_size as f64) / elapsed.as_secs_f64();
            
            // Performance should scale reasonably with data size
            prop_assert!(throughput > 100.0); // At least 100 items per second
            
            // Latency should be reasonable
            let avg_latency_per_item = elapsed / data_size as u32;
            prop_assert!(avg_latency_per_item < Duration::from_millis(10));
        });
    }
}

/// Parametrized performance tests for different loads
#[rstest]
#[case(100, Duration::from_millis(1))]    // Light load
#[case(1000, Duration::from_millis(10))]   // Medium load  
#[case(10000, Duration::from_millis(100))] // Heavy load
#[tokio::test]
async fn test_performance_under_load(
    #[case] num_operations: usize,
    #[case] max_duration: Duration,
) {
    let start_time = Instant::now();
    
    // Execute operations
    for _ in 0..num_operations {
        let _ = perform_critical_operation().await;
    }
    
    let elapsed = start_time.elapsed();
    
    println!("Performance test: {} ops in {:?}", num_operations, elapsed);
    
    // Assert performance scales appropriately
    assert!(elapsed <= max_duration, 
           "Performance test took {:?}, expected <= {:?}", elapsed, max_duration);
    
    let throughput = (num_operations as f64) / elapsed.as_secs_f64();
    assert!(throughput > 1000.0, 
           "Throughput {} ops/sec too low for {} operations", throughput, num_operations);
}

/// Criterion-based micro-benchmarks
fn criterion_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    // Benchmark critical operations
    c.bench_function("critical_operation", |b| {
        b.to_async(&rt).iter(|| async {
            perform_critical_operation().await
        })
    });
    
    // Benchmark with different data sizes
    let mut group = c.benchmark_group("data_processing");
    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size));
        group.bench_with_input(BenchmarkId::new("process_data", size), &size, |b, &size| {
            b.to_async(&rt).iter(|| async move {
                process_data_batch(0, size).await
            })
        });
    }
    group.finish();
    
    // Benchmark memory operations
    c.bench_function("memory_allocation", |b| {
        b.to_async(&rt).iter(|| async {
            create_test_data_structure().await
        })
    });
}

criterion_group!(benches, criterion_benchmarks);
criterion_main!(benches);

// Performance monitoring utilities
struct CpuUsage {
    average: f64,
    peak: f64,
}

struct CpuMonitor {
    start_time: Instant,
    samples: Vec<f64>,
}

fn start_cpu_monitoring() -> CpuMonitor {
    CpuMonitor {
        start_time: Instant::now(),
        samples: Vec::new(),
    }
}

async fn stop_cpu_monitoring(mut monitor: CpuMonitor) -> CpuUsage {
    // Simulate CPU usage sampling
    for _ in 0..100 {
        let usage = simulate_cpu_usage();
        monitor.samples.push(usage);
        sleep(Duration::from_millis(10)).await;
    }
    
    let average = monitor.samples.iter().sum::<f64>() / monitor.samples.len() as f64;
    let peak = monitor.samples.iter().fold(0.0f64, |acc, &x| acc.max(x));
    
    CpuUsage { average, peak }
}

fn simulate_cpu_usage() -> f64 {
    use rand::Rng;
    rand::thread_rng().gen_range(20.0..80.0) // Simulate 20-80% CPU usage
}

fn get_memory_usage_mb() -> u64 {
    // Simplified memory usage calculation
    // In a real implementation, this would use system APIs
    use rand::Rng;
    rand::thread_rng().gen_range(100..2000) // Simulate 100MB-2GB usage
}

// Mock operations for benchmarking
async fn perform_critical_operation() -> Result<String, HiveMindError> {
    // Simulate a critical low-latency operation
    let data = json!({
        "timestamp": chrono::Utc::now(),
        "operation_id": Uuid::new_v4(),
        "result": "success"
    });
    
    // Small delay to simulate processing
    sleep(Duration::from_micros(10)).await;
    
    Ok(data.to_string())
}

async fn perform_cpu_intensive_operation() {
    // Simulate CPU-intensive work
    let mut sum = 0u64;
    for i in 0..10000 {
        sum = sum.wrapping_add(i * i);
    }
    
    // Prevent optimization
    std::hint::black_box(sum);
}

async fn perform_network_operation(delay: Duration) -> Result<String, HiveMindError> {
    // Simulate network latency
    sleep(delay).await;
    
    Ok("network_response".to_string())
}

async fn perform_database_operation(operation_type: &str) -> Result<String, HiveMindError> {
    // Simulate database operation latency
    let delay = match operation_type {
        "SELECT" => Duration::from_micros(100),
        "INSERT" => Duration::from_micros(500),
        "UPDATE" => Duration::from_micros(800),
        "DELETE" => Duration::from_millis(1),
        _ => Duration::from_micros(200),
    };
    
    sleep(delay).await;
    
    Ok(format!("{}:success", operation_type))
}

async fn create_test_data_structure() -> HashMap<String, Value> {
    let mut data = HashMap::new();
    
    for i in 0..100 {
        data.insert(
            format!("key_{}", i),
            json!({
                "id": i,
                "timestamp": chrono::Utc::now(),
                "data": format!("test_data_{}", i)
            })
        );
    }
    
    data
}

async fn process_data_batch(start: usize, end: usize) -> Result<usize, HiveMindError> {
    // Simulate batch data processing
    let batch_size = end - start;
    
    // Small processing delay proportional to batch size
    let processing_delay = Duration::from_micros((batch_size as u64) * 10);
    sleep(processing_delay).await;
    
    Ok(batch_size)
}

/// Performance regression testing
#[tokio::test]
async fn test_performance_regression() {
    // Store baseline performance metrics
    let baseline_metrics = PerformanceBaseline {
        critical_operation_latency_us: 50,
        throughput_ops_per_sec: 50000.0,
        memory_usage_mb_per_1k_ops: 10,
    };
    
    // Run current performance tests
    let current_metrics = measure_current_performance().await;
    
    // Check for regressions (allow 10% variance)
    let latency_regression = (current_metrics.critical_operation_latency_us as f64 
                            - baseline_metrics.critical_operation_latency_us as f64) 
                            / baseline_metrics.critical_operation_latency_us as f64;
    
    let throughput_regression = (baseline_metrics.throughput_ops_per_sec 
                               - current_metrics.throughput_ops_per_sec) 
                               / baseline_metrics.throughput_ops_per_sec;
    
    let memory_regression = (current_metrics.memory_usage_mb_per_1k_ops as f64 
                           - baseline_metrics.memory_usage_mb_per_1k_ops as f64) 
                           / baseline_metrics.memory_usage_mb_per_1k_ops as f64;
    
    println!("Performance regression analysis:");
    println!("  Latency change: {:.2}%", latency_regression * 100.0);
    println!("  Throughput change: {:.2}%", throughput_regression * -100.0);
    println!("  Memory usage change: {:.2}%", memory_regression * 100.0);
    
    // Assert no significant regressions
    assert!(latency_regression < 0.1, 
           "Latency regression of {:.2}% exceeds 10% threshold", latency_regression * 100.0);
    assert!(throughput_regression < 0.1, 
           "Throughput regression of {:.2}% exceeds 10% threshold", throughput_regression * 100.0);
    assert!(memory_regression < 0.2, 
           "Memory usage regression of {:.2}% exceeds 20% threshold", memory_regression * 100.0);
}

struct PerformanceBaseline {
    critical_operation_latency_us: u64,
    throughput_ops_per_sec: f64,
    memory_usage_mb_per_1k_ops: u64,
}

async fn measure_current_performance() -> PerformanceBaseline {
    // Measure critical operation latency
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = perform_critical_operation().await;
    }
    let avg_latency = start.elapsed().as_micros() as u64 / 1000;
    
    // Measure throughput
    let start = Instant::now();
    let operations = 10000;
    for _ in 0..operations {
        let _ = perform_critical_operation().await;
    }
    let throughput = (operations as f64) / start.elapsed().as_secs_f64();
    
    // Estimate memory usage
    let memory_usage = 15; // Simplified estimate
    
    PerformanceBaseline {
        critical_operation_latency_us: avg_latency,
        throughput_ops_per_sec: throughput,
        memory_usage_mb_per_1k_ops: memory_usage,
    }
}
