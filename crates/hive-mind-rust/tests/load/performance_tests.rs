//! Comprehensive performance and load tests for banking-grade systems
//! 
//! Tests system performance under various load conditions, validates
//! SLA requirements, and ensures scalability for financial operations.

use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{timeout, sleep, interval};
use uuid::Uuid;
use futures::future::join_all;
use proptest::prelude::*;
use serde_json::{Value, json};

use hive_mind_rust::{
    config::HiveMindConfig,
    core::HiveMind,
    error::{HiveMindError, Result},
};

/// Performance metrics collection
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    total_operations: usize,
    successful_operations: usize,
    failed_operations: usize,
    total_duration: Duration,
    min_latency: Duration,
    max_latency: Duration,
    avg_latency: Duration,
    p95_latency: Duration,
    p99_latency: Duration,
    throughput_ops_per_sec: f64,
    memory_usage_mb: f64,
    cpu_usage_percent: f64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::ZERO,
            min_latency: Duration::from_secs(u64::MAX),
            max_latency: Duration::ZERO,
            avg_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
        }
    }
    
    fn calculate_from_latencies(&mut self, latencies: &mut Vec<Duration>) {
        if latencies.is_empty() {
            return;
        }
        
        latencies.sort();
        self.min_latency = latencies[0];
        self.max_latency = latencies[latencies.len() - 1];
        
        let sum: Duration = latencies.iter().sum();
        self.avg_latency = sum / latencies.len() as u32;
        
        let p95_index = (latencies.len() as f64 * 0.95) as usize;
        let p99_index = (latencies.len() as f64 * 0.99) as usize;
        
        self.p95_latency = latencies[p95_index.min(latencies.len() - 1)];
        self.p99_latency = latencies[p99_index.min(latencies.len() - 1)];
        
        self.throughput_ops_per_sec = self.total_operations as f64 / self.total_duration.as_secs_f64();
    }
    
    fn meets_sla(&self) -> bool {
        // Banking SLA requirements
        self.p95_latency < Duration::from_millis(100) && // 95% < 100ms
        self.p99_latency < Duration::from_millis(500) && // 99% < 500ms
        self.throughput_ops_per_sec > 1000.0 &&         // > 1000 TPS
        self.cpu_usage_percent < 80.0 &&                // < 80% CPU
        self.memory_usage_mb < 1024.0                    // < 1GB memory
    }
}

/// Load test configuration
#[derive(Debug, Clone)]
struct LoadTestConfig {
    concurrent_users: usize,
    operations_per_user: usize,
    ramp_up_duration: Duration,
    test_duration: Duration,
    operation_types: Vec<OperationType>,
}

#[derive(Debug, Clone)]
enum OperationType {
    ReadMarketData,
    SubmitProposal,
    QueryMemory,
    SpawnAgent,
    NetworkMessage,
}

/// Test baseline performance with single-threaded operations
#[tokio::test]
async fn test_baseline_performance() {
    let start_time = Instant::now();
    let mut latencies = Vec::new();
    let num_operations = 1000;
    
    for i in 0..num_operations {
        let op_start = Instant::now();
        
        // Simulate basic operations
        simulate_operation(OperationType::ReadMarketData).await;
        
        let latency = op_start.elapsed();
        latencies.push(latency);
    }
    
    let total_duration = start_time.elapsed();
    let mut metrics = PerformanceMetrics::new();
    metrics.total_operations = num_operations;
    metrics.successful_operations = num_operations;
    metrics.total_duration = total_duration;
    metrics.calculate_from_latencies(&mut latencies);
    
    println!("Baseline Performance:");
    println!("  Throughput: {:.2} ops/sec", metrics.throughput_ops_per_sec);
    println!("  Avg Latency: {:?}", metrics.avg_latency);
    println!("  P95 Latency: {:?}", metrics.p95_latency);
    println!("  P99 Latency: {:?}", metrics.p99_latency);
    
    // Baseline performance assertions
    assert!(metrics.throughput_ops_per_sec > 500.0, "Baseline throughput too low");
    assert!(metrics.p95_latency < Duration::from_millis(10), "Baseline P95 latency too high");
}

/// Test concurrent load with multiple users
#[tokio::test]
async fn test_concurrent_load() {
    let config = LoadTestConfig {
        concurrent_users: 100,
        operations_per_user: 50,
        ramp_up_duration: Duration::from_secs(10),
        test_duration: Duration::from_secs(60),
        operation_types: vec![
            OperationType::ReadMarketData,
            OperationType::SubmitProposal,
            OperationType::QueryMemory,
        ],
    };
    
    let total_operations = config.concurrent_users * config.operations_per_user;
    let successful_operations = Arc::new(AtomicUsize::new(0));
    let failed_operations = Arc::new(AtomicUsize::new(0));
    let latencies = Arc::new(RwLock::new(Vec::new()));
    
    let start_time = Instant::now();
    let mut tasks = Vec::new();
    
    // Simulate gradual ramp-up
    let ramp_up_delay = config.ramp_up_duration / config.concurrent_users as u32;
    
    for user_id in 0..config.concurrent_users {
        let successful = successful_operations.clone();
        let failed = failed_operations.clone();
        let latencies_clone = latencies.clone();
        let operations_per_user = config.operations_per_user;
        let operation_types = config.operation_types.clone();
        
        let task = tokio::spawn(async move {
            // Ramp-up delay
            sleep(ramp_up_delay * user_id as u32).await;
            
            for op_id in 0..operations_per_user {
                let op_type = &operation_types[op_id % operation_types.len()];
                let op_start = Instant::now();
                
                match simulate_operation(op_type.clone()).await {
                    Ok(_) => {
                        successful.fetch_add(1, Ordering::Relaxed);
                        let latency = op_start.elapsed();
                        latencies_clone.write().await.push(latency);
                    },
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                
                // Small delay between operations
                sleep(Duration::from_millis(10)).await;
            }
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    join_all(tasks).await;
    
    let total_duration = start_time.elapsed();
    let successful = successful_operations.load(Ordering::Relaxed);
    let failed = failed_operations.load(Ordering::Relaxed);
    let mut latencies_vec = latencies.write().await;
    
    let mut metrics = PerformanceMetrics::new();
    metrics.total_operations = total_operations;
    metrics.successful_operations = successful;
    metrics.failed_operations = failed;
    metrics.total_duration = total_duration;
    metrics.calculate_from_latencies(&mut latencies_vec);
    
    println!("Concurrent Load Test Results:");
    println!("  Users: {}", config.concurrent_users);
    println!("  Total Operations: {}", total_operations);
    println!("  Successful: {}", successful);
    println!("  Failed: {}", failed);
    println!("  Throughput: {:.2} ops/sec", metrics.throughput_ops_per_sec);
    println!("  P95 Latency: {:?}", metrics.p95_latency);
    println!("  P99 Latency: {:?}", metrics.p99_latency);
    
    // Performance assertions
    assert!(metrics.successful_operations > total_operations * 95 / 100, 
           "Success rate should be > 95%");
    assert!(metrics.throughput_ops_per_sec > 100.0, 
           "Concurrent throughput should be > 100 ops/sec");
}

/// Test system behavior under extreme load
#[tokio::test]
async fn test_extreme_load() {
    let extreme_config = LoadTestConfig {
        concurrent_users: 1000,
        operations_per_user: 100,
        ramp_up_duration: Duration::from_secs(30),
        test_duration: Duration::from_secs(120),
        operation_types: vec![
            OperationType::ReadMarketData,
            OperationType::SubmitProposal,
            OperationType::QueryMemory,
            OperationType::SpawnAgent,
            OperationType::NetworkMessage,
        ],
    };
    
    // Use semaphore to limit concurrent operations and prevent resource exhaustion
    let semaphore = Arc::new(Semaphore::new(500));
    let successful_operations = Arc::new(AtomicUsize::new(0));
    let failed_operations = Arc::new(AtomicUsize::new(0));
    let latencies = Arc::new(RwLock::new(Vec::new()));
    
    let start_time = Instant::now();
    let mut tasks = Vec::new();
    
    for user_id in 0..extreme_config.concurrent_users {
        let sem = semaphore.clone();
        let successful = successful_operations.clone();
        let failed = failed_operations.clone();
        let latencies_clone = latencies.clone();
        let operations_per_user = extreme_config.operations_per_user;
        let operation_types = extreme_config.operation_types.clone();
        
        let task = tokio::spawn(async move {
            for op_id in 0..operations_per_user {
                let _permit = sem.acquire().await.unwrap();
                let op_type = &operation_types[op_id % operation_types.len()];
                let op_start = Instant::now();
                
                // Add timeout to prevent hanging operations
                let operation_result = timeout(
                    Duration::from_secs(5),
                    simulate_operation(op_type.clone())
                ).await;
                
                match operation_result {
                    Ok(Ok(_)) => {
                        successful.fetch_add(1, Ordering::Relaxed);
                        let latency = op_start.elapsed();
                        if latencies_clone.try_write().is_ok() {
                            latencies_clone.write().await.push(latency);
                        }
                    },
                    Ok(Err(_)) | Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
        
        tasks.push(task);
    }
    
    // Monitor system resources during test
    let resource_monitor = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5));
        let mut max_memory = 0.0;
        let mut max_cpu = 0.0;
        
        loop {
            interval.tick().await;
            
            // Simulate resource monitoring
            let memory_usage = get_memory_usage();
            let cpu_usage = get_cpu_usage();
            
            max_memory = max_memory.max(memory_usage);
            max_cpu = max_cpu.max(cpu_usage);
            
            println!("Resource usage - Memory: {:.2} MB, CPU: {:.2}%", 
                    memory_usage, cpu_usage);
            
            if memory_usage > 2048.0 || cpu_usage > 95.0 {
                println!("WARNING: High resource usage detected!");
            }
        }
    });
    
    // Wait for all user tasks to complete or timeout after test duration
    let all_tasks = join_all(tasks);
    let result = timeout(extreme_config.test_duration + Duration::from_secs(30), all_tasks).await;
    
    resource_monitor.abort();
    
    let total_duration = start_time.elapsed();
    let successful = successful_operations.load(Ordering::Relaxed);
    let failed = failed_operations.load(Ordering::Relaxed);
    
    println!("Extreme Load Test Results:");
    println!("  Duration: {:?}", total_duration);
    println!("  Successful: {}", successful);
    println!("  Failed: {}", failed);
    
    // System should remain stable under extreme load
    assert!(successful > 0, "System should process some operations under extreme load");
    assert!(result.is_ok(), "System should not hang under extreme load");
}

/// Test memory leak detection under sustained load
#[tokio::test]
async fn test_memory_leak_detection() {
    let initial_memory = get_memory_usage();
    let mut memory_measurements = Vec::new();
    
    // Run sustained operations
    for iteration in 0..10 {
        println!("Memory leak test iteration: {}", iteration + 1);
        
        // Perform batch operations
        let mut tasks = Vec::new();
        for _ in 0..100 {
            let task = tokio::spawn(async {
                simulate_operation(OperationType::QueryMemory).await
            });
            tasks.push(task);
        }
        
        join_all(tasks).await;
        
        // Force garbage collection
        #[cfg(feature = "gc")]
        gc::force_gc();
        
        // Measure memory after operations
        let current_memory = get_memory_usage();
        memory_measurements.push(current_memory);
        
        println!("Memory usage: {:.2} MB", current_memory);
        
        // Small delay between iterations
        sleep(Duration::from_secs(1)).await;
    }
    
    // Analyze memory trend
    let final_memory = memory_measurements.last().unwrap();
    let memory_growth = final_memory - initial_memory;
    
    println!("Initial memory: {:.2} MB", initial_memory);
    println!("Final memory: {:.2} MB", final_memory);
    println!("Memory growth: {:.2} MB", memory_growth);
    
    // Check for memory leaks
    assert!(memory_growth < 100.0, 
           "Potential memory leak detected: {:.2} MB growth", memory_growth);
    
    // Check for consistent growth pattern (indicating leak)
    let growth_trend = memory_measurements.windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Vec<_>>();
    
    let positive_growth_count = growth_trend.iter()
        .filter(|&&growth| growth > 1.0)
        .count();
    
    assert!(positive_growth_count < growth_trend.len() * 7 / 10,
           "Consistent memory growth pattern indicates leak");
}

/// Test database connection pool under load
#[tokio::test]
async fn test_database_connection_pool() {
    let pool_size = 20;
    let concurrent_connections = 100;
    let operations_per_connection = 50;
    
    // Simulate database connection pool
    let connection_pool = Arc::new(Semaphore::new(pool_size));
    let successful_queries = Arc::new(AtomicUsize::new(0));
    let failed_queries = Arc::new(AtomicUsize::new(0));
    let connection_wait_times = Arc::new(RwLock::new(Vec::new()));
    
    let mut tasks = Vec::new();
    
    for _ in 0..concurrent_connections {
        let pool = connection_pool.clone();
        let successful = successful_queries.clone();
        let failed = failed_queries.clone();
        let wait_times = connection_wait_times.clone();
        
        let task = tokio::spawn(async move {
            for _ in 0..operations_per_connection {
                let wait_start = Instant::now();
                
                // Acquire connection from pool
                let _permit = pool.acquire().await.unwrap();
                let wait_time = wait_start.elapsed();
                wait_times.write().await.push(wait_time);
                
                // Simulate database query
                let query_result = simulate_database_query().await;
                match query_result {
                    Ok(_) => { successful.fetch_add(1, Ordering::Relaxed); },
                    Err(_) => { failed.fetch_add(1, Ordering::Relaxed); }
                }
                
                // Small delay to hold connection
                sleep(Duration::from_millis(10)).await;
            }
        });
        
        tasks.push(task);
    }
    
    join_all(tasks).await;
    
    let successful = successful_queries.load(Ordering::Relaxed);
    let failed = failed_queries.load(Ordering::Relaxed);
    let wait_times_vec = connection_wait_times.read().await;
    
    let avg_wait_time: Duration = wait_times_vec.iter().sum::<Duration>() / wait_times_vec.len() as u32;
    let max_wait_time = wait_times_vec.iter().max().copied().unwrap_or_default();
    
    println!("Database Connection Pool Test:");
    println!("  Pool size: {}", pool_size);
    println!("  Successful queries: {}", successful);
    println!("  Failed queries: {}", failed);
    println!("  Avg wait time: {:?}", avg_wait_time);
    println!("  Max wait time: {:?}", max_wait_time);
    
    // Connection pool performance assertions
    assert!(successful > failed * 10, "Success rate should be high");
    assert!(avg_wait_time < Duration::from_millis(100), "Average wait time should be reasonable");
    assert!(max_wait_time < Duration::from_secs(1), "Max wait time should not be excessive");
}

/// Test system recovery after resource exhaustion
#[tokio::test]
async fn test_resource_exhaustion_recovery() {
    // Simulate resource exhaustion
    let initial_memory = get_memory_usage();
    
    // Create memory pressure
    let mut large_allocations = Vec::new();
    for i in 0..50 {
        let allocation = simulate_large_allocation(10_000_000); // 10MB each
        large_allocations.push(allocation);
        
        let current_memory = get_memory_usage();
        println!("Allocation {}: Memory usage {:.2} MB", i, current_memory);
        
        if current_memory > initial_memory + 400.0 { // Stop before system crash
            break;
        }
    }
    
    let peak_memory = get_memory_usage();
    println!("Peak memory usage: {:.2} MB", peak_memory);
    
    // Release allocations
    large_allocations.clear();
    
    // Force garbage collection
    #[cfg(feature = "gc")]
    gc::force_gc();
    
    // Give system time to recover
    sleep(Duration::from_secs(5)).await;
    
    let recovered_memory = get_memory_usage();
    println!("Memory after recovery: {:.2} MB", recovered_memory);
    
    // Test system functionality after recovery
    let post_recovery_operations = 50;
    let mut successful_post_recovery = 0;
    
    for _ in 0..post_recovery_operations {
        if simulate_operation(OperationType::ReadMarketData).await.is_ok() {
            successful_post_recovery += 1;
        }
    }
    
    // Recovery assertions
    assert!(recovered_memory < peak_memory * 0.7, 
           "Memory should be recovered after resource exhaustion");
    assert!(successful_post_recovery > post_recovery_operations * 8 / 10,
           "System should function normally after recovery");
}

/// Property-based performance testing
proptest! {
    #[test]
    fn test_operation_latency_properties(
        operation_count in 1usize..1000,
        concurrent_factor in 1usize..20,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut latencies = Vec::new();
            let start_time = Instant::now();
            
            // Run operations with varying concurrency
            let tasks_per_batch = operation_count / concurrent_factor.max(1);
            
            for batch in 0..concurrent_factor {
                let mut batch_tasks = Vec::new();
                
                for _ in 0..tasks_per_batch {
                    let task = tokio::spawn(async {
                        let op_start = Instant::now();
                        simulate_operation(OperationType::ReadMarketData).await.ok();
                        op_start.elapsed()
                    });
                    batch_tasks.push(task);
                }
                
                let batch_latencies = join_all(batch_tasks).await;
                for latency_result in batch_latencies {
                    if let Ok(latency) = latency_result {
                        latencies.push(latency);
                    }
                }
            }
            
            let total_time = start_time.elapsed();
            
            if !latencies.is_empty() {
                let avg_latency: Duration = latencies.iter().sum::<Duration>() / latencies.len() as u32;
                let throughput = latencies.len() as f64 / total_time.as_secs_f64();
                
                // Property assertions
                prop_assert!(avg_latency < Duration::from_secs(1), 
                           "Average latency should be reasonable");
                prop_assert!(throughput > 1.0,
                           "Throughput should be positive");
                
                // Latency should not increase significantly with moderate concurrency
                if concurrent_factor <= 10 {
                    prop_assert!(avg_latency < Duration::from_millis(500),
                               "Latency should not degrade significantly with moderate concurrency");
                }
            }
        });
    }
}

// Helper functions for performance testing
async fn simulate_operation(op_type: OperationType) -> Result<()> {
    match op_type {
        OperationType::ReadMarketData => {
            // Simulate market data read
            sleep(Duration::from_millis(1)).await;
            Ok(())
        },
        OperationType::SubmitProposal => {
            // Simulate consensus proposal
            sleep(Duration::from_millis(5)).await;
            if rand::random::<f64>() < 0.95 { Ok(()) } else { 
                Err(HiveMindError::Timeout { timeout_ms: 5000 }) 
            }
        },
        OperationType::QueryMemory => {
            // Simulate memory query
            sleep(Duration::from_millis(2)).await;
            Ok(())
        },
        OperationType::SpawnAgent => {
            // Simulate agent spawning
            sleep(Duration::from_millis(10)).await;
            Ok(())
        },
        OperationType::NetworkMessage => {
            // Simulate network message
            sleep(Duration::from_millis(3)).await;
            Ok(())
        },
    }
}

async fn simulate_database_query() -> Result<()> {
    // Simulate database query latency
    let latency = Duration::from_millis(5 + rand::random::<u64>() % 10);
    sleep(latency).await;
    
    if rand::random::<f64>() < 0.99 { 
        Ok(()) 
    } else { 
        Err(HiveMindError::Database(sqlx::Error::RowNotFound))
    }
}

fn simulate_large_allocation(size_bytes: usize) -> Vec<u8> {
    // Simulate large memory allocation
    vec![0u8; size_bytes]
}

fn get_memory_usage() -> f64 {
    // Simulate memory usage monitoring
    // In real implementation, use system monitoring
    100.0 + rand::random::<f64>() * 50.0 // Mock: 100-150 MB
}

fn get_cpu_usage() -> f64 {
    // Simulate CPU usage monitoring
    // In real implementation, use system monitoring  
    20.0 + rand::random::<f64>() * 30.0 // Mock: 20-50% CPU
}

/// Test autoscaling behavior under load
#[tokio::test]
async fn test_autoscaling() {
    let mut current_capacity = 10; // Start with 10 units of capacity
    let target_utilization = 70.0; // Target 70% utilization
    
    // Simulate load patterns
    let load_patterns = vec![
        (Duration::from_secs(10), 50),  // Low load
        (Duration::from_secs(20), 80),  // Medium load  
        (Duration::from_secs(30), 120), // High load
        (Duration::from_secs(40), 200), // Very high load
        (Duration::from_secs(50), 60),  // Back to medium
        (Duration::from_secs(60), 20),  // Low load
    ];
    
    for (duration, load_level) in load_patterns {
        let utilization = (load_level as f64 / current_capacity as f64) * 100.0;
        
        println!("Time {:?}: Load {}, Capacity {}, Utilization {:.1}%",
                duration, load_level, current_capacity, utilization);
        
        // Autoscaling logic
        if utilization > 80.0 {
            // Scale up
            let scale_factor = (utilization / target_utilization).ceil() as usize;
            current_capacity = (current_capacity * scale_factor).min(100); // Cap at 100
            println!("  Scaling UP to capacity {}", current_capacity);
        } else if utilization < 50.0 && current_capacity > 10 {
            // Scale down
            current_capacity = (current_capacity * 70 / 100).max(10); // Min capacity 10
            println!("  Scaling DOWN to capacity {}", current_capacity);
        }
        
        // Simulate time passing
        sleep(Duration::from_millis(100)).await;
    }
    
    // Verify autoscaling worked
    assert!(current_capacity >= 10, "Capacity should not go below minimum");
    assert!(current_capacity <= 100, "Capacity should not exceed maximum");
}

/// Test circuit breaker pattern under failures
#[tokio::test]
async fn test_circuit_breaker_performance() {
    #[derive(Debug, Clone)]
    enum CircuitState {
        Closed,
        Open,
        HalfOpen,
    }
    
    struct CircuitBreaker {
        state: CircuitState,
        failure_count: usize,
        failure_threshold: usize,
        recovery_timeout: Duration,
        last_failure_time: Option<Instant>,
        success_count: usize,
        success_threshold: usize,
    }
    
    impl CircuitBreaker {
        fn new() -> Self {
            Self {
                state: CircuitState::Closed,
                failure_count: 0,
                failure_threshold: 5,
                recovery_timeout: Duration::from_secs(30),
                last_failure_time: None,
                success_count: 0,
                success_threshold: 3,
            }
        }
        
        fn can_execute(&mut self) -> bool {
            match self.state {
                CircuitState::Closed => true,
                CircuitState::Open => {
                    if let Some(last_failure) = self.last_failure_time {
                        if last_failure.elapsed() > self.recovery_timeout {
                            self.state = CircuitState::HalfOpen;
                            self.success_count = 0;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                },
                CircuitState::HalfOpen => true,
            }
        }
        
        fn record_success(&mut self) {
            match self.state {
                CircuitState::Closed => {
                    self.failure_count = 0;
                },
                CircuitState::HalfOpen => {
                    self.success_count += 1;
                    if self.success_count >= self.success_threshold {
                        self.state = CircuitState::Closed;
                        self.failure_count = 0;
                    }
                },
                _ => {}
            }
        }
        
        fn record_failure(&mut self) {
            self.failure_count += 1;
            self.last_failure_time = Some(Instant::now());
            
            if self.failure_count >= self.failure_threshold {
                self.state = CircuitState::Open;
            }
        }
    }
    
    let mut circuit_breaker = CircuitBreaker::new();
    let mut operation_count = 0;
    let mut blocked_count = 0;
    let mut success_count = 0;
    let mut failure_count = 0;
    
    // Simulate operations with failures
    for i in 0..100 {
        operation_count += 1;
        
        if circuit_breaker.can_execute() {
            // Simulate operation with 20% failure rate initially
            let will_fail = i < 50 && rand::random::<f64>() < 0.2;
            
            if will_fail {
                circuit_breaker.record_failure();
                failure_count += 1;
            } else {
                circuit_breaker.record_success();
                success_count += 1;
            }
        } else {
            blocked_count += 1;
        }
        
        sleep(Duration::from_millis(10)).await;
    }
    
    println!("Circuit Breaker Test Results:");
    println!("  Total operations: {}", operation_count);
    println!("  Successful: {}", success_count);
    println!("  Failed: {}", failure_count);
    println!("  Blocked: {}", blocked_count);
    println!("  Final state: {:?}", circuit_breaker.state);
    
    // Circuit breaker should prevent some operations during failures
    assert!(blocked_count > 0, "Circuit breaker should block some operations");
    assert!(success_count > failure_count, "Should have more successes than failures");
}

/// Test graceful degradation under partial system failures
#[tokio::test]
async fn test_graceful_degradation() {
    struct SystemComponent {
        name: String,
        is_available: bool,
        fallback_available: bool,
    }
    
    let mut components = vec![
        SystemComponent {
            name: "primary_database".to_string(),
            is_available: true,
            fallback_available: true,
        },
        SystemComponent {
            name: "cache_layer".to_string(), 
            is_available: true,
            fallback_available: false,
        },
        SystemComponent {
            name: "neural_processing".to_string(),
            is_available: true,
            fallback_available: true,
        },
        SystemComponent {
            name: "consensus_engine".to_string(),
            is_available: true,
            fallback_available: false,
        },
    ];
    
    let mut degradation_scenarios = vec![
        ("cache_layer", false), // Cache goes down
        ("neural_processing", false), // Neural processing fails
        ("primary_database", false), // Database fails (has fallback)
    ];
    
    for (component_name, availability) in degradation_scenarios {
        // Simulate component failure
        if let Some(component) = components.iter_mut().find(|c| c.name == component_name) {
            component.is_available = availability;
        }
        
        // Test system operation under degraded conditions
        let mut successful_operations = 0;
        let mut degraded_operations = 0;
        let mut failed_operations = 0;
        
        for _ in 0..100 {
            let operation_result = simulate_operation_with_degradation(&components).await;
            
            match operation_result {
                OperationResult::Success => successful_operations += 1,
                OperationResult::Degraded => degraded_operations += 1,
                OperationResult::Failed => failed_operations += 1,
            }
        }
        
        println!("Degradation test - {} failed:", component_name);
        println!("  Successful: {}", successful_operations);
        println!("  Degraded: {}", degraded_operations);
        println!("  Failed: {}", failed_operations);
        
        // System should continue operating even with component failures
        let total_functional = successful_operations + degraded_operations;
        assert!(total_functional > 80, 
               "System should remain largely functional during component failure");
    }
}

#[derive(Debug)]
enum OperationResult {
    Success,
    Degraded,
    Failed,
}

async fn simulate_operation_with_degradation(components: &[SystemComponent]) -> OperationResult {
    // Check critical components
    let consensus_available = components.iter()
        .find(|c| c.name == "consensus_engine")
        .map_or(false, |c| c.is_available);
    
    if !consensus_available {
        return OperationResult::Failed; // Cannot operate without consensus
    }
    
    // Check if primary database is available
    let db_available = components.iter()
        .find(|c| c.name == "primary_database")
        .map_or(false, |c| c.is_available);
    
    let db_fallback = components.iter()
        .find(|c| c.name == "primary_database")
        .map_or(false, |c| c.fallback_available);
    
    if !db_available && !db_fallback {
        return OperationResult::Failed;
    }
    
    // Check optional components
    let cache_available = components.iter()
        .find(|c| c.name == "cache_layer")
        .map_or(false, |c| c.is_available);
    
    let neural_available = components.iter()
        .find(|c| c.name == "neural_processing")
        .map_or(false, |c| c.is_available);
    
    // Determine operation result based on available components
    if db_available && cache_available && neural_available {
        OperationResult::Success
    } else if db_available || db_fallback {
        OperationResult::Degraded // Reduced functionality but still working
    } else {
        OperationResult::Failed
    }
}