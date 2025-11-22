// CWTS Performance Benchmark Runner
// Comprehensive latency and throughput testing for orderbook implementations

use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::thread;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    pub test_name: String,
    pub implementation: String,
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub throughput_ops_per_sec: u64,
    pub memory_usage_mb: u64,
    pub cpu_utilization_pct: f64,
    pub test_duration: Duration,
    pub orders_processed: u64,
    pub timestamp: DateTime<Utc>,
}

pub struct BenchmarkRunner {
    results: Arc<Mutex<Vec<PerformanceBenchmark>>>,
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Test safe orderbook latency with sub-100μs requirement validation
    pub fn benchmark_safe_orderbook_latency(&self) -> Result<PerformanceBenchmark, Box<dyn std::error::Error>> {
        use crate::core::algorithms::safe_orderbook::{SafeOrderBook, Order, OrderSide, OrderType};
        use rust_decimal::Decimal;
        
        let orderbook = Arc::new(SafeOrderBook::new());
        let mut latencies = Vec::with_capacity(10000);
        let test_start = Instant::now();
        
        // Warmup phase
        for i in 0..1000 {
            let order = Order {
                order_id: i,
                price: Decimal::from(100),
                quantity: Decimal::from(10),
                side: OrderSide::Buy,
                timestamp: Utc::now(),
                order_type: OrderType::Limit,
                user_id: format!("user_{}", i),
            };
            let _ = orderbook.add_order(order);
        }
        
        // Benchmark phase - measure individual operation latency
        for i in 1000..11000 {
            let order = Order {
                order_id: i,
                price: Decimal::from(100 + (i % 50)),
                quantity: Decimal::from(10),
                side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
                timestamp: Utc::now(),
                order_type: OrderType::Limit,
                user_id: format!("user_{}", i),
            };
            
            let start = Instant::now();
            let result = orderbook.add_order(order);
            let latency = start.elapsed();
            
            if result.is_ok() {
                latencies.push(latency);
            }
        }
        
        let test_duration = test_start.elapsed();
        latencies.sort();
        
        let p50_idx = latencies.len() / 2;
        let p95_idx = (latencies.len() * 95) / 100;
        let p99_idx = (latencies.len() * 99) / 100;
        
        let benchmark = PerformanceBenchmark {
            test_name: "Safe Orderbook Latency Test".to_string(),
            implementation: "safe_orderbook".to_string(),
            latency_p50: latencies[p50_idx],
            latency_p95: latencies[p95_idx],
            latency_p99: latencies[p99_idx],
            throughput_ops_per_sec: (latencies.len() as u64 * 1_000_000) / test_duration.as_micros() as u64,
            memory_usage_mb: self.get_memory_usage(),
            cpu_utilization_pct: 0.0, // Would need system monitoring
            test_duration,
            orders_processed: latencies.len() as u64,
            timestamp: Utc::now(),
        };
        
        // Validate sub-100μs requirement
        if benchmark.latency_p99 > Duration::from_micros(100) {
            eprintln!("WARNING: P99 latency {} exceeds 100μs requirement", 
                     benchmark.latency_p99.as_micros());
        }
        
        self.results.lock().unwrap().push(benchmark.clone());
        Ok(benchmark)
    }

    /// Test throughput with 1M+ orders/sec target
    pub fn benchmark_throughput_test(&self, target_ops_per_sec: u64) -> Result<PerformanceBenchmark, Box<dyn std::error::Error>> {
        use crate::core::algorithms::safe_orderbook::{SafeOrderBook, Order, OrderSide, OrderType};
        use rust_decimal::Decimal;
        
        let orderbook = Arc::new(SafeOrderBook::new());
        let test_duration = Duration::from_secs(10);
        let test_start = Instant::now();
        let mut operations_count = 0u64;
        let mut latencies = Vec::new();
        
        while test_start.elapsed() < test_duration {
            let batch_start = Instant::now();
            
            // Process batch of orders
            for i in 0..1000 {
                let order = Order {
                    order_id: operations_count + i,
                    price: Decimal::from(100 + (i % 100)),
                    quantity: Decimal::from(10),
                    side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
                    timestamp: Utc::now(),
                    order_type: OrderType::Limit,
                    user_id: format!("user_{}", i),
                };
                
                let op_start = Instant::now();
                let _ = orderbook.add_order(order);
                latencies.push(op_start.elapsed());
            }
            
            operations_count += 1000;
            
            // Rate limiting to avoid overwhelming system
            let batch_duration = batch_start.elapsed();
            let target_batch_duration = Duration::from_millis(1);
            if batch_duration < target_batch_duration {
                thread::sleep(target_batch_duration - batch_duration);
            }
        }
        
        let actual_test_duration = test_start.elapsed();
        latencies.sort();
        
        let p50_idx = latencies.len() / 2;
        let p95_idx = (latencies.len() * 95) / 100;
        let p99_idx = (latencies.len() * 99) / 100;
        
        let actual_throughput = (operations_count * 1000) / actual_test_duration.as_millis() as u64;
        
        let benchmark = PerformanceBenchmark {
            test_name: format!("Throughput Test (Target: {} ops/sec)", target_ops_per_sec),
            implementation: "safe_orderbook".to_string(),
            latency_p50: latencies[p50_idx],
            latency_p95: latencies[p95_idx],
            latency_p99: latencies[p99_idx],
            throughput_ops_per_sec: actual_throughput,
            memory_usage_mb: self.get_memory_usage(),
            cpu_utilization_pct: 0.0,
            test_duration: actual_test_duration,
            orders_processed: operations_count,
            timestamp: Utc::now(),
        };
        
        // Check if we hit target throughput
        if benchmark.throughput_ops_per_sec < target_ops_per_sec {
            eprintln!("WARNING: Actual throughput {} ops/sec below target {} ops/sec",
                     benchmark.throughput_ops_per_sec, target_ops_per_sec);
        }
        
        self.results.lock().unwrap().push(benchmark.clone());
        Ok(benchmark)
    }

    /// Concurrent stress test with multiple threads
    pub fn benchmark_concurrent_stress_test(&self, thread_count: usize) -> Result<PerformanceBenchmark, Box<dyn std::error::Error>> {
        use crate::core::algorithms::safe_orderbook::{SafeOrderBook, Order, OrderSide, OrderType};
        use rust_decimal::Decimal;
        
        let orderbook = Arc::new(SafeOrderBook::new());
        let test_start = Instant::now();
        let operations_per_thread = 10000;
        let mut handles = Vec::new();
        let latencies = Arc::new(Mutex::new(Vec::new()));
        
        for thread_id in 0..thread_count {
            let orderbook_clone = orderbook.clone();
            let latencies_clone = latencies.clone();
            
            let handle = thread::spawn(move || {
                let mut local_latencies = Vec::with_capacity(operations_per_thread);
                
                for i in 0..operations_per_thread {
                    let order_id = (thread_id * operations_per_thread + i) as u64;
                    let order = Order {
                        order_id,
                        price: Decimal::from(100 + (i % 50)),
                        quantity: Decimal::from(10),
                        side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
                        timestamp: Utc::now(),
                        order_type: OrderType::Limit,
                        user_id: format!("thread_{}_user_{}", thread_id, i),
                    };
                    
                    let start = Instant::now();
                    let result = orderbook_clone.add_order(order);
                    let latency = start.elapsed();
                    
                    if result.is_ok() {
                        local_latencies.push(latency);
                    }
                }
                
                latencies_clone.lock().unwrap().extend(local_latencies);
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let test_duration = test_start.elapsed();
        let mut all_latencies = latencies.lock().unwrap().clone();
        all_latencies.sort();
        
        let p50_idx = all_latencies.len() / 2;
        let p95_idx = (all_latencies.len() * 95) / 100;
        let p99_idx = (all_latencies.len() * 99) / 100;
        
        let benchmark = PerformanceBenchmark {
            test_name: format!("Concurrent Stress Test ({} threads)", thread_count),
            implementation: "safe_orderbook".to_string(),
            latency_p50: all_latencies[p50_idx],
            latency_p95: all_latencies[p95_idx],
            latency_p99: all_latencies[p99_idx],
            throughput_ops_per_sec: (all_latencies.len() as u64 * 1_000_000) / test_duration.as_micros() as u64,
            memory_usage_mb: self.get_memory_usage(),
            cpu_utilization_pct: 0.0,
            test_duration,
            orders_processed: all_latencies.len() as u64,
            timestamp: Utc::now(),
        };
        
        self.results.lock().unwrap().push(benchmark.clone());
        Ok(benchmark)
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> String {
        let results = self.results.lock().unwrap();
        let mut report = String::new();
        
        report.push_str("# CWTS Performance Benchmark Report\n\n");
        report.push_str(&format!("Generated: {}\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("Total Tests: {}\n\n", results.len()));
        
        for benchmark in results.iter() {
            report.push_str(&format!("## {}\n", benchmark.test_name));
            report.push_str(&format!("**Implementation**: {}\n", benchmark.implementation));
            report.push_str(&format!("**Test Duration**: {:?}\n", benchmark.test_duration));
            report.push_str(&format!("**Orders Processed**: {}\n", benchmark.orders_processed));
            report.push_str(&format!("**Throughput**: {} ops/sec\n", benchmark.throughput_ops_per_sec));
            report.push_str("**Latency**:\n");
            report.push_str(&format!("  - P50: {:?}\n", benchmark.latency_p50));
            report.push_str(&format!("  - P95: {:?}\n", benchmark.latency_p95));
            report.push_str(&format!("  - P99: {:?}\n", benchmark.latency_p99));
            report.push_str(&format!("**Memory Usage**: {} MB\n", benchmark.memory_usage_mb));
            
            // Performance requirements validation
            if benchmark.latency_p99 <= Duration::from_micros(100) {
                report.push_str("✅ **Sub-100μs Latency**: PASSED\n");
            } else {
                report.push_str("❌ **Sub-100μs Latency**: FAILED\n");
            }
            
            if benchmark.throughput_ops_per_sec >= 1_000_000 {
                report.push_str("✅ **1M+ ops/sec Throughput**: PASSED\n");
            } else {
                report.push_str("❌ **1M+ ops/sec Throughput**: FAILED\n");
            }
            
            report.push_str("\n---\n\n");
        }
        
        report
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage(&self) -> u64 {
        // In a real implementation, would use system APIs to get actual memory usage
        // For now, return a placeholder value
        0
    }

    pub fn run_all_benchmarks(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Running comprehensive CWTS performance benchmarks...");
        
        // 1. Latency benchmark
        println!("1. Running latency benchmark...");
        self.benchmark_safe_orderbook_latency()?;
        
        // 2. Throughput benchmark
        println!("2. Running throughput benchmark...");
        self.benchmark_throughput_test(1_000_000)?;
        
        // 3. Concurrent stress test
        println!("3. Running concurrent stress test...");
        self.benchmark_concurrent_stress_test(8)?;
        
        // Generate and print report
        let report = self.generate_report();
        println!("{}", report);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new();
        let results = runner.results.lock().unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test] 
    fn test_performance_benchmark_serialization() {
        let benchmark = PerformanceBenchmark {
            test_name: "Test".to_string(),
            implementation: "safe_orderbook".to_string(),
            latency_p50: Duration::from_micros(10),
            latency_p95: Duration::from_micros(20),
            latency_p99: Duration::from_micros(30),
            throughput_ops_per_sec: 1000000,
            memory_usage_mb: 100,
            cpu_utilization_pct: 50.0,
            test_duration: Duration::from_secs(10),
            orders_processed: 1000000,
            timestamp: Utc::now(),
        };
        
        let json = serde_json::to_string(&benchmark).unwrap();
        let deserialized: PerformanceBenchmark = serde_json::from_str(&json).unwrap();
        assert_eq!(benchmark.test_name, deserialized.test_name);
    }
}