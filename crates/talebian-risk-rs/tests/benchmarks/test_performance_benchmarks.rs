//! Performance benchmarks for the Talebian Risk RS system
//! Tests latency, throughput, and memory efficiency requirements

use talebian_risk_rs::{
    risk_engine::*, MacchiavelianConfig, MarketData, TalebianRiskError
};
use chrono::{Utc, Duration};
use std::time::{Instant, Duration as StdDuration};
use std::collections::VecDeque;

/// Helper to create realistic market data for benchmarking
fn create_benchmark_market_data(id: usize) -> MarketData {
    MarketData {
        timestamp: Utc::now() + Duration::seconds(id as i64),
        timestamp_unix: 1640995200 + (id as i64),
        price: 50000.0 + (id as f64 * 0.1),
        volume: 1000.0 + (id as f64 * 0.5),
        bid: 49990.0 + (id as f64 * 0.1),
        ask: 50010.0 + (id as f64 * 0.1),
        bid_volume: 500.0 + (id as f64 * 0.1),
        ask_volume: 500.0 + (id as f64 * 0.1),
        volatility: 0.02 + (id as f64 * 0.0001),
        returns: vec![0.001, 0.002, -0.001, 0.0015],
        volume_history: vec![1000.0, 1100.0, 950.0, 1050.0, 1000.0],
    }
}

/// Helper to create whale activity data for benchmarking
fn create_whale_benchmark_data(id: usize) -> MarketData {
    MarketData {
        timestamp: Utc::now() + Duration::seconds(id as i64),
        timestamp_unix: 1640995200 + (id as i64),
        price: 50000.0 + (id as f64 * 2.0),
        volume: 5000.0, // Whale volume
        bid: 49980.0 + (id as f64 * 2.0),
        ask: 50020.0 + (id as f64 * 2.0),
        bid_volume: 2000.0,
        ask_volume: 800.0,
        volatility: 0.05,
        returns: vec![0.02, 0.025, 0.015, 0.03],
        volume_history: vec![1000.0; 5],
    }
}

/// Performance metrics collector
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    latencies: VecDeque<StdDuration>,
    throughput_samples: VecDeque<f64>,
    memory_usage_samples: VecDeque<usize>,
    error_count: usize,
    total_operations: usize,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            latencies: VecDeque::with_capacity(10000),
            throughput_samples: VecDeque::with_capacity(1000),
            memory_usage_samples: VecDeque::with_capacity(1000),
            error_count: 0,
            total_operations: 0,
        }
    }
    
    fn record_latency(&mut self, latency: StdDuration) {
        self.latencies.push_back(latency);
        if self.latencies.len() > 10000 {
            self.latencies.pop_front();
        }
    }
    
    fn record_throughput(&mut self, ops_per_second: f64) {
        self.throughput_samples.push_back(ops_per_second);
        if self.throughput_samples.len() > 1000 {
            self.throughput_samples.pop_front();
        }
    }
    
    fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage_samples.push_back(bytes);
        if self.memory_usage_samples.len() > 1000 {
            self.memory_usage_samples.pop_front();
        }
    }
    
    fn get_latency_stats(&self) -> (f64, f64, f64, f64) {
        if self.latencies.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        
        let mut sorted_latencies: Vec<f64> = self.latencies.iter()
            .map(|d| d.as_secs_f64() * 1000.0) // Convert to milliseconds
            .collect();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted_latencies.len();
        let mean = sorted_latencies.iter().sum::<f64>() / len as f64;
        let p50 = sorted_latencies[len / 2];
        let p95 = sorted_latencies[(len as f64 * 0.95) as usize];
        let p99 = sorted_latencies[(len as f64 * 0.99) as usize];
        
        (mean, p50, p95, p99)
    }
    
    fn get_throughput_stats(&self) -> (f64, f64, f64) {
        if self.throughput_samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let sum: f64 = self.throughput_samples.iter().sum();
        let mean = sum / self.throughput_samples.len() as f64;
        let min = self.throughput_samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.throughput_samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        (mean, min, max)
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn bench_single_assessment_latency() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let mut metrics = PerformanceMetrics::new();
        
        // Warm up the engine
        for i in 0..100 {
            let market_data = create_benchmark_market_data(i);
            let _ = engine.assess_risk(&market_data);
        }
        
        // Benchmark single assessment latency
        for i in 0..1000 {
            let market_data = create_benchmark_market_data(i);
            
            let start = Instant::now();
            let result = engine.assess_risk(&market_data);
            let duration = start.elapsed();
            
            metrics.total_operations += 1;
            
            match result {
                Ok(_) => {
                    metrics.record_latency(duration);
                },
                Err(_) => {
                    metrics.error_count += 1;
                }
            }
        }
        
        let (mean_ms, p50_ms, p95_ms, p99_ms) = metrics.get_latency_stats();
        
        println!(\"Single Assessment Latency Benchmark:\");
        println!(\"  Mean: {:.3}ms\", mean_ms);
        println!(\"  P50:  {:.3}ms\", p50_ms);
        println!(\"  P95:  {:.3}ms\", p95_ms);
        println!(\"  P99:  {:.3}ms\", p99_ms);
        println!(\"  Error rate: {:.2}%\", (metrics.error_count as f64 / metrics.total_operations as f64) * 100.0);
        
        // Performance requirements
        assert!(mean_ms < 1.0, \"Mean latency should be under 1ms (actual: {:.3}ms)\", mean_ms);
        assert!(p95_ms < 2.0, \"P95 latency should be under 2ms (actual: {:.3}ms)\", p95_ms);
        assert!(p99_ms < 5.0, \"P99 latency should be under 5ms (actual: {:.3}ms)\", p99_ms);
        assert!(metrics.error_count == 0, \"Should have zero errors during normal operation\");
    }

    #[test]
    fn bench_whale_detection_latency() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let mut metrics = PerformanceMetrics::new();
        
        // Benchmark whale detection specifically
        for i in 0..1000 {
            let market_data = create_whale_benchmark_data(i);
            
            let start = Instant::now();
            let result = engine.assess_risk(&market_data);
            let duration = start.elapsed();
            
            metrics.total_operations += 1;
            
            match result {
                Ok(assessment) => {
                    metrics.record_latency(duration);
                    
                    // Verify whale detection is working
                    if i == 0 { // First iteration should detect whale
                        assert!(assessment.whale_detection.is_whale_detected);
                    }
                },
                Err(_) => {
                    metrics.error_count += 1;
                }
            }
        }
        
        let (mean_ms, p50_ms, p95_ms, p99_ms) = metrics.get_latency_stats();
        
        println!(\"Whale Detection Latency Benchmark:\");
        println!(\"  Mean: {:.3}ms\", mean_ms);
        println!(\"  P50:  {:.3}ms\", p50_ms);
        println!(\"  P95:  {:.3}ms\", p95_ms);
        println!(\"  P99:  {:.3}ms\", p99_ms);
        
        // Whale detection should be similarly fast
        assert!(mean_ms < 1.5, \"Whale detection mean latency should be under 1.5ms (actual: {:.3}ms)\", mean_ms);
        assert!(p95_ms < 3.0, \"Whale detection P95 latency should be under 3ms (actual: {:.3}ms)\", p95_ms);
    }

    #[test]
    fn bench_throughput_sustained() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let mut metrics = PerformanceMetrics::new();
        
        let test_duration = StdDuration::from_secs(10); // 10 second test
        let batch_size = 100;
        let start_time = Instant::now();
        let mut total_operations = 0;
        
        println!(\"Running sustained throughput test for {} seconds...\", test_duration.as_secs());
        
        while start_time.elapsed() < test_duration {
            let batch_start = Instant::now();
            let mut successful_ops = 0;
            
            // Process a batch
            for i in 0..batch_size {
                let market_data = create_benchmark_market_data(total_operations + i);
                
                if engine.assess_risk(&market_data).is_ok() {
                    successful_ops += 1;
                }
            }
            
            let batch_duration = batch_start.elapsed();
            let ops_per_second = successful_ops as f64 / batch_duration.as_secs_f64();
            
            metrics.record_throughput(ops_per_second);
            total_operations += batch_size;
        }
        
        let actual_duration = start_time.elapsed();
        let overall_throughput = total_operations as f64 / actual_duration.as_secs_f64();
        let (mean_throughput, min_throughput, max_throughput) = metrics.get_throughput_stats();
        
        println!(\"Sustained Throughput Benchmark:\");
        println!(\"  Overall throughput: {:.0} ops/sec\", overall_throughput);
        println!(\"  Mean batch throughput: {:.0} ops/sec\", mean_throughput);
        println!(\"  Min batch throughput: {:.0} ops/sec\", min_throughput);
        println!(\"  Max batch throughput: {:.0} ops/sec\", max_throughput);
        println!(\"  Total operations: {}\", total_operations);
        
        // Throughput requirements
        assert!(overall_throughput > 1000.0, \"Overall throughput should exceed 1000 ops/sec (actual: {:.0})\", overall_throughput);
        assert!(mean_throughput > 800.0, \"Mean throughput should exceed 800 ops/sec (actual: {:.0})\", mean_throughput);
        assert!(min_throughput > 500.0, \"Minimum throughput should exceed 500 ops/sec (actual: {:.0})\", min_throughput);
    }

    #[test]
    fn bench_memory_usage() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let mut metrics = PerformanceMetrics::new();
        
        // Record initial memory usage (approximate)
        let initial_size = std::mem::size_of_val(&engine);
        metrics.record_memory_usage(initial_size);
        
        println!(\"Memory Usage Benchmark:\");
        println!(\"  Initial engine size: {} bytes\", initial_size);
        
        // Process increasing amounts of data and monitor memory
        let test_sizes = vec![1000, 5000, 10000, 20000, 50000];
        
        for &test_size in &test_sizes {
            // Process data
            for i in 0..test_size {
                let market_data = create_benchmark_market_data(i);
                let _ = engine.assess_risk(&market_data);
                
                // Occasionally record trade outcomes
                if i % 100 == 0 {
                    let _ = engine.record_trade_outcome(0.01, i % 3 == 0, 0.5);
                }
            }
            
            // Estimate memory usage
            let current_size = std::mem::size_of_val(&engine) + 
                              engine.assessment_history.len() * std::mem::size_of::<TalebianRiskAssessment>();
            
            metrics.record_memory_usage(current_size);
            
            println!(\"  After {} operations: {} bytes ({:.2} MB)\", 
                    test_size, current_size, current_size as f64 / 1024.0 / 1024.0);
            
            // Check memory bounds
            let mb_usage = current_size as f64 / 1024.0 / 1024.0;
            assert!(mb_usage < 100.0, \"Memory usage should stay under 100MB (actual: {:.2}MB)\", mb_usage);
            
            // Check that assessment history is bounded
            assert!(engine.assessment_history.len() <= 10000, \"Assessment history should be bounded\");
        }
        
        // Memory should not grow linearly with operations due to bounds
        let final_size = metrics.memory_usage_samples.back().unwrap();
        let growth_ratio = *final_size as f64 / initial_size as f64;
        
        println!(\"  Memory growth ratio: {:.2}x\", growth_ratio);
        assert!(growth_ratio < 50.0, \"Memory growth should be bounded (actual: {:.2}x)\", growth_ratio);
    }

    #[test]
    fn bench_recommendation_generation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        let mut metrics = PerformanceMetrics::new();
        
        // Benchmark recommendation generation (more complex operation)
        for i in 0..500 {
            let market_data = create_whale_benchmark_data(i);
            
            let start = Instant::now();
            let result = engine.generate_recommendations(&market_data);
            let duration = start.elapsed();
            
            metrics.total_operations += 1;
            
            match result {
                Ok(_) => {
                    metrics.record_latency(duration);
                },
                Err(_) => {
                    metrics.error_count += 1;
                }
            }
        }
        
        let (mean_ms, p50_ms, p95_ms, p99_ms) = metrics.get_latency_stats();
        
        println!(\"Recommendation Generation Benchmark:\");
        println!(\"  Mean: {:.3}ms\", mean_ms);
        println!(\"  P50:  {:.3}ms\", p50_ms);
        println!(\"  P95:  {:.3}ms\", p95_ms);
        println!(\"  P99:  {:.3}ms\", p99_ms);
        
        // Recommendations are more complex but should still be fast
        assert!(mean_ms < 2.0, \"Recommendation mean latency should be under 2ms (actual: {:.3}ms)\", mean_ms);
        assert!(p95_ms < 5.0, \"Recommendation P95 latency should be under 5ms (actual: {:.3}ms)\", p95_ms);
        assert!(p99_ms < 10.0, \"Recommendation P99 latency should be under 10ms (actual: {:.3}ms)\", p99_ms);
    }

    #[test]
    fn bench_concurrent_performance() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = Arc::new(Mutex::new(TalebianRiskEngine::new(config)));
        
        let num_threads = 4;
        let operations_per_thread = 250;
        let mut handles = vec![];
        
        let start_time = Instant::now();
        
        // Spawn concurrent threads
        for thread_id in 0..num_threads {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let mut thread_latencies = Vec::new();
                
                for i in 0..operations_per_thread {
                    let market_data = create_benchmark_market_data(thread_id * operations_per_thread + i);
                    
                    let start = Instant::now();
                    let mut engine_guard = engine_clone.lock().unwrap();
                    let result = engine_guard.assess_risk(&market_data);
                    drop(engine_guard); // Release lock quickly
                    let duration = start.elapsed();
                    
                    if result.is_ok() {
                        thread_latencies.push(duration);
                    }
                }
                
                thread_latencies
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut all_latencies = Vec::new();
        for handle in handles {
            let thread_latencies = handle.join().unwrap();
            all_latencies.extend(thread_latencies);
        }
        
        let total_duration = start_time.elapsed();
        let total_operations = all_latencies.len();
        let throughput = total_operations as f64 / total_duration.as_secs_f64();
        
        // Calculate latency statistics
        let mut latency_ms: Vec<f64> = all_latencies.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        latency_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean_ms = latency_ms.iter().sum::<f64>() / latency_ms.len() as f64;
        let p95_ms = latency_ms[(latency_ms.len() as f64 * 0.95) as usize];
        let p99_ms = latency_ms[(latency_ms.len() as f64 * 0.99) as usize];
        
        println!(\"Concurrent Performance Benchmark ({} threads):\", num_threads);
        println!(\"  Total operations: {}\", total_operations);
        println!(\"  Throughput: {:.0} ops/sec\", throughput);
        println!(\"  Mean latency: {:.3}ms\", mean_ms);
        println!(\"  P95 latency: {:.3}ms\", p95_ms);
        println!(\"  P99 latency: {:.3}ms\", p99_ms);
        
        // Concurrent performance requirements (accounting for contention)
        assert!(throughput > 500.0, \"Concurrent throughput should exceed 500 ops/sec (actual: {:.0})\", throughput);
        assert!(mean_ms < 5.0, \"Concurrent mean latency should be under 5ms (actual: {:.3}ms)\", mean_ms);
        assert!(p99_ms < 20.0, \"Concurrent P99 latency should be under 20ms (actual: {:.3}ms)\", p99_ms);
    }

    #[test]
    fn bench_cold_start_performance() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut cold_start_times = Vec::new();
        
        // Test multiple cold starts
        for _ in 0..10 {
            let start = Instant::now();
            let mut engine = TalebianRiskEngine::new(config.clone());
            let market_data = create_benchmark_market_data(0);
            let _ = engine.assess_risk(&market_data);
            let duration = start.elapsed();
            
            cold_start_times.push(duration.as_millis());
        }
        
        let mean_cold_start = cold_start_times.iter().sum::<u128>() as f64 / cold_start_times.len() as f64;
        let max_cold_start = *cold_start_times.iter().max().unwrap();
        
        println!(\"Cold Start Performance Benchmark:\");
        println!(\"  Mean cold start: {:.1}ms\", mean_cold_start);
        println!(\"  Max cold start: {}ms\", max_cold_start);
        
        // Cold start should be reasonably fast
        assert!(mean_cold_start < 10.0, \"Mean cold start should be under 10ms (actual: {:.1}ms)\", mean_cold_start);
        assert!(max_cold_start < 50, \"Max cold start should be under 50ms (actual: {}ms)\", max_cold_start);
    }

    #[test]
    fn bench_degradation_under_load() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let load_levels = vec![100, 500, 1000, 2000, 5000];
        let mut degradation_results = Vec::new();
        
        for &load_size in &load_levels {
            let start_time = Instant::now();
            let mut successful_ops = 0;
            
            // Process load
            for i in 0..load_size {
                let market_data = create_benchmark_market_data(i);
                
                if engine.assess_risk(&market_data).is_ok() {
                    successful_ops += 1;
                }
            }
            
            let duration = start_time.elapsed();
            let throughput = successful_ops as f64 / duration.as_secs_f64();
            let avg_latency_ms = duration.as_millis() as f64 / successful_ops as f64;
            
            degradation_results.push((load_size, throughput, avg_latency_ms));
            
            println!(\"Load {} operations: {:.0} ops/sec, {:.3}ms avg latency\", 
                    load_size, throughput, avg_latency_ms);
        }
        
        // Check that performance doesn't degrade dramatically
        let baseline_throughput = degradation_results[0].1;
        let max_load_throughput = degradation_results.last().unwrap().1;
        let throughput_retention = max_load_throughput / baseline_throughput;
        
        println!(\"Throughput retention at max load: {:.2}%\", throughput_retention * 100.0);
        
        // Should maintain at least 50% of baseline throughput under high load
        assert!(throughput_retention > 0.5, 
               \"Should maintain >50% throughput under load (actual: {:.2}%)\", 
               throughput_retention * 100.0);
        
        // Latency should not grow excessively
        let baseline_latency = degradation_results[0].2;
        let max_load_latency = degradation_results.last().unwrap().2;
        let latency_degradation = max_load_latency / baseline_latency;
        
        println!(\"Latency degradation factor: {:.2}x\", latency_degradation);
        
        assert!(latency_degradation < 5.0, 
               \"Latency degradation should be <5x (actual: {:.2}x)\", 
               latency_degradation);
    }

    #[test]
    fn bench_stability_over_time() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = TalebianRiskEngine::new(config);
        
        let measurement_intervals = 1000; // Measure every 1000 operations
        let total_operations = 10000;
        let mut stability_measurements = Vec::new();
        
        for chunk in 0..(total_operations / measurement_intervals) {
            let chunk_start = Instant::now();
            let mut successful_ops = 0;
            
            // Process chunk of operations
            for i in 0..measurement_intervals {
                let operation_id = chunk * measurement_intervals + i;
                let market_data = create_benchmark_market_data(operation_id);
                
                if engine.assess_risk(&market_data).is_ok() {
                    successful_ops += 1;
                }
            }
            
            let chunk_duration = chunk_start.elapsed();
            let chunk_throughput = successful_ops as f64 / chunk_duration.as_secs_f64();
            let chunk_avg_latency = chunk_duration.as_millis() as f64 / successful_ops as f64;
            
            stability_measurements.push((chunk_throughput, chunk_avg_latency));
            
            println!(\"Chunk {}: {:.0} ops/sec, {:.3}ms avg latency\", 
                    chunk, chunk_throughput, chunk_avg_latency);
        }
        
        // Calculate stability metrics
        let throughputs: Vec<f64> = stability_measurements.iter().map(|(t, _)| *t).collect();
        let latencies: Vec<f64> = stability_measurements.iter().map(|(_, l)| *l).collect();
        
        let throughput_mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let throughput_std = {
            let variance = throughputs.iter()
                .map(|t| (t - throughput_mean).powi(2))
                .sum::<f64>() / throughputs.len() as f64;
            variance.sqrt()
        };
        let throughput_cv = throughput_std / throughput_mean; // Coefficient of variation
        
        let latency_mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let latency_std = {
            let variance = latencies.iter()
                .map(|l| (l - latency_mean).powi(2))
                .sum::<f64>() / latencies.len() as f64;
            variance.sqrt()
        };
        let latency_cv = latency_std / latency_mean;
        
        println!(\"Stability over {} operations:\", total_operations);
        println!(\"  Throughput: {:.0} ± {:.0} ops/sec (CV: {:.3})\", 
                throughput_mean, throughput_std, throughput_cv);
        println!(\"  Latency: {:.3} ± {:.3}ms (CV: {:.3})\", 
                latency_mean, latency_std, latency_cv);
        
        // Stability requirements
        assert!(throughput_cv < 0.2, \"Throughput should be stable (CV < 0.2, actual: {:.3})\", throughput_cv);
        assert!(latency_cv < 0.3, \"Latency should be stable (CV < 0.3, actual: {:.3})\", latency_cv);
        
        // Performance should remain good throughout
        assert!(throughput_mean > 1000.0, \"Mean throughput should exceed 1000 ops/sec\");
        assert!(latency_mean < 1.0, \"Mean latency should be under 1ms\");
    }
}