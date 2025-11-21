//! Performance Validation Tests for ATS-Core
//!
//! This module contains comprehensive performance tests that validate sub-100μs latency
//! requirements for high-frequency trading applications.

use ats_core::{config::AtsCpConfig, prelude::*, test_utils::*};
use std::time::{Duration, Instant};

const LATENCY_TARGET_TOTAL: Duration = Duration::from_micros(100);
const LATENCY_TARGET_TEMPERATURE: Duration = Duration::from_micros(20);
const LATENCY_TARGET_CONFORMAL: Duration = Duration::from_micros(50);
const LATENCY_TARGET_SIMD: Duration = Duration::from_micros(5);

#[test]
fn test_sub_100_microsecond_full_pipeline() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test data sizes typical for HFT
    let predictions = generate_predictions(32); // 32 predictions
    let calibration_data = generate_calibration_data(100); // 100 calibration points
    let temperature = 1.5;
    
    // Warm up to avoid cold start penalty
    for _ in 0..10 {
        let _ = engine.temperature_scale(&predictions, temperature).unwrap();
    }
    
    // Measure latency over multiple runs
    let mut latencies = Vec::new();
    for _ in 0..1000 {
        let start = Instant::now();
        
        // Full ATS-CP pipeline
        let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
        let _intervals = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        
        let elapsed = start.elapsed();
        latencies.push(elapsed);
    }
    
    // Statistical analysis of latencies
    latencies.sort();
    let median = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let max = latencies[latencies.len() - 1];
    
    println!("Full Pipeline Latency Stats:");
    println!("  Median: {:?}", median);
    println!("  95th percentile: {:?}", p95);
    println!("  99th percentile: {:?}", p99);
    println!("  Maximum: {:?}", max);
    
    // Assertions for latency targets
    assert!(median < LATENCY_TARGET_TOTAL, 
            "Median latency {} exceeds target {}", 
            median.as_micros(), LATENCY_TARGET_TOTAL.as_micros());
    
    assert!(p95 < Duration::from_micros(150), 
            "95th percentile latency {} exceeds 150μs", 
            p95.as_micros());
    
    // Check that at least 90% of operations meet the target
    let within_target = latencies.iter()
        .filter(|&&lat| lat < LATENCY_TARGET_TOTAL)
        .count();
    let percentage = (within_target as f64 / latencies.len() as f64) * 100.0;
    
    assert!(percentage >= 90.0, 
            "Only {:.1}% of operations met the 100μs target", percentage);
}

#[test]
fn test_temperature_scaling_latency() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    for size in [16, 32, 64, 128] {
        let predictions = generate_predictions(size);
        let temperature = 1.5;
        
        // Warm up
        for _ in 0..10 {
            let _ = engine.temperature_scale(&predictions, temperature).unwrap();
        }
        
        let mut latencies = Vec::new();
        for _ in 0..1000 {
            let start = Instant::now();
            let _ = engine.temperature_scale(&predictions, temperature).unwrap();
            latencies.push(start.elapsed());
        }
        
        latencies.sort();
        let median = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
        
        println!("Temperature Scaling (size {}): median={:?}, p99={:?}", 
                 size, median, p99);
        
        assert!(median < LATENCY_TARGET_TEMPERATURE, 
                "Temperature scaling median latency {} exceeds target {} for size {}", 
                median.as_micros(), LATENCY_TARGET_TEMPERATURE.as_micros(), size);
    }
}

#[test]
fn test_conformal_prediction_latency() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let predictions = generate_predictions(32);
    
    for calib_size in [50, 100, 200, 500] {
        let calibration_data = generate_calibration_data(calib_size);
        
        // Warm up
        for _ in 0..10 {
            let _ = engine.conformal_predict(&predictions, &calibration_data).unwrap();
        }
        
        let mut latencies = Vec::new();
        for _ in 0..1000 {
            let start = Instant::now();
            let _ = engine.conformal_predict(&predictions, &calibration_data).unwrap();
            latencies.push(start.elapsed());
        }
        
        latencies.sort();
        let median = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
        
        println!("Conformal Prediction (calib_size {}): median={:?}, p99={:?}", 
                 calib_size, median, p99);
        
        assert!(median < LATENCY_TARGET_CONFORMAL, 
                "Conformal prediction median latency {} exceeds target {} for calib_size {}", 
                median.as_micros(), LATENCY_TARGET_CONFORMAL.as_micros(), calib_size);
    }
}

#[test]
fn test_simd_operations_latency() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    for size in [64, 128, 256, 512, 1024] {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) * 0.01).collect();
        
        // Warm up
        for _ in 0..10 {
            let _ = engine.simd_vector_add(&a, &b).unwrap();
        }
        
        let mut latencies = Vec::new();
        for _ in 0..1000 {
            let start = Instant::now();
            let _ = engine.simd_vector_add(&a, &b).unwrap();
            latencies.push(start.elapsed());
        }
        
        latencies.sort();
        let median = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
        
        println!("SIMD Vector Add (size {}): median={:?}, p99={:?}", 
                 size, median, p99);
        
        // SIMD operations should be extremely fast
        assert!(median < LATENCY_TARGET_SIMD, 
                "SIMD operation median latency {} exceeds target {} for size {}", 
                median.as_micros(), LATENCY_TARGET_SIMD.as_micros(), size);
    }
}

#[test]
fn test_memory_allocation_performance() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test that repeated operations don't degrade performance (no memory leaks)
    let predictions = generate_predictions(64);
    let calibration_data = generate_calibration_data(200);
    let temperature = 1.5;
    
    let mut first_batch_times = Vec::new();
    let mut later_batch_times = Vec::new();
    
    // First batch (potential allocation overhead)
    for _ in 0..100 {
        let start = Instant::now();
        let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
        let _ = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        first_batch_times.push(start.elapsed());
    }
    
    // Later batch (should be stable)
    for _ in 0..100 {
        let start = Instant::now();
        let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
        let _ = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        later_batch_times.push(start.elapsed());
    }
    
    let first_avg: f64 = first_batch_times.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / first_batch_times.len() as f64;
    
    let later_avg: f64 = later_batch_times.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / later_batch_times.len() as f64;
    
    println!("First batch average: {:.0}ns", first_avg);
    println!("Later batch average: {:.0}ns", later_avg);
    
    // Performance should not degrade by more than 10%
    let degradation = (later_avg - first_avg) / first_avg;
    assert!(degradation < 0.10, 
            "Performance degraded by {:.1}% over time", degradation * 100.0);
}

#[test]
fn test_concurrent_performance() {
    use std::sync::Arc;
    use std::thread;
    
    let config = AtsCpConfig::high_performance();
    let engine = Arc::new(AtsCpEngine::new(config).unwrap());
    
    let predictions = Arc::new(generate_predictions(32));
    let calibration_data = Arc::new(generate_calibration_data(100));
    let temperature = 1.5;
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads
    for thread_id in 0..4 {
        let engine_clone = Arc::clone(&engine);
        let predictions_clone = Arc::clone(&predictions);
        let calibration_clone = Arc::clone(&calibration_data);
        
        let handle = thread::spawn(move || {
            let mut latencies = Vec::new();
            
            // Each thread performs 250 operations
            for _ in 0..250 {
                let start = Instant::now();
                let scaled = engine_clone.temperature_scale(&predictions_clone, temperature).unwrap();
                let _ = engine_clone.conformal_predict(&scaled, &calibration_clone).unwrap();
                latencies.push(start.elapsed());
            }
            
            (thread_id, latencies)
        });
        
        handles.push(handle);
    }
    
    // Collect results from all threads
    let mut all_latencies = Vec::new();
    for handle in handles {
        let (thread_id, mut latencies) = handle.join().unwrap();
        println!("Thread {}: {} operations completed", thread_id, latencies.len());
        all_latencies.append(&mut latencies);
    }
    
    // Analyze concurrent performance
    all_latencies.sort();
    let median = all_latencies[all_latencies.len() / 2];
    let p95 = all_latencies[(all_latencies.len() as f64 * 0.95) as usize];
    
    println!("Concurrent performance: median={:?}, p95={:?}", median, p95);
    
    // Concurrent operations should still meet latency targets
    assert!(median < Duration::from_micros(120), 
            "Concurrent median latency {} exceeds 120μs", median.as_micros());
    
    assert!(p95 < Duration::from_micros(200), 
            "Concurrent 95th percentile {} exceeds 200μs", p95.as_micros());
}

#[test]
fn test_cache_efficiency_performance() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Test cache-friendly access patterns
    let small_predictions = generate_predictions(32); // Fits in L1 cache
    let large_predictions = generate_predictions(2048); // Exceeds L1 cache
    let calibration_data = generate_calibration_data(100);
    let temperature = 1.5;
    
    // Small data (cache-friendly)
    let mut small_latencies = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let scaled = engine.temperature_scale(&small_predictions, temperature).unwrap();
        let _ = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        small_latencies.push(start.elapsed());
    }
    
    // Large data (cache-unfriendly)
    let mut large_latencies = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let scaled = engine.temperature_scale(&large_predictions, temperature).unwrap();
        large_latencies.push(start.elapsed());
    }
    
    let small_avg: f64 = small_latencies.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / small_latencies.len() as f64;
    
    let large_avg: f64 = large_latencies.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / large_latencies.len() as f64;
    
    println!("Cache-friendly (32 elements): {:.0}ns", small_avg);
    println!("Cache-unfriendly (2048 elements): {:.0}ns", large_avg);
    
    // Cache-friendly operations should be much faster per element
    let small_per_element = small_avg / 32.0;
    let large_per_element = large_avg / 2048.0;
    
    println!("Per-element: small={:.0}ns, large={:.0}ns", small_per_element, large_per_element);
    
    // Small operations should still meet the absolute target
    assert!(Duration::from_nanos(small_avg as u64) < LATENCY_TARGET_TOTAL,
            "Cache-friendly operations should meet latency target");
}

#[test]
fn test_precision_vs_performance_tradeoff() {
    let config_default = AtsCpConfig::default();
    let config_high_perf = AtsCpConfig::high_performance();
    
    let engine_default = AtsCpEngine::new(config_default).unwrap();
    let engine_high_perf = AtsCpEngine::new(config_high_perf).unwrap();
    
    let predictions = generate_predictions(64);
    let temperature = 1.5;
    
    // Measure precision and performance for both configs
    let mut default_times = Vec::new();
    let mut high_perf_times = Vec::new();
    
    let mut default_results = Vec::new();
    let mut high_perf_results = Vec::new();
    
    for _ in 0..100 {
        // Default config
        let start = Instant::now();
        let result_default = engine_default.temperature_scale(&predictions, temperature).unwrap();
        default_times.push(start.elapsed());
        default_results.push(result_default);
        
        // High performance config
        let start = Instant::now();
        let result_high_perf = engine_high_perf.temperature_scale(&predictions, temperature).unwrap();
        high_perf_times.push(start.elapsed());
        high_perf_results.push(result_high_perf);
    }
    
    let default_avg: f64 = default_times.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / default_times.len() as f64;
    
    let high_perf_avg: f64 = high_perf_times.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / high_perf_times.len() as f64;
    
    println!("Default config average: {:.0}ns", default_avg);
    println!("High performance config average: {:.0}ns", high_perf_avg);
    
    // High performance config should be faster or comparable
    assert!(high_perf_avg <= default_avg * 1.1, 
            "High performance config should not be significantly slower");
    
    // Results should be numerically close
    for (default, high_perf) in default_results[0].iter().zip(high_perf_results[0].iter()) {
        let relative_error = (default - high_perf).abs() / default.max(high_perf);
        assert!(relative_error < 1e-6, 
                "Results should be numerically close: {} vs {}", default, high_perf);
    }
}

#[test]
fn test_latency_stability_over_time() {
    let config = AtsCpConfig::high_performance();
    let engine = AtsCpEngine::new(config).unwrap();
    
    let predictions = generate_predictions(32);
    let calibration_data = generate_calibration_data(100);
    let temperature = 1.5;
    
    // Collect latencies over extended period
    let mut latencies = Vec::new();
    let start_time = Instant::now();
    
    while start_time.elapsed() < Duration::from_secs(10) {
        let op_start = Instant::now();
        let scaled = engine.temperature_scale(&predictions, temperature).unwrap();
        let _ = engine.conformal_predict(&scaled, &calibration_data).unwrap();
        latencies.push(op_start.elapsed());
        
        // Small pause to avoid busy loop
        std::thread::sleep(Duration::from_micros(10));
    }
    
    println!("Collected {} latency measurements over 10 seconds", latencies.len());
    
    // Analyze stability
    let window_size = latencies.len() / 10; // 10 windows
    let mut window_medians = Vec::new();
    
    for i in 0..10 {
        let start_idx = i * window_size;
        let end_idx = ((i + 1) * window_size).min(latencies.len());
        
        if end_idx > start_idx {
            let mut window = latencies[start_idx..end_idx].to_vec();
            window.sort();
            window_medians.push(window[window.len() / 2]);
        }
    }
    
    // Check stability (coefficient of variation)
    let mean: f64 = window_medians.iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>() / window_medians.len() as f64;
    
    let variance: f64 = window_medians.iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / window_medians.len() as f64;
    
    let std_dev = variance.sqrt();
    let coefficient_of_variation = std_dev / mean;
    
    println!("Latency stability - CV: {:.3}", coefficient_of_variation);
    
    // Coefficient of variation should be low (stable performance)
    assert!(coefficient_of_variation < 0.2, 
            "Latency should be stable over time, CV: {:.3}", coefficient_of_variation);
}