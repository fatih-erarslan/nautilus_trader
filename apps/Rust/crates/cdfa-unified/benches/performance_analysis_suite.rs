//! Performance Analysis and Bottleneck Detection Suite
//!
//! This module provides detailed performance analysis and bottleneck identification
//! for the CDFA unified system, including:
//! - Automated bottleneck detection
//! - Performance profiling with detailed metrics
//! - Resource utilization analysis
//! - Optimization recommendations

use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale
};
use cdfa_unified::{
    detectors::black_swan::{BlackSwanDetector, BlackSwanConfig},
    analyzers::{
        soc::{SOCAnalyzer, SOCParameters},
        antifragility::{AntifragilityAnalyzer, AntifragilityParameters},
    },
    optimizers::stdp::{STDPOptimizer, STDPConfig},
    types::{CdfaArray, CdfaMatrix},
    perf_tools::PerformanceProfiler,
};
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use sysinfo::{System, SystemExt, ProcessExt, Pid};

/// Performance metrics collection
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time: Duration,
    pub memory_usage: u64,
    pub cpu_usage: f32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub operations_per_second: f64,
    pub memory_allocations: u64,
    pub peak_memory: u64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            execution_time: Duration::default(),
            memory_usage: 0,
            cpu_usage: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            operations_per_second: 0.0,
            memory_allocations: 0,
            peak_memory: 0,
        }
    }
    
    pub fn efficiency_score(&self) -> f64 {
        let time_score = 1.0 / (self.execution_time.as_secs_f64() + 1e-9);
        let memory_score = 1.0 / (self.memory_usage as f64 + 1.0);
        let cpu_score = 1.0 / (self.cpu_usage as f64 + 1.0);
        
        (time_score * memory_score * cpu_score).cbrt()
    }
    
    pub fn cache_efficiency(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            1.0
        } else {
            self.cache_hits as f64 / total_accesses as f64
        }
    }
}

/// Bottleneck identification system
#[derive(Debug, Clone)]
pub struct BottleneckAnalyzer {
    system: System,
    baseline_metrics: HashMap<String, PerformanceMetrics>,
}

impl BottleneckAnalyzer {
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
            baseline_metrics: HashMap::new(),
        }
    }
    
    pub fn measure_performance<F, R>(&mut self, name: &str, operation: F) -> (R, PerformanceMetrics)
    where
        F: FnOnce() -> R,
    {
        // Pre-measurement system state
        self.system.refresh_all();
        let initial_memory = self.system.total_memory() - self.system.available_memory();
        let pid = Pid::from(std::process::id() as usize);
        let initial_cpu = self.system.process(pid)
            .map(|p| p.cpu_usage())
            .unwrap_or(0.0);
        
        let start_time = Instant::now();
        
        // Execute operation
        let result = operation();
        
        let execution_time = start_time.elapsed();
        
        // Post-measurement system state
        self.system.refresh_all();
        let final_memory = self.system.total_memory() - self.system.available_memory();
        let final_cpu = self.system.process(pid)
            .map(|p| p.cpu_usage())
            .unwrap_or(0.0);
        
        let metrics = PerformanceMetrics {
            execution_time,
            memory_usage: final_memory.saturating_sub(initial_memory),
            cpu_usage: final_cpu - initial_cpu,
            cache_hits: 0, // Would need hardware counters
            cache_misses: 0,
            operations_per_second: 1.0 / execution_time.as_secs_f64(),
            memory_allocations: 0, // Would need allocation tracking
            peak_memory: final_memory,
        };
        
        (result, metrics)
    }
    
    pub fn identify_bottlenecks(&self, metrics: &PerformanceMetrics) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        
        // Time-based bottlenecks
        if metrics.execution_time.as_millis() > 100 {
            bottlenecks.push("Execution time exceeds 100ms threshold".to_string());
        }
        
        // Memory-based bottlenecks
        if metrics.memory_usage > 100 * 1024 * 1024 { // 100MB
            bottlenecks.push("High memory usage detected".to_string());
        }
        
        // CPU-based bottlenecks
        if metrics.cpu_usage > 80.0 {
            bottlenecks.push("High CPU utilization detected".to_string());
        }
        
        // Cache efficiency bottlenecks
        if metrics.cache_efficiency() < 0.8 {
            bottlenecks.push("Poor cache efficiency detected".to_string());
        }
        
        // Operations per second bottlenecks
        if metrics.operations_per_second < 1000.0 {
            bottlenecks.push("Low throughput detected".to_string());
        }
        
        bottlenecks
    }
    
    pub fn generate_optimization_recommendations(&self, metrics: &PerformanceMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        let bottlenecks = self.identify_bottlenecks(metrics);
        
        for bottleneck in &bottlenecks {
            if bottleneck.contains("Execution time") {
                recommendations.push("Consider enabling SIMD optimizations".to_string());
                recommendations.push("Evaluate parallel processing opportunities".to_string());
                recommendations.push("Profile algorithm complexity and optimize hot paths".to_string());
            }
            
            if bottleneck.contains("memory usage") {
                recommendations.push("Implement memory pooling for frequent allocations".to_string());
                recommendations.push("Consider using bump allocators for temporary data".to_string());
                recommendations.push("Evaluate data structure efficiency".to_string());
            }
            
            if bottleneck.contains("CPU utilization") {
                recommendations.push("Implement load balancing across CPU cores".to_string());
                recommendations.push("Consider algorithmic optimizations".to_string());
                recommendations.push("Evaluate GPU acceleration opportunities".to_string());
            }
            
            if bottleneck.contains("cache efficiency") {
                recommendations.push("Improve data locality and access patterns".to_string());
                recommendations.push("Consider cache-friendly data structures".to_string());
                recommendations.push("Implement data prefetching strategies".to_string());
            }
            
            if bottleneck.contains("throughput") {
                recommendations.push("Implement batch processing".to_string());
                recommendations.push("Consider pipeline parallelization".to_string());
                recommendations.push("Optimize I/O operations".to_string());
            }
        }
        
        recommendations.dedup();
        recommendations
    }
}

// === PERFORMANCE ANALYSIS BENCHMARKS ===

fn bench_black_swan_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/black_swan");
    let mut analyzer = BottleneckAnalyzer::new();
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("performance_analysis", size),
            size,
            |b, &size| {
                let config = BlackSwanConfig {
                    window_size: size.min(1000),
                    use_simd: true,
                    parallel_processing: true,
                    ..Default::default()
                };
                
                let detector = BlackSwanDetector::new(config).expect("Failed to create detector");
                let data = Array1::linspace(0.0, 1.0, size);
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("black_swan_detection", || {
                        detector.detect_anomalies(black_box(&data))
                    });
                    
                    let bottlenecks = analyzer.identify_bottlenecks(&metrics);
                    let recommendations = analyzer.generate_optimization_recommendations(&metrics);
                    
                    if !bottlenecks.is_empty() {
                        eprintln!("Black Swan bottlenecks for size {}: {:?}", size, bottlenecks);
                        eprintln!("Recommendations: {:?}", recommendations);
                    }
                    
                    eprintln!("Black Swan metrics for size {}: efficiency={:.3}, cache_efficiency={:.3}, ops/sec={:.0}",
                             size, metrics.efficiency_score(), metrics.cache_efficiency(), metrics.operations_per_second);
                    
                    black_box((result, metrics))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_soc_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/soc");
    let mut analyzer = BottleneckAnalyzer::new();
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("performance_analysis", size),
            size,
            |b, &size| {
                let params = SOCParameters::default();
                let soc_analyzer = SOCAnalyzer::new(params);
                let data = Array1::linspace(0.0, 1.0, size);
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("soc_analysis", || {
                        soc_analyzer.analyze_series(black_box(&data))
                    });
                    
                    let bottlenecks = analyzer.identify_bottlenecks(&metrics);
                    let recommendations = analyzer.generate_optimization_recommendations(&metrics);
                    
                    if !bottlenecks.is_empty() {
                        eprintln!("SOC bottlenecks for size {}: {:?}", size, bottlenecks);
                        eprintln!("Recommendations: {:?}", recommendations);
                    }
                    
                    eprintln!("SOC metrics for size {}: efficiency={:.3}, cache_efficiency={:.3}, ops/sec={:.0}",
                             size, metrics.efficiency_score(), metrics.cache_efficiency(), metrics.operations_per_second);
                    
                    black_box((result, metrics))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_antifragility_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/antifragility");
    let mut analyzer = BottleneckAnalyzer::new();
    
    for size in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("performance_analysis", size),
            size,
            |b, &size| {
                let mut params = AntifragilityParameters::default();
                params.min_data_points = (size / 10).max(50);
                params.enable_simd = true;
                params.enable_parallel = true;
                
                let af_analyzer = AntifragilityAnalyzer::with_params(params)
                    .expect("Failed to create analyzer");
                let prices = Array1::linspace(100.0, 200.0, size);
                let volumes = Array1::linspace(1000.0, 2000.0, size);
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("antifragility_analysis", || {
                        af_analyzer.analyze_prices(black_box(&prices), black_box(&volumes))
                    });
                    
                    let bottlenecks = analyzer.identify_bottlenecks(&metrics);
                    let recommendations = analyzer.generate_optimization_recommendations(&metrics);
                    
                    if !bottlenecks.is_empty() {
                        eprintln!("Antifragility bottlenecks for size {}: {:?}", size, bottlenecks);
                        eprintln!("Recommendations: {:?}", recommendations);
                    }
                    
                    eprintln!("Antifragility metrics for size {}: efficiency={:.3}, cache_efficiency={:.3}, ops/sec={:.0}",
                             size, metrics.efficiency_score(), metrics.cache_efficiency(), metrics.operations_per_second);
                    
                    black_box((result, metrics))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_stdp_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/stdp");
    let mut analyzer = BottleneckAnalyzer::new();
    
    for neurons in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("performance_analysis", neurons),
            neurons,
            |b, &neurons| {
                let config = STDPConfig {
                    simd_width: 8,
                    parallel_enabled: true,
                    ..Default::default()
                };
                
                let mut stdp_optimizer = STDPOptimizer::new(config)
                    .expect("Failed to create optimizer");
                let weights = Array2::from_elem((neurons, neurons), 0.5);
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("stdp_optimization", || {
                        stdp_optimizer.update_weights(black_box(&weights), 0.01)
                    });
                    
                    let bottlenecks = analyzer.identify_bottlenecks(&metrics);
                    let recommendations = analyzer.generate_optimization_recommendations(&metrics);
                    
                    if !bottlenecks.is_empty() {
                        eprintln!("STDP bottlenecks for {} neurons: {:?}", neurons, bottlenecks);
                        eprintln!("Recommendations: {:?}", recommendations);
                    }
                    
                    eprintln!("STDP metrics for {} neurons: efficiency={:.3}, cache_efficiency={:.3}, ops/sec={:.0}",
                             neurons, metrics.efficiency_score(), metrics.cache_efficiency(), metrics.operations_per_second);
                    
                    black_box((result, metrics))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/memory_patterns");
    let mut analyzer = BottleneckAnalyzer::new();
    
    for size in [1000, 10000, 100000].iter() {
        // Analyze different allocation patterns
        group.bench_with_input(
            BenchmarkId::new("vec_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("vec_allocation", || {
                        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
                        black_box(data)
                    });
                    
                    eprintln!("Vec allocation for size {}: memory={}KB, time={}μs", 
                             size, metrics.memory_usage / 1024, metrics.execution_time.as_micros());
                    
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ndarray_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("ndarray_allocation", || {
                        let data = Array1::<f64>::from_iter((0..size).map(|i| i as f64));
                        black_box(data)
                    });
                    
                    eprintln!("ndarray allocation for size {}: memory={}KB, time={}μs", 
                             size, metrics.memory_usage / 1024, metrics.execution_time.as_micros());
                    
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_allocation", size),
            size,
            |b, &size| {
                let dim = (size as f64).sqrt() as usize;
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("matrix_allocation", || {
                        let data = Array2::<f64>::zeros((dim, dim));
                        black_box(data)
                    });
                    
                    eprintln!("Matrix allocation for {}x{}: memory={}KB, time={}μs", 
                             dim, dim, metrics.memory_usage / 1024, metrics.execution_time.as_micros());
                    
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_algorithmic_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis/complexity");
    let mut analyzer = BottleneckAnalyzer::new();
    
    // Analyze complexity scaling for different algorithms
    for size in [100, 500, 1000, 2000].iter() {
        // O(n) algorithm
        group.bench_with_input(
            BenchmarkId::new("linear_complexity", size),
            size,
            |b, &size| {
                let data = Array1::linspace(0.0, 1.0, size);
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("linear_scan", || {
                        data.iter().sum::<f64>()
                    });
                    
                    let ops_per_ns = size as f64 / metrics.execution_time.as_nanos() as f64;
                    eprintln!("Linear O(n) for size {}: {:.3} ops/ns", size, ops_per_ns);
                    
                    black_box(result)
                })
            },
        );
        
        // O(n log n) algorithm
        group.bench_with_input(
            BenchmarkId::new("nlogn_complexity", size),
            size,
            |b, &size| {
                let mut data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
                
                b.iter(|| {
                    let (result, metrics) = analyzer.measure_performance("sort_operation", || {
                        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        data.clone()
                    });
                    
                    let theoretical_ops = size as f64 * (size as f64).log2();
                    let actual_ns = metrics.execution_time.as_nanos() as f64;
                    let efficiency = theoretical_ops / actual_ns;
                    eprintln!("Sort O(n log n) for size {}: {:.3} efficiency", size, efficiency);
                    
                    black_box(result)
                })
            },
        );
        
        // O(n²) algorithm (for smaller sizes only)
        if *size <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("quadratic_complexity", size),
                size,
                |b, &size| {
                    let data = Array1::linspace(0.0, 1.0, size);
                    
                    b.iter(|| {
                        let (result, metrics) = analyzer.measure_performance("quadratic_operation", || {
                            let mut sum = 0.0;
                            for i in 0..size {
                                for j in 0..size {
                                    sum += data[i] * data[j];
                                }
                            }
                            sum
                        });
                        
                        let theoretical_ops = (size * size) as f64;
                        let actual_ns = metrics.execution_time.as_nanos() as f64;
                        let efficiency = theoretical_ops / actual_ns;
                        eprintln!("Quadratic O(n²) for size {}: {:.3} efficiency", size, efficiency);
                        
                        black_box(result)
                    })
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    performance_analysis,
    bench_black_swan_analysis,
    bench_soc_analysis,
    bench_antifragility_analysis,
    bench_stdp_analysis,
    bench_memory_allocation_patterns,
    bench_algorithmic_complexity
);

criterion_main!(performance_analysis);