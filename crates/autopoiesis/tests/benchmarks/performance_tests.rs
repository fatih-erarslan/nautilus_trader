//! Performance benchmarks for autopoiesis framework
//! Tests system performance under various loads and conditions

#[cfg(feature = "benchmarks")]
use autopoiesis::core::*;

#[cfg(feature = "benchmarks")]
use autopoiesis::consciousness::*;

#[cfg(feature = "benchmarks")]
use autopoiesis::emergence::*;

#[cfg(feature = "benchmarks")]
use autopoiesis::dynamics::*;

#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

/// Benchmark data generators
#[cfg(feature = "benchmarks")]
pub mod bench_utils {
    use super::*;
    use rand::Rng;
    use std::collections::VecDeque;
    
    pub fn generate_consciousness_system(size: usize) -> ConsciousnessSystem {
        ConsciousnessSystem::new((size, size, size), 40.0)
    }
    
    pub fn generate_emergence_history(length: usize) -> EmergenceHistory {
        let mut history = EmergenceHistory {
            metrics_history: VecDeque::new(),
            phase_trajectories: VecDeque::new(),
            avalanche_events: VecDeque::new(),
            fitness_evolution: VecDeque::new(),
            lattice_states: VecDeque::new(),
        };
        
        let mut rng = rand::thread_rng();
        for i in 0..length {
            let metrics = SystemMetrics {
                timestamp: i as f64,
                system_size: 100,
                total_energy: 1000.0 + rng.gen_range(-100.0..100.0),
                entropy: 50.0 + rng.gen_range(-10.0..10.0),
                information: rng.gen_range(0.0..1.0),
                complexity: rng.gen_range(0.0..1.0),
                coherence: rng.gen_range(0.0..1.0),
                coupling: 0.5,
            };
            history.metrics_history.push_back(metrics);
        }
        
        history
    }
    
    pub fn generate_test_swarm(size: usize) -> TestSwarmSystem {
        use crate::tests::integration::swarm_dynamics_tests::TestSwarmSystem;
        TestSwarmSystem::new(size)
    }
}

/// Benchmark consciousness system performance
fn bench_consciousness_cycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("consciousness_cycle");
    
    for size in [3, 5, 7, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("process_cycle", size),
            size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let mut system = bench_utils::generate_consciousness_system(size);
                    system.initialize_coherent_consciousness();
                    
                    black_box(system.process_consciousness_cycle(0.01, None));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark consciousness simulation performance
fn bench_consciousness_simulation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("consciousness_simulation");
    
    for duration in [0.1, 0.5, 1.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("run_simulation", duration),
            duration,
            |b, &duration| {
                b.to_async(&rt).iter(|| async {
                    let mut system = bench_utils::generate_consciousness_system(5);
                    system.initialize_coherent_consciousness();
                    
                    let states = black_box(system.run_consciousness(duration, 0.001));
                    states.len()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark emergence detection performance
fn bench_emergence_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("emergence_detection");
    
    for history_length in [50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("detect_emergence", history_length),
            history_length,
            |b, &history_length| {
                let history = bench_utils::generate_emergence_history(history_length);
                let mut detector = EmergenceDetector::new(DetectionParameters::default());
                
                b.iter(|| {
                    detector.update_from_history(black_box(&history));
                    black_box(detector.get_emergence_state());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pattern recognition performance
fn bench_pattern_recognition(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_recognition");
    
    for history_length in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("analyze_patterns", history_length),
            history_length,
            |b, &history_length| {
                let history = bench_utils::generate_emergence_history(history_length);
                let recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
                
                b.iter(|| {
                    let patterns = black_box(recognizer.analyze_patterns(&history));
                    patterns.len()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark complete emergence analysis
fn bench_emergence_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("emergence_analysis");
    
    for history_length in [100, 300, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("full_analysis", history_length),
            history_length,
            |b, &history_length| {
                let history = bench_utils::generate_emergence_history(history_length);
                let mut analysis_system = EmergenceAnalysisSystem::new(
                    DetectionParameters::default(),
                    PatternParameters::default(),
                    AnalysisParameters::default(),
                );
                
                b.iter(|| {
                    let result = black_box(analysis_system.analyze_emergence(&history));
                    result.confidence
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    // Benchmark large data structure operations
    group.bench_function("large_vector_operations", |b| {
        b.iter(|| {
            let mut data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
            
            // Simulate complex operations
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let sum: f64 = data.iter().sum();
            let mean = sum / data.len() as f64;
            let variance: f64 = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / data.len() as f64;
            
            black_box(variance.sqrt());
        });
    });
    
    group.bench_function("hash_map_operations", |b| {
        use std::collections::HashMap;
        
        b.iter(|| {
            let mut map = HashMap::new();
            
            // Insert many values
            for i in 0..1000 {
                map.insert(format!("key_{}", i), i as f64);
            }
            
            // Lookup values
            let mut sum = 0.0;
            for i in 0..1000 {
                if let Some(&value) = map.get(&format!("key_{}", i)) {
                    sum += value;
                }
            }
            
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_operations");
    
    group.bench_function("parallel_consciousness_systems", |b| {
        b.to_async(&rt).iter(|| async {
            let mut handles = Vec::new();
            
            // Spawn multiple consciousness systems
            for i in 0..4 {
                let handle = tokio::spawn(async move {
                    let mut system = bench_utils::generate_consciousness_system(3);
                    system.initialize_coherent_consciousness();
                    
                    // Run short simulation
                    for _ in 0..10 {
                        system.process_consciousness_cycle(0.01, None);
                    }
                    
                    system.get_consciousness_state().integration_metrics.overall_integration
                });
                handles.push(handle);
            }
            
            // Wait for all to complete
            let mut results = Vec::new();
            for handle in handles {
                results.push(handle.await.unwrap());
            }
            
            black_box(results);
        });
    });
    
    group.finish();
}

/// Benchmark mathematical operations
fn bench_mathematical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mathematical_operations");
    
    group.bench_function("matrix_operations", |b| {
        use nalgebra as na;
        
        b.iter(|| {
            let size = 50;
            let mut matrix = na::DMatrix::<f64>::zeros(size, size);
            
            // Fill matrix with random values
            for i in 0..size {
                for j in 0..size {
                    matrix[(i, j)] = (i * j) as f64 / (size * size) as f64;
                }
            }
            
            // Perform operations
            let transpose = matrix.transpose();
            let product = &matrix * &transpose;
            let trace = product.trace();
            
            black_box(trace);
        });
    });
    
    group.bench_function("statistical_calculations", |b| {
        b.iter(|| {
            let data: Vec<f64> = (0..1000).map(|i| {
                (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos()
            }).collect();
            
            // Calculate statistics
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / data.len() as f64;
            let std_dev = variance.sqrt();
            
            // Calculate autocorrelation at lag 1
            let mut autocorr = 0.0;
            for i in 1..data.len() {
                autocorr += (data[i] - mean) * (data[i-1] - mean);
            }
            autocorr /= (data.len() - 1) as f64 * variance;
            
            black_box((std_dev, autocorr));
        });
    });
    
    group.finish();
}

/// Benchmark serialization/deserialization
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    group.bench_function("json_serialization", |b| {
        use serde_json;
        
        let metrics = SystemMetrics {
            timestamp: 123.456,
            system_size: 1000,
            total_energy: 5000.0,
            entropy: 250.0,
            information: 0.75,
            complexity: 0.85,
            coherence: 0.65,
            coupling: 0.55,
        };
        
        b.iter(|| {
            let json = serde_json::to_string(black_box(&metrics)).unwrap();
            let deserialized: SystemMetrics = serde_json::from_str(&json).unwrap();
            black_box(deserialized);
        });
    });
    
    group.finish();
}

/// Benchmark data processing pipelines
fn bench_data_pipelines(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_pipelines");
    
    group.bench_function("emergence_data_pipeline", |b| {
        b.iter(|| {
            // Generate raw data
            let history = bench_utils::generate_emergence_history(500);
            
            // Process through detection pipeline
            let mut detector = EmergenceDetector::new(DetectionParameters::default());
            detector.update_from_history(&history);
            let emergence_state = detector.get_emergence_state();
            
            // Process through pattern recognition pipeline
            let recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
            let patterns = recognizer.analyze_patterns(&history);
            
            // Generate analysis
            let mut analysis_system = EmergenceAnalysisSystem::new(
                DetectionParameters::default(),
                PatternParameters::default(),
                AnalysisParameters::default(),
            );
            let result = analysis_system.analyze_emergence(&history);
            
            black_box((emergence_state, patterns, result));
        });
    });
    
    group.finish();
}

/// Benchmark edge case performance
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");
    
    group.bench_function("empty_data_structures", |b| {
        b.iter(|| {
            let empty_history = EmergenceHistory {
                metrics_history: std::collections::VecDeque::new(),
                phase_trajectories: std::collections::VecDeque::new(),
                avalanche_events: std::collections::VecDeque::new(),
                fitness_evolution: std::collections::VecDeque::new(),
                lattice_states: std::collections::VecDeque::new(),
            };
            
            let mut detector = EmergenceDetector::new(DetectionParameters::default());
            detector.update_from_history(black_box(&empty_history));
            
            let state = detector.get_emergence_state();
            black_box(state);
        });
    });
    
    group.bench_function("single_element_operations", |b| {
        b.iter(|| {
            let mut single_history = EmergenceHistory {
                metrics_history: std::collections::VecDeque::new(),
                phase_trajectories: std::collections::VecDeque::new(),
                avalanche_events: std::collections::VecDeque::new(),
                fitness_evolution: std::collections::VecDeque::new(),
                lattice_states: std::collections::VecDeque::new(),
            };
            
            single_history.metrics_history.push_back(SystemMetrics {
                timestamp: 0.0,
                system_size: 1,
                total_energy: 100.0,
                entropy: 10.0,
                information: 0.5,
                complexity: 0.3,
                coherence: 0.7,
                coupling: 0.4,
            });
            
            let recognizer = TemporalPatternRecognizer::new(PatternParameters::default());
            let patterns = recognizer.analyze_patterns(black_box(&single_history));
            
            black_box(patterns);
        });
    });
    
    group.finish();
}

/// Benchmark stress testing scenarios
fn bench_stress_tests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("stress_tests");
    group.sample_size(10); // Fewer samples for stress tests
    
    group.bench_function("high_frequency_updates", |b| {
        b.to_async(&rt).iter(|| async {
            let mut system = bench_utils::generate_consciousness_system(4);
            system.initialize_coherent_consciousness();
            
            // Simulate high-frequency updates
            for _ in 0..1000 {
                system.process_consciousness_cycle(0.0001, None); // Very small time step
            }
            
            black_box(system.get_consciousness_state());
        });
    });
    
    group.bench_function("large_scale_analysis", |b| {
        b.iter(|| {
            let large_history = bench_utils::generate_emergence_history(5000);
            
            let mut analysis_system = EmergenceAnalysisSystem::new(
                DetectionParameters::default(),
                PatternParameters::default(),
                AnalysisParameters {
                    memory_retention: 5000,
                    ..AnalysisParameters::default()
                },
            );
            
            let result = analysis_system.analyze_emergence(black_box(&large_history));
            black_box(result.confidence);
        });
    });
    
    group.finish();
}

/// Custom benchmark runner for integration with existing test framework
pub fn run_custom_benchmarks() {
    println!("Running custom performance benchmarks...");
    
    // Benchmark consciousness system scaling
    benchmark_consciousness_scaling();
    
    // Benchmark emergence detection scaling
    benchmark_emergence_scaling();
    
    // Benchmark memory usage patterns
    benchmark_memory_usage();
    
    println!("Custom benchmarks completed.");
}

fn benchmark_consciousness_scaling() {
    println!("Benchmarking consciousness system scaling...");
    
    let rt = Runtime::new().unwrap();
    let dimensions = vec![(2, 2, 2), (3, 3, 3), (5, 5, 5), (7, 7, 7), (10, 10, 10)];
    
    for dim in dimensions {
        let start = Instant::now();
        
        rt.block_on(async {
            let mut system = ConsciousnessSystem::new(dim, 40.0);
            system.initialize_coherent_consciousness();
            
            // Run standard benchmark
            for _ in 0..100 {
                system.process_consciousness_cycle(0.01, None);
            }
        });
        
        let duration = start.elapsed();
        let size = dim.0 * dim.1 * dim.2;
        println!("Size {}: {:?} ({:.2} µs per cycle)", 
                size, duration, duration.as_micros() as f64 / 100.0);
    }
}

fn benchmark_emergence_scaling() {
    println!("Benchmarking emergence detection scaling...");
    
    let history_sizes = vec![100, 500, 1000, 2000, 5000];
    
    for size in history_sizes {
        let history = bench_utils::generate_emergence_history(size);
        let mut detector = EmergenceDetector::new(DetectionParameters::default());
        
        let start = Instant::now();
        
        // Run detection multiple times
        for _ in 0..10 {
            detector.update_from_history(&history);
            detector.get_emergence_state();
        }
        
        let duration = start.elapsed();
        println!("History size {}: {:?} ({:.2} ms per detection)", 
                size, duration, duration.as_millis() as f64 / 10.0);
    }
}

fn benchmark_memory_usage() {
    println!("Benchmarking memory usage patterns...");
    
    // This is a simplified memory usage test
    // In a real implementation, you'd use tools like valgrind or custom allocators
    
    let large_system_start = Instant::now();
    {
        let _large_system = bench_utils::generate_consciousness_system(15);
        // System goes out of scope here
    }
    let large_system_duration = large_system_start.elapsed();
    
    let large_history_start = Instant::now();
    {
        let _large_history = bench_utils::generate_emergence_history(10000);
        // History goes out of scope here
    }
    let large_history_duration = large_history_start.elapsed();
    
    println!("Large consciousness system creation/destruction: {:?}", large_system_duration);
    println!("Large emergence history creation/destruction: {:?}", large_history_duration);
}

// Criterion benchmark groups
#[cfg(feature = "benchmarks")]
criterion_group!(
    benches,
    bench_consciousness_cycle,
    bench_consciousness_simulation,
    bench_emergence_detection,
    bench_pattern_recognition,
    bench_emergence_analysis,
    bench_memory_operations,
    bench_concurrent_operations,
    bench_mathematical_operations,
    bench_serialization,
    bench_data_pipelines,
    bench_edge_cases,
    bench_stress_tests
);

#[cfg(feature = "benchmarks")]
criterion_main!(benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    
    #[test]
    fn test_custom_benchmarks() {
        // Quick smoke test to ensure benchmarks can run
        run_custom_benchmarks();
    }
    
    #[test]
    fn test_benchmark_utilities() {
        let system = bench_utils::generate_consciousness_system(3);
        assert_eq!(system.dimensions, (3, 3, 3));
        
        let history = bench_utils::generate_emergence_history(10);
        assert_eq!(history.metrics_history.len(), 10);
    }
}