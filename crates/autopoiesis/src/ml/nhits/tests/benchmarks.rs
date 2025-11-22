#[cfg(feature = "benchmarks")]
use super::*;

#[cfg(feature = "benchmarks")]
use crate::ml::nhits::{NHITSModel, NHITSConfig};

#[cfg(feature = "benchmarks")]
use crate::ml::nhits::consciousness::ConsciousnessIntegration;

#[cfg(feature = "benchmarks")]
use ndarray::{Array2, Array3};

#[cfg(feature = "benchmarks")]
use std::time::{Duration, Instant};

#[cfg(feature = "benchmarks")]
use std::collections::HashMap;

#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "benchmarks")]
use criterion::{BenchmarkId, Throughput};

#[cfg(feature = "benchmarks")]
pub struct BenchmarkSuite {
    models: HashMap<String, NHITSModel>,
    test_data: HashMap<String, (Array2<f32>, Array2<f32>)>,
}

#[cfg(feature = "benchmarks")]
impl BenchmarkSuite {
    pub fn new() -> Self {
        let mut suite = BenchmarkSuite {
            models: HashMap::new(),
            test_data: HashMap::new(),
        };
        
        suite.setup_models();
        suite.setup_test_data();
        suite
    }
    
    fn setup_models(&mut self) {
        // Small model for fast benchmarks
        let small_config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 2,
            num_blocks: [1, 1],
            num_layers: [2, 2],
            layer_widths: [128, 128],
            pooling_kernel_sizes: [2, 2],
            n_freq_downsample: [4, 2],
            batch_size: 32,
            dropout: 0.1,
            ..Default::default()
        };
        self.models.insert("small", NHITSModel::new(small_config));
        
        // Medium model
        let medium_config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 3,
            num_blocks: [1, 1, 1],
            num_layers: [3, 3, 3],
            layer_widths: [256, 256, 256],
            pooling_kernel_sizes: [2, 2, 2],
            n_freq_downsample: [8, 4, 2],
            batch_size: 32,
            dropout: 0.1,
            ..Default::default()
        };
        self.models.insert("medium", NHITSModel::new(medium_config));
        
        // Large model
        let large_config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 4,
            num_blocks: [2, 2, 2, 2],
            num_layers: [4, 4, 4, 4],
            layer_widths: [512, 512, 512, 512],
            pooling_kernel_sizes: [2, 2, 2, 2],
            n_freq_downsample: [16, 8, 4, 2],
            batch_size: 32,
            dropout: 0.1,
            ..Default::default()
        };
        self.models.insert("large", NHITSModel::new(large_config));
        
        // Consciousness-enabled model
        let mut consciousness_model = NHITSModel::new(medium_config.clone());
        consciousness_model.enable_consciousness(512, 8, 4);
        self.models.insert("consciousness", consciousness_model);
    }
    
    fn setup_test_data(&mut self) {
        // Single sample for latency testing
        let single_x = Array2::from_shape_fn((1, 168), |(_, j)| (j as f32 * 0.01).sin());
        let single_y = Array2::from_shape_fn((1, 24), |(_, j)| (j as f32 * 0.01 + 1.0).sin());
        self.test_data.insert("single".to_string(), (single_x, single_y));
        
        // Small batch
        let small_batch_x = Array2::from_shape_fn((8, 168), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01).sin());
        let small_batch_y = Array2::from_shape_fn((8, 24), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01 + 1.0).sin());
        self.test_data.insert("small_batch".to_string(), (small_batch_x, small_batch_y));
        
        // Medium batch
        let medium_batch_x = Array2::from_shape_fn((32, 168), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01).sin());
        let medium_batch_y = Array2::from_shape_fn((32, 24), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01 + 1.0).sin());
        self.test_data.insert("medium_batch".to_string(), (medium_batch_x, medium_batch_y));
        
        // Large batch
        let large_batch_x = Array2::from_shape_fn((128, 168), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01).sin());
        let large_batch_y = Array2::from_shape_fn((128, 24), |(i, j)| 
            (i as f32 * 0.1 + j as f32 * 0.01 + 1.0).sin());
        self.test_data.insert("large_batch".to_string(), (large_batch_x, large_batch_y));
    }
}

// Benchmark functions
pub fn benchmark_inference_latency(c: &mut Criterion) {
    let suite = BenchmarkSuite::new();
    let mut group = c.benchmark_group("inference_latency");
    
    for (model_name, model) in &suite.models {
        let (input, _) = &suite.test_data["single"];
        
        group.bench_with_input(
            BenchmarkId::new("single_sample", model_name),
            model_name,
            |b, _| {
                b.iter(|| {
                    black_box(model.forward(black_box(input)))
                })
            },
        );
    }
    
    group.finish();
}

pub fn benchmark_batch_throughput(c: &mut Criterion) {
    let suite = BenchmarkSuite::new();
    let mut group = c.benchmark_group("batch_throughput");
    
    for batch_size in &["small_batch", "medium_batch", "large_batch"] {
        for (model_name, model) in &suite.models {
            let (input, _) = &suite.test_data[*batch_size];
            let samples_per_batch = input.shape()[0];
            
            group.throughput(Throughput::Elements(samples_per_batch as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", model_name, batch_size), samples_per_batch),
                &(model, input),
                |b, (model, input)| {
                    b.iter(|| {
                        black_box(model.forward(black_box(input)))
                    })
                },
            );
        }
    }
    
    group.finish();
}

pub fn benchmark_training_step(c: &mut Criterion) {
    let suite = BenchmarkSuite::new();
    let mut group = c.benchmark_group("training_step");
    
    for (model_name, model) in &suite.models {
        let mut model_clone = model.clone();
        let (input, target) = &suite.test_data["medium_batch"];
        
        group.bench_with_input(
            BenchmarkId::new("training_step", model_name),
            model_name,
            |b, _| {
                b.iter(|| {
                    black_box(model_clone.train_step(black_box(input), black_box(target)))
                })
            },
        );
    }
    
    group.finish();
}

pub fn benchmark_consciousness_overhead(c: &mut Criterion) {
    let suite = BenchmarkSuite::new();
    let mut group = c.benchmark_group("consciousness_overhead");
    
    let regular_model = &suite.models["medium"];
    let consciousness_model = &suite.models["consciousness"];
    let (input, _) = &suite.test_data["medium_batch"];
    
    group.bench_function("regular_forward", |b| {
        b.iter(|| {
            black_box(regular_model.forward(black_box(input)))
        })
    });
    
    group.bench_function("consciousness_forward", |b| {
        b.iter(|| {
            black_box(consciousness_model.forward_with_consciousness(black_box(input)))
        })
    });
    
    group.finish();
}

pub fn benchmark_memory_usage(c: &mut Criterion) {
    let suite = BenchmarkSuite::new();
    let mut group = c.benchmark_group("memory_usage");
    
    for (model_name, model) in &suite.models {
        group.bench_with_input(
            BenchmarkId::new("memory_footprint", model_name),
            model_name,
            |b, _| {
                b.iter(|| {
                    let params = black_box(model.get_parameters());
                    let memory_size = params.len() * std::mem::size_of::<f32>();
                    black_box(memory_size)
                })
            },
        );
    }
    
    group.finish();
}

// Performance profiling functions
pub struct PerformanceProfiler {
    results: HashMap<String, Vec<Duration>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        PerformanceProfiler {
            results: HashMap::new(),
        }
    }
    
    pub fn profile_inference_scaling(&mut self) {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128];
        
        for &batch_size in &batch_sizes {
            let input = Array2::ones((batch_size, 168));
            let mut times = Vec::new();
            
            // Warm up
            for _ in 0..10 {
                let _ = model.forward(&input);
            }
            
            // Actual measurements
            for _ in 0..100 {
                let start = Instant::now();
                let _ = model.forward(&input);
                times.push(start.elapsed());
            }
            
            self.results.insert(format!("batch_size_{}", batch_size), times);
        }
    }
    
    pub fn profile_model_size_scaling(&mut self) {
        let layer_widths = vec![64, 128, 256, 512, 1024];
        
        for &width in &layer_widths {
            let config = NHITSConfig {
                layer_widths: [width, width, width],
                ..Default::default()
            };
            
            let model = NHITSModel::new(config);
            let input = Array2::ones((32, 168));
            let mut times = Vec::new();
            
            // Warm up
            for _ in 0..10 {
                let _ = model.forward(&input);
            }
            
            // Actual measurements
            for _ in 0..50 {
                let start = Instant::now();
                let _ = model.forward(&input);
                times.push(start.elapsed());
            }
            
            self.results.insert(format!("width_{}", width), times);
        }
    }
    
    pub fn profile_consciousness_modes(&mut self) {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        model.enable_consciousness(512, 8, 4);
        
        let input = Array2::ones((32, 168));
        let modes = vec!["analytical", "creative", "balanced"];
        
        for mode in modes {
            model.set_consciousness_mode(mode);
            let mut times = Vec::new();
            
            // Warm up
            for _ in 0..10 {
                let _ = model.forward_with_consciousness(&input);
            }
            
            // Actual measurements
            for _ in 0..50 {
                let start = Instant::now();
                let _ = model.forward_with_consciousness(&input);
                times.push(start.elapsed());
            }
            
            self.results.insert(format!("consciousness_{}", mode), times);
        }
    }
    
    pub fn generate_report(&self) -> String {
        let mut report = String::from("Performance Profile Report\n");
        report.push_str("==========================\n\n");
        
        for (test_name, times) in &self.results {
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let min_time = times.iter().min().unwrap();
            let max_time = times.iter().max().unwrap();
            
            let avg_micros = avg_time.as_micros();
            let min_micros = min_time.as_micros();
            let max_micros = max_time.as_micros();
            
            report.push_str(&format!(
                "{}: avg={}μs, min={}μs, max={}μs\n",
                test_name, avg_micros, min_micros, max_micros
            ));
        }
        
        report
    }
}

// Memory profiling
pub struct MemoryProfiler {
    baseline_memory: usize,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        MemoryProfiler {
            baseline_memory: Self::get_memory_usage(),
        }
    }
    
    fn get_memory_usage() -> usize {
        // Simplified memory usage estimation
        // In a real implementation, you would use proper memory profiling tools
        std::mem::size_of::<NHITSModel>()
    }
    
    pub fn profile_model_memory(&self, model: &NHITSModel) -> HashMap<String, usize> {
        let mut memory_usage = HashMap::new();
        
        // Parameter memory
        let params = model.get_parameters();
        let param_memory = params.len() * std::mem::size_of::<f32>();
        memory_usage.insert("parameters".to_string(), param_memory);
        
        // Model structure memory
        let model_memory = std::mem::size_of_val(model);
        memory_usage.insert("model_structure".to_string(), model_memory);
        
        // Consciousness memory (if enabled)
        if model.consciousness.is_some() {
            let consciousness_memory = std::mem::size_of::<ConsciousnessIntegration>();
            memory_usage.insert("consciousness".to_string(), consciousness_memory);
        }
        
        memory_usage
    }
    
    pub fn profile_training_memory(&self, model: &mut NHITSModel, batch_size: usize) -> HashMap<String, usize> {
        let mut memory_usage = HashMap::new();
        
        let input = Array2::ones((batch_size, 168));
        let target = Array2::zeros((batch_size, 24));
        
        let start_memory = Self::get_memory_usage();
        
        // Forward pass memory
        let _ = model.forward(&input);
        let forward_memory = Self::get_memory_usage() - start_memory;
        memory_usage.insert("forward_pass".to_string(), forward_memory);
        
        // Gradient computation memory
        let _ = model.compute_gradients(&input, &target);
        let gradient_memory = Self::get_memory_usage() - start_memory - forward_memory;
        memory_usage.insert("gradients".to_string(), gradient_memory);
        
        memory_usage
    }
}

// Criterion benchmark registration
#[cfg(feature = "benchmarks")]
criterion_group!(
    benches,
    benchmark_inference_latency,
    benchmark_batch_throughput,
    benchmark_training_step,
    benchmark_consciousness_overhead,
    benchmark_memory_usage
);

#[cfg(feature = "benchmarks")]
criterion_main!(benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();
        profiler.profile_inference_scaling();
        
        let report = profiler.generate_report();
        assert!(!report.is_empty());
        assert!(report.contains("Performance Profile Report"));
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let memory_usage = profiler.profile_model_memory(&model);
        assert!(memory_usage.contains_key("parameters"));
        assert!(memory_usage.contains_key("model_structure"));
    }

    #[test]
    fn test_benchmark_suite_setup() {
        let suite = BenchmarkSuite::new();
        
        assert!(suite.models.contains_key("small"));
        assert!(suite.models.contains_key("medium"));
        assert!(suite.models.contains_key("large"));
        assert!(suite.models.contains_key("consciousness"));
        
        assert!(suite.test_data.contains_key("single"));
        assert!(suite.test_data.contains_key("small_batch"));
        assert!(suite.test_data.contains_key("medium_batch"));
        assert!(suite.test_data.contains_key("large_batch"));
    }

    #[test]
    fn test_consciousness_performance_overhead() {
        let config = NHITSConfig::default();
        let regular_model = NHITSModel::new(config.clone());
        
        let mut consciousness_model = NHITSModel::new(config);
        consciousness_model.enable_consciousness(256, 8, 3);
        
        let input = Array2::ones((32, 168));
        
        // Measure regular model
        let start = Instant::now();
        for _ in 0..100 {
            let _ = regular_model.forward(&input);
        }
        let regular_time = start.elapsed();
        
        // Measure consciousness model
        let start = Instant::now();
        for _ in 0..100 {
            let _ = consciousness_model.forward_with_consciousness(&input);
        }
        let consciousness_time = start.elapsed();
        
        // Consciousness should have some overhead but not excessive
        let overhead_ratio = consciousness_time.as_secs_f64() / regular_time.as_secs_f64();
        assert!(overhead_ratio < 5.0, "Consciousness overhead too high: {}x", overhead_ratio);
    }
}