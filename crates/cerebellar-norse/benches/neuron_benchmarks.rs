//! Performance benchmarks for neuron implementations
//! 
//! These benchmarks validate that the cerebellar SNN implementation
//! meets ultra-low latency requirements for trading applications.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;

use cerebellar_norse::*;
use cerebellar_norse::neuron_types::*;

/// Benchmark LIF neuron step function
fn bench_lif_neuron_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("lif_neuron_step");
    
    // Test different batch sizes
    for batch_size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        let device = Device::Cpu;
        let size = *batch_size;
        let params = LIFParameters::default();
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        
        let input = Tensor::randn(&[size], (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &input,
            |b, input| {
                b.iter(|| {
                    state.update(black_box(input), black_box(dt)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark AdEx neuron step function
fn bench_adex_neuron_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adex_neuron_step");
    
    for batch_size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        let device = Device::Cpu;
        let size = *batch_size;
        let params = AdExParameters::default();
        let mut state = AdExState::new(size, params, device.clone()).unwrap();
        
        let input = Tensor::randn(&[size], (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &input,
            |b, input| {
                b.iter(|| {
                    state.update(black_box(input), black_box(dt)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark trading-optimized LIF neuron
fn bench_trading_lif_neuron(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_lif_neuron");
    group.measurement_time(Duration::from_secs(10));
    
    // Test ultra-low latency requirements
    let mut neuron = LIFNeuron::new_trading_optimized();
    
    group.bench_function("single_step", |b| {
        b.iter(|| {
            neuron.step(black_box(1.5))
        })
    });
    
    // Benchmark batch processing
    let neurons = vec![LIFNeuron::new_trading_optimized(); 1000];
    let mut processor = BatchNeuronProcessor::new(neurons);
    let inputs = vec![1.0; 1000];
    
    group.throughput(Throughput::Elements(1000));
    group.bench_function("batch_processing", |b| {
        b.iter(|| {
            processor.process_batch(black_box(&inputs))
        })
    });
    
    group.finish();
}

/// Benchmark neuron factory performance
fn bench_neuron_factory(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_factory");
    
    group.bench_function("granule_cell", |b| {
        b.iter(|| {
            NeuronFactory::create_granule_cell()
        })
    });
    
    group.bench_function("purkinje_cell", |b| {
        b.iter(|| {
            NeuronFactory::create_purkinje_cell()
        })
    });
    
    group.bench_function("golgi_cell", |b| {
        b.iter(|| {
            NeuronFactory::create_golgi_cell()
        })
    });
    
    group.bench_function("dcn_cell", |b| {
        b.iter(|| {
            NeuronFactory::create_dcn_cell()
        })
    });
    
    group.finish();
}

/// Benchmark neuron state creation
fn bench_neuron_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_state_creation");
    let device = Device::Cpu;
    
    for size in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        let lif_params = LIFParameters::default();
        let adex_params = AdExParameters::default();
        
        group.bench_with_input(
            BenchmarkId::new("lif_state", size),
            size,
            |b, &size| {
                b.iter(|| {
                    LIFState::new(size, lif_params.clone(), device.clone()).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("adex_state", size),
            size,
            |b, &size| {
                b.iter(|| {
                    AdExState::new(size, adex_params.clone(), device.clone()).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark neuron update with different input patterns
fn bench_neuron_input_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_input_patterns");
    let device = Device::Cpu;
    let size = 100;
    let params = LIFParameters::default();
    
    // Test with different input patterns
    let patterns = vec![
        ("constant", Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap()),
        ("random", Tensor::randn(&[size], (DType::F32, &device)).unwrap()),
        ("sparse", {
            let mut tensor = Tensor::zeros(&[size], (DType::F32, &device)).unwrap();
            // Set every 10th element to 1.0
            for i in (0..size).step_by(10) {
                tensor = tensor.slice_set(&[i as i64], &Tensor::full(&[], 1.0, (DType::F32, &device)).unwrap()).unwrap();
            }
            tensor
        }),
    ];
    
    for (name, input) in patterns {
        let mut state = LIFState::new(size, params.clone(), device.clone()).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("input_pattern", name),
            &input,
            |b, input| {
                b.iter(|| {
                    state.update(black_box(input), black_box(dt)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark neuron reset performance
fn bench_neuron_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_reset");
    let device = Device::Cpu;
    
    for size in [10, 100, 1000, 10000].iter() {
        let lif_params = LIFParameters::default();
        let adex_params = AdExParameters::default();
        
        group.bench_with_input(
            BenchmarkId::new("lif_reset", size),
            size,
            |b, &size| {
                let mut state = LIFState::new(size, lif_params.clone(), device.clone()).unwrap();
                b.iter(|| {
                    state.reset().unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("adex_reset", size),
            size,
            |b, &size| {
                let mut state = AdExState::new(size, adex_params.clone(), device.clone()).unwrap();
                b.iter(|| {
                    state.reset().unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark neuron dynamics over time
fn bench_neuron_dynamics(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_dynamics");
    let device = Device::Cpu;
    let size = 100;
    
    // Different time constants
    let time_constants = vec![
        ("fast", 5.0, 2.0),
        ("medium", 10.0, 5.0),
        ("slow", 20.0, 10.0),
    ];
    
    for (name, tau_mem, tau_syn) in time_constants {
        let params = LIFParameters {
            tau_mem,
            tau_syn,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("time_constants", name),
            &input,
            |b, input| {
                b.iter(|| {
                    for _ in 0..100 {
                        state.update(black_box(input), black_box(dt)).unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage of neuron states
fn bench_neuron_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_memory");
    let device = Device::Cpu;
    
    // Test memory allocation performance
    for size in [100, 1000, 10000].iter() {
        let params = LIFParameters::default();
        
        group.bench_with_input(
            BenchmarkId::new("memory_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    // Create and immediately drop to test allocation/deallocation
                    let _state = LIFState::new(size, params.clone(), device.clone()).unwrap();
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent neuron operations
fn bench_concurrent_neurons(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_neurons");
    let device = Device::Cpu;
    let size = 1000;
    
    // Test parallel processing
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        
        let neurons: Vec<LIFNeuron> = (0..size)
            .map(|_| LIFNeuron::new_trading_optimized())
            .collect();
        
        let inputs = vec![1.0; size];
        
        group.bench_function("parallel_processing", |b| {
            b.iter(|| {
                let mut neurons = neurons.clone();
                neurons.par_iter_mut()
                    .zip(inputs.par_iter())
                    .map(|(neuron, &input)| neuron.step(input))
                    .collect::<Vec<_>>()
            })
        });
    }
    
    group.finish();
}

/// Ultra-low latency benchmark for trading applications
fn bench_ultra_low_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_low_latency");
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(5));
    
    // Single neuron performance (target: < 10ns per step)
    let mut neuron = LIFNeuron::new_trading_optimized();
    
    group.bench_function("single_neuron_10ns_target", |b| {
        b.iter(|| {
            // This should complete in < 10ns for trading requirements
            neuron.step(black_box(1.5))
        })
    });
    
    // Small network performance (target: < 1μs)
    let mut processor = TradingCerebellarProcessor::new();
    
    group.bench_function("small_network_1us_target", |b| {
        b.iter(|| {
            processor.process_tick(black_box(100.0), black_box(1000.0), black_box(1234567890))
        })
    });
    
    // Batch processing performance
    let neurons = vec![LIFNeuron::new_trading_optimized(); 100];
    let mut batch_processor = BatchNeuronProcessor::new(neurons);
    let inputs = vec![1.0; 100];
    
    group.bench_function("batch_100_neurons", |b| {
        b.iter(|| {
            batch_processor.process_batch(black_box(&inputs))
        })
    });
    
    group.finish();
}

/// Benchmark spike generation patterns
fn bench_spike_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_patterns");
    let device = Device::Cpu;
    let size = 100;
    let params = LIFParameters::default();
    
    // Test different spike generation scenarios
    let scenarios = vec![
        ("no_spikes", 0.1),      // Below threshold
        ("sparse_spikes", 0.8),   // Just above threshold
        ("dense_spikes", 2.0),    // Well above threshold
        ("very_dense", 5.0),      // Very high input
    ];
    
    for (name, input_amplitude) in scenarios {
        let mut state = LIFState::new(size, params.clone(), device.clone()).unwrap();
        let input = Tensor::full(&[size], input_amplitude, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("spike_scenario", name),
            &input,
            |b, input| {
                b.iter(|| {
                    state.update(black_box(input), black_box(dt)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark neuron parameter variations
fn bench_parameter_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_variations");
    let device = Device::Cpu;
    let size = 50;
    
    // Test different parameter combinations
    let parameter_sets = vec![
        ("granule_like", 8.0, 3.0, 0.8),
        ("purkinje_like", 12.0, 5.0, 1.2),
        ("golgi_like", 15.0, 8.0, 1.0),
        ("dcn_like", 10.0, 4.0, 1.5),
    ];
    
    for (name, tau_mem, tau_syn, v_th) in parameter_sets {
        let params = LIFParameters {
            tau_mem,
            tau_syn,
            v_th,
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        group.bench_with_input(
            BenchmarkId::new("parameter_set", name),
            &input,
            |b, input| {
                b.iter(|| {
                    state.update(black_box(input), black_box(dt)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_lif_neuron_step,
    bench_adex_neuron_step,
    bench_trading_lif_neuron,
    bench_neuron_factory,
    bench_neuron_state_creation,
    bench_neuron_input_patterns,
    bench_neuron_reset,
    bench_neuron_dynamics,
    bench_neuron_memory,
    bench_concurrent_neurons,
    bench_ultra_low_latency,
    bench_spike_patterns,
    bench_parameter_variations,
);

criterion_main!(benches);