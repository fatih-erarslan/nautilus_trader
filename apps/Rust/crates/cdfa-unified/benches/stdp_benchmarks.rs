//! STDP Optimizer Performance Benchmarks
//!
//! Comprehensive benchmarks for the STDP optimizer to validate sub-microsecond performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

use cdfa_unified::optimizers::*;
use ndarray::Array2;

fn bench_stdp_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp_initialization");
    
    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("weights", size), size, |b, &size| {
            let config = STDPConfig::default();
            let optimizer = STDPOptimizer::new(config).unwrap();
            
            b.iter(|| {
                let weights = optimizer.initialize_weights(size, size);
                black_box(weights)
            });
        });
    }
    
    group.finish();
}

fn bench_stdp_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp_application");
    group.measurement_time(Duration::from_secs(10));
    
    let config = STDPConfig {
        simd_width: 8,
        parallel_enabled: true,
        ..Default::default()
    };
    let optimizer = STDPOptimizer::new(config).unwrap();
    
    for &network_size in [50, 100, 200].iter() {
        for &spike_count in [10, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new("network_spikes", format!("{}x{}_spikes_{}", network_size, network_size, spike_count)),
                &(network_size, spike_count),
                |b, &(net_size, spikes)| {
                    let weights = optimizer.initialize_weights(net_size, net_size).unwrap();
                    
                    let pre_spikes: Vec<SpikeEvent> = (0..spikes)
                        .map(|i| SpikeEvent {
                            neuron_id: i % net_size,
                            timestamp: i as f64 * 0.1,
                            amplitude: 1.0,
                        })
                        .collect();
                    
                    let post_spikes: Vec<SpikeEvent> = (0..spikes)
                        .map(|i| SpikeEvent {
                            neuron_id: (i + 10) % net_size,
                            timestamp: i as f64 * 0.1 + 5.0,
                            amplitude: 1.0,
                        })
                        .collect();
                    
                    b.iter(|| {
                        let result = optimizer.apply_stdp(&pre_spikes, &post_spikes, &weights, None);
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_temporal_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_patterns");
    
    let optimizer = STDPOptimizer::new(STDPConfig::default()).unwrap();
    
    for &sequence_count in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("pattern_learning", sequence_count),
            &sequence_count,
            |b, &count| {
                let spike_sequences: Vec<Vec<SpikeEvent>> = (0..count)
                    .map(|seq_id| {
                        (0..5)
                            .map(|i| SpikeEvent {
                                neuron_id: i,
                                timestamp: seq_id as f64 * 100.0 + i as f64 * 10.0,
                                amplitude: 1.0,
                            })
                            .collect()
                    })
                    .collect();
                
                b.iter(|| {
                    let patterns = optimizer.learn_temporal_patterns(&spike_sequences);
                    black_box(patterns)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    for &pool_size in [64 * 1024, 256 * 1024, 1024 * 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_pool", pool_size),
            &pool_size,
            |b, &size| {
                let config = STDPConfig {
                    memory_pool_size: size,
                    ..Default::default()
                };
                
                b.iter(|| {
                    let optimizer = STDPOptimizer::new(config.clone());
                    black_box(optimizer)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_simd_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_performance");
    
    // Compare different SIMD widths
    for &simd_width in [1, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("simd_width", simd_width),
            &simd_width,
            |b, &width| {
                let config = STDPConfig {
                    simd_width: width,
                    ..Default::default()
                };
                let optimizer = STDPOptimizer::new(config).unwrap();
                let weights = optimizer.initialize_weights(200, 200).unwrap();
                
                let pre_spikes: Vec<SpikeEvent> = (0..50)
                    .map(|i| SpikeEvent {
                        neuron_id: i % 200,
                        timestamp: i as f64,
                        amplitude: 1.0,
                    })
                    .collect();
                
                let post_spikes: Vec<SpikeEvent> = (0..50)
                    .map(|i| SpikeEvent {
                        neuron_id: (i + 20) % 200,
                        timestamp: i as f64 + 5.0,
                        amplitude: 1.0,
                    })
                    .collect();
                
                b.iter(|| {
                    let result = optimizer.apply_stdp(&pre_spikes, &post_spikes, &weights, None);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_large_scale_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_networks");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    
    let config = STDPConfig {
        simd_width: 8,
        parallel_enabled: true,
        memory_pool_size: 10 * 1024 * 1024, // 10MB
        ..Default::default()
    };
    let optimizer = STDPOptimizer::new(config).unwrap();
    
    for &size in [500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("network_size", size),
            &size,
            |b, &network_size| {
                let weights = optimizer.initialize_weights(network_size, network_size).unwrap();
                
                let pre_spikes: Vec<SpikeEvent> = (0..network_size)
                    .step_by(2)
                    .map(|i| SpikeEvent {
                        neuron_id: i,
                        timestamp: i as f64 * 0.1,
                        amplitude: 1.0,
                    })
                    .collect();
                
                let post_spikes: Vec<SpikeEvent> = (0..network_size)
                    .step_by(3)
                    .map(|i| SpikeEvent {
                        neuron_id: i,
                        timestamp: i as f64 * 0.1 + 10.0,
                        amplitude: 1.0,
                    })
                    .collect();
                
                b.iter(|| {
                    let result = optimizer.apply_stdp(&pre_spikes, &post_spikes, &weights, None);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_stdp_initialization,
    bench_stdp_application,
    bench_temporal_patterns,
    bench_memory_efficiency,
    bench_simd_performance,
    bench_large_scale_networks
);
criterion_main!(benches);