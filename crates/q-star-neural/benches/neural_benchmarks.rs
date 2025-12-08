//! Q* Neural Network Benchmarks
//!
//! This benchmark suite measures the performance of neural network components
//! used in the Q* algorithm.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Mock neural network structures for benchmarking
#[derive(Debug, Clone)]
pub struct QStarNeuralNetwork {
    pub layers: Vec<Layer>,
    pub activation_function: ActivationFunction,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub neurons: usize,
}

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Leaky,
}

impl QStarNeuralNetwork {
    pub fn new(layers: Vec<usize>) -> Self {
        let mut network_layers = Vec::new();
        
        for i in 0..layers.len() - 1 {
            let input_size = layers[i];
            let output_size = layers[i + 1];
            
            let weights = vec![vec![0.1; input_size]; output_size];
            let biases = vec![0.0; output_size];
            
            network_layers.push(Layer {
                weights,
                biases,
                neurons: output_size,
            });
        }
        
        Self {
            layers: network_layers,
            activation_function: ActivationFunction::ReLU,
        }
    }
    
    pub async fn forward(&self, input: &[f64]) -> Vec<f64> {
        // Mock forward pass
        tokio::time::sleep(Duration::from_micros(10)).await;
        vec![0.5; self.layers.last().unwrap().neurons]
    }
    
    pub async fn backward(&self, _gradient: &[f64]) -> Vec<f64> {
        // Mock backward pass
        tokio::time::sleep(Duration::from_micros(15)).await;
        vec![0.1; self.layers.first().unwrap().weights[0].len()]
    }
    
    pub async fn train(&mut self, _inputs: &[Vec<f64>], _targets: &[Vec<f64>]) {
        // Mock training
        tokio::time::sleep(Duration::from_micros(50)).await;
    }
    
    pub fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            activation_function: self.activation_function.clone(),
        }
    }
}

/// Benchmark neural network creation
fn bench_neural_network_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network_creation");
    
    let architectures = vec![
        vec![10, 5, 1],
        vec![20, 10, 5, 1],
        vec![50, 25, 10, 1],
        vec![100, 50, 25, 10, 1],
        vec![256, 128, 64, 32, 1],
    ];
    
    for (i, architecture) in architectures.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("architecture", i),
            architecture,
            |b, architecture| {
                b.iter(|| {
                    let network = QStarNeuralNetwork::new(architecture.clone());
                    black_box(network)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark forward pass
fn bench_forward_pass(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("forward_pass");
    
    let input_sizes = vec![10, 50, 100, 256, 512];
    
    for size in input_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("input_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let network = QStarNeuralNetwork::new(vec![size, size / 2, 1]);
                        let input = vec![0.5; size];
                        (network, input)
                    },
                    |(network, input)| async move {
                        let result = network.forward(&input).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark backward pass
fn bench_backward_pass(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backward_pass");
    
    let gradient_sizes = vec![1, 5, 10, 25, 50];
    
    for size in gradient_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("gradient_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let network = QStarNeuralNetwork::new(vec![100, 50, size]);
                        let gradient = vec![0.1; size];
                        (network, gradient)
                    },
                    |(network, gradient)| async move {
                        let result = network.backward(&gradient).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark training
fn bench_training(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("training");
    
    let batch_sizes = vec![1, 8, 16, 32, 64];
    
    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let mut network = QStarNeuralNetwork::new(vec![10, 5, 1]);
                        let inputs: Vec<Vec<f64>> = (0..batch_size)
                            .map(|_| vec![0.5; 10])
                            .collect();
                        let targets: Vec<Vec<f64>> = (0..batch_size)
                            .map(|_| vec![0.8])
                            .collect();
                        (network, inputs, targets)
                    },
                    |(mut network, inputs, targets)| async move {
                        network.train(&inputs, &targets).await;
                        black_box(network)
                    },
                );
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_neural_network_creation,
    bench_forward_pass,
    bench_backward_pass,
    bench_training
);

criterion_main!(benches);