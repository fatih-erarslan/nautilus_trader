//! Inference benchmarks for hyperphysics-ml
//!
//! Measures inference latency for various model configurations using
//! real quantum-inspired encoding and neural network operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hyperphysics_ml::prelude::*;
use hyperphysics_ml::backends::Device;
use hyperphysics_ml::tensor::{Tensor, DType};

/// Benchmark quantum-inspired state encoding operations
fn bench_quantum_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_encoding");

    // Different input sizes (powers of 2 for quantum state dimensions)
    for num_qubits in [3, 4, 5, 6] {
        let input_size = 1 << num_qubits; // 8, 16, 32, 64
        let input: Vec<f32> = (0..input_size).map(|i| (i as f32 * 0.01).sin()).collect();

        group.throughput(Throughput::Elements(input_size as u64));

        // Amplitude encoding benchmark
        group.bench_with_input(
            BenchmarkId::new("amplitude", input_size),
            &input,
            |b, input| {
                let encoder = StateEncoder::amplitude(num_qubits);
                b.iter(|| {
                    black_box(encoder.encode(input))
                });
            },
        );

        // Angle encoding benchmark
        group.bench_with_input(
            BenchmarkId::new("angle", input_size),
            &input,
            |b, input| {
                let encoder = StateEncoder::angle(num_qubits);
                b.iter(|| {
                    black_box(encoder.encode(input))
                });
            },
        );

        // Phase encoding benchmark
        group.bench_with_input(
            BenchmarkId::new("phase", input_size),
            &input,
            |b, input| {
                let encoder = StateEncoder::phase(num_qubits);
                b.iter(|| {
                    black_box(encoder.encode(input))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark time series encoding for financial data
fn bench_time_series_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_series_encoding");

    // Different sequence lengths typical for HFT
    for seq_len in [10, 20, 50, 100] {
        // Simulate OHLCV market data (5 features per timestep)
        let features = 5;
        let data: Vec<f32> = (0..seq_len * features)
            .map(|i| ((i as f32 * 0.1).sin() + 1.0) * 50.0) // Price-like values
            .collect();

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("ohlcv_encoding", seq_len),
            &data,
            |b, data| {
                let encoder = TimeSeriesEncoder::new(EncodingType::Amplitude, 4);
                // Create windows of size 4 from the data
                let windows: Vec<Vec<f32>> = data.chunks(4)
                    .map(|c| c.to_vec())
                    .collect();
                b.iter(|| {
                    black_box(encoder.encode_batch(&windows))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tensor operations used in neural network inference
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_ops");

    // Get default device (CPU)
    let device = Device::default();

    for size in [64, 128, 256, 512] {
        // Create test tensors
        let data_a: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();
        let data_b: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.002).cos()).collect();

        let tensor_a = Tensor::from_slice(&data_a, vec![size, size], &device).unwrap();
        let tensor_b = Tensor::from_slice(&data_b, vec![size, size], &device).unwrap();

        group.throughput(Throughput::Elements((size * size) as u64));

        // Matrix multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &(tensor_a.clone(), tensor_b.clone()),
            |b, (a, b_tensor)| {
                b.iter(|| {
                    black_box(a.matmul(b_tensor))
                });
            },
        );

        // Element-wise operations
        group.bench_with_input(
            BenchmarkId::new("elementwise_add", size),
            &(tensor_a.clone(), tensor_b.clone()),
            |b, (a, b_tensor)| {
                b.iter(|| {
                    black_box(a.add(b_tensor))
                });
            },
        );

        // Activation function (tanh)
        group.bench_with_input(
            BenchmarkId::new("tanh_activation", size),
            &tensor_a,
            |b, tensor| {
                b.iter(|| {
                    black_box(tensor.tanh())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bio-cognitive pattern recognition
fn bench_bio_cognitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("bio_cognitive");

    let device = Device::default();

    for seq_len in [10, 20, 50] {
        let input_size = 4;
        let hidden_size = 64;

        // Create bio-cognitive config
        let config = BioCognitiveConfig::new(input_size, hidden_size);

        // Simulate biological signal patterns (EEG-like)
        let input: Vec<f32> = (0..seq_len * input_size)
            .map(|i| {
                let t = i as f32 * 0.01;
                // Simulate alpha wave (10Hz) + beta wave (20Hz)
                (t * 10.0 * std::f32::consts::PI * 2.0).sin() * 0.5 +
                (t * 20.0 * std::f32::consts::PI * 2.0).sin() * 0.3
            })
            .collect();

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("pattern_recognition", seq_len),
            &input,
            |b, input| {
                b.iter(|| {
                    let mut lstm = BioCognitiveLSTM::from_config(config.clone());
                    let mut state = BioCognitiveState::new(hidden_size);

                    // Process in chunks of input_size
                    for chunk in input.chunks(input_size) {
                        let (output, new_state) = lstm.forward_simple(chunk, &state);
                        state = new_state;
                        black_box(&output);
                    }
                    state
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complex number operations (foundation of quantum-inspired computations)
fn bench_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_ops");

    for size in [64, 256, 1024, 4096] {
        // Create complex number arrays
        let complexes: Vec<Complex> = (0..size)
            .map(|i| Complex::new((i as f32 * 0.01).sin(), (i as f32 * 0.01).cos()))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Complex multiplication
        group.bench_with_input(
            BenchmarkId::new("complex_mul", size),
            &complexes,
            |b, data| {
                b.iter(|| {
                    let mut result = Complex::one();
                    for c in data.iter() {
                        result = result * *c;
                    }
                    black_box(result)
                });
            },
        );

        // Complex exponential (e^(i*theta))
        group.bench_with_input(
            BenchmarkId::new("complex_exp", size),
            &complexes,
            |b, data| {
                b.iter(|| {
                    let results: Vec<Complex> = data.iter()
                        .map(|c| c.exp())
                        .collect();
                    black_box(results)
                });
            },
        );

        // Norm calculation
        group.bench_with_input(
            BenchmarkId::new("complex_norm", size),
            &complexes,
            |b, data| {
                b.iter(|| {
                    let norm_sq: f32 = data.iter()
                        .map(|c| c.norm_sq())
                        .sum();
                    black_box(norm_sq.sqrt())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state vector operations (quantum-inspired superposition)
fn bench_state_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector");

    for num_qubits in [4, 6, 8, 10] {
        let dim = 1 << num_qubits;

        group.throughput(Throughput::Elements(dim as u64));

        // State vector creation
        group.bench_with_input(
            BenchmarkId::new("creation", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    black_box(StateVector::new(n))
                });
            },
        );

        // Uniform superposition
        group.bench_with_input(
            BenchmarkId::new("uniform_superposition", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    black_box(StateVector::uniform(n))
                });
            },
        );

        // Normalization
        let state = StateVector::uniform(num_qubits);
        group.bench_with_input(
            BenchmarkId::new("normalize", num_qubits),
            &state,
            |b, s| {
                b.iter(|| {
                    let mut state_copy = s.clone();
                    state_copy.normalize();
                    black_box(state_copy)
                });
            },
        );

        // Inner product
        let state1 = StateVector::uniform(num_qubits);
        let state2 = StateVector::uniform(num_qubits);
        group.bench_with_input(
            BenchmarkId::new("inner_product", num_qubits),
            &(state1, state2),
            |b, (s1, s2)| {
                b.iter(|| {
                    black_box(s1.inner_product(s2))
                });
            },
        );

        // Probability calculation
        let state = StateVector::uniform(num_qubits);
        group.bench_with_input(
            BenchmarkId::new("probabilities", num_qubits),
            &state,
            |b, s| {
                b.iter(|| {
                    black_box(s.probabilities())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantum_encoding,
    bench_time_series_encoding,
    bench_tensor_operations,
    bench_bio_cognitive,
    bench_complex_operations,
    bench_state_vector,
);

criterion_main!(benches);
