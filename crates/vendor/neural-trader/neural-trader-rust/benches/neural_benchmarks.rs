// Benchmarks for neural network operations
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

fn benchmark_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply");

    for size in [10, 50, 100, 200].iter() {
        let matrix_a: Vec<Vec<f64>> = (0..*size)
            .map(|i| (0..*size).map(|j| (i * j) as f64).collect())
            .collect();

        let matrix_b: Vec<Vec<f64>> = (0..*size)
            .map(|i| (0..*size).map(|j| (i + j) as f64).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _size| {
                b.iter(|| {
                    let mut result = vec![vec![0.0; *size]; *size];
                    for i in 0..*size {
                        for j in 0..*size {
                            for k in 0..*size {
                                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
                            }
                        }
                    }
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_activation_functions(c: &mut Criterion) {
    let inputs: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01 - 5.0).collect();

    c.bench_function("relu", |b| {
        b.iter(|| {
            let outputs: Vec<f64> = inputs.iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect();
            black_box(outputs);
        });
    });

    c.bench_function("sigmoid", |b| {
        b.iter(|| {
            let outputs: Vec<f64> = inputs.iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            black_box(outputs);
        });
    });

    c.bench_function("tanh", |b| {
        b.iter(|| {
            let outputs: Vec<f64> = inputs.iter()
                .map(|&x| x.tanh())
                .collect();
            black_box(outputs);
        });
    });
}

fn benchmark_forward_pass(c: &mut Criterion) {
    // Simple 2-layer network
    let input_size = 50;
    let hidden_size = 100;
    let output_size = 10;

    let input: Vec<f64> = (0..input_size).map(|i| i as f64 * 0.1).collect();
    let weights1: Vec<Vec<f64>> = (0..hidden_size)
        .map(|_| (0..input_size).map(|i| i as f64 * 0.01).collect())
        .collect();
    let weights2: Vec<Vec<f64>> = (0..output_size)
        .map(|_| (0..hidden_size).map(|i| i as f64 * 0.01).collect())
        .collect();

    c.bench_function("neural_forward_pass", |b| {
        b.iter(|| {
            // Hidden layer
            let mut hidden: Vec<f64> = vec![0.0; hidden_size];
            for i in 0..hidden_size {
                for j in 0..input_size {
                    hidden[i] += input[j] * weights1[i][j];
                }
                hidden[i] = if hidden[i] > 0.0 { hidden[i] } else { 0.0 }; // ReLU
            }

            // Output layer
            let mut output: Vec<f64> = vec![0.0; output_size];
            for i in 0..output_size {
                for j in 0..hidden_size {
                    output[i] += hidden[j] * weights2[i][j];
                }
            }

            black_box(output);
        });
    });
}

fn benchmark_batch_normalization(c: &mut Criterion) {
    let batch_size = 32;
    let features = 100;

    let batch: Vec<Vec<f64>> = (0..batch_size)
        .map(|i| (0..features).map(|j| (i * j) as f64).collect())
        .collect();

    c.bench_function("batch_normalization", |b| {
        b.iter(|| {
            // Calculate mean for each feature
            let mut means = vec![0.0; features];
            for sample in &batch {
                for (j, &val) in sample.iter().enumerate() {
                    means[j] += val;
                }
            }
            for mean in &mut means {
                *mean /= batch_size as f64;
            }

            // Calculate variance
            let mut variances = vec![0.0; features];
            for sample in &batch {
                for (j, &val) in sample.iter().enumerate() {
                    variances[j] += (val - means[j]).powi(2);
                }
            }
            for var in &mut variances {
                *var /= batch_size as f64;
            }

            // Normalize
            let mut normalized = vec![vec![0.0; features]; batch_size];
            for (i, sample) in batch.iter().enumerate() {
                for (j, &val) in sample.iter().enumerate() {
                    normalized[i][j] = (val - means[j]) / (variances[j] + 1e-8).sqrt();
                }
            }

            black_box(normalized);
        });
    });
}

fn benchmark_softmax(c: &mut Criterion) {
    let logits: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

    c.bench_function("softmax", |b| {
        b.iter(|| {
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter()
                .map(|&x| (x - max_logit).exp())
                .sum();

            let probs: Vec<f64> = logits.iter()
                .map(|&x| (x - max_logit).exp() / exp_sum)
                .collect();

            black_box(probs);
        });
    });
}

fn benchmark_gradient_descent(c: &mut Criterion) {
    let params: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
    let gradients: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001).sin()).collect();
    let learning_rate = 0.01;

    c.bench_function("gradient_update", |b| {
        b.iter(|| {
            let updated: Vec<f64> = params.iter()
                .zip(gradients.iter())
                .map(|(p, g)| p - learning_rate * g)
                .collect();
            black_box(updated);
        });
    });
}

criterion_group! {
    name = neural_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = benchmark_matrix_multiply,
              benchmark_activation_functions,
              benchmark_forward_pass,
              benchmark_batch_normalization,
              benchmark_softmax,
              benchmark_gradient_descent
}

criterion_main!(neural_benches);
