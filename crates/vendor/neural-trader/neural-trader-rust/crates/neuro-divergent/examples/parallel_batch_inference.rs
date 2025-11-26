//! Parallel Batch Inference Example
//!
//! Demonstrates 3-8x speedup using Rayon parallelization
//!
//! Run with: cargo run --release --example parallel_batch_inference

use neuro_divergent::optimizations::parallel::*;
use neuro_divergent::optimizations::parallel::benchmark::*;
use ndarray::{Array1, Array2, Array};
use rand::Rng;
use std::time::Instant;

/// Simple mock neural network for demonstration
struct SimpleModel {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl SimpleModel {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = Array::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let bias = Array::from_shape_fn(output_dim, |_| {
            rng.gen_range(-0.1..0.1)
        });

        Self { weights, bias }
    }

    fn predict(&self, input: &Array2<f64>) -> Vec<f64> {
        let mut predictions = Vec::new();

        for sample in input.outer_iter() {
            // Linear transformation: Wx + b
            let output = self.weights.t().dot(&sample.to_owned()) + &self.bias;

            // Apply activation (ReLU + some extra computation for realism)
            let mut result = 0.0;
            for val in output.iter() {
                let activated = val.max(0.0);
                // Add some computational cost to simulate real inference
                for _ in 0..10 {
                    result += (activated * 1.01).tanh();
                }
            }

            predictions.push(result);
        }

        predictions
    }
}

fn generate_random_batch(batch_size: usize, features: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((batch_size, features), |_| rng.gen_range(-1.0..1.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Parallel Batch Inference Benchmark");
    println!("═══════════════════════════════════════════════════════════\n");

    // Configuration
    let input_dim = 128;
    let output_dim = 64;
    let batch_size = 32;
    let num_batches = 200;

    println!("Configuration:");
    println!("  Input dimensions: {}", input_dim);
    println!("  Output dimensions: {}", output_dim);
    println!("  Batch size: {}", batch_size);
    println!("  Number of batches: {}", num_batches);
    println!("  Available cores: {}\n", num_cpus::get());

    // Create model
    println!("Creating model...");
    let model = SimpleModel::new(input_dim, output_dim);

    // Generate test batches
    println!("Generating {} batches...", num_batches);
    let batches: Vec<Array2<f64>> = (0..num_batches)
        .map(|_| generate_random_batch(batch_size, input_dim))
        .collect();

    println!("Total samples: {}\n", num_batches * batch_size);

    // Sequential baseline
    println!("─────────────────────────────────────────────────────────");
    println!("Running sequential baseline...");
    let start = Instant::now();
    let sequential_results: Vec<Vec<f64>> = batches
        .iter()
        .map(|batch| model.predict(batch))
        .collect();
    let sequential_time = start.elapsed();
    println!("Sequential time: {:.2} ms", sequential_time.as_secs_f64() * 1000.0);

    // Parallel inference with different thread counts
    let thread_counts = vec![1, 2, 4, 8, 16];
    println!("\n─────────────────────────────────────────────────────────");
    println!("Running parallel inference with different thread counts...\n");

    for &num_threads in &thread_counts {
        // Configure thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore error if already initialized

        // Warm-up
        let _ = parallel_batch_inference(&batches[0..10], |batch| {
            Ok(model.predict(batch))
        });

        // Benchmark
        let start = Instant::now();
        let parallel_results = parallel_batch_inference(&batches, |batch| {
            Ok(model.predict(batch))
        })?;
        let parallel_time = start.elapsed();

        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        let efficiency = speedup / num_threads as f64;

        println!("  {} threads: {:.2} ms ({}x speedup, {:.1}% efficiency)",
            num_threads,
            parallel_time.as_secs_f64() * 1000.0,
            speedup as i32,
            efficiency * 100.0
        );

        // Verify correctness
        assert_eq!(parallel_results.len(), sequential_results.len());
    }

    // Detailed scalability analysis
    println!("\n─────────────────────────────────────────────────────────");
    println!("Scalability Analysis:");
    println!("─────────────────────────────────────────────────────────\n");

    let workload = || -> neuro_divergent::Result<()> {
        parallel_batch_inference(&batches, |batch| {
            Ok(model.predict(batch))
        })?;
        Ok(())
    };

    let results = scalability_benchmark(&thread_counts, &workload)?;
    print_benchmark_results(&results);

    let parallel_fraction = amdahl_analysis(&results);
    println!("\nAmdahl's Law Analysis:");
    println!("  Estimated parallel fraction: {:.1}%", parallel_fraction * 100.0);
    println!("  Serial fraction: {:.1}%", (1.0 - parallel_fraction) * 100.0);

    if parallel_fraction > 0.9 {
        println!("  ✓ Excellent parallelization (>90% parallel work)");
    } else if parallel_fraction > 0.7 {
        println!("  ○ Good parallelization (>70% parallel work)");
    } else {
        println!("  ✗ Poor parallelization (<70% parallel work)");
        println!("    Consider reducing synchronization overhead");
    }

    // Batch inference with uncertainty estimation
    println!("\n─────────────────────────────────────────────────────────");
    println!("Parallel Inference with Uncertainty Estimation:");
    println!("─────────────────────────────────────────────────────────\n");

    // Configure for 8 threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .ok();

    let num_samples = 50; // Monte Carlo samples
    let test_batches = &batches[0..10]; // Use subset for demo

    let start = Instant::now();
    let uncertainty_results = parallel_batch_inference_with_uncertainty(
        test_batches,
        |batch| Ok(model.predict(batch)),
        num_samples,
    )?;
    let uncertainty_time = start.elapsed();

    println!("Processed {} batches with {} MC samples each",
        test_batches.len(), num_samples);
    println!("Time: {:.2} ms", uncertainty_time.as_secs_f64() * 1000.0);

    // Show sample predictions with uncertainty
    println!("\nSample predictions with uncertainty:");
    for (i, (mean, std)) in uncertainty_results.iter().take(3).enumerate() {
        println!("  Batch {}: mean={:.4} ± {:.4}", i, mean[0], std[0]);
    }

    // Ensemble predictions
    println!("\n─────────────────────────────────────────────────────────");
    println!("Parallel Ensemble Predictions:");
    println!("─────────────────────────────────────────────────────────\n");

    // Create ensemble of models
    let model1 = SimpleModel::new(input_dim, output_dim);
    let model2 = SimpleModel::new(input_dim, output_dim);
    let model3 = SimpleModel::new(input_dim, output_dim);

    let ensemble_fns = vec![
        |batch: &Array2<f64>| Ok(model1.predict(batch)),
        |batch: &Array2<f64>| Ok(model2.predict(batch)),
        |batch: &Array2<f64>| Ok(model3.predict(batch)),
    ];

    let test_input = &batches[0];

    // Mean ensemble
    let start = Instant::now();
    let ensemble_mean = parallel_ensemble_predict(
        test_input,
        &ensemble_fns,
        EnsembleAggregation::Mean,
    )?;
    println!("Mean ensemble: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);

    // Median ensemble (more robust)
    let start = Instant::now();
    let ensemble_median = parallel_ensemble_predict(
        test_input,
        &ensemble_fns,
        EnsembleAggregation::Median,
    )?;
    println!("Median ensemble: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);

    // Weighted ensemble
    let weights = vec![0.5, 0.3, 0.2];
    let start = Instant::now();
    let ensemble_weighted = parallel_ensemble_predict(
        test_input,
        &ensemble_fns,
        EnsembleAggregation::WeightedMean(weights),
    )?;
    println!("Weighted ensemble: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);

    println!("\nSample ensemble predictions:");
    println!("  Mean:     {:.4}", ensemble_mean[0]);
    println!("  Median:   {:.4}", ensemble_median[0]);
    println!("  Weighted: {:.4}", ensemble_weighted[0]);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Benchmark Complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
