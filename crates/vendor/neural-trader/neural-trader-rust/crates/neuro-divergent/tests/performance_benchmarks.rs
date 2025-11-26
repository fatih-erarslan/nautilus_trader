//! Performance Benchmark Tests
//!
//! Validates that Rust implementation achieves performance targets:
//! - 3-5x faster training than Python
//! - 2-4x faster inference than Python
//! - Lower memory footprint

use neuro_divergent::*;
use std::time::Instant;

/// Performance metrics for comparison
#[derive(Debug)]
struct PerformanceMetrics {
    training_time_ms: f64,
    inference_time_ms: f64,
    memory_mb: f64,
    throughput_samples_per_sec: f64,
}

/// Generate large dataset for benchmarking
fn generate_benchmark_data(samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64;
            10.0 * (t / 100.0).sin() + 0.5 * t + rand::random::<f64>() * 2.0
        })
        .collect()
}

#[test]
#[ignore] // Run manually with `cargo test --release -- --ignored --nocapture`
fn benchmark_nhits_training_speed() {
    let training_data = generate_benchmark_data(10000);
    let python_baseline_ms = 45000.0; // 45 seconds from Python baseline

    // TODO: Implement NHITS model
    // let mut model = NHITSModel::new(config);
    //
    // let start = Instant::now();
    // model.fit(&training_data).unwrap();
    // let rust_time_ms = start.elapsed().as_millis() as f64;
    //
    // let speedup = python_baseline_ms / rust_time_ms;
    //
    // println!("NHITS Training Performance:");
    // println!("  Python baseline: {:.2}s", python_baseline_ms / 1000.0);
    // println!("  Rust implementation: {:.2}s", rust_time_ms / 1000.0);
    // println!("  Speedup: {:.2}x", speedup);
    //
    // assert!(
    //     speedup >= 3.0,
    //     "Training speedup {:.2}x is below 3.0x target",
    //     speedup
    // );

    println!("NHITS training benchmark placeholder - target: 3-5x speedup");
}

#[test]
#[ignore]
fn benchmark_nhits_inference_speed() {
    let training_data = generate_benchmark_data(1000);
    let python_baseline_ms = 120.0; // 120ms from Python baseline

    // TODO: Implement NHITS model
    // let mut model = NHITSModel::new(config);
    // model.fit(&training_data).unwrap();
    //
    // let start = Instant::now();
    // let _ = model.predict(24).unwrap();
    // let rust_time_ms = start.elapsed().as_millis() as f64;
    //
    // let speedup = python_baseline_ms / rust_time_ms;
    //
    // println!("NHITS Inference Performance:");
    // println!("  Python baseline: {:.2}ms", python_baseline_ms);
    // println!("  Rust implementation: {:.2}ms", rust_time_ms);
    // println!("  Speedup: {:.2}x", speedup);
    //
    // assert!(
    //     speedup >= 2.0,
    //     "Inference speedup {:.2}x is below 2.0x target",
    //     speedup
    // );

    println!("NHITS inference benchmark placeholder - target: 2-4x speedup");
}

#[test]
#[ignore]
fn benchmark_batch_inference() {
    let training_data = generate_benchmark_data(1000);
    let batch_sizes = vec![1, 8, 16, 32, 64];

    println!("Batch Inference Benchmarks:");
    println!("{:>10} | {:>15} | {:>20}", "Batch Size", "Time (ms)", "Throughput (pred/s)");
    println!("{:-<50}", "");

    for &batch_size in &batch_sizes {
        // TODO: Implement batch prediction
        // let throughput = measure_batch_throughput(&model, batch_size);
        // println!("{:>10} | {:>15.2} | {:>20.2}", batch_size, time_ms, throughput);

        println!("{:>10} | {:>15} | {:>20}", batch_size, "TODO", "TODO");
    }
}

#[test]
#[ignore]
fn benchmark_memory_usage() {
    // TODO: Implement memory profiling
    // let initial_memory = get_process_memory_mb();
    //
    // let training_data = generate_benchmark_data(10000);
    // let mut model = NHITSModel::new(config);
    // model.fit(&training_data).unwrap();
    //
    // let final_memory = get_process_memory_mb();
    // let memory_used = final_memory - initial_memory;
    //
    // println!("Memory Usage:");
    // println!("  Training data: 10,000 samples");
    // println!("  Memory used: {:.2} MB", memory_used);
    //
    // // Python baseline uses ~500MB for this dataset
    // assert!(
    //     memory_used < 250.0,
    //     "Memory usage {:.2}MB exceeds 250MB target",
    //     memory_used
    // );

    println!("Memory benchmark placeholder - target: <250MB for 10k samples");
}

#[test]
#[ignore]
fn benchmark_cross_validation_performance() {
    let data = generate_benchmark_data(1000);
    let n_splits = 5;

    // TODO: Implement cross-validation
    // let start = Instant::now();
    // let cv_results = cross_validate(&model, &data, n_splits).unwrap();
    // let total_time_ms = start.elapsed().as_millis() as f64;
    //
    // println!("Cross-Validation Performance:");
    // println!("  Splits: {}", n_splits);
    // println!("  Total time: {:.2}s", total_time_ms / 1000.0);
    // println!("  Time per split: {:.2}s", total_time_ms / (n_splits as f64 * 1000.0));

    println!("Cross-validation benchmark placeholder");
}

/// Compare performance across all implemented models
#[test]
#[ignore]
fn benchmark_all_models_comparison() {
    let training_data = generate_benchmark_data(1000);
    let models = vec!["nhits", "nbeats", "tft", "lstm", "gru"];

    println!("Model Performance Comparison:");
    println!("{:>15} | {:>15} | {:>15} | {:>10}", "Model", "Train (ms)", "Infer (ms)", "Speedup");
    println!("{:-<60}", "");

    for model_name in models {
        // TODO: Benchmark each model
        println!("{:>15} | {:>15} | {:>15} | {:>10}", model_name, "TODO", "TODO", "TODO");
    }
}

#[test]
#[ignore]
fn stress_test_large_dataset() {
    // Test with very large dataset
    let sizes = vec![10_000, 50_000, 100_000, 500_000];

    println!("Large Dataset Stress Test:");
    println!("{:>10} | {:>15} | {:>15}", "Samples", "Train (s)", "Infer (ms)");
    println!("{:-<45}", "");

    for size in sizes {
        let data = generate_benchmark_data(size);

        // TODO: Test with large dataset
        // let start = Instant::now();
        // model.fit(&data).unwrap();
        // let train_time = start.elapsed().as_secs_f64();
        //
        // let start = Instant::now();
        // model.predict(24).unwrap();
        // let infer_time = start.elapsed().as_millis();
        //
        // println!("{:>10} | {:>15.2} | {:>15}", size, train_time, infer_time);

        println!("{:>10} | {:>15} | {:>15}", size, "TODO", "TODO");
    }
}

#[test]
fn benchmark_data_preprocessing() {
    let data = generate_benchmark_data(10000);

    let start = Instant::now();
    let normalized = normalize_data(&data);
    let preprocess_time = start.elapsed().as_micros();

    println!("Data Preprocessing:");
    println!("  Samples: {}", data.len());
    println!("  Time: {}μs", preprocess_time);
    println!("  Throughput: {:.2} samples/μs", data.len() as f64 / preprocess_time as f64);

    assert_eq!(normalized.len(), data.len());
}

fn normalize_data(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64)
        .sqrt();

    data.iter()
        .map(|&x| (x - mean) / std)
        .collect()
}
