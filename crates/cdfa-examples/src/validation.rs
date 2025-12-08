//! CDFA validation binary for Python comparison tests
//!
//! This binary reads test inputs from JSON, runs CDFA algorithms,
//! and outputs results for comparison with Python reference implementation.

use cdfa_core::prelude::*;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::time::Instant;

#[derive(Deserialize)]
struct DiversityInput {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Serialize)]
struct DiversityOutput {
    kendall_tau: f64,
    spearman: f64,
    pearson: f64,
}

#[derive(Deserialize)]
struct JensenShannonInput {
    p: Vec<f64>,
    q: Vec<f64>,
}

#[derive(Serialize)]
struct JensenShannonOutput {
    divergence: f64,
    distance: f64,
}

#[derive(Deserialize)]
struct FusionInput {
    scores: Vec<Vec<f64>>,
    weights: Vec<f64>,
    diversity_threshold: f64,
    score_weight: f64,
}

#[derive(Serialize)]
struct FusionOutput {
    average: Vec<f64>,
    weighted: Vec<f64>,
    borda: Vec<f64>,
    adaptive: Vec<f64>,
}

#[derive(Deserialize)]
struct BenchmarkInput {
    scores: Vec<Vec<f64>>,
    benchmark: bool,
}

#[derive(Serialize)]
struct BenchmarkOutput {
    execution_time_ms: f64,
    result_length: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <test_name> <input_file>", args[0]);
        std::process::exit(1);
    }
    
    let test_name = &args[1];
    let input_file = &args[2];
    
    // Read input
    let input_data = fs::read_to_string(input_file)
        .expect("Failed to read input file");
    
    // Run appropriate test
    match test_name.as_str() {
        "diversity_metrics" => run_diversity_test(&input_data),
        name if name.starts_with("jensen_shannon_") => run_jensen_shannon_test(&input_data, name),
        "fusion_methods" => run_fusion_test(&input_data),
        "performance_benchmark" => run_benchmark_test(&input_data),
        _ => {
            eprintln!("Unknown test: {}", test_name);
            std::process::exit(1);
        }
    }
}

fn run_diversity_test(input_data: &str) {
    let input: DiversityInput = serde_json::from_str(input_data)
        .expect("Failed to parse diversity input");
    
    let x = Array1::from_vec(input.x);
    let y = Array1::from_vec(input.y);
    
    let kendall = kendall_tau(&x, &y).expect("Kendall tau failed");
    let spearman = spearman_correlation(&x, &y).expect("Spearman failed");
    let pearson = pearson_correlation(&x, &y).expect("Pearson failed");
    
    let output = DiversityOutput {
        kendall_tau: kendall,
        spearman,
        pearson,
    };
    
    write_output("diversity_metrics", &output);
}

fn run_jensen_shannon_test(input_data: &str, test_name: &str) {
    let input: JensenShannonInput = serde_json::from_str(input_data)
        .expect("Failed to parse JS input");
    
    let p = Array1::from_vec(input.p);
    let q = Array1::from_vec(input.q);
    
    let divergence = jensen_shannon_divergence(&p, &q).expect("JS divergence failed");
    let distance = jensen_shannon_distance(&p, &q).expect("JS distance failed");
    
    let output = JensenShannonOutput {
        divergence,
        distance,
    };
    
    write_output(test_name, &output);
}

fn run_fusion_test(input_data: &str) {
    let input: FusionInput = serde_json::from_str(input_data)
        .expect("Failed to parse fusion input");
    
    // Convert to ndarray
    let n_sources = input.scores.len();
    let n_items = input.scores[0].len();
    let scores_flat: Vec<f64> = input.scores.into_iter().flatten().collect();
    let scores = Array2::from_shape_vec((n_sources, n_items), scores_flat)
        .expect("Failed to create scores array");
    
    // Run fusion methods
    let average = CdfaFusion::fuse(&scores.view(), FusionMethod::Average, None)
        .expect("Average fusion failed");
    
    let weights = Array1::from_vec(input.weights);
    let weighted_params = FusionParams {
        weights: Some(weights),
        ..Default::default()
    };
    let weighted = CdfaFusion::fuse(&scores.view(), FusionMethod::WeightedAverage, Some(weighted_params))
        .expect("Weighted fusion failed");
    
    let borda = CdfaFusion::fuse(&scores.view(), FusionMethod::BordaCount, None)
        .expect("Borda fusion failed");
    
    let adaptive_params = FusionParams {
        diversity_threshold: input.diversity_threshold,
        score_weight: input.score_weight,
        ..Default::default()
    };
    let adaptive = CdfaFusion::fuse(&scores.view(), FusionMethod::Adaptive, Some(adaptive_params))
        .expect("Adaptive fusion failed");
    
    let output = FusionOutput {
        average: average.to_vec(),
        weighted: weighted.to_vec(),
        borda: borda.to_vec(),
        adaptive: adaptive.to_vec(),
    };
    
    write_output("fusion_methods", &output);
}

fn run_benchmark_test(input_data: &str) {
    let input: BenchmarkInput = serde_json::from_str(input_data)
        .expect("Failed to parse benchmark input");
    
    // Convert to ndarray
    let n_sources = input.scores.len();
    let n_items = input.scores[0].len();
    let scores_flat: Vec<f64> = input.scores.into_iter().flatten().collect();
    let scores = Array2::from_shape_vec((n_sources, n_items), scores_flat)
        .expect("Failed to create scores array");
    
    // Benchmark fusion and correlation matrix
    let start = Instant::now();
    
    // Fusion
    let _ = CdfaFusion::fuse(&scores.view(), FusionMethod::Average, None)
        .expect("Fusion failed");
    
    // Correlation matrix
    let _ = pearson_correlation_matrix(&scores.view())
        .expect("Correlation matrix failed");
    
    let elapsed = start.elapsed();
    
    let output = BenchmarkOutput {
        execution_time_ms: elapsed.as_secs_f64() * 1000.0,
        result_length: n_items,
    };
    
    write_output("performance_benchmark", &output);
}

fn write_output<T: Serialize>(test_name: &str, output: &T) {
    let output_file = format!("/tmp/cdfa_test_output_{}.json", test_name);
    let output_json = serde_json::to_string_pretty(output)
        .expect("Failed to serialize output");
    
    fs::write(&output_file, output_json)
        .expect("Failed to write output file");
    
    println!("Test {} completed. Output written to {}", test_name, output_file);
}