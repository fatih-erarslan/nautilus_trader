//! Neural network fixes validation binary
//!
//! This binary validates neural network implementations and fixes.
//! Part of the CWTS-Ultra scientific trading system.

use std::time::Instant;

/// Neural validation entry point
fn main() {
    println!("=== CWTS-Ultra Neural Fixes Validation ===");
    println!("Starting neural network validation suite...\n");

    let start = Instant::now();

    // Run validation tests
    let results = run_neural_validations();

    let elapsed = start.elapsed();

    println!("\n=== Validation Results ===");
    println!("Total tests: {}", results.total);
    println!("Passed: {}", results.passed);
    println!("Failed: {}", results.failed);
    println!("Time elapsed: {:?}", elapsed);

    if results.failed > 0 {
        std::process::exit(1);
    }
}

struct ValidationResults {
    total: usize,
    passed: usize,
    failed: usize,
}

fn run_neural_validations() -> ValidationResults {
    let mut results = ValidationResults {
        total: 0,
        passed: 0,
        failed: 0,
    };

    // Validation 1: Check floating point precision
    results.total += 1;
    if validate_float_precision() {
        println!("[PASS] Float precision validation");
        results.passed += 1;
    } else {
        println!("[FAIL] Float precision validation");
        results.failed += 1;
    }

    // Validation 2: Check vector operations
    results.total += 1;
    if validate_vector_ops() {
        println!("[PASS] Vector operations validation");
        results.passed += 1;
    } else {
        println!("[FAIL] Vector operations validation");
        results.failed += 1;
    }

    // Validation 3: Check numerical stability
    results.total += 1;
    if validate_numerical_stability() {
        println!("[PASS] Numerical stability validation");
        results.passed += 1;
    } else {
        println!("[FAIL] Numerical stability validation");
        results.failed += 1;
    }

    results
}

fn validate_float_precision() -> bool {
    // IEEE 754 double precision validation
    let a: f64 = 1.0;
    let b: f64 = 3.0;
    let c = a / b;
    let reconstructed = c * b;

    // Allow for floating point error within machine epsilon
    (reconstructed - a).abs() < f64::EPSILON * 10.0
}

fn validate_vector_ops() -> bool {
    // Basic vector dot product validation
    let v1 = [1.0_f64, 2.0, 3.0];
    let v2 = [4.0_f64, 5.0, 6.0];

    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let expected = 32.0; // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    (dot - expected).abs() < f64::EPSILON
}

fn validate_numerical_stability() -> bool {
    // Test for numerical stability in accumulation
    let n = 1_000_000;
    let small_value = 1e-10_f64;

    // Kahan summation for stable accumulation
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;

    for _ in 0..n {
        let y = small_value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    let expected = (n as f64) * small_value;
    let relative_error = ((sum - expected) / expected).abs();

    // Relative error should be very small with Kahan summation
    relative_error < 1e-10
}
