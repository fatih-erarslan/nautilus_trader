//! SIMD Correctness Tests
//!
//! Verify that SIMD implementations produce identical results to scalar versions.

use neuro_divergent::optimizations::simd::{matmul, activations, losses, utils};

const EPSILON: f32 = 1e-4;

fn assert_vec_approx_eq(a: &[f32], b: &[f32], epsilon: f32) {
    assert_eq!(a.len(), b.len(), "Vector lengths differ");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < epsilon,
            "Mismatch at index {}: {} vs {} (diff: {})",
            i, av, bv, (av - bv).abs()
        );
    }
}

#[test]
fn test_gemm_correctness() {
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let b = vec![
        vec![7.0, 8.0],
        vec![9.0, 10.0],
        vec![11.0, 12.0],
    ];

    let c = matmul::gemm(&a, &b);

    // Expected result:
    // [1*7+2*9+3*11  1*8+2*10+3*12]   [58  64]
    // [4*7+5*9+6*11  4*8+5*10+6*12] = [139 154]

    assert_eq!(c.len(), 2);
    assert_eq!(c[0].len(), 2);
    assert!((c[0][0] - 58.0).abs() < EPSILON);
    assert!((c[0][1] - 64.0).abs() < EPSILON);
    assert!((c[1][0] - 139.0).abs() < EPSILON);
    assert!((c[1][1] - 154.0).abs() < EPSILON);
}

#[test]
fn test_dot_product_correctness() {
    let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32) * 2.0).collect();

    let result = matmul::dot_product(&a, &b);

    // Compute expected result
    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!((result - expected).abs() < EPSILON);
}

#[test]
fn test_relu_correctness() {
    let x: Vec<f32> = vec![-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0];
    let result = activations::relu(&x);

    assert_vec_approx_eq(&result, &[0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 5.0], EPSILON);
}

#[test]
fn test_sigmoid_correctness() {
    let x = vec![0.0, 1.0, -1.0, 2.0];
    let result = activations::sigmoid(&x);

    let expected: Vec<f32> = x.iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();

    assert_vec_approx_eq(&result, &expected, 0.01); // Higher epsilon for approximations
}

#[test]
fn test_softmax_correctness() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = activations::softmax(&x);

    // Verify sum equals 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < EPSILON);

    // Verify monotonicity (larger inputs = larger outputs)
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
    assert!(result[2] < result[3]);
}

#[test]
fn test_mse_correctness() {
    let pred = vec![1.0, 2.0, 3.0, 4.0];
    let target = vec![1.5, 2.5, 2.5, 4.5];

    let result = losses::mse(&pred, &target);

    // Manual calculation: ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 0.25
    let expected = 0.25;

    assert!((result - expected).abs() < EPSILON);
}

#[test]
fn test_mae_correctness() {
    let pred = vec![1.0, 2.0, 3.0, 4.0];
    let target = vec![1.5, 2.5, 2.5, 4.5];

    let result = losses::mae(&pred, &target);

    // Manual calculation: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
    let expected = 0.5;

    assert!((result - expected).abs() < EPSILON);
}

#[test]
fn test_mse_gradient_correctness() {
    let pred = vec![1.0, 2.0, 3.0, 4.0];
    let target = vec![1.5, 2.5, 2.5, 4.5];

    let gradient = losses::mse_gradient(&pred, &target);

    // Expected: 2 * (pred - target) / n
    let expected: Vec<f32> = pred.iter()
        .zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t) / 4.0)
        .collect();

    assert_vec_approx_eq(&gradient, &expected, EPSILON);
}

#[test]
fn test_reduce_sum_correctness() {
    let x: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let result = utils::reduce_sum(&x);

    // Sum of 1 to 100 = 100 * 101 / 2 = 5050
    let expected = 5050.0;

    assert!((result - expected).abs() < EPSILON);
}

#[test]
fn test_reduce_max_correctness() {
    let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let result = utils::reduce_max(&x);

    assert_eq!(result, 9.0);
}

#[test]
fn test_norm_l2_correctness() {
    let x = vec![3.0, 4.0]; // 3-4-5 triangle
    let result = utils::norm_l2(&x);

    assert!((result - 5.0).abs() < EPSILON);
}

#[test]
fn test_large_vectors() {
    // Test with larger vectors to ensure SIMD loops work correctly
    let size = 10000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    let dot = matmul::dot_product(&a, &b);
    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!((dot - expected).abs() / expected < 0.0001); // Relative error

    let sum = utils::reduce_sum(&a);
    let expected_sum: f32 = a.iter().sum();

    assert!((sum - expected_sum).abs() < EPSILON);
}

#[test]
fn test_edge_cases() {
    // Empty vectors
    let empty: Vec<f32> = vec![];
    assert_eq!(utils::reduce_sum(&empty), 0.0);

    // Single element
    let single = vec![42.0];
    assert_eq!(utils::reduce_sum(&single), 42.0);
    assert_eq!(utils::reduce_max(&single), 42.0);

    // All zeros
    let zeros = vec![0.0; 100];
    assert_eq!(losses::mse(&zeros, &zeros), 0.0);
    assert_eq!(losses::mae(&zeros, &zeros), 0.0);
}

#[test]
fn test_activation_ranges() {
    // Test activations produce expected ranges
    let x: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();

    let sigmoid = activations::sigmoid(&x);
    for &v in &sigmoid {
        assert!(v >= 0.0 && v <= 1.0, "Sigmoid out of range: {}", v);
    }

    let softmax = activations::softmax(&x);
    let sum: f32 = softmax.iter().sum();
    assert!((sum - 1.0).abs() < 0.001, "Softmax doesn't sum to 1: {}", sum);
}
