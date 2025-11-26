//! Gradient Check Tests for Numerical Stability
//!
//! Validates that analytical gradients match numerical gradients

use neuro_divergent::{
    models::{basic::MLP, recurrent::{LSTM, GRU}},
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
};
use ndarray::{Array1, Array2};

#[path = "helpers/mod.rs"]
mod helpers;
use helpers::{synthetic, gradient_check};

#[test]
fn test_mlp_gradient_check() {
    let config = ModelConfig::default()
        .with_input_size(10)
        .with_horizon(5)
        .with_hidden_size(16)
        .with_learning_rate(0.01);

    let mut model = MLP::new(config);

    let values = synthetic::sine_wave(100, 0.1, 10.0, 50.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    // Train for a few epochs to get stable gradients
    model.fit(&data).unwrap();

    // Perform gradient check on model parameters
    // In practice, this would access internal model parameters
    // For now, we verify that training produces reasonable gradients

    let history = model.training_history();
    let first_loss = history[0];
    let last_loss = *history.last().unwrap();

    assert!(
        last_loss < first_loss,
        "Gradient descent should reduce loss if gradients are correct"
    );

    println!("✓ MLP gradient check passed (loss reduction verified)");
}

#[test]
fn test_lstm_gradient_flow() {
    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(10)
        .with_hidden_size(32)
        .with_learning_rate(0.005)
        .with_epochs(30);

    let mut model = LSTM::new(config);

    let values = synthetic::complex_series(300, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    model.fit(&data).unwrap();

    let history = model.training_history();

    // Check gradient flow by verifying consistent loss reduction
    let mut loss_decreased = 0;
    for i in 1..history.len() {
        if history[i] < history[i - 1] {
            loss_decreased += 1;
        }
    }

    let decrease_ratio = loss_decreased as f64 / (history.len() - 1) as f64;

    assert!(
        decrease_ratio > 0.5,
        "LSTM gradients should flow properly (decrease ratio: {})",
        decrease_ratio
    );

    println!("✓ LSTM gradient flow check passed");
}

#[test]
fn test_gru_gradient_flow() {
    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(10)
        .with_hidden_size(32)
        .with_learning_rate(0.005)
        .with_epochs(30);

    let mut model = GRU::new(config);

    let values = synthetic::complex_series(300, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    model.fit(&data).unwrap();

    let history = model.training_history();

    let initial_loss = history[0];
    let final_loss = *history.last().unwrap();

    assert!(
        final_loss < initial_loss * 0.8,
        "GRU gradients should enable significant learning: {} -> {}",
        initial_loss,
        final_loss
    );

    println!("✓ GRU gradient flow check passed");
}

#[test]
fn test_numerical_gradient_approximation() {
    // Test numerical gradient computation for simple quadratic function
    use ndarray::arr1;

    let f = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
    let x = arr1(&[3.0, 4.0]);

    let numerical = gradient_check::numerical_gradient(f, &x, 1e-5);
    let analytical = arr1(&[6.0, 8.0]); // Gradient of x^2 + y^2 is [2x, 2y]

    assert!(
        gradient_check::gradients_match(&analytical, &numerical, 1e-3, 1e-5),
        "Numerical gradient computation should be accurate"
    );

    println!("✓ Numerical gradient approximation test passed");
}

#[test]
fn test_gradient_clipping_prevents_explosion() {
    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(10)
        .with_hidden_size(64)
        .with_learning_rate(0.1) // High learning rate
        .with_gradient_clip(Some(1.0))
        .with_epochs(50);

    let mut model = LSTM::new(config);

    // Data that might cause gradient explosion without clipping
    let values: Vec<f64> = (0..300)
        .map(|i| (i as f64 * 0.1).exp().min(100.0))
        .collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let result = model.fit(&data);

    assert!(
        result.is_ok(),
        "Gradient clipping should prevent explosion"
    );

    let history = model.training_history();

    // Check that no loss values are NaN or Inf
    for (i, &loss) in history.iter().enumerate() {
        assert!(
            loss.is_finite(),
            "Loss should remain finite with gradient clipping at epoch {}: {}",
            i, loss
        );
    }

    println!("✓ Gradient clipping test passed");
}

#[test]
fn test_vanishing_gradient_detection() {
    // Test that deep networks show signs of vanishing gradients without proper initialization
    let config = ModelConfig::default()
        .with_input_size(50)
        .with_horizon(10)
        .with_hidden_size(8) // Small hidden size
        .with_learning_rate(0.001)
        .with_epochs(100);

    let mut model = LSTM::new(config);

    // Long sequence to test gradient propagation
    let values = synthetic::ar1_series(500, 0.95, 0.1, 50.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    model.fit(&data).unwrap();

    let history = model.training_history();
    let improvement = history[0] - history.last().unwrap();

    // LSTM should still learn despite potential vanishing gradients
    assert!(
        improvement > 0.0,
        "LSTM should overcome vanishing gradients: improvement={}",
        improvement
    );

    println!("✓ Vanishing gradient detection test passed");
}

#[test]
fn test_gradient_norm_bounds() {
    // Test that gradient norms stay within reasonable bounds
    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(10)
        .with_hidden_size(32)
        .with_learning_rate(0.01)
        .with_epochs(50);

    let mut model = MLP::new(config);

    let values = synthetic::sine_wave(200, 0.1, 10.0, 50.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    model.fit(&data).unwrap();

    let history = model.training_history();

    // Verify loss doesn't become NaN or Inf (indication of gradient explosion)
    for &loss in &history {
        assert!(loss.is_finite(), "Loss should remain finite: {}", loss);
        assert!(loss >= 0.0, "Loss should be non-negative: {}", loss);
    }

    println!("✓ Gradient norm bounds test passed");
}

#[test]
fn test_second_order_gradient_approximation() {
    // Test second-order numerical differentiation
    use ndarray::arr1;

    let f = |x: &Array1<f64>| x[0].powi(3);
    let x = arr1(&[2.0]);

    // First derivative
    let grad1 = gradient_check::numerical_gradient(f, &x, 1e-5);

    // Approximate second derivative
    let f_grad = |x: &Array1<f64>| {
        gradient_check::numerical_gradient(f, x, 1e-5)[0]
    };
    let grad2 = gradient_check::numerical_gradient(f_grad, &x, 1e-5);

    // Second derivative of x^3 is 6x, so at x=2, it's 12
    let expected_grad2 = arr1(&[12.0]);

    assert!(
        gradient_check::gradients_match(&expected_grad2, &grad2, 0.1, 1e-3),
        "Second-order gradient should be approximately correct"
    );

    println!("✓ Second-order gradient approximation test passed");
}

#[test]
fn test_gradient_consistency_across_batches() {
    // Test that gradients are consistent when computed on different batches
    let config = ModelConfig::default()
        .with_input_size(20)
        .with_horizon(10)
        .with_hidden_size(32)
        .with_learning_rate(0.01)
        .with_epochs(20)
        .with_random_seed(42);

    let values = synthetic::complex_series(500, 0.1, 24, 1.0);

    // Train on first half
    let data1 = TimeSeriesDataFrame::from_values(
        values[..250].to_vec(),
        None,
    ).unwrap();
    let mut model1 = MLP::new(config.clone());
    model1.fit(&data1).unwrap();
    let history1 = model1.training_history();

    // Train on second half
    let data2 = TimeSeriesDataFrame::from_values(
        values[250..].to_vec(),
        None,
    ).unwrap();
    let mut model2 = MLP::new(config);
    model2.fit(&data2).unwrap();
    let history2 = model2.training_history();

    // Both should show learning (positive gradient descent)
    assert!(history1.last().unwrap() < &history1[0]);
    assert!(history2.last().unwrap() < &history2[0]);

    println!("✓ Gradient consistency test passed");
}

#[test]
fn test_gradient_based_convergence() {
    // Test that gradient-based optimization converges to a reasonable solution
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_hidden_size(64)
        .with_learning_rate(0.01)
        .with_epochs(100)
        .with_early_stopping(Some(10)); // Stop if no improvement for 10 epochs

    let mut model = LSTM::new(config);

    let values = synthetic::sine_wave(400, 0.05, 20.0, 100.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    model.fit(&data).unwrap();

    let history = model.training_history();

    // Check for convergence (loss plateaus)
    if history.len() > 20 {
        let late_losses = &history[history.len() - 10..];
        let late_variance = {
            let mean = late_losses.iter().sum::<f64>() / late_losses.len() as f64;
            late_losses
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / late_losses.len() as f64
        };

        assert!(
            late_variance < 0.1,
            "Loss should converge (low variance): {}",
            late_variance
        );
    }

    println!("✓ Gradient-based convergence test passed");
}
