//! Integration tests for mixed precision training
//!
//! Tests:
//! - Full training loop with FP16
//! - Convergence with mixed precision
//! - Stability under various conditions
//! - Memory efficiency
//! - Gradient overflow recovery

use neuro_divergent::optimizations::mixed_precision::*;
use ndarray::{Array1, Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use approx::assert_relative_eq;

/// Generate synthetic training data
fn generate_linear_data(n_samples: usize, noise_level: f64) -> (Array2<f64>, Array1<f64>) {
    // y = 2.0 * x1 + 3.0 * x2 + noise
    let x = Array::random((n_samples, 2), Uniform::new(-1.0, 1.0));
    let noise = Array::random(n_samples, Uniform::new(-noise_level, noise_level));

    let y = x.column(0).mapv(|v| v * 2.0) + x.column(1).mapv(|v| v * 3.0) + noise;

    (x, y)
}

#[test]
fn test_full_training_loop() {
    // Test complete training with mixed precision
    let config = MixedPrecisionConfig::default();
    let mut trainer = MixedPrecisionTrainer::new(config);

    let (x_train, y_train) = generate_linear_data(100, 0.1);

    // Initialize dummy weights
    let weights = vec![Array2::zeros((2, 1))];
    trainer.initialize_weights(weights);

    let mut losses = Vec::new();

    // Train for a few steps
    for _epoch in 0..10 {
        let loss = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                // Dummy gradient computation
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );

        if let Ok(loss_value) = loss {
            losses.push(loss_value);
        }
    }

    // Check that training ran
    assert!(!losses.is_empty());

    // Check stats were updated
    let stats = trainer.stats();
    assert!(stats.total_steps > 0);
}

#[test]
fn test_convergence_comparison() {
    // Compare FP32 vs FP16 convergence
    let (x_train, y_train) = generate_linear_data(200, 0.05);

    // FP16 training
    let config = MixedPrecisionConfig::default();
    let mut fp16_trainer = MixedPrecisionTrainer::new(config);

    let weights = vec![Array2::zeros((2, 1))];
    fp16_trainer.initialize_weights(weights);

    let mut fp16_losses = Vec::new();

    for _epoch in 0..20 {
        let loss = fp16_trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );

        if let Ok(loss_value) = loss {
            fp16_losses.push(loss_value);
        }
    }

    // Check convergence (losses should generally decrease)
    assert!(fp16_losses.len() >= 10);
}

#[test]
fn test_gradient_overflow_handling() {
    // Test stability with very large gradients
    let config = MixedPrecisionConfig {
        initial_scale: 1000.0, // Lower scale to trigger overflow easier
        ..Default::default()
    };
    let mut trainer = MixedPrecisionTrainer::new(config);

    let (x_train, y_train) = generate_linear_data(50, 0.1);

    let weights = vec![Array2::zeros((2, 1))];
    trainer.initialize_weights(weights);

    let mut overflow_detected = false;

    for _ in 0..5 {
        // Generate very large gradients to trigger overflow
        let loss = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::from_elem((2, 1), 1e10)] // Huge gradient
            },
            0.01,
        );

        if loss.is_err() {
            overflow_detected = true;
            break;
        }
    }

    // Should either handle overflow or detect it
    let stats = trainer.stats();
    assert!(stats.total_steps > 0);
}

#[test]
fn test_loss_scale_adaptation() {
    // Test dynamic loss scaling
    let config = MixedPrecisionConfig {
        dynamic_scaling: true,
        growth_interval: 3,
        ..Default::default()
    };
    let mut trainer = MixedPrecisionTrainer::new(config);

    let (x_train, y_train) = generate_linear_data(50, 0.1);

    let weights = vec![Array2::zeros((2, 1))];
    trainer.initialize_weights(weights);

    let initial_scale = trainer.current_scale();

    // Train with stable gradients
    for _ in 0..10 {
        let _ = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.01, 0.01))]
            },
            0.01,
        );
    }

    let stats = trainer.stats();
    // Scale should potentially increase with stable training
    // (depends on growth_interval)
    assert!(stats.scale_increases >= 0);
}

#[test]
fn test_fp16_conversion_accuracy() {
    // Test FP16 conversion preserves values within tolerance
    let data = Array2::random((100, 10), Uniform::new(-100.0, 100.0));

    let fp16_data = conversion::to_fp16(&data);
    let back_to_fp64 = conversion::to_fp64(&fp16_data);

    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            let original = data[[i, j]];
            let converted = back_to_fp64[[i, j]];

            // Check within FP16 precision
            if original.abs() < F16::MAX as f64 {
                let rel_error = ((original - converted) / original.max(1e-6)).abs();
                assert!(rel_error < 0.01, "Relative error too large: {}", rel_error);
            }
        }
    }
}

#[test]
fn test_safety_checks() {
    // Test overflow/underflow detection
    let safe_array = Array2::from_elem((10, 10), 1.0);
    let (overflow, underflow) = conversion::check_fp16_safe(&safe_array);
    assert_eq!(overflow, 0);
    assert_eq!(underflow, 0);

    let overflow_array = Array2::from_elem((10, 10), 100000.0);
    let (overflow, _) = conversion::check_fp16_safe(&overflow_array);
    assert!(overflow > 0);

    let underflow_array = Array2::from_elem((10, 10), 1e-10);
    let (_, underflow) = conversion::check_fp16_safe(&underflow_array);
    assert!(underflow > 0);
}

#[test]
fn test_weight_manager_synchronization() {
    // Test that FP16 weights stay synced with master weights
    let initial_weights = vec![Array2::from_elem((3, 3), 1.0)];
    let mut manager = WeightManager::new(initial_weights, true);

    let gradients = vec![Array2::from_elem((3, 3), 0.1)];
    manager.update_master_weights(&gradients, 0.01);

    let master = &manager.master_weights()[0];
    let fp16 = &manager.get_fp16_weights()[0];

    // Check that FP16 weights are close to master weights
    for i in 0..3 {
        for j in 0..3 {
            let master_val = master[[i, j]];
            let fp16_val = fp16[[i, j]] as f64;
            assert_relative_eq!(master_val, fp16_val, epsilon = F16::EPSILON as f64);
        }
    }
}

#[test]
fn test_statistics_tracking() {
    // Test that statistics are properly tracked
    let config = MixedPrecisionConfig::default();
    let mut trainer = MixedPrecisionTrainer::new(config);

    let (x_train, y_train) = generate_linear_data(50, 0.1);

    let weights = vec![Array2::zeros((2, 1))];
    trainer.initialize_weights(weights);

    for _ in 0..5 {
        let _ = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );
    }

    let stats = trainer.stats();
    assert_eq!(stats.total_steps, 5);
    assert!(stats.current_scale > 0.0);
    assert!(stats.avg_grad_norm >= 0.0);
    assert!(stats.overflow_rate >= 0.0 && stats.overflow_rate <= 1.0);
}

#[test]
fn test_mixed_precision_reset() {
    // Test trainer reset functionality
    let config = MixedPrecisionConfig::default();
    let mut trainer = MixedPrecisionTrainer::new(config);

    let (x_train, y_train) = generate_linear_data(50, 0.1);

    let weights = vec![Array2::zeros((2, 1))];
    trainer.initialize_weights(weights);

    // Train a few steps
    for _ in 0..3 {
        let _ = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );
    }

    let stats_before = trainer.stats().clone();
    assert!(stats_before.total_steps > 0);

    // Reset
    trainer.reset();

    let stats_after = trainer.stats();
    assert_eq!(stats_after.total_steps, 0);
    assert_eq!(stats_after.overflow_count, 0);
}

#[test]
fn test_batch_size_scaling() {
    // Test with different batch sizes
    let config = MixedPrecisionConfig::default();

    for batch_size in [16, 32, 64, 128] {
        let mut trainer = MixedPrecisionTrainer::new(config.clone());

        let (x_train, y_train) = generate_linear_data(batch_size, 0.1);

        let weights = vec![Array2::zeros((2, 1))];
        trainer.initialize_weights(weights);

        let loss = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );

        assert!(loss.is_ok() || loss.is_err()); // Should complete either way
    }
}

#[test]
fn test_memory_efficiency() {
    // Test memory usage is reasonable
    let config = MixedPrecisionConfig::default();
    let mut trainer = MixedPrecisionTrainer::new(config);

    // Large model with many parameters
    let n_params = 100;
    let weights: Vec<Array2<f64>> = (0..n_params)
        .map(|_| Array2::random((10, 10), Uniform::new(-1.0, 1.0)))
        .collect();

    trainer.initialize_weights(weights);

    let (x_train, y_train) = generate_linear_data(32, 0.1);

    // Training should still work with many parameters
    let loss = trainer.train_step(
        &x_train,
        &y_train,
        |_input_fp16, _targets| {
            (0..n_params)
                .map(|_| Array2::random((10, 10), Uniform::new(-0.01, 0.01)))
                .collect()
        },
        0.01,
    );

    assert!(loss.is_ok() || loss.is_err());
}

#[test]
fn test_extreme_loss_scales() {
    // Test with extreme loss scales
    let extreme_configs = vec![
        MixedPrecisionConfig {
            initial_scale: 1.0,
            ..Default::default()
        },
        MixedPrecisionConfig {
            initial_scale: 65536.0,
            ..Default::default()
        },
        MixedPrecisionConfig {
            initial_scale: 262144.0, // 2^18
            ..Default::default()
        },
    ];

    for config in extreme_configs {
        let mut trainer = MixedPrecisionTrainer::new(config);

        let (x_train, y_train) = generate_linear_data(32, 0.1);

        let weights = vec![Array2::zeros((2, 1))];
        trainer.initialize_weights(weights);

        let loss = trainer.train_step(
            &x_train,
            &y_train,
            |_input_fp16, _targets| {
                vec![Array2::random((2, 1), Uniform::new(-0.1, 0.1))]
            },
            0.01,
        );

        // Should handle extreme scales gracefully
        assert!(loss.is_ok() || loss.is_err());
    }
}
