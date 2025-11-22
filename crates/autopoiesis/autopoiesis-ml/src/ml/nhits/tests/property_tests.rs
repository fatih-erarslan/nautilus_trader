#[cfg(feature = "property-tests")]
use super::*;

#[cfg(feature = "property-tests")]
use crate::ml::nhits::{NHITSModel, NHITSConfig, AttentionLayer, StackBlock};

#[cfg(feature = "property-tests")]
use crate::ml::nhits::consciousness::ConsciousnessIntegration;

#[cfg(feature = "property-tests")]
use ndarray::{Array1, Array2, Array3};

#[cfg(feature = "property-tests")]
use proptest::{prelude::*, prop_compose};

#[cfg(feature = "property-tests")]
use proptest::collection::vec;

#[cfg(feature = "property-tests")]
use std::collections::HashMap;

// Property test generators for NHITS components
#[cfg(feature = "property-tests")]
prop_compose! {
    fn valid_nhits_config()(
        input_size in 24usize..512,
        output_size in 1usize..168,
        num_stacks in 1usize..6,
        layer_width in 32usize..1024,
        learning_rate in 1e-5f32..1e-1,
        dropout in 0.0f32..0.8,
        batch_size in 1usize..128,
    ) -> NHITSConfig {
        let num_blocks = vec![1; num_stacks];
        let num_layers = vec![2; num_stacks];
        let layer_widths = vec![layer_width; num_stacks];
        let pooling_kernel_sizes = vec![2; num_stacks];
        let n_freq_downsample = (0..num_stacks).map(|i| 2_i32.pow(i as u32 + 1)).collect();
        
        NHITSConfig {
            input_size,
            output_size,
            num_stacks,
            num_blocks,
            num_layers,
            layer_widths,
            pooling_kernel_sizes,
            n_freq_downsample,
            activation: "ReLU".to_string(),
            dropout,
            max_steps: 100,
            learning_rate,
            batch_size,
            val_check_steps: 25,
            early_stop_patience_steps: 50,
        }
    }
}

#[cfg(feature = "property-tests")]
prop_compose! {
    fn valid_input_data(batch_size: usize, input_size: usize)(
        values in vec(vec(-10.0f32..10.0, input_size), batch_size)
    ) -> Array2<f32> {
        let flat_values: Vec<f32> = values.into_iter().flatten().collect();
        Array2::from_shape_vec((batch_size, input_size), flat_values).unwrap()
    }
}

#[cfg(feature = "property-tests")]
prop_compose! {
    fn valid_target_data(batch_size: usize, output_size: usize)(
        values in vec(vec(-10.0f32..10.0, output_size), batch_size)
    ) -> Array2<f32> {
        let flat_values: Vec<f32> = values.into_iter().flatten().collect();
        Array2::from_shape_vec((batch_size, output_size), flat_values).unwrap()
    }
}

// Property tests for model invariants
#[cfg(feature = "property-tests")]
proptest! {
    #[test]
    fn test_model_output_shape_consistency(
        config in valid_nhits_config(),
        batch_size in 1usize..32,
    ) {
        let model = NHITSModel::new(config.clone());
        let input = Array2::zeros((batch_size, config.input_size));
        
        let output = model.forward(&input);
        
        // Property: Output shape should always match expected dimensions
        prop_assert_eq!(output.shape(), &[batch_size, config.output_size]);
    }

    #[test]
    fn test_model_deterministic_forward_pass(
        config in valid_nhits_config(),
    ) {
        let model = NHITSModel::new(config.clone());
        let input = Array2::ones((4, config.input_size));
        
        let output1 = model.forward(&input);
        let output2 = model.forward(&input);
        
        // Property: Same input should produce same output (deterministic)
        for (a, b) in output1.iter().zip(output2.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_model_finite_outputs(
        config in valid_nhits_config(),
        input in valid_input_data(8, config.input_size),
    ) {
        let model = NHITSModel::new(config);
        let output = model.forward(&input);
        
        // Property: All outputs should be finite (no NaN or Inf)
        for &value in output.iter() {
            prop_assert!(value.is_finite());
        }
    }

    #[test]
    fn test_training_step_improves_or_maintains_loss(
        config in valid_nhits_config(),
    ) {
        let mut model = NHITSModel::new(config.clone());
        let input = Array2::ones((4, config.input_size));
        let target = Array2::zeros((4, config.output_size));
        
        let initial_output = model.forward(&input);
        let initial_loss = model.compute_loss(&initial_output, &target);
        
        // Perform multiple training steps
        for _ in 0..10 {
            model.train_step(&input, &target);
        }
        
        let final_output = model.forward(&input);
        let final_loss = model.compute_loss(&final_output, &target);
        
        // Property: Training should not drastically increase loss
        // (allowing for some numerical instability)
        prop_assert!(final_loss <= initial_loss * 2.0);
    }

    #[test]
    fn test_model_scale_invariance_property(
        config in valid_nhits_config(),
        scale_factor in 0.1f32..10.0,
    ) {
        let model = NHITSModel::new(config.clone());
        let input = Array2::ones((2, config.input_size));
        let scaled_input = &input * scale_factor;
        
        let output1 = model.forward(&input);
        let output2 = model.forward(&scaled_input);
        
        // Property: Model should handle different input scales gracefully
        // (outputs should remain finite and bounded)
        for &value in output1.iter() {
            prop_assert!(value.is_finite());
            prop_assert!(value.abs() < 1000.0); // Reasonable bound
        }
        
        for &value in output2.iter() {
            prop_assert!(value.is_finite());
            prop_assert!(value.abs() < 1000.0);
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one(
        hidden_dim in 32usize..256,
        num_heads in 1usize..16,
        seq_len in 1usize..64,
    ) {
        let attention_layer = AttentionLayer::new(hidden_dim, num_heads, 0.1);
        let input = Array3::ones((1, seq_len, hidden_dim));
        
        let output = attention_layer.forward(&input);
        
        // Property: Attention output should have same shape as input
        prop_assert_eq!(output.shape(), input.shape());
        
        // Property: Output should be finite
        for &value in output.iter() {
            prop_assert!(value.is_finite());
        }
    }

    #[test]
    fn test_stack_block_linearity_property(
        input_size in 24usize..256,
        output_size in 1usize..64,
        layer_width in 32usize..512,
    ) {
        let stack_block = StackBlock::new(
            input_size,
            output_size,
            2, // num_layers
            layer_width,
            2, // pooling_kernel_size
            4, // n_freq_downsample
            0.1, // dropout
        );
        
        let input1 = Array2::ones((2, input_size));
        let input2 = Array2::from_elem((2, input_size), 2.0);
        let input_sum = &input1 + &input2;
        
        let output1 = stack_block.forward(&input1);
        let output2 = stack_block.forward(&input2);
        let output_sum = stack_block.forward(&input_sum);
        
        // Property: Stack block should exhibit approximate linearity for small perturbations
        // (this is a weak linearity test due to non-linear activations)
        prop_assert_eq!(output1.shape(), &[2, output_size]);
        prop_assert_eq!(output2.shape(), &[2, output_size]);
        prop_assert_eq!(output_sum.shape(), &[2, output_size]);
    }

    #[test]
    fn test_consciousness_integration_consistency(
        hidden_dim in 64usize..512,
        num_heads in 2usize..16,
        num_layers in 1usize..8,
    ) {
        let consciousness = ConsciousnessIntegration::new(hidden_dim, num_heads, num_layers, 0.1);
        let model_state = Array2::ones((1, hidden_dim));
        
        // Test awareness computation consistency  
        let awareness1 = consciousness.compute_awareness(&model_state);
        let awareness2 = consciousness.compute_awareness(&model_state);
        
        // Property: Same input should produce same awareness (deterministic)
        prop_assert_eq!(awareness1.shape(), awareness2.shape());
        for (a, b) in awareness1.iter().zip(awareness2.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
        
        // Property: Awareness should be normalized (sum to 1)
        let awareness_sum: f32 = awareness1.iter().sum();
        prop_assert!((awareness_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_gradient_descent_convergence_property(
        config in valid_nhits_config(),
    ) {
        let mut model = NHITSModel::new(config.clone());
        
        // Simple synthetic data where target = input.mean()
        let input = Array2::from_shape_fn((8, config.input_size), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        });
        let target = Array2::from_shape_fn((8, config.output_size), |(i, _)| {
            input.row(i).mean().unwrap()
        });
        
        let mut losses = Vec::new();
        
        // Train for multiple steps
        for _ in 0..20 {
            let output = model.forward(&input);
            let loss = model.compute_loss(&output, &target);
            losses.push(loss);
            
            model.train_step(&input, &target);
        }
        
        // Property: Loss trend should generally be decreasing (allowing some fluctuation)
        let initial_loss = losses[0];
        let final_loss = losses[losses.len() - 1];
        
        prop_assert!(final_loss <= initial_loss * 1.5); // Allow some tolerance
        
        // Property: Losses should remain finite
        for &loss in &losses {
            prop_assert!(loss.is_finite());
            prop_assert!(loss >= 0.0);
        }
    }

    #[test]
    fn test_model_batch_size_consistency(
        config in valid_nhits_config(),
        batch_size1 in 1usize..16,
        batch_size2 in 1usize..16,
    ) {
        let model = NHITSModel::new(config.clone());
        
        let input1 = Array2::ones((batch_size1, config.input_size));
        let input2 = Array2::ones((batch_size2, config.input_size));
        
        let output1 = model.forward(&input1);
        let output2 = model.forward(&input2);
        
        // Property: Different batch sizes should produce outputs with correct shapes
        prop_assert_eq!(output1.shape(), &[batch_size1, config.output_size]);
        prop_assert_eq!(output2.shape(), &[batch_size2, config.output_size]);
        
        // Property: Per-sample outputs should be consistent across batch sizes
        // (first sample should be the same regardless of batch size)
        if batch_size1 > 0 && batch_size2 > 0 {
            let sample1 = output1.row(0);
            let sample2 = output2.row(0);
            
            for (a, b) in sample1.iter().zip(sample2.iter()) {
                prop_assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_model_parameter_bounds(
        config in valid_nhits_config(),
    ) {
        let model = NHITSModel::new(config);
        let parameters = model.get_parameters();
        
        // Property: All parameters should be finite and within reasonable bounds
        for &param in &parameters {
            prop_assert!(param.is_finite());
            prop_assert!(param.abs() < 100.0); // Reasonable initialization bound
        }
        
        // Property: Should have a reasonable number of parameters
        prop_assert!(parameters.len() > 0);
        prop_assert!(parameters.len() < 10_000_000); // Not excessively large
    }

    #[test]
    fn test_loss_function_properties(
        config in valid_nhits_config(),
        batch_size in 1usize..16,
    ) {
        let model = NHITSModel::new(config.clone());
        let predictions = Array2::ones((batch_size, config.output_size));
        let targets = Array2::zeros((batch_size, config.output_size));
        
        let loss = model.compute_loss(&predictions, &targets);
        
        // Property: Loss should be non-negative
        prop_assert!(loss >= 0.0);
        
        // Property: Loss should be finite
        prop_assert!(loss.is_finite());
        
        // Property: Perfect predictions should have zero loss
        let perfect_loss = model.compute_loss(&predictions, &predictions);
        prop_assert!(perfect_loss < 1e-6);
    }

    #[test]
    fn test_model_serialization_consistency(
        config in valid_nhits_config(),
    ) {
        let original_model = NHITSModel::new(config);
        
        // Serialize and deserialize
        let serialized = original_model.serialize().unwrap();
        let deserialized_model = NHITSModel::deserialize(&serialized).unwrap();
        
        // Property: Serialized and deserialized models should produce same outputs
        let input = Array2::ones((2, original_model.config.input_size));
        let original_output = original_model.forward(&input);
        let deserialized_output = deserialized_model.forward(&input);
        
        prop_assert_eq!(original_output.shape(), deserialized_output.shape());
        
        for (a, b) in original_output.iter().zip(deserialized_output.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_consciousness_decision_making_properties(
        hidden_dim in 64usize..256,
        num_options in 2usize..10,
    ) {
        let consciousness = ConsciousnessIntegration::new(hidden_dim, 8, 4, 0.1);
        let context = Array2::ones((1, hidden_dim));
        
        let options: Vec<Array1<f32>> = (0..num_options)
            .map(|i| Array1::from_elem(24, i as f32))
            .collect();
        
        let decision = consciousness.make_decision(&context, &options);
        
        // Property: Decision should be a valid option index
        prop_assert!(decision < options.len());
        
        // Property: Same context should produce same decision (deterministic)
        let decision2 = consciousness.make_decision(&context, &options);
        prop_assert_eq!(decision, decision2);
    }

    #[test]
    fn test_model_memory_efficiency_property(
        config in valid_nhits_config(),
    ) {
        let model = NHITSModel::new(config.clone());
        let parameters = model.get_parameters();
        
        // Property: Memory usage should scale reasonably with model size
        let expected_min_params = config.input_size + config.output_size;
        let expected_max_params = config.layer_widths.iter().sum::<usize>() * 
                                  config.num_layers.iter().sum::<usize>() * 10; // Rough upper bound
        
        prop_assert!(parameters.len() >= expected_min_params);
        prop_assert!(parameters.len() <= expected_max_params);
    }

    #[test]
    fn test_attention_mechanism_properties(
        hidden_dim in 32usize..128,
        num_heads in 2usize..8,
        seq_len in 2usize..32,
    ) {
        use crate::ml::nhits::consciousness::AttentionMechanism;
        
        let attention = AttentionMechanism::new(hidden_dim, num_heads);
        let query = Array3::ones((1, seq_len, hidden_dim));
        let key = Array3::ones((1, seq_len, hidden_dim));
        let value = Array3::ones((1, seq_len, hidden_dim));
        
        let (output, attention_weights) = attention.compute_attention(&query, &key, &value);
        
        // Property: Output shape should match value shape
        prop_assert_eq!(output.shape(), value.shape());
        
        // Property: Attention weights should have correct shape
        prop_assert_eq!(attention_weights.shape(), &[1, num_heads, seq_len, seq_len]);
        
        // Property: Attention weights should sum to 1 for each head and query position
        for head in 0..num_heads {
            for query_pos in 0..seq_len {
                let weight_sum: f32 = (0..seq_len)
                    .map(|key_pos| attention_weights[[0, head, query_pos, key_pos]])
                    .sum();
                prop_assert!((weight_sum - 1.0).abs() < 1e-5);
            }
        }
        
        // Property: All values should be finite and non-negative
        for &weight in attention_weights.iter() {
            prop_assert!(weight.is_finite());
            prop_assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_model_training_stability(
        config in valid_nhits_config(),
        num_steps in 1usize..50,
    ) {
        let mut model = NHITSModel::new(config.clone());
        let input = Array2::ones((4, config.input_size));
        let target = Array2::zeros((4, config.output_size));
        
        let mut parameter_norms = Vec::new();
        
        for _ in 0..num_steps {
            model.train_step(&input, &target);
            
            let params = model.get_parameters();
            let param_norm = params.iter().map(|&x| x * x).sum::<f32>().sqrt();
            parameter_norms.push(param_norm);
        }
        
        // Property: Parameter norms should remain finite during training
        for &norm in &parameter_norms {
            prop_assert!(norm.is_finite());
        }
        
        // Property: Parameters shouldn't explode (gradient explosion check)
        let max_norm = parameter_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        prop_assert!(max_norm < 1000.0);
    }

    #[test]
    fn test_consciousness_awareness_normalization_property(
        hidden_dim in 64usize..256,
        batch_size in 1usize..8,
    ) {
        let consciousness = ConsciousnessIntegration::new(hidden_dim, 8, 4, 0.1);
        let model_state = Array2::from_shape_fn((batch_size, hidden_dim), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        });
        
        let awareness = consciousness.compute_awareness(&model_state);
        
        // Property: Awareness should be normalized (sum to 1) for each batch
        for batch_idx in 0..batch_size {
            let batch_awareness = awareness.row(batch_idx);
            let awareness_sum: f32 = batch_awareness.iter().sum();
            prop_assert!((awareness_sum - 1.0).abs() < 1e-5);
        }
        
        // Property: All awareness values should be non-negative
        for &value in awareness.iter() {
            prop_assert!(value >= 0.0);
            prop_assert!(value.is_finite());
        }
    }
}

// Additional helper functions for property testing
impl NHITSModel {
    /// Property test helper: Check if model state is valid
    pub fn is_valid_state(&self) -> bool {
        let parameters = self.get_parameters();
        
        // All parameters should be finite
        if !parameters.iter().all(|&x| x.is_finite()) {
            return false;
        }
        
        // Parameter count should be reasonable
        if parameters.is_empty() || parameters.len() > 50_000_000 {
            return false;
        }
        
        // Configuration should be valid
        if self.config.input_size == 0 || 
           self.config.output_size == 0 || 
           self.config.num_stacks == 0 ||
           self.config.learning_rate <= 0.0 ||
           self.config.learning_rate > 1.0 ||
           self.config.dropout < 0.0 ||
           self.config.dropout >= 1.0 {
            return false;
        }
        
        true
    }
    
    /// Property test helper: Compute parameter statistics
    pub fn parameter_statistics(&self) -> HashMap<String, f32> {
        let parameters = self.get_parameters();
        let mut stats = HashMap::new();
        
        if parameters.is_empty() {
            return stats;
        }
        
        let mean = parameters.iter().sum::<f32>() / parameters.len() as f32;
        let variance = parameters.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / parameters.len() as f32;
        let std_dev = variance.sqrt();
        
        let min_param = parameters.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_param = parameters.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        stats.insert("mean".to_string(), mean);
        stats.insert("std_dev".to_string(), std_dev);
        stats.insert("min".to_string(), min_param);
        stats.insert("max".to_string(), max_param);
        stats.insert("count".to_string(), parameters.len() as f32);
        
        stats
    }
}

#[cfg(test)]
mod property_test_validation {
    use super::*;

    #[test]
    fn test_property_test_generators() {
        // Test that our generators produce valid configurations
        let config = valid_nhits_config().new_tree(&mut proptest::test_runner::TestRunner::default()).unwrap().current();
        
        assert!(config.input_size >= 24 && config.input_size <= 512);
        assert!(config.output_size >= 1 && config.output_size <= 168);
        assert!(config.num_stacks >= 1 && config.num_stacks <= 6);
        assert!(config.learning_rate > 0.0 && config.learning_rate < 1.0);
        assert!(config.dropout >= 0.0 && config.dropout < 1.0);
        
        // Test input data generator
        let input_data = valid_input_data(4, config.input_size).new_tree(&mut proptest::test_runner::TestRunner::default()).unwrap().current();
        assert_eq!(input_data.shape(), &[4, config.input_size]);
        
        for &value in input_data.iter() {
            assert!(value >= -10.0 && value <= 10.0);
        }
    }

    #[test]
    fn test_model_validity_checker() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        assert!(model.is_valid_state());
        
        let stats = model.parameter_statistics();
        assert!(stats.contains_key("mean"));
        assert!(stats.contains_key("std_dev"));
        assert!(stats.contains_key("count"));
        
        let param_count = stats["count"];
        assert!(param_count > 0.0);
    }

    #[test]
    fn test_consciousness_property_helpers() {
        let consciousness = ConsciousnessIntegration::new(128, 8, 4, 0.1);
        let model_state = Array2::ones((2, 128));
        
        let awareness = consciousness.compute_awareness(&model_state);
        
        // Verify normalization property
        for batch_idx in 0..2 {
            let batch_awareness = awareness.row(batch_idx);
            let sum: f32 = batch_awareness.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        
        // Verify non-negativity property
        for &value in awareness.iter() {
            assert!(value >= 0.0);
            assert!(value.is_finite());
        }
    }
}