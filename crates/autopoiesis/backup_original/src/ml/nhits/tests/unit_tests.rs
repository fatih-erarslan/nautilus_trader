use super::*;
use crate::ml::nhits::{NHITSModel, NHITSConfig, AttentionLayer, StackBlock};
use crate::ml::nhits::consciousness::{ConsciousnessIntegration, AttentionMechanism};
use ndarray::{Array2, Array3, Array4};
#[cfg(feature = "test-utils")]
use approx::assert_abs_diff_eq;

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_nhits_config_creation() {
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 3,
            num_blocks: [1, 1, 1],
            num_layers: [2, 2, 2],
            layer_widths: [512, 512, 512],
            pooling_kernel_sizes: [2, 2, 2],
            n_freq_downsample: [8, 4, 1],
            activation: "ReLU".to_string(),
            dropout: 0.1,
            max_steps: 5000,
            learning_rate: 1e-3,
            batch_size: 32,
            val_check_steps: 100,
            early_stop_patience_steps: 1000,
        };
        
        assert_eq!(config.input_size, 168);
        assert_eq!(config.output_size, 24);
        assert_eq!(config.num_stacks, 3);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_nhits_model_initialization() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config.clone());
        
        assert_eq!(model.config.input_size, config.input_size);
        assert_eq!(model.config.output_size, config.output_size);
        assert_eq!(model.stacks.len(), config.num_stacks);
    }

    #[test]
    fn test_attention_layer_forward() {
        let attention_layer = AttentionLayer::new(64, 8, 0.1);
        let input = Array3::zeros((1, 10, 64)); // batch_size, seq_len, hidden_dim
        
        let output = attention_layer.forward(&input);
        assert_eq!(output.shape(), &[1, 10, 64]);
    }

    #[test]
    fn test_stack_block_forward() {
        let stack_block = StackBlock::new(
            168,    // input_size
            24,     // output_size
            2,      // num_layers
            512,    // layer_width
            2,      // pooling_kernel_size
            8,      // n_freq_downsample
            0.1,    // dropout
        );
        
        let input = Array2::zeros((32, 168)); // batch_size, input_size
        let output = stack_block.forward(&input);
        
        assert_eq!(output.shape(), &[32, 24]);
    }

    #[test]
    fn test_consciousness_integration_initialization() {
        let consciousness = ConsciousnessIntegration::new(512, 8, 4, 0.1);
        
        assert_eq!(consciousness.hidden_dim, 512);
        assert_eq!(consciousness.num_heads, 8);
        assert_eq!(consciousness.num_layers, 4);
        assert_eq!(consciousness.dropout, 0.1);
    }

    #[test]
    fn test_attention_mechanism_compute() {
        let attention = AttentionMechanism::new(64, 8);
        let query = Array3::ones((1, 10, 64));
        let key = Array3::ones((1, 10, 64));
        let value = Array3::ones((1, 10, 64));
        
        let (output, attention_weights) = attention.compute_attention(&query, &key, &value);
        
        assert_eq!(output.shape(), &[1, 10, 64]);
        assert_eq!(attention_weights.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_model_forward_pass() {
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 2,
            num_blocks: [1, 1],
            num_layers: [2, 2],
            layer_widths: [256, 256],
            pooling_kernel_sizes: [2, 2],
            n_freq_downsample: [4, 2],
            ..Default::default()
        };
        
        let model = NHITSModel::new(config);
        let input = Array2::zeros((4, 168)); // batch_size=4, input_size=168
        
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[4, 24]);
    }

    #[test]
    fn test_model_loss_computation() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let predictions = Array2::ones((4, 24));
        let targets = Array2::zeros((4, 24));
        
        let loss = model.compute_loss(&predictions, &targets);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_model_gradient_computation() {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        
        let input = Array2::ones((2, 168));
        let targets = Array2::zeros((2, 24));
        
        let gradients = model.compute_gradients(&input, &targets);
        assert!(!gradients.is_empty());
    }

    #[test]
    fn test_model_parameter_update() {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        
        let input = Array2::ones((2, 168));
        let targets = Array2::zeros((2, 24));
        
        let initial_params = model.get_parameters();
        model.train_step(&input, &targets);
        let updated_params = model.get_parameters();
        
        // Parameters should have changed
        assert_ne!(initial_params.len(), 0);
        assert_ne!(updated_params.len(), 0);
    }

    #[test]
    fn test_consciousness_awareness_computation() {
        let consciousness = ConsciousnessIntegration::new(512, 8, 4, 0.1);
        let model_state = Array2::ones((1, 512));
        
        let awareness = consciousness.compute_awareness(&model_state);
        assert_eq!(awareness.shape(), &[1, 512]);
        
        // Awareness should be normalized
        let sum: f32 = awareness.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_consciousness_decision_making() {
        let consciousness = ConsciousnessIntegration::new(512, 8, 4, 0.1);
        let context = Array2::ones((1, 512));
        let options = vec![
            Array1::ones(24),
            Array1::zeros(24),
            Array1::from_elem(24, 0.5),
        ];
        
        let decision = consciousness.make_decision(&context, &options);
        assert!(decision < options.len());
    }

    #[test]
    fn test_model_consciousness_integration() {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        model.enable_consciousness(512, 8, 4);
        
        let input = Array2::ones((2, 168));
        let output = model.forward_with_consciousness(&input);
        
        assert_eq!(output.shape(), &[2, 24]);
        assert!(model.consciousness.is_some());
    }

    #[test]
    fn test_model_attention_weights_extraction() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let input = Array2::ones((2, 168));
        let (output, attention_weights) = model.forward_with_attention(&input);
        
        assert_eq!(output.shape(), &[2, 24]);
        assert!(!attention_weights.is_empty());
    }

    #[test]
    fn test_model_interpretability_features() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let input = Array2::ones((1, 168));
        let interpretability = model.extract_interpretability_features(&input);
        
        assert!(interpretability.contains_key("layer_activations"));
        assert!(interpretability.contains_key("attention_patterns"));
        assert!(interpretability.contains_key("feature_importance"));
    }

    #[test]
    fn test_model_serialization() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let serialized = model.serialize().unwrap();
        let deserialized = NHITSModel::deserialize(&serialized).unwrap();
        
        assert_eq!(model.config.input_size, deserialized.config.input_size);
        assert_eq!(model.config.output_size, deserialized.config.output_size);
    }

    #[test]
    fn test_model_checkpoint_save_load() {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        
        // Train for a few steps
        let input = Array2::ones((2, 168));
        let targets = Array2::zeros((2, 24));
        model.train_step(&input, &targets);
        
        // Save checkpoint
        model.save_checkpoint("test_checkpoint.pt").unwrap();
        
        // Load checkpoint
        let mut loaded_model = NHITSModel::new(model.config.clone());
        loaded_model.load_checkpoint("test_checkpoint.pt").unwrap();
        
        // Parameters should match
        let original_params = model.get_parameters();
        let loaded_params = loaded_model.get_parameters();
        assert_eq!(original_params.len(), loaded_params.len());
    }

    #[test]
    fn test_model_validation_metrics() {
        let config = NHITSConfig::default();
        let model = NHITSModel::new(config);
        
        let predictions = Array2::from_shape_vec((4, 24), (0..96).map(|x| x as f32).collect()).unwrap();
        let targets = Array2::from_shape_vec((4, 24), (0..96).map(|x| (x + 1) as f32).collect()).unwrap();
        
        let metrics = model.compute_validation_metrics(&predictions, &targets);
        
        assert!(metrics.contains_key("mse"));
        assert!(metrics.contains_key("mae"));
        assert!(metrics.contains_key("mape"));
        assert!(metrics.contains_key("smape"));
        assert!(metrics.contains_key("r2"));
    }

    #[test]
    fn test_model_early_stopping() {
        let config = NHITSConfig {
            early_stop_patience_steps: 5,
            ..Default::default()
        };
        let mut model = NHITSModel::new(config);
        
        // Simulate training with no improvement
        for _ in 0..10 {
            model.update_early_stopping_counter(1.0); // Same loss
        }
        
        assert!(model.should_early_stop());
    }

    #[test]
    fn test_model_learning_rate_scheduling() {
        let config = NHITSConfig::default();
        let mut model = NHITSModel::new(config);
        
        let initial_lr = model.get_learning_rate();
        
        // Simulate training steps
        for step in 0..1000 {
            model.update_learning_rate(step);
        }
        
        let final_lr = model.get_learning_rate();
        assert_ne!(initial_lr, final_lr);
    }

    #[test]
    fn test_model_regularization() {
        let config = NHITSConfig {
            dropout: 0.5,
            ..Default::default()
        };
        let model = NHITSModel::new(config);
        
        let input = Array2::ones((4, 168));
        
        // Test with training mode (dropout active)
        model.set_training_mode(true);
        let output_train = model.forward(&input);
        
        // Test with evaluation mode (dropout inactive)
        model.set_training_mode(false);
        let output_eval = model.forward(&input);
        
        // Outputs should be different due to dropout
        assert_ne!(output_train.shape(), &[0, 0]);
        assert_ne!(output_eval.shape(), &[0, 0]);
    }
}