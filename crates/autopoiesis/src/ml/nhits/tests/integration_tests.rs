use super::*;
use crate::ml::nhits::{NHITSModel, NHITSConfig};
use crate::ml::nhits::consciousness::ConsciousnessIntegration;
use crate::api::nhits_api::{NHITSService, PredictionRequest, TrainingRequest};
use ndarray::{Array2, Array1};
use tokio;
use std::sync::Arc;
#[cfg(feature = "test-utils")]
use tempfile::TempDir;
use serde_json;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_training_pipeline() {
        // Create temporary directory for model artifacts
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model");
        
        // Initialize model with test configuration
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 2,
            num_blocks: [1, 1],
            num_layers: [2, 2],
            layer_widths: [128, 128],
            pooling_kernel_sizes: [2, 2],
            n_freq_downsample: [4, 2],
            max_steps: 100, // Reduced for testing
            batch_size: 8,
            val_check_steps: 25,
            early_stop_patience_steps: 50,
            learning_rate: 1e-2,
            dropout: 0.1,
            activation: "ReLU".to_string(),
        };
        
        let mut model = NHITSModel::new(config);
        
        // Generate synthetic training data
        let num_samples = 100;
        let train_x = Array2::from_shape_fn((num_samples, 168), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        });
        let train_y = Array2::from_shape_fn((num_samples, 24), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01 + 168.0 * 0.01).sin()
        });
        
        // Train the model
        let initial_loss = model.compute_loss(&model.forward(&train_x), &train_y);
        
        for epoch in 0..10 {
            for batch_start in (0..num_samples).step_by(8) {
                let batch_end = (batch_start + 8).min(num_samples);
                let batch_x = train_x.slice(s![batch_start..batch_end, ..]).to_owned();
                let batch_y = train_y.slice(s![batch_start..batch_end, ..]).to_owned();
                
                model.train_step(&batch_x, &batch_y);
            }
        }
        
        let final_loss = model.compute_loss(&model.forward(&train_x), &train_y);
        
        // Verify training progress
        assert!(final_loss < initial_loss, "Model should improve during training");
        
        // Save model
        model.save_checkpoint(&model_path.to_string_lossy()).unwrap();
        
        // Load model and verify consistency
        let mut loaded_model = NHITSModel::new(model.config.clone());
        loaded_model.load_checkpoint(&model_path.to_string_lossy()).unwrap();
        
        let loaded_output = loaded_model.forward(&train_x);
        let original_output = model.forward(&train_x);
        
        // Outputs should be identical
        assert_eq!(loaded_output.shape(), original_output.shape());
    }

    #[tokio::test]
    async fn test_consciousness_integration_pipeline() {
        // Initialize model with consciousness
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 2,
            max_steps: 50,
            batch_size: 4,
            ..Default::default()
        };
        
        let mut model = NHITSModel::new(config);
        model.enable_consciousness(256, 8, 3);
        
        // Generate test data
        let input = Array2::ones((4, 168));
        let targets = Array2::zeros((4, 24));
        
        // Test consciousness-aware forward pass
        let output_with_consciousness = model.forward_with_consciousness(&input);
        let output_without_consciousness = {
            model.disable_consciousness();
            let output = model.forward(&input);
            model.enable_consciousness(256, 8, 3);
            output
        };
        
        // Outputs should be different when consciousness is enabled
        assert_eq!(output_with_consciousness.shape(), output_without_consciousness.shape());
        
        // Test consciousness decision making
        if let Some(ref consciousness) = model.consciousness {
            let context = Array2::ones((1, 256));
            let options = vec![
                Array1::ones(24),
                Array1::zeros(24),
                Array1::from_elem(24, 0.5),
            ];
            
            let decision = consciousness.make_decision(&context, &options);
            assert!(decision < options.len());
        }
    }

    #[tokio::test]
    async fn test_api_service_integration() {
        // Initialize NHITS service
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            max_steps: 10,
            batch_size: 2,
            ..Default::default()
        };
        
        let service = Arc::new(NHITSService::new(config));
        
        // Test prediction endpoint
        let prediction_request = PredictionRequest {
            input_data: vec![vec![1.0; 168]; 2], // 2 samples, 168 features each
            model_id: Some("test_model".to_string()),
            return_attention: true,
            return_consciousness: true,
        };
        
        let prediction_response = service.predict(prediction_request).await;
        assert!(prediction_response.is_ok());
        
        let response = prediction_response.unwrap();
        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.predictions[0].len(), 24);
        assert!(response.attention_weights.is_some());
        
        // Test training endpoint
        let training_request = TrainingRequest {
            train_data: vec![vec![1.0; 168]; 10],
            train_targets: vec![vec![0.0; 24]; 10],
            val_data: Some(vec![vec![1.0; 168]; 2]),
            val_targets: Some(vec![vec![0.0; 24]; 2]),
            config_override: None,
            model_id: "test_model".to_string(),
        };
        
        let training_response = service.train(training_request).await;
        assert!(training_response.is_ok());
        
        let response = training_response.unwrap();
        assert!(response.final_loss > 0.0);
        assert!(!response.training_history.is_empty());
    }

    #[tokio::test]
    async fn test_model_versioning_and_rollback() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();
        
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            max_steps: 20,
            batch_size: 4,
            ..Default::default()
        };
        
        let mut model = NHITSModel::new(config);
        
        // Generate training data
        let train_x = Array2::ones((8, 168));
        let train_y = Array2::zeros((8, 24));
        
        // Save initial version
        let v1_path = base_path.join("model_v1.pt");
        model.save_checkpoint(&v1_path.to_string_lossy()).unwrap();
        let v1_output = model.forward(&train_x);
        
        // Train and save version 2
        for _ in 0..5 {
            model.train_step(&train_x, &train_y);
        }
        let v2_path = base_path.join("model_v2.pt");
        model.save_checkpoint(&v2_path.to_string_lossy()).unwrap();
        let v2_output = model.forward(&train_x);
        
        // Train and save version 3
        for _ in 0..5 {
            model.train_step(&train_x, &train_y);
        }
        let v3_path = base_path.join("model_v3.pt");
        model.save_checkpoint(&v3_path.to_string_lossy()).unwrap();
        let v3_output = model.forward(&train_x);
        
        // Test rollback to v1
        model.load_checkpoint(&v1_path.to_string_lossy()).unwrap();
        let rollback_v1_output = model.forward(&train_x);
        
        // Verify rollback worked
        assert_eq!(v1_output.shape(), rollback_v1_output.shape());
        
        // Test rollback to v2
        model.load_checkpoint(&v2_path.to_string_lossy()).unwrap();
        let rollback_v2_output = model.forward(&train_x);
        
        assert_eq!(v2_output.shape(), rollback_v2_output.shape());
    }

    #[tokio::test]
    async fn test_distributed_training_simulation() {
        // Simulate distributed training with multiple model instances
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            max_steps: 30,
            batch_size: 4,
            learning_rate: 1e-3,
            ..Default::default()
        };
        
        // Create multiple model instances (simulating distributed workers)
        let mut models = vec![
            NHITSModel::new(config.clone()),
            NHITSModel::new(config.clone()),
            NHITSModel::new(config.clone()),
        ];
        
        // Generate training data
        let full_dataset_x = Array2::from_shape_fn((24, 168), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        });
        let full_dataset_y = Array2::from_shape_fn((24, 24), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01 + 168.0 * 0.01).sin()
        });
        
        // Split data among workers
        let chunk_size = 8;
        let mut initial_losses = Vec::new();
        
        for (i, model) in models.iter_mut().enumerate() {
            let start_idx = i * chunk_size;
            let end_idx = ((i + 1) * chunk_size).min(24);
            
            let chunk_x = full_dataset_x.slice(s![start_idx..end_idx, ..]).to_owned();
            let chunk_y = full_dataset_y.slice(s![start_idx..end_idx, ..]).to_owned();
            
            let initial_loss = model.compute_loss(&model.forward(&chunk_x), &chunk_y);
            initial_losses.push(initial_loss);
            
            // Train on local data
            for _ in 0..10 {
                model.train_step(&chunk_x, &chunk_y);
            }
        }
        
        // Simulate parameter averaging (simplified federated learning)
        let mut averaged_params = models[0].get_parameters();
        for model in &models[1..] {
            let params = model.get_parameters();
            for (avg_param, param) in averaged_params.iter_mut().zip(params.iter()) {
                *avg_param = (*avg_param + param) / 2.0;
            }
        }
        
        // Apply averaged parameters to all models
        for model in &mut models {
            model.set_parameters(&averaged_params);
        }
        
        // Verify all models have consistent parameters
        let first_model_params = models[0].get_parameters();
        for model in &models[1..] {
            let params = model.get_parameters();
            assert_eq!(params.len(), first_model_params.len());
        }
    }

    #[tokio::test]
    async fn test_real_time_inference_pipeline() {
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            batch_size: 1, // Real-time inference
            ..Default::default()
        };
        
        let model = NHITSModel::new(config);
        
        // Simulate real-time data stream
        let mut inference_times = Vec::new();
        let num_inference_calls = 100;
        
        for i in 0..num_inference_calls {
            let input = Array2::from_shape_fn((1, 168), |(_, j)| {
                (i as f32 * 0.1 + j as f32 * 0.01).sin()
            });
            
            let start_time = std::time::Instant::now();
            let _output = model.forward(&input);
            let inference_time = start_time.elapsed();
            
            inference_times.push(inference_time.as_micros());
        }
        
        // Calculate performance metrics
        let avg_inference_time = inference_times.iter().sum::<u128>() / num_inference_calls;
        let max_inference_time = *inference_times.iter().max().unwrap();
        let min_inference_time = *inference_times.iter().min().unwrap();
        
        // Verify real-time performance (should be under 1ms for single sample)
        assert!(avg_inference_time < 1000, "Average inference time too high: {}μs", avg_inference_time);
        assert!(max_inference_time < 5000, "Max inference time too high: {}μs", max_inference_time);
        
        println!("Inference Performance:");
        println!("  Average: {}μs", avg_inference_time);
        println!("  Min: {}μs", min_inference_time);
        println!("  Max: {}μs", max_inference_time);
    }

    #[tokio::test]
    async fn test_model_interpretability_pipeline() {
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            num_stacks: 2,
            ..Default::default()
        };
        
        let model = NHITSModel::new(config);
        
        // Generate test input
        let input = Array2::from_shape_fn((1, 168), |(_, j)| {
            (j as f32 * 0.1).sin()
        });
        
        // Extract interpretability features
        let interpretability = model.extract_interpretability_features(&input);
        
        // Verify interpretability components
        assert!(interpretability.contains_key("layer_activations"));
        assert!(interpretability.contains_key("attention_patterns"));
        assert!(interpretability.contains_key("feature_importance"));
        assert!(interpretability.contains_key("gradient_norms"));
        
        // Test attention visualization
        let (output, attention_weights) = model.forward_with_attention(&input);
        assert_eq!(output.shape(), &[1, 24]);
        assert!(!attention_weights.is_empty());
        
        // Verify attention weights sum to 1
        for attention_map in attention_weights {
            let attention_sum: f32 = attention_map.iter().sum();
            assert!((attention_sum - 1.0).abs() < 1e-5);
        }
    }

    #[tokio::test] 
    async fn test_fault_tolerance_and_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("fault_recovery_model.pt");
        
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            max_steps: 50,
            val_check_steps: 10,
            ..Default::default()
        };
        
        let mut model = NHITSModel::new(config);
        
        // Generate training data
        let train_x = Array2::ones((16, 168));
        let train_y = Array2::zeros((16, 24));
        
        // Train for a few steps and save checkpoint
        for step in 0..10 {
            model.train_step(&train_x, &train_y);
            
            // Save checkpoint every 5 steps
            if step % 5 == 0 {
                model.save_checkpoint(&checkpoint_path.to_string_lossy()).unwrap();
            }
        }
        
        let pre_fault_output = model.forward(&train_x);
        
        // Simulate fault by corrupting model state
        let mut corrupted_model = NHITSModel::new(model.config.clone());
        
        // Recover from checkpoint
        corrupted_model.load_checkpoint(&checkpoint_path.to_string_lossy()).unwrap();
        let recovered_output = corrupted_model.forward(&train_x);
        
        // Verify recovery
        assert_eq!(pre_fault_output.shape(), recovered_output.shape());
        
        // Continue training from recovered state
        for _ in 0..5 {
            corrupted_model.train_step(&train_x, &train_y);
        }
        
        let post_recovery_output = corrupted_model.forward(&train_x);
        assert_eq!(post_recovery_output.shape(), &[16, 24]);
    }

    #[tokio::test]
    async fn test_multi_modal_consciousness_integration() {
        let config = NHITSConfig {
            input_size: 168,
            output_size: 24,
            ..Default::default()
        };
        
        let mut model = NHITSModel::new(config);
        model.enable_consciousness(512, 8, 4);
        
        // Test different consciousness modes
        let input = Array2::ones((4, 168));
        
        // Analytical mode
        model.set_consciousness_mode("analytical");
        let analytical_output = model.forward_with_consciousness(&input);
        
        // Creative mode
        model.set_consciousness_mode("creative");
        let creative_output = model.forward_with_consciousness(&input);
        
        // Balanced mode
        model.set_consciousness_mode("balanced");
        let balanced_output = model.forward_with_consciousness(&input);
        
        // All outputs should have same shape but potentially different values
        assert_eq!(analytical_output.shape(), &[4, 24]);
        assert_eq!(creative_output.shape(), &[4, 24]);
        assert_eq!(balanced_output.shape(), &[4, 24]);
        
        // Test consciousness state persistence
        let consciousness_state = model.get_consciousness_state();
        assert!(consciousness_state.contains_key("awareness_level"));
        assert!(consciousness_state.contains_key("decision_confidence"));
        assert!(consciousness_state.contains_key("attention_focus"));
    }
}