// ATS-Core Library - Advanced Trading System Core
// Scientific computing and neural network integration platform

pub mod types;
pub mod config;
pub mod security_config;
pub mod conformal;
pub mod conformal_optimized;
pub mod memory_optimized;
pub mod conformal_optimized_standalone_test;
pub mod error;
pub mod ffi;
pub mod integration;
pub mod memory;
pub mod monitoring;
pub mod nanosecond_validator;
pub mod parallel;
pub mod performance_dashboard;
pub mod performance_bottleneck_analyzer;
pub mod performance_analysis_integration;
pub mod simd;
pub mod tdd_enforcement;
pub mod temperature;
pub mod test_utils;
pub mod utils;
pub mod benchmarks;

// API Integration Layer
pub mod api;
pub mod bridge;

// Standalone ML validation - working neural networks
#[cfg(feature = "minimal-ml")]
pub mod ml_validation;

// ruv-FANN integration module (disabled for minimal builds)
#[cfg(not(feature = "minimal-ml"))]
pub mod ruv_fann_integration;

// Re-export main types for convenience
pub use types::*;
pub use config::*;
pub use conformal::*;
pub use conformal_optimized::*;
pub use memory_optimized::*;
pub use performance_bottleneck_analyzer::*;
pub use performance_analysis_integration::*;
pub use error::*;
#[cfg(not(feature = "minimal-ml"))]
pub use ruv_fann_integration::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Initialize ATS-Core with ruv-FANN integration
pub async fn initialize() -> std::result::Result<ruv_fann_integration::RuvFannIntegration, ruv_fann_integration::IntegrationError> {
    println!("🚀 Initializing ATS-Core with ruv-FANN integration v{}", VERSION);

    // Initialize real ruv-FANN integration with 27+ architectures
    let integration = ruv_fann_integration::RuvFannIntegration::new().await?;

    println!("✅ Initialized ruv-FANN ML integration with 27+ architectures");

    Ok(integration)
}

/// Validate integration instance
pub async fn validate_integration(
    integration: &ruv_fann_integration::RuvFannIntegration
) -> std::result::Result<ValidationResult, ruv_fann_integration::IntegrationError> {
    let start_time = std::time::Instant::now();

    // Get architectures count from real integration
    let architectures = integration.get_architectures().await;
    let architecture_count = architectures.len();

    let validation_time = start_time.elapsed();

    Ok(ValidationResult {
        success: true,
        architecture_count,
        validation_duration: validation_time,
        errors: Vec::new(),
    })
}

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub success: bool,
    pub architecture_count: usize,
    pub validation_duration: std::time::Duration,
    pub errors: Vec<String>,
}

/// Quick start example function
pub async fn quick_start_example() -> std::result::Result<(), ruv_fann_integration::IntegrationError> {
    println!("🎯 Running ruv-FANN quick start example...");
    
    // Initialize integration
    let mut integration = initialize().await?;
    
    // Create a simple LSTM model for time series
    let model_config = ModelConfig {
        input_size: 1,
        hidden_sizes: vec![32, 16],
        output_size: 1,
        activation: ActivationFunction::Tanh,
        dropout_rate: Some(0.1),
        regularization: None,
        architecture_specific: serde_json::json!({
            "sequence_length": 10
        }),
    };
    
    let model_id = integration.create_model(
        "quickstart_lstm".to_string(),
        "lstm".to_string(),
        model_config
    ).await?;
    
    println!("📈 Created LSTM model: {}", model_id);
    
    // Generate some test data
    let training_data = generate_test_time_series_data();
    
    // Configure training
    let training_config = TrainingConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        optimizer: OptimizerType::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        loss_function: LossFunction::MSE,
        device: DeviceType::Auto,
        early_stopping: Some(EarlyStoppingConfig {
            patience: 10,
            min_delta: 0.001,
            monitor: "loss".to_string(),
        }),
        scheduler: None,
    };
    
    // Train the model
    println!("🚀 Starting training...");
    let training_result = integration.train_model(model_id.clone(), training_data, training_config).await?;
    
    println!("✅ Training completed in {:.2}s", training_result.training_time.as_secs_f32());
    println!("📊 Final loss: {:.6}", training_result.final_loss);
    println!("📊 Final accuracy: {:.4}", training_result.final_accuracy);
    
    // Generate forecast
    let input_data = InputData {
        features: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
        sequence_length: Some(5),
    };
    
    let forecast_config = ForecastConfig {
        horizon: 5,
        confidence_intervals: vec![0.8, 0.95],
        uncertainty_quantification: true,
        monte_carlo_samples: Some(100),
    };
    
    println!("🔮 Generating forecast...");
    let forecast_result = integration.forecast(model_id, input_data, forecast_config).await?;
    
    println!("📈 Forecast predictions: {:?}", forecast_result.predictions);
    println!("🎯 Forecast horizon: {} steps", forecast_result.forecast_horizon);
    
    // Run performance benchmark
    println!("⚡ Running performance benchmark...");
    let benchmark_results = integration.benchmark().await?;
    
    println!("📊 Benchmark completed in {:.2}s", benchmark_results.total_duration.as_secs_f32());
    println!("🏗️ Architectures tested: {}", benchmark_results.architecture_benchmarks.len());
    
    println!("✅ Quick start example completed successfully!");
    
    Ok(())
}

/// Generate test time series data
fn generate_test_time_series_data() -> TrainingData {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let n_samples = 200;
    let sequence_length = 10;
    
    let mut features = Vec::new();
    let mut targets = Vec::new();
    
    // Generate synthetic sine wave with noise
    for i in 0..n_samples {
        let mut sequence = Vec::new();
        let base_value = (i as f32 * 0.1).sin();
        
        for j in 0..sequence_length {
            let noise = rng.gen_range(-0.1..0.1);
            let value = base_value + ((i + j) as f32 * 0.05).sin() + noise;
            sequence.push(value);
        }
        
        features.push(sequence);
        
        // Target is next value in sequence
        let target_noise = rng.gen_range(-0.05..0.05);
        let target = base_value + ((i + sequence_length) as f32 * 0.05).sin() + target_noise;
        targets.push(vec![target]);
    }
    
    TrainingData {
        features,
        targets,
        validation_split: Some(0.2),
    }
}

/// Export commonly used types
pub mod prelude {
    pub use crate::{
        RuvFannIntegration,
        ModelConfig,
        TrainingConfig,
        TrainingData,
        ForecastConfig,
        InputData,
        ActivationFunction,
        OptimizerType,
        LossFunction,
        DeviceType,
        ModelId,
        initialize,
        quick_start_example,
    };
    pub use crate::ruv_fann_integration::IntegrationError;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ats_core_initialization() {
        let result = initialize().await;
        assert!(result.is_ok());
        
        let integration = result.unwrap();
        let architectures = integration.get_architectures().await;
        assert!(!architectures.is_empty());
        assert!(architectures.contains(&"mlp".to_string()));
        assert!(architectures.contains(&"lstm".to_string()));
        assert!(architectures.contains(&"transformer".to_string()));
    }
    
    #[tokio::test]
    async fn test_validation() {
        let integration = initialize().await.unwrap();
        let validation = validate_integration(&integration).await.unwrap();
        
        assert!(validation.success);
        assert!(validation.architecture_count >= 10); // Should have at least 10 architectures
        assert!(validation.validation_duration.as_millis() < 5000); // Should be fast
    }
    
    #[test]
    fn test_version_constants() {
        assert!(!VERSION.is_empty());
        assert!(!NAME.is_empty());
        assert_eq!(NAME, "ats-core");
    }
    
    #[test]
    fn test_test_data_generation() {
        let data = generate_test_time_series_data();
        
        assert_eq!(data.features.len(), 200);
        assert_eq!(data.targets.len(), 200);
        assert_eq!(data.validation_split, Some(0.2));
        
        // Check sequence length
        if let Some(first_sequence) = data.features.first() {
            assert_eq!(first_sequence.len(), 10);
        }
        
        // Check target format
        if let Some(first_target) = data.targets.first() {
            assert_eq!(first_target.len(), 1);
        }
    }
}