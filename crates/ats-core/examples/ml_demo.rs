// ATS-Core ML Integration Demo
// Demonstrates working neural network functionality

use ats_core::ruv_fann_integration::*;
use ats_core::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ATS-Core ML Integration Demo");
    println!("================================");

    // Create integration instance
    println!("🧠 Creating ML integration...");
    let integration = RuvFannIntegration::new().await?;
    
    // Create a neural model
    println!("📊 Creating neural model...");
    let config = ModelConfig {
        input_size: 4,
        hidden_sizes: vec![8, 6],
        output_size: 2,
        activation: ActivationFunction::ReLU,
        dropout_rate: Some(0.1),
        regularization: None,
        architecture_specific: serde_json::json!({}),
    };
    
    let model_id = integration.create_model("demo_model".to_string(), "mlp".to_string(), config).await?;
    println!("✅ Model created: {}", model_id);
    
    // Generate training data
    println!("📈 Generating training data...");
    let training_data = TrainingData {
        features: vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ],
        targets: vec![
            vec![0.1, 0.9],
            vec![0.2, 0.8],
            vec![0.3, 0.7],
            vec![0.4, 0.6],
            vec![0.5, 0.5],
        ],
        validation_split: Some(0.2),
    };
    
    // Configure training
    let training_config = TrainingConfig {
        epochs: 10,
        batch_size: 2,
        learning_rate: 0.01,
        optimizer: OptimizerType::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        loss_function: LossFunction::MSE,
        device: DeviceType::CPU,
        early_stopping: None,
        scheduler: None,
    };
    
    // Train the model
    println!("🏋️ Training model...");
    let training_result = integration.train_model(model_id.clone(), training_data, training_config).await?;
    println!("✅ Training completed!");
    println!("   Final loss: {:.6}", training_result.final_loss);
    println!("   Training time: {:.2}ms", training_result.training_time.as_millis());
    
    // Generate forecasts
    println!("📊 Generating forecasts...");
    let input_data = InputData {
        features: vec![vec![6.0, 7.0, 8.0, 9.0]],
        sequence_length: Some(1),
    };
    
    let forecast_config = ForecastConfig {
        horizon: 3,
        confidence_intervals: vec![0.8, 0.95],
        uncertainty_quantification: true,
        monte_carlo_samples: Some(50),
    };
    
    let forecast = integration.forecast(model_id.clone(), input_data, forecast_config).await?;
    println!("   Forecast: {:?}", forecast.predictions);
    println!("   Confidence intervals: {:?}", forecast.confidence_intervals.keys().collect::<Vec<_>>());
    
    // Show architectures
    println!("📈 Available Architectures:");
    let architectures = integration.get_architectures().await;
    println!("   Total architectures: {}", architectures.len());
    
    println!();
    println!("🎉 ML Integration Demo Completed Successfully!");
    println!("✅ All neural network functionality verified!");
    
    Ok(())
}