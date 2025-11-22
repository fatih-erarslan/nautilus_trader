use anyhow::Result;
use tokio;
use tracing::{info, error};
use tracing_subscriber::fmt;
use uuid::Uuid;

use crate::ml::nhits::api::{
    client::*,
    models::*,
};

/// Basic NHITS API client example
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    fmt::init();

    // Create client with default configuration
    let client = NHITSClient::with_base_url("http://localhost:8080/api/v1")?;

    // Check API health
    info!("Checking API health...");
    match client.health_check().await {
        Ok(health) => info!("API is healthy: {:?}", health),
        Err(e) => {
            error!("Health check failed: {}", e);
            return Err(e);
        }
    }

    // Create a new model
    info!("Creating a new model...");
    let create_request = CreateModelRequest {
        name: "Example NHITS Model".to_string(),
        description: Some("An example NHITS model for demonstration".to_string()),
        config: ModelConfig {
            input_size: 24,
            output_size: 6,
            n_stacks: 3,
            n_blocks: vec![1, 1, 1],
            n_layers: vec![2, 2, 2],
            layer_widths: vec![512, 512, 512],
            pooling_sizes: vec![2, 2, 1],
            interpolation_mode: "linear".to_string(),
            dropout: 0.1,
            activation: "relu".to_string(),
            max_epochs: 50,
            learning_rate: 0.001,
            batch_size: 32,
            patience: 10,
        },
        tags: Some(vec!["example".to_string(), "demo".to_string()]),
    };

    let model = client.create_model(create_request).await?;
    info!("Created model: {} (ID: {})", model.name, model.id);

    // Train the model with sample data
    info!("Training the model...");
    let training_data = generate_sample_training_data();
    let train_request = TrainModelRequest {
        data: training_data,
        validation_split: Some(0.2),
        save_checkpoints: Some(true),
        callback_url: None,
    };

    match client.train_model(&model.id, train_request).await {
        Ok(result) => info!("Training started: {:?}", result),
        Err(e) => error!("Training failed: {}", e),
    }

    // Wait for training to complete in real scenarios
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    // Create a forecast
    info!("Creating a forecast...");
    let forecast_data = generate_sample_forecast_data();
    let forecast_request = ForecastRequest {
        data: forecast_data,
        forecast_steps: Some(12),
        feature_names: Some(vec!["temperature".to_string(), "humidity".to_string()]),
        timestamps: None,
        confidence_intervals: Some(vec![0.95, 0.99]),
        return_intervals: Some(true),
        callback_url: None,
        priority: Some(ForecastPriority::High),
    };

    let forecast_job = client.create_forecast(&model.id, forecast_request).await?;
    info!("Created forecast job: {}", forecast_job.id);

    // Wait for forecast completion
    info!("Waiting for forecast completion...");
    match client.wait_for_forecast(forecast_job.id, tokio::time::Duration::from_secs(1)).await {
        Ok(completed_job) => {
            info!("Forecast completed successfully!");
            if let Some(result) = completed_job.result {
                info!("Predictions: {:?}", result.predictions);
                info!("Computation time: {:.2}s", result.computation_time);
                if let Some(metrics) = result.metrics {
                    info!("Forecast metrics: {:?}", metrics);
                }
            }
        }
        Err(e) => error!("Forecast failed: {}", e),
    }

    // List all models
    info!("Listing all models...");
    let query = ListModelsQuery {
        page: Some(1),
        page_size: Some(10),
        status: Some("trained".to_string()),
        ..Default::default()
    };

    match client.list_models(Some(query)).await {
        Ok(models_response) => {
            info!("Found {} models", models_response.total);
            for model in models_response.items {
                info!("  - {} ({}): {}", model.name, model.id, model.status);
            }
        }
        Err(e) => error!("Failed to list models: {}", e),
    }

    // List forecast jobs
    info!("Listing forecast jobs...");
    let jobs_query = ListJobsQuery {
        page: Some(1),
        page_size: Some(10),
        model_id: Some(model.id.clone()),
        ..Default::default()
    };

    match client.list_forecasts(Some(jobs_query)).await {
        Ok(jobs_response) => {
            info!("Found {} forecast jobs", jobs_response.total);
            for job in jobs_response.items {
                info!("  - {} ({}): {}", job.id, job.model_id, job.status);
            }
        }
        Err(e) => error!("Failed to list forecast jobs: {}", e),
    }

    // Clean up - delete the model
    info!("Cleaning up - deleting model...");
    match client.delete_model(&model.id).await {
        Ok(_) => info!("Model deleted successfully"),
        Err(e) => error!("Failed to delete model: {}", e),
    }

    info!("Example completed successfully!");
    Ok(())
}

/// Generate sample training data
fn generate_sample_training_data() -> TrainingData {
    let mut data = Vec::new();
    let mut targets = Vec::new();

    // Generate 100 time series samples
    for _ in 0..100 {
        let mut series = Vec::new();
        let mut target_series = Vec::new();

        // Generate 24 time steps of input data (temperature, humidity)
        for t in 0..24 {
            let temperature = 20.0 + 10.0 * (t as f64 * 0.1).sin() + rand::random::<f64>() * 2.0;
            let humidity = 60.0 + 20.0 * (t as f64 * 0.05).cos() + rand::random::<f64>() * 5.0;
            series.push(vec![temperature, humidity]);
        }

        // Generate 6 time steps of target data
        for t in 24..30 {
            let temperature = 20.0 + 10.0 * (t as f64 * 0.1).sin() + rand::random::<f64>() * 2.0;
            let humidity = 60.0 + 20.0 * (t as f64 * 0.05).cos() + rand::random::<f64>() * 5.0;
            target_series.push(vec![temperature, humidity]);
        }

        data.push(series);
        targets.push(target_series);
    }

    TrainingData {
        data,
        targets: Some(targets),
        feature_names: vec!["temperature".to_string(), "humidity".to_string()],
        timestamps: None,
        preprocessing: Some(PreprocessingConfig {
            normalize: true,
            standardize: true,
            remove_trend: false,
            seasonal_adjustment: false,
            fill_missing: "mean".to_string(),
        }),
    }
}

/// Generate sample forecast data
fn generate_sample_forecast_data() -> Vec<Vec<f64>> {
    let mut data = Vec::new();

    // Generate 24 time steps of recent data
    for t in 0..24 {
        let temperature = 22.0 + 8.0 * (t as f64 * 0.1).sin() + rand::random::<f64>() * 1.5;
        let humidity = 65.0 + 15.0 * (t as f64 * 0.05).cos() + rand::random::<f64>() * 3.0;
        data.push(vec![temperature, humidity]);
    }

    data
}

impl Default for ListModelsQuery {
    fn default() -> Self {
        Self {
            page: None,
            page_size: None,
            status: None,
            tags: None,
            search: None,
            sort_by: None,
            sort_order: None,
        }
    }
}

impl Default for ListJobsQuery {
    fn default() -> Self {
        Self {
            page: None,
            page_size: None,
            status: None,
            model_id: None,
            priority: None,
            created_after: None,
            created_before: None,
        }
    }
}