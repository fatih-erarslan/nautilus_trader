use anyhow::Result;
use std::collections::HashMap;
use tokio;
use tracing::{info, error, warn};
use tracing_subscriber::fmt;

use crate::ml::nhits::api::{
    client::*,
    models::*,
    websocket::WsMessage,
};

/// WebSocket client example for real-time updates
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    fmt::init();

    // Create HTTP client
    let client = NHITSClient::with_base_url("http://localhost:8080/api/v1")?;

    // First, create a model and forecast job
    info!("Setting up model and forecast job...");
    let (model_id, job_id) = setup_forecast_job(&client).await?;

    // Connect to WebSocket with subscription parameters
    info!("Connecting to WebSocket...");
    let mut params = HashMap::new();
    params.insert("model_id".to_string(), model_id.clone());
    params.insert("job_id".to_string(), job_id.to_string());
    params.insert("client_id".to_string(), "example_client".to_string());

    let mut ws_client = client.connect_websocket(Some(params)).await?;
    info!("WebSocket connected successfully!");

    // Listen for messages with a custom handler
    let result = ws_client.listen(|msg| {
        Box::pin(async move {
            handle_websocket_message(msg).await;
        })
    }).await;

    match result {
        Ok(_) => info!("WebSocket connection closed normally"),
        Err(e) => error!("WebSocket error: {}", e),
    }

    Ok(())
}

/// Handle incoming WebSocket messages
async fn handle_websocket_message(message: WsMessage) {
    match message {
        WsMessage::Connected { connection_id, timestamp } => {
            info!("Connected to WebSocket: {} at {}", connection_id, timestamp);
        }
        WsMessage::Ping { timestamp } => {
            info!("Received ping at {}", timestamp);
            // Pong response is handled automatically by the client
        }
        WsMessage::Pong { timestamp } => {
            info!("Received pong at {}", timestamp);
        }
        WsMessage::ForecastUpdate { 
            job_id, 
            model_id, 
            progress, 
            partial_results, 
            timestamp 
        } => {
            info!("Forecast update - Job: {}, Model: {}, Progress: {:.1}% at {}", 
                  job_id, model_id, progress * 100.0, timestamp);
            
            if let Some(results) = partial_results {
                info!("Partial results: {:?}", results);
            }
            
            if progress >= 1.0 {
                info!("Forecast completed!");
            }
        }
        WsMessage::TrainingUpdate { 
            model_id, 
            epoch, 
            loss, 
            metrics, 
            timestamp 
        } => {
            info!("Training update - Model: {}, Epoch: {}, Loss: {:.4} at {}", 
                  model_id, epoch, loss, timestamp);
            
            for (metric_name, metric_value) in metrics {
                info!("  {}: {:.4}", metric_name, metric_value);
            }
        }
        WsMessage::SystemUpdate { 
            active_models, 
            active_jobs, 
            memory_usage, 
            cpu_usage, 
            timestamp 
        } => {
            info!("System update at {} - Models: {}, Jobs: {}, Memory: {:.1}MB, CPU: {:.1}%", 
                  timestamp, active_models, active_jobs, memory_usage / 1024.0 / 1024.0, cpu_usage);
        }
        WsMessage::Error { code, message, timestamp } => {
            error!("WebSocket error at {}: {} - {}", timestamp, code, message);
        }
        WsMessage::Subscribed { subscription, timestamp } => {
            info!("Subscribed to {} at {}", subscription, timestamp);
        }
        WsMessage::Unsubscribed { subscription, timestamp } => {
            info!("Unsubscribed from {} at {}", subscription, timestamp);
        }
    }
}

/// Setup a model and forecast job for demonstration
async fn setup_forecast_job(client: &NHITSClient) -> Result<(String, uuid::Uuid)> {
    // Create a model
    let create_request = CreateModelRequest {
        name: "WebSocket Demo Model".to_string(),
        description: Some("Model for WebSocket demonstration".to_string()),
        config: ModelConfig {
            input_size: 12,
            output_size: 6,
            ..Default::default()
        },
        tags: Some(vec!["websocket".to_string(), "demo".to_string()]),
    };

    let model = client.create_model(create_request).await?;
    info!("Created model: {}", model.id);

    // Start training (optional, to see training updates)
    let training_data = generate_sample_data();
    let train_request = TrainModelRequest {
        data: training_data,
        validation_split: Some(0.2),
        save_checkpoints: Some(false),
        callback_url: None,
    };

    client.train_model(&model.id, train_request).await?;
    info!("Started training for model: {}", model.id);

    // Create a forecast job
    let forecast_data = generate_forecast_data();
    let forecast_request = ForecastRequest {
        data: forecast_data,
        forecast_steps: Some(6),
        feature_names: Some(vec!["value".to_string()]),
        timestamps: None,
        confidence_intervals: None,
        return_intervals: None,
        callback_url: None,
        priority: Some(ForecastPriority::Normal),
    };

    let job = client.create_forecast(&model.id, forecast_request).await?;
    info!("Created forecast job: {}", job.id);

    Ok((model.id, job.id))
}

/// Generate sample training data
fn generate_sample_data() -> TrainingData {
    let mut data = Vec::new();
    let mut targets = Vec::new();

    // Generate 50 time series samples
    for i in 0..50 {
        let mut series = Vec::new();
        let mut target_series = Vec::new();

        let base_value = (i as f64) * 0.1;

        // Generate 12 time steps of input data
        for t in 0..12 {
            let value = base_value + (t as f64 * 0.2).sin() + rand::random::<f64>() * 0.1;
            series.push(vec![value]);
        }

        // Generate 6 time steps of target data
        for t in 12..18 {
            let value = base_value + (t as f64 * 0.2).sin() + rand::random::<f64>() * 0.1;
            target_series.push(vec![value]);
        }

        data.push(series);
        targets.push(target_series);
    }

    TrainingData {
        data,
        targets: Some(targets),
        feature_names: vec!["value".to_string()],
        timestamps: None,
        preprocessing: Some(PreprocessingConfig {
            normalize: true,
            standardize: false,
            remove_trend: false,
            seasonal_adjustment: false,
            fill_missing: "mean".to_string(),
        }),
    }
}

/// Generate sample forecast data
fn generate_forecast_data() -> Vec<Vec<f64>> {
    let mut data = Vec::new();

    // Generate 12 time steps of recent data
    for t in 0..12 {
        let value = (t as f64 * 0.2).sin() + rand::random::<f64>() * 0.1;
        data.push(vec![value]);
    }

    data
}

/// Alternative example: Multiple WebSocket connections
pub async fn multiple_connections_example() -> Result<()> {
    fmt::init();

    let client = NHITSClient::with_base_url("http://localhost:8080/api/v1")?;

    // Create multiple WebSocket connections for different purposes
    let connections = vec![
        ("system_monitor", HashMap::new()),
        ("model_updates", {
            let mut params = HashMap::new();
            params.insert("model_id".to_string(), "model_123".to_string());
            params
        }),
        ("job_tracker", {
            let mut params = HashMap::new();
            params.insert("job_id".to_string(), "job_456".to_string());
            params
        }),
    ];

    let mut handles = Vec::new();

    for (name, params) in connections {
        let client_clone = client.clone();
        let name = name.to_string();
        
        let handle = tokio::spawn(async move {
            match client_clone.connect_websocket(Some(params)).await {
                Ok(mut ws_client) => {
                    info!("Connected WebSocket: {}", name);
                    
                    let result = ws_client.listen(|msg| {
                        let conn_name = name.clone();
                        Box::pin(async move {
                            info!("[{}] Received message: {:?}", conn_name, msg);
                        })
                    }).await;
                    
                    match result {
                        Ok(_) => info!("WebSocket {} closed normally", name),
                        Err(e) => error!("WebSocket {} error: {}", name, e),
                    }
                }
                Err(e) => error!("Failed to connect WebSocket {}: {}", name, e),
            }
        });
        
        handles.push(handle);
    }

    // Wait for all connections
    for handle in handles {
        let _ = handle.await;
    }

    Ok(())
}

/// Example: Sending messages to WebSocket
pub async fn bidirectional_websocket_example() -> Result<()> {
    fmt::init();

    let client = NHITSClient::with_base_url("http://localhost:8080/api/v1")?;
    let mut ws_client = client.connect_websocket(None).await?;

    // Send a ping message
    let ping_msg = WsMessage::Ping {
        timestamp: chrono::Utc::now(),
    };
    
    ws_client.send(&ping_msg).await?;
    info!("Sent ping message");

    // Listen for responses
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        ws_client.listen(|msg| {
            Box::pin(async move {
                match msg {
                    WsMessage::Pong { timestamp } => {
                        info!("Received pong response at {}", timestamp);
                    }
                    _ => {
                        info!("Received other message: {:?}", msg);
                    }
                }
            })
        }).await
    }).await??;

    ws_client.close().await?;
    info!("WebSocket connection closed");

    Ok(())
}