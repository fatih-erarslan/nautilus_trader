use anyhow::Result;
use std::collections::HashMap;
use tokio;
use tracing::{info, error, warn};
use tracing_subscriber::fmt;
use uuid::Uuid;
use ndarray::{Array2, Array3};
use chrono::Utc;
use rand;

use crate::ml::nhits::{
    NHITSModelTrait, NHITSConfig, ModelState,
};
use crate::ml::nhits::model::{
    TrainingMetadata, DataSource, StackType, InterpolationMode, AttentionType,
};
use crate::ml::nhits::api::{
    models::*,
    streaming::*,
};

/// Real-time streaming forecast example
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    fmt::init();

    info!("Starting NHITS streaming example...");

    // Configure streaming
    let config = StreamingConfig {
        buffer_size: 1000,
        batch_size: 10,
        processing_interval_ms: 500,
        max_latency_ms: 2000,
        auto_scaling: true,
        max_concurrent_streams: 10,
    };

    // Create stream manager
    let stream_manager = StreamManager::new(config.clone());

    // Create a data stream
    info!("Creating data stream...");
    let stream_id = stream_manager.create_stream().await?;
    let stream = stream_manager.get_stream(stream_id).await
        .ok_or_else(|| anyhow::anyhow!("Failed to get created stream"))?;

    // Add NHITS processor
    info!("Setting up NHITS stream processor...");
    let model = create_mock_nhits_model();
    let processor = NHITSStreamProcessor::new(
        model,
        "streaming_model_v1".to_string(),
        ModelConfig {
            input_size: 10,
            output_size: 3,
            ..Default::default()
        },
    );

    // Note: We can't modify the stream after creation in this design
    // In a real implementation, you'd add processors during stream creation
    // For this example, we'll simulate the processing

    // Subscribe to stream results
    info!("Subscribing to stream results...");
    let (subscription_id, mut result_receiver) = stream.subscribe().await;

    // Start the stream processing
    info!("Starting stream processing...");
    stream.start().await?;

    // Spawn a task to handle results
    let result_handler = tokio::spawn(async move {
        let mut result_count = 0;
        
        while let Some(result) = result_receiver.recv().await {
            result_count += 1;
            info!("Received forecast result #{}: {:?}", result_count, result.predictions);
            info!("  Latency: {:.2}ms", result.latency_ms);
            info!("  Timestamp: {}", result.timestamp);
            
            // Stop after receiving 20 results
            if result_count >= 20 {
                info!("Received {} results, stopping...", result_count);
                break;
            }
        }
    });

    // Generate and push sample data
    info!("Generating sample time series data...");
    let data_generator = tokio::spawn(async move {
        for i in 0..100 {
            // Generate realistic time series data (temperature readings)
            let base_temp = 20.0;
            let seasonal = 5.0 * (i as f64 * 0.1).sin();
            let noise = rand::random::<f64>() * 2.0 - 1.0;
            let temperature = base_temp + seasonal + noise;
            
            // Create data point with multiple features
            let values = vec![
                temperature,
                60.0 + 10.0 * (i as f64 * 0.05).cos(), // humidity
                1013.0 + 20.0 * (i as f64 * 0.02).sin(), // pressure
            ];
            
            let mut metadata = HashMap::new();
            metadata.insert(
                "sensor_id".to_string(), 
                serde_json::Value::String("temp_sensor_001".to_string())
            );
            metadata.insert(
                "location".to_string(),
                serde_json::Value::String("Building A, Floor 2".to_string())
            );
            
            let data_point = DataPoint::new(values)
                .with_metadata(metadata);
            
            if let Err(e) = stream.push(data_point).await {
                error!("Failed to push data point: {}", e);
            }
            
            // Add some jitter to simulate real-world data arrival
            let delay = 50 + (rand::random::<u64>() % 100);
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
        }
        
        info!("Finished generating data points");
    });

    // Monitor stream statistics
    let stats_monitor = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
        
        for _ in 0..10 {
            interval.tick().await;
            
            if let Some(stream) = stream_manager.get_stream(stream_id).await {
                let stats = stream.get_stats().await;
                info!("Stream stats - Buffer: {}, Subscribers: {}, Active: {}", 
                      stats.buffer_size, stats.subscriber_count, stats.is_active);
                info!("  Metrics - Received: {}, Processed: {}, Dropped: {}, Errors: {}", 
                      stats.metrics.received_points, 
                      stats.metrics.processed_points,
                      stats.metrics.dropped_points,
                      stats.metrics.processing_errors);
                info!("  Performance - Avg processing: {:.2}ms, Throughput: {:.1}/s, Success rate: {:.1}%",
                      stats.metrics.average_processing_time_ms,
                      stats.metrics.throughput_per_second,
                      stats.metrics.success_rate * 100.0);
            }
        }
    });

    // Wait for tasks to complete
    let _ = tokio::join!(data_generator, result_handler, stats_monitor);

    // Clean up
    info!("Cleaning up...");
    stream.unsubscribe(subscription_id).await;
    stream.stop().await;
    stream_manager.remove_stream(stream_id).await?;

    info!("Streaming example completed!");
    Ok(())
}

/// Create a mock NHITS model for demonstration
fn create_mock_nhits_model() -> std::sync::Arc<dyn crate::ml::nhits::NHITSModelTrait + Send + Sync> {
    std::sync::Arc::new(MockNHITSModel::new())
}

/// Mock NHITS model implementation for testing
struct MockNHITSModel {
    id: String,
    config: NHITSConfig,
    model_state: ModelState,
    training_metadata: TrainingMetadata,
}

impl MockNHITSModel {
    fn new() -> Self {
        let config = NHITSConfig::default();
        let model_state = ModelState {
            epoch: 0,
            loss: f64::INFINITY,
            validation_loss: None,
            weights: HashMap::new(),
            optimizer_state: None,
            last_updated: Utc::now(),
            training_step: 0,
        };
        let training_metadata = TrainingMetadata {
            start_time: Utc::now(),
            end_time: None,
            total_epochs: 0,
            best_loss: f64::INFINITY,
            training_history: Vec::new(),
            data_source: DataSource {
                source_type: "mock".to_string(),
                api_endpoint: "mock".to_string(),
                authentication_required: true,
                data_interval: "1h".to_string(),
                symbols: vec!["MOCK".to_string()],
                last_update: Utc::now(),
            },
        };
        
        Self {
            id: Uuid::new_v4().to_string(),
            config,
            model_state,
            training_metadata,
        }
    }
}

impl crate::ml::nhits::NHITSModelTrait for MockNHITSModel {
    fn get_config(&self) -> &NHITSConfig {
        &self.config
    }
    
    fn get_model_state(&self) -> &ModelState {
        &self.model_state
    }
    
    fn get_model_state_mut(&mut self) -> &mut ModelState {
        &mut self.model_state
    }
    
    fn get_training_metadata(&self) -> &TrainingMetadata {
        &self.training_metadata
    }
    
    fn forward<'a>(&'a self, input: &'a Array3<f64>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Array3<f64>>> + Send + 'a>> {
        Box::pin(async move {
            // Simple mock forward pass
            let (batch_size, seq_len, features) = input.dim();
            let output_shape = (batch_size, self.config.output_size, features);
            Ok(Array3::<f64>::zeros(output_shape))
        })
    }
    
    fn train<'a>(&'a mut self, data_source: DataSource) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Simulate training
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            self.model_state.epoch += 1;
            self.model_state.loss = 0.2;
            self.training_metadata.data_source = data_source;
            Ok(())
        })
    }
    
    fn save_checkpoint<'a>(&'a self, _path: &'a str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            Ok(())
        })
    }
    
    fn predict(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        // Simple mock prediction with realistic output
        let (rows, _cols) = data.dim();
        let output_rows = self.config.output_size.min(rows);
        
        // Generate predictions based on input pattern
        let mut predictions = Vec::new();
        for i in 0..output_rows {
            for j in 0..1 { // Single output feature for simplicity
                let base_value = data.get((i % rows, 0)).unwrap_or(&0.0);
                let trend = 0.1 * (i as f32);
                let seasonal = 2.0 * ((i as f32) * 0.3).sin();
                let noise = (rand::random::<f32>() - 0.5) * 0.5;
                predictions.push(base_value + trend + seasonal + noise);
            }
        }
        
        Array2::from_shape_vec((output_rows, 1), predictions)
            .map_err(|e| anyhow::anyhow!("Failed to create output array: {}", e))
    }
    
    fn train_simple(&mut self, _x: &Array2<f32>, _y: &Array2<f32>) -> Result<()> {
        // Simple mock training
        self.model_state.epoch += 1;
        self.model_state.loss = 0.1;
        self.model_state.training_step += 1;
        Ok(())
    }
}

/// Example of batch streaming processing
pub async fn batch_streaming_example() -> Result<()> {
    fmt::init();
    
    info!("Starting batch streaming example...");

    let config = StreamingConfig {
        buffer_size: 5000,
        batch_size: 50, // Larger batches for efficiency
        processing_interval_ms: 1000,
        max_latency_ms: 5000,
        auto_scaling: true,
        max_concurrent_streams: 5,
    };

    let stream_manager = StreamManager::new(config);
    let stream_id = stream_manager.create_stream().await?;
    let stream = stream_manager.get_stream(stream_id).await.unwrap();

    // Subscribe to results
    let (_subscription_id, mut receiver) = stream.subscribe().await;

    // Start processing
    stream.start().await?;

    // Generate batch data
    let batch_data: Vec<DataPoint> = (0..200)
        .map(|i| {
            let values = vec![
                10.0 + 5.0 * (i as f64 * 0.1).sin(), // sensor 1
                20.0 + 3.0 * (i as f64 * 0.15).cos(), // sensor 2
                30.0 + 2.0 * (i as f64 * 0.05).sin(), // sensor 3
            ];
            DataPoint::new(values)
        })
        .collect();

    info!("Pushing batch of {} data points...", batch_data.len());
    stream.push_batch(batch_data).await?;

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = receiver.recv().await {
        results.push(result);
        if results.len() >= 10 { // Collect some results
            break;
        }
    }

    info!("Collected {} forecast results", results.len());
    for (i, result) in results.iter().enumerate() {
        info!("Result {}: {:?} (latency: {:.2}ms)", 
              i + 1, result.predictions, result.latency_ms);
    }

    // Cleanup
    stream.stop().await;
    stream_manager.remove_stream(stream_id).await?;

    info!("Batch streaming example completed!");
    Ok(())
}

/// Example of multi-stream processing
pub async fn multi_stream_example() -> Result<()> {
    fmt::init();
    
    info!("Starting multi-stream example...");

    let config = StreamingConfig::default();
    let stream_manager = StreamManager::new(config);

    // Create multiple streams for different data sources
    let stream_ids: Vec<Uuid> = futures_util::future::try_join_all(
        (0..3).map(|_| stream_manager.create_stream())
    ).await?;

    info!("Created {} streams", stream_ids.len());

    // Start all streams
    for &stream_id in &stream_ids {
        if let Some(stream) = stream_manager.get_stream(stream_id).await {
            stream.start().await?;
        }
    }

    // Generate data for each stream
    let handles: Vec<_> = stream_ids.iter().enumerate().map(|(i, &stream_id)| {
        let stream_manager = &stream_manager;
        tokio::spawn(async move {
            if let Some(stream) = stream_manager.get_stream(stream_id).await {
                for j in 0..50 {
                    let base_value = (i as f64) * 10.0;
                    let values = vec![
                        base_value + (j as f64 * 0.2).sin(),
                        base_value + (j as f64 * 0.3).cos(),
                    ];
                    
                    let data_point = DataPoint::new(values);
                    if let Err(e) = stream.push(data_point).await {
                        error!("Stream {} push error: {}", i, e);
                    }
                    
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        })
    }).collect();

    // Wait for data generation to complete
    futures_util::future::join_all(handles).await;

    // Get statistics for all streams
    let all_stats = stream_manager.list_streams().await;
    info!("Stream statistics:");
    for stats in all_stats {
        info!("  Stream {}: {} points processed, {:.1}% success rate",
              stats.stream_id, 
              stats.metrics.processed_points,
              stats.metrics.success_rate * 100.0);
    }

    // Cleanup all streams
    for stream_id in stream_ids {
        stream_manager.remove_stream(stream_id).await?;
    }

    info!("Multi-stream example completed!");
    Ok(())
}