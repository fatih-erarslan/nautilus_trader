use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::ml::nhits::api::{
    models::*,
    server::AppState,
    websocket::{send_forecast_update, send_training_update},
};

/// List all models with optional filtering
pub async fn list_models(
    Query(query): Query<ListModelsQuery>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<PaginatedResponse<ModelMetadata>>>, StatusCode> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20).min(100); // Max 100 items per page
    
    let models = state.models.read().await;
    let total = models.len() as u64;
    
    // Apply filtering (simplified implementation)
    let mut filtered_models: Vec<ModelMetadata> = Vec::new();
    
    // In a real implementation, you would:
    // 1. Filter by status, tags, search term
    // 2. Sort by the specified field
    // 3. Apply pagination
    
    // For now, return a simple paginated response
    let total_pages = ((total as f64) / (page_size as f64)).ceil() as u32;
    
    let response = PaginatedResponse {
        items: filtered_models,
        total,
        page,
        page_size,
        total_pages,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get a specific model by ID
pub async fn get_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<ModelMetadata>>, StatusCode> {
    let models = state.models.read().await;
    
    // In a real implementation, you would fetch the model metadata
    // For now, return a sample model
    let model_metadata = ModelMetadata {
        id: model_id.clone(),
        name: format!("NHITS Model {}", model_id),
        description: Some("Sample NHITS forecasting model".to_string()),
        config: ModelConfig::default(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        status: ModelStatus::Trained,
        version: "1.0.0".to_string(),
        tags: vec!["forecasting".to_string(), "time-series".to_string()],
        metrics: None,
    };
    
    Ok(Json(ApiResponse::success(model_metadata)))
}

/// Create a new model
pub async fn create_model(
    State(state): State<AppState>,
    Json(request): Json<CreateModelRequest>,
) -> Result<Json<ApiResponse<ModelMetadata>>, StatusCode> {
    let model_id = Uuid::new_v4().to_string();
    
    // Create model metadata
    let model_metadata = ModelMetadata {
        id: model_id.clone(),
        name: request.name,
        description: request.description,
        config: request.config,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        status: ModelStatus::Created,
        version: "1.0.0".to_string(),
        tags: request.tags.unwrap_or_default(),
        metrics: None,
    };
    
    // In a real implementation, you would:
    // 1. Validate the configuration
    // 2. Initialize the NHITS model
    // 3. Store the model metadata in the database
    
    info!("Created new model: {}", model_id);
    
    // Update metrics
    state.metrics.increment_models_created().await;
    
    Ok(Json(ApiResponse::success(model_metadata)))
}

/// Update an existing model
pub async fn update_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<UpdateModelRequest>,
) -> Result<Json<ApiResponse<ModelMetadata>>, StatusCode> {
    let models = state.models.read().await;
    
    // In a real implementation, you would:
    // 1. Check if the model exists
    // 2. Validate the updates
    // 3. Update the model metadata
    // 4. Handle model retraining if config changed
    
    // For now, return an updated model metadata
    let updated_metadata = ModelMetadata {
        id: model_id.clone(),
        name: request.name.unwrap_or_else(|| format!("NHITS Model {}", model_id)),
        description: request.description,
        config: request.config.unwrap_or_default(),
        created_at: chrono::Utc::now() - chrono::Duration::hours(1), // Sample created time
        updated_at: chrono::Utc::now(),
        status: ModelStatus::Trained,
        version: "1.1.0".to_string(),
        tags: request.tags.unwrap_or_default(),
        metrics: None,
    };
    
    info!("Updated model: {}", model_id);
    
    Ok(Json(ApiResponse::success(updated_metadata)))
}

/// Delete a model
pub async fn delete_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let mut models = state.models.write().await;
    
    // In a real implementation, you would:
    // 1. Check if the model exists
    // 2. Check if there are active jobs using this model
    // 3. Clean up model files and resources
    // 4. Remove from database
    
    models.remove(&model_id);
    
    info!("Deleted model: {}", model_id);
    
    Ok(Json(ApiResponse::success(json!({
        "message": "Model deleted successfully",
        "model_id": model_id
    }))))
}

/// Train a model with provided data
pub async fn train_model(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<TrainModelRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let models = state.models.read().await;
    
    // In a real implementation, you would:
    // 1. Validate the training data
    // 2. Start an async training job
    // 3. Send progress updates via WebSocket
    // 4. Update model status and metrics
    
    // Simulate training process
    let training_job_id = Uuid::new_v4();
    
    // Start background training task
    let websocket_connections = state.websocket_connections.clone();
    let model_id_clone = model_id.clone();
    
    tokio::spawn(async move {
        // Simulate training epochs
        for epoch in 1..=10 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            let loss = 1.0 / (epoch as f64); // Decreasing loss
            let mut metrics = HashMap::new();
            metrics.insert("accuracy".to_string(), 0.85 + (epoch as f64 * 0.01));
            
            // Send training update via WebSocket
            if let Err(e) = send_training_update(
                &websocket_connections,
                model_id_clone.clone(),
                epoch,
                loss,
                metrics,
            ).await {
                error!("Failed to send training update: {}", e);
            }
        }
        
        info!("Training completed for model: {}", model_id_clone);
    });
    
    info!("Started training for model: {}", model_id);
    
    Ok(Json(ApiResponse::success(json!({
        "message": "Training started",
        "model_id": model_id,
        "training_job_id": training_job_id
    }))))
}

/// Create a forecast request
pub async fn create_forecast(
    Path(model_id): Path<String>,
    State(state): State<AppState>,
    Json(request): Json<ForecastRequest>,
) -> Result<Json<ApiResponse<ForecastJob>>, StatusCode> {
    let models = state.models.read().await;
    
    // Validate model exists (in real implementation)
    if !models.contains_key(&model_id) {
        return Err(StatusCode::NOT_FOUND);
    }
    
    // Create forecast job
    let priority = request.priority.unwrap_or(ForecastPriority::Normal);
    let mut job = ForecastJob::new(model_id.clone(), priority);
    job.status = JobStatus::Queued;
    
    let job_id = job.id;
    
    // Store job
    {
        let mut jobs = state.forecast_jobs.write().await;
        jobs.insert(job_id, job.clone());
    }
    
    // Start background forecast task
    let state_clone = state.clone();
    let forecast_data = request.data.clone();
    let forecast_steps = request.forecast_steps.unwrap_or(12);
    
    tokio::spawn(async move {
        if let Err(e) = execute_forecast(
            state_clone,
            job_id,
            model_id,
            forecast_data,
            forecast_steps,
        ).await {
            error!("Forecast execution failed: {}", e);
            
            // Update job status to failed
            if let Ok(mut jobs) = state_clone.forecast_jobs.write().await {
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed;
                    job.error = Some(e.to_string());
                    job.completed_at = Some(chrono::Utc::now());
                }
            }
        }
    });
    
    info!("Created forecast job: {} for model: {}", job_id, model_id);
    
    // Update metrics
    state.metrics.increment_forecasts_created().await;
    
    Ok(Json(ApiResponse::success(job)))
}

/// Execute forecast in background
async fn execute_forecast(
    state: AppState,
    job_id: Uuid,
    model_id: String,
    data: Vec<Vec<f64>>,
    forecast_steps: usize,
) -> Result<()> {
    let start_time = Instant::now();
    
    // Update job status to running
    {
        let mut jobs = state.forecast_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running;
            job.started_at = Some(chrono::Utc::now());
        }
    }
    
    // Simulate forecast computation with progress updates
    for step in 1..=forecast_steps {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let progress = (step as f32) / (forecast_steps as f32);
        
        // Update job progress
        {
            let mut jobs = state.forecast_jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.progress = progress;
            }
        }
        
        // Send progress update via WebSocket
        if let Err(e) = send_forecast_update(
            &state.websocket_connections,
            job_id,
            model_id.clone(),
            progress,
            None, // In real implementation, send partial results
        ).await {
            warn!("Failed to send forecast update: {}", e);
        }
    }
    
    // Generate mock forecast results
    let predictions = (0..forecast_steps)
        .map(|i| vec![100.0 + (i as f64) * 2.5]) // Mock predictions
        .collect();
    
    let result = ForecastResult {
        predictions,
        timestamps: None,
        confidence_intervals: None,
        metrics: Some({
            let mut metrics = HashMap::new();
            metrics.insert("mae".to_string(), 2.3);
            metrics.insert("rmse".to_string(), 3.1);
            metrics.insert("mape".to_string(), 0.05);
            metrics
        }),
        feature_importance: None,
        computation_time: start_time.elapsed().as_secs_f64(),
    };
    
    // Update job with results
    {
        let mut jobs = state.forecast_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Completed;
            job.progress = 1.0;
            job.completed_at = Some(chrono::Utc::now());
            job.result = Some(result);
        }
    }
    
    info!("Forecast completed for job: {}", job_id);
    
    Ok(())
}

/// List forecast jobs
pub async fn list_forecasts(
    Query(query): Query<ListJobsQuery>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<PaginatedResponse<ForecastJob>>>, StatusCode> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20).min(100);
    
    let jobs = state.forecast_jobs.read().await;
    let mut job_list: Vec<ForecastJob> = jobs.values().cloned().collect();
    
    // Apply filtering and sorting (simplified)
    if let Some(status) = &query.status {
        job_list.retain(|job| job.status.to_string() == *status);
    }
    
    if let Some(model_id) = &query.model_id {
        job_list.retain(|job| job.model_id == *model_id);
    }
    
    // Sort by creation time (newest first)
    job_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    
    let total = job_list.len() as u64;
    let total_pages = ((total as f64) / (page_size as f64)).ceil() as u32;
    
    // Apply pagination
    let start = ((page - 1) * page_size) as usize;
    let end = (start + page_size as usize).min(job_list.len());
    let items = job_list[start..end].to_vec();
    
    let response = PaginatedResponse {
        items,
        total,
        page,
        page_size,
        total_pages,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get a specific forecast job
pub async fn get_forecast(
    Path(job_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<ForecastJob>>, StatusCode> {
    let jobs = state.forecast_jobs.read().await;
    
    match jobs.get(&job_id) {
        Some(job) => Ok(Json(ApiResponse::success(job.clone()))),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Cancel a forecast job
pub async fn cancel_forecast(
    Path(job_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let mut jobs = state.forecast_jobs.write().await;
    
    match jobs.get_mut(&job_id) {
        Some(job) => {
            if job.is_active() {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(chrono::Utc::now());
                
                info!("Cancelled forecast job: {}", job_id);
                
                Ok(Json(ApiResponse::success(json!({
                    "message": "Forecast job cancelled",
                    "job_id": job_id
                }))))
            } else {
                Err(StatusCode::BAD_REQUEST)
            }
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::nhits::api::server::ServerConfig;

    fn create_test_state() -> AppState {
        AppState::new(ServerConfig::default())
    }

    #[tokio::test]
    async fn test_create_model() {
        let state = create_test_state();
        let request = CreateModelRequest {
            name: "Test Model".to_string(),
            description: Some("Test description".to_string()),
            config: ModelConfig::default(),
            tags: Some(vec!["test".to_string()]),
        };

        let result = create_model(State(state), Json(request)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_forecast_job_creation() {
        let state = create_test_state();
        let model_id = "test_model".to_string();
        
        // Add a mock model
        {
            let mut models = state.models.write().await;
            models.insert(model_id.clone(), Box::new(crate::ml::nhits::NHITSModel::default()));
        }
        
        let request = ForecastRequest {
            data: vec![vec![1.0, 2.0, 3.0]],
            forecast_steps: Some(5),
            feature_names: None,
            timestamps: None,
            confidence_intervals: None,
            return_intervals: None,
            callback_url: None,
            priority: Some(ForecastPriority::High),
        };

        let result = create_forecast(Path(model_id), State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}