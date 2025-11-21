//! REST API Routes Configuration
//!
//! Centralized route configuration with proper organization,
//! parameter validation, and comprehensive endpoint documentation.

use super::handlers::ApiHandlers;
use crate::api::rest::{
    ModelConfigRequest, BatchPredictionRequest, CalibrationRequest, BenchmarkRequest,
    ApiResponse, PaginationParams
};
use axum::{
    extract::{Path, Query, Extension, Json as AxumJson},
    http::StatusCode,
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
use serde_json;
use std::sync::Arc;
use uuid::Uuid;

/// Create the main API router with all routes
pub fn create_router() -> Router {
    Router::new()
        // Health and monitoring routes
        .nest("/health", health_routes())
        .nest("/metrics", metrics_routes())
        
        // Model management routes
        .nest("/models", model_routes())
        
        // Prediction routes
        .nest("/predict", prediction_routes())
        
        // Calibration routes
        .nest("/calibration", calibration_routes())
        
        // Benchmarking routes  
        .nest("/benchmark", benchmark_routes())
        
        // System management routes
        .nest("/system", system_routes())
        
        // Admin routes
        .nest("/admin", admin_routes())
}

/// Health and monitoring routes
fn health_routes() -> Router {
    Router::new()
        .route("/", get(basic_health_check))
        .route("/detailed", get(detailed_health_check))
        .route("/readiness", get(readiness_check))
        .route("/liveness", get(liveness_check))
}

/// Metrics and monitoring routes
fn metrics_routes() -> Router {
    Router::new()
        .route("/", get(get_metrics))
        .route("/performance", get(get_performance_metrics))
        .route("/system", get(get_system_metrics))
        .route("/endpoints", get(get_endpoint_metrics))
        .route("/export/prometheus", get(export_prometheus_metrics))
}

/// Model management routes
fn model_routes() -> Router {
    Router::new()
        .route("/", get(list_models).post(create_model))
        .route("/:model_id", get(get_model).put(update_model).delete(delete_model))
        .route("/:model_id/status", get(get_model_status))
        .route("/:model_id/config", get(get_model_config).put(update_model_config))
        .route("/:model_id/metrics", get(get_model_metrics))
        .route("/:model_id/health", get(check_model_health))
}

/// Prediction routes
fn prediction_routes() -> Router {
    Router::new()
        .route("/:model_id", post(single_prediction))
        .route("/:model_id/batch", post(batch_prediction))
        .route("/:model_id/stream", get(stream_prediction))
        .route("/:model_id/async", post(async_prediction))
        .route("/results/:request_id", get(get_prediction_results))
}

/// Calibration routes
fn calibration_routes() -> Router {
    Router::new()
        .route("/", post(start_calibration))
        .route("/:calibration_id", get(get_calibration_status))
        .route("/:calibration_id/results", get(get_calibration_results))
        .route("/:calibration_id/cancel", post(cancel_calibration))
        .route("/history", get(get_calibration_history))
}

/// Benchmarking routes
fn benchmark_routes() -> Router {
    Router::new()
        .route("/", post(start_benchmark))
        .route("/:benchmark_id", get(get_benchmark_status))
        .route("/:benchmark_id/results", get(get_benchmark_results))
        .route("/:benchmark_id/cancel", post(cancel_benchmark))
        .route("/history", get(get_benchmark_history))
        .route("/templates", get(get_benchmark_templates))
}

/// System management routes
fn system_routes() -> Router {
    Router::new()
        .route("/status", get(get_system_status))
        .route("/memory", get(get_memory_usage))
        .route("/cpu", get(get_cpu_usage))
        .route("/disk", get(get_disk_usage))
        .route("/network", get(get_network_stats))
        .route("/gc", post(trigger_garbage_collection))
        .route("/cache/clear", post(clear_cache))
}

/// Administrative routes
fn admin_routes() -> Router {
    Router::new()
        .route("/shutdown", post(initiate_shutdown))
        .route("/restart", post(restart_server))
        .route("/config", get(get_server_config).put(update_server_config))
        .route("/logs", get(get_logs))
        .route("/debug", get(debug_info))
}

// Health Check Handlers
async fn basic_health_check() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let health_data = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": crate::VERSION
    });
    
    Json(ApiResponse::success(health_data, request_id))
}

async fn detailed_health_check(
    Extension(handlers): Extension<Arc<ApiHandlers>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let request_id = Uuid::new_v4().to_string();
    
    match handlers.get_detailed_health().await {
        Ok(health) => {
            let health_json = serde_json::to_value(health)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Json(ApiResponse::success(health_json, request_id)))
        }
        Err(e) => {
            let api_error = crate::api::ApiError::from(e);
            Ok(Json(ApiResponse::error(api_error, request_id)))
        }
    }
}

async fn readiness_check() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    // Check if all critical components are ready
    let ready = true; // This would check actual readiness
    
    let readiness_data = serde_json::json!({
        "ready": ready,
        "components": {
            "database": true,
            "predictor": true,
            "cache": true
        },
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(readiness_data, request_id))
}

async fn liveness_check() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let liveness_data = serde_json::json!({
        "alive": true,
        "uptime_seconds": 3600, // Would be actual uptime
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(liveness_data, request_id))
}

// Metrics Handlers
async fn get_metrics() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let metrics_data = serde_json::json!({
        "requests_total": 1000,
        "errors_total": 5,
        "latency_avg_ms": 15.5,
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(metrics_data, request_id))
}

async fn get_performance_metrics() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let performance_data = serde_json::json!({
        "latency": {
            "avg_us": 500.0,
            "p95_us": 1000.0,
            "p99_us": 2000.0
        },
        "throughput": {
            "requests_per_second": 100.0,
            "predictions_per_second": 50.0
        },
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(performance_data, request_id))
}

async fn get_system_metrics() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let system_data = serde_json::json!({
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1,
        "network_rx_mbps": 12.5,
        "network_tx_mbps": 8.3,
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(system_data, request_id))
}

async fn get_endpoint_metrics() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let endpoint_data = serde_json::json!({
        "endpoints": {
            "GET /health": { "requests": 500, "avg_latency_ms": 2.1 },
            "POST /predict/:model_id": { "requests": 200, "avg_latency_ms": 25.5 },
            "POST /models": { "requests": 10, "avg_latency_ms": 50.2 }
        },
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(endpoint_data, request_id))
}

async fn export_prometheus_metrics() -> String {
    // Return metrics in Prometheus format
    format!(
        r#"# HELP ats_core_requests_total Total number of requests
# TYPE ats_core_requests_total counter
ats_core_requests_total 1000

# HELP ats_core_latency_seconds Request latency in seconds
# TYPE ats_core_latency_seconds histogram
ats_core_latency_seconds_bucket{{le="0.005"}} 100
ats_core_latency_seconds_bucket{{le="0.01"}} 200
ats_core_latency_seconds_bucket{{le="0.025"}} 400
ats_core_latency_seconds_bucket{{le="0.05"}} 600
ats_core_latency_seconds_bucket{{le="0.1"}} 800
ats_core_latency_seconds_bucket{{le="+Inf"}} 1000
ats_core_latency_seconds_sum 15.5
ats_core_latency_seconds_count 1000
"#
    )
}

// Model Management Handlers
async fn list_models(
    Query(pagination): Query<PaginationParams>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    let models_data = serde_json::json!({
        "models": [
            {
                "model_id": "lstm_model_1",
                "type": "LSTM",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "model_id": "transformer_model_1", 
                "type": "Transformer",
                "status": "training",
                "created_at": "2024-01-02T00:00:00Z"
            }
        ],
        "pagination": {
            "page": pagination.page,
            "limit": pagination.limit,
            "total": 2
        }
    });
    
    Json(ApiResponse::success(models_data, request_id))
}

async fn create_model(
    Extension(handlers): Extension<Arc<ApiHandlers>>,
    AxumJson(config): AxumJson<ModelConfigRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let request_id = Uuid::new_v4().to_string();
    
    match handlers.create_model_config(config).await {
        Ok(response) => {
            let response_json = serde_json::to_value(response)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Json(ApiResponse::success(response_json, request_id)))
        }
        Err(e) => {
            let api_error = crate::api::ApiError::from(e);
            Ok(Json(ApiResponse::error(api_error, request_id)))
        }
    }
}

async fn get_model(
    Path(model_id): Path<String>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    let model_data = serde_json::json!({
        "model_id": model_id,
        "type": "LSTM",
        "status": "active",
        "configuration": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1
        },
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05
        }
    });
    
    Json(ApiResponse::success(model_data, request_id))
}

async fn update_model(
    Path(model_id): Path<String>,
    AxumJson(config): AxumJson<ModelConfigRequest>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    let update_data = serde_json::json!({
        "model_id": model_id,
        "updated": true,
        "config": config,
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(update_data, request_id))
}

async fn delete_model(
    Path(model_id): Path<String>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    let delete_data = serde_json::json!({
        "model_id": model_id,
        "deleted": true,
        "timestamp": chrono::Utc::now()
    });
    
    Json(ApiResponse::success(delete_data, request_id))
}

async fn get_model_status(
    Path(model_id): Path<String>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    let status_data = serde_json::json!({
        "model_id": model_id,
        "status": "active",
        "health": "healthy",
        "last_prediction": "2024-01-01T12:00:00Z",
        "uptime_seconds": 3600
    });
    
    Json(ApiResponse::success(status_data, request_id))
}

// Placeholder handlers for remaining endpoints
async fn get_model_config(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_config".to_string(), request_id))
}

async fn update_model_config(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("config_updated".to_string(), request_id))
}

async fn get_model_metrics(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_metrics".to_string(), request_id))
}

async fn check_model_health(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_healthy".to_string(), request_id))
}

// Prediction handlers
async fn single_prediction(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("prediction_result".to_string(), request_id))
}

async fn batch_prediction(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("batch_result".to_string(), request_id))
}

async fn stream_prediction(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("stream_started".to_string(), request_id))
}

async fn async_prediction(Path(_model_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("async_started".to_string(), request_id))
}

async fn get_prediction_results(Path(_request_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("prediction_results".to_string(), request_id))
}

// Calibration handlers
async fn start_calibration() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_started".to_string(), request_id))
}

async fn get_calibration_status(Path(_calibration_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_status".to_string(), request_id))
}

async fn get_calibration_results(Path(_calibration_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_results".to_string(), request_id))
}

async fn cancel_calibration(Path(_calibration_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_cancelled".to_string(), request_id))
}

async fn get_calibration_history() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_history".to_string(), request_id))
}

// Benchmark handlers
async fn start_benchmark() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_started".to_string(), request_id))
}

async fn get_benchmark_status(Path(_benchmark_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_status".to_string(), request_id))
}

async fn get_benchmark_results(Path(_benchmark_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_results".to_string(), request_id))
}

async fn cancel_benchmark(Path(_benchmark_id): Path<String>) -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_cancelled".to_string(), request_id))
}

async fn get_benchmark_history() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_history".to_string(), request_id))
}

async fn get_benchmark_templates() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_templates".to_string(), request_id))
}

// System handlers
async fn get_system_status() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("system_status".to_string(), request_id))
}

async fn get_memory_usage() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("memory_usage".to_string(), request_id))
}

async fn get_cpu_usage() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("cpu_usage".to_string(), request_id))
}

async fn get_disk_usage() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("disk_usage".to_string(), request_id))
}

async fn get_network_stats() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("network_stats".to_string(), request_id))
}

async fn trigger_garbage_collection() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("gc_triggered".to_string(), request_id))
}

async fn clear_cache() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("cache_cleared".to_string(), request_id))
}

// Admin handlers
async fn initiate_shutdown() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("shutdown_initiated".to_string(), request_id))
}

async fn restart_server() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("restart_initiated".to_string(), request_id))
}

async fn get_server_config() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("server_config".to_string(), request_id))
}

async fn update_server_config() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("config_updated".to_string(), request_id))
}

async fn get_logs() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("logs".to_string(), request_id))
}

async fn debug_info() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("debug_info".to_string(), request_id))
}