use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    time::{Duration, Instant},
};
use uuid::Uuid;

/// API Response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

/// Model configuration for NHITS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of input time steps
    pub input_size: usize,
    /// Number of output time steps to predict
    pub output_size: usize,
    /// Number of stacks in the model
    pub n_stacks: usize,
    /// Number of blocks per stack
    pub n_blocks: Vec<usize>,
    /// Number of layers in each block
    pub n_layers: Vec<usize>,
    /// Hidden layer sizes
    pub layer_widths: Vec<usize>,
    /// Pooling kernel sizes for each stack
    pub pooling_sizes: Vec<usize>,
    /// Interpolation mode
    pub interpolation_mode: String,
    /// Dropout rate
    pub dropout: f32,
    /// Activation function
    pub activation: String,
    /// Maximum number of training epochs
    pub max_epochs: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 32,
            output_size: 12,
            n_stacks: 3,
            n_blocks: vec![1, 1, 1],
            n_layers: vec![2, 2, 2],
            layer_widths: vec![512, 512, 512],
            pooling_sizes: vec![2, 2, 1],
            interpolation_mode: "linear".to_string(),
            dropout: 0.1,
            activation: "relu".to_string(),
            max_epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            patience: 10,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub config: ModelConfig,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub status: ModelStatus,
    pub version: String,
    pub tags: Vec<String>,
    pub metrics: Option<TrainingMetrics>,
}

/// Model status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Created,
    Training,
    Trained,
    Failed,
    Deprecated,
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelStatus::Created => write!(f, "created"),
            ModelStatus::Training => write!(f, "training"),
            ModelStatus::Trained => write!(f, "trained"),
            ModelStatus::Failed => write!(f, "failed"),
            ModelStatus::Deprecated => write!(f, "deprecated"),
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub best_epoch: u32,
    pub best_loss: f64,
    pub training_time: f64,
    pub convergence_rate: f64,
    pub final_metrics: HashMap<String, f64>,
}

/// Request to create a new model
#[derive(Debug, Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    pub description: Option<String>,
    pub config: ModelConfig,
    pub tags: Option<Vec<String>>,
}

/// Request to update an existing model
#[derive(Debug, Deserialize)]
pub struct UpdateModelRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub config: Option<ModelConfig>,
    pub tags: Option<Vec<String>>,
}

/// Training data format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// Input time series data [batch_size, sequence_length, features]
    pub data: Vec<Vec<Vec<f64>>>,
    /// Target values for forecasting [batch_size, forecast_length, features]
    pub targets: Option<Vec<Vec<Vec<f64>>>>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps for the data points
    pub timestamps: Option<Vec<chrono::DateTime<chrono::Utc>>>,
    /// Data preprocessing parameters
    pub preprocessing: Option<PreprocessingConfig>,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub normalize: bool,
    pub standardize: bool,
    pub remove_trend: bool,
    pub seasonal_adjustment: bool,
    pub fill_missing: String, // "mean", "median", "forward", "backward"
}

/// Request to train a model
#[derive(Debug, Deserialize)]
pub struct TrainModelRequest {
    pub data: TrainingData,
    pub validation_split: Option<f32>,
    pub save_checkpoints: Option<bool>,
    pub callback_url: Option<String>,
}

/// Forecasting request
#[derive(Debug, Deserialize)]
pub struct ForecastRequest {
    /// Input time series data
    pub data: Vec<Vec<f64>>,
    /// Number of steps to forecast (overrides model config if provided)
    pub forecast_steps: Option<usize>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Timestamps for input data
    pub timestamps: Option<Vec<chrono::DateTime<chrono::Utc>>>,
    /// Confidence intervals to compute
    pub confidence_intervals: Option<Vec<f64>>,
    /// Whether to return prediction intervals
    pub return_intervals: Option<bool>,
    /// Callback URL for async results
    pub callback_url: Option<String>,
    /// Request priority
    pub priority: Option<ForecastPriority>,
}

/// Forecast priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Forecast job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastJob {
    pub id: Uuid,
    pub model_id: String,
    pub status: JobStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub progress: f32,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    pub priority: ForecastPriority,
    pub error: Option<String>,
    pub result: Option<ForecastResult>,
    #[serde(skip)]
    pub created_instant: Instant,
}

impl ForecastJob {
    pub fn new(model_id: String, priority: ForecastPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            model_id,
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            estimated_completion: None,
            priority,
            error: None,
            result: None,
            created_instant: Instant::now(),
        }
    }

    pub fn is_expired(&self) -> bool {
        // Jobs expire after 24 hours
        self.created_instant.elapsed() > Duration::from_secs(24 * 3600)
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, JobStatus::Queued | JobStatus::Running)
    }
}

/// Job status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobStatus::Queued => write!(f, "queued"),
            JobStatus::Running => write!(f, "running"),
            JobStatus::Completed => write!(f, "completed"),
            JobStatus::Failed => write!(f, "failed"),
            JobStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Predicted values [forecast_length, features]
    pub predictions: Vec<Vec<f64>>,
    /// Prediction timestamps
    pub timestamps: Option<Vec<chrono::DateTime<chrono::Utc>>>,
    /// Confidence intervals if requested
    pub confidence_intervals: Option<HashMap<String, Vec<Vec<f64>>>>,
    /// Model performance metrics
    pub metrics: Option<HashMap<String, f64>>,
    /// Feature importance scores
    pub feature_importance: Option<Vec<f64>>,
    /// Computation time in seconds
    pub computation_time: f64,
}

/// Request to list models with filtering
#[derive(Debug, Deserialize)]
pub struct ListModelsQuery {
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub status: Option<String>,
    pub tags: Option<Vec<String>>,
    pub search: Option<String>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
}

/// Paginated response
#[derive(Debug, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: u64,
    pub page: u32,
    pub page_size: u32,
    pub total_pages: u32,
}

/// Request to list forecast jobs
#[derive(Debug, Deserialize)]
pub struct ListJobsQuery {
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub status: Option<String>,
    pub model_id: Option<String>,
    pub priority: Option<String>,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
}

/// Batch forecast request
#[derive(Debug, Deserialize)]
pub struct BatchForecastRequest {
    pub requests: Vec<ForecastRequest>,
    pub callback_url: Option<String>,
    pub max_parallel: Option<usize>,
}

/// System metrics response
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub active_models: usize,
    pub active_jobs: usize,
    pub completed_jobs_today: usize,
    pub failed_jobs_today: usize,
    pub average_response_time: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub uptime: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorResponse {
    pub fn new(error: String, code: String) -> Self {
        Self {
            error,
            code,
            details: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.details = Some(details);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.input_size, 32);
        assert_eq!(config.output_size, 12);
        assert_eq!(config.n_stacks, 3);
    }

    #[test]
    fn test_forecast_job_creation() {
        let job = ForecastJob::new("test_model".to_string(), ForecastPriority::Normal);
        assert_eq!(job.model_id, "test_model");
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.progress, 0.0);
        assert!(job.is_active());
        assert!(!job.is_expired());
    }

    #[test]
    fn test_api_response_success() {
        let response = ApiResponse::success("test data");
        assert!(response.success);
        assert_eq!(response.data, Some("test data"));
        assert!(response.error.is_none());
    }

    #[test]
    fn test_api_response_error() {
        let response: ApiResponse<String> = ApiResponse::error("test error".to_string());
        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some("test error".to_string()));
    }

    #[test]
    fn test_model_status_display() {
        assert_eq!(ModelStatus::Created.to_string(), "created");
        assert_eq!(ModelStatus::Training.to_string(), "training");
        assert_eq!(ModelStatus::Trained.to_string(), "trained");
    }

    #[test]
    fn test_job_status_display() {
        assert_eq!(JobStatus::Queued.to_string(), "queued");
        assert_eq!(JobStatus::Running.to_string(), "running");
        assert_eq!(JobStatus::Completed.to_string(), "completed");
    }
}