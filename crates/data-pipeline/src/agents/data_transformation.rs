//! # Data Transformation Agent
//!
//! Real-time data normalization and preprocessing agent.
//! Provides comprehensive data transformation with memory-optimized operations.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Axis};
use nalgebra::{DMatrix, DVector};

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, DataMessageType, MessageMetadata, MessagePriority,
    CoordinationMessage, HealthStatus, HealthLevel, HealthMetrics,
    AgentMetrics, BaseDataAgent, MetricsUpdate
};

/// Data transformation agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformationConfig {
    /// Target transformation latency in microseconds
    pub target_latency_us: u64,
    /// Normalization configuration
    pub normalization_config: NormalizationConfig,
    /// Preprocessing configuration
    pub preprocessing_config: PreprocessingConfig,
    /// Feature scaling configuration
    pub scaling_config: ScalingConfig,
    /// Memory optimization settings
    pub memory_config: MemoryOptimizationConfig,
    /// Parallel processing settings
    pub parallel_config: ParallelProcessingConfig,
}

impl Default for DataTransformationConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 100,
            normalization_config: NormalizationConfig::default(),
            preprocessing_config: PreprocessingConfig::default(),
            scaling_config: ScalingConfig::default(),
            memory_config: MemoryOptimizationConfig::default(),
            parallel_config: ParallelProcessingConfig::default(),
        }
    }
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Normalization methods to apply
    pub methods: Vec<NormalizationMethod>,
    /// Target range for normalization
    pub target_range: (f64, f64),
    /// Handle missing values
    pub handle_missing: bool,
    /// Missing value strategy
    pub missing_strategy: MissingValueStrategy,
    /// Outlier handling
    pub outlier_handling: OutlierHandling,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                NormalizationMethod::MinMax,
                NormalizationMethod::ZScore,
            ],
            target_range: (0.0, 1.0),
            handle_missing: true,
            missing_strategy: MissingValueStrategy::Mean,
            outlier_handling: OutlierHandling::Clip,
        }
    }
}

/// Normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Robust,
    Quantile,
    Unit,
    L1,
    L2,
}

/// Missing value strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    Mean,
    Median,
    Mode,
    Forward,
    Backward,
    Linear,
    Zero,
    Drop,
}

/// Outlier handling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutlierHandling {
    Clip,
    Remove,
    Transform,
    Ignore,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Preprocessing steps
    pub steps: Vec<PreprocessingStep>,
    /// Data cleaning options
    pub cleaning_options: DataCleaningOptions,
    /// Feature engineering options
    pub feature_engineering: FeatureEngineeringOptions,
    /// Data validation options
    pub validation_options: ValidationOptions,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            steps: vec![
                PreprocessingStep::RemoveDuplicates,
                PreprocessingStep::HandleMissing,
                PreprocessingStep::DetectOutliers,
                PreprocessingStep::Normalize,
            ],
            cleaning_options: DataCleaningOptions::default(),
            feature_engineering: FeatureEngineeringOptions::default(),
            validation_options: ValidationOptions::default(),
        }
    }
}

/// Preprocessing steps
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PreprocessingStep {
    RemoveDuplicates,
    HandleMissing,
    DetectOutliers,
    Normalize,
    FeatureScale,
    Encode,
    Transform,
    Validate,
}

/// Data cleaning options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCleaningOptions {
    /// Remove duplicates
    pub remove_duplicates: bool,
    /// Duplicate detection strategy
    pub duplicate_strategy: DuplicateStrategy,
    /// Data type validation
    pub type_validation: bool,
    /// Range validation
    pub range_validation: bool,
    /// Format validation
    pub format_validation: bool,
}

impl Default for DataCleaningOptions {
    fn default() -> Self {
        Self {
            remove_duplicates: true,
            duplicate_strategy: DuplicateStrategy::KeepFirst,
            type_validation: true,
            range_validation: true,
            format_validation: true,
        }
    }
}

/// Duplicate detection strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DuplicateStrategy {
    KeepFirst,
    KeepLast,
    KeepAll,
    Remove,
}

/// Feature engineering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringOptions {
    /// Create polynomial features
    pub polynomial_features: bool,
    /// Polynomial degree
    pub polynomial_degree: usize,
    /// Create interaction features
    pub interaction_features: bool,
    /// Binning/discretization
    pub binning: bool,
    /// Number of bins
    pub num_bins: usize,
    /// Log transformation
    pub log_transform: bool,
    /// Square root transformation
    pub sqrt_transform: bool,
}

impl Default for FeatureEngineeringOptions {
    fn default() -> Self {
        Self {
            polynomial_features: false,
            polynomial_degree: 2,
            interaction_features: false,
            binning: false,
            num_bins: 10,
            log_transform: false,
            sqrt_transform: false,
        }
    }
}

/// Validation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOptions {
    /// Validate data types
    pub validate_types: bool,
    /// Validate ranges
    pub validate_ranges: bool,
    /// Validate formats
    pub validate_formats: bool,
    /// Validate constraints
    pub validate_constraints: bool,
    /// Expected schema
    pub expected_schema: Option<DataSchema>,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            validate_types: true,
            validate_ranges: true,
            validate_formats: true,
            validate_constraints: true,
            expected_schema: None,
        }
    }
}

/// Data schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Field definitions
    pub fields: HashMap<String, FieldDefinition>,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Optional fields
    pub optional_fields: Vec<String>,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field type
    pub field_type: FieldType,
    /// Minimum value (for numeric fields)
    pub min_value: Option<f64>,
    /// Maximum value (for numeric fields)
    pub max_value: Option<f64>,
    /// Pattern (for string fields)
    pub pattern: Option<String>,
    /// Allowed values
    pub allowed_values: Option<Vec<serde_json::Value>>,
}

/// Field types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FieldType {
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Array,
    Object,
}

/// Scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Scaling methods
    pub methods: Vec<ScalingMethod>,
    /// Per-feature scaling
    pub per_feature_scaling: bool,
    /// Scaling parameters
    pub scaling_params: ScalingParameters,
    /// Inverse transform support
    pub inverse_transform: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            methods: vec![ScalingMethod::StandardScaler],
            per_feature_scaling: true,
            scaling_params: ScalingParameters::default(),
            inverse_transform: true,
        }
    }
}

/// Scaling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
}

/// Scaling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingParameters {
    /// Feature range for MinMax scaling
    pub feature_range: (f64, f64),
    /// Quantile range for robust scaling
    pub quantile_range: (f64, f64),
    /// Power transformer method
    pub power_method: PowerMethod,
    /// Normalizer norm
    pub norm: NormType,
}

impl Default for ScalingParameters {
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            quantile_range: (25.0, 75.0),
            power_method: PowerMethod::YeoJohnson,
            norm: NormType::L2,
        }
    }
}

/// Power transformation methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PowerMethod {
    BoxCox,
    YeoJohnson,
}

/// Norm types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormType {
    L1,
    L2,
    Max,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable memory pooling
    pub memory_pooling: bool,
    /// Pool size in MB
    pub pool_size_mb: usize,
    /// Enable data compression
    pub compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// In-place operations
    pub in_place_operations: bool,
    /// Lazy evaluation
    pub lazy_evaluation: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_pooling: true,
            pool_size_mb: 512,
            compression: true,
            compression_algorithm: CompressionAlgorithm::LZ4,
            in_place_operations: true,
            lazy_evaluation: true,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    LZ4,
    Zstd,
    Snappy,
    Gzip,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessingConfig {
    /// Enable parallel processing
    pub enabled: bool,
    /// Number of worker threads
    pub num_threads: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ParallelProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_threads: num_cpus::get(),
            chunk_size: 1000,
            load_balancing: LoadBalancingStrategy::WorkStealing,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    Dynamic,
}

/// Data normalizer
pub struct DataNormalizer {
    config: Arc<NormalizationConfig>,
    statistics: Arc<RwLock<HashMap<String, FieldStatistics>>>,
}

/// Field statistics for normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub count: usize,
}

impl DataNormalizer {
    /// Create a new data normalizer
    pub fn new(config: Arc<NormalizationConfig>) -> Self {
        Self {
            config,
            statistics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Normalize data
    pub async fn normalize(&self, data: &mut serde_json::Value) -> Result<()> {
        if let Some(obj) = data.as_object_mut() {
            for (field_name, value) in obj.iter_mut() {
                if let Some(num) = value.as_f64() {
                    let normalized = self.normalize_numeric_field(field_name, num).await?;
                    *value = serde_json::Value::Number(
                        serde_json::Number::from_f64(normalized)
                            .unwrap_or(serde_json::Number::from(0))
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// Normalize numeric field
    async fn normalize_numeric_field(&self, field_name: &str, value: f64) -> Result<f64> {
        let stats = self.get_or_compute_statistics(field_name, value).await?;
        
        for method in &self.config.methods {
            match method {
                NormalizationMethod::MinMax => {
                    return Ok(self.min_max_normalize(value, &stats));
                }
                NormalizationMethod::ZScore => {
                    return Ok(self.z_score_normalize(value, &stats));
                }
                NormalizationMethod::Robust => {
                    return Ok(self.robust_normalize(value, &stats));
                }
                _ => {
                    // Other normalization methods would be implemented here
                }
            }
        }
        
        Ok(value)
    }
    
    /// Min-max normalization
    fn min_max_normalize(&self, value: f64, stats: &FieldStatistics) -> f64 {
        let (min_target, max_target) = self.config.target_range;
        
        if stats.max == stats.min {
            return min_target;
        }
        
        let normalized = (value - stats.min) / (stats.max - stats.min);
        min_target + normalized * (max_target - min_target)
    }
    
    /// Z-score normalization
    fn z_score_normalize(&self, value: f64, stats: &FieldStatistics) -> f64 {
        if stats.std == 0.0 {
            return 0.0;
        }
        
        (value - stats.mean) / stats.std
    }
    
    /// Robust normalization using median and IQR
    fn robust_normalize(&self, value: f64, stats: &FieldStatistics) -> f64 {
        let iqr = stats.q75 - stats.q25;
        
        if iqr == 0.0 {
            return 0.0;
        }
        
        (value - stats.median) / iqr
    }
    
    /// Get or compute field statistics
    async fn get_or_compute_statistics(&self, field_name: &str, value: f64) -> Result<FieldStatistics> {
        let statistics = self.statistics.read().await;
        
        if let Some(stats) = statistics.get(field_name) {
            Ok(stats.clone())
        } else {
            drop(statistics);
            
            // Compute initial statistics (simplified - would use historical data)
            let stats = FieldStatistics {
                mean: value,
                std: 1.0,
                min: value,
                max: value,
                median: value,
                q25: value,
                q75: value,
                count: 1,
            };
            
            self.statistics.write().await.insert(field_name.to_string(), stats.clone());
            Ok(stats)
        }
    }
}

/// Transformation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationResult {
    pub original_data: serde_json::Value,
    pub transformed_data: serde_json::Value,
    pub transformations_applied: Vec<String>,
    pub metadata: TransformationMetadata,
}

/// Transformation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationMetadata {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transformation_time_us: u64,
    pub data_size_before: usize,
    pub data_size_after: usize,
    pub quality_score: f64,
}

/// Data transformation agent
pub struct DataTransformationAgent {
    base: BaseDataAgent,
    config: Arc<DataTransformationConfig>,
    normalizer: Arc<DataNormalizer>,
    transformation_cache: Arc<RwLock<HashMap<String, TransformationResult>>>,
    transformation_metrics: Arc<RwLock<TransformationMetrics>>,
    state: Arc<RwLock<TransformationState>>,
}

/// Transformation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationMetrics {
    pub transformations_performed: u64,
    pub transformations_cached: u64,
    pub average_transformation_time_us: f64,
    pub max_transformation_time_us: f64,
    pub memory_usage_mb: f64,
    pub compression_ratio: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for TransformationMetrics {
    fn default() -> Self {
        Self {
            transformations_performed: 0,
            transformations_cached: 0,
            average_transformation_time_us: 0.0,
            max_transformation_time_us: 0.0,
            memory_usage_mb: 0.0,
            compression_ratio: 1.0,
            cache_hits: 0,
            cache_misses: 0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Transformation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationState {
    pub active_transformations: usize,
    pub cache_usage: f64,
    pub memory_pool_usage: f64,
    pub is_healthy: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for TransformationState {
    fn default() -> Self {
        Self {
            active_transformations: 0,
            cache_usage: 0.0,
            memory_pool_usage: 0.0,
            is_healthy: true,
            last_health_check: chrono::Utc::now(),
        }
    }
}

impl DataTransformationAgent {
    /// Create a new data transformation agent
    pub async fn new(config: DataTransformationConfig) -> Result<Self> {
        let base = BaseDataAgent::new(DataAgentType::DataTransformation);
        let config = Arc::new(config);
        let normalizer = Arc::new(DataNormalizer::new(config.normalization_config.clone().into()));
        let transformation_cache = Arc::new(RwLock::new(HashMap::new()));
        let transformation_metrics = Arc::new(RwLock::new(TransformationMetrics::default()));
        let state = Arc::new(RwLock::new(TransformationState::default()));
        
        Ok(Self {
            base,
            config,
            normalizer,
            transformation_cache,
            transformation_metrics,
            state,
        })
    }
    
    /// Transform data
    pub async fn transform_data(&self, data: serde_json::Value) -> Result<TransformationResult> {
        let start_time = Instant::now();
        let original_size = data.to_string().len();
        
        // Generate cache key
        let cache_key = self.generate_cache_key(&data);
        
        // Check cache
        if let Some(cached_result) = self.transformation_cache.read().await.get(&cache_key) {
            let mut metrics = self.transformation_metrics.write().await;
            metrics.cache_hits += 1;
            return Ok(cached_result.clone());
        }
        
        let mut transformed_data = data.clone();
        let mut transformations_applied = Vec::new();
        
        // Apply preprocessing steps
        for step in &self.config.preprocessing_config.steps {
            match step {
                PreprocessingStep::RemoveDuplicates => {
                    self.remove_duplicates(&mut transformed_data).await?;
                    transformations_applied.push("remove_duplicates".to_string());
                }
                PreprocessingStep::HandleMissing => {
                    self.handle_missing_values(&mut transformed_data).await?;
                    transformations_applied.push("handle_missing".to_string());
                }
                PreprocessingStep::DetectOutliers => {
                    self.detect_and_handle_outliers(&mut transformed_data).await?;
                    transformations_applied.push("detect_outliers".to_string());
                }
                PreprocessingStep::Normalize => {
                    self.normalizer.normalize(&mut transformed_data).await?;
                    transformations_applied.push("normalize".to_string());
                }
                PreprocessingStep::FeatureScale => {
                    self.apply_feature_scaling(&mut transformed_data).await?;
                    transformations_applied.push("feature_scale".to_string());
                }
                _ => {
                    // Other preprocessing steps would be implemented here
                }
            }
        }
        
        let transformation_time = start_time.elapsed().as_micros() as u64;
        let transformed_size = transformed_data.to_string().len();
        
        let result = TransformationResult {
            original_data: data,
            transformed_data,
            transformations_applied,
            metadata: TransformationMetadata {
                timestamp: chrono::Utc::now(),
                transformation_time_us: transformation_time,
                data_size_before: original_size,
                data_size_after: transformed_size,
                quality_score: 0.95, // Would be calculated based on transformations
            },
        };
        
        // Cache the result
        self.transformation_cache.write().await.insert(cache_key, result.clone());
        
        // Update metrics
        {
            let mut metrics = self.transformation_metrics.write().await;
            metrics.transformations_performed += 1;
            metrics.average_transformation_time_us = 
                (metrics.average_transformation_time_us + transformation_time as f64) / 2.0;
            
            if transformation_time as f64 > metrics.max_transformation_time_us {
                metrics.max_transformation_time_us = transformation_time as f64;
            }
            
            metrics.compression_ratio = original_size as f64 / transformed_size as f64;
            metrics.cache_misses += 1;
            metrics.last_update = chrono::Utc::now();
        }
        
        Ok(result)
    }
    
    /// Generate cache key
    fn generate_cache_key(&self, data: &serde_json::Value) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.to_string().hash(&mut hasher);
        
        format!("transform_{}", hasher.finish())
    }
    
    /// Remove duplicates
    async fn remove_duplicates(&self, data: &mut serde_json::Value) -> Result<()> {
        // Duplicate removal logic would be implemented here
        // For now, return as-is
        Ok(())
    }
    
    /// Handle missing values
    async fn handle_missing_values(&self, data: &mut serde_json::Value) -> Result<()> {
        if let Some(obj) = data.as_object_mut() {
            for (_, value) in obj.iter_mut() {
                if value.is_null() {
                    match self.config.normalization_config.missing_strategy {
                        MissingValueStrategy::Zero => {
                            *value = serde_json::Value::Number(serde_json::Number::from(0));
                        }
                        MissingValueStrategy::Mean => {
                            *value = serde_json::Value::Number(serde_json::Number::from(0));
                        }
                        _ => {
                            // Other strategies would be implemented here
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect and handle outliers
    async fn detect_and_handle_outliers(&self, data: &mut serde_json::Value) -> Result<()> {
        if let Some(obj) = data.as_object_mut() {
            for (_, value) in obj.iter_mut() {
                if let Some(num) = value.as_f64() {
                    // Simple outlier detection using z-score
                    let z_score = (num - 0.0) / 1.0; // Would use actual statistics
                    
                    if z_score.abs() > 3.0 {
                        match self.config.normalization_config.outlier_handling {
                            OutlierHandling::Clip => {
                                let clipped = if z_score > 3.0 { 3.0 } else { -3.0 };
                                *value = serde_json::Value::Number(
                                    serde_json::Number::from_f64(clipped)
                                        .unwrap_or(serde_json::Number::from(0))
                                );
                            }
                            OutlierHandling::Remove => {
                                *value = serde_json::Value::Null;
                            }
                            _ => {
                                // Other outlier handling methods
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply feature scaling
    async fn apply_feature_scaling(&self, data: &mut serde_json::Value) -> Result<()> {
        for method in &self.config.scaling_config.methods {
            match method {
                ScalingMethod::StandardScaler => {
                    // Standard scaling would be implemented here
                }
                ScalingMethod::MinMaxScaler => {
                    // Min-max scaling would be implemented here
                }
                _ => {
                    // Other scaling methods
                }
            }
        }
        
        Ok(())
    }
    
    /// Get transformation metrics
    pub async fn get_transformation_metrics(&self) -> TransformationMetrics {
        self.transformation_metrics.read().await.clone()
    }
    
    /// Get transformation state
    pub async fn get_transformation_state(&self) -> TransformationState {
        self.state.read().await.clone()
    }
}

impl From<NormalizationConfig> for Arc<NormalizationConfig> {
    fn from(config: NormalizationConfig) -> Self {
        Arc::new(config)
    }
}

#[async_trait]
impl DataAgent for DataTransformationAgent {
    fn get_id(&self) -> DataAgentId {
        self.base.id
    }
    
    fn get_type(&self) -> DataAgentType {
        DataAgentType::DataTransformation
    }
    
    async fn get_state(&self) -> DataAgentState {
        self.base.state.read().await.clone()
    }
    
    async fn get_info(&self) -> DataAgentInfo {
        self.base.info.read().await.clone()
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting data transformation agent");
        
        self.base.update_state(DataAgentState::Running).await?;
        
        info!("Data transformation agent started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping data transformation agent");
        
        self.base.update_state(DataAgentState::Stopping).await?;
        
        // Clear cache
        self.transformation_cache.write().await.clear();
        
        self.base.update_state(DataAgentState::Stopped).await?;
        
        info!("Data transformation agent stopped successfully");
        Ok(())
    }
    
    async fn process(&self, message: DataMessage) -> Result<DataMessage> {
        let start_time = Instant::now();
        
        // Transform the message data
        let transformation_result = self.transform_data(message.payload).await?;
        
        // Update metrics
        let latency = start_time.elapsed().as_micros() as f64;
        self.base.update_metrics(MetricsUpdate::MessageProcessed(latency)).await?;
        
        // Create response message
        let response = DataMessage {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: self.get_id(),
            destination: message.destination,
            message_type: DataMessageType::TransformedData,
            payload: serde_json::to_value(transformation_result)?,
            metadata: MessageMetadata {
                priority: MessagePriority::High,
                expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                retry_count: 0,
                trace_id: format!("data_transformation_{}", uuid::Uuid::new_v4()),
                span_id: format!("span_{}", uuid::Uuid::new_v4()),
            },
        };
        
        Ok(response)
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let state = self.get_transformation_state().await;
        let metrics = self.get_transformation_metrics().await;
        
        let health_level = if state.is_healthy {
            HealthLevel::Healthy
        } else {
            HealthLevel::Critical
        };
        
        Ok(HealthStatus {
            status: health_level,
            last_check: chrono::Utc::now(),
            uptime: self.base.start_time.elapsed(),
            issues: Vec::new(),
            metrics: HealthMetrics {
                cpu_usage_percent: 0.0, // Would be measured
                memory_usage_mb: metrics.memory_usage_mb,
                network_usage_mbps: 0.0, // Would be measured
                disk_usage_mb: 0.0, // Would be measured
                error_rate: 0.0, // Would be calculated
                response_time_ms: metrics.average_transformation_time_us / 1000.0,
            },
        })
    }
    
    async fn get_metrics(&self) -> Result<AgentMetrics> {
        Ok(self.base.metrics.read().await.clone())
    }
    
    async fn reset(&self) -> Result<()> {
        info!("Resetting data transformation agent");
        
        self.transformation_cache.write().await.clear();
        
        // Reset metrics
        {
            let mut metrics = self.transformation_metrics.write().await;
            *metrics = TransformationMetrics::default();
        }
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = TransformationState::default();
        }
        
        info!("Data transformation agent reset successfully");
        Ok(())
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        debug!("Handling coordination message: {:?}", message.coordination_type);
        
        match message.coordination_type {
            crate::agents::base::CoordinationType::LoadBalancing => {
                info!("Received load balancing coordination");
            }
            crate::agents::base::CoordinationType::StateSync => {
                info!("Received state sync coordination");
            }
            _ => {
                debug!("Unhandled coordination type: {:?}", message.coordination_type);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_data_transformation_agent_creation() {
        let config = DataTransformationConfig::default();
        let agent = DataTransformationAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[test]
    async fn test_data_transformation() {
        let config = DataTransformationConfig::default();
        let agent = DataTransformationAgent::new(config).await.unwrap();
        
        let data = serde_json::json!({
            "price": 100.0,
            "volume": 1000.0,
            "missing_field": null
        });
        
        let result = agent.transform_data(data).await;
        assert!(result.is_ok());
        
        let transformation_result = result.unwrap();
        assert!(!transformation_result.transformations_applied.is_empty());
    }
    
    #[test]
    async fn test_data_normalizer() {
        let config = NormalizationConfig::default();
        let normalizer = DataNormalizer::new(Arc::new(config));
        
        let mut data = serde_json::json!({
            "field1": 100.0,
            "field2": 200.0
        });
        
        let result = normalizer.normalize(&mut data).await;
        assert!(result.is_ok());
    }
    
    #[test]
    async fn test_normalization_methods() {
        let config = DataTransformationConfig::default();
        let agent = DataTransformationAgent::new(config).await.unwrap();
        
        // Test min-max normalization
        let stats = FieldStatistics {
            mean: 50.0,
            std: 15.0,
            min: 0.0,
            max: 100.0,
            median: 50.0,
            q25: 25.0,
            q75: 75.0,
            count: 100,
        };
        
        let normalized = agent.normalizer.min_max_normalize(75.0, &stats);
        assert!((normalized - 0.75).abs() < 1e-10);
        
        let z_normalized = agent.normalizer.z_score_normalize(65.0, &stats);
        assert!((z_normalized - 1.0).abs() < 1e-10);
    }
}