//! Configuration for the data pipeline

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration for the data pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPipelineConfig {
    /// Streaming configuration
    pub streaming: StreamingConfig,
    /// Sentiment analysis configuration
    pub sentiment: SentimentConfig,
    /// Technical indicators configuration
    pub indicators: IndicatorsConfig,
    /// Data fusion configuration
    pub fusion: FusionConfig,
    /// Data validation configuration
    pub validation: ValidationConfig,
    /// Feature extraction configuration
    pub features: FeatureConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
}

impl Default for DataPipelineConfig {
    fn default() -> Self {
        Self {
            streaming: StreamingConfig::default(),
            sentiment: SentimentConfig::default(),
            indicators: IndicatorsConfig::default(),
            fusion: FusionConfig::default(),
            validation: ValidationConfig::default(),
            features: FeatureConfig::default(),
            monitoring: MonitoringConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Kafka broker addresses
    pub kafka_brokers: Vec<String>,
    /// Consumer group ID
    pub consumer_group: String,
    /// Topic names to consume from
    pub topics: Vec<String>,
    /// Batch size for processing
    pub batch_size: usize,
    /// Flush interval
    pub flush_interval: Duration,
    /// Consumer timeout
    pub consumer_timeout: Duration,
    /// Enable SSL
    pub enable_ssl: bool,
    /// SSL configuration
    pub ssl_config: Option<SslConfig>,
    /// SASL configuration
    pub sasl_config: Option<SaslConfig>,
    /// Compression type
    pub compression: CompressionType,
    /// Auto-offset reset
    pub auto_offset_reset: AutoOffsetReset,
    /// Enable auto-commit
    pub enable_auto_commit: bool,
    /// Auto-commit interval
    pub auto_commit_interval: Duration,
    /// Session timeout
    pub session_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Max poll records
    pub max_poll_records: usize,
    /// Fetch min bytes
    pub fetch_min_bytes: usize,
    /// Fetch max wait
    pub fetch_max_wait: Duration,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            kafka_brokers: vec!["localhost:9092".to_string()],
            consumer_group: "data-pipeline".to_string(),
            topics: vec!["market-data".to_string(), "news-feed".to_string()],
            batch_size: 1000,
            flush_interval: Duration::from_millis(100),
            consumer_timeout: Duration::from_secs(30),
            enable_ssl: false,
            ssl_config: None,
            sasl_config: None,
            compression: CompressionType::Snappy,
            auto_offset_reset: AutoOffsetReset::Latest,
            enable_auto_commit: true,
            auto_commit_interval: Duration::from_secs(5),
            session_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(3),
            max_poll_records: 500,
            fetch_min_bytes: 1024,
            fetch_max_wait: Duration::from_millis(500),
        }
    }
}

/// SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfig {
    pub ca_cert_path: String,
    pub client_cert_path: String,
    pub client_key_path: String,
    pub verify_hostname: bool,
}

/// SASL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaslConfig {
    pub mechanism: SaslMechanism,
    pub username: String,
    pub password: String,
}

/// SASL mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaslMechanism {
    Plain,
    ScramSha256,
    ScramSha512,
    Gssapi,
}

/// Compression type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

/// Auto offset reset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoOffsetReset {
    Earliest,
    Latest,
    None,
}

/// Sentiment analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    /// Model path for sentiment analysis
    pub model_path: String,
    /// Tokenizer path
    pub tokenizer_path: String,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU device ID
    pub gpu_device_id: usize,
    /// Model precision
    pub precision: ModelPrecision,
    /// Cache size for embeddings
    pub cache_size: usize,
    /// Threshold for sentiment classification
    pub sentiment_threshold: f32,
    /// Language models to use
    pub language_models: Vec<String>,
    /// Enable multilingual support
    pub enable_multilingual: bool,
    /// Preprocessing options
    pub preprocessing: PreprocessingConfig,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            model_path: "models/sentiment/bert-base-uncased".to_string(),
            tokenizer_path: "models/sentiment/tokenizer".to_string(),
            max_sequence_length: 512,
            batch_size: 32,
            enable_gpu: true,
            gpu_device_id: 0,
            precision: ModelPrecision::Float16,
            cache_size: 10000,
            sentiment_threshold: 0.5,
            language_models: vec![
                "bert-base-uncased".to_string(),
                "roberta-base".to_string(),
                "distilbert-base-uncased".to_string(),
            ],
            enable_multilingual: false,
            preprocessing: PreprocessingConfig::default(),
        }
    }
}

/// Model precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelPrecision {
    Float32,
    Float16,
    Int8,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Remove stop words
    pub remove_stopwords: bool,
    /// Stem words
    pub stem_words: bool,
    /// Lemmatize words
    pub lemmatize: bool,
    /// Remove URLs
    pub remove_urls: bool,
    /// Remove mentions
    pub remove_mentions: bool,
    /// Remove hashtags
    pub remove_hashtags: bool,
    /// Remove emojis
    pub remove_emojis: bool,
    /// Normalize whitespace
    pub normalize_whitespace: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            remove_stopwords: false,
            stem_words: false,
            lemmatize: false,
            remove_urls: true,
            remove_mentions: true,
            remove_hashtags: false,
            remove_emojis: false,
            normalize_whitespace: true,
        }
    }
}

/// Technical indicators configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorsConfig {
    /// Window sizes for moving averages
    pub ma_windows: Vec<usize>,
    /// RSI period
    pub rsi_period: usize,
    /// MACD configuration
    pub macd_config: MacdConfig,
    /// Bollinger Bands configuration
    pub bollinger_config: BollingerConfig,
    /// Stochastic configuration
    pub stochastic_config: StochasticConfig,
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Parallel processing threads
    pub parallel_threads: usize,
    /// Cache size for indicators
    pub cache_size: usize,
    /// Update frequency
    pub update_frequency: Duration,
    /// Custom indicators
    pub custom_indicators: Vec<CustomIndicatorConfig>,
}

impl Default for IndicatorsConfig {
    fn default() -> Self {
        Self {
            ma_windows: vec![5, 10, 20, 50, 100, 200],
            rsi_period: 14,
            macd_config: MacdConfig::default(),
            bollinger_config: BollingerConfig::default(),
            stochastic_config: StochasticConfig::default(),
            enable_simd: true,
            parallel_threads: num_cpus::get(),
            cache_size: 10000,
            update_frequency: Duration::from_millis(100),
            custom_indicators: vec![],
        }
    }
}

/// MACD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl Default for MacdConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

/// Bollinger Bands configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerConfig {
    pub period: usize,
    pub std_dev: f64,
}

impl Default for BollingerConfig {
    fn default() -> Self {
        Self {
            period: 20,
            std_dev: 2.0,
        }
    }
}

/// Stochastic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticConfig {
    pub k_period: usize,
    pub d_period: usize,
    pub smooth: usize,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
            smooth: 3,
        }
    }
}

/// Custom indicator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIndicatorConfig {
    pub name: String,
    pub formula: String,
    pub parameters: Vec<f64>,
    pub window_size: usize,
}

/// Data fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fusion algorithm
    pub algorithm: FusionAlgorithm,
    /// Weights for different data sources
    pub source_weights: SourceWeights,
    /// Temporal alignment window
    pub alignment_window: Duration,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    /// Outlier detection threshold
    pub outlier_threshold: f64,
    /// Missing data handling
    pub missing_data_strategy: MissingDataStrategy,
    /// Quality score threshold
    pub quality_threshold: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            algorithm: FusionAlgorithm::WeightedAverage,
            source_weights: SourceWeights::default(),
            alignment_window: Duration::from_millis(100),
            interpolation: InterpolationMethod::Linear,
            outlier_threshold: 3.0,
            missing_data_strategy: MissingDataStrategy::Interpolate,
            quality_threshold: 0.8,
        }
    }
}

/// Fusion algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionAlgorithm {
    WeightedAverage,
    KalmanFilter,
    ParticleFilter,
    BayesianFusion,
    NeuralFusion,
}

/// Source weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWeights {
    pub market_data: f64,
    pub news_sentiment: f64,
    pub technical_indicators: f64,
    pub social_media: f64,
    pub economic_data: f64,
}

impl Default for SourceWeights {
    fn default() -> Self {
        Self {
            market_data: 0.4,
            news_sentiment: 0.2,
            technical_indicators: 0.2,
            social_media: 0.1,
            economic_data: 0.1,
        }
    }
}

/// Interpolation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    Cubic,
    Spline,
    Nearest,
}

/// Missing data strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingDataStrategy {
    Drop,
    Interpolate,
    Forward,
    Backward,
    Zero,
    Mean,
}

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable schema validation
    pub enable_schema_validation: bool,
    /// Schema file path
    pub schema_path: String,
    /// Enable range validation
    pub enable_range_validation: bool,
    /// Value ranges
    pub value_ranges: ValueRanges,
    /// Enable duplicate detection
    pub enable_duplicate_detection: bool,
    /// Duplicate detection window
    pub duplicate_window: Duration,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
    /// Enable data quality scoring
    pub enable_quality_scoring: bool,
    /// Quality score components
    pub quality_components: QualityComponents,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_schema_validation: true,
            schema_path: "schemas/market-data.json".to_string(),
            enable_range_validation: true,
            value_ranges: ValueRanges::default(),
            enable_duplicate_detection: true,
            duplicate_window: Duration::from_secs(60),
            enable_anomaly_detection: true,
            anomaly_threshold: 2.0,
            enable_quality_scoring: true,
            quality_components: QualityComponents::default(),
        }
    }
}

/// Value ranges for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueRanges {
    pub price_min: f64,
    pub price_max: f64,
    pub volume_min: f64,
    pub volume_max: f64,
    pub timestamp_tolerance: Duration,
}

impl Default for ValueRanges {
    fn default() -> Self {
        Self {
            price_min: 0.0,
            price_max: 1_000_000.0,
            volume_min: 0.0,
            volume_max: 1_000_000_000.0,
            timestamp_tolerance: Duration::from_secs(60),
        }
    }
}

/// Quality score components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityComponents {
    pub completeness_weight: f64,
    pub accuracy_weight: f64,
    pub consistency_weight: f64,
    pub timeliness_weight: f64,
    pub validity_weight: f64,
}

impl Default for QualityComponents {
    fn default() -> Self {
        Self {
            completeness_weight: 0.2,
            accuracy_weight: 0.3,
            consistency_weight: 0.2,
            timeliness_weight: 0.15,
            validity_weight: 0.15,
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enable statistical features
    pub enable_statistical: bool,
    /// Statistical window sizes
    pub statistical_windows: Vec<usize>,
    /// Enable frequency domain features
    pub enable_frequency: bool,
    /// FFT window size
    pub fft_window_size: usize,
    /// Enable wavelet features
    pub enable_wavelet: bool,
    /// Wavelet type
    pub wavelet_type: WaveletType,
    /// Enable time series features
    pub enable_time_series: bool,
    /// Lag features
    pub lag_features: Vec<usize>,
    /// Enable polynomial features
    pub enable_polynomial: bool,
    /// Polynomial degree
    pub polynomial_degree: usize,
    /// Enable interaction features
    pub enable_interactions: bool,
    /// Feature scaling method
    pub scaling_method: ScalingMethod,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Number of features to select
    pub num_features: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            enable_statistical: true,
            statistical_windows: vec![5, 10, 20, 50],
            enable_frequency: true,
            fft_window_size: 256,
            enable_wavelet: true,
            wavelet_type: WaveletType::Daubechies,
            enable_time_series: true,
            lag_features: vec![1, 2, 3, 5, 10],
            enable_polynomial: false,
            polynomial_degree: 2,
            enable_interactions: false,
            scaling_method: ScalingMethod::StandardScaler,
            selection_method: FeatureSelectionMethod::MutualInformation,
            num_features: 100,
        }
    }
}

/// Wavelet type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaveletType {
    Daubechies,
    Haar,
    Biorthogonal,
    Coiflets,
    Symlets,
}

/// Feature scaling method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    None,
}

/// Feature selection method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    MutualInformation,
    ChiSquare,
    FScore,
    VarianceThreshold,
    Lasso,
    Ridge,
    ElasticNet,
    None,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enable logging
    pub enable_logging: bool,
    /// Log level
    pub log_level: LogLevel,
    /// Enable tracing
    pub enable_tracing: bool,
    /// Tracing configuration
    pub tracing_config: TracingConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval: Duration::from_secs(10),
            enable_health_checks: true,
            health_check_interval: Duration::from_secs(30),
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
            enable_logging: true,
            log_level: LogLevel::Info,
            enable_tracing: false,
            tracing_config: TracingConfig::default(),
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub error_rate_threshold: f64,
    pub latency_threshold_ms: f64,
    pub throughput_threshold: f64,
    pub memory_threshold_mb: f64,
    pub cpu_threshold_percent: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            error_rate_threshold: 0.05,
            latency_threshold_ms: 100.0,
            throughput_threshold: 1000.0,
            memory_threshold_mb: 1024.0,
            cpu_threshold_percent: 80.0,
        }
    }
}

/// Log level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub jaeger_endpoint: String,
    pub service_name: String,
    pub sample_rate: f64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            jaeger_endpoint: "http://localhost:14268/api/traces".to_string(),
            service_name: "data-pipeline".to_string(),
            sample_rate: 0.1,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Thread stack size
    pub thread_stack_size: usize,
    /// Enable NUMA awareness
    pub enable_numa: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable memory mapping
    pub enable_memory_mapping: bool,
    /// Enable lock-free data structures
    pub enable_lock_free: bool,
    /// Batch processing size
    pub batch_size: usize,
    /// Enable prefetching
    pub enable_prefetching: bool,
    /// Prefetch distance
    pub prefetch_distance: usize,
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
    /// CPU affinity mask
    pub cpu_affinity_mask: Vec<usize>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            enable_work_stealing: true,
            thread_stack_size: 8 * 1024 * 1024, // 8MB
            enable_numa: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_memory_mapping: true,
            enable_lock_free: true,
            batch_size: 1000,
            enable_prefetching: true,
            prefetch_distance: 64,
            enable_cpu_affinity: false,
            cpu_affinity_mask: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DataPipelineConfig::default();
        assert_eq!(config.streaming.batch_size, 1000);
        assert_eq!(config.sentiment.max_sequence_length, 512);
        assert_eq!(config.indicators.rsi_period, 14);
    }

    #[test]
    fn test_config_serialization() {
        let config = DataPipelineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DataPipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.streaming.batch_size, deserialized.streaming.batch_size);
    }
}