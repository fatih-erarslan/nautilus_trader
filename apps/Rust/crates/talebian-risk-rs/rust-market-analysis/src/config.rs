//! Configuration management for the market analysis engine

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration for the market analysis engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Whale analysis configuration
    pub whale_analysis: WhaleAnalysisConfig,
    
    /// Regime detection configuration
    pub regime_detection: RegimeDetectionConfig,
    
    /// Pattern recognition configuration
    pub pattern_recognition: PatternRecognitionConfig,
    
    /// Predictive modeling configuration
    pub predictive_models: PredictiveModelsConfig,
    
    /// Market microstructure configuration
    pub microstructure: MicrostructureConfig,
    
    /// Performance and caching configuration
    pub performance: PerformanceConfig,
    
    /// Data processing configuration
    pub data_processing: DataProcessingConfig,
    
    /// Alert and notification configuration
    pub alerts: AlertConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            whale_analysis: WhaleAnalysisConfig::default(),
            regime_detection: RegimeDetectionConfig::default(),
            pattern_recognition: PatternRecognitionConfig::default(),
            predictive_models: PredictiveModelsConfig::default(),
            microstructure: MicrostructureConfig::default(),
            performance: PerformanceConfig::default(),
            data_processing: DataProcessingConfig::default(),
            alerts: AlertConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAnalysisConfig {
    /// Minimum BTC equivalent for whale classification
    pub whale_threshold_btc: f64,
    
    /// Minimum volume spike multiplier for detection
    pub volume_spike_threshold: f64,
    
    /// Order flow imbalance threshold (0.0 to 1.0)
    pub imbalance_threshold: f64,
    
    /// Minimum confidence for whale signal emission
    pub confidence_threshold: f64,
    
    /// Number of historical periods for pattern analysis
    pub lookback_periods: usize,
    
    /// Exponential smoothing factor for indicators
    pub smoothing_factor: f64,
    
    /// Detection sensitivity (0.0 = conservative, 1.0 = aggressive)
    pub detection_sensitivity: f64,
    
    /// Volume profile analysis settings
    pub volume_profile: VolumeProfileConfig,
    
    /// Order flow analysis settings
    pub order_flow: OrderFlowConfig,
}

impl Default for WhaleAnalysisConfig {
    fn default() -> Self {
        Self {
            whale_threshold_btc: 10.0,
            volume_spike_threshold: 2.5,
            imbalance_threshold: 0.7,
            confidence_threshold: 0.8,
            lookback_periods: 100,
            smoothing_factor: 0.3,
            detection_sensitivity: 0.8,
            volume_profile: VolumeProfileConfig::default(),
            order_flow: OrderFlowConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfileConfig {
    /// Number of price levels for volume distribution
    pub price_levels: usize,
    
    /// Value area percentage (typically 70%)
    pub value_area_percentage: f64,
    
    /// Minimum volume for significant level
    pub min_significant_volume: f64,
    
    /// Time period for volume profile calculation (minutes)
    pub profile_period_minutes: u32,
}

impl Default for VolumeProfileConfig {
    fn default() -> Self {
        Self {
            price_levels: 100,
            value_area_percentage: 0.70,
            min_significant_volume: 1000.0,
            profile_period_minutes: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowConfig {
    /// Minimum trade size for order flow analysis
    pub min_trade_size: f64,
    
    /// Time window for imbalance calculation (seconds)
    pub imbalance_window_seconds: u32,
    
    /// Aggressive trade detection threshold
    pub aggressive_trade_threshold: f64,
    
    /// Order size classification thresholds (BTC equivalent)
    pub order_size_thresholds: OrderSizeThresholds,
}

impl Default for OrderFlowConfig {
    fn default() -> Self {
        Self {
            min_trade_size: 0.001,
            imbalance_window_seconds: 60,
            aggressive_trade_threshold: 0.8,
            order_size_thresholds: OrderSizeThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSizeThresholds {
    pub small_order_max: f64,    // < 1 BTC
    pub medium_order_max: f64,   // < 10 BTC
    pub large_order_max: f64,    // < 100 BTC
    // whale_order: > large_order_max
}

impl Default for OrderSizeThresholds {
    fn default() -> Self {
        Self {
            small_order_max: 1.0,
            medium_order_max: 10.0,
            large_order_max: 100.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionConfig {
    /// Number of data points required for regime detection
    pub min_data_points: usize,
    
    /// Lookback window for feature extraction
    pub feature_lookback_window: usize,
    
    /// Number of features to extract
    pub feature_count: usize,
    
    /// Minimum confidence for regime classification
    pub confidence_threshold: f64,
    
    /// Number of consecutive periods for regime stability
    pub regime_stability_threshold: usize,
    
    /// Volatility calculation windows
    pub volatility_windows: Vec<usize>,
    
    /// Trend detection threshold
    pub trend_threshold: f64,
    
    /// Volume threshold multiplier for regime changes
    pub volume_threshold_multiplier: f64,
    
    /// Machine learning model settings
    pub ml_models: MLModelConfig,
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            min_data_points: 50,
            feature_lookback_window: 200,
            feature_count: 25,
            confidence_threshold: 0.75,
            regime_stability_threshold: 5,
            volatility_windows: vec![10, 20, 50, 100],
            trend_threshold: 0.02,
            volume_threshold_multiplier: 1.5,
            ml_models: MLModelConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    /// Random forest parameters
    pub random_forest: RandomForestConfig,
    
    /// Logistic regression parameters
    pub logistic_regression: LogisticRegressionConfig,
    
    /// Neural network parameters
    pub neural_network: NeuralNetworkConfig,
    
    /// Ensemble method weights
    pub ensemble_weights: EnsembleWeights,
}

impl Default for MLModelConfig {
    fn default() -> Self {
        Self {
            random_forest: RandomForestConfig::default(),
            logistic_regression: LogisticRegressionConfig::default(),
            neural_network: NeuralNetworkConfig::default(),
            ensemble_weights: EnsembleWeights::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: Some(10),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegressionConfig {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub regularization: f64,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-6,
            regularization: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    pub hidden_layers: Vec<usize>,
    pub activation_function: String,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub dropout_rate: f64,
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 32, 16],
            activation_function: "relu".to_string(),
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            dropout_rate: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleWeights {
    pub random_forest: f64,
    pub logistic_regression: f64,
    pub neural_network: f64,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        Self {
            random_forest: 0.4,
            logistic_regression: 0.3,
            neural_network: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionConfig {
    /// Minimum pattern duration (number of periods)
    pub min_pattern_duration: usize,
    
    /// Maximum pattern duration (number of periods)
    pub max_pattern_duration: usize,
    
    /// Confidence threshold for pattern detection
    pub confidence_threshold: f64,
    
    /// Volume confirmation requirement
    pub require_volume_confirmation: bool,
    
    /// Pattern validation settings
    pub validation: PatternValidationConfig,
    
    /// Support and resistance detection
    pub support_resistance: SupportResistanceConfig,
    
    /// Technical pattern settings
    pub technical_patterns: TechnicalPatternConfig,
}

impl Default for PatternRecognitionConfig {
    fn default() -> Self {
        Self {
            min_pattern_duration: 3,
            max_pattern_duration: 50,
            confidence_threshold: 0.7,
            require_volume_confirmation: true,
            validation: PatternValidationConfig::default(),
            support_resistance: SupportResistanceConfig::default(),
            technical_patterns: TechnicalPatternConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternValidationConfig {
    /// Minimum price movement for pattern validity (percentage)
    pub min_price_movement: f64,
    
    /// Maximum noise tolerance (percentage)
    pub max_noise_tolerance: f64,
    
    /// Volume confirmation threshold
    pub volume_confirmation_threshold: f64,
    
    /// Time-based pattern expiry (minutes)
    pub pattern_expiry_minutes: u32,
}

impl Default for PatternValidationConfig {
    fn default() -> Self {
        Self {
            min_price_movement: 0.01, // 1%
            max_noise_tolerance: 0.02, // 2%
            volume_confirmation_threshold: 1.2,
            pattern_expiry_minutes: 240, // 4 hours
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportResistanceConfig {
    /// Minimum touch count for level validation
    pub min_touch_count: usize,
    
    /// Price tolerance for level matching (percentage)
    pub price_tolerance: f64,
    
    /// Time window for level validation (minutes)
    pub validation_window_minutes: u32,
    
    /// Strength calculation method
    pub strength_method: String, // "touch_count", "volume_weighted", "time_weighted"
}

impl Default for SupportResistanceConfig {
    fn default() -> Self {
        Self {
            min_touch_count: 2,
            price_tolerance: 0.005, // 0.5%
            validation_window_minutes: 1440, // 24 hours
            strength_method: "volume_weighted".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalPatternConfig {
    /// Head and shoulders pattern settings
    pub head_shoulders: HeadShouldersConfig,
    
    /// Double top/bottom pattern settings
    pub double_patterns: DoublePatternsConfig,
    
    /// Triangle pattern settings
    pub triangles: TriangleConfig,
    
    /// Flag and pennant settings
    pub flags_pennants: FlagsPennantsConfig,
}

impl Default for TechnicalPatternConfig {
    fn default() -> Self {
        Self {
            head_shoulders: HeadShouldersConfig::default(),
            double_patterns: DoublePatternsConfig::default(),
            triangles: TriangleConfig::default(),
            flags_pennants: FlagsPennantsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadShouldersConfig {
    pub shoulder_height_tolerance: f64,
    pub neckline_slope_tolerance: f64,
    pub volume_decline_requirement: bool,
}

impl Default for HeadShouldersConfig {
    fn default() -> Self {
        Self {
            shoulder_height_tolerance: 0.05, // 5%
            neckline_slope_tolerance: 0.02,  // 2%
            volume_decline_requirement: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoublePatternsConfig {
    pub peak_height_tolerance: f64,
    pub valley_depth_tolerance: f64,
    pub time_separation_min_periods: usize,
    pub time_separation_max_periods: usize,
}

impl Default for DoublePatternsConfig {
    fn default() -> Self {
        Self {
            peak_height_tolerance: 0.03, // 3%
            valley_depth_tolerance: 0.03, // 3%
            time_separation_min_periods: 5,
            time_separation_max_periods: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleConfig {
    pub convergence_tolerance: f64,
    pub min_touch_points: usize,
    pub breakout_volume_multiplier: f64,
}

impl Default for TriangleConfig {
    fn default() -> Self {
        Self {
            convergence_tolerance: 0.01, // 1%
            min_touch_points: 4,
            breakout_volume_multiplier: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagsPennantsConfig {
    pub pole_min_height: f64,
    pub consolidation_max_duration: usize,
    pub consolidation_slope_tolerance: f64,
}

impl Default for FlagsPennantsConfig {
    fn default() -> Self {
        Self {
            pole_min_height: 0.05, // 5%
            consolidation_max_duration: 20,
            consolidation_slope_tolerance: 0.01, // 1%
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveModelsConfig {
    /// Time series forecasting settings
    pub time_series: TimeSeriesConfig,
    
    /// Neural network prediction settings
    pub neural_prediction: NeuralPredictionConfig,
    
    /// Ensemble prediction settings
    pub ensemble_prediction: EnsemblePredictionConfig,
    
    /// Prediction horizons (in minutes)
    pub prediction_horizons: Vec<u32>,
    
    /// Model validation settings
    pub validation: ModelValidationConfig,
}

impl Default for PredictiveModelsConfig {
    fn default() -> Self {
        Self {
            time_series: TimeSeriesConfig::default(),
            neural_prediction: NeuralPredictionConfig::default(),
            ensemble_prediction: EnsemblePredictionConfig::default(),
            prediction_horizons: vec![1, 5, 15, 30, 60],
            validation: ModelValidationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    pub arima_order: (usize, usize, usize), // (p, d, q)
    pub seasonal_periods: Option<usize>,
    pub max_forecast_periods: usize,
    pub confidence_intervals: Vec<f64>,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            arima_order: (1, 1, 1),
            seasonal_periods: Some(24), // 24 hours for daily seasonality
            max_forecast_periods: 60,
            confidence_intervals: vec![0.80, 0.95],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPredictionConfig {
    pub lstm_layers: Vec<usize>,
    pub sequence_length: usize,
    pub prediction_steps: usize,
    pub feature_dimensions: usize,
    pub training_epochs: usize,
    pub batch_size: usize,
}

impl Default for NeuralPredictionConfig {
    fn default() -> Self {
        Self {
            lstm_layers: vec![128, 64, 32],
            sequence_length: 60,
            prediction_steps: 10,
            feature_dimensions: 20,
            training_epochs: 200,
            batch_size: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePredictionConfig {
    pub model_weights: PredictionModelWeights,
    pub voting_method: String, // "average", "weighted", "stacking"
    pub meta_learner: Option<String>, // For stacking
}

impl Default for EnsemblePredictionConfig {
    fn default() -> Self {
        Self {
            model_weights: PredictionModelWeights::default(),
            voting_method: "weighted".to_string(),
            meta_learner: Some("linear_regression".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModelWeights {
    pub arima: f64,
    pub lstm: f64,
    pub garch: f64,
    pub linear_regression: f64,
}

impl Default for PredictionModelWeights {
    fn default() -> Self {
        Self {
            arima: 0.3,
            lstm: 0.4,
            garch: 0.2,
            linear_regression: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationConfig {
    pub train_test_split: f64,
    pub cross_validation_folds: usize,
    pub validation_metrics: Vec<String>,
    pub early_stopping_patience: usize,
    pub retraining_frequency_hours: u32,
}

impl Default for ModelValidationConfig {
    fn default() -> Self {
        Self {
            train_test_split: 0.8,
            cross_validation_folds: 5,
            validation_metrics: vec![
                "mse".to_string(),
                "mae".to_string(),
                "mape".to_string(),
                "directional_accuracy".to_string(),
            ],
            early_stopping_patience: 10,
            retraining_frequency_hours: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureConfig {
    /// Order book analysis settings
    pub order_book: OrderBookConfig,
    
    /// Trade analysis settings
    pub trade_analysis: TradeAnalysisConfig,
    
    /// Liquidity measurement settings
    pub liquidity: LiquidityConfig,
    
    /// Market efficiency metrics
    pub efficiency: EfficiencyConfig,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            order_book: OrderBookConfig::default(),
            trade_analysis: TradeAnalysisConfig::default(),
            liquidity: LiquidityConfig::default(),
            efficiency: EfficiencyConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookConfig {
    pub max_depth_levels: usize,
    pub depth_aggregation_levels: Vec<f64>,
    pub imbalance_calculation_method: String,
    pub snapshot_frequency_ms: u64,
}

impl Default for OrderBookConfig {
    fn default() -> Self {
        Self {
            max_depth_levels: 20,
            depth_aggregation_levels: vec![1.0, 5.0, 10.0, 50.0, 100.0],
            imbalance_calculation_method: "volume_weighted".to_string(),
            snapshot_frequency_ms: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAnalysisConfig {
    pub classification_methods: Vec<String>,
    pub aggressor_detection: AggressorDetectionConfig,
    pub size_classification: TradeSizeClassificationConfig,
}

impl Default for TradeAnalysisConfig {
    fn default() -> Self {
        Self {
            classification_methods: vec![
                "lee_ready".to_string(),
                "tick_rule".to_string(),
                "quote_rule".to_string(),
            ],
            aggressor_detection: AggressorDetectionConfig::default(),
            size_classification: TradeSizeClassificationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggressorDetectionConfig {
    pub time_tolerance_ms: u64,
    pub price_tolerance: f64,
    pub fallback_method: String,
}

impl Default for AggressorDetectionConfig {
    fn default() -> Self {
        Self {
            time_tolerance_ms: 10,
            price_tolerance: 0.0001,
            fallback_method: "tick_rule".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSizeClassificationConfig {
    pub percentile_thresholds: Vec<f64>,
    pub absolute_thresholds: Vec<f64>,
    pub classification_method: String, // "percentile", "absolute", "adaptive"
}

impl Default for TradeSizeClassificationConfig {
    fn default() -> Self {
        Self {
            percentile_thresholds: vec![0.25, 0.75, 0.95],
            absolute_thresholds: vec![1.0, 10.0, 100.0],
            classification_method: "adaptive".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityConfig {
    pub measurement_methods: Vec<String>,
    pub time_windows: Vec<u32>,
    pub depth_levels: Vec<f64>,
    pub resilience_calculation: ResilienceConfig,
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            measurement_methods: vec![
                "amihud".to_string(),
                "roll".to_string(),
                "effective_spread".to_string(),
                "kyle_lambda".to_string(),
            ],
            time_windows: vec![60, 300, 900, 3600], // 1min, 5min, 15min, 1hour
            depth_levels: vec![1.0, 5.0, 10.0, 25.0, 50.0],
            resilience_calculation: ResilienceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    pub shock_size_percentages: Vec<f64>,
    pub recovery_time_thresholds: Vec<u32>,
    pub measurement_frequency_seconds: u32,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            shock_size_percentages: vec![0.1, 0.5, 1.0, 2.0],
            recovery_time_thresholds: vec![60, 300, 900], // 1min, 5min, 15min
            measurement_frequency_seconds: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyConfig {
    pub tests: Vec<String>,
    pub significance_level: f64,
    pub sample_sizes: Vec<usize>,
    pub frequency_bands: Vec<(f64, f64)>,
}

impl Default for EfficiencyConfig {
    fn default() -> Self {
        Self {
            tests: vec![
                "variance_ratio".to_string(),
                "runs_test".to_string(),
                "ljung_box".to_string(),
                "hurst_exponent".to_string(),
            ],
            significance_level: 0.05,
            sample_sizes: vec![100, 500, 1000],
            frequency_bands: vec![(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Cache settings
    pub cache: CacheConfig,
    
    /// Parallel processing settings
    pub parallel: ParallelConfig,
    
    /// Memory management
    pub memory: MemoryConfig,
    
    /// SIMD optimization settings
    pub simd: SIMDConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cache: CacheConfig::default(),
            parallel: ParallelConfig::default(),
            memory: MemoryConfig::default(),
            simd: SIMDConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_size_mb: usize,
    pub ttl_minutes: u32,
    pub eviction_policy: String, // "lru", "lfu", "ttl"
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size_mb: 512,
            ttl_minutes: 15,
            eviction_policy: "lru".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub thread_pool_size: Option<usize>, // None = auto-detect
    pub chunk_size: usize,
    pub enable_rayon: bool,
    pub enable_tokio: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            thread_pool_size: None,
            chunk_size: 1000,
            enable_rayon: true,
            enable_tokio: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_history_size: usize,
    pub gc_threshold_mb: usize,
    pub streaming_buffer_size: usize,
    pub compression_enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            gc_threshold_mb: 1024,
            streaming_buffer_size: 1000,
            compression_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIMDConfig {
    pub enabled: bool,
    pub instruction_set: String, // "auto", "avx2", "avx512", "neon"
    pub vector_size: usize,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instruction_set: "auto".to_string(),
            vector_size: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessingConfig {
    /// Real-time data ingestion
    pub ingestion: IngestionConfig,
    
    /// Data validation and cleaning
    pub validation: DataValidationConfig,
    
    /// Feature engineering
    pub feature_engineering: FeatureEngineeringConfig,
    
    /// Data storage settings
    pub storage: StorageConfig,
}

impl Default for DataProcessingConfig {
    fn default() -> Self {
        Self {
            ingestion: IngestionConfig::default(),
            validation: DataValidationConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub max_queue_size: usize,
    pub backpressure_threshold: f64,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            flush_interval_ms: 1000,
            max_queue_size: 10000,
            backpressure_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    pub price_range_check: bool,
    pub volume_range_check: bool,
    pub timestamp_validation: bool,
    pub outlier_detection: OutlierDetectionConfig,
}

impl Default for DataValidationConfig {
    fn default() -> Self {
        Self {
            price_range_check: true,
            volume_range_check: true,
            timestamp_validation: true,
            outlier_detection: OutlierDetectionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetectionConfig {
    pub method: String, // "iqr", "zscore", "isolation_forest"
    pub threshold: f64,
    pub window_size: usize,
}

impl Default for OutlierDetectionConfig {
    fn default() -> Self {
        Self {
            method: "iqr".to_string(),
            threshold: 3.0,
            window_size: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub technical_indicators: TechnicalIndicatorConfig,
    pub statistical_features: StatisticalFeatureConfig,
    pub time_features: TimeFeatureConfig,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            technical_indicators: TechnicalIndicatorConfig::default(),
            statistical_features: StatisticalFeatureConfig::default(),
            time_features: TimeFeatureConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorConfig {
    pub moving_averages: Vec<usize>,
    pub rsi_periods: Vec<usize>,
    pub bollinger_periods: Vec<usize>,
    pub macd_config: MACDConfig,
}

impl Default for TechnicalIndicatorConfig {
    fn default() -> Self {
        Self {
            moving_averages: vec![5, 10, 20, 50, 100, 200],
            rsi_periods: vec![14, 21],
            bollinger_periods: vec![20],
            macd_config: MACDConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl Default for MACDConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatureConfig {
    pub rolling_windows: Vec<usize>,
    pub correlation_windows: Vec<usize>,
    pub distribution_moments: bool,
    pub cointegration_tests: bool,
}

impl Default for StatisticalFeatureConfig {
    fn default() -> Self {
        Self {
            rolling_windows: vec![10, 20, 50],
            correlation_windows: vec![20, 50, 100],
            distribution_moments: true,
            cointegration_tests: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFeatureConfig {
    pub hour_of_day: bool,
    pub day_of_week: bool,
    pub month_of_year: bool,
    pub market_session: bool,
    pub holiday_effects: bool,
}

impl Default for TimeFeatureConfig {
    fn default() -> Self {
        Self {
            hour_of_day: true,
            day_of_week: true,
            month_of_year: true,
            market_session: true,
            holiday_effects: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub format: String, // "parquet", "csv", "binary"
    pub compression: String, // "none", "gzip", "lz4", "zstd"
    pub partitioning: PartitioningConfig,
    pub retention_policy: RetentionConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            format: "parquet".to_string(),
            compression: "zstd".to_string(),
            partitioning: PartitioningConfig::default(),
            retention_policy: RetentionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningConfig {
    pub strategy: String, // "date", "symbol", "hybrid"
    pub partition_size_mb: usize,
    pub max_partitions: usize,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            strategy: "hybrid".to_string(),
            partition_size_mb: 100,
            max_partitions: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    pub raw_data_days: u32,
    pub processed_data_days: u32,
    pub aggregated_data_days: u32,
    pub model_data_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            raw_data_days: 30,
            processed_data_days: 90,
            aggregated_data_days: 365,
            model_data_days: 365,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Whale activity alerts
    pub whale_alerts: WhaleAlertConfig,
    
    /// Regime change alerts
    pub regime_alerts: RegimeAlertConfig,
    
    /// Pattern detection alerts
    pub pattern_alerts: PatternAlertConfig,
    
    /// Risk alerts
    pub risk_alerts: RiskAlertConfig,
    
    /// System performance alerts
    pub performance_alerts: PerformanceAlertConfig,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            whale_alerts: WhaleAlertConfig::default(),
            regime_alerts: RegimeAlertConfig::default(),
            pattern_alerts: PatternAlertConfig::default(),
            risk_alerts: RiskAlertConfig::default(),
            performance_alerts: PerformanceAlertConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlertConfig {
    pub enabled: bool,
    pub min_confidence: f64,
    pub min_btc_size: f64,
    pub cooldown_minutes: u32,
}

impl Default for WhaleAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.8,
            min_btc_size: 10.0,
            cooldown_minutes: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAlertConfig {
    pub enabled: bool,
    pub min_confidence: f64,
    pub regime_change_only: bool,
    pub cooldown_minutes: u32,
}

impl Default for RegimeAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.7,
            regime_change_only: true,
            cooldown_minutes: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAlertConfig {
    pub enabled: bool,
    pub min_confidence: f64,
    pub pattern_types: Vec<String>,
    pub breakout_alerts: bool,
}

impl Default for PatternAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.75,
            pattern_types: vec![
                "HeadAndShoulders".to_string(),
                "DoubleTop".to_string(),
                "DoubleBottom".to_string(),
                "Breakout".to_string(),
            ],
            breakout_alerts: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlertConfig {
    pub enabled: bool,
    pub var_threshold: f64,
    pub drawdown_threshold: f64,
    pub volatility_spike_threshold: f64,
}

impl Default for RiskAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            var_threshold: 0.05, // 5% VaR
            drawdown_threshold: 0.10, // 10% drawdown
            volatility_spike_threshold: 2.0, // 2x normal volatility
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlertConfig {
    pub enabled: bool,
    pub latency_threshold_ms: u64,
    pub memory_threshold_mb: usize,
    pub cpu_threshold_percent: f64,
    pub error_rate_threshold: f64,
}

impl Default for PerformanceAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency_threshold_ms: 100,
            memory_threshold_mb: 1024,
            cpu_threshold_percent: 80.0,
            error_rate_threshold: 0.01, // 1% error rate
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = match path.ends_with(".toml") {
            true => toml::from_str(&content)?,
            false => serde_json::from_str(&content)?,
        };
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = match path.ends_with(".toml") {
            true => toml::to_string_pretty(self)?,
            false => serde_json::to_string_pretty(self)?,
        };
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate whale analysis config
        if self.whale_analysis.whale_threshold_btc <= 0.0 {
            return Err("Whale threshold must be positive".to_string());
        }
        
        if self.whale_analysis.confidence_threshold < 0.0 || self.whale_analysis.confidence_threshold > 1.0 {
            return Err("Confidence threshold must be between 0 and 1".to_string());
        }
        
        // Validate regime detection config
        if self.regime_detection.min_data_points == 0 {
            return Err("Minimum data points must be positive".to_string());
        }
        
        // Validate performance config
        if self.performance.cache.max_size_mb == 0 {
            return Err("Cache size must be positive".to_string());
        }
        
        // Validate alert config thresholds
        if self.alerts.whale_alerts.min_confidence < 0.0 || self.alerts.whale_alerts.min_confidence > 1.0 {
            return Err("Whale alert confidence must be between 0 and 1".to_string());
        }
        
        Ok(())
    }
    
    /// Get configuration summary
    pub fn summary(&self) -> String {
        format!(
            "Market Analysis Config:\n\
             - Whale threshold: {} BTC\n\
             - Regime confidence: {}\n\
             - Pattern confidence: {}\n\
             - Cache size: {} MB\n\
             - Parallel threads: {:?}",
            self.whale_analysis.whale_threshold_btc,
            self.regime_detection.confidence_threshold,
            self.pattern_recognition.confidence_threshold,
            self.performance.cache.max_size_mb,
            self.performance.parallel.thread_pool_size
        )
    }
}

// Convenience getters
impl Config {
    pub fn whale_signal_threshold(&self) -> f64 {
        self.whale_analysis.confidence_threshold
    }
    
    pub fn regime_change_threshold(&self) -> f64 {
        self.regime_detection.confidence_threshold
    }
    
    pub fn pattern_signal_threshold(&self) -> f64 {
        self.pattern_recognition.confidence_threshold
    }
    
    pub fn cache_ttl_minutes(&self) -> i64 {
        self.performance.cache.ttl_minutes as i64
    }
    
    pub fn required_data_points(&self) -> usize {
        self.regime_detection.min_data_points
    }
}