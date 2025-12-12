//! Common types and data structures for the unified CDFA library

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Standard floating point type used throughout the library
pub type Float = f64;

/// Standard array type for 1D data
pub type FloatArray1 = ndarray::Array1<Float>;

/// Standard array type for 2D data
pub type FloatArray2 = ndarray::Array2<Float>;

/// Standard array view type for 1D data
pub type FloatArrayView1<'a> = ndarray::ArrayView1<'a, Float>;

/// Standard array view type for 2D data
pub type FloatArrayView2<'a> = ndarray::ArrayView2<'a, Float>;

// Re-export ndarray types for convenience
pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Index type for array indexing
pub type Index = usize;

/// Time stamp type
pub type Timestamp = i64;

/// Configuration for CDFA operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CdfaConfig {
    /// Number of parallel threads to use (0 = auto-detect)
    pub num_threads: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Numerical tolerance for comparisons
    pub tolerance: Float,
    
    /// Maximum number of iterations for iterative algorithms
    pub max_iterations: usize,
    
    /// Convergence threshold
    pub convergence_threshold: Float,
    
    /// Cache size for memoization (in MB)
    pub cache_size_mb: usize,
    
    /// Enable distributed processing
    pub enable_distributed: bool,
    
    /// Default diversity method to use
    pub diversity_method: Option<String>,
    
    /// Default fusion method to use
    pub fusion_method: Option<String>,
    
    /// List of enabled detectors (None = all enabled)
    pub enabled_detectors: Option<Vec<String>>,
    
    /// List of enabled analyzers (None = all enabled)
    pub enabled_analyzers: Option<Vec<String>>,
    
    /// List of enabled algorithms (None = all enabled)
    pub enabled_algorithms: Option<Vec<String>>,
}

impl Default for CdfaConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            enable_simd: true,
            enable_gpu: false,
            tolerance: 1e-10,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            cache_size_mb: 100,
            enable_distributed: false,
            diversity_method: None,
            fusion_method: None,
            enabled_detectors: None,
            enabled_analyzers: None,
            enabled_algorithms: None,
        }
    }
}

/// Performance metrics for CDFA operations
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    /// Execution time in microseconds
    pub execution_time_us: u64,
    
    /// Memory used in bytes
    pub memory_used_bytes: usize,
    
    /// Number of SIMD operations performed
    pub simd_operations: usize,
    
    /// Number of parallel tasks executed
    pub parallel_tasks: usize,
    
    /// GPU memory used in bytes
    #[cfg(feature = "gpu")]
    pub gpu_memory_bytes: usize,
    
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: Float,
    
    /// Number of iterations for convergent algorithms
    pub iterations: usize,
    
    /// Final convergence error
    pub convergence_error: Float,
}

/// Data quality metrics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataQuality {
    /// Percentage of missing values
    pub missing_percentage: Float,
    
    /// Number of outliers detected
    pub outlier_count: usize,
    
    /// Signal-to-noise ratio
    pub snr_db: Float,
    
    /// Data stationarity p-value
    pub stationarity_pvalue: Float,
    
    /// Skewness of the data
    pub skewness: Float,
    
    /// Kurtosis of the data
    pub kurtosis: Float,
    
    /// Autocorrelation at lag 1
    pub autocorrelation_lag1: Float,
}

/// Pattern information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pattern {
    /// Pattern type identifier
    pub pattern_type: String,
    
    /// Start index in the data
    pub start_index: Index,
    
    /// End index in the data
    pub end_index: Index,
    
    /// Confidence score (0.0 - 1.0)
    pub confidence: Float,
    
    /// Pattern-specific parameters
    pub parameters: std::collections::HashMap<String, Float>,
    
    /// Timestamp when pattern was detected
    pub timestamp: Timestamp,
}

/// Market data point
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MarketDataPoint {
    /// Timestamp
    pub timestamp: Timestamp,
    
    /// Open price
    pub open: Float,
    
    /// High price
    pub high: Float,
    
    /// Low price
    pub low: Float,
    
    /// Close price
    pub close: Float,
    
    /// Volume
    pub volume: Float,
    
    /// Additional features (e.g., technical indicators)
    pub features: std::collections::HashMap<String, Float>,
}

/// Time series data structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeSeries {
    /// Time stamps
    pub timestamps: Vec<Timestamp>,
    
    /// Values
    pub values: FloatArray1,
    
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(timestamps: Vec<Timestamp>, values: FloatArray1) -> Self {
        Self {
            timestamps,
            values,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Get length of the time series
    pub fn len(&self) -> usize {
        self.values.len()
    }
    
    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    
    /// Get a slice of the time series
    pub fn slice(&self, start: usize, end: usize) -> TimeSeries {
        TimeSeries {
            timestamps: self.timestamps[start..end].to_vec(),
            values: self.values.slice(ndarray::s![start..end]).to_owned(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Multi-dimensional time series
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiTimeSeries {
    /// Time stamps
    pub timestamps: Vec<Timestamp>,
    
    /// Values (time x features)
    pub values: FloatArray2,
    
    /// Feature names
    pub feature_names: Vec<String>,
    
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl MultiTimeSeries {
    /// Create a new multi-dimensional time series
    pub fn new(timestamps: Vec<Timestamp>, values: FloatArray2, feature_names: Vec<String>) -> Self {
        Self {
            timestamps,
            values,
            feature_names,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Get number of time points
    pub fn len(&self) -> usize {
        self.values.nrows()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    
    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.values.ncols()
    }
    
    /// Get a specific feature as a time series
    pub fn get_feature(&self, feature_index: usize) -> Option<TimeSeries> {
        if feature_index >= self.num_features() {
            return None;
        }
        
        Some(TimeSeries::new(
            self.timestamps.clone(),
            self.values.column(feature_index).to_owned(),
        ))
    }
}

/// Algorithm parameters for various CDFA algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlgorithmParams {
    /// General parameters
    pub general: std::collections::HashMap<String, Float>,
    
    /// Algorithm-specific parameters
    pub specific: std::collections::HashMap<String, std::collections::HashMap<String, Float>>,
    
    /// Boolean flags
    pub flags: std::collections::HashMap<String, bool>,
    
    /// String parameters
    pub strings: std::collections::HashMap<String, String>,
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        Self {
            general: std::collections::HashMap::new(),
            specific: std::collections::HashMap::new(),
            flags: std::collections::HashMap::new(),
            strings: std::collections::HashMap::new(),
        }
    }
}

impl AlgorithmParams {
    /// Get a general parameter
    pub fn get_float(&self, key: &str) -> Option<Float> {
        self.general.get(key).copied()
    }
    
    /// Set a general parameter
    pub fn set_float(&mut self, key: String, value: Float) {
        self.general.insert(key, value);
    }
    
    /// Get an algorithm-specific parameter
    pub fn get_algorithm_float(&self, algorithm: &str, key: &str) -> Option<Float> {
        self.specific.get(algorithm)?.get(key).copied()
    }
    
    /// Set an algorithm-specific parameter
    pub fn set_algorithm_float(&mut self, algorithm: String, key: String, value: Float) {
        self.specific.entry(algorithm).or_default().insert(key, value);
    }
    
    /// Get a boolean flag
    pub fn get_flag(&self, key: &str) -> Option<bool> {
        self.flags.get(key).copied()
    }
    
    /// Set a boolean flag
    pub fn set_flag(&mut self, key: String, value: bool) {
        self.flags.insert(key, value);
    }
    
    /// Get a string parameter
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(|s| s.as_str())
    }
    
    /// Set a string parameter
    pub fn set_string(&mut self, key: String, value: String) {
        self.strings.insert(key, value);
    }
}

/// Result of a CDFA analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalysisResult {
    /// Primary result data
    pub data: FloatArray1,
    
    /// Secondary results (e.g., intermediate calculations)
    pub secondary_data: std::collections::HashMap<String, FloatArray1>,
    
    /// Scalar metrics
    pub metrics: std::collections::HashMap<String, Float>,
    
    /// Patterns detected
    pub patterns: Vec<Pattern>,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Data quality assessment
    pub data_quality: DataQuality,
    
    /// Configuration used
    pub config: CdfaConfig,
    
    /// Timestamp of analysis
    pub timestamp: Timestamp,
}

impl AnalysisResult {
    /// Create a new analysis result
    pub fn new(data: FloatArray1, config: CdfaConfig) -> Self {
        Self {
            data,
            secondary_data: std::collections::HashMap::new(),
            metrics: std::collections::HashMap::new(),
            patterns: Vec::new(),
            performance: PerformanceMetrics::default(),
            data_quality: DataQuality::default(),
            config,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
    
    /// Add a metric
    pub fn add_metric(&mut self, name: String, value: Float) {
        self.metrics.insert(name, value);
    }
    
    /// Add secondary data
    pub fn add_secondary_data(&mut self, name: String, data: FloatArray1) {
        self.secondary_data.insert(name, data);
    }
    
    /// Add a detected pattern
    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.patterns.push(pattern);
    }
    
    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<Float> {
        self.metrics.get(name).copied()
    }
    
    /// Get secondary data by name
    pub fn get_secondary_data(&self, name: &str) -> Option<&FloatArray1> {
        self.secondary_data.get(name)
    }
}

/// Batch processing result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BatchResult {
    /// Individual results
    pub results: Vec<AnalysisResult>,
    
    /// Aggregate metrics
    pub aggregate_metrics: std::collections::HashMap<String, Float>,
    
    /// Total processing time
    pub total_time_ms: u64,
    
    /// Number of successful analyses
    pub successful_count: usize,
    
    /// Number of failed analyses
    pub failed_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_time_series() {
        let timestamps = vec![1000, 2000, 3000, 4000];
        let values = array![1.0, 2.0, 3.0, 4.0];
        
        let ts = TimeSeries::new(timestamps, values);
        assert_eq!(ts.len(), 4);
        assert!(!ts.is_empty());
        
        let slice = ts.slice(1, 3);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice.values[0], 2.0);
        assert_eq!(slice.values[1], 3.0);
    }
    
    #[test]
    fn test_multi_time_series() {
        let timestamps = vec![1000, 2000, 3000];
        let values = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];
        
        let mts = MultiTimeSeries::new(timestamps, values, feature_names);
        assert_eq!(mts.len(), 3);
        assert_eq!(mts.num_features(), 2);
        
        let feature = mts.get_feature(0).unwrap();
        assert_eq!(feature.values[0], 1.0);
        assert_eq!(feature.values[1], 3.0);
        assert_eq!(feature.values[2], 5.0);
    }
    
    #[test]
    fn test_algorithm_params() {
        let mut params = AlgorithmParams::default();
        
        params.set_float("tolerance".to_string(), 1e-6);
        params.set_flag("enable_cache".to_string(), true);
        params.set_string("method".to_string(), "fast".to_string());
        
        assert_eq!(params.get_float("tolerance"), Some(1e-6));
        assert_eq!(params.get_flag("enable_cache"), Some(true));
        assert_eq!(params.get_string("method"), Some("fast"));
        
        params.set_algorithm_float("wavelet".to_string(), "levels".to_string(), 4.0);
        assert_eq!(params.get_algorithm_float("wavelet", "levels"), Some(4.0));
    }
    
    #[test]
    fn test_analysis_result() {
        let data = array![1.0, 2.0, 3.0];
        let config = CdfaConfig::default();
        let mut result = AnalysisResult::new(data, config);
        
        result.add_metric("correlation".to_string(), 0.95);
        result.add_secondary_data("residuals".to_string(), array![0.1, -0.1, 0.05]);
        
        assert_eq!(result.get_metric("correlation"), Some(0.95));
        assert!(result.get_secondary_data("residuals").is_some());
    }
}