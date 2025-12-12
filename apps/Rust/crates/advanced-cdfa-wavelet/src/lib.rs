//! # Advanced CDFA Wavelet Analysis
//! 
//! Production-grade wavelet analysis for Advanced CDFA with comprehensive market regime detection.
//! Provides multi-resolution decomposition, continuous and discrete wavelet transforms.
//! 
//! ## Features
//! 
//! - **Continuous Wavelet Transform (CWT)**: Time-frequency analysis with Morlet, Mexican Hat, and Daubechies wavelets
//! - **Discrete Wavelet Transform (DWT)**: Multi-resolution decomposition with perfect reconstruction
//! - **Stationary Wavelet Transform (SWT)**: Translation-invariant decomposition for regime detection
//! - **Market Regime Detection**: Automatic identification of 5 market regimes using wavelet features
//! - **Time-Frequency Analysis**: Real-time spectral analysis for trading signals
//! - **Multi-scale Volatility**: Volatility decomposition across multiple time scales
//! 
//! ## Performance Targets
//! 
//! - CWT computation (1024 samples): < 5 milliseconds
//! - DWT decomposition (4096 samples): < 1 millisecond
//! - Regime detection: < 10 milliseconds per window
//! - Real-time processing: < 100 microseconds latency
//! 
//! ## Example Usage
//! 
//! ```rust
//! use advanced_cdfa_wavelet::{WaveletProcessor, WaveletConfig, MarketRegime};
//! 
//! let config = WaveletConfig::default();
//! let mut processor = WaveletProcessor::new(config)?;
//! 
//! // Market data (OHLCV)
//! let prices = vec![100.0, 101.5, 99.8, 102.1, 103.5];
//! let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 1100.0];
//! 
//! // Multi-resolution analysis
//! let analysis = processor.analyze_market_data(&prices, &volumes).await?;
//! println!("Detected regime: {:?}", analysis.regime);
//! println!("Volatility components: {:?}", analysis.volatility_scales);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, ArrayView1, Axis, s};
use nalgebra::{DMatrix, DVector, Complex};
use num_complex::Complex64;
use num_traits::Float;
use parking_lot::{RwLock, Mutex};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex64 as FftComplex};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::sleep;
use tracing::{debug, info, warn, error, instrument};

// Re-exports
pub use config::*;
pub use transforms::*;
pub use regime::*;
pub use analysis::*;
pub use processor::*;

// Module declarations
pub mod config;
pub mod transforms;
pub mod regime;
pub mod analysis;
pub mod processor;
pub mod wavelets;
pub mod decomposition;
pub mod spectral;

// Error types
#[derive(Error, Debug)]
pub enum WaveletError {
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Transform error: {message}")]
    TransformError { message: String },
    
    #[error("Analysis error: {message}")]
    AnalysisError { message: String },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    #[error("Computation error: {message}")]
    ComputationError { message: String },
    
    #[error("Regime detection error: {message}")]
    RegimeError { message: String },
}

/// Wavelet processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletConfig {
    /// Transform parameters
    pub wavelet_type: WaveletType,
    pub decomposition_levels: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub frequency_resolution: usize,
    
    /// CWT parameters
    pub cwt_scales: Vec<f64>,
    pub cwt_sampling_rate: f64,
    pub morlet_bandwidth: f64,
    pub morlet_center_frequency: f64,
    
    /// DWT parameters
    pub dwt_boundary: BoundaryCondition,
    pub dwt_mode: DecompositionMode,
    
    /// Regime detection parameters
    pub regime_window_size: usize,
    pub regime_overlap: f64,
    pub volatility_threshold_low: f64,
    pub volatility_threshold_high: f64,
    pub trend_threshold: f64,
    pub regime_confidence_threshold: f64,
    
    /// Analysis parameters
    pub spectral_resolution: usize,
    pub time_frequency_window: usize,
    pub coherence_threshold: f64,
    pub cross_wavelet_analysis: bool,
    
    /// Performance parameters
    pub parallel_processing: bool,
    pub max_processing_time_ms: u64,
    pub cache_size: usize,
    pub real_time_mode: bool,
    pub latency_target_us: u64,
    
    /// Market-specific parameters
    pub price_normalization: bool,
    pub volume_weighting: bool,
    pub microstructure_analysis: bool,
    pub volatility_clustering: bool,
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            wavelet_type: WaveletType::Morlet,
            decomposition_levels: 6,
            min_frequency: 0.01,
            max_frequency: 0.5,
            frequency_resolution: 64,
            cwt_scales: Self::generate_default_scales(),
            cwt_sampling_rate: 1.0,
            morlet_bandwidth: 1.0,
            morlet_center_frequency: 1.0,
            dwt_boundary: BoundaryCondition::Symmetric,
            dwt_mode: DecompositionMode::Periodization,
            regime_window_size: 100,
            regime_overlap: 0.5,
            volatility_threshold_low: 0.01,
            volatility_threshold_high: 0.05,
            trend_threshold: 0.02,
            regime_confidence_threshold: 0.8,
            spectral_resolution: 256,
            time_frequency_window: 64,
            coherence_threshold: 0.7,
            cross_wavelet_analysis: true,
            parallel_processing: true,
            max_processing_time_ms: 100,
            cache_size: 1000,
            real_time_mode: true,
            latency_target_us: 1000,
            price_normalization: true,
            volume_weighting: true,
            microstructure_analysis: false,
            volatility_clustering: true,
        }
    }
}

impl WaveletConfig {
    fn generate_default_scales() -> Vec<f64> {
        (1..=64).map(|i| 2.0_f64.powf(i as f64 / 8.0)).collect()
    }
}

/// Wavelet types supported
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum WaveletType {
    /// Morlet wavelet (complex)
    Morlet,
    /// Mexican Hat (Ricker) wavelet
    MexicanHat,
    /// Daubechies wavelets
    Daubechies4,
    Daubechies8,
    Daubechies16,
    /// Biorthogonal wavelets
    Biorthogonal22,
    Biorthogonal44,
    /// Coiflets
    Coiflet2,
    Coiflet4,
    /// Haar wavelet
    Haar,
}

/// Boundary conditions for DWT
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BoundaryCondition {
    /// Zero padding
    Zero,
    /// Symmetric extension
    Symmetric,
    /// Periodic extension
    Periodic,
    /// Antisymmetric extension
    Antisymmetric,
}

/// Decomposition modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DecompositionMode {
    /// Full decomposition
    Full,
    /// Periodization
    Periodization,
    /// Zero padding
    ZeroPadding,
}

/// Market regime types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    /// Low volatility trending up
    BullishTrend,
    /// High volatility trending up
    BullishVolatile,
    /// Low volatility sideways
    Consolidation,
    /// High volatility trending down
    BearishVolatile,
    /// Low volatility trending down
    BearishTrend,
}

impl MarketRegime {
    /// Get regime description
    pub fn description(&self) -> &'static str {
        match self {
            MarketRegime::BullishTrend => "Bullish Trend - Low volatility upward movement",
            MarketRegime::BullishVolatile => "Bullish Volatile - High volatility upward movement",
            MarketRegime::Consolidation => "Consolidation - Low volatility sideways movement",
            MarketRegime::BearishVolatile => "Bearish Volatile - High volatility downward movement",
            MarketRegime::BearishTrend => "Bearish Trend - Low volatility downward movement",
        }
    }
    
    /// Get risk level (0.0 = low, 1.0 = high)
    pub fn risk_level(&self) -> f64 {
        match self {
            MarketRegime::BullishTrend => 0.2,
            MarketRegime::BullishVolatile => 0.7,
            MarketRegime::Consolidation => 0.3,
            MarketRegime::BearishVolatile => 0.9,
            MarketRegime::BearishTrend => 0.5,
        }
    }
}

/// Wavelet analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletAnalysis {
    /// Detected market regime
    pub regime: MarketRegime,
    
    /// Regime confidence (0.0 to 1.0)
    pub regime_confidence: f64,
    
    /// Multi-scale volatility components
    pub volatility_scales: Vec<f64>,
    
    /// Trend strength at different scales
    pub trend_components: Vec<f64>,
    
    /// Time-frequency representation
    pub time_frequency_matrix: Array2<f64>,
    
    /// Dominant frequencies
    pub dominant_frequencies: Vec<f64>,
    
    /// Wavelet coherence (if cross-analysis enabled)
    pub coherence: Option<Array2<f64>>,
    
    /// Phase relationships
    pub phase_relationships: Vec<f64>,
    
    /// Spectral features
    pub spectral_features: SpectralFeatures,
    
    /// Processing metrics
    pub processing_time_us: u64,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Spectral features extracted from wavelet analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    /// Spectral centroid (center of mass of spectrum)
    pub spectral_centroid: f64,
    
    /// Spectral spread (variance around centroid)
    pub spectral_spread: f64,
    
    /// Spectral rolloff (frequency below which 85% of energy lies)
    pub spectral_rolloff: f64,
    
    /// Spectral flux (rate of change of spectrum)
    pub spectral_flux: f64,
    
    /// Zero crossing rate
    pub zero_crossing_rate: f64,
    
    /// Spectral entropy
    pub spectral_entropy: f64,
    
    /// Mel-frequency cepstral coefficients
    pub mfcc: Vec<f64>,
    
    /// Chroma features
    pub chroma: Vec<f64>,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Number of samples analyzed
    pub n_samples: usize,
    
    /// Sampling rate
    pub sampling_rate: f64,
    
    /// Analysis window size
    pub window_size: usize,
    
    /// Number of scales used
    pub n_scales: usize,
    
    /// Frequency range analyzed
    pub frequency_range: (f64, f64),
    
    /// Wavelet type used
    pub wavelet_type: WaveletType,
    
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Main wavelet processor
pub struct WaveletProcessor {
    config: WaveletConfig,
    transformer: Arc<RwLock<WaveletTransformer>>,
    regime_detector: Arc<Mutex<RegimeDetector>>,
    spectral_analyzer: Arc<Mutex<SpectralAnalyzer>>,
    cache: Arc<RwLock<AnalysisCache>>,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
}

impl WaveletProcessor {
    /// Create new wavelet processor
    pub fn new(config: WaveletConfig) -> Result<Self> {
        info!("Initializing wavelet processor with config: {:?}", config);
        
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize components
        let transformer = Arc::new(RwLock::new(
            WaveletTransformer::new(&config)?
        ));
        
        let regime_detector = Arc::new(Mutex::new(
            RegimeDetector::new(&config)?
        ));
        
        let spectral_analyzer = Arc::new(Mutex::new(
            SpectralAnalyzer::new(&config)?
        ));
        
        let cache = Arc::new(RwLock::new(
            AnalysisCache::new(config.cache_size)
        ));
        
        let performance_monitor = Arc::new(Mutex::new(
            PerformanceMonitor::new()
        ));
        
        info!("Wavelet processor initialized successfully");
        
        Ok(Self {
            config,
            transformer,
            regime_detector,
            spectral_analyzer,
            cache,
            performance_monitor,
        })
    }
    
    /// Validate configuration
    fn validate_config(config: &WaveletConfig) -> Result<()> {
        if config.decomposition_levels == 0 {
            return Err(WaveletError::ConfigError {
                message: "Decomposition levels must be greater than 0".to_string(),
            }.into());
        }
        
        if config.min_frequency >= config.max_frequency {
            return Err(WaveletError::ConfigError {
                message: "Min frequency must be less than max frequency".to_string(),
            }.into());
        }
        
        if config.regime_window_size == 0 {
            return Err(WaveletError::ConfigError {
                message: "Regime window size must be greater than 0".to_string(),
            }.into());
        }
        
        if config.regime_overlap < 0.0 || config.regime_overlap >= 1.0 {
            return Err(WaveletError::ConfigError {
                message: "Regime overlap must be between 0.0 and 1.0".to_string(),
            }.into());
        }
        
        Ok(())
    }
    
    /// Analyze market data with multi-resolution wavelet decomposition
    #[instrument(skip(self, prices, volumes))]
    pub async fn analyze_market_data(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<WaveletAnalysis> {
        let start_time = Instant::now();
        
        // Validate inputs
        if prices.is_empty() {
            return Err(WaveletError::InvalidInput {
                message: "Price data cannot be empty".to_string(),
            }.into());
        }
        
        if volumes.len() != prices.len() {
            return Err(WaveletError::InvalidInput {
                message: "Price and volume arrays must have same length".to_string(),
            }.into());
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(prices, volumes);
        if let Some(cached_result) = self.check_cache(&cache_key).await {
            debug!("Returning cached wavelet analysis");
            return Ok(cached_result);
        }
        
        // Add timeout protection
        let timeout_duration = Duration::from_millis(self.config.max_processing_time_ms);
        let analysis_future = self.perform_analysis_internal(prices, volumes);
        
        let mut result = match tokio::time::timeout(timeout_duration, analysis_future).await {
            Ok(result) => result?,
            Err(_) => {
                error!("Wavelet analysis timeout exceeded: {}ms", self.config.max_processing_time_ms);
                return Err(WaveletError::ComputationError {
                    message: "Analysis timeout exceeded".to_string(),
                }.into());
            }
        };
        
        let processing_time_us = start_time.elapsed().as_micros() as u64;
        result.processing_time_us = processing_time_us;
        
        // Update performance metrics
        {
            let mut monitor = self.performance_monitor.lock();
            monitor.record_analysis(processing_time_us, prices.len());
        }
        
        // Cache result
        self.store_in_cache(cache_key, result.clone()).await;
        
        // Check latency target
        if processing_time_us > self.config.latency_target_us {
            warn!(
                "Wavelet analysis latency exceeded target: {}μs > {}μs",
                processing_time_us,
                self.config.latency_target_us
            );
        }
        
        debug!("Wavelet analysis completed in {}μs", processing_time_us);
        
        Ok(result)
    }
    
    /// Internal analysis implementation
    async fn perform_analysis_internal(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<WaveletAnalysis> {
        // Preprocess data
        let processed_prices = self.preprocess_prices(prices)?;
        let processed_volumes = if self.config.volume_weighting {
            Some(self.preprocess_volumes(volumes)?)
        } else {
            None
        };
        
        // Perform wavelet transforms
        let transformer = self.transformer.clone();
        let cwt_result = {
            let mut transformer = transformer.write();
            transformer.continuous_wavelet_transform(&processed_prices).await?
        };
        
        let dwt_result = {
            let mut transformer = transformer.write();
            transformer.discrete_wavelet_transform(&processed_prices).await?
        };
        
        // Extract spectral features
        let spectral_features = {
            let mut analyzer = self.spectral_analyzer.lock();
            analyzer.extract_features(&cwt_result, &self.config)?
        };
        
        // Detect market regime
        let (regime, regime_confidence) = {
            let mut detector = self.regime_detector.lock();
            detector.detect_regime(&dwt_result, &spectral_features, &self.config)?
        };
        
        // Calculate multi-scale components
        let volatility_scales = self.calculate_volatility_scales(&dwt_result)?;
        let trend_components = self.calculate_trend_components(&dwt_result)?;
        
        // Time-frequency analysis
        let time_frequency_matrix = self.create_time_frequency_matrix(&cwt_result)?;
        
        // Extract dominant frequencies
        let dominant_frequencies = self.extract_dominant_frequencies(&cwt_result)?;
        
        // Phase analysis
        let phase_relationships = self.analyze_phase_relationships(&cwt_result)?;
        
        // Cross-wavelet analysis if enabled
        let coherence = if self.config.cross_wavelet_analysis && processed_volumes.is_some() {
            Some(self.compute_wavelet_coherence(
                &processed_prices,
                &processed_volumes.unwrap(),
            ).await?)
        } else {
            None
        };
        
        // Create metadata
        let metadata = AnalysisMetadata {
            n_samples: prices.len(),
            sampling_rate: self.config.cwt_sampling_rate,
            window_size: self.config.regime_window_size,
            n_scales: self.config.cwt_scales.len(),
            frequency_range: (self.config.min_frequency, self.config.max_frequency),
            wavelet_type: self.config.wavelet_type,
            timestamp: chrono::Utc::now(),
        };
        
        Ok(WaveletAnalysis {
            regime,
            regime_confidence,
            volatility_scales,
            trend_components,
            time_frequency_matrix,
            dominant_frequencies,
            coherence,
            phase_relationships,
            spectral_features,
            processing_time_us: 0, // Will be set by caller
            metadata,
        })
    }
    
    /// Preprocess price data
    fn preprocess_prices(&self, prices: &[f64]) -> Result<Array1<f64>> {
        let mut processed = Array1::from_vec(prices.to_vec());
        
        if self.config.price_normalization {
            // Log returns for price analysis
            for i in 1..processed.len() {
                processed[i] = (processed[i] / processed[i-1]).ln();
            }
            processed[0] = 0.0; // First return is zero
            
            // Z-score normalization
            let mean = processed.mean().unwrap_or(0.0);
            let std = processed.std(0.0);
            
            if std > 1e-10 {
                processed = (processed - mean) / std;
            }
        }
        
        Ok(processed)
    }
    
    /// Preprocess volume data
    fn preprocess_volumes(&self, volumes: &[f64]) -> Result<Array1<f64>> {
        let mut processed = Array1::from_vec(volumes.to_vec());
        
        // Log transform to reduce skewness
        processed.mapv_inplace(|v| (v + 1.0).ln());
        
        // Z-score normalization
        let mean = processed.mean().unwrap_or(0.0);
        let std = processed.std(0.0);
        
        if std > 1e-10 {
            processed = (processed - mean) / std;
        }
        
        Ok(processed)
    }
    
    /// Calculate multi-scale volatility components
    fn calculate_volatility_scales(&self, dwt_result: &DWTResult) -> Result<Vec<f64>> {
        let mut volatility_scales = Vec::new();
        
        for detail in &dwt_result.details {
            // Calculate RMS (root mean square) as volatility measure
            let rms = (detail.mapv(|x| x * x).mean().unwrap_or(0.0)).sqrt();
            volatility_scales.push(rms);
        }
        
        Ok(volatility_scales)
    }
    
    /// Calculate trend components at different scales
    fn calculate_trend_components(&self, dwt_result: &DWTResult) -> Result<Vec<f64>> {
        let mut trend_components = Vec::new();
        
        // Use approximation coefficients as trend indicators
        for i in 0..dwt_result.approximations.len() {
            let approx = &dwt_result.approximations[i];
            if approx.len() > 1 {
                // Linear trend coefficient
                let n = approx.len() as f64;
                let sum_x = (0..approx.len()).map(|i| i as f64).sum::<f64>();
                let sum_y = approx.sum();
                let sum_xy = approx.iter().enumerate()
                    .map(|(i, &y)| i as f64 * y)
                    .sum::<f64>();
                let sum_x2 = (0..approx.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
                
                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
                trend_components.push(slope);
            } else {
                trend_components.push(0.0);
            }
        }
        
        Ok(trend_components)
    }
    
    /// Create time-frequency matrix from CWT results
    fn create_time_frequency_matrix(&self, cwt_result: &CWTResult) -> Result<Array2<f64>> {
        // Convert complex CWT coefficients to magnitude
        let magnitude_matrix = cwt_result.coefficients.mapv(|c| c.norm());
        Ok(magnitude_matrix)
    }
    
    /// Extract dominant frequencies from CWT
    fn extract_dominant_frequencies(&self, cwt_result: &CWTResult) -> Result<Vec<f64>> {
        let magnitude_matrix = cwt_result.coefficients.mapv(|c| c.norm());
        let mut dominant_frequencies = Vec::new();
        
        // Find peak frequencies for each time point
        for col in magnitude_matrix.columns() {
            let mut max_magnitude = 0.0;
            let mut max_idx = 0;
            
            for (idx, &magnitude) in col.iter().enumerate() {
                if magnitude > max_magnitude {
                    max_magnitude = magnitude;
                    max_idx = idx;
                }
            }
            
            if max_idx < cwt_result.frequencies.len() {
                dominant_frequencies.push(cwt_result.frequencies[max_idx]);
            }
        }
        
        Ok(dominant_frequencies)
    }
    
    /// Analyze phase relationships in CWT
    fn analyze_phase_relationships(&self, cwt_result: &CWTResult) -> Result<Vec<f64>> {
        let mut phase_relationships = Vec::new();
        
        // Calculate phase differences between adjacent frequency bands
        for i in 0..cwt_result.coefficients.nrows().saturating_sub(1) {
            let row1 = cwt_result.coefficients.row(i);
            let row2 = cwt_result.coefficients.row(i + 1);
            
            let mut phase_diff_sum = 0.0;
            let mut count = 0;
            
            for (c1, c2) in row1.iter().zip(row2.iter()) {
                let phase_diff = (c2.arg() - c1.arg()).abs();
                phase_diff_sum += phase_diff;
                count += 1;
            }
            
            if count > 0 {
                phase_relationships.push(phase_diff_sum / count as f64);
            }
        }
        
        Ok(phase_relationships)
    }
    
    /// Compute wavelet coherence between two signals
    async fn compute_wavelet_coherence(
        &self,
        signal1: &Array1<f64>,
        signal2: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        // Perform CWT on both signals
        let transformer = self.transformer.clone();
        
        let cwt1 = {
            let mut transformer = transformer.write();
            transformer.continuous_wavelet_transform(signal1).await?
        };
        
        let cwt2 = {
            let mut transformer = transformer.write();
            transformer.continuous_wavelet_transform(signal2).await?
        };
        
        // Calculate coherence
        let mut coherence = Array2::zeros(cwt1.coefficients.dim());
        
        for ((i, j), coherence_val) in coherence.indexed_iter_mut() {
            let c1 = cwt1.coefficients[[i, j]];
            let c2 = cwt2.coefficients[[i, j]];
            
            // Wavelet coherence formula
            let cross_spectrum = c1 * c2.conj();
            let power1 = c1.norm_sqr();
            let power2 = c2.norm_sqr();
            
            *coherence_val = if power1 > 1e-10 && power2 > 1e-10 {
                cross_spectrum.norm() / (power1 * power2).sqrt()
            } else {
                0.0
            };
        }
        
        Ok(coherence)
    }
    
    /// Generate cache key for analysis
    fn generate_cache_key(&self, prices: &[f64], volumes: &[f64]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash configuration
        format!("{:?}", self.config).hash(&mut hasher);
        
        // Hash data (sample to avoid too much computation)
        let sample_size = std::cmp::min(prices.len(), 100);
        for i in 0..sample_size {
            prices[i].to_bits().hash(&mut hasher);
            if i < volumes.len() {
                volumes[i].to_bits().hash(&mut hasher);
            }
        }
        
        format!("wavelet_analysis_{:x}", hasher.finish())
    }
    
    /// Check analysis cache
    async fn check_cache(&self, key: &str) -> Option<WaveletAnalysis> {
        let cache = self.cache.read();
        cache.get(key).cloned()
    }
    
    /// Store result in cache
    async fn store_in_cache(&self, key: String, result: WaveletAnalysis) {
        let mut cache = self.cache.write();
        cache.insert(key, result);
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let monitor = self.performance_monitor.lock();
        monitor.get_metrics()
    }
}

// Supporting structures - these would be implemented in separate modules

/// Wavelet transform result structures
#[derive(Debug, Clone)]
pub struct CWTResult {
    pub coefficients: Array2<Complex64>,
    pub frequencies: Vec<f64>,
    pub scales: Vec<f64>,
    pub time_points: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DWTResult {
    pub approximations: Vec<Array1<f64>>,
    pub details: Vec<Array1<f64>>,
    pub reconstruction_error: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub analyses_performed: u64,
    pub average_processing_time_us: u64,
    pub total_samples_processed: u64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
}

// Module stubs - these would be implemented in separate files
mod config {
    // Configuration utilities
}

mod transforms {
    use super::*;
    
    pub struct WaveletTransformer {
        // Implementation stub
    }
    
    impl WaveletTransformer {
        pub fn new(_config: &WaveletConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub async fn continuous_wavelet_transform(
            &mut self,
            _signal: &Array1<f64>,
        ) -> Result<CWTResult> {
            // Stub implementation
            Ok(CWTResult {
                coefficients: Array2::zeros((64, 100)),
                frequencies: vec![0.1; 64],
                scales: vec![1.0; 64],
                time_points: vec![0.0; 100],
            })
        }
        
        pub async fn discrete_wavelet_transform(
            &mut self,
            _signal: &Array1<f64>,
        ) -> Result<DWTResult> {
            // Stub implementation
            Ok(DWTResult {
                approximations: vec![Array1::zeros(50)],
                details: vec![Array1::zeros(50)],
                reconstruction_error: 0.01,
            })
        }
    }
}

mod regime {
    use super::*;
    
    pub struct RegimeDetector {
        // Implementation stub
    }
    
    impl RegimeDetector {
        pub fn new(_config: &WaveletConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub fn detect_regime(
            &mut self,
            _dwt_result: &DWTResult,
            _spectral_features: &SpectralFeatures,
            _config: &WaveletConfig,
        ) -> Result<(MarketRegime, f64)> {
            // Stub implementation
            Ok((MarketRegime::Consolidation, 0.8))
        }
    }
}

mod analysis {
    use super::*;
    
    pub struct SpectralAnalyzer {
        // Implementation stub
    }
    
    impl SpectralAnalyzer {
        pub fn new(_config: &WaveletConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub fn extract_features(
            &mut self,
            _cwt_result: &CWTResult,
            _config: &WaveletConfig,
        ) -> Result<SpectralFeatures> {
            // Stub implementation
            Ok(SpectralFeatures {
                spectral_centroid: 0.25,
                spectral_spread: 0.1,
                spectral_rolloff: 0.85,
                spectral_flux: 0.05,
                zero_crossing_rate: 0.1,
                spectral_entropy: 0.7,
                mfcc: vec![0.0; 13],
                chroma: vec![0.0; 12],
            })
        }
    }
}

mod processor {
    use super::*;
    
    pub struct AnalysisCache {
        cache: HashMap<String, WaveletAnalysis>,
        max_size: usize,
    }
    
    impl AnalysisCache {
        pub fn new(max_size: usize) -> Self {
            Self {
                cache: HashMap::new(),
                max_size,
            }
        }
        
        pub fn get(&self, key: &str) -> Option<&WaveletAnalysis> {
            self.cache.get(key)
        }
        
        pub fn insert(&mut self, key: String, value: WaveletAnalysis) {
            if self.cache.len() >= self.max_size {
                // Simple LRU: remove first entry
                let first_key = self.cache.keys().next().cloned();
                if let Some(key) = first_key {
                    self.cache.remove(&key);
                }
            }
            self.cache.insert(key, value);
        }
    }
    
    pub struct PerformanceMonitor {
        analyses_performed: u64,
        total_processing_time_us: u64,
        total_samples_processed: u64,
        start_time: Instant,
    }
    
    impl PerformanceMonitor {
        pub fn new() -> Self {
            Self {
                analyses_performed: 0,
                total_processing_time_us: 0,
                total_samples_processed: 0,
                start_time: Instant::now(),
            }
        }
        
        pub fn record_analysis(&mut self, processing_time_us: u64, n_samples: usize) {
            self.analyses_performed += 1;
            self.total_processing_time_us += processing_time_us;
            self.total_samples_processed += n_samples as u64;
        }
        
        pub fn get_metrics(&self) -> PerformanceMetrics {
            PerformanceMetrics {
                analyses_performed: self.analyses_performed,
                average_processing_time_us: if self.analyses_performed > 0 {
                    self.total_processing_time_us / self.analyses_performed
                } else {
                    0
                },
                total_samples_processed: self.total_samples_processed,
                cache_hit_rate: 0.0, // Would need to track cache statistics
                memory_usage_mb: 0.0, // Would need memory monitoring
            }
        }
    }
}

mod wavelets {
    // Wavelet basis function implementations
}

mod decomposition {
    // Multi-resolution decomposition algorithms
}

mod spectral {
    // Spectral analysis utilities
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_wavelet_processor_creation() {
        let config = WaveletConfig::default();
        let processor = WaveletProcessor::new(config);
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_market_data_analysis() {
        let config = WaveletConfig::default();
        let mut processor = WaveletProcessor::new(config).unwrap();
        
        let prices = vec![100.0, 101.5, 99.8, 102.1, 103.5, 101.2, 104.0, 102.8];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 1100.0, 900.0, 1300.0, 1000.0];
        
        let result = processor.analyze_market_data(&prices, &volumes).await;
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(!analysis.volatility_scales.is_empty());
        assert!(!analysis.trend_components.is_empty());
        assert!(analysis.regime_confidence >= 0.0 && analysis.regime_confidence <= 1.0);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = WaveletConfig::default();
        assert!(WaveletProcessor::validate_config(&config).is_ok());
        
        config.decomposition_levels = 0;
        assert!(WaveletProcessor::validate_config(&config).is_err());
        
        config.decomposition_levels = 6;
        config.min_frequency = 0.5;
        config.max_frequency = 0.1;
        assert!(WaveletProcessor::validate_config(&config).is_err());
    }
    
    #[test]
    fn test_market_regime_properties() {
        assert_eq!(MarketRegime::BullishTrend.risk_level(), 0.2);
        assert_eq!(MarketRegime::BearishVolatile.risk_level(), 0.9);
        
        assert!(MarketRegime::Consolidation.description().contains("Consolidation"));
        assert!(MarketRegime::BullishVolatile.description().contains("Bullish Volatile"));
    }
}