//! Multi-source data fusion algorithms

use crate::{config::FusionConfig, error::{FusionError, FusionResult}, ComponentHealth, sentiment::SentimentResult, indicators::IndicatorValue, types::DataItem};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use statrs::statistics::Statistics;

/// Data fusion engine for combining multiple data sources
pub struct DataFusion {
    config: Arc<FusionConfig>,
    fusion_state: Arc<RwLock<FusionState>>,
    kalman_filter: Arc<RwLock<KalmanFilter>>,
    bayesian_fusioner: Arc<BayesianFusioner>,
    metrics: Arc<RwLock<FusionMetrics>>,
}

/// Fusion input containing all data sources
#[derive(Debug, Clone)]
pub struct FusionInput {
    pub raw_data: DataItem,
    pub features: ProcessedData,
    pub indicators: HashMap<String, IndicatorValue>,
    pub sentiment: Option<SentimentResult>,
}

/// Processed data from fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub market_data: MarketDataFused,
    pub technical_indicators: HashMap<String, f64>,
    pub sentiment_scores: Option<SentimentScores>,
    pub features: Vec<f64>,
    pub quality_score: f64,
    pub confidence: f64,
    pub fusion_metadata: FusionMetadata,
}

/// Fused market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataFused {
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
    pub volatility: f64,
    pub trend_strength: f64,
    pub momentum: f64,
}

/// Sentiment scores from fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    pub overall_sentiment: f64,
    pub news_sentiment: f64,
    pub social_sentiment: f64,
    pub market_sentiment: f64,
    pub composite_score: f64,
}

/// Fusion metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    pub algorithm_used: String,
    pub sources_count: usize,
    pub processing_time: Duration,
    pub quality_metrics: QualityMetrics,
    pub outliers_detected: usize,
    pub interpolated_points: usize,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub completeness: f64,
    pub accuracy: f64,
    pub consistency: f64,
    pub timeliness: f64,
    pub overall_quality: f64,
}

/// Fusion state
#[derive(Debug, Clone)]
pub struct FusionState {
    pub last_fusion_time: Option<chrono::DateTime<chrono::Utc>>,
    pub active_sources: Vec<String>,
    pub source_weights: HashMap<String, f64>,
    pub alignment_window_data: HashMap<String, Vec<AlignedDataPoint>>,
    pub outlier_history: Vec<OutlierRecord>,
}

/// Aligned data point
#[derive(Debug, Clone)]
pub struct AlignedDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub source: String,
    pub quality: f64,
}

/// Outlier record
#[derive(Debug, Clone)]
pub struct OutlierRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub value: f64,
    pub z_score: f64,
    pub action_taken: OutlierAction,
}

/// Outlier action
#[derive(Debug, Clone)]
pub enum OutlierAction {
    Removed,
    Adjusted,
    Flagged,
    Ignored,
}

/// Kalman filter for data fusion
pub struct KalmanFilter {
    state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    measurement_noise: DMatrix<f64>,
    transition_matrix: DMatrix<f64>,
    observation_matrix: DMatrix<f64>,
}

impl KalmanFilter {
    pub fn new(state_size: usize, measurement_size: usize) -> Self {
        Self {
            state: DVector::zeros(state_size),
            covariance: DMatrix::identity(state_size, state_size),
            process_noise: DMatrix::identity(state_size, state_size) * 0.01,
            measurement_noise: DMatrix::identity(measurement_size, measurement_size) * 0.1,
            transition_matrix: DMatrix::identity(state_size, state_size),
            observation_matrix: DMatrix::identity(measurement_size, state_size),
        }
    }

    pub fn predict(&mut self) -> FusionResult<()> {
        // Predict state
        self.state = &self.transition_matrix * &self.state;
        
        // Predict covariance
        self.covariance = &self.transition_matrix * &self.covariance * self.transition_matrix.transpose() + &self.process_noise;
        
        Ok(())
    }

    pub fn update(&mut self, measurement: &DVector<f64>) -> FusionResult<()> {
        // Calculate Kalman gain
        let predicted_measurement = &self.observation_matrix * &self.state;
        let innovation = measurement - predicted_measurement;
        
        let innovation_covariance = &self.observation_matrix * &self.covariance * self.observation_matrix.transpose() + &self.measurement_noise;
        
        if let Some(inv_cov) = innovation_covariance.try_inverse() {
            let kalman_gain = &self.covariance * self.observation_matrix.transpose() * inv_cov;
            
            // Update state
            self.state += &kalman_gain * innovation;
            
            // Update covariance
            let identity = DMatrix::identity(self.state.len(), self.state.len());
            self.covariance = (identity - &kalman_gain * &self.observation_matrix) * &self.covariance;
            
            Ok(())
        } else {
            Err(FusionError::KalmanFilter("Failed to invert innovation covariance".to_string()))
        }
    }

    pub fn get_state(&self) -> &DVector<f64> {
        &self.state
    }
}

/// Bayesian fusioner for probabilistic data fusion
pub struct BayesianFusioner {
    config: Arc<FusionConfig>,
    priors: HashMap<String, f64>,
    likelihoods: HashMap<String, f64>,
}

impl BayesianFusioner {
    pub fn new(config: Arc<FusionConfig>) -> Self {
        Self {
            config,
            priors: HashMap::new(),
            likelihoods: HashMap::new(),
        }
    }

    pub fn fuse_data(&self, sources: &[DataSource]) -> FusionResult<f64> {
        if sources.is_empty() {
            return Err(FusionError::MissingSource("No data sources provided".to_string()));
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for source in sources {
            let prior = self.get_prior(&source.name);
            let likelihood = self.calculate_likelihood(&source.value, &source.uncertainty);
            let posterior = prior * likelihood;
            
            weighted_sum += source.value * posterior;
            total_weight += posterior;
        }

        if total_weight == 0.0 {
            return Err(FusionError::BayesianFusion("Total weight is zero".to_string()));
        }

        Ok(weighted_sum / total_weight)
    }

    fn get_prior(&self, source_name: &str) -> f64 {
        self.priors.get(source_name).copied().unwrap_or(1.0)
    }

    fn calculate_likelihood(&self, value: &f64, uncertainty: &f64) -> f64 {
        if *uncertainty <= 0.0 {
            return 1.0;
        }
        
        // Simple Gaussian likelihood
        let sigma = uncertainty;
        1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())
    }

    pub fn update_priors(&mut self, source_name: String, performance: f64) {
        let current_prior = self.priors.get(&source_name).copied().unwrap_or(1.0);
        let learning_rate = 0.1;
        let new_prior = current_prior * (1.0 - learning_rate) + performance * learning_rate;
        self.priors.insert(source_name, new_prior);
    }
}

/// Data source for fusion
#[derive(Debug, Clone)]
pub struct DataSource {
    pub name: String,
    pub value: f64,
    pub uncertainty: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub quality: f64,
}

/// Fusion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetrics {
    pub fusions_performed: u64,
    pub average_processing_time: Duration,
    pub average_quality_score: f64,
    pub sources_used: HashMap<String, u64>,
    pub outliers_detected: u64,
    pub interpolations_performed: u64,
    pub error_count: u64,
    pub last_reset: chrono::DateTime<chrono::Utc>,
}

impl Default for FusionMetrics {
    fn default() -> Self {
        Self {
            fusions_performed: 0,
            average_processing_time: Duration::from_millis(0),
            average_quality_score: 0.0,
            sources_used: HashMap::new(),
            outliers_detected: 0,
            interpolations_performed: 0,
            error_count: 0,
            last_reset: chrono::Utc::now(),
        }
    }
}

use crate::config::{FusionAlgorithm, InterpolationMethod, MissingDataStrategy};

impl DataFusion {
    /// Create a new data fusion engine
    pub fn new(config: Arc<FusionConfig>) -> Result<Self> {
        info!("Initializing data fusion engine");
        
        let fusion_state = Arc::new(RwLock::new(FusionState {
            last_fusion_time: None,
            active_sources: Vec::new(),
            source_weights: HashMap::new(),
            alignment_window_data: HashMap::new(),
            outlier_history: Vec::new(),
        }));
        
        let kalman_filter = Arc::new(RwLock::new(KalmanFilter::new(4, 2)));
        let bayesian_fusioner = Arc::new(BayesianFusioner::new(config.clone()));
        let metrics = Arc::new(RwLock::new(FusionMetrics::default()));
        
        Ok(Self {
            config,
            fusion_state,
            kalman_filter,
            bayesian_fusioner,
            metrics,
        })
    }

    /// Fuse data from multiple sources
    #[instrument(skip(self, input))]
    pub async fn fuse(&self, input: FusionInput) -> FusionResult<ProcessedData> {
        let start_time = Instant::now();
        
        // Prepare data sources
        let sources = self.prepare_data_sources(&input).await?;
        
        // Temporal alignment
        let aligned_sources = self.temporal_alignment(&sources).await?;
        
        // Outlier detection and handling
        let clean_sources = self.handle_outliers(&aligned_sources).await?;
        
        // Perform fusion based on configured algorithm
        let fused_data = match self.config.algorithm {
            FusionAlgorithm::WeightedAverage => self.weighted_average_fusion(&clean_sources).await?,
            FusionAlgorithm::KalmanFilter => self.kalman_filter_fusion(&clean_sources).await?,
            FusionAlgorithm::BayesianFusion => self.bayesian_fusion(&clean_sources).await?,
            FusionAlgorithm::ParticleFilter => self.particle_filter_fusion(&clean_sources).await?,
            FusionAlgorithm::NeuralFusion => self.neural_fusion(&clean_sources).await?,
        };
        
        // Calculate quality score
        let quality_score = self.calculate_quality_score(&clean_sources, &fused_data).await?;
        
        // Build result
        let result = ProcessedData {
            timestamp: input.raw_data.timestamp,
            symbol: input.raw_data.symbol.clone(),
            market_data: self.extract_market_data(&fused_data, &input),
            technical_indicators: input.indicators.iter()
                .map(|(k, v)| (k.clone(), v.value))
                .collect(),
            sentiment_scores: input.sentiment.as_ref().map(|s| SentimentScores {
                overall_sentiment: s.scores.compound,
                news_sentiment: s.scores.compound,
                social_sentiment: 0.0, // Would be extracted from social data
                market_sentiment: 0.0, // Would be calculated from market indicators
                composite_score: s.scores.compound,
            }),
            features: fused_data.clone(),
            quality_score,
            confidence: self.calculate_confidence(&clean_sources).await?,
            fusion_metadata: FusionMetadata {
                algorithm_used: format!("{:?}", self.config.algorithm),
                sources_count: sources.len(),
                processing_time: start_time.elapsed(),
                quality_metrics: QualityMetrics {
                    completeness: 1.0,
                    accuracy: quality_score,
                    consistency: 0.95,
                    timeliness: 1.0,
                    overall_quality: quality_score,
                },
                outliers_detected: 0,
                interpolated_points: 0,
            },
        };
        
        // Update metrics
        self.update_metrics(&result).await?;
        
        Ok(result)
    }

    /// Prepare data sources from input
    async fn prepare_data_sources(&self, input: &FusionInput) -> FusionResult<Vec<DataSource>> {
        let mut sources = Vec::new();
        
        // Add market data source
        sources.push(DataSource {
            name: "market_data".to_string(),
            value: input.raw_data.price,
            uncertainty: input.raw_data.volume.sqrt() * 0.01, // Simple uncertainty model
            timestamp: input.raw_data.timestamp,
            quality: 0.95,
        });
        
        // Add technical indicators as sources
        for (name, indicator) in &input.indicators {
            sources.push(DataSource {
                name: format!("indicator_{}", name),
                value: indicator.value,
                uncertainty: 1.0 - indicator.confidence,
                timestamp: indicator.timestamp,
                quality: indicator.confidence,
            });
        }
        
        // Add sentiment as a source if available
        if let Some(sentiment) = &input.sentiment {
            sources.push(DataSource {
                name: "sentiment".to_string(),
                value: sentiment.scores.compound,
                uncertainty: 1.0 - sentiment.confidence,
                timestamp: chrono::Utc::now(), // Use current time for sentiment
                quality: sentiment.confidence,
            });
        }
        
        Ok(sources)
    }

    /// Temporal alignment of data sources
    async fn temporal_alignment(&self, sources: &[DataSource]) -> FusionResult<Vec<DataSource>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        
        let reference_time = sources.iter()
            .map(|s| s.timestamp)
            .max()
            .unwrap();
        
        let mut aligned_sources = Vec::new();
        
        for source in sources {
            let time_diff = (reference_time - source.timestamp).num_milliseconds().abs() as u64;
            
            if Duration::from_millis(time_diff) <= self.config.alignment_window {
                // Within alignment window, use as-is
                aligned_sources.push(source.clone());
            } else {
                // Outside window, need interpolation or dropping
                match self.config.interpolation {
                    InterpolationMethod::Linear => {
                        // For simplicity, just use the value with adjusted quality
                        let mut adjusted_source = source.clone();
                        adjusted_source.quality *= 0.8; // Reduce quality for old data
                        adjusted_source.timestamp = reference_time;
                        aligned_sources.push(adjusted_source);
                    }
                    InterpolationMethod::Nearest => {
                        let mut adjusted_source = source.clone();
                        adjusted_source.timestamp = reference_time;
                        aligned_sources.push(adjusted_source);
                    }
                    _ => {
                        // For other methods, skip for now
                        continue;
                    }
                }
            }
        }
        
        Ok(aligned_sources)
    }

    /// Handle outliers in the data
    async fn handle_outliers(&self, sources: &[DataSource]) -> FusionResult<Vec<DataSource>> {
        if sources.len() < 3 {
            return Ok(sources.to_vec());
        }
        
        let values: Vec<f64> = sources.iter().map(|s| s.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        let mut clean_sources = Vec::new();
        
        for source in sources {
            let z_score = (source.value - mean) / std_dev;
            
            if z_score.abs() <= self.config.outlier_threshold {
                clean_sources.push(source.clone());
            } else {
                // Handle outlier based on strategy
                let outlier_record = OutlierRecord {
                    timestamp: source.timestamp,
                    source: source.name.clone(),
                    value: source.value,
                    z_score,
                    action_taken: OutlierAction::Flagged,
                };
                
                // For now, just flag outliers but include them with reduced weight
                let mut adjusted_source = source.clone();
                adjusted_source.quality *= 0.5;
                clean_sources.push(adjusted_source);
                
                // Store outlier record
                {
                    let mut state = self.fusion_state.write().await;
                    state.outlier_history.push(outlier_record);
                }
            }
        }
        
        Ok(clean_sources)
    }

    /// Weighted average fusion
    async fn weighted_average_fusion(&self, sources: &[DataSource]) -> FusionResult<Vec<f64>> {
        if sources.is_empty() {
            return Err(FusionError::MissingSource("No sources for fusion".to_string()));
        }
        
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for source in sources {
            let weight = self.get_source_weight(&source.name) * source.quality;
            weighted_sum += source.value * weight;
            total_weight += weight;
        }
        
        if total_weight == 0.0 {
            return Err(FusionError::Weight("Total weight is zero".to_string()));
        }
        
        let fused_value = weighted_sum / total_weight;
        Ok(vec![fused_value])
    }

    /// Kalman filter fusion
    async fn kalman_filter_fusion(&self, sources: &[DataSource]) -> FusionResult<Vec<f64>> {
        if sources.is_empty() {
            return Err(FusionError::MissingSource("No sources for Kalman filter".to_string()));
        }
        
        let mut kalman = self.kalman_filter.write().await;
        
        // Predict step
        kalman.predict()?;
        
        // Update step with measurements
        for source in sources {
            let measurement = DVector::from_vec(vec![source.value, source.quality]);
            kalman.update(&measurement)?;
        }
        
        let state = kalman.get_state();
        Ok(state.iter().cloned().collect())
    }

    /// Bayesian fusion
    async fn bayesian_fusion(&self, sources: &[DataSource]) -> FusionResult<Vec<f64>> {
        let fused_value = self.bayesian_fusioner.fuse_data(sources)?;
        Ok(vec![fused_value])
    }

    /// Particle filter fusion (simplified implementation)
    async fn particle_filter_fusion(&self, sources: &[DataSource]) -> FusionResult<Vec<f64>> {
        // Simplified particle filter - just return weighted average for now
        self.weighted_average_fusion(sources).await
    }

    /// Neural fusion (placeholder for neural network-based fusion)
    async fn neural_fusion(&self, sources: &[DataSource]) -> FusionResult<Vec<f64>> {
        // Placeholder - would use trained neural network
        self.weighted_average_fusion(sources).await
    }

    /// Get source weight from configuration
    fn get_source_weight(&self, source_name: &str) -> f64 {
        match source_name {
            name if name.contains("market_data") => self.config.source_weights.market_data,
            name if name.contains("sentiment") => self.config.source_weights.news_sentiment,
            name if name.contains("indicator") => self.config.source_weights.technical_indicators,
            name if name.contains("social") => self.config.source_weights.social_media,
            name if name.contains("economic") => self.config.source_weights.economic_data,
            _ => 0.1, // Default weight for unknown sources
        }
    }

    /// Calculate quality score for fused data
    async fn calculate_quality_score(&self, sources: &[DataSource], _fused_data: &[f64]) -> FusionResult<f64> {
        if sources.is_empty() {
            return Ok(0.0);
        }
        
        let total_quality: f64 = sources.iter().map(|s| s.quality).sum();
        let average_quality = total_quality / sources.len() as f64;
        
        // Adjust based on number of sources (more sources = higher confidence)
        let source_factor = (sources.len() as f64).min(5.0) / 5.0;
        
        Ok(average_quality * source_factor)
    }

    /// Calculate confidence score
    async fn calculate_confidence(&self, sources: &[DataSource]) -> FusionResult<f64> {
        if sources.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate variance of source values
        let values: Vec<f64> = sources.iter().map(|s| s.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        // Higher variance = lower confidence
        let confidence = 1.0 / (1.0 + variance);
        
        Ok(confidence.min(1.0))
    }

    /// Extract market data from fused result
    fn extract_market_data(&self, fused_data: &[f64], input: &FusionInput) -> MarketDataFused {
        let base_price = fused_data.get(0).copied().unwrap_or(input.raw_data.price);
        
        MarketDataFused {
            price: base_price,
            volume: input.raw_data.volume,
            bid: input.raw_data.bid.unwrap_or(base_price - 0.01),
            ask: input.raw_data.ask.unwrap_or(base_price + 0.01),
            spread: input.raw_data.ask.unwrap_or(base_price + 0.01) - input.raw_data.bid.unwrap_or(base_price - 0.01),
            volatility: 0.02, // Would be calculated from historical data
            trend_strength: 0.5, // Would be derived from indicators
            momentum: 0.0, // Would be calculated from price changes
        }
    }

    /// Update fusion metrics
    async fn update_metrics(&self, result: &ProcessedData) -> FusionResult<()> {
        let mut metrics = self.metrics.write().await;
        
        metrics.fusions_performed += 1;
        metrics.average_processing_time = Duration::from_millis(
            ((metrics.average_processing_time.as_millis() as u64 * (metrics.fusions_performed - 1)) + 
             result.fusion_metadata.processing_time.as_millis() as u64) / metrics.fusions_performed
        );
        metrics.average_quality_score = 
            ((metrics.average_quality_score * (metrics.fusions_performed - 1) as f64) + 
             result.quality_score) / metrics.fusions_performed as f64;
        
        Ok(())
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let metrics = self.metrics.read().await;
        
        if metrics.error_count > 10 {
            Ok(ComponentHealth::Unhealthy)
        } else if metrics.average_quality_score < 0.5 {
            Ok(ComponentHealth::Degraded)
        } else {
            Ok(ComponentHealth::Healthy)
        }
    }

    /// Get fusion metrics
    pub async fn get_metrics(&self) -> FusionMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset fusion engine
    pub async fn reset(&self) -> Result<()> {
        info!("Resetting data fusion engine");
        
        // Reset state
        {
            let mut state = self.fusion_state.write().await;
            state.last_fusion_time = None;
            state.active_sources.clear();
            state.source_weights.clear();
            state.alignment_window_data.clear();
            state.outlier_history.clear();
        }
        
        // Reset Kalman filter
        {
            let mut kalman = self.kalman_filter.write().await;
            *kalman = KalmanFilter::new(4, 2);
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = FusionMetrics::default();
        }
        
        info!("Data fusion engine reset successfully");
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_kalman_filter() {
        let mut filter = KalmanFilter::new(2, 1);
        
        // Test prediction
        assert!(filter.predict().is_ok());
        
        // Test update
        let measurement = DVector::from_vec(vec![1.0]);
        assert!(filter.update(&measurement).is_ok());
        
        let state = filter.get_state();
        assert_eq!(state.len(), 2);
    }

    #[test]
    async fn test_bayesian_fusioner() {
        let config = Arc::new(FusionConfig::default());
        let fusioner = BayesianFusioner::new(config);
        
        let sources = vec![
            DataSource {
                name: "source1".to_string(),
                value: 100.0,
                uncertainty: 0.1,
                timestamp: chrono::Utc::now(),
                quality: 0.9,
            },
            DataSource {
                name: "source2".to_string(),
                value: 102.0,
                uncertainty: 0.2,
                timestamp: chrono::Utc::now(),
                quality: 0.8,
            },
        ];
        
        let result = fusioner.fuse_data(&sources).unwrap();
        assert!(result > 99.0 && result < 103.0);
    }

    #[test]
    async fn test_data_fusion() {
        let config = Arc::new(FusionConfig::default());
        let fusion = DataFusion::new(config).unwrap();
        
        let data_item = DataItem {
            symbol: "TEST".to_string(),
            timestamp: chrono::Utc::now(),
            price: 100.0,
            volume: 1000.0,
            bid: Some(99.5),
            ask: Some(100.5),
            text: Some("Test news".to_string()),
            raw_data: vec![],
        };
        
        let input = FusionInput {
            raw_data: data_item,
            features: ProcessedData {
                timestamp: chrono::Utc::now(),
                symbol: "TEST".to_string(),
                market_data: MarketDataFused {
                    price: 100.0,
                    volume: 1000.0,
                    bid: 99.5,
                    ask: 100.5,
                    spread: 1.0,
                    volatility: 0.02,
                    trend_strength: 0.5,
                    momentum: 0.0,
                },
                technical_indicators: HashMap::new(),
                sentiment_scores: None,
                features: vec![],
                quality_score: 0.9,
                confidence: 0.8,
                fusion_metadata: FusionMetadata {
                    algorithm_used: "test".to_string(),
                    sources_count: 1,
                    processing_time: Duration::from_millis(10),
                    quality_metrics: QualityMetrics {
                        completeness: 1.0,
                        accuracy: 0.9,
                        consistency: 0.95,
                        timeliness: 1.0,
                        overall_quality: 0.9,
                    },
                    outliers_detected: 0,
                    interpolated_points: 0,
                },
            },
            indicators: HashMap::new(),
            sentiment: None,
        };
        
        let result = fusion.fuse(input).await.unwrap();
        assert_eq!(result.symbol, "TEST");
        assert!(result.quality_score > 0.0);
    }
}