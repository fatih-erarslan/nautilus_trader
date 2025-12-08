//! # Rust Market Analysis Engine
//! 
//! Advanced financial market analysis engine with machine learning, whale tracking,
//! and real-time analytics for cryptocurrency and traditional markets.
//! 
//! ## Features
//! 
//! - **Whale Detection**: Advanced volume profile and order flow analysis
//! - **Market Regime Detection**: ML-based classification of market conditions
//! - **Pattern Recognition**: Technical analysis with statistical validation
//! - **Predictive Analytics**: Time series forecasting with neural networks
//! - **Market Microstructure**: Order book and liquidity analysis
//! - **Real-time Processing**: SIMD-optimized streaming analytics
//! 
//! ## Quick Start
//! 
//! ```rust
//! use rust_market_analysis::{MarketAnalyzer, Config};
//! 
//! let config = Config::default();
//! let analyzer = MarketAnalyzer::new(config)?;
//! 
//! // Analyze market data
//! let analysis = analyzer.analyze_market(&market_data).await?;
//! println!("Whale activity detected: {}", analysis.whale_signals.len());
//! ```

pub mod config;
pub mod error;
pub mod types;
pub mod data;
pub mod whale_analysis;
pub mod regime_detection;
pub mod pattern_recognition;
pub mod predictive_models;
pub mod market_microstructure;
pub mod performance;
pub mod utils;

#[cfg(feature = "python")]
pub mod python_bindings;

// Re-exports for convenience
pub use config::Config;
pub use error::{AnalysisError, Result};
pub use types::*;
pub use whale_analysis::WhaleAnalyzer;
pub use regime_detection::RegimeDetector;
pub use pattern_recognition::PatternRecognizer;
pub use predictive_models::PredictiveEngine;
pub use market_microstructure::MicrostructureAnalyzer;

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn, error};

/// Main market analysis engine
pub struct MarketAnalyzer {
    config: Config,
    whale_analyzer: WhaleAnalyzer,
    regime_detector: RegimeDetector,
    pattern_recognizer: PatternRecognizer,
    predictive_engine: PredictiveEngine,
    microstructure_analyzer: MicrostructureAnalyzer,
    
    // State management
    market_state: Arc<RwLock<MarketState>>,
    analysis_cache: Arc<DashMap<String, CachedAnalysis>>,
    
    // Event system
    signal_sender: broadcast::Sender<AnalysisSignal>,
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl MarketAnalyzer {
    /// Create a new market analyzer with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing MarketAnalyzer with config: {:?}", config);
        
        let whale_analyzer = WhaleAnalyzer::new(&config)?;
        let regime_detector = RegimeDetector::new(&config)?;
        let pattern_recognizer = PatternRecognizer::new(&config)?;
        let predictive_engine = PredictiveEngine::new(&config)?;
        let microstructure_analyzer = MicrostructureAnalyzer::new(&config)?;
        
        let (signal_sender, _) = broadcast::channel(1000);
        
        Ok(Self {
            config,
            whale_analyzer,
            regime_detector,
            pattern_recognizer,
            predictive_engine,
            microstructure_analyzer,
            market_state: Arc::new(RwLock::new(MarketState::default())),
            analysis_cache: Arc::new(DashMap::new()),
            signal_sender,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        })
    }
    
    /// Perform comprehensive market analysis
    pub async fn analyze_market(&self, data: &MarketData) -> Result<MarketAnalysis> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(data);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            if !cached.is_expired() {
                return Ok(cached.analysis.clone());
            }
        }
        
        // Parallel analysis execution
        let (whale_signals, regime_info, patterns, predictions, microstructure) = tokio::try_join!(
            self.whale_analyzer.analyze(data),
            self.regime_detector.detect_regime(data),
            self.pattern_recognizer.recognize_patterns(data),
            self.predictive_engine.generate_predictions(data),
            self.microstructure_analyzer.analyze_structure(data)
        )?;
        
        // Combine analysis results
        let analysis = MarketAnalysis {
            timestamp: chrono::Utc::now(),
            symbol: data.symbol.clone(),
            whale_signals,
            regime_info,
            patterns,
            predictions,
            microstructure,
            confidence_score: self.calculate_confidence_score(&data),
            risk_metrics: self.calculate_risk_metrics(&data).await?,
        };
        
        // Cache the result
        self.analysis_cache.insert(
            cache_key,
            CachedAnalysis {
                analysis: analysis.clone(),
                timestamp: chrono::Utc::now(),
                ttl: chrono::Duration::minutes(self.config.cache_ttl_minutes),
            }
        );
        
        // Update metrics
        let processing_time = start_time.elapsed();
        self.update_metrics(processing_time, &analysis).await;
        
        // Emit signals for important findings
        self.emit_signals(&analysis).await;
        
        Ok(analysis)
    }
    
    /// Stream real-time market analysis
    pub async fn stream_analysis(
        &self, 
        mut data_stream: impl futures::Stream<Item = MarketData> + Unpin
    ) -> Result<impl futures::Stream<Item = Result<MarketAnalysis>>> {
        use futures::StreamExt;
        
        Ok(data_stream.then(move |data| {
            let analyzer = self.clone();
            async move {
                analyzer.analyze_market(&data).await
            }
        }))
    }
    
    /// Get current market state
    pub fn get_market_state(&self) -> MarketState {
        self.market_state.read().clone()
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().clone()
    }
    
    /// Subscribe to analysis signals
    pub fn subscribe_signals(&self) -> broadcast::Receiver<AnalysisSignal> {
        self.signal_sender.subscribe()
    }
    
    /// Update model parameters based on feedback
    pub async fn update_models(&mut self, feedback: ModelFeedback) -> Result<()> {
        info!("Updating models with feedback: {:?}", feedback);
        
        // Update individual analyzers
        self.whale_analyzer.update_parameters(&feedback.whale_feedback).await?;
        self.regime_detector.update_model(&feedback.regime_feedback).await?;
        self.pattern_recognizer.update_weights(&feedback.pattern_feedback).await?;
        self.predictive_engine.retrain(&feedback.prediction_feedback).await?;
        
        Ok(())
    }
    
    // Private helper methods
    
    fn generate_cache_key(&self, data: &MarketData) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.symbol.hash(&mut hasher);
        data.timestamp.hash(&mut hasher);
        data.prices.len().hash(&mut hasher);
        
        format!("{}_{}", data.symbol, hasher.finish())
    }
    
    fn calculate_confidence_score(&self, data: &MarketData) -> f64 {
        // Implement confidence scoring based on data quality and analysis consistency
        let data_quality = self.assess_data_quality(data);
        let analysis_consistency = self.assess_analysis_consistency(data);
        
        (data_quality + analysis_consistency) / 2.0
    }
    
    fn assess_data_quality(&self, data: &MarketData) -> f64 {
        let completeness = data.prices.len() as f64 / self.config.required_data_points as f64;
        let freshness = {
            let age = chrono::Utc::now() - data.timestamp;
            (1.0 - (age.num_seconds() as f64 / 3600.0)).max(0.0)
        };
        
        (completeness.min(1.0) + freshness) / 2.0
    }
    
    fn assess_analysis_consistency(&self, _data: &MarketData) -> f64 {
        // Implement consistency checking across different analysis modules
        0.8 // Placeholder
    }
    
    async fn calculate_risk_metrics(&self, data: &MarketData) -> Result<RiskMetrics> {
        // Integration with talebian-risk-rs for comprehensive risk analysis
        // Note: talebian_risk_rs integration disabled - requires external dependency
        // use talebian_risk_rs::{TailRiskAnalyzer, VolatilityRegimeDetector};
        
        // Placeholder implementations until talebian_risk_rs is available
        let var_95 = self.calculate_var_placeholder(&data.prices, 0.05);
        let expected_shortfall = self.calculate_es_placeholder(&data.prices, 0.05);
        let max_drawdown = self.calculate_max_drawdown_placeholder(&data.prices);
        let volatility_regime = "Normal".to_string();
        
        Ok(RiskMetrics {
            value_at_risk_95: var_95,
            expected_shortfall_95: expected_shortfall,
            maximum_drawdown: max_drawdown,
            volatility_regime,
            tail_ratio: self.calculate_tail_ratio_placeholder(&data.prices),
            skewness: self.calculate_skewness(&data.prices),
            kurtosis: self.calculate_kurtosis(&data.prices),
        })
    }
    
    fn calculate_skewness(&self, prices: &[f64]) -> f64 {
        use statrs::statistics::{Statistics, Distribution};
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
            
        if returns.len() < 3 {
            return 0.0;
        }
        
        let mean = returns.as_slice().mean();
        let std_dev = returns.as_slice().std_dev();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = returns.len() as f64;
        let skew_sum: f64 = returns.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum();
            
        (n / ((n - 1.0) * (n - 2.0))) * skew_sum
    }
    
    fn calculate_kurtosis(&self, prices: &[f64]) -> f64 {
        use statrs::statistics::{Statistics, Distribution};
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
            
        if returns.len() < 4 {
            return 0.0;
        }
        
        let mean = returns.as_slice().mean();
        let std_dev = returns.as_slice().std_dev();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = returns.len() as f64;
        let kurt_sum: f64 = returns.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum();
            
        let kurtosis = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * kurt_sum
            - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
            
        kurtosis
    }
    
    async fn update_metrics(&self, processing_time: std::time::Duration, analysis: &MarketAnalysis) {
        let mut metrics = self.metrics.write();
        metrics.total_analyses += 1;
        metrics.average_processing_time = 
            (metrics.average_processing_time * (metrics.total_analyses - 1) as f64 
             + processing_time.as_secs_f64()) / metrics.total_analyses as f64;
        metrics.last_analysis_time = chrono::Utc::now();
        
        // Update accuracy metrics based on analysis quality
        if analysis.confidence_score > 0.8 {
            metrics.high_confidence_analyses += 1;
        }
    }
    
    async fn emit_signals(&self, analysis: &MarketAnalysis) {
        // Emit whale signals
        for whale_signal in &analysis.whale_signals {
            if whale_signal.confidence > self.config.whale_signal_threshold {
                let signal = AnalysisSignal::WhaleActivity {
                    symbol: analysis.symbol.clone(),
                    signal: whale_signal.clone(),
                    timestamp: chrono::Utc::now(),
                };
                let _ = self.signal_sender.send(signal);
            }
        }
        
        // Emit regime change signals
        if analysis.regime_info.confidence > self.config.regime_change_threshold {
            let signal = AnalysisSignal::RegimeChange {
                symbol: analysis.symbol.clone(),
                old_regime: analysis.regime_info.previous_regime.clone(),
                new_regime: analysis.regime_info.current_regime.clone(),
                confidence: analysis.regime_info.confidence,
                timestamp: chrono::Utc::now(),
            };
            let _ = self.signal_sender.send(signal);
        }
        
        // Emit pattern signals
        for pattern in &analysis.patterns {
            if pattern.confidence > self.config.pattern_signal_threshold {
                let signal = AnalysisSignal::PatternDetected {
                    symbol: analysis.symbol.clone(),
                    pattern: pattern.clone(),
                    timestamp: chrono::Utc::now(),
                };
                let _ = self.signal_sender.send(signal);
            }
        }
    }
    
    // Placeholder methods for talebian_risk_rs integration
    fn calculate_var_placeholder(&self, prices: &[f64], confidence: f64) -> f64 {
        // Simple VaR calculation as placeholder
        if prices.len() < 2 { return 0.0; }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        sorted_returns.get(index).copied().unwrap_or(0.0)
    }
    
    fn calculate_es_placeholder(&self, prices: &[f64], confidence: f64) -> f64 {
        // Simple Expected Shortfall calculation as placeholder
        let var = self.calculate_var_placeholder(prices, confidence);
        var * 1.3 // Simple approximation
    }
    
    fn calculate_max_drawdown_placeholder(&self, prices: &[f64]) -> f64 {
        // Simple maximum drawdown calculation
        if prices.is_empty() { return 0.0; }
        
        let mut max_price = prices[0];
        let mut max_drawdown = 0.0;
        
        for &price in prices.iter() {
            if price > max_price {
                max_price = price;
            }
            let drawdown = (max_price - price) / max_price;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
    
    fn calculate_tail_ratio_placeholder(&self, prices: &[f64]) -> f64 {
        // Simple tail ratio approximation
        if prices.len() < 10 { return 1.0; }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_returns.len();
        let right_tail_idx = (0.95 * n as f64) as usize;
        let left_tail_idx = (0.05 * n as f64) as usize;
        
        if right_tail_idx < n && left_tail_idx < n {
            let right_tail = sorted_returns[right_tail_idx].abs();
            let left_tail = sorted_returns[left_tail_idx].abs();
            if left_tail > 0.0 { right_tail / left_tail } else { 1.0 }
        } else {
            1.0
        }
    }
}

impl Clone for MarketAnalyzer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            whale_analyzer: self.whale_analyzer.clone(),
            regime_detector: self.regime_detector.clone(),
            pattern_recognizer: self.pattern_recognizer.clone(),
            predictive_engine: self.predictive_engine.clone(),
            microstructure_analyzer: self.microstructure_analyzer.clone(),
            market_state: Arc::clone(&self.market_state),
            analysis_cache: Arc::clone(&self.analysis_cache),
            signal_sender: self.signal_sender.clone(),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

/// Cached analysis result with TTL
#[derive(Debug, Clone)]
struct CachedAnalysis {
    analysis: MarketAnalysis,
    timestamp: chrono::DateTime<chrono::Utc>,
    ttl: chrono::Duration,
}

impl CachedAnalysis {
    fn is_expired(&self) -> bool {
        chrono::Utc::now() - self.timestamp > self.ttl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_market_analyzer_creation() {
        let config = Config::default();
        let analyzer = MarketAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }
    
    #[tokio::test]
    async fn test_market_analysis() {
        let config = Config::default();
        let analyzer = MarketAnalyzer::new(config).unwrap();
        
        let market_data = MarketData::mock_data();
        let analysis = analyzer.analyze_market(&market_data).await;
        assert!(analysis.is_ok());
        
        let result = analysis.unwrap();
        assert!(!result.symbol.is_empty());
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
    }
}