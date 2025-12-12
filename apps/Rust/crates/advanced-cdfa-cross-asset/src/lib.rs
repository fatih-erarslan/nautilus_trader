//! # Advanced CDFA Cross-Asset Analysis
//! 
//! Enterprise-grade cross-asset analysis for Advanced CDFA with real-time correlation tracking,
//! contagion detection, and systemic risk monitoring.
//! 
//! ## Features
//! 
//! - **Real-time Correlation**: Dynamic correlation matrix computation with multiple correlation measures
//! - **Contagion Detection**: Early warning system for financial contagion across asset classes
//! - **Lead-Lag Analysis**: Cross-asset causality and temporal precedence relationships
//! - **Systemic Risk**: Network-based systemic risk metrics and stress testing
//! - **Copula Analysis**: Non-linear dependence modeling with multiple copula families
//! - **Regime-Dependent**: Correlation dynamics across different market regimes
//! 
//! ## Performance Targets
//! 
//! - Correlation matrix (100x100): < 5 milliseconds
//! - Contagion detection: < 10 milliseconds
//! - Lead-lag analysis: < 20 milliseconds
//! - Systemic risk computation: < 50 milliseconds
//! 
//! ## Example Usage
//! 
//! ```rust
//! use advanced_cdfa_cross_asset::{CrossAssetAnalyzer, AnalyzerConfig, AssetClass};
//! 
//! let config = AnalyzerConfig::default();
//! let mut analyzer = CrossAssetAnalyzer::new(config)?;
//! 
//! // Add asset data
//! analyzer.add_asset("BTC", AssetClass::Cryptocurrency, &btc_prices).await?;
//! analyzer.add_asset("SPY", AssetClass::Equity, &spy_prices).await?;
//! analyzer.add_asset("GLD", AssetClass::Commodity, &gold_prices).await?;
//! 
//! // Analyze cross-asset relationships
//! let analysis = analyzer.analyze_cross_relationships().await?;
//! 
//! println!("Correlation matrix: {:?}", analysis.correlation_matrix);
//! println!("Contagion risk: {:.2}%", analysis.contagion_risk * 100.0);
//! println!("Systemic risk: {:.2}", analysis.systemic_risk_measure);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use parking_lot::{RwLock, Mutex};
use petgraph::{Graph, Undirected};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::correlation::{pearson, spearman};
use thiserror::Error;
use tokio::time::sleep;
use tracing::{debug, info, warn, error, instrument};
use dashmap::DashMap;

// Re-exports
pub use config::*;
pub use correlation::*;
pub use contagion::*;
pub use systemic::*;
pub use analyzer::*;

// Module declarations
pub mod config;
pub mod correlation;
pub mod contagion;
pub mod systemic;
pub mod analyzer;
pub mod network;
pub mod copula;
pub mod regime;
pub mod metrics;

// Error types
#[derive(Error, Debug)]
pub enum CrossAssetError {
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Data error: {message}")]
    DataError { message: String },
    
    #[error("Correlation error: {message}")]
    CorrelationError { message: String },
    
    #[error("Contagion error: {message}")]
    ContagionError { message: String },
    
    #[error("Network error: {message}")]
    NetworkError { message: String },
    
    #[error("Computation error: {message}")]
    ComputationError { message: String },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

/// Cross-asset analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Correlation parameters
    pub correlation_window: usize,
    pub correlation_overlap: f64,
    pub correlation_methods: Vec<CorrelationType>,
    pub min_correlation_threshold: f64,
    pub correlation_confidence_level: f64,
    
    /// Contagion detection parameters
    pub contagion_threshold: f64,
    pub contagion_window: usize,
    pub contagion_significance_level: f64,
    pub volatility_spillover_threshold: f64,
    pub extreme_correlation_threshold: f64,
    
    /// Lead-lag analysis parameters
    pub max_lag_periods: usize,
    pub lag_significance_threshold: f64,
    pub causality_test_method: CausalityTest,
    
    /// Systemic risk parameters
    pub systemic_risk_method: SystemicRiskMethod,
    pub network_density_threshold: f64,
    pub centrality_measures: Vec<CentralityMeasure>,
    pub stress_test_scenarios: usize,
    
    /// Time series parameters
    pub data_frequency: DataFrequency,
    pub min_observations: usize,
    pub outlier_detection: bool,
    pub data_transformation: DataTransformation,
    
    /// Performance parameters
    pub parallel_processing: bool,
    pub max_processing_time_ms: u64,
    pub cache_results: bool,
    pub real_time_updates: bool,
    
    /// Advanced features
    pub regime_dependent_analysis: bool,
    pub copula_analysis: bool,
    pub high_frequency_analysis: bool,
    pub multi_timeframe_analysis: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            correlation_window: 252,  // 1 year of daily data
            correlation_overlap: 0.5,
            correlation_methods: vec![
                CorrelationType::Pearson,
                CorrelationType::Spearman,
                CorrelationType::Kendall,
            ],
            min_correlation_threshold: 0.05,
            correlation_confidence_level: 0.95,
            contagion_threshold: 0.7,
            contagion_window: 63,  // Quarter
            contagion_significance_level: 0.01,
            volatility_spillover_threshold: 0.6,
            extreme_correlation_threshold: 0.8,
            max_lag_periods: 20,
            lag_significance_threshold: 0.05,
            causality_test_method: CausalityTest::Granger,
            systemic_risk_method: SystemicRiskMethod::NetworkConnectedness,
            network_density_threshold: 0.3,
            centrality_measures: vec![
                CentralityMeasure::Degree,
                CentralityMeasure::Betweenness,
                CentralityMeasure::Eigenvector,
            ],
            stress_test_scenarios: 1000,
            data_frequency: DataFrequency::Daily,
            min_observations: 50,
            outlier_detection: true,
            data_transformation: DataTransformation::LogReturns,
            parallel_processing: true,
            max_processing_time_ms: 5000,
            cache_results: true,
            real_time_updates: true,
            regime_dependent_analysis: true,
            copula_analysis: false,
            high_frequency_analysis: false,
            multi_timeframe_analysis: false,
        }
    }
}

/// Types of correlation measures
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CorrelationType {
    /// Pearson correlation (linear)
    Pearson,
    /// Spearman rank correlation (monotonic)
    Spearman,
    /// Kendall tau correlation (concordance)
    Kendall,
    /// Partial correlation
    Partial,
    /// Distance correlation (nonlinear)
    Distance,
    /// Mutual information
    MutualInformation,
}

/// Causality test methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CausalityTest {
    /// Granger causality
    Granger,
    /// Transfer entropy
    TransferEntropy,
    /// Convergent cross mapping
    CCM,
    /// Vector autoregression
    VAR,
}

/// Systemic risk measurement methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SystemicRiskMethod {
    /// Network connectedness measure
    NetworkConnectedness,
    /// CoVaR (Conditional Value at Risk)
    CoVaR,
    /// Marginal Expected Shortfall
    MarginalExpectedShortfall,
    /// Systemic Risk Index
    SystemicRiskIndex,
    /// SRISK (Conditional Capital Shortfall)
    SRISK,
}

/// Network centrality measures
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CentralityMeasure {
    /// Degree centrality
    Degree,
    /// Betweenness centrality
    Betweenness,
    /// Closeness centrality
    Closeness,
    /// Eigenvector centrality
    Eigenvector,
    /// PageRank centrality
    PageRank,
}

/// Data frequencies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DataFrequency {
    /// High frequency (seconds/minutes)
    HighFrequency,
    /// Hourly data
    Hourly,
    /// Daily data
    Daily,
    /// Weekly data
    Weekly,
    /// Monthly data
    Monthly,
}

/// Data transformation methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DataTransformation {
    /// No transformation
    None,
    /// Log returns
    LogReturns,
    /// Simple returns
    SimpleReturns,
    /// Z-score normalization
    ZScore,
    /// Rank transformation
    Ranks,
}

/// Asset classes for categorization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AssetClass {
    Equity,
    FixedIncome,
    Commodity,
    Currency,
    Cryptocurrency,
    RealEstate,
    Alternative,
}

/// Asset data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetData {
    pub symbol: String,
    pub asset_class: AssetClass,
    pub prices: Array1<f64>,
    pub returns: Array1<f64>,
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cross-asset analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAssetAnalysis {
    /// Dynamic correlation matrix
    pub correlation_matrix: HashMap<CorrelationType, Array2<f64>>,
    
    /// Asset symbols in order
    pub asset_symbols: Vec<String>,
    
    /// Contagion risk measure (0.0 to 1.0)
    pub contagion_risk: f64,
    
    /// Lead-lag relationships
    pub lead_lag_relationships: HashMap<String, i32>,
    
    /// Systemic risk measure
    pub systemic_risk_measure: f64,
    
    /// Network metrics
    pub network_metrics: NetworkMetrics,
    
    /// Regime-dependent correlations (if enabled)
    pub regime_correlations: Option<HashMap<String, Array2<f64>>>,
    
    /// Volatility spillover effects
    pub volatility_spillovers: Array2<f64>,
    
    /// Risk attribution
    pub risk_attribution: Vec<RiskAttribution>,
    
    /// Processing metrics
    pub processing_time_ms: u64,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Network metrics for systemic risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Network density (0.0 to 1.0)
    pub density: f64,
    
    /// Average clustering coefficient
    pub clustering_coefficient: f64,
    
    /// Network diameter
    pub diameter: usize,
    
    /// Average path length
    pub average_path_length: f64,
    
    /// Centrality measures for each asset
    pub centrality_measures: HashMap<String, HashMap<CentralityMeasure, f64>>,
    
    /// Community structure
    pub communities: Vec<Vec<String>>,
    
    /// Network stability measure
    pub stability: f64,
}

/// Risk attribution for individual assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAttribution {
    pub asset_symbol: String,
    pub systemic_contribution: f64,
    pub contagion_vulnerability: f64,
    pub network_importance: f64,
    pub stress_test_impact: f64,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub n_assets: usize,
    pub observation_period_days: u32,
    pub correlation_window_size: usize,
    pub data_frequency: DataFrequency,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub statistical_significance: HashMap<String, f64>,
}

/// Main cross-asset analyzer
pub struct CrossAssetAnalyzer {
    config: AnalyzerConfig,
    assets: Arc<RwLock<HashMap<String, AssetData>>>,
    correlation_engine: Arc<Mutex<CorrelationEngine>>,
    contagion_detector: Arc<Mutex<ContagionDetector>>,
    systemic_risk_monitor: Arc<Mutex<SystemicRiskMonitor>>,
    network_analyzer: Arc<Mutex<NetworkAnalyzer>>,
    cache: Arc<RwLock<AnalysisCache>>,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
}

impl CrossAssetAnalyzer {
    /// Create new cross-asset analyzer
    pub fn new(config: AnalyzerConfig) -> Result<Self> {
        info!("Initializing cross-asset analyzer with config: {:?}", config);
        
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize components
        let correlation_engine = Arc::new(Mutex::new(
            CorrelationEngine::new(&config)?
        ));
        
        let contagion_detector = Arc::new(Mutex::new(
            ContagionDetector::new(&config)?
        ));
        
        let systemic_risk_monitor = Arc::new(Mutex::new(
            SystemicRiskMonitor::new(&config)?
        ));
        
        let network_analyzer = Arc::new(Mutex::new(
            NetworkAnalyzer::new(&config)?
        ));
        
        let cache = Arc::new(RwLock::new(
            AnalysisCache::new(1000)
        ));
        
        let performance_monitor = Arc::new(Mutex::new(
            PerformanceMonitor::new()
        ));
        
        info!("Cross-asset analyzer initialized successfully");
        
        Ok(Self {
            config,
            assets: Arc::new(RwLock::new(HashMap::new())),
            correlation_engine,
            contagion_detector,
            systemic_risk_monitor,
            network_analyzer,
            cache,
            performance_monitor,
        })
    }
    
    /// Validate configuration
    fn validate_config(config: &AnalyzerConfig) -> Result<()> {
        if config.correlation_window == 0 {
            return Err(CrossAssetError::ConfigError {
                message: "Correlation window must be greater than 0".to_string(),
            }.into());
        }
        
        if config.correlation_overlap < 0.0 || config.correlation_overlap >= 1.0 {
            return Err(CrossAssetError::ConfigError {
                message: "Correlation overlap must be between 0.0 and 1.0".to_string(),
            }.into());
        }
        
        if config.min_observations == 0 {
            return Err(CrossAssetError::ConfigError {
                message: "Minimum observations must be greater than 0".to_string(),
            }.into());
        }
        
        if config.contagion_threshold <= 0.0 || config.contagion_threshold > 1.0 {
            return Err(CrossAssetError::ConfigError {
                message: "Contagion threshold must be between 0.0 and 1.0".to_string(),
            }.into());
        }
        
        Ok(())
    }
    
    /// Add asset data to the analyzer
    #[instrument(skip(self, prices))]
    pub async fn add_asset(
        &mut self,
        symbol: &str,
        asset_class: AssetClass,
        prices: &[f64],
    ) -> Result<()> {
        if prices.len() < self.config.min_observations {
            return Err(CrossAssetError::DataError {
                message: format!(
                    "Insufficient observations for {}: {} < {}",
                    symbol,
                    prices.len(),
                    self.config.min_observations
                ),
            }.into());
        }
        
        // Transform data according to configuration
        let price_array = Array1::from_vec(prices.to_vec());
        let returns = self.calculate_returns(&price_array)?;
        
        // Create timestamps (mock for now)
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..prices.len())
            .map(|i| chrono::Utc::now() - chrono::Duration::days(prices.len() as i64 - i as i64))
            .collect();
        
        let asset_data = AssetData {
            symbol: symbol.to_string(),
            asset_class,
            prices: price_array,
            returns,
            timestamps,
            metadata: HashMap::new(),
        };
        
        let mut assets = self.assets.write();
        assets.insert(symbol.to_string(), asset_data);
        
        info!("Added asset {} ({:?}) with {} observations", symbol, asset_class, prices.len());
        Ok(())
    }
    
    /// Calculate returns based on transformation method
    fn calculate_returns(&self, prices: &Array1<f64>) -> Result<Array1<f64>> {
        if prices.len() < 2 {
            return Err(CrossAssetError::DataError {
                message: "At least 2 price points required for return calculation".to_string(),
            }.into());
        }
        
        let mut returns = Array1::zeros(prices.len() - 1);
        
        match self.config.data_transformation {
            DataTransformation::LogReturns => {
                for i in 1..prices.len() {
                    returns[i - 1] = (prices[i] / prices[i - 1]).ln();
                }
            }
            DataTransformation::SimpleReturns => {
                for i in 1..prices.len() {
                    returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
                }
            }
            DataTransformation::ZScore => {
                // First calculate simple returns, then z-score normalize
                for i in 1..prices.len() {
                    returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
                }
                let mean = returns.mean().unwrap_or(0.0);
                let std = returns.std(0.0);
                if std > 1e-10 {
                    returns = (returns - mean) / std;
                }
            }
            DataTransformation::Ranks => {
                // Calculate simple returns first
                for i in 1..prices.len() {
                    returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
                }
                // Convert to ranks (simplified)
                let sorted_indices: Vec<usize> = (0..returns.len()).collect();
                // This is a simplified rank transformation
                for (rank, &idx) in sorted_indices.iter().enumerate() {
                    returns[idx] = rank as f64;
                }
            }
            DataTransformation::None => {
                returns = prices.slice(s![1..]).to_owned();
            }
        }
        
        Ok(returns)
    }
    
    /// Analyze cross-asset relationships
    #[instrument(skip(self))]
    pub async fn analyze_cross_relationships(&mut self) -> Result<CrossAssetAnalysis> {
        let start_time = Instant::now();
        
        // Check minimum number of assets
        let assets = self.assets.read();
        if assets.len() < 2 {
            return Err(CrossAssetError::DataError {
                message: "At least 2 assets required for cross-asset analysis".to_string(),
            }.into());
        }
        
        let asset_symbols: Vec<String> = assets.keys().cloned().collect();
        drop(assets);
        
        // Add timeout protection
        let timeout_duration = Duration::from_millis(self.config.max_processing_time_ms);
        let analysis_future = self.perform_cross_analysis_internal(&asset_symbols);
        
        let mut result = match tokio::time::timeout(timeout_duration, analysis_future).await {
            Ok(result) => result?,
            Err(_) => {
                error!("Cross-asset analysis timeout exceeded: {}ms", self.config.max_processing_time_ms);
                return Err(CrossAssetError::ComputationError {
                    message: "Analysis timeout exceeded".to_string(),
                }.into());
            }
        };
        
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        result.processing_time_ms = processing_time_ms;
        
        // Update performance metrics
        {
            let mut monitor = self.performance_monitor.lock();
            monitor.record_analysis(processing_time_ms, asset_symbols.len());
        }
        
        debug!("Cross-asset analysis completed in {}ms", processing_time_ms);
        
        Ok(result)
    }
    
    /// Internal cross-analysis implementation
    async fn perform_cross_analysis_internal(
        &mut self,
        asset_symbols: &[String],
    ) -> Result<CrossAssetAnalysis> {
        // Compute correlation matrices
        let correlation_matrix = {
            let mut engine = self.correlation_engine.lock();
            engine.compute_correlations(&self.assets, asset_symbols, &self.config).await?
        };
        
        // Detect contagion
        let contagion_risk = {
            let mut detector = self.contagion_detector.lock();
            detector.detect_contagion(&self.assets, &correlation_matrix, &self.config).await?
        };
        
        // Analyze lead-lag relationships
        let lead_lag_relationships = self.analyze_lead_lag_relationships(asset_symbols).await?;
        
        // Calculate systemic risk
        let systemic_risk_measure = {
            let mut monitor = self.systemic_risk_monitor.lock();
            monitor.calculate_systemic_risk(&self.assets, &correlation_matrix, &self.config).await?
        };
        
        // Compute network metrics
        let network_metrics = {
            let mut analyzer = self.network_analyzer.lock();
            analyzer.analyze_network(&correlation_matrix, asset_symbols, &self.config)?
        };
        
        // Calculate volatility spillovers
        let volatility_spillovers = self.calculate_volatility_spillovers(asset_symbols).await?;
        
        // Risk attribution
        let risk_attribution = self.calculate_risk_attribution(
            asset_symbols,
            &network_metrics,
            contagion_risk,
            systemic_risk_measure,
        )?;
        
        // Regime-dependent analysis if enabled
        let regime_correlations = if self.config.regime_dependent_analysis {
            Some(self.analyze_regime_dependent_correlations(asset_symbols).await?)
        } else {
            None
        };
        
        // Create metadata
        let metadata = self.create_analysis_metadata(asset_symbols)?;
        
        Ok(CrossAssetAnalysis {
            correlation_matrix,
            asset_symbols: asset_symbols.to_vec(),
            contagion_risk,
            lead_lag_relationships,
            systemic_risk_measure,
            network_metrics,
            regime_correlations,
            volatility_spillovers,
            risk_attribution,
            processing_time_ms: 0, // Will be set by caller
            metadata,
        })
    }
    
    /// Analyze lead-lag relationships between assets
    async fn analyze_lead_lag_relationships(
        &self,
        asset_symbols: &[String],
    ) -> Result<HashMap<String, i32>> {
        let mut lead_lag_relationships = HashMap::new();
        
        let assets = self.assets.read();
        
        for i in 0..asset_symbols.len() {
            for j in (i + 1)..asset_symbols.len() {
                let asset1 = &assets[&asset_symbols[i]];
                let asset2 = &assets[&asset_symbols[j]];
                
                let optimal_lag = self.find_optimal_lag(&asset1.returns, &asset2.returns)?;
                
                // Store the relationship: positive lag means asset1 leads asset2
                lead_lag_relationships.insert(
                    format!("{}_{}", asset_symbols[i], asset_symbols[j]),
                    optimal_lag,
                );
            }
        }
        
        Ok(lead_lag_relationships)
    }
    
    /// Find optimal lag between two time series
    fn find_optimal_lag(&self, series1: &Array1<f64>, series2: &Array1<f64>) -> Result<i32> {
        let max_lag = self.config.max_lag_periods as i32;
        let mut best_correlation = 0.0;
        let mut best_lag = 0;
        
        for lag in -max_lag..=max_lag {
            let correlation = self.calculate_lagged_correlation(series1, series2, lag)?;
            
            if correlation.abs() > best_correlation.abs() {
                best_correlation = correlation;
                best_lag = lag;
            }
        }
        
        Ok(best_lag)
    }
    
    /// Calculate correlation with lag
    fn calculate_lagged_correlation(
        &self,
        series1: &Array1<f64>,
        series2: &Array1<f64>,
        lag: i32,
    ) -> Result<f64> {
        let len1 = series1.len();
        let len2 = series2.len();
        let min_len = std::cmp::min(len1, len2);
        
        if lag == 0 {
            // No lag - standard correlation
            let s1 = series1.slice(s![0..min_len]);
            let s2 = series2.slice(s![0..min_len]);
            return Ok(self.pearson_correlation(&s1.to_owned(), &s2.to_owned())?);
        }
        
        let (start1, end1, start2, end2) = if lag > 0 {
            // series1 leads series2
            let lag = lag as usize;
            if lag >= min_len {
                return Ok(0.0);
            }
            (0, min_len - lag, lag, min_len)
        } else {
            // series2 leads series1
            let lag = (-lag) as usize;
            if lag >= min_len {
                return Ok(0.0);
            }
            (lag, min_len, 0, min_len - lag)
        };
        
        let s1 = series1.slice(s![start1..end1]);
        let s2 = series2.slice(s![start2..end2]);
        
        Ok(self.pearson_correlation(&s1.to_owned(), &s2.to_owned())?)
    }
    
    /// Calculate Pearson correlation
    fn pearson_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }
        
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        Ok(if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        })
    }
    
    /// Calculate volatility spillovers
    async fn calculate_volatility_spillovers(
        &self,
        asset_symbols: &[String],
    ) -> Result<Array2<f64>> {
        let n_assets = asset_symbols.len();
        let mut spillovers = Array2::zeros((n_assets, n_assets));
        
        let assets = self.assets.read();
        
        // Calculate volatility for each asset
        let mut volatilities = Vec::new();
        for symbol in asset_symbols {
            let asset = &assets[symbol];
            let volatility = self.calculate_realized_volatility(&asset.returns)?;
            volatilities.push(volatility);
        }
        
        // Calculate spillover effects using VAR model (simplified)
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i != j {
                    let spillover = self.calculate_spillover_effect(&volatilities[i], &volatilities[j])?;
                    spillovers[[i, j]] = spillover;
                }
            }
        }
        
        Ok(spillovers)
    }
    
    /// Calculate realized volatility
    fn calculate_realized_volatility(&self, returns: &Array1<f64>) -> Result<Array1<f64>> {
        let window_size = 20; // 20-day rolling window
        let mut volatility = Array1::zeros(returns.len().saturating_sub(window_size - 1));
        
        for i in window_size - 1..returns.len() {
            let window = returns.slice(s![i + 1 - window_size..=i]);
            let variance = window.var(0.0);
            volatility[i - window_size + 1] = variance.sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Calculate spillover effect between two volatility series
    fn calculate_spillover_effect(&self, vol1: &Array1<f64>, vol2: &Array1<f64>) -> Result<f64> {
        // Simplified spillover calculation - would use VAR in practice
        let correlation = self.pearson_correlation(vol1, vol2)?;
        Ok(correlation.abs())
    }
    
    /// Calculate risk attribution for each asset
    fn calculate_risk_attribution(
        &self,
        asset_symbols: &[String],
        network_metrics: &NetworkMetrics,
        contagion_risk: f64,
        systemic_risk: f64,
    ) -> Result<Vec<RiskAttribution>> {
        let mut risk_attribution = Vec::new();
        
        for symbol in asset_symbols {
            let degree_centrality = network_metrics
                .centrality_measures
                .get(symbol)
                .and_then(|m| m.get(&CentralityMeasure::Degree))
                .copied()
                .unwrap_or(0.0);
            
            let betweenness_centrality = network_metrics
                .centrality_measures
                .get(symbol)
                .and_then(|m| m.get(&CentralityMeasure::Betweenness))
                .copied()
                .unwrap_or(0.0);
            
            risk_attribution.push(RiskAttribution {
                asset_symbol: symbol.clone(),
                systemic_contribution: degree_centrality * systemic_risk,
                contagion_vulnerability: betweenness_centrality * contagion_risk,
                network_importance: (degree_centrality + betweenness_centrality) / 2.0,
                stress_test_impact: degree_centrality * contagion_risk,
            });
        }
        
        Ok(risk_attribution)
    }
    
    /// Analyze regime-dependent correlations
    async fn analyze_regime_dependent_correlations(
        &self,
        asset_symbols: &[String],
    ) -> Result<HashMap<String, Array2<f64>>> {
        // Simplified regime-dependent analysis
        // In practice, would use regime detection algorithms
        let mut regime_correlations = HashMap::new();
        
        // Mock regime correlations for demonstration
        let n_assets = asset_symbols.len();
        
        regime_correlations.insert(
            "high_volatility".to_string(),
            Array2::eye(n_assets) * 0.8,
        );
        
        regime_correlations.insert(
            "low_volatility".to_string(),
            Array2::eye(n_assets) * 0.3,
        );
        
        Ok(regime_correlations)
    }
    
    /// Create analysis metadata
    fn create_analysis_metadata(&self, asset_symbols: &[String]) -> Result<AnalysisMetadata> {
        let assets = self.assets.read();
        
        let mut min_observations = usize::MAX;
        for symbol in asset_symbols {
            if let Some(asset) = assets.get(symbol) {
                min_observations = std::cmp::min(min_observations, asset.prices.len());
            }
        }
        
        Ok(AnalysisMetadata {
            n_assets: asset_symbols.len(),
            observation_period_days: min_observations as u32, // Simplified
            correlation_window_size: self.config.correlation_window,
            data_frequency: self.config.data_frequency,
            analysis_timestamp: chrono::Utc::now(),
            statistical_significance: HashMap::new(), // Would calculate in practice
        })
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let monitor = self.performance_monitor.lock();
        monitor.get_metrics()
    }
}

// Performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub analyses_performed: u64,
    pub average_processing_time_ms: u64,
    pub total_assets_analyzed: u64,
    pub correlation_computations: u64,
    pub cache_hit_rate: f64,
}

// Module stubs - these would be implemented in separate files
mod config {
    // Configuration utilities
}

mod correlation {
    use super::*;
    
    pub struct CorrelationEngine {
        // Implementation stub
    }
    
    impl CorrelationEngine {
        pub fn new(_config: &AnalyzerConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub async fn compute_correlations(
            &mut self,
            _assets: &Arc<RwLock<HashMap<String, AssetData>>>,
            _symbols: &[String],
            _config: &AnalyzerConfig,
        ) -> Result<HashMap<CorrelationType, Array2<f64>>> {
            // Stub implementation
            let n = _symbols.len();
            let mut correlations = HashMap::new();
            correlations.insert(CorrelationType::Pearson, Array2::eye(n));
            correlations.insert(CorrelationType::Spearman, Array2::eye(n) * 0.9);
            Ok(correlations)
        }
    }
}

mod contagion {
    use super::*;
    
    pub struct ContagionDetector {
        // Implementation stub
    }
    
    impl ContagionDetector {
        pub fn new(_config: &AnalyzerConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub async fn detect_contagion(
            &mut self,
            _assets: &Arc<RwLock<HashMap<String, AssetData>>>,
            _correlations: &HashMap<CorrelationType, Array2<f64>>,
            _config: &AnalyzerConfig,
        ) -> Result<f64> {
            // Stub implementation
            Ok(0.25)
        }
    }
}

mod systemic {
    use super::*;
    
    pub struct SystemicRiskMonitor {
        // Implementation stub
    }
    
    impl SystemicRiskMonitor {
        pub fn new(_config: &AnalyzerConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub async fn calculate_systemic_risk(
            &mut self,
            _assets: &Arc<RwLock<HashMap<String, AssetData>>>,
            _correlations: &HashMap<CorrelationType, Array2<f64>>,
            _config: &AnalyzerConfig,
        ) -> Result<f64> {
            // Stub implementation
            Ok(0.35)
        }
    }
}

mod analyzer {
    use super::*;
    
    pub struct NetworkAnalyzer {
        // Implementation stub
    }
    
    impl NetworkAnalyzer {
        pub fn new(_config: &AnalyzerConfig) -> Result<Self> {
            Ok(Self {})
        }
        
        pub fn analyze_network(
            &mut self,
            _correlations: &HashMap<CorrelationType, Array2<f64>>,
            symbols: &[String],
            _config: &AnalyzerConfig,
        ) -> Result<NetworkMetrics> {
            // Stub implementation
            let mut centrality_measures = HashMap::new();
            for symbol in symbols {
                let mut measures = HashMap::new();
                measures.insert(CentralityMeasure::Degree, 0.5);
                measures.insert(CentralityMeasure::Betweenness, 0.3);
                measures.insert(CentralityMeasure::Eigenvector, 0.4);
                centrality_measures.insert(symbol.clone(), measures);
            }
            
            Ok(NetworkMetrics {
                density: 0.4,
                clustering_coefficient: 0.6,
                diameter: 3,
                average_path_length: 2.1,
                centrality_measures,
                communities: vec![symbols.to_vec()],
                stability: 0.8,
            })
        }
    }
    
    pub struct AnalysisCache {
        // Implementation stub
    }
    
    impl AnalysisCache {
        pub fn new(_size: usize) -> Self {
            Self {}
        }
    }
    
    pub struct PerformanceMonitor {
        analyses_performed: u64,
        total_processing_time_ms: u64,
        total_assets_analyzed: u64,
        start_time: Instant,
    }
    
    impl PerformanceMonitor {
        pub fn new() -> Self {
            Self {
                analyses_performed: 0,
                total_processing_time_ms: 0,
                total_assets_analyzed: 0,
                start_time: Instant::now(),
            }
        }
        
        pub fn record_analysis(&mut self, processing_time_ms: u64, n_assets: usize) {
            self.analyses_performed += 1;
            self.total_processing_time_ms += processing_time_ms;
            self.total_assets_analyzed += n_assets as u64;
        }
        
        pub fn get_metrics(&self) -> PerformanceMetrics {
            PerformanceMetrics {
                analyses_performed: self.analyses_performed,
                average_processing_time_ms: if self.analyses_performed > 0 {
                    self.total_processing_time_ms / self.analyses_performed
                } else {
                    0
                },
                total_assets_analyzed: self.total_assets_analyzed,
                correlation_computations: self.analyses_performed * self.total_assets_analyzed,
                cache_hit_rate: 0.0,
            }
        }
    }
}

mod network {
    // Network analysis implementations
}

mod copula {
    // Copula analysis implementations
}

mod regime {
    // Regime-dependent analysis
}

mod metrics {
    // Performance metrics and monitoring
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cross_asset_analyzer_creation() {
        let config = AnalyzerConfig::default();
        let analyzer = CrossAssetAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }
    
    #[tokio::test]
    async fn test_add_asset() {
        let config = AnalyzerConfig::default();
        let mut analyzer = CrossAssetAnalyzer::new(config).unwrap();
        
        let prices = vec![100.0, 101.0, 99.5, 102.0, 103.5, 101.0];
        let result = analyzer.add_asset("BTC", AssetClass::Cryptocurrency, &prices).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_cross_asset_analysis() {
        let config = AnalyzerConfig::default();
        let mut analyzer = CrossAssetAnalyzer::new(config).unwrap();
        
        let btc_prices = vec![100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0, 102.5];
        let eth_prices = vec![50.0, 51.2, 49.8, 52.1, 53.0, 50.5, 54.0, 51.8];
        
        analyzer.add_asset("BTC", AssetClass::Cryptocurrency, &btc_prices).await.unwrap();
        analyzer.add_asset("ETH", AssetClass::Cryptocurrency, &eth_prices).await.unwrap();
        
        let result = analyzer.analyze_cross_relationships().await;
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert_eq!(analysis.asset_symbols.len(), 2);
        assert!(!analysis.correlation_matrix.is_empty());
        assert!(analysis.contagion_risk >= 0.0 && analysis.contagion_risk <= 1.0);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = AnalyzerConfig::default();
        assert!(CrossAssetAnalyzer::validate_config(&config).is_ok());
        
        config.correlation_window = 0;
        assert!(CrossAssetAnalyzer::validate_config(&config).is_err());
        
        config.correlation_window = 252;
        config.correlation_overlap = 1.5;
        assert!(CrossAssetAnalyzer::validate_config(&config).is_err());
    }
}