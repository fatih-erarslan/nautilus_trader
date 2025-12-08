//! Market regime detection module using machine learning
//! 
//! Implements advanced algorithms for classifying market conditions and detecting
//! regime changes using ensemble methods, time series analysis, and volatility modeling.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::ml,
};
use ndarray::{Array1, Array2, ArrayView1};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use rayon::prelude::*;
use tracing::{info, debug, warn};

/// Market regime detection and classification engine
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    config: RegimeConfig,
    feature_extractor: FeatureExtractor,
    ensemble_classifier: EnsembleClassifier,
    volatility_detector: VolatilityRegimeDetector,
    regime_history: VecDeque<RegimeSnapshot>,
    transition_matrix: HashMap<(MarketRegime, MarketRegime), f64>,
}

#[derive(Debug, Clone)]
pub struct RegimeConfig {
    pub lookback_window: usize,
    pub feature_count: usize,
    pub confidence_threshold: f64,
    pub regime_stability_threshold: usize,
    pub volatility_windows: Vec<usize>,
    pub trend_threshold: f64,
    pub volume_threshold_multiplier: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            lookback_window: 200,
            feature_count: 25,
            confidence_threshold: 0.75,
            regime_stability_threshold: 5,
            volatility_windows: vec![10, 20, 50, 100],
            trend_threshold: 0.02,
            volume_threshold_multiplier: 1.5,
        }
    }
}

impl RegimeDetector {
    pub fn new(config: &Config) -> Result<Self> {
        let regime_config = RegimeConfig::default();
        
        Ok(Self {
            config: regime_config.clone(),
            feature_extractor: FeatureExtractor::new(&regime_config)?,
            ensemble_classifier: EnsembleClassifier::new(&regime_config)?,
            volatility_detector: VolatilityRegimeDetector::new(&regime_config)?,
            regime_history: VecDeque::with_capacity(1000),
            transition_matrix: HashMap::new(),
        })
    }
    
    /// Detect current market regime
    pub async fn detect_regime(&self, data: &MarketData) -> Result<RegimeInfo> {
        let start_time = std::time::Instant::now();
        debug!("Starting regime detection for {}", data.symbol);
        
        // Extract features from market data
        let features = self.feature_extractor.extract_features(data).await?;
        
        // Parallel regime classification
        let (market_regime, volatility_regime, confidence) = tokio::try_join!(
            self.classify_market_regime(&features),
            self.detect_volatility_regime(data),
            self.calculate_regime_confidence(&features, data)
        )?;
        
        // Calculate transition probabilities
        let transition_probability = self.calculate_transition_probabilities(&market_regime)?;
        
        // Get previous regime from history
        let previous_regime = self.regime_history.back().map(|r| r.regime.clone());
        
        // Calculate regime duration
        let regime_duration = self.calculate_regime_duration(&market_regime);
        
        let regime_info = RegimeInfo {
            current_regime: market_regime.clone(),
            previous_regime,
            confidence,
            regime_duration,
            transition_probability,
            volatility_regime,
        };
        
        // Update regime history
        self.update_regime_history(&market_regime, confidence).await?;
        
        let processing_time = start_time.elapsed();
        debug!("Regime detection completed in {:?}", processing_time);
        
        Ok(regime_info)
    }
    
    /// Classify market regime using ensemble methods
    async fn classify_market_regime(&self, features: &MarketFeatures) -> Result<MarketRegime> {
        self.ensemble_classifier.classify(features).await
    }
    
    /// Detect volatility regime
    async fn detect_volatility_regime(&self, data: &MarketData) -> Result<VolatilityRegime> {
        self.volatility_detector.detect_regime(data).await
    }
    
    /// Calculate confidence in regime classification
    async fn calculate_regime_confidence(&self, features: &MarketFeatures, data: &MarketData) -> Result<f64> {
        // Multi-factor confidence calculation
        let classification_confidence = self.ensemble_classifier.get_classification_confidence(features).await?;
        let stability_score = self.calculate_regime_stability_score(data)?;
        let consensus_score = self.calculate_ensemble_consensus(features).await?;
        
        // Weighted average of confidence factors
        let weights = [0.4, 0.3, 0.3]; // classification, stability, consensus
        let scores = [classification_confidence, stability_score, consensus_score];
        
        let weighted_confidence = weights.iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>();
            
        Ok(weighted_confidence.clamp(0.0, 1.0))
    }
    
    /// Calculate regime stability score based on recent consistency
    fn calculate_regime_stability_score(&self, data: &MarketData) -> Result<f64> {
        if self.regime_history.len() < self.config.regime_stability_threshold {
            return Ok(0.5); // Neutral score for insufficient history
        }
        
        let recent_regimes: Vec<&MarketRegime> = self.regime_history
            .iter()
            .rev()
            .take(self.config.regime_stability_threshold)
            .map(|snapshot| &snapshot.regime)
            .collect();
        
        // Calculate consistency (same regime frequency)
        let most_common_regime = self.find_most_common_regime(&recent_regimes);
        let consistency = recent_regimes.iter()
            .filter(|&&r| r == most_common_regime)
            .count() as f64 / recent_regimes.len() as f64;
        
        Ok(consistency)
    }
    
    /// Find the most common regime in recent history
    fn find_most_common_regime(&self, regimes: &[&MarketRegime]) -> &MarketRegime {
        let mut counts = HashMap::new();
        for regime in regimes {
            *counts.entry(*regime).or_insert(0) += 1;
        }
        
        counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(regime, _)| regime)
            .unwrap_or(&MarketRegime::Sideways)
    }
    
    /// Calculate ensemble consensus score
    async fn calculate_ensemble_consensus(&self, features: &MarketFeatures) -> Result<f64> {
        self.ensemble_classifier.get_consensus_score(features).await
    }
    
    /// Calculate transition probabilities to other regimes
    fn calculate_transition_probabilities(&self, current_regime: &MarketRegime) -> Result<HashMap<MarketRegime, f64>> {
        let mut probabilities = HashMap::new();
        
        // Initialize with uniform probabilities if no history
        if self.transition_matrix.is_empty() {
            let regimes = [
                MarketRegime::Bull,
                MarketRegime::Bear,
                MarketRegime::Sideways,
                MarketRegime::HighVolatility,
                MarketRegime::LowVolatility,
                MarketRegime::TrendingUp,
                MarketRegime::TrendingDown,
                MarketRegime::Accumulation,
                MarketRegime::Distribution,
                MarketRegime::Breakout,
            ];
            
            let uniform_prob = 1.0 / regimes.len() as f64;
            for regime in &regimes {
                probabilities.insert(regime.clone(), uniform_prob);
            }
            
            return Ok(probabilities);
        }
        
        // Calculate empirical transition probabilities
        let total_transitions: f64 = self.transition_matrix
            .iter()
            .filter(|((from, _), _)| from == current_regime)
            .map(|(_, count)| count)
            .sum();
        
        if total_transitions > 0.0 {
            for ((from, to), count) in &self.transition_matrix {
                if from == current_regime {
                    probabilities.insert(to.clone(), count / total_transitions);
                }
            }
        }
        
        Ok(probabilities)
    }
    
    /// Calculate how long the current regime has been active
    fn calculate_regime_duration(&self, current_regime: &MarketRegime) -> Duration {
        if self.regime_history.is_empty() {
            return Duration::zero();
        }
        
        let mut duration_periods = 0;
        for snapshot in self.regime_history.iter().rev() {
            if &snapshot.regime == current_regime {
                duration_periods += 1;
            } else {
                break;
            }
        }
        
        Duration::minutes(duration_periods as i64) // Assuming 1-minute periods
    }
    
    /// Update regime history and transition matrix
    async fn update_regime_history(&self, regime: &MarketRegime, confidence: f64) -> Result<()> {
        let snapshot = RegimeSnapshot {
            regime: regime.clone(),
            confidence,
            timestamp: Utc::now(),
        };
        
        // Update transition matrix if we have a previous regime
        if let Some(previous_snapshot) = self.regime_history.back() {
            let transition = (previous_snapshot.regime.clone(), regime.clone());
            *self.transition_matrix.entry(transition).or_insert(0.0) += 1.0;
        }
        
        // Add to history (this would need interior mutability in real implementation)
        // self.regime_history.push_back(snapshot);
        
        // Keep only recent history
        // if self.regime_history.len() > 1000 {
        //     self.regime_history.pop_front();
        // }
        
        Ok(())
    }
    
    /// Update model based on feedback
    pub async fn update_model(&mut self, feedback: &RegimeFeedback) -> Result<()> {
        info!("Updating regime detection model with feedback");
        
        // Calculate accuracy and adjust confidence threshold
        let total_classifications = feedback.correct_classifications + feedback.incorrect_classifications;
        if total_classifications > 0 {
            let accuracy = feedback.correct_classifications as f64 / total_classifications as f64;
            
            if accuracy < 0.7 {
                // Low accuracy, be more conservative
                self.config.confidence_threshold = (self.config.confidence_threshold + 0.05).min(0.9);
            } else if accuracy > 0.85 {
                // High accuracy, can be more aggressive
                self.config.confidence_threshold = (self.config.confidence_threshold - 0.02).max(0.6);
            }
        }
        
        // Update ensemble classifier with new training data
        self.ensemble_classifier.retrain(feedback).await?;
        
        // Apply model updates
        for (key, value) in &feedback.model_updates {
            match key.as_str() {
                "lookback_window" => {
                    if let Some(window) = value.as_u64() {
                        self.config.lookback_window = window as usize;
                    }
                }
                "trend_threshold" => {
                    if let Some(threshold) = value.as_f64() {
                        self.config.trend_threshold = threshold;
                    }
                }
                "volume_threshold_multiplier" => {
                    if let Some(multiplier) = value.as_f64() {
                        self.config.volume_threshold_multiplier = multiplier;
                    }
                }
                _ => {}
            }
        }
        
        info!("Updated regime detector configuration: {:?}", self.config);
        Ok(())
    }
}

/// Feature extraction for regime classification
#[derive(Debug, Clone)]
struct FeatureExtractor {
    config: RegimeConfig,
}

impl FeatureExtractor {
    fn new(config: &RegimeConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    /// Extract comprehensive features from market data
    async fn extract_features(&self, data: &MarketData) -> Result<MarketFeatures> {
        if data.prices.len() < self.config.lookback_window {
            return Err(AnalysisError::InsufficientData(
                format!("Need {} data points, got {}", self.config.lookback_window, data.prices.len())
            ));
        }
        
        let (price_features, volume_features, volatility_features) = tokio::try_join!(
            self.extract_price_features(data),
            self.extract_volume_features(data),
            self.extract_volatility_features(data)
        )?;
        
        let technical_features = self.extract_technical_features(data)?;
        let momentum_features = self.extract_momentum_features(data)?;
        let microstructure_features = self.extract_microstructure_features(data)?;
        
        Ok(MarketFeatures {
            price_features,
            volume_features,
            volatility_features,
            technical_features,
            momentum_features,
            microstructure_features,
            timestamp: Utc::now(),
        })
    }
    
    /// Extract price-based features
    async fn extract_price_features(&self, data: &MarketData) -> Result<PriceFeatures> {
        let returns = data.get_returns();
        let prices = &data.prices;
        
        // Trend analysis
        let short_term_trend = self.calculate_trend(&prices[prices.len()-20..]);
        let medium_term_trend = self.calculate_trend(&prices[prices.len()-50..]);
        let long_term_trend = self.calculate_trend(&prices[prices.len()-100..]);
        
        // Support and resistance levels
        let (support_levels, resistance_levels) = self.calculate_support_resistance(prices);
        
        // Price momentum
        let price_momentum = self.calculate_price_momentum(prices);
        
        // Moving averages and crossovers
        let sma_20 = self.calculate_sma(prices, 20);
        let sma_50 = self.calculate_sma(prices, 50);
        let ema_12 = self.calculate_ema(prices, 12);
        let ema_26 = self.calculate_ema(prices, 26);
        
        Ok(PriceFeatures {
            short_term_trend,
            medium_term_trend,
            long_term_trend,
            support_levels,
            resistance_levels,
            price_momentum,
            sma_crossover: if sma_20 > sma_50 { 1.0 } else { -1.0 },
            ema_crossover: if ema_12 > ema_26 { 1.0 } else { -1.0 },
            price_position: self.calculate_price_position(prices),
            breakout_strength: self.calculate_breakout_strength(prices),
        })
    }
    
    /// Extract volume-based features
    async fn extract_volume_features(&self, data: &MarketData) -> Result<VolumeFeatures> {
        let volumes = &data.volumes;
        
        let average_volume = volumes.mean();
        let volume_trend = self.calculate_trend(volumes);
        let volume_spikes = self.detect_volume_spikes(volumes);
        let volume_profile_strength = self.calculate_volume_profile_strength(data);
        let on_balance_volume = self.calculate_obv(data);
        
        Ok(VolumeFeatures {
            average_volume,
            volume_trend,
            volume_spikes,
            volume_profile_strength,
            on_balance_volume,
            volume_price_correlation: self.calculate_volume_price_correlation(data),
        })
    }
    
    /// Extract volatility-based features
    async fn extract_volatility_features(&self, data: &MarketData) -> Result<VolatilityFeatures> {
        let returns = data.get_returns();
        
        let mut volatilities = Vec::new();
        for &window in &self.config.volatility_windows {
            if returns.len() >= window {
                let recent_returns = &returns[returns.len()-window..];
                let vol = recent_returns.std_dev();
                volatilities.push(vol);
            }
        }
        
        let volatility_clustering = self.calculate_volatility_clustering(&returns);
        let volatility_persistence = self.calculate_volatility_persistence(&returns);
        let volatility_regime_probability = self.estimate_volatility_regime_probability(&returns);
        
        Ok(VolatilityFeatures {
            short_term_volatility: volatilities.get(0).copied().unwrap_or(0.0),
            medium_term_volatility: volatilities.get(1).copied().unwrap_or(0.0),
            long_term_volatility: volatilities.get(2).copied().unwrap_or(0.0),
            volatility_clustering,
            volatility_persistence,
            volatility_regime_probability,
        })
    }
    
    /// Extract technical indicator features
    fn extract_technical_features(&self, data: &MarketData) -> Result<TechnicalFeatures> {
        let prices = &data.prices;
        
        let rsi = self.calculate_rsi(prices, 14);
        let macd = self.calculate_macd(prices);
        let bollinger_position = self.calculate_bollinger_position(prices);
        let stochastic = self.calculate_stochastic(data);
        let williams_r = self.calculate_williams_r(data);
        let atr = self.calculate_atr(data);
        
        Ok(TechnicalFeatures {
            rsi,
            macd: macd.0,
            macd_signal: macd.1,
            macd_histogram: macd.2,
            bollinger_position,
            stochastic,
            williams_r,
            atr,
        })
    }
    
    /// Extract momentum features
    fn extract_momentum_features(&self, data: &MarketData) -> Result<MomentumFeatures> {
        let prices = &data.prices;
        
        let price_momentum_1d = self.calculate_momentum(prices, 1);
        let price_momentum_3d = self.calculate_momentum(prices, 3);
        let price_momentum_7d = self.calculate_momentum(prices, 7);
        let volume_momentum = self.calculate_momentum(&data.volumes, 3);
        let momentum_acceleration = self.calculate_momentum_acceleration(prices);
        
        Ok(MomentumFeatures {
            price_momentum_1d,
            price_momentum_3d,
            price_momentum_7d,
            volume_momentum,
            momentum_acceleration,
            momentum_divergence: self.calculate_momentum_divergence(data),
        })
    }
    
    /// Extract market microstructure features
    fn extract_microstructure_features(&self, data: &MarketData) -> Result<MicrostructureFeatures> {
        let bid_ask_spread = if let Some(ref order_book) = data.order_book {
            if !order_book.asks.is_empty() && !order_book.bids.is_empty() {
                order_book.asks[0].price - order_book.bids[0].price
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let trade_intensity = data.trades.len() as f64;
        let order_flow_imbalance = self.calculate_order_flow_imbalance(data);
        let market_depth = self.calculate_market_depth(data);
        
        Ok(MicrostructureFeatures {
            bid_ask_spread,
            trade_intensity,
            order_flow_imbalance,
            market_depth,
            liquidity_score: self.calculate_liquidity_score(data),
        })
    }
    
    // Helper calculation methods
    
    fn calculate_trend(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let first = prices[0];
        let last = prices[prices.len() - 1];
        (last - first) / first
    }
    
    fn calculate_support_resistance(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Simplified support/resistance calculation
        let window = 20.min(prices.len());
        let mut support_levels = Vec::new();
        let mut resistance_levels = Vec::new();
        
        for i in window..prices.len() {
            let local_min = prices[i-window..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let local_max = prices[i-window..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            if prices[i] == local_min {
                support_levels.push(prices[i]);
            }
            if prices[i] == local_max {
                resistance_levels.push(prices[i]);
            }
        }
        
        (support_levels, resistance_levels)
    }
    
    fn calculate_price_momentum(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }
        
        let recent = &prices[prices.len()-5..];
        let older = &prices[prices.len()-10..prices.len()-5];
        
        let recent_avg = recent.mean();
        let older_avg = older.mean();
        
        (recent_avg - older_avg) / older_avg
    }
    
    fn calculate_sma(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.mean();
        }
        
        prices[prices.len()-period..].mean()
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.mean();
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        
        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        
        ema
    }
    
    fn calculate_price_position(&self, prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.5;
        }
        
        let current_price = prices[prices.len() - 1];
        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_price == min_price {
            return 0.5;
        }
        
        (current_price - min_price) / (max_price - min_price)
    }
    
    fn calculate_breakout_strength(&self, prices: &[f64]) -> f64 {
        // Simplified breakout strength calculation
        if prices.len() < 20 {
            return 0.0;
        }
        
        let recent_high = prices[prices.len()-5..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let resistance = prices[prices.len()-20..prices.len()-5].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if resistance > 0.0 {
            (recent_high - resistance) / resistance
        } else {
            0.0
        }
    }
    
    fn detect_volume_spikes(&self, volumes: &[f64]) -> f64 {
        if volumes.len() < 20 {
            return 0.0;
        }
        
        let current_volume = volumes[volumes.len() - 1];
        let avg_volume = volumes[volumes.len()-20..volumes.len()-1].mean();
        
        if avg_volume > 0.0 {
            current_volume / avg_volume
        } else {
            1.0
        }
    }
    
    fn calculate_volume_profile_strength(&self, data: &MarketData) -> f64 {
        // Simplified volume profile strength
        if data.volumes.is_empty() {
            return 0.0;
        }
        
        let total_volume: f64 = data.volumes.iter().sum();
        let max_volume = data.volumes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if total_volume > 0.0 {
            max_volume / (total_volume / data.volumes.len() as f64)
        } else {
            0.0
        }
    }
    
    fn calculate_obv(&self, data: &MarketData) -> f64 {
        if data.prices.len() < 2 || data.volumes.is_empty() {
            return 0.0;
        }
        
        let mut obv = 0.0;
        
        for i in 1..data.prices.len().min(data.volumes.len()) {
            if data.prices[i] > data.prices[i - 1] {
                obv += data.volumes[i];
            } else if data.prices[i] < data.prices[i - 1] {
                obv -= data.volumes[i];
            }
        }
        
        obv
    }
    
    fn calculate_volume_price_correlation(&self, data: &MarketData) -> f64 {
        if data.prices.len() != data.volumes.len() || data.prices.len() < 2 {
            return 0.0;
        }
        
        let price_changes: Vec<f64> = data.prices.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
            
        if price_changes.len() != data.volumes.len() - 1 {
            return 0.0;
        }
        
        let volumes = &data.volumes[1..];
        
        // Calculate correlation coefficient
        let price_mean = price_changes.mean();
        let volume_mean = volumes.mean();
        
        let numerator: f64 = price_changes.iter()
            .zip(volumes.iter())
            .map(|(p, v)| (p - price_mean) * (v - volume_mean))
            .sum();
            
        let price_sq_sum: f64 = price_changes.iter()
            .map(|p| (p - price_mean).powi(2))
            .sum();
            
        let volume_sq_sum: f64 = volumes.iter()
            .map(|v| (v - volume_mean).powi(2))
            .sum();
        
        let denominator = (price_sq_sum * volume_sq_sum).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn calculate_volatility_clustering(&self, returns: &[f64]) -> f64 {
        if returns.len() < 10 {
            return 0.0;
        }
        
        // Calculate autocorrelation of squared returns
        let squared_returns: Vec<f64> = returns.iter().map(|r| r.powi(2)).collect();
        let mean_sq = squared_returns.mean();
        
        let lag1_correlation = if squared_returns.len() > 1 {
            let numerator: f64 = squared_returns[1..].iter()
                .zip(squared_returns[..squared_returns.len()-1].iter())
                .map(|(x, y)| (x - mean_sq) * (y - mean_sq))
                .sum();
                
            let denominator: f64 = squared_returns.iter()
                .map(|x| (x - mean_sq).powi(2))
                .sum();
                
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        lag1_correlation
    }
    
    fn calculate_volatility_persistence(&self, returns: &[f64]) -> f64 {
        // Simplified volatility persistence using ARCH effect
        self.calculate_volatility_clustering(returns)
    }
    
    fn estimate_volatility_regime_probability(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 0.5;
        }
        
        let recent_vol = returns[returns.len()-10..].std_dev();
        let historical_vol = returns[returns.len()-20..returns.len()-10].std_dev();
        
        if historical_vol > 0.0 {
            (recent_vol / historical_vol).min(2.0) / 2.0
        } else {
            0.5
        }
    }
    
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in prices.len()-period..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64, f64) {
        let ema12 = self.calculate_ema(prices, 12);
        let ema26 = self.calculate_ema(prices, 26);
        let macd_line = ema12 - ema26;
        
        // Simplified signal line (would need historical MACD values for proper EMA)
        let signal_line = macd_line * 0.9; // Approximation
        let histogram = macd_line - signal_line;
        
        (macd_line, signal_line, histogram)
    }
    
    fn calculate_bollinger_position(&self, prices: &[f64]) -> f64 {
        if prices.len() < 20 {
            return 0.5;
        }
        
        let period = 20;
        let recent_prices = &prices[prices.len()-period..];
        let sma = recent_prices.mean();
        let std_dev = recent_prices.std_dev();
        let current_price = prices[prices.len() - 1];
        
        if std_dev == 0.0 {
            return 0.5;
        }
        
        let upper_band = sma + (2.0 * std_dev);
        let lower_band = sma - (2.0 * std_dev);
        
        (current_price - lower_band) / (upper_band - lower_band)
    }
    
    fn calculate_stochastic(&self, data: &MarketData) -> f64 {
        if data.prices.len() < 14 {
            return 50.0;
        }
        
        let period = 14;
        let recent_prices = &data.prices[data.prices.len()-period..];
        let current_price = data.prices[data.prices.len() - 1];
        let lowest_low = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let highest_high = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if highest_high == lowest_low {
            return 50.0;
        }
        
        ((current_price - lowest_low) / (highest_high - lowest_low)) * 100.0
    }
    
    fn calculate_williams_r(&self, data: &MarketData) -> f64 {
        100.0 - self.calculate_stochastic(data)
    }
    
    fn calculate_atr(&self, data: &MarketData) -> f64 {
        if data.prices.len() < 2 {
            return 0.0;
        }
        
        // Simplified ATR using price changes
        let price_changes: Vec<f64> = data.prices.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
            
        if price_changes.is_empty() {
            return 0.0;
        }
        
        price_changes.mean()
    }
    
    fn calculate_momentum(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period + 1 {
            return 0.0;
        }
        
        let current = values[values.len() - 1];
        let previous = values[values.len() - 1 - period];
        
        if previous != 0.0 {
            (current - previous) / previous
        } else {
            0.0
        }
    }
    
    fn calculate_momentum_acceleration(&self, prices: &[f64]) -> f64 {
        if prices.len() < 6 {
            return 0.0;
        }
        
        let momentum_1 = self.calculate_momentum(prices, 1);
        let momentum_3 = self.calculate_momentum(&prices[..prices.len()-2], 1);
        
        momentum_1 - momentum_3
    }
    
    fn calculate_momentum_divergence(&self, data: &MarketData) -> f64 {
        // Simplified momentum divergence
        if data.prices.len() < 10 || data.volumes.len() < 10 {
            return 0.0;
        }
        
        let price_momentum = self.calculate_momentum(&data.prices, 5);
        let volume_momentum = self.calculate_momentum(&data.volumes, 5);
        
        price_momentum - volume_momentum
    }
    
    fn calculate_order_flow_imbalance(&self, data: &MarketData) -> f64 {
        if data.trades.is_empty() {
            return 0.0;
        }
        
        let (buy_volume, sell_volume) = data.trades.iter().fold((0.0, 0.0), |(buy, sell), trade| {
            match trade.side {
                TradeSide::Buy => (buy + trade.quantity, sell),
                TradeSide::Sell => (buy, sell + trade.quantity),
            }
        });
        
        let total_volume = buy_volume + sell_volume;
        if total_volume > 0.0 {
            (buy_volume - sell_volume) / total_volume
        } else {
            0.0
        }
    }
    
    fn calculate_market_depth(&self, data: &MarketData) -> f64 {
        if let Some(ref order_book) = data.order_book {
            let bid_depth: f64 = order_book.bids.iter().map(|level| level.quantity).sum();
            let ask_depth: f64 = order_book.asks.iter().map(|level| level.quantity).sum();
            bid_depth + ask_depth
        } else {
            data.volumes.iter().sum::<f64>() / data.volumes.len() as f64
        }
    }
    
    fn calculate_liquidity_score(&self, data: &MarketData) -> f64 {
        let market_depth = self.calculate_market_depth(data);
        let trade_intensity = data.trades.len() as f64;
        let volume_consistency = 1.0 / (1.0 + data.volumes.std_dev());
        
        (market_depth.ln() + trade_intensity.ln() + volume_consistency) / 3.0
    }
}

/// Ensemble classifier for regime detection
#[derive(Debug, Clone)]
struct EnsembleClassifier {
    config: RegimeConfig,
    // In a real implementation, these would be actual ML models
    // random_forest: Option<RandomForestClassifier<f64>>,
    // logistic_regression: Option<LogisticRegression<f64>>,
    // decision_tree: Option<DecisionTreeClassifier<f64>>,
}

impl EnsembleClassifier {
    fn new(config: &RegimeConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            // Initialize ML models here
        })
    }
    
    async fn classify(&self, features: &MarketFeatures) -> Result<MarketRegime> {
        // Simplified classification logic
        // In a real implementation, this would use trained ML models
        
        let trend_score = features.price_features.short_term_trend;
        let volatility_score = features.volatility_features.short_term_volatility;
        let volume_score = features.volume_features.volume_trend;
        
        // Simple rule-based classification for demonstration
        if volatility_score > 0.05 {
            Ok(MarketRegime::HighVolatility)
        } else if trend_score > 0.02 {
            Ok(MarketRegime::TrendingUp)
        } else if trend_score < -0.02 {
            Ok(MarketRegime::TrendingDown)
        } else if volume_score > 1.5 {
            Ok(MarketRegime::Breakout)
        } else {
            Ok(MarketRegime::Sideways)
        }
    }
    
    async fn get_classification_confidence(&self, features: &MarketFeatures) -> Result<f64> {
        // Calculate confidence based on feature strength
        let trend_strength = features.price_features.short_term_trend.abs();
        let volatility_consistency = 1.0 / (1.0 + features.volatility_features.short_term_volatility);
        let volume_confirmation = features.volume_features.volume_spikes.min(2.0) / 2.0;
        
        let confidence = (trend_strength + volatility_consistency + volume_confirmation) / 3.0;
        Ok(confidence.clamp(0.0, 1.0))
    }
    
    async fn get_consensus_score(&self, features: &MarketFeatures) -> Result<f64> {
        // Calculate how much different indicators agree
        let technical_consensus = self.calculate_technical_consensus(features)?;
        let momentum_consensus = self.calculate_momentum_consensus(features)?;
        let volume_consensus = self.calculate_volume_consensus(features)?;
        
        Ok((technical_consensus + momentum_consensus + volume_consensus) / 3.0)
    }
    
    fn calculate_technical_consensus(&self, features: &MarketFeatures) -> Result<f64> {
        let tech = &features.technical_features;
        
        // Count bullish vs bearish signals
        let mut bullish_signals = 0;
        let mut total_signals = 0;
        
        // RSI signal
        if tech.rsi > 30.0 && tech.rsi < 70.0 {
            total_signals += 1;
            if tech.rsi > 50.0 {
                bullish_signals += 1;
            }
        }
        
        // MACD signal
        total_signals += 1;
        if tech.macd > tech.macd_signal {
            bullish_signals += 1;
        }
        
        // Bollinger position
        total_signals += 1;
        if tech.bollinger_position > 0.5 {
            bullish_signals += 1;
        }
        
        if total_signals > 0 {
            Ok(bullish_signals as f64 / total_signals as f64)
        } else {
            Ok(0.5)
        }
    }
    
    fn calculate_momentum_consensus(&self, features: &MarketFeatures) -> Result<f64> {
        let momentum = &features.momentum_features;
        
        let mut positive_momentum = 0;
        let mut total_momentum = 0;
        
        if momentum.price_momentum_1d.abs() > 0.001 {
            total_momentum += 1;
            if momentum.price_momentum_1d > 0.0 {
                positive_momentum += 1;
            }
        }
        
        if momentum.price_momentum_3d.abs() > 0.001 {
            total_momentum += 1;
            if momentum.price_momentum_3d > 0.0 {
                positive_momentum += 1;
            }
        }
        
        if momentum.volume_momentum.abs() > 0.001 {
            total_momentum += 1;
            if momentum.volume_momentum > 0.0 {
                positive_momentum += 1;
            }
        }
        
        if total_momentum > 0 {
            Ok(positive_momentum as f64 / total_momentum as f64)
        } else {
            Ok(0.5)
        }
    }
    
    fn calculate_volume_consensus(&self, features: &MarketFeatures) -> Result<f64> {
        let volume = &features.volume_features;
        
        // Volume confirmation of price moves
        let volume_price_alignment = if volume.volume_price_correlation.abs() > 0.3 {
            if volume.volume_price_correlation > 0.0 { 1.0 } else { 0.0 }
        } else {
            0.5
        };
        
        let volume_trend_strength = (volume.volume_trend.abs().min(1.0) + 1.0) / 2.0;
        
        Ok((volume_price_alignment + volume_trend_strength) / 2.0)
    }
    
    async fn retrain(&mut self, feedback: &RegimeFeedback) -> Result<()> {
        // Implement model retraining based on feedback
        // This would update the ML models with new training data
        Ok(())
    }
}

/// Volatility regime detector
#[derive(Debug, Clone)]
struct VolatilityRegimeDetector {
    config: RegimeConfig,
}

impl VolatilityRegimeDetector {
    fn new(config: &RegimeConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn detect_regime(&self, data: &MarketData) -> Result<VolatilityRegime> {
        let returns = data.get_returns();
        
        if returns.len() < 20 {
            return Ok(VolatilityRegime::Medium);
        }
        
        let current_vol = returns[returns.len()-10..].std_dev();
        let historical_vol = returns.std_dev();
        
        let vol_ratio = if historical_vol > 0.0 {
            current_vol / historical_vol
        } else {
            1.0
        };
        
        let regime = match vol_ratio {
            x if x < 0.5 => VolatilityRegime::Low,
            x if x < 1.5 => VolatilityRegime::Medium,
            x if x < 3.0 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        };
        
        Ok(regime)
    }
}

/// Market features for regime classification
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    pub price_features: PriceFeatures,
    pub volume_features: VolumeFeatures,
    pub volatility_features: VolatilityFeatures,
    pub technical_features: TechnicalFeatures,
    pub momentum_features: MomentumFeatures,
    pub microstructure_features: MicrostructureFeatures,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PriceFeatures {
    pub short_term_trend: f64,
    pub medium_term_trend: f64,
    pub long_term_trend: f64,
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub price_momentum: f64,
    pub sma_crossover: f64,
    pub ema_crossover: f64,
    pub price_position: f64,
    pub breakout_strength: f64,
}

#[derive(Debug, Clone)]
pub struct VolumeFeatures {
    pub average_volume: f64,
    pub volume_trend: f64,
    pub volume_spikes: f64,
    pub volume_profile_strength: f64,
    pub on_balance_volume: f64,
    pub volume_price_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct VolatilityFeatures {
    pub short_term_volatility: f64,
    pub medium_term_volatility: f64,
    pub long_term_volatility: f64,
    pub volatility_clustering: f64,
    pub volatility_persistence: f64,
    pub volatility_regime_probability: f64,
}

#[derive(Debug, Clone)]
pub struct TechnicalFeatures {
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_histogram: f64,
    pub bollinger_position: f64,
    pub stochastic: f64,
    pub williams_r: f64,
    pub atr: f64,
}

#[derive(Debug, Clone)]
pub struct MomentumFeatures {
    pub price_momentum_1d: f64,
    pub price_momentum_3d: f64,
    pub price_momentum_7d: f64,
    pub volume_momentum: f64,
    pub momentum_acceleration: f64,
    pub momentum_divergence: f64,
}

#[derive(Debug, Clone)]
pub struct MicrostructureFeatures {
    pub bid_ask_spread: f64,
    pub trade_intensity: f64,
    pub order_flow_imbalance: f64,
    pub market_depth: f64,
    pub liquidity_score: f64,
}

/// Regime snapshot for historical tracking
#[derive(Debug, Clone)]
struct RegimeSnapshot {
    regime: MarketRegime,
    confidence: f64,
    timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_regime_detector_creation() {
        let config = Config::default();
        let detector = RegimeDetector::new(&config);
        assert!(detector.is_ok());
    }
    
    #[tokio::test]
    async fn test_regime_detection() {
        let config = Config::default();
        let detector = RegimeDetector::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let regime_info = detector.detect_regime(&market_data).await;
        assert!(regime_info.is_ok());
        
        let info = regime_info.unwrap();
        assert!(info.confidence >= 0.0 && info.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let config = RegimeConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let features = extractor.extract_features(&market_data).await;
        assert!(features.is_ok());
        
        let f = features.unwrap();
        assert!(f.price_features.price_position >= 0.0 && f.price_features.price_position <= 1.0);
        assert!(f.technical_features.rsi >= 0.0 && f.technical_features.rsi <= 100.0);
    }
}