//! Market regime detection and transition prediction using consciousness-aware NHITS
//! 
//! This module implements sophisticated market regime identification and transition
//! forecasting, leveraging consciousness mechanisms to detect regime changes
//! before they fully manifest in traditional market indicators.

use super::*;
use ndarray::{Array1, Array2, s, Axis};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Market regime detector using consciousness-aware analysis
#[derive(Debug)]
pub struct MarketRegimeDetector {
    pub nhits_model: FinancialNHITS,
    pub regime_states: Vec<MarketRegime>,
    pub transition_matrix: Array2<f32>,
    pub consciousness_threshold: f32,
    pub regime_history: Vec<RegimeObservation>,
    pub feature_extractors: HashMap<String, Box<dyn RegimeFeatureExtractor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,           // Rising market with optimism
    Bear,           // Declining market with pessimism
    Sideways,       // Range-bound market
    HighVolatility, // Volatile market regardless of direction
    LowVolatility,  // Calm market
    Crisis,         // Market stress/panic
    Recovery,       // Post-crisis recovery
    Euphoria,       // Extreme bullish sentiment
    Capitulation,   // Extreme bearish sentiment
    ConsciousnessShift, // Regime change driven by consciousness shift
}

#[derive(Debug, Clone)]
pub struct RegimeObservation {
    pub timestamp: i64,
    pub current_regime: MarketRegime,
    pub regime_probability: f32,
    pub consciousness_state: f32,
    pub regime_strength: f32,
    pub transition_signals: HashMap<MarketRegime, f32>,
    pub duration_in_regime: u32,  // Days in current regime
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransitionForecast {
    pub current_regime: MarketRegime,
    pub most_likely_next_regime: MarketRegime,
    pub transition_probability: f32,
    pub expected_transition_time: u32,  // Days
    pub consciousness_factor: f32,
    pub transition_triggers: Vec<String>,
    pub regime_probabilities: HashMap<MarketRegime, f32>,
    pub forecast_timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct RegimeFeatures {
    pub volatility_regime: f32,
    pub trend_strength: f32,
    pub momentum: f32,
    pub volume_profile: f32,
    pub correlation_structure: f32,
    pub sentiment_indicator: f32,
    pub liquidity_conditions: f32,
    pub consciousness_coherence: f32,
    pub cross_asset_signals: f32,
}

/// Trait for regime feature extraction
pub trait RegimeFeatureExtractor: std::fmt::Debug {
    fn extract_features(&self, data: &Array2<f32>) -> RegimeFeatures;
    fn calculate_regime_probability(&self, features: &RegimeFeatures) -> HashMap<MarketRegime, f32>;
}

/// Volatility-based regime detector
#[derive(Debug)]
pub struct VolatilityRegimeExtractor {
    pub vol_threshold_high: f32,
    pub vol_threshold_low: f32,
    pub lookback_window: usize,
}

impl RegimeFeatureExtractor for VolatilityRegimeExtractor {
    fn extract_features(&self, data: &Array2<f32>) -> RegimeFeatures {
        let returns = data.slice(s![.., 5]).to_vec();  // Returns column
        let volatility = self.calculate_rolling_volatility(&returns);
        
        let current_vol = volatility.last().copied().unwrap_or(0.0);
        let vol_regime = if current_vol > self.vol_threshold_high {
            1.0  // High volatility
        } else if current_vol < self.vol_threshold_low {
            -1.0  // Low volatility
        } else {
            0.0  // Normal volatility
        };
        
        RegimeFeatures {
            volatility_regime: vol_regime,
            trend_strength: self.calculate_trend_strength(&returns),
            momentum: self.calculate_momentum(&returns),
            volume_profile: 0.0,  // Would extract from volume data
            correlation_structure: 0.0,
            sentiment_indicator: 0.0,
            liquidity_conditions: 0.0,
            consciousness_coherence: self.calculate_consciousness_coherence(&returns),
            cross_asset_signals: 0.0,
        }
    }
    
    fn calculate_regime_probability(&self, features: &RegimeFeatures) -> HashMap<MarketRegime, f32> {
        let mut probabilities = HashMap::new();
        
        // High volatility regime
        if features.volatility_regime > 0.5 {
            probabilities.insert(MarketRegime::HighVolatility, 0.8);
            probabilities.insert(MarketRegime::Crisis, 0.3);
        } else if features.volatility_regime < -0.5 {
            probabilities.insert(MarketRegime::LowVolatility, 0.8);
            probabilities.insert(MarketRegime::Sideways, 0.4);
        }
        
        // Trend-based regimes
        if features.trend_strength > 0.3 && features.momentum > 0.2 {
            probabilities.insert(MarketRegime::Bull, 0.7);
        } else if features.trend_strength < -0.3 && features.momentum < -0.2 {
            probabilities.insert(MarketRegime::Bear, 0.7);
        }
        
        // Consciousness-driven regimes
        if features.consciousness_coherence > 0.8 {
            probabilities.insert(MarketRegime::ConsciousnessShift, 0.6);
        }
        
        probabilities
    }
}

impl VolatilityRegimeExtractor {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            vol_threshold_high: 0.02,  // 2% daily volatility
            vol_threshold_low: 0.005,  // 0.5% daily volatility
            lookback_window,
        }
    }
    
    fn calculate_rolling_volatility(&self, returns: &[f32]) -> Vec<f32> {
        if returns.len() < self.lookback_window {
            return vec![0.0];
        }
        
        returns.windows(self.lookback_window)
            .map(|window| {
                let mean = window.iter().sum::<f32>() / window.len() as f32;
                let variance = window.iter()
                    .map(|&r| (r - mean).powi(2))
                    .sum::<f32>() / (window.len() - 1) as f32;
                variance.sqrt()
            })
            .collect()
    }
    
    fn calculate_trend_strength(&self, returns: &[f32]) -> f32 {
        if returns.len() < 20 {
            return 0.0;
        }
        
        // Linear regression slope over lookback window
        let n = self.lookback_window.min(returns.len());
        let recent_returns = &returns[returns.len() - n..];
        
        let x_mean = (n - 1) as f32 / 2.0;
        let y_mean = recent_returns.iter().sum::<f32>() / n as f32;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &ret) in recent_returns.iter().enumerate() {
            let x_diff = i as f32 - x_mean;
            numerator += x_diff * (ret - y_mean);
            denominator += x_diff * x_diff;
        }
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn calculate_momentum(&self, returns: &[f32]) -> f32 {
        if returns.len() < 10 {
            return 0.0;
        }
        
        let n = 10.min(returns.len());
        let recent_avg = returns[returns.len() - n..].iter().sum::<f32>() / n as f32;
        
        if returns.len() < 20 {
            return recent_avg;
        }
        
        let older_avg = returns[returns.len() - 20..returns.len() - 10].iter().sum::<f32>() / 10.0;
        recent_avg - older_avg
    }
    
    fn calculate_consciousness_coherence(&self, returns: &[f32]) -> f32 {
        if returns.len() < 20 {
            return 0.5;
        }
        
        // Consciousness coherence based on return predictability
        let autocorr = self.calculate_autocorrelation(returns, 1);
        let volatility_clustering = self.calculate_volatility_clustering(returns);
        
        // Higher coherence = more predictable patterns
        (autocorr.abs() + volatility_clustering) / 2.0
    }
    
    fn calculate_autocorrelation(&self, series: &[f32], lag: usize) -> f32 {
        if series.len() <= lag {
            return 0.0;
        }
        
        let n = series.len() - lag;
        let mean = series.iter().sum::<f32>() / series.len() as f32;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            let x_i = series[i] - mean;
            let x_lag = series[i + lag] - mean;
            numerator += x_i * x_lag;
        }
        
        for &x in series {
            denominator += (x - mean).powi(2);
        }
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn calculate_volatility_clustering(&self, returns: &[f32]) -> f32 {
        let squared_returns: Vec<f32> = returns.iter().map(|&r| r.powi(2)).collect();
        self.calculate_autocorrelation(&squared_returns, 1).abs()
    }
}

/// Sentiment-based regime detector
#[derive(Debug)]
pub struct SentimentRegimeExtractor {
    pub sentiment_thresholds: (f32, f32),  // (bearish, bullish)
    pub momentum_window: usize,
}

impl RegimeFeatureExtractor for SentimentRegimeExtractor {
    fn extract_features(&self, data: &Array2<f32>) -> RegimeFeatures {
        let returns = data.slice(s![.., 5]).to_vec();
        let volumes = if data.ncols() > 6 { 
            data.slice(s![.., 4]).to_vec()  // Volume column
        } else { 
            vec![1.0; returns.len()] 
        };
        
        let sentiment = self.calculate_market_sentiment(&returns, &volumes);
        let momentum = self.calculate_price_momentum(&returns);
        
        RegimeFeatures {
            volatility_regime: 0.0,
            trend_strength: 0.0,
            momentum,
            volume_profile: self.calculate_volume_profile(&volumes),
            correlation_structure: 0.0,
            sentiment_indicator: sentiment,
            liquidity_conditions: 0.0,
            consciousness_coherence: 0.0,
            cross_asset_signals: 0.0,
        }
    }
    
    fn calculate_regime_probability(&self, features: &RegimeFeatures) -> HashMap<MarketRegime, f32> {
        let mut probabilities = HashMap::new();
        
        let sentiment = features.sentiment_indicator;
        
        if sentiment > self.sentiment_thresholds.1 {
            probabilities.insert(MarketRegime::Bull, 0.8);
            if sentiment > 0.8 {
                probabilities.insert(MarketRegime::Euphoria, 0.6);
            }
        } else if sentiment < self.sentiment_thresholds.0 {
            probabilities.insert(MarketRegime::Bear, 0.8);
            if sentiment < -0.8 {
                probabilities.insert(MarketRegime::Capitulation, 0.6);
            }
        } else {
            probabilities.insert(MarketRegime::Sideways, 0.6);
        }
        
        // Volume-based adjustments
        if features.volume_profile > 1.5 {
            // High volume suggests regime change
            for prob in probabilities.values_mut() {
                *prob *= 1.2;  // Increase confidence
            }
        }
        
        probabilities
    }
}

impl SentimentRegimeExtractor {
    pub fn new(momentum_window: usize) -> Self {
        Self {
            sentiment_thresholds: (-0.3, 0.3),
            momentum_window,
        }
    }
    
    fn calculate_market_sentiment(&self, returns: &[f32], volumes: &[f32]) -> f32 {
        if returns.len() != volumes.len() || returns.is_empty() {
            return 0.0;
        }
        
        // Volume-weighted sentiment
        let mut positive_volume = 0.0;
        let mut negative_volume = 0.0;
        let mut total_volume = 0.0;
        
        for (i, &ret) in returns.iter().enumerate() {
            let vol = volumes.get(i).copied().unwrap_or(1.0);
            total_volume += vol;
            
            if ret > 0.0 {
                positive_volume += vol;
            } else if ret < 0.0 {
                negative_volume += vol;
            }
        }
        
        if total_volume == 0.0 {
            0.0
        } else {
            (positive_volume - negative_volume) / total_volume
        }
    }
    
    fn calculate_price_momentum(&self, returns: &[f32]) -> f32 {
        if returns.len() < self.momentum_window {
            return 0.0;
        }
        
        let recent_window = &returns[returns.len() - self.momentum_window..];
        let cumulative_return: f32 = recent_window.iter().sum();
        
        // Normalize by window size
        cumulative_return / self.momentum_window as f32
    }
    
    fn calculate_volume_profile(&self, volumes: &[f32]) -> f32 {
        if volumes.len() < 20 {
            return 1.0;
        }
        
        let recent_avg = volumes[volumes.len() - 10..].iter().sum::<f32>() / 10.0;
        let historical_avg = volumes.iter().sum::<f32>() / volumes.len() as f32;
        
        if historical_avg == 0.0 {
            1.0
        } else {
            recent_avg / historical_avg
        }
    }
}

impl MarketRegimeDetector {
    pub fn new(input_dim: usize, lookback_window: usize, forecast_horizon: usize) -> Self {
        let regime_states = vec![
            MarketRegime::Bull,
            MarketRegime::Bear,
            MarketRegime::Sideways,
            MarketRegime::HighVolatility,
            MarketRegime::LowVolatility,
            MarketRegime::Crisis,
            MarketRegime::Recovery,
            MarketRegime::Euphoria,
            MarketRegime::Capitulation,
            MarketRegime::ConsciousnessShift,
        ];
        
        let n_regimes = regime_states.len();
        let transition_matrix = Array2::eye(n_regimes) * 0.8 + Array2::from_elem((n_regimes, n_regimes), 0.02);
        
        let mut feature_extractors: HashMap<String, Box<dyn RegimeFeatureExtractor>> = HashMap::new();
        feature_extractors.insert(
            "volatility".to_string(),
            Box::new(VolatilityRegimeExtractor::new(lookback_window))
        );
        feature_extractors.insert(
            "sentiment".to_string(),
            Box::new(SentimentRegimeExtractor::new(20))
        );
        
        Self {
            nhits_model: FinancialNHITS::new(
                input_dim,
                64,  // hidden_dim for regime detection
                3,   // num_stacks
                4,   // num_blocks
                forecast_horizon,
            ).with_financial_components(),
            regime_states,
            transition_matrix,
            consciousness_threshold: 0.6,
            regime_history: Vec::new(),
            feature_extractors,
        }
    }
    
    /// Detect current market regime
    pub fn detect_current_regime(&mut self, market_data: &Array2<f32>) -> Result<RegimeObservation, String> {
        if market_data.is_empty() {
            return Err("No market data provided".to_string());
        }
        
        // Extract features using all extractors
        let mut all_regime_probs = HashMap::new();
        let mut consciousness_states = Vec::new();
        
        for (name, extractor) in &self.feature_extractors {
            let features = extractor.extract_features(market_data);
            let regime_probs = extractor.calculate_regime_probability(&features);
            
            // Accumulate probabilities
            for (regime, prob) in regime_probs {
                *all_regime_probs.entry(regime).or_insert(0.0) += prob;
            }
            
            consciousness_states.push(features.consciousness_coherence);
        }
        
        // Normalize probabilities
        let total_prob: f32 = all_regime_probs.values().sum();
        if total_prob > 0.0 {
            for prob in all_regime_probs.values_mut() {
                *prob /= total_prob;
            }
        }
        
        // Find most likely regime
        let (current_regime, regime_probability) = all_regime_probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(regime, &prob)| (regime.clone(), prob))
            .unwrap_or((MarketRegime::Sideways, 0.5));
        
        // Calculate consciousness state
        let consciousness_state = consciousness_states.iter().sum::<f32>() / consciousness_states.len() as f32;
        
        // Calculate regime strength
        let regime_strength = regime_probability * (1.0 + consciousness_state);
        
        // Calculate transition signals
        let transition_signals = self.calculate_transition_signals(&all_regime_probs, consciousness_state);
        
        // Calculate duration in current regime
        let duration_in_regime = self.calculate_regime_duration(&current_regime);
        
        let observation = RegimeObservation {
            timestamp: chrono::Utc::now().timestamp(),
            current_regime,
            regime_probability,
            consciousness_state,
            regime_strength,
            transition_signals,
            duration_in_regime,
        };
        
        self.regime_history.push(observation.clone());
        
        // Keep only recent history
        if self.regime_history.len() > 1000 {
            self.regime_history.drain(..500);
        }
        
        Ok(observation)
    }
    
    /// Predict regime transitions
    pub fn predict_regime_transition(
        &mut self, 
        current_observation: &RegimeObservation,
        market_data: &Array2<f32>
    ) -> Result<RegimeTransitionForecast, String> {
        let current_regime_idx = self.regime_index(&current_observation.current_regime);
        
        // Get transition probabilities from transition matrix
        let transition_probs = self.transition_matrix.row(current_regime_idx);
        
        // Adjust probabilities based on consciousness state
        let consciousness_adjustment = self.calculate_consciousness_transition_adjustment(
            current_observation.consciousness_state
        );
        
        let mut adjusted_probs = HashMap::new();
        for (i, &prob) in transition_probs.iter().enumerate() {
            let regime = &self.regime_states[i];
            let adjusted_prob = prob * consciousness_adjustment.get(regime).copied().unwrap_or(1.0);
            adjusted_probs.insert(regime.clone(), adjusted_prob);
        }
        
        // Find most likely next regime (excluding current)
        let (most_likely_next_regime, transition_probability) = adjusted_probs
            .iter()
            .filter(|(regime, _)| **regime != current_observation.current_regime)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(regime, &prob)| (regime.clone(), prob))
            .unwrap_or((MarketRegime::Sideways, 0.1));
        
        // Estimate transition time using consciousness-aware model
        let expected_transition_time = self.estimate_transition_time(
            &current_observation.current_regime,
            &most_likely_next_regime,
            current_observation.consciousness_state
        );
        
        // Identify transition triggers
        let transition_triggers = self.identify_transition_triggers(
            &current_observation.current_regime,
            &most_likely_next_regime,
            market_data
        );
        
        Ok(RegimeTransitionForecast {
            current_regime: current_observation.current_regime.clone(),
            most_likely_next_regime,
            transition_probability,
            expected_transition_time,
            consciousness_factor: current_observation.consciousness_state,
            transition_triggers,
            regime_probabilities: adjusted_probs,
            forecast_timestamp: chrono::Utc::now().timestamp(),
        })
    }
    
    /// Train regime detection model
    pub fn train_regime_model(&mut self, historical_data: &[Array2<f32>], regime_labels: &[MarketRegime], epochs: usize) -> Result<(), String> {
        if historical_data.len() != regime_labels.len() {
            return Err("Data and labels must have same length".to_string());
        }
        
        // Update transition matrix based on historical regime sequences
        self.update_transition_matrix(regime_labels)?;
        
        // Train NHITS model for regime prediction
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (i, data) in historical_data.iter().enumerate() {
                let features = self.extract_comprehensive_features(data);
                let target_regime = &regime_labels[i];
                
                // Calculate consciousness-aware loss
                let consciousness_state = self.calculate_data_consciousness(data);
                let prediction_loss = self.calculate_regime_prediction_loss(&features, target_regime, consciousness_state);
                
                total_loss += prediction_loss;
            }
            
            if epoch % 100 == 0 {
                println!("Regime Training Epoch {}: Loss = {:.6}", epoch, total_loss / historical_data.len() as f32);
            }
        }
        
        Ok(())
    }
    
    /// Backtest regime detection accuracy
    pub fn backtest_regime_detection(
        &mut self,
        test_data: &[Array2<f32>],
        true_regimes: &[MarketRegime],
    ) -> RegimeBacktestResults {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        let mut regime_confusion_matrix = HashMap::new();
        let mut transition_accuracy = 0;
        let mut total_transitions = 0;
        
        let mut previous_prediction = None;
        let mut previous_true = None;
        
        for (i, data) in test_data.iter().enumerate() {
            if let Ok(observation) = self.detect_current_regime(data) {
                let predicted_regime = &observation.current_regime;
                let true_regime = &true_regimes[i];
                
                // Accuracy
                if predicted_regime == true_regime {
                    correct_predictions += 1;
                }
                total_predictions += 1;
                
                // Confusion matrix
                let entry = regime_confusion_matrix
                    .entry((true_regime.clone(), predicted_regime.clone()))
                    .or_insert(0);
                *entry += 1;
                
                // Transition accuracy
                if let (Some(prev_pred), Some(prev_true)) = (&previous_prediction, &previous_true) {
                    if prev_true != true_regime {  // True transition occurred
                        total_transitions += 1;
                        if prev_pred != predicted_regime {  // Predicted a transition
                            transition_accuracy += 1;
                        }
                    }
                }
                
                previous_prediction = Some(predicted_regime.clone());
                previous_true = Some(true_regime.clone());
            }
        }
        
        let overall_accuracy = if total_predictions > 0 {
            correct_predictions as f32 / total_predictions as f32
        } else {
            0.0
        };
        
        let transition_detection_rate = if total_transitions > 0 {
            transition_accuracy as f32 / total_transitions as f32
        } else {
            0.0
        };
        
        RegimeBacktestResults {
            overall_accuracy,
            transition_detection_rate,
            regime_confusion_matrix,
            total_predictions,
            correct_predictions,
            total_transitions,
            detected_transitions: transition_accuracy,
        }
    }
    
    // Private helper methods
    
    fn regime_index(&self, regime: &MarketRegime) -> usize {
        self.regime_states.iter()
            .position(|r| std::mem::discriminant(r) == std::mem::discriminant(regime))
            .unwrap_or(0)
    }
    
    fn calculate_transition_signals(&self, regime_probs: &HashMap<MarketRegime, f32>, consciousness: f32) -> HashMap<MarketRegime, f32> {
        let mut signals = HashMap::new();
        
        for (regime, &prob) in regime_probs {
            // Higher consciousness increases transition signal sensitivity
            let consciousness_multiplier = 1.0 + consciousness * 0.5;
            let signal_strength = prob * consciousness_multiplier;
            
            if signal_strength > 0.3 {  // Threshold for meaningful signal
                signals.insert(regime.clone(), signal_strength);
            }
        }
        
        signals
    }
    
    fn calculate_regime_duration(&self, current_regime: &MarketRegime) -> u32 {
        if self.regime_history.is_empty() {
            return 1;
        }
        
        let mut duration = 1;
        for observation in self.regime_history.iter().rev() {
            if std::mem::discriminant(&observation.current_regime) == std::mem::discriminant(current_regime) {
                duration += 1;
            } else {
                break;
            }
        }
        
        duration
    }
    
    fn calculate_consciousness_transition_adjustment(&self, consciousness: f32) -> HashMap<MarketRegime, f32> {
        let mut adjustments = HashMap::new();
        
        // High consciousness increases probability of stable regimes
        if consciousness > self.consciousness_threshold {
            adjustments.insert(MarketRegime::Bull, 1.2);
            adjustments.insert(MarketRegime::LowVolatility, 1.3);
            adjustments.insert(MarketRegime::Sideways, 1.1);
            adjustments.insert(MarketRegime::Crisis, 0.7);
            adjustments.insert(MarketRegime::HighVolatility, 0.8);
        } else {
            // Low consciousness increases probability of volatile regimes
            adjustments.insert(MarketRegime::Crisis, 1.4);
            adjustments.insert(MarketRegime::HighVolatility, 1.3);
            adjustments.insert(MarketRegime::Bear, 1.2);
            adjustments.insert(MarketRegime::Bull, 0.8);
            adjustments.insert(MarketRegime::LowVolatility, 0.7);
        }
        
        // ConsciousnessShift regime is more likely during consciousness transitions
        let consciousness_change_rate = self.calculate_consciousness_change_rate();
        if consciousness_change_rate > 0.1 {
            adjustments.insert(MarketRegime::ConsciousnessShift, 1.5);
        }
        
        adjustments
    }
    
    fn calculate_consciousness_change_rate(&self) -> f32 {
        if self.regime_history.len() < 2 {
            return 0.0;
        }
        
        let recent_consciousness: Vec<f32> = self.regime_history
            .iter()
            .rev()
            .take(10)
            .map(|obs| obs.consciousness_state)
            .collect();
        
        if recent_consciousness.len() < 2 {
            return 0.0;
        }
        
        let mut total_change = 0.0;
        for i in 1..recent_consciousness.len() {
            total_change += (recent_consciousness[i] - recent_consciousness[i-1]).abs();
        }
        
        total_change / (recent_consciousness.len() - 1) as f32
    }
    
    fn estimate_transition_time(&self, current_regime: &MarketRegime, next_regime: &MarketRegime, consciousness: f32) -> u32 {
        // Base transition times (in days) for different regime changes
        let base_time = match (current_regime, next_regime) {
            (MarketRegime::Bull, MarketRegime::Bear) => 30,
            (MarketRegime::Bear, MarketRegime::Bull) => 45,
            (MarketRegime::LowVolatility, MarketRegime::HighVolatility) => 7,
            (MarketRegime::HighVolatility, MarketRegime::LowVolatility) => 21,
            (_, MarketRegime::Crisis) => 3,
            (MarketRegime::Crisis, _) => 60,
            _ => 14,  // Default
        };
        
        // Consciousness affects transition speed
        let consciousness_factor = if consciousness > 0.7 {
            0.8  // High consciousness = faster, smoother transitions
        } else {
            1.2  // Low consciousness = slower, more chaotic transitions
        };
        
        (base_time as f32 * consciousness_factor) as u32
    }
    
    fn identify_transition_triggers(&self, current_regime: &MarketRegime, next_regime: &MarketRegime, market_data: &Array2<f32>) -> Vec<String> {
        let mut triggers = Vec::new();
        
        // Analyze market data for transition signals
        let returns = market_data.slice(s![.., 5]).to_vec();
        let volatility = self.calculate_recent_volatility(&returns);
        let momentum = self.calculate_recent_momentum(&returns);
        
        match (current_regime, next_regime) {
            (MarketRegime::Bull, MarketRegime::Bear) => {
                if volatility > 0.02 {
                    triggers.push("Volatility spike detected".to_string());
                }
                if momentum < -0.01 {
                    triggers.push("Negative momentum building".to_string());
                }
            },
            (MarketRegime::LowVolatility, MarketRegime::HighVolatility) => {
                triggers.push("Volatility breakout imminent".to_string());
            },
            (_, MarketRegime::Crisis) => {
                triggers.push("Market stress indicators elevated".to_string());
                triggers.push("Liquidity conditions deteriorating".to_string());
            },
            _ => {
                triggers.push("Regime change patterns detected".to_string());
            }
        }
        
        triggers
    }
    
    fn update_transition_matrix(&mut self, regime_sequence: &[MarketRegime]) -> Result<(), String> {
        if regime_sequence.len() < 2 {
            return Err("Need at least 2 regime observations".to_string());
        }
        
        let n_regimes = self.regime_states.len();
        let mut transition_counts = Array2::zeros((n_regimes, n_regimes));
        
        // Count transitions
        for i in 1..regime_sequence.len() {
            let from_idx = self.regime_index(&regime_sequence[i-1]);
            let to_idx = self.regime_index(&regime_sequence[i]);
            transition_counts[[from_idx, to_idx]] += 1.0;
        }
        
        // Normalize to get probabilities
        for i in 0..n_regimes {
            let row_sum = transition_counts.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_regimes {
                    self.transition_matrix[[i, j]] = transition_counts[[i, j]] / row_sum;
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_comprehensive_features(&self, data: &Array2<f32>) -> RegimeFeatures {
        // Combine features from all extractors
        let mut combined_features = RegimeFeatures {
            volatility_regime: 0.0,
            trend_strength: 0.0,
            momentum: 0.0,
            volume_profile: 0.0,
            correlation_structure: 0.0,
            sentiment_indicator: 0.0,
            liquidity_conditions: 0.0,
            consciousness_coherence: 0.0,
            cross_asset_signals: 0.0,
        };
        
        let num_extractors = self.feature_extractors.len() as f32;
        
        for extractor in self.feature_extractors.values() {
            let features = extractor.extract_features(data);
            
            combined_features.volatility_regime += features.volatility_regime / num_extractors;
            combined_features.trend_strength += features.trend_strength / num_extractors;
            combined_features.momentum += features.momentum / num_extractors;
            combined_features.volume_profile += features.volume_profile / num_extractors;
            combined_features.sentiment_indicator += features.sentiment_indicator / num_extractors;
            combined_features.consciousness_coherence += features.consciousness_coherence / num_extractors;
        }
        
        combined_features
    }
    
    fn calculate_data_consciousness(&self, data: &Array2<f32>) -> f32 {
        let returns = data.slice(s![.., 5]).to_vec();
        
        if returns.len() < 10 {
            return 0.5;
        }
        
        // Consciousness based on market coherence
        let volatility = {
            let mean = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / returns.len() as f32;
            variance.sqrt()
        };
        
        let trend_consistency = self.calculate_trend_consistency(&returns);
        
        // Higher consciousness = lower volatility + higher trend consistency
        (trend_consistency / (1.0 + volatility * 10.0)).min(1.0).max(0.0)
    }
    
    fn calculate_trend_consistency(&self, returns: &[f32]) -> f32 {
        if returns.len() < 2 {
            return 0.5;
        }
        
        let mut consistent_periods = 0;
        for i in 1..returns.len() {
            if (returns[i] > 0.0) == (returns[i-1] > 0.0) {
                consistent_periods += 1;
            }
        }
        
        consistent_periods as f32 / (returns.len() - 1) as f32
    }
    
    fn calculate_regime_prediction_loss(&self, _features: &RegimeFeatures, _target_regime: &MarketRegime, consciousness: f32) -> f32 {
        // Simplified loss calculation
        // In practice, this would be a proper loss function for regime classification
        let base_loss = 0.5;  // Cross-entropy or similar
        
        // Consciousness affects loss weighting
        base_loss * (1.0 - consciousness * 0.2)
    }
    
    fn calculate_recent_volatility(&self, returns: &[f32]) -> f32 {
        if returns.len() < 10 {
            return 0.0;
        }
        
        let recent_returns = &returns[returns.len() - 10..];
        let mean = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        let variance = recent_returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f32>() / (recent_returns.len() - 1) as f32;
        
        variance.sqrt()
    }
    
    fn calculate_recent_momentum(&self, returns: &[f32]) -> f32 {
        if returns.len() < 10 {
            return 0.0;
        }
        
        let recent_returns = &returns[returns.len() - 10..];
        recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
    }
}

#[derive(Debug, Clone)]
pub struct RegimeBacktestResults {
    pub overall_accuracy: f32,
    pub transition_detection_rate: f32,
    pub regime_confusion_matrix: HashMap<(MarketRegime, MarketRegime), i32>,
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub total_transitions: usize,
    pub detected_transitions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_regime_detector_creation() {
        let detector = MarketRegimeDetector::new(10, 60, 10);
        assert_eq!(detector.regime_states.len(), 10);
        assert_eq!(detector.transition_matrix.nrows(), 10);
        assert_eq!(detector.transition_matrix.ncols(), 10);
    }
    
    #[test]
    fn test_volatility_regime_extractor() {
        let extractor = VolatilityRegimeExtractor::new(20);
        let test_data = Array2::zeros((100, 10));
        
        let features = extractor.extract_features(&test_data);
        let probabilities = extractor.calculate_regime_probability(&features);
        
        assert!(!probabilities.is_empty());
        assert!(probabilities.values().all(|&p| p >= 0.0 && p <= 1.0));
    }
    
    #[test]
    fn test_sentiment_regime_extractor() {
        let extractor = SentimentRegimeExtractor::new(20);
        let mut test_data = Array2::zeros((100, 10));
        
        // Add some test returns (positive trend)
        for i in 0..100 {
            test_data[[i, 5]] = 0.01;  // 1% daily returns
        }
        
        let features = extractor.extract_features(&test_data);
        let probabilities = extractor.calculate_regime_probability(&features);
        
        // Should detect bullish regime
        assert!(probabilities.contains_key(&MarketRegime::Bull));
    }
    
    #[test]
    fn test_regime_transition_matrix() {
        let mut detector = MarketRegimeDetector::new(10, 60, 10);
        
        let regime_sequence = vec![
            MarketRegime::Bull,
            MarketRegime::Bull,
            MarketRegime::Sideways,
            MarketRegime::Bear,
            MarketRegime::Bear,
        ];
        
        let result = detector.update_transition_matrix(&regime_sequence);
        assert!(result.is_ok());
        
        // Check that transition matrix has been updated
        let bull_idx = detector.regime_index(&MarketRegime::Bull);
        let sideways_idx = detector.regime_index(&MarketRegime::Sideways);
        
        // Bull -> Sideways transition should have non-zero probability
        assert!(detector.transition_matrix[[bull_idx, sideways_idx]] > 0.0);
    }
}