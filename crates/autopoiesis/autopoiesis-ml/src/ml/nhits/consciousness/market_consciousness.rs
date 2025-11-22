/// Market Consciousness - Financial Consciousness Predictor
///
/// This module implements consciousness-aware financial market prediction that models
/// market dynamics as collective consciousness phenomena. It uses field coherence to
/// detect market sentiment and predict price movements through consciousness patterns.

use ndarray::{Array2, Array1};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, VecDeque};
use crate::consciousness::core::ConsciousnessState;

/// Market sentiment state representing collective financial consciousness
#[derive(Clone, Debug)]
pub struct MarketSentiment {
    pub bullish_energy: f64,
    pub bearish_energy: f64,
    pub volatility_consciousness: f64,
    pub momentum_coherence: f64,
    pub fear_greed_index: f64,
    pub market_field_strength: f64,
}

impl MarketSentiment {
    pub fn new() -> Self {
        Self {
            bullish_energy: 0.5,
            bearish_energy: 0.5,
            volatility_consciousness: 0.3,
            momentum_coherence: 0.5,
            fear_greed_index: 0.5,
            market_field_strength: 0.4,
        }
    }
    
    /// Update sentiment from market data
    pub fn update_from_market_data(&mut self, prices: &Array1<f64>, volumes: &Array1<f64>) {
        if prices.len() < 2 || volumes.len() < 2 {
            return;
        }
        
        // Compute price momentum
        let price_momentum = self.compute_price_momentum(prices);
        let volume_energy = self.compute_volume_energy(volumes);
        
        // Update bullish/bearish energies
        if price_momentum > 0.0 {
            self.bullish_energy = (self.bullish_energy + price_momentum * 0.1).min(1.0);
            self.bearish_energy = (self.bearish_energy - price_momentum * 0.05).max(0.0);
        } else {
            self.bearish_energy = (self.bearish_energy + (-price_momentum) * 0.1).min(1.0);
            self.bullish_energy = (self.bullish_energy - (-price_momentum) * 0.05).max(0.0);
        }
        
        // Update volatility consciousness
        let price_volatility = self.compute_price_volatility(prices);
        self.volatility_consciousness = (price_volatility * volume_energy).clamp(0.0, 1.0);
        
        // Update momentum coherence
        self.momentum_coherence = self.compute_momentum_coherence(prices);
        
        // Update fear-greed index
        self.fear_greed_index = self.compute_fear_greed_index(price_momentum, price_volatility);
        
        // Update market field strength
        self.market_field_strength = (self.bullish_energy - self.bearish_energy).abs() * 
                                   self.momentum_coherence * volume_energy;
    }
    
    /// Compute price momentum from price series
    fn compute_price_momentum(&self, prices: &Array1<f64>) -> f64 {
        let mut momentum = 0.0;
        let window_size = 5.min(prices.len());
        
        for i in 1..window_size {
            let idx = prices.len() - i - 1;
            let price_change = (prices[idx] - prices[idx - 1]) / prices[idx - 1];
            momentum += price_change * (window_size - i) as f64; // Weight recent changes higher
        }
        
        momentum / window_size as f64
    }
    
    /// Compute volume energy
    fn compute_volume_energy(&self, volumes: &Array1<f64>) -> f64 {
        if volumes.is_empty() { return 0.5; }
        
        let mean_volume = volumes.mean().unwrap_or(1.0);
        let recent_volume = volumes[volumes.len() - 1];
        
        (recent_volume / mean_volume).clamp(0.1, 2.0) / 2.0
    }
    
    /// Compute price volatility
    fn compute_price_volatility(&self, prices: &Array1<f64>) -> f64 {
        if prices.len() < 2 { return 0.3; }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt().clamp(0.0, 1.0)
    }
    
    /// Compute momentum coherence
    fn compute_momentum_coherence(&self, prices: &Array1<f64>) -> f64 {
        if prices.len() < 3 { return 0.5; }
        
        let mut coherence_sum = 0.0;
        let mut coherence_count = 0;
        
        // Check directional consistency
        for i in 2..prices.len() {
            let change1 = prices[i - 1] - prices[i - 2];
            let change2 = prices[i] - prices[i - 1];
            
            if change1 * change2 > 0.0 { 
                coherence_sum += 1.0; // Same direction
            }
            coherence_count += 1;
        }
        
        if coherence_count > 0 {
            coherence_sum / coherence_count as f64
        } else {
            0.5
        }
    }
    
    /// Compute fear-greed index
    fn compute_fear_greed_index(&self, momentum: f64, volatility: f64) -> f64 {
        // Greed increases with positive momentum and low volatility
        // Fear increases with negative momentum and high volatility
        let greed_signal = momentum / (1.0 + volatility);
        let fear_signal = -momentum * volatility;
        
        let combined_signal = greed_signal - fear_signal;
        (combined_signal * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// Consciousness-aware financial predictor
pub struct MarketConsciousness {
    pub market_sentiment: MarketSentiment,
    pub consciousness_weights: Array2<f64>,
    pub prediction_network: Array2<f64>,
    pub sentiment_history: VecDeque<MarketSentiment>,
    pub price_memory: HashMap<String, Array1<f64>>,
    pub pattern_recognition: PatternRecognition,
    pub input_dimension: usize,
}

impl MarketConsciousness {
    pub fn new(input_dimension: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let consciousness_weights = Array2::from_shape_fn((input_dimension, input_dimension), |_| {
            rng.gen_range(-0.1..0.1)
        });
        
        let prediction_network = Array2::from_shape_fn((input_dimension, input_dimension), |_| {
            rng.gen_range(-0.2..0.2) / (input_dimension as f64).sqrt()
        });
        
        Self {
            market_sentiment: MarketSentiment::new(),
            consciousness_weights,
            prediction_network,
            sentiment_history: VecDeque::with_capacity(100),
            price_memory: HashMap::new(),
            pattern_recognition: PatternRecognition::new(input_dimension),
            input_dimension,
        }
    }
    
    /// Predict market movements using consciousness
    pub fn predict(&mut self, market_data: &Array2<f64>) -> Array1<f64> {
        let batch_size = market_data.nrows();
        let mut predictions = Vec::new();
        
        for batch_idx in 0..batch_size {
            let data_sample = market_data.row(batch_idx).to_owned();
            
            // Update market sentiment from data
            self.update_market_sentiment(&data_sample);
            
            // Generate consciousness-aware prediction
            let prediction = self.generate_consciousness_prediction(&data_sample);
            
            // Store pattern in memory
            self.store_pattern_memory(&data_sample, &prediction);
            
            predictions.push(prediction);
        }
        
        // Aggregate predictions with consciousness field effects
        self.aggregate_consciousness_predictions(&predictions)
    }
    
    /// Update market sentiment from input data
    fn update_market_sentiment(&mut self, data: &Array1<f64>) {
        // Assume data contains [price, volume, ...] features
        if data.len() >= 2 {
            let prices = Array1::from_vec(vec![data[0]]);  // Single price point
            let volumes = Array1::from_vec(vec![data[1]]); // Single volume point
            
            // For proper sentiment update, we need price history
            if let Some(price_history) = self.price_memory.get("recent_prices") {
                let mut extended_prices = price_history.clone();
                extended_prices.push(ndarray::Axis(0), ndarray::ArrayView::from(&[data[0]])).unwrap();
                
                let volumes_extended = Array1::from_vec(vec![data[1]]);
                self.market_sentiment.update_from_market_data(&extended_prices, &volumes_extended);
                
                // Update price memory
                if extended_prices.len() > 50 {
                    let trimmed = extended_prices.slice(ndarray::s![25..]).to_owned();
                    self.price_memory.insert("recent_prices".to_string(), trimmed);
                } else {
                    self.price_memory.insert("recent_prices".to_string(), extended_prices);
                }
            } else {
                // Initialize price memory
                self.price_memory.insert("recent_prices".to_string(), prices);
            }
        }
        
        // Store sentiment history
        self.sentiment_history.push_back(self.market_sentiment.clone());
        if self.sentiment_history.len() > 100 {
            self.sentiment_history.pop_front();
        }
    }
    
    /// Generate consciousness-aware market prediction
    fn generate_consciousness_prediction(&self, data: &Array1<f64>) -> Array1<f64> {
        // Apply consciousness modulation to input
        let consciousness_modulated = self.apply_consciousness_modulation(data);
        
        // Apply market sentiment influence
        let sentiment_influenced = self.apply_sentiment_influence(&consciousness_modulated);
        
        // Generate base prediction
        let base_prediction = self.prediction_network.dot(&sentiment_influenced);
        
        // Apply pattern recognition
        let pattern_enhanced = self.pattern_recognition.enhance_prediction(&base_prediction, &self.market_sentiment);
        
        // Apply consciousness field effects
        self.apply_consciousness_field_effects(&pattern_enhanced)
    }
    
    /// Apply consciousness modulation to market data
    fn apply_consciousness_modulation(&self, data: &Array1<f64>) -> Array1<f64> {
        let consciousness_state = self.infer_market_consciousness(data);
        let modulation_strength = consciousness_state.coherence_level * consciousness_state.field_coherence;
        
        let mut modulated = Array1::zeros(data.len());
        
        for i in 0..data.len() {
            let mut consciousness_influence = 0.0;
            
            for j in 0..data.len() {
                consciousness_influence += self.consciousness_weights[(i, j)] * data[j];
            }
            
            modulated[i] = data[i] * (1.0 + consciousness_influence * modulation_strength * 0.1);
        }
        
        modulated
    }
    
    /// Infer consciousness state from market data
    fn infer_market_consciousness(&self, data: &Array1<f64>) -> ConsciousnessState {
        let coherence = self.market_sentiment.momentum_coherence;
        let field_coherence = self.market_sentiment.market_field_strength;
        
        let mut consciousness = ConsciousnessState::new();
        consciousness.coherence_level = coherence;
        consciousness.field_coherence = field_coherence;
        consciousness
    }
    
    /// Apply market sentiment influence
    fn apply_sentiment_influence(&self, data: &Array1<f64>) -> Array1<f64> {
        let mut influenced = data.clone();
        
        // Apply bullish/bearish bias
        let sentiment_bias = self.market_sentiment.bullish_energy - self.market_sentiment.bearish_energy;
        
        // Apply volatility consciousness
        let volatility_factor = self.market_sentiment.volatility_consciousness;
        
        // Apply fear-greed modulation
        let fear_greed_factor = (self.market_sentiment.fear_greed_index - 0.5) * 2.0; // Center around 0
        
        for val in influenced.iter_mut() {
            *val *= 1.0 + sentiment_bias * 0.05;
            *val *= 1.0 + volatility_factor * 0.03;
            *val += fear_greed_factor * 0.02;
        }
        
        influenced
    }
    
    /// Apply consciousness field effects
    fn apply_consciousness_field_effects(&self, prediction: &Array1<f64>) -> Array1<f64> {
        let field_strength = self.market_sentiment.market_field_strength;
        let mut enhanced = prediction.clone();
        
        // Apply field resonance based on historical sentiment patterns
        if self.sentiment_history.len() > 5 {
            let recent_field_strength: f64 = self.sentiment_history.iter()
                .rev()
                .take(5)
                .map(|s| s.market_field_strength)
                .sum::<f64>() / 5.0;
            
            let field_resonance = (field_strength - recent_field_strength) * 0.1;
            
            for val in enhanced.iter_mut() {
                *val += field_resonance * val.abs().sqrt() * val.signum();
            }
        }
        
        enhanced
    }
    
    /// Store pattern in memory for future reference
    fn store_pattern_memory(&mut self, input: &Array1<f64>, prediction: &Array1<f64>) {
        // Create pattern key from input characteristics
        let pattern_key = self.compute_pattern_key(input);
        
        // Store prediction for this pattern
        self.price_memory.insert(format!("pattern_{}", pattern_key), prediction.clone());
        
        // Limit memory size
        if self.price_memory.len() > 1000 {
            // Remove oldest patterns (simplified cleanup)
            let keys_to_remove: Vec<String> = self.price_memory.keys()
                .filter(|k| k.starts_with("pattern_"))
                .take(100)
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                self.price_memory.remove(&key);
            }
        }
    }
    
    /// Compute pattern key for memory storage
    fn compute_pattern_key(&self, input: &Array1<f64>) -> String {
        let mean = input.mean().unwrap_or(0.0);
        let std = input.std(0.0);
        let momentum = if input.len() > 1 { 
            input[input.len() - 1] - input[0] 
        } else { 
            0.0 
        };
        
        format!("{}_{}_{}_{}", 
                (mean * 1000.0) as i32,
                (std * 1000.0) as i32,
                (momentum * 1000.0) as i32,
                (self.market_sentiment.market_field_strength * 1000.0) as i32)
    }
    
    /// Aggregate predictions with consciousness field effects
    fn aggregate_consciousness_predictions(&self, predictions: &[Array1<f64>]) -> Array1<f64> {
        if predictions.is_empty() {
            return Array1::zeros(self.input_dimension);
        }
        
        let mut aggregated = Array1::zeros(predictions[0].len());
        let mut total_weight = 0.0;
        
        // Weight predictions by market consciousness strength
        for (i, prediction) in predictions.iter().enumerate() {
            let sentiment_idx = (i.min(self.sentiment_history.len().saturating_sub(1)));
            let weight = if !self.sentiment_history.is_empty() {
                self.sentiment_history[sentiment_idx].market_field_strength
            } else {
                1.0 / predictions.len() as f64
            };
            
            aggregated = &aggregated + &(prediction * weight);
            total_weight += weight;
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            aggregated = aggregated / total_weight;
        }
        
        // Apply final consciousness coherence
        let overall_coherence = if !self.sentiment_history.is_empty() {
            self.sentiment_history.iter()
                .map(|s| s.momentum_coherence)
                .sum::<f64>() / self.sentiment_history.len() as f64
        } else {
            0.5
        };
        
        aggregated.mapv(|x| x * overall_coherence + x * (1.0 - overall_coherence) * 0.5)
    }
    
    /// Update model based on prediction performance
    pub fn update_from_performance(&mut self, actual_returns: &Array1<f64>, predicted_returns: &Array1<f64>) {
        let prediction_error = self.compute_prediction_error(actual_returns, predicted_returns);
        let learning_rate = 0.001;
        
        // Update consciousness weights based on performance
        if prediction_error < 0.1 {
            // Good performance - strengthen consciousness influence
            self.consciousness_weights.mapv_inplace(|w| w * (1.0 + learning_rate));
        } else {
            // Poor performance - reduce consciousness influence
            self.consciousness_weights.mapv_inplace(|w| w * (1.0 - learning_rate * 0.5));
        }
        
        // Update prediction network
        self.update_prediction_network(actual_returns, predicted_returns, learning_rate);
        
        // Update pattern recognition based on performance
        self.pattern_recognition.update_from_performance(prediction_error);
    }
    
    /// Compute prediction error
    fn compute_prediction_error(&self, actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
        if actual.len() != predicted.len() || actual.is_empty() {
            return 1.0;
        }
        
        let mse = actual.iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f64>() / actual.len() as f64;
        
        mse.sqrt()
    }
    
    /// Update prediction network weights
    fn update_prediction_network(&mut self, actual: &Array1<f64>, predicted: &Array1<f64>, learning_rate: f64) {
        let error = actual - predicted;
        
        // Simple gradient update (simplified for consciousness integration)
        for i in 0..self.prediction_network.nrows() {
            for j in 0..self.prediction_network.ncols() {
                if i < error.len() && j < predicted.len() {
                    let gradient = error[i] * predicted[j];
                    self.prediction_network[(i, j)] += learning_rate * gradient;
                }
            }
        }
    }
    
    /// Get market consciousness statistics
    pub fn get_consciousness_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("bullish_energy".to_string(), self.market_sentiment.bullish_energy);
        stats.insert("bearish_energy".to_string(), self.market_sentiment.bearish_energy);
        stats.insert("volatility_consciousness".to_string(), self.market_sentiment.volatility_consciousness);
        stats.insert("momentum_coherence".to_string(), self.market_sentiment.momentum_coherence);
        stats.insert("fear_greed_index".to_string(), self.market_sentiment.fear_greed_index);
        stats.insert("market_field_strength".to_string(), self.market_sentiment.market_field_strength);
        
        if !self.sentiment_history.is_empty() {
            let avg_field_strength = self.sentiment_history.iter()
                .map(|s| s.market_field_strength)
                .sum::<f64>() / self.sentiment_history.len() as f64;
            stats.insert("avg_field_strength".to_string(), avg_field_strength);
        }
        
        stats
    }
}

/// Pattern recognition system for market consciousness
struct PatternRecognition {
    pub patterns: HashMap<String, PatternTemplate>,
    pub recognition_threshold: f64,
    pub adaptation_rate: f64,
}

#[derive(Clone)]
struct PatternTemplate {
    pub template: Array1<f64>,
    pub confidence: f64,
    pub success_rate: f64,
    pub consciousness_affinity: f64,
}

impl PatternRecognition {
    fn new(dimension: usize) -> Self {
        Self {
            patterns: HashMap::new(),
            recognition_threshold: 0.7,
            adaptation_rate: 0.01,
        }
    }
    
    /// Enhance prediction using pattern recognition
    fn enhance_prediction(&self, prediction: &Array1<f64>, sentiment: &MarketSentiment) -> Array1<f64> {
        let mut enhanced = prediction.clone();
        
        // Find matching patterns
        let matching_patterns = self.find_matching_patterns(prediction);
        
        for (pattern_name, similarity) in matching_patterns {
            if let Some(pattern) = self.patterns.get(&pattern_name) {
                if similarity > self.recognition_threshold {
                    let enhancement_strength = similarity * pattern.confidence * 
                                             pattern.consciousness_affinity * sentiment.market_field_strength;
                    
                    for i in 0..enhanced.len() {
                        if i < pattern.template.len() {
                            enhanced[i] += pattern.template[i] * enhancement_strength * 0.1;
                        }
                    }
                }
            }
        }
        
        enhanced
    }
    
    /// Find patterns matching current prediction
    fn find_matching_patterns(&self, prediction: &Array1<f64>) -> Vec<(String, f64)> {
        let mut matches = Vec::new();
        
        for (name, pattern) in &self.patterns {
            let similarity = self.compute_pattern_similarity(prediction, &pattern.template);
            matches.push((name.clone(), similarity));
        }
        
        // Sort by similarity
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches
    }
    
    /// Compute similarity between prediction and pattern template
    fn compute_pattern_similarity(&self, prediction: &Array1<f64>, template: &Array1<f64>) -> f64 {
        if prediction.len() != template.len() {
            return 0.0;
        }
        
        let dot_product = prediction.dot(template);
        let norm_pred = prediction.mapv(|x| x * x).sum().sqrt();
        let norm_template = template.mapv(|x| x * x).sum().sqrt();
        
        if norm_pred > 1e-10 && norm_template > 1e-10 {
            dot_product / (norm_pred * norm_template)
        } else {
            0.0
        }
    }
    
    /// Update pattern recognition based on performance
    fn update_from_performance(&mut self, prediction_error: f64) {
        let performance_score = (1.0 - prediction_error).clamp(0.0, 1.0);
        
        // Update all pattern confidence based on performance
        for pattern in self.patterns.values_mut() {
            pattern.success_rate = pattern.success_rate * (1.0 - self.adaptation_rate) + 
                                 performance_score * self.adaptation_rate;
            pattern.confidence = pattern.success_rate;
        }
    }
}