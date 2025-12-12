//! Machiavellian strategic framework for market manipulation detection
//! 
//! High-performance Rust implementation with SIMD acceleration for real-time
//! detection of market manipulation patterns including spoofing, layering,
//! wash trading, pump & dump, and front-running.

use crate::{
    config::HardwareConfig,
    error::{QBMIAError, Result},
    strategy::{OrderEvent, ManipulationPattern, StrategyRecommendation, StrategyAction},
};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::time::Instant;

#[cfg(feature = "simd")]
use wide::*;

/// Result of manipulation detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationDetectionResult {
    /// Whether manipulation was detected
    pub detected: bool,
    /// Overall confidence score [0, 1]
    pub confidence: f64,
    /// Individual manipulation pattern scores
    pub manipulation_scores: HashMap<String, f64>,
    /// Primary manipulation pattern
    pub primary_pattern: String,
    /// Execution time in milliseconds
    pub execution_time: f64,
    /// Recommended action
    pub recommended_action: String,
}

/// Machiavellian strategic framework
pub struct MachiavellianFramework {
    /// Configuration
    config: HardwareConfig,
    /// Detection sensitivity [0, 1]
    sensitivity: f64,
    
    /// Manipulation pattern weights
    manipulation_patterns: HashMap<String, f64>,
    
    /// Historical data buffers
    order_history: VecDeque<OrderEvent>,
    price_history: VecDeque<f64>,
    detection_history: VecDeque<ManipulationDetectionResult>,
    
    /// Strategic deception parameters
    deception_strategies: HashMap<String, HashMap<String, f64>>,
    
    /// Performance statistics
    detection_stats: DetectionStats,
    
    /// SIMD optimization buffers
    #[cfg(feature = "simd")]
    simd_buffer: Vec<f64x4>,
}

/// Performance statistics for manipulation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStats {
    pub total_detections: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub detection_latency: Vec<f64>,
    pub average_latency: f64,
}

impl Default for DetectionStats {
    fn default() -> Self {
        Self {
            total_detections: 0,
            true_positives: 0,
            false_positives: 0,
            detection_latency: Vec::new(),
            average_latency: 0.0,
        }
    }
}

impl MachiavellianFramework {
    /// Create a new Machiavellian framework
    pub fn new(config: HardwareConfig, sensitivity: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&sensitivity) {
            return Err(QBMIAError::validation("Sensitivity must be between 0 and 1"));
        }
        
        let mut manipulation_patterns = HashMap::new();
        manipulation_patterns.insert("spoofing".to_string(), 0.3);
        manipulation_patterns.insert("layering".to_string(), 0.25);
        manipulation_patterns.insert("wash_trading".to_string(), 0.2);
        manipulation_patterns.insert("pump_dump".to_string(), 0.15);
        manipulation_patterns.insert("front_running".to_string(), 0.1);
        
        let mut deception_strategies = HashMap::new();
        let mut noise_injection = HashMap::new();
        noise_injection.insert("enabled".to_string(), 1.0);
        noise_injection.insert("intensity".to_string(), 0.1);
        deception_strategies.insert("noise_injection".to_string(), noise_injection);
        
        #[cfg(feature = "simd")]
        let simd_buffer = vec![f64x4::splat(0.0); 1024]; // Pre-allocated SIMD buffer
        
        Ok(Self {
            config,
            sensitivity,
            manipulation_patterns,
            order_history: VecDeque::with_capacity(10000),
            price_history: VecDeque::with_capacity(10000),
            detection_history: VecDeque::with_capacity(1000),
            deception_strategies,
            detection_stats: DetectionStats::default(),
            #[cfg(feature = "simd")]
            simd_buffer,
        })
    }
    
    /// Detect market manipulation patterns
    pub async fn detect_manipulation(
        &mut self,
        order_flow: &[OrderEvent],
        price_history: &[f64],
    ) -> Result<ManipulationDetectionResult> {
        let start_time = Instant::now();
        
        // Update internal buffers
        self.update_buffers(order_flow, price_history);
        
        // Preprocess data for analysis
        let order_data = self.preprocess_order_flow(order_flow)?;
        let prices = Array1::from_vec(price_history.to_vec());
        
        // Run detection algorithms in parallel
        let detection_tasks = vec![
            self.detect_spoofing(&order_data).await,
            self.detect_layering(&order_data).await,
            self.detect_wash_trading(&order_data, &prices).await,
            self.detect_pump_dump(&prices).await,
            self.detect_front_running(&order_data, &prices).await,
        ];
        
        // Collect results
        let results: Result<Vec<f64>> = detection_tasks.into_iter().collect();
        let scores = results?;
        
        // Map scores to pattern names
        let pattern_names = vec!["spoofing", "layering", "wash_trading", "pump_dump", "front_running"];
        let mut manipulation_scores = HashMap::new();
        
        for (i, &score) in scores.iter().enumerate() {
            manipulation_scores.insert(pattern_names[i].to_string(), score);
        }
        
        // Calculate weighted overall score
        let overall_score = manipulation_scores
            .iter()
            .map(|(pattern, &score)| score * self.manipulation_patterns.get(pattern).unwrap_or(&0.0))
            .sum::<f64>();
        
        // Determine if manipulation detected
        let detected = overall_score > self.sensitivity;
        
        // Find primary pattern
        let primary_pattern = manipulation_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "none".to_string());
        
        // Generate recommendation
        let recommended_action = self.recommend_action(&manipulation_scores, overall_score);
        
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Update statistics
        self.update_detection_stats(detected, execution_time);
        
        let result = ManipulationDetectionResult {
            detected,
            confidence: overall_score,
            manipulation_scores,
            primary_pattern,
            execution_time,
            recommended_action,
        };
        
        // Store in history
        if detected {
            self.detection_history.push_back(result.clone());
            if self.detection_history.len() > 1000 {
                self.detection_history.pop_front();
            }
        }
        
        Ok(result)
    }
    
    /// Update internal data buffers
    fn update_buffers(&mut self, order_flow: &[OrderEvent], price_history: &[f64]) {
        // Update order history
        for order in order_flow {
            self.order_history.push_back(order.clone());
            if self.order_history.len() > 10000 {
                self.order_history.pop_front();
            }
        }
        
        // Update price history
        for &price in price_history {
            self.price_history.push_back(price);
            if self.price_history.len() > 10000 {
                self.price_history.pop_front();
            }
        }
    }
    
    /// Preprocess order flow data for analysis
    fn preprocess_order_flow(&self, order_flow: &[OrderEvent]) -> Result<OrderFlowData> {
        if order_flow.is_empty() {
            return Ok(OrderFlowData::empty());
        }
        
        let sizes: Vec<f64> = order_flow.iter().map(|o| o.size).collect();
        let prices: Vec<f64> = order_flow.iter().map(|o| o.price).collect();
        let timestamps: Vec<f64> = order_flow.iter().map(|o| o.timestamp).collect();
        let sides: Vec<f64> = order_flow.iter()
            .map(|o| if o.side == "buy" { 1.0 } else { -1.0 })
            .collect();
        let cancelled: Vec<bool> = order_flow.iter().map(|o| o.cancelled).collect();
        
        Ok(OrderFlowData {
            sizes: Array1::from_vec(sizes),
            prices: Array1::from_vec(prices),
            timestamps: Array1::from_vec(timestamps),
            sides: Array1::from_vec(sides),
            cancelled,
        })
    }
    
    /// Detect spoofing patterns using SIMD optimization
    async fn detect_spoofing(&self, order_data: &OrderFlowData) -> Result<f64> {
        if order_data.sizes.len() < 100 {
            return Ok(0.0);
        }
        
        #[cfg(feature = "simd")]
        {
            self.detect_spoofing_simd(order_data).await
        }
        
        #[cfg(not(feature = "simd"))]
        {
            self.detect_spoofing_cpu(order_data).await
        }
    }
    
    #[cfg(feature = "simd")]
    async fn detect_spoofing_simd(&self, order_data: &OrderFlowData) -> Result<f64> {
        const WINDOW_SIZE: usize = 100;
        let n = order_data.sizes.len();
        
        if n < WINDOW_SIZE {
            return Ok(0.0);
        }
        
        let mut spoofing_score = 0.0f64;
        let mut windows_analyzed = 0usize;
        
        // Process in SIMD chunks
        for i in (WINDOW_SIZE..n).step_by(4) {
            let end_idx = std::cmp::min(i + 4, n);
            let chunk_size = end_idx - i;
            
            if chunk_size < 4 {
                // Handle remaining elements with scalar code
                for j in i..end_idx {
                    let window_start = j - WINDOW_SIZE;
                    let window_sizes = &order_data.sizes.slice(ndarray::s![window_start..j]);
                    let window_cancels = &order_data.cancelled[window_start..j];
                    
                    let score = self.calculate_spoofing_score(window_sizes, window_cancels);
                    if score > 0.0 {
                        spoofing_score += score;
                        windows_analyzed += 1;
                    }
                }
            } else {
                // SIMD processing for 4 windows simultaneously
                let mut scores = f64x4::splat(0.0);
                
                for lane in 0..4 {
                    let window_idx = i + lane;
                    if window_idx < n {
                        let window_start = window_idx - WINDOW_SIZE;
                        let window_sizes = &order_data.sizes.slice(ndarray::s![window_start..window_idx]);
                        let window_cancels = &order_data.cancelled[window_start..window_idx];
                        
                        let score = self.calculate_spoofing_score(window_sizes, window_cancels);
                        let mut scores_array = scores.to_array();
                        scores_array[lane] = score;
                        scores = f64x4::new(scores_array);
                        
                        if score > 0.0 {
                            windows_analyzed += 1;
                        }
                    }
                }
                
                spoofing_score += scores.to_array().iter().sum::<f64>();
            }
        }
        
        Ok(spoofing_score / (windows_analyzed.max(1) as f64))
    }
    
    #[cfg(not(feature = "simd"))]
    async fn detect_spoofing_cpu(&self, order_data: &OrderFlowData) -> Result<f64> {
        const WINDOW_SIZE: usize = 100;
        let n = order_data.sizes.len();
        
        if n < WINDOW_SIZE {
            return Ok(0.0);
        }
        
        let mut spoofing_score = 0.0;
        let mut windows_analyzed = 0;
        
        for i in WINDOW_SIZE..n {
            let window_start = i - WINDOW_SIZE;
            let window_sizes = &order_data.sizes.slice(ndarray::s![window_start..i]);
            let window_cancels = &order_data.cancelled[window_start..i];
            
            let score = self.calculate_spoofing_score(window_sizes, window_cancels);
            if score > 0.0 {
                spoofing_score += score;
                windows_analyzed += 1;
            }
        }
        
        Ok(spoofing_score / (windows_analyzed.max(1) as f64))
    }
    
    /// Calculate spoofing score for a window
    fn calculate_spoofing_score(&self, window_sizes: &ArrayView1<f64>, window_cancels: &[bool]) -> f64 {
        if window_sizes.is_empty() {
            return 0.0;
        }
        
        // Calculate percentiles using parallel processing
        let mut sorted_sizes: Vec<f64> = window_sizes.to_vec();
        sorted_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let p90_idx = (sorted_sizes.len() as f64 * 0.9) as usize;
        let large_order_threshold = sorted_sizes.get(p90_idx).copied().unwrap_or(0.0);
        
        // Identify large orders and cancelled orders
        let large_orders: Vec<bool> = window_sizes.iter()
            .map(|&size| size > large_order_threshold)
            .collect();
        
        let large_cancelled_count = large_orders.iter()
            .zip(window_cancels.iter())
            .filter(|(&is_large, &is_cancelled)| is_large && is_cancelled)
            .count();
        
        let large_orders_count = large_orders.iter().filter(|&&is_large| is_large).count();
        
        if large_orders_count > 0 {
            let cancel_rate = large_cancelled_count as f64 / large_orders_count as f64;
            if cancel_rate > 0.7 {
                return cancel_rate;
            }
        }
        
        0.0
    }
    
    /// Detect layering patterns
    async fn detect_layering(&self, order_data: &OrderFlowData) -> Result<f64> {
        if order_data.prices.len() < 10 {
            return Ok(0.0);
        }
        
        let mut layering_score = 0.0;
        let time_window = 60.0; // 60 seconds
        
        let mut i = 0;
        while i < order_data.timestamps.len() {
            let current_time = order_data.timestamps[i];
            let mut window_end = i;
            
            // Find end of time window
            while window_end < order_data.timestamps.len() && 
                  order_data.timestamps[window_end] - current_time <= time_window {
                window_end += 1;
            }
            
            if window_end - i >= 5 {
                // Analyze window for layering
                let window_prices = &order_data.prices.slice(ndarray::s![i..window_end]);
                let window_sizes = &order_data.sizes.slice(ndarray::s![i..window_end]);
                
                let unique_prices = self.count_unique_prices(window_prices);
                
                if unique_prices >= 3 {
                    let size_variance = self.calculate_variance(window_sizes);
                    let mean_size = window_sizes.mean().unwrap_or(0.0);
                    let normalized_variance = if mean_size > 0.0 {
                        size_variance / (mean_size * mean_size)
                    } else {
                        f64::INFINITY
                    };
                    
                    if normalized_variance < 0.1 {
                        layering_score += 1.0;
                    }
                }
            }
            
            i = window_end.max(i + 1);
        }
        
        Ok((layering_score / 10.0_f64).min(1.0_f64))
    }
    
    /// Detect wash trading patterns
    async fn detect_wash_trading(&self, order_data: &OrderFlowData, prices: &Array1<f64>) -> Result<f64> {
        if prices.len() < 100 {
            return Ok(0.0);
        }
        
        // Calculate price volatility using parallel processing
        let returns: Vec<f64> = prices.windows(2)
            .into_iter()
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();
        
        let volatility = self.calculate_standard_deviation(&returns);
        
        // Calculate volume intensity
        if !order_data.sizes.is_empty() {
            let volume_intensity = order_data.sizes.sum() / order_data.sizes.len() as f64;
            let typical_volume = order_data.sizes.mean().unwrap_or(0.0);
            let typical_volatility = 0.02;
            
            // Low volatility + high volume = potential wash trading
            if volatility < typical_volatility * 0.5 && volume_intensity > typical_volume * 2.0 {
                let score = ((volume_intensity / typical_volume) * 
                           (typical_volatility / (volatility + 1e-8))).min(1.0);
                return Ok(score);
            }
        }
        
        Ok(0.0)
    }
    
    /// Detect pump and dump patterns
    async fn detect_pump_dump(&self, prices: &Array1<f64>) -> Result<f64> {
        if prices.len() < 200 {
            return Ok(0.0);
        }
        
        const WINDOW: usize = 20;
        
        // Calculate rolling returns
        let returns: Vec<f64> = prices.windows(2)
            .into_iter()
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();
        
        if returns.len() < WINDOW * 2 {
            return Ok(0.0);
        }
        
        // Calculate percentile thresholds
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let pump_threshold = sorted_returns[(sorted_returns.len() as f64 * 0.95) as usize];
        let dump_threshold = sorted_returns[(sorted_returns.len() as f64 * 0.05) as usize];
        
        let mut max_score: f64 = 0.0;
        
        // Look for pump-dump patterns in parallel
        let scores: Vec<f64> = (WINDOW..returns.len() - WINDOW)
            .into_par_iter()
            .map(|i| {
                // Check for pump
                let recent_returns = &returns[i - WINDOW..i];
                let recent_mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                
                if recent_mean > pump_threshold {
                    // Check for subsequent dump
                    let future_returns = &returns[i..i + WINDOW];
                    let future_mean = future_returns.iter().sum::<f64>() / future_returns.len() as f64;
                    
                    if future_mean < dump_threshold {
                        let pump_magnitude = recent_mean;
                        let dump_magnitude = future_mean.abs();
                        let pattern_strength = (pump_magnitude + dump_magnitude) / 2.0;
                        return (pattern_strength * 10.0).min(1.0);
                    }
                }
                
                0.0
            })
            .collect();
        
        max_score = scores.into_iter().fold(0.0f64, f64::max);
        
        Ok(max_score)
    }
    
    /// Detect front-running patterns
    async fn detect_front_running(&self, order_data: &OrderFlowData, prices: &Array1<f64>) -> Result<f64> {
        if order_data.timestamps.len() < 50 || prices.len() < 50 {
            return Ok(0.0);
        }
        
        // Identify large price moves
        let price_changes: Vec<f64> = prices.windows(2)
            .into_iter()
            .map(|window| (window[1] - window[0]).abs())
            .collect();
        
        let mut sorted_changes = price_changes.clone();
        sorted_changes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let large_move_threshold = sorted_changes[(sorted_changes.len() as f64 * 0.9) as usize];
        
        let large_moves: Vec<usize> = price_changes.iter()
            .enumerate()
            .filter(|(_, &change)| change > large_move_threshold)
            .map(|(i, _)| i)
            .collect();
        
        if large_moves.is_empty() {
            return Ok(0.0);
        }
        
        let mut max_score: f64 = 0.0;
        
        // Check for suspicious order patterns before large moves
        for &move_idx in &large_moves {
            if move_idx < 10 {
                continue;
            }
            
            let move_time = if move_idx < order_data.timestamps.len() {
                order_data.timestamps[move_idx]
            } else {
                continue;
            };
            
            // Look at orders in the 5 seconds before the move
            let pre_move_orders: Vec<f64> = order_data.timestamps.iter()
                .zip(order_data.sizes.iter())
                .filter(|(&timestamp, _)| timestamp < move_time && timestamp > move_time - 5.0)
                .map(|(_, &size)| size)
                .collect();
            
            if !pre_move_orders.is_empty() {
                let max_pre_move_size = pre_move_orders.iter().fold(0.0f64, |a, &b| a.max(b));
                let average_size = order_data.sizes.mean().unwrap_or(0.0);
                
                if average_size > 0.0 {
                    let size_anomaly = max_pre_move_size / average_size;
                    if size_anomaly > 3.0 {
                        max_score = max_score.max((size_anomaly / 10.0).min(1.0));
                    }
                }
            }
        }
        
        Ok(max_score)
    }
    
    /// Helper function to count unique prices in a window
    fn count_unique_prices(&self, prices: &ArrayView1<f64>) -> usize {
        let mut unique_prices = std::collections::HashSet::new();
        for &price in prices.iter() {
            unique_prices.insert(price.to_bits()); // Use bit representation for exact comparison
        }
        unique_prices.len()
    }
    
    /// Helper function to calculate variance
    fn calculate_variance(&self, data: &ArrayView1<f64>) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance
    }
    
    /// Helper function to calculate standard deviation
    fn calculate_standard_deviation(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance.sqrt()
    }
    
    /// Recommend action based on manipulation detection
    fn recommend_action(&self, manipulation_scores: &HashMap<String, f64>, overall_score: f64) -> String {
        if overall_score < 0.3 {
            "CONTINUE_NORMAL".to_string()
        } else if overall_score < 0.5 {
            "INCREASE_VIGILANCE".to_string()
        } else if overall_score < 0.7 {
            "ADJUST_STRATEGY".to_string()
        } else {
            // High manipulation detected
            let primary_pattern = manipulation_scores
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.as_str())
                .unwrap_or("unknown");
            
            match primary_pattern {
                "spoofing" => "IGNORE_LARGE_ORDERS".to_string(),
                "layering" => "FOCUS_ON_EXECUTED_ORDERS".to_string(),
                "wash_trading" => "REDUCE_POSITION_SIZE".to_string(),
                "pump_dump" => "AVOID_MOMENTUM_TRADING".to_string(),
                "front_running" => "RANDOMIZE_EXECUTION_TIMING".to_string(),
                _ => "DEFENSIVE_TRADING".to_string(),
            }
        }
    }
    
    /// Update detection statistics
    fn update_detection_stats(&mut self, detected: bool, execution_time: f64) {
        self.detection_stats.total_detections += 1;
        self.detection_stats.detection_latency.push(execution_time);
        
        // Update average latency
        let latency_sum: f64 = self.detection_stats.detection_latency.iter().sum();
        self.detection_stats.average_latency = latency_sum / self.detection_stats.detection_latency.len() as f64;
        
        // Keep only recent latency measurements
        if self.detection_stats.detection_latency.len() > 1000 {
            self.detection_stats.detection_latency.remove(0);
        }
        
        if detected {
            // Would update true_positives based on ground truth data if available
        }
    }
    
    /// Generate strategic response based on detected manipulation
    pub async fn generate_strategy(
        &self,
        manipulation_detected: &ManipulationDetectionResult,
        competitors: &HashMap<String, f64>,
    ) -> Result<StrategyRecommendation> {
        let mut strategy = StrategyRecommendation {
            action: StrategyAction::Hold,
            confidence: 0.5,
            reasoning: "Default strategy".to_string(),
            tactics: Vec::new(),
        };
        
        // Adjust strategy based on manipulation
        if manipulation_detected.detected {
            let confidence_penalty = manipulation_detected.confidence * 0.3;
            strategy.confidence = (0.5 - confidence_penalty).max(0.1);
            
            match manipulation_detected.primary_pattern.as_str() {
                "pump_dump" => {
                    strategy.action = StrategyAction::Sell;
                    strategy.tactics.push("EXIT_BEFORE_DUMP".to_string());
                    strategy.reasoning = "Pump-dump pattern detected, selling to avoid losses".to_string();
                }
                "spoofing" => {
                    strategy.tactics.push("IGNORE_FAKE_ORDERS".to_string());
                    strategy.reasoning = "Spoofing detected, ignoring deceptive order book signals".to_string();
                }
                "front_running" => {
                    strategy.tactics.push("DELAYED_EXECUTION".to_string());
                    strategy.reasoning = "Front-running detected, randomizing execution timing".to_string();
                }
                _ => {
                    strategy.reasoning = format!("Manipulation pattern '{}' detected", manipulation_detected.primary_pattern);
                }
            }
        }
        
        // Consider competitive landscape
        if competitors.len() > 5 {
            strategy.tactics.push("INCREASE_DECEPTION".to_string());
            
            // Enable strategic deception if configured
            if self.deception_strategies.get("noise_injection")
                .and_then(|config| config.get("enabled"))
                .map(|&enabled| enabled > 0.0)
                .unwrap_or(false) {
                strategy.tactics.push("INJECT_NOISE".to_string());
            }
        }
        
        Ok(strategy)
    }
    
    /// Get current state for serialization
    pub fn get_state(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        
        state.insert("sensitivity".to_string(), 
                    serde_json::to_value(self.sensitivity).unwrap());
        state.insert("manipulation_patterns".to_string(), 
                    serde_json::to_value(&self.manipulation_patterns).unwrap());
        state.insert("deception_strategies".to_string(), 
                    serde_json::to_value(&self.deception_strategies).unwrap());
        state.insert("detection_stats".to_string(), 
                    serde_json::to_value(&self.detection_stats).unwrap());
        
        state
    }
    
    /// Set state from serialized data
    pub fn set_state(&mut self, state: &HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(sensitivity) = state.get("sensitivity") {
            self.sensitivity = serde_json::from_value(sensitivity.clone())?;
        }
        
        if let Some(patterns) = state.get("manipulation_patterns") {
            self.manipulation_patterns = serde_json::from_value(patterns.clone())?;
        }
        
        if let Some(strategies) = state.get("deception_strategies") {
            self.deception_strategies = serde_json::from_value(strategies.clone())?;
        }
        
        if let Some(stats) = state.get("detection_stats") {
            self.detection_stats = serde_json::from_value(stats.clone())?;
        }
        
        Ok(())
    }
}

/// Order flow data structure for analysis
#[derive(Debug, Clone)]
struct OrderFlowData {
    sizes: Array1<f64>,
    prices: Array1<f64>,
    timestamps: Array1<f64>,
    sides: Array1<f64>,
    cancelled: Vec<bool>,
}

impl OrderFlowData {
    fn empty() -> Self {
        Self {
            sizes: Array1::zeros(0),
            prices: Array1::zeros(0),
            timestamps: Array1::zeros(0),
            sides: Array1::zeros(0),
            cancelled: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HardwareConfig;
    
    #[tokio::test]
    async fn test_machiavellian_framework_creation() {
        let config = HardwareConfig::default();
        let framework = MachiavellianFramework::new(config, 0.7);
        assert!(framework.is_ok());
    }
    
    #[tokio::test]
    async fn test_manipulation_detection_empty_data() {
        let config = HardwareConfig::default();
        let mut framework = MachiavellianFramework::new(config, 0.7).unwrap();
        
        let result = framework.detect_manipulation(&[], &[]).await;
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        assert!(!detection.detected);
        assert_eq!(detection.confidence, 0.0);
    }
    
    #[test]
    fn test_spoofing_score_calculation() {
        let config = HardwareConfig::default();
        let framework = MachiavellianFramework::new(config, 0.7).unwrap();
        
        let sizes = Array1::from_vec(vec![1.0, 2.0, 10.0, 1.0, 1.0]);
        let cancelled = vec![false, false, true, false, false]; // Large order cancelled
        
        let score = framework.calculate_spoofing_score(&sizes.view(), &cancelled);
        assert!(score > 0.0);
    }
}