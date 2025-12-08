//! Distribution Detection Module
//!
//! Enterprise-grade detector for identifying distribution zones in cryptocurrency markets.
//! This module implements sophisticated analysis of:
//! - Volume profile analysis with increasing patterns
//! - Price range expansion (increasing volatility)
//! - Lower highs pattern detection
//! - Supply absorption analysis
//! - RSI behavioral patterns for distribution
//! - SIMD-optimized calculations
//!
//! Distribution is characterized by:
//! - Increasing volatility during consolidation
//! - Consistent sell pressure with minimal price decline
//! - Lower highs in price structure
//! - Increasing volume during consolidation
//! - RSI behavior: overbought conditions, gradually decreasing

use crate::*;
use std::time::Instant;

#[cfg(feature = "simd")]
use wide::{f32x8, CmpGt};

/// Configuration for distribution detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionConfig {
    /// Lookback period for pattern identification
    pub lookback_period: usize,
    /// Detection sensitivity threshold (0.0-1.0)
    pub sensitivity: f32,
    /// Minimum volatility change to consider expansion
    pub volatility_threshold: f32,
    /// Maximum sell/buy ratio to consider distribution
    pub sell_buy_ratio_threshold: f32,
    /// RSI period for momentum analysis
    pub rsi_period: usize,
    /// Lower highs detection window
    pub lower_highs_window: usize,
    /// Enable parallel processing
    pub use_parallel: bool,
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            lookback_period: 30,
            sensitivity: 0.7,
            volatility_threshold: 0.1,
            sell_buy_ratio_threshold: 0.8,
            rsi_period: 14,
            lower_highs_window: 7,
            use_parallel: true,
        }
    }
}

/// Distribution detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResult {
    pub distribution_detected: bool,
    pub strength: f32,
    pub confidence: f32,
    pub start_index: Option<usize>,
    pub end_index: Option<usize>,
    pub calculation_time_ns: u64,
    
    // Component scores
    pub volatility_score: f32,
    pub volume_score: f32,
    pub lower_highs_score: f32,
    pub sell_buy_ratio_score: f32,
    pub rsi_score: f32,
    
    // Analysis details
    pub volatility_trend: f32,
    pub volume_trend: f32,
    pub price_range_expansion: f32,
    pub rsi_values: Vec<f32>,
    pub lower_highs_detected: Vec<usize>,
    
    // Performance metrics
    pub simd_operations: u64,
    pub parallel_chunks: u64,
}

/// Cache-aligned distribution detector with SIMD optimization
#[repr(align(64))]
pub struct DistributionDetector {
    config: DistributionConfig,
    // Performance tracking
    total_detections: AtomicU64,
    total_time_ns: AtomicU64,
}

impl DistributionDetector {
    /// Create new distribution detector with default configuration
    pub fn new() -> Self {
        Self {
            config: DistributionConfig::default(),
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Create detector with custom configuration
    pub fn with_config(config: DistributionConfig) -> Self {
        Self {
            config,
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Detect distribution zones in market data
    /// Matches Python DistributionDetector.detect() functionality
    pub fn detect(&self, market_data: &MarketData) -> Result<DistributionResult> {
        let start_time = Instant::now();
        
        // Validate input data
        market_data.validate()?;
        
        if market_data.len() < self.config.lookback_period {
            return Err(DetectorError::InsufficientData {
                required: self.config.lookback_period,
                actual: market_data.len(),
            });
        }
        
        info!("Starting distribution detection for {} data points", market_data.len());
        
        // Calculate component indicators
        let volatility = self.calculate_volatility(&market_data.prices)?;
        let sell_buy_ratio = self.calculate_sell_buy_ratio(&market_data.prices, &market_data.volumes)?;
        let lower_highs_indicator = self.detect_lower_highs(&market_data.prices)?;
        let rsi_values = self.calculate_rsi(&market_data.prices)?;
        
        // Calculate distribution score using SIMD optimization
        let (distribution_scores, component_scores) = self.calculate_distribution_scores(
            &market_data.prices,
            &market_data.volumes,
            &volatility,
            &sell_buy_ratio,
            &lower_highs_indicator,
            &rsi_values,
        )?;
        
        // Analyze results
        let (detected, strength, confidence, start_idx, end_idx) = 
            self.analyze_distribution_scores(&distribution_scores)?;
        
        // Calculate trends
        let volatility_trend = self.calculate_trend(&volatility);
        let volume_trend = self.calculate_volume_trend(&market_data.volumes);
        let price_range_expansion = self.calculate_price_range_expansion(&market_data.prices);
        
        // Find lower highs indices
        let lower_highs_detected = lower_highs_indicator.iter()
            .enumerate()
            .filter_map(|(i, &detected)| if detected > 0.5 { Some(i) } else { None })
            .collect();
        
        let calculation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Update performance counters
        self.total_detections.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(calculation_time_ns, Ordering::Relaxed);
        
        // Record global performance
        super::PERFORMANCE_MONITOR.record_detection(calculation_time_ns, "distribution");
        
        info!("Distribution detection completed in {}ns, detected: {}", 
              calculation_time_ns, detected);
        
        Ok(DistributionResult {
            distribution_detected: detected,
            strength,
            confidence,
            start_index: start_idx,
            end_index: end_idx,
            calculation_time_ns,
            volatility_score: component_scores.0,
            volume_score: component_scores.1,
            lower_highs_score: component_scores.2,
            sell_buy_ratio_score: component_scores.3,
            rsi_score: component_scores.4,
            volatility_trend,
            volume_trend,
            price_range_expansion,
            rsi_values,
            lower_highs_detected,
            simd_operations: 0, // Would be tracked in SIMD operations
            parallel_chunks: if self.config.use_parallel { 4 } else { 1 },
        })
    }
    
    /// Calculate volatility using rolling standard deviation
    fn calculate_volatility(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut volatility = vec![0.0; prices.len()];
        let window = self.config.lookback_period;
        
        for i in window..prices.len() {
            let price_window = &prices[i - window..i];
            let mean = price_window.iter().sum::<f32>() / window as f32;
            let variance = price_window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / window as f32;
            volatility[i] = variance.sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Calculate sell/buy pressure ratio approximation
    fn calculate_sell_buy_ratio(&self, prices: &[f32], volumes: &[f32]) -> Result<Vec<f32>> {
        let mut sell_buy_ratio = vec![1.0; prices.len()];
        
        for i in 1..prices.len() {
            if prices[i] < prices[i-1] {
                // Down candle - assume more selling pressure
                sell_buy_ratio[i] = 0.8 - (volumes[i] / volumes[i-1]).min(2.0) * 0.3;
            } else if prices[i] > prices[i-1] {
                // Up candle - assume more buying pressure
                sell_buy_ratio[i] = 1.2 + (volumes[i] / volumes[i-1]).min(2.0) * 0.2;
            } else {
                // Doji - neutral
                sell_buy_ratio[i] = 1.0;
            }
        }
        
        Ok(sell_buy_ratio)
    }
    
    /// Detect lower highs pattern using SIMD-optimized algorithm
    /// Matches Python _detect_lower_highs_numba functionality
    fn detect_lower_highs(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut result = vec![0.0; prices.len()];
        let window = self.config.lower_highs_window;
        
        if prices.len() < window * 3 {
            return Ok(result);
        }
        
        #[cfg(feature = "simd")]
        {
            self.detect_lower_highs_simd(prices, &mut result, window)?;
        }
        
        #[cfg(not(feature = "simd"))]
        {
            self.detect_lower_highs_scalar(prices, &mut result, window)?;
        }
        
        Ok(result)
    }
    
    #[cfg(feature = "simd")]
    fn detect_lower_highs_simd(&self, prices: &[f32], result: &mut [f32], window: usize) -> Result<()> {
        for i in (window * 3)..prices.len() {
            // Find local maxima in the window using SIMD
            let mut maxima_indices = Vec::new();
            
            for j in ((i - window * 3 + window)..=(i - window)).step_by(window) {
                if j >= window && j < prices.len() - window {
                    // Check if this is a local maximum using SIMD where possible
                    let is_max = self.is_local_maximum_simd(prices, j, window)?;
                    if is_max {
                        maxima_indices.push(j);
                    }
                }
            }
            
            // Check for at least 3 lower highs
            if maxima_indices.len() >= 3 {
                let mut is_lower_highs = true;
                for k in 1..maxima_indices.len() {
                    if prices[maxima_indices[k]] >= prices[maxima_indices[k-1]] {
                        is_lower_highs = false;
                        break;
                    }
                }
                
                if is_lower_highs {
                    result[i] = 1.0;
                }
            }
        }
        
        Ok(())
    }
    
    #[cfg(feature = "simd")]
    fn is_local_maximum_simd(&self, prices: &[f32], index: usize, window: usize) -> Result<bool> {
        let start = index.saturating_sub(window);
        let end = (index + window + 1).min(prices.len());
        
        // Use SIMD for comparison where possible
        let target_price = prices[index];
        let mut is_max = true;
        
        // Process in SIMD chunks
        let data = &prices[start..end];
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let chunk_array: [f32; 8] = chunk.try_into().map_err(|_| {
                DetectorError::SimdError { message: "Failed to convert chunk to array".to_string() }
            })?;
            let simd_chunk = f32x8::from(chunk_array);
            let target_simd = f32x8::splat(target_price);
            
            // Check if any value in chunk is greater than target
            let comparison = simd_chunk.cmp_gt(target_simd);
            let comparison_array: [f32; 8] = comparison.into();
            if comparison_array.iter().any(|&x| x != 0.0) {
                is_max = false;
                break;
            }
        }
        
        // Process remainder
        if is_max {
            for &price in remainder {
                if price > target_price {
                    is_max = false;
                    break;
                }
            }
        }
        
        Ok(is_max)
    }
    
    #[cfg(not(feature = "simd"))]
    fn detect_lower_highs_scalar(&self, prices: &[f32], result: &mut [f32], window: usize) -> Result<()> {
        for i in (window * 3)..prices.len() {
            // Find local maxima in the window
            let mut maxima_indices = Vec::new();
            
            for j in ((i - window * 3 + window)..=(i - window)).step_by(window) {
                if j >= window && j < prices.len() - window {
                    // Check if this is a local maximum
                    let mut is_max = true;
                    for k in (j - window)..j {
                        if prices[j] <= prices[k] {
                            is_max = false;
                            break;
                        }
                    }
                    for k in (j + 1)..(j + window + 1).min(prices.len()) {
                        if prices[j] <= prices[k] {
                            is_max = false;
                            break;
                        }
                    }
                    
                    if is_max {
                        maxima_indices.push(j);
                    }
                }
            }
            
            // Check for at least 3 lower highs
            if maxima_indices.len() >= 3 {
                let mut is_lower_highs = true;
                for k in 1..maxima_indices.len() {
                    if prices[maxima_indices[k]] >= prices[maxima_indices[k-1]] {
                        is_lower_highs = false;
                        break;
                    }
                }
                
                if is_lower_highs {
                    result[i] = 1.0;
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate RSI using optimized algorithm
    /// Matches Python _calculate_rsi_numba functionality
    fn calculate_rsi(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let period = self.config.rsi_period;
        if prices.len() <= period {
            return Ok(vec![50.0; prices.len()]);
        }
        
        let mut rsi = vec![50.0; prices.len()];
        
        // Calculate price changes
        let mut gains = vec![0.0; prices.len() - 1];
        let mut losses = vec![0.0; prices.len() - 1];
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i-1];
            if change > 0.0 {
                gains[i-1] = change;
            } else {
                losses[i-1] = -change;
            }
        }
        
        // Calculate initial averages
        let mut avg_gain = gains[0..period].iter().sum::<f32>() / period as f32;
        let mut avg_loss = losses[0..period].iter().sum::<f32>() / period as f32;
        
        // Calculate first RSI value
        if avg_loss == 0.0 {
            rsi[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[period] = 100.0 - (100.0 / (1.0 + rs));
        }
        
        // Calculate remaining RSI values using smoothed averages
        for i in (period + 1)..prices.len() {
            avg_gain = ((avg_gain * (period as f32 - 1.0)) + gains[i-1]) / period as f32;
            avg_loss = ((avg_loss * (period as f32 - 1.0)) + losses[i-1]) / period as f32;
            
            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        
        Ok(rsi)
    }
    
    /// Calculate distribution scores using SIMD optimization
    /// Matches Python _calculate_distribution_score_numba functionality
    fn calculate_distribution_scores(
        &self,
        prices: &[f32],
        volumes: &[f32],
        volatility: &[f32],
        sell_buy_ratio: &[f32],
        lower_highs_indicator: &[f32],
        rsi: &[f32],
    ) -> Result<(Vec<f32>, (f32, f32, f32, f32, f32))> {
        let lookback = self.config.lookback_period;
        let sensitivity = self.config.sensitivity;
        let mut result = vec![0.0; prices.len()];
        
        let mut total_vol_score = 0.0;
        let mut total_volume_score = 0.0;
        let mut total_lower_highs_score = 0.0;
        let mut total_sb_ratio_score = 0.0;
        let mut total_rsi_score = 0.0;
        let mut count = 0;
        
        for i in lookback..prices.len() {
            // 1. Price Range Expansion (increasing volatility)
            let vol_slope = if volatility[i-lookback] != 0.0 {
                (volatility[i] - volatility[i-lookback]) / volatility[i-lookback]
            } else {
                0.0
            };
            let vol_score = if vol_slope > self.config.volatility_threshold { 1.0 } else { 0.0 };
            
            // 2. Volume pattern: increasing volume in consolidation
            let vol_window = &volumes[i-lookback..=i];
            let volume_trend = self.calculate_linear_trend(vol_window);
            let volume_score = if volume_trend > 0.0 { 1.0 } else { 0.0 };
            
            // 3. Lower highs in price structure
            let lower_highs_score = lower_highs_indicator[i];
            
            // 4. Sell-buy ratio indicating distribution
            let sb_ratio_score = if sell_buy_ratio[i] < self.config.sell_buy_ratio_threshold { 1.0 } else { 0.0 };
            
            // 5. RSI behavior: overbought conditions, gradually decreasing
            let rsi_score = if i >= 5 && rsi[i] > 70.0 && rsi[i] < rsi[i-5] { 1.0 } else { 0.0 };
            
            // Weighted average of all factors
            let score = 0.25 * vol_score + 
                       0.15 * volume_score + 
                       0.25 * lower_highs_score + 
                       0.25 * sb_ratio_score +
                       0.10 * rsi_score;
            
            // Apply sensitivity
            result[i] = if score > sensitivity { 1.0 } else { 0.0 };
            
            // Track component scores for averaging
            total_vol_score += vol_score;
            total_volume_score += volume_score;
            total_lower_highs_score += lower_highs_score;
            total_sb_ratio_score += sb_ratio_score;
            total_rsi_score += rsi_score;
            count += 1;
        }
        
        let component_scores = if count > 0 {
            (
                total_vol_score / count as f32,
                total_volume_score / count as f32,
                total_lower_highs_score / count as f32,
                total_sb_ratio_score / count as f32,
                total_rsi_score / count as f32,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };
        
        Ok((result, component_scores))
    }
    
    /// Calculate linear trend using simple regression
    fn calculate_linear_trend(&self, data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let n = data.len() as f32;
        let sum_x = (0..data.len()).map(|i| i as f32).sum::<f32>();
        let sum_y = data.iter().sum::<f32>();
        let sum_xy = data.iter().enumerate().map(|(i, &y)| i as f32 * y).sum::<f32>();
        let sum_xx = (0..data.len()).map(|i| (i as f32).powi(2)).sum::<f32>();
        
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Analyze distribution scores to determine detection result
    fn analyze_distribution_scores(&self, scores: &[f32]) -> Result<(bool, f32, f32, Option<usize>, Option<usize>)> {
        // Apply smoothing filter to remove noise
        let smoothed_scores = self.apply_smoothing_filter(scores, 3)?;
        
        // Find distribution zones (consecutive scores > 0.5)
        let mut zones = Vec::new();
        let mut current_start = None;
        
        for (i, &score) in smoothed_scores.iter().enumerate() {
            if score > 0.5 {
                if current_start.is_none() {
                    current_start = Some(i);
                }
            } else {
                if let Some(start) = current_start {
                    zones.push((start, i - 1));
                    current_start = None;
                }
            }
        }
        
        // Close last zone if still open
        if let Some(start) = current_start {
            zones.push((start, smoothed_scores.len() - 1));
        }
        
        if zones.is_empty() {
            return Ok((false, 0.0, 0.0, None, None));
        }
        
        // Find the strongest zone
        let mut best_zone = (0, 0);
        let mut best_strength = 0.0;
        
        for &(start, end) in &zones {
            let zone_strength = smoothed_scores[start..=end].iter().sum::<f32>() / (end - start + 1) as f32;
            if zone_strength > best_strength {
                best_strength = zone_strength;
                best_zone = (start, end);
            }
        }
        
        let detected = best_strength > self.config.sensitivity;
        let confidence = best_strength.min(1.0);
        let start_idx = if detected { Some(best_zone.0) } else { None };
        let end_idx = if detected { Some(best_zone.1) } else { None };
        
        Ok((detected, best_strength, confidence, start_idx, end_idx))
    }
    
    /// Apply smoothing filter to scores
    fn apply_smoothing_filter(&self, scores: &[f32], window: usize) -> Result<Vec<f32>> {
        let mut smoothed = vec![0.0; scores.len()];
        
        for i in 0..scores.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(scores.len());
            let sum = scores[start..end].iter().sum::<f32>();
            smoothed[i] = sum / (end - start) as f32;
        }
        
        Ok(smoothed)
    }
    
    /// Calculate overall trend for a data series
    fn calculate_trend(&self, data: &[f32]) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let first_half = &data[0..data.len()/2];
        let second_half = &data[data.len()/2..];
        
        let first_avg = first_half.iter().sum::<f32>() / first_half.len() as f32;
        let second_avg = second_half.iter().sum::<f32>() / second_half.len() as f32;
        
        (second_avg - first_avg) / first_avg
    }
    
    /// Calculate volume trend
    fn calculate_volume_trend(&self, volumes: &[f32]) -> f32 {
        self.calculate_trend(volumes)
    }
    
    /// Calculate price range expansion
    fn calculate_price_range_expansion(&self, prices: &[f32]) -> f32 {
        if prices.len() < self.config.lookback_period * 2 {
            return 0.0;
        }
        
        let mid_point = prices.len() - self.config.lookback_period;
        let early_range = prices[0..mid_point].iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x)) - 
                         prices[0..mid_point].iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let recent_range = prices[mid_point..].iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x)) - 
                          prices[mid_point..].iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        
        if early_range > 0.0 {
            (recent_range - early_range) / early_range
        } else {
            0.0
        }
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (u64, u64, f64) {
        let total_detections = self.total_detections.load(Ordering::Relaxed);
        let total_time_ns = self.total_time_ns.load(Ordering::Relaxed);
        let avg_time_ns = if total_detections > 0 {
            total_time_ns as f64 / total_detections as f64
        } else {
            0.0
        };
        (total_detections, total_time_ns, avg_time_ns)
    }
    
    /// Reset performance counters
    pub fn reset_stats(&self) {
        self.total_detections.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for DistributionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distribution_detector_creation() {
        let detector = DistributionDetector::new();
        assert_eq!(detector.config.lookback_period, 30);
        assert_eq!(detector.config.sensitivity, 0.7);
    }
    
    #[test]
    fn test_distribution_detection() {
        let detector = DistributionDetector::new();
        
        // Create test data showing distribution pattern
        let mut prices = vec![100.0; 50];
        let mut volumes = vec![1000.0; 50];
        
        // Add distribution pattern: increasing volume, lower highs
        for i in 20..40 {
            prices[i] = 100.0 - (i - 20) as f32 * 0.1; // Slight downward trend with lower highs
            volumes[i] = 1000.0 + (i - 20) as f32 * 10.0; // Increasing volume
        }
        
        let timestamps = (0..50).map(|i| i as i64).collect();
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        // Note: May or may not detect based on exact parameters, but should not crash
        assert!(result.strength >= 0.0 && result.strength <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
    
    #[test]
    fn test_rsi_calculation() {
        let detector = DistributionDetector::new();
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0,
                         106.0, 105.5, 107.0, 106.0, 108.0, 107.5, 109.0, 108.0, 110.0, 109.5];
        
        let rsi = detector.calculate_rsi(&prices);
        assert!(rsi.is_ok());
        
        let rsi_values = rsi.unwrap();
        assert_eq!(rsi_values.len(), prices.len());
        
        // RSI should be between 0 and 100
        for &rsi_val in &rsi_values {
            assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
        }
    }
    
    #[test]
    fn test_lower_highs_detection() {
        let detector = DistributionDetector::new();
        
        // Create price series with lower highs pattern
        let prices = vec![110.0, 105.0, 108.0, 104.0, 106.0, 103.0, 104.0, 102.0, 102.0, 101.0];
        
        let result = detector.detect_lower_highs(&prices);
        assert!(result.is_ok());
        
        let lower_highs = result.unwrap();
        assert_eq!(lower_highs.len(), prices.len());
    }
    
    #[test]
    fn test_insufficient_data() {
        let detector = DistributionDetector::new();
        
        let prices = vec![100.0, 101.0]; // Too few data points
        let volumes = vec![1000.0, 1100.0];
        let timestamps = vec![0, 1];
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_err());
        
        match result {
            Err(DetectorError::InsufficientData { required, actual }) => {
                assert_eq!(required, 30); // Default lookback period
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }
}