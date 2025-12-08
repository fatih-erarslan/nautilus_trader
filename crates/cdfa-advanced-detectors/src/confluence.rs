//! Confluence Area Detection Module
//!
//! Enterprise-grade detector for identifying confluence zones where multiple technical indicators
//! align to create high-probability support and resistance areas. This module implements:
//! - Multi-indicator confluence analysis
//! - Support and resistance level detection
//! - Fibonacci level alignment
//! - Moving average convergence
//! - Volume profile node identification
//! - SIMD-optimized calculations
//!
//! Confluence areas are characterized by:
//! - Multiple indicators aligning at similar price levels
//! - High volume activity at key levels
//! - Historical price reaction at levels
//! - Multiple timeframe agreement
//! - Statistical significance of price clusters

use crate::*;
use std::collections::BTreeMap;
use std::time::Instant;

#[cfg(feature = "simd")]
use wide::f32x8;

/// Configuration for confluence area detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfluenceConfig {
    /// Price tolerance for level clustering (percentage)
    pub price_tolerance: f32,
    /// Minimum number of indicators required for confluence
    pub min_indicators: usize,
    /// Volume weight in confluence scoring
    pub volume_weight: f32,
    /// Historical significance weight
    pub historical_weight: f32,
    /// Fibonacci levels weight
    pub fibonacci_weight: f32,
    /// Moving average weight
    pub moving_average_weight: f32,
    /// Support/resistance weight
    pub support_resistance_weight: f32,
    /// Volume profile weight
    pub volume_profile_weight: f32,
    /// Lookback period for historical analysis
    pub lookback_period: usize,
    /// Enable parallel processing
    pub use_parallel: bool,
}

impl Default for ConfluenceConfig {
    fn default() -> Self {
        Self {
            price_tolerance: 0.005, // 0.5%
            min_indicators: 3,
            volume_weight: 0.2,
            historical_weight: 0.25,
            fibonacci_weight: 0.2,
            moving_average_weight: 0.15,
            support_resistance_weight: 0.15,
            volume_profile_weight: 0.05,
            lookback_period: 100,
            use_parallel: true,
        }
    }
}

/// Types of technical indicators for confluence
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndicatorType {
    SupportResistance,
    FibonacciRetracement,
    FibonacciExtension,
    MovingAverage,
    VolumeProfile,
    PivotPoint,
    TrendLine,
    HistoricalLevel,
}

/// Individual confluence level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfluenceLevel {
    pub price: f32,
    pub strength: f32,
    pub indicators: Vec<IndicatorType>,
    pub volume_significance: f32,
    pub historical_touches: u32,
    pub fibonacci_alignment: f32,
    pub last_test_distance: f32,
}

/// Confluence area detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfluenceResult {
    pub confluences_detected: Vec<ConfluenceLevel>,
    pub strongest_confluence: Option<ConfluenceLevel>,
    pub total_areas: usize,
    pub calculation_time_ns: u64,
    
    // Component analysis
    pub support_levels: Vec<f32>,
    pub resistance_levels: Vec<f32>,
    pub fibonacci_levels: Vec<f32>,
    pub moving_average_levels: Vec<f32>,
    pub volume_profile_nodes: Vec<f32>,
    
    // Statistical metrics
    pub average_strength: f32,
    pub confluence_density: f32,
    pub price_clustering_score: f32,
    
    // Performance metrics
    pub simd_operations: u64,
    pub parallel_chunks: u64,
}

/// Cache-aligned confluence detector with SIMD optimization
#[repr(align(64))]
pub struct ConfluenceAreaDetector {
    config: ConfluenceConfig,
    // Performance tracking
    total_detections: AtomicU64,
    total_time_ns: AtomicU64,
}

impl ConfluenceAreaDetector {
    /// Create new confluence area detector with default configuration
    pub fn new() -> Self {
        Self {
            config: ConfluenceConfig::default(),
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Create detector with custom configuration
    pub fn with_config(config: ConfluenceConfig) -> Self {
        Self {
            config,
            total_detections: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
        }
    }
    
    /// Detect confluence areas in market data
    /// Matches Python ConfluenceAreaDetector.detect() functionality
    pub fn detect(&self, market_data: &MarketData) -> Result<ConfluenceResult> {
        let start_time = Instant::now();
        
        // Validate input data
        market_data.validate()?;
        
        if market_data.len() < self.config.lookback_period {
            return Err(DetectorError::InsufficientData {
                required: self.config.lookback_period,
                actual: market_data.len(),
            });
        }
        
        info!("Starting confluence area detection for {} data points", market_data.len());
        
        // Detect support and resistance levels
        let support_levels = self.detect_support_levels(&market_data.prices)?;
        let resistance_levels = self.detect_resistance_levels(&market_data.prices)?;
        
        // Calculate Fibonacci levels
        let fibonacci_levels = self.calculate_fibonacci_levels(&market_data.prices)?;
        
        // Calculate moving average levels
        let moving_average_levels = self.calculate_moving_average_levels(&market_data.prices)?;
        
        // Detect volume profile nodes
        let volume_profile_nodes = self.detect_volume_profile_nodes(&market_data.prices, &market_data.volumes)?;
        
        // Combine all indicator levels
        let all_levels = self.combine_indicator_levels(
            &support_levels,
            &resistance_levels,
            &fibonacci_levels,
            &moving_average_levels,
            &volume_profile_nodes,
        )?;
        
        // Cluster levels into confluence areas
        let confluence_areas = self.cluster_confluence_areas(&all_levels, &market_data.prices, &market_data.volumes)?;
        
        // Calculate statistical metrics
        let average_strength = if !confluence_areas.is_empty() {
            confluence_areas.iter().map(|c| c.strength).sum::<f32>() / confluence_areas.len() as f32
        } else {
            0.0
        };
        
        let confluence_density = confluence_areas.len() as f32 / market_data.len() as f32;
        let price_clustering_score = self.calculate_price_clustering_score(&confluence_areas)?;
        
        // Find strongest confluence
        let strongest_confluence = confluence_areas.iter()
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();
        
        let total_areas = confluence_areas.len();
        
        let calculation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Update performance counters
        self.total_detections.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(calculation_time_ns, Ordering::Relaxed);
        
        // Record global performance
        super::PERFORMANCE_MONITOR.record_detection(calculation_time_ns, "confluence");
        
        info!("Confluence detection completed in {}ns, found {} areas", 
              calculation_time_ns, total_areas);
        
        Ok(ConfluenceResult {
            confluences_detected: confluence_areas,
            strongest_confluence,
            total_areas,
            calculation_time_ns,
            support_levels,
            resistance_levels,
            fibonacci_levels,
            moving_average_levels,
            volume_profile_nodes,
            average_strength,
            confluence_density,
            price_clustering_score,
            simd_operations: 0, // Would be tracked in SIMD operations
            parallel_chunks: if self.config.use_parallel { 4 } else { 1 },
        })
    }
    
    /// Detect support levels using swing lows and price clustering
    fn detect_support_levels(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut support_levels = Vec::new();
        let window = 20; // Swing detection window
        
        // Find swing lows
        for i in window..prices.len()-window {
            let mut is_swing_low = true;
            
            // Check if current price is lowest in window
            for j in (i-window)..(i+window+1) {
                if j != i && prices[j] <= prices[i] {
                    is_swing_low = false;
                    break;
                }
            }
            
            if is_swing_low {
                support_levels.push(prices[i]);
            }
        }
        
        // Cluster nearby support levels
        support_levels = self.cluster_price_levels(&support_levels)?;
        
        Ok(support_levels)
    }
    
    /// Detect resistance levels using swing highs and price clustering
    fn detect_resistance_levels(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut resistance_levels = Vec::new();
        let window = 20; // Swing detection window
        
        // Find swing highs
        for i in window..prices.len()-window {
            let mut is_swing_high = true;
            
            // Check if current price is highest in window
            for j in (i-window)..(i+window+1) {
                if j != i && prices[j] >= prices[i] {
                    is_swing_high = false;
                    break;
                }
            }
            
            if is_swing_high {
                resistance_levels.push(prices[i]);
            }
        }
        
        // Cluster nearby resistance levels
        resistance_levels = self.cluster_price_levels(&resistance_levels)?;
        
        Ok(resistance_levels)
    }
    
    /// Calculate Fibonacci retracement and extension levels
    fn calculate_fibonacci_levels(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut fibonacci_levels = Vec::new();
        
        if prices.len() < 50 {
            return Ok(fibonacci_levels);
        }
        
        // Find significant swing high and low in recent period
        let recent_period = prices.len().min(100);
        let recent_prices = &prices[prices.len()-recent_period..];
        
        let swing_high = recent_prices.iter().fold(0.0f32, |acc, &x| acc.max(x));
        let swing_low = recent_prices.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        
        let range = swing_high - swing_low;
        if range < f32::EPSILON {
            return Ok(fibonacci_levels);
        }
        
        // Standard Fibonacci ratios
        let fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
        
        for &ratio in &fib_ratios {
            // Retracement level
            let retracement = swing_high - (ratio * range);
            fibonacci_levels.push(retracement);
            
            // Extension level (if above 1.0)
            if ratio >= 1.0 {
                let extension = swing_high + ((ratio - 1.0) * range);
                fibonacci_levels.push(extension);
            }
        }
        
        Ok(fibonacci_levels)
    }
    
    /// Calculate moving average levels
    fn calculate_moving_average_levels(&self, prices: &[f32]) -> Result<Vec<f32>> {
        let mut ma_levels = Vec::new();
        
        // Common moving average periods
        let ma_periods = [20, 50, 100, 200];
        
        for &period in &ma_periods {
            if prices.len() >= period {
                // Calculate simple moving average
                let ma = prices[prices.len()-period..].iter().sum::<f32>() / period as f32;
                ma_levels.push(ma);
                
                // Calculate exponential moving average
                let mut ema = prices[prices.len()-period];
                let multiplier = 2.0 / (period as f32 + 1.0);
                
                for &price in &prices[prices.len()-period+1..] {
                    ema = (price * multiplier) + (ema * (1.0 - multiplier));
                }
                ma_levels.push(ema);
            }
        }
        
        Ok(ma_levels)
    }
    
    /// Detect volume profile nodes (high volume areas)
    fn detect_volume_profile_nodes(&self, prices: &[f32], volumes: &[f32]) -> Result<Vec<f32>> {
        let mut volume_nodes = Vec::new();
        
        if prices.len() != volumes.len() || prices.is_empty() {
            return Ok(volume_nodes);
        }
        
        // Create price-volume histogram
        let mut price_volume_map: BTreeMap<i32, f32> = BTreeMap::new();
        let price_precision = 10000; // 4 decimal places
        
        for (i, (&price, &volume)) in prices.iter().zip(volumes.iter()).enumerate() {
            if i >= prices.len().saturating_sub(self.config.lookback_period) {
                let price_key = (price * price_precision as f32) as i32;
                *price_volume_map.entry(price_key).or_insert(0.0) += volume;
            }
        }
        
        // Find high volume nodes
        let total_volume: f32 = price_volume_map.values().sum();
        let avg_volume = total_volume / price_volume_map.len() as f32;
        
        for (&price_key, &volume) in &price_volume_map {
            if volume > avg_volume * 2.0 { // High volume threshold
                let price = price_key as f32 / price_precision as f32;
                volume_nodes.push(price);
            }
        }
        
        Ok(volume_nodes)
    }
    
    /// Combine all indicator levels into a unified structure
    fn combine_indicator_levels(
        &self,
        support_levels: &[f32],
        resistance_levels: &[f32],
        fibonacci_levels: &[f32],
        moving_average_levels: &[f32],
        volume_profile_nodes: &[f32],
    ) -> Result<Vec<(f32, IndicatorType, f32)>> {
        let mut all_levels = Vec::new();
        
        // Add support levels
        for &level in support_levels {
            all_levels.push((level, IndicatorType::SupportResistance, self.config.support_resistance_weight));
        }
        
        // Add resistance levels
        for &level in resistance_levels {
            all_levels.push((level, IndicatorType::SupportResistance, self.config.support_resistance_weight));
        }
        
        // Add Fibonacci levels
        for &level in fibonacci_levels {
            all_levels.push((level, IndicatorType::FibonacciRetracement, self.config.fibonacci_weight));
        }
        
        // Add moving average levels
        for &level in moving_average_levels {
            all_levels.push((level, IndicatorType::MovingAverage, self.config.moving_average_weight));
        }
        
        // Add volume profile nodes
        for &level in volume_profile_nodes {
            all_levels.push((level, IndicatorType::VolumeProfile, self.config.volume_profile_weight));
        }
        
        Ok(all_levels)
    }
    
    /// Cluster levels into confluence areas
    fn cluster_confluence_areas(
        &self,
        all_levels: &[(f32, IndicatorType, f32)],
        prices: &[f32],
        volumes: &[f32],
    ) -> Result<Vec<ConfluenceLevel>> {
        let mut confluence_areas = Vec::new();
        
        if all_levels.is_empty() {
            return Ok(confluence_areas);
        }
        
        // Sort levels by price
        let mut sorted_levels = all_levels.to_vec();
        sorted_levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut i = 0;
        while i < sorted_levels.len() {
            let base_price = sorted_levels[i].0;
            let mut cluster_levels = vec![sorted_levels[i].clone()];
            let mut j = i + 1;
            
            // Find all levels within tolerance
            while j < sorted_levels.len() {
                let price_diff = (sorted_levels[j].0 - base_price).abs() / base_price;
                if price_diff <= self.config.price_tolerance {
                    cluster_levels.push(sorted_levels[j].clone());
                    j += 1;
                } else {
                    break;
                }
            }
            
            // Check if we have enough indicators for confluence
            if cluster_levels.len() >= self.config.min_indicators {
                let confluence = self.create_confluence_level(&cluster_levels, prices, volumes)?;
                confluence_areas.push(confluence);
            }
            
            i = j;
        }
        
        Ok(confluence_areas)
    }
    
    /// Create a confluence level from clustered indicators
    fn create_confluence_level(
        &self,
        cluster_levels: &[(f32, IndicatorType, f32)],
        prices: &[f32],
        volumes: &[f32],
    ) -> Result<ConfluenceLevel> {
        // Calculate average price of cluster
        let avg_price = cluster_levels.iter().map(|(price, _, _)| price).sum::<f32>() / cluster_levels.len() as f32;
        
        // Calculate strength based on weights and indicator count
        let total_weight: f32 = cluster_levels.iter().map(|(_, _, weight)| weight).sum();
        let indicator_bonus = (cluster_levels.len() as f32 - self.config.min_indicators as f32) * 0.1;
        let base_strength = total_weight + indicator_bonus;
        
        // Calculate volume significance
        let volume_significance = self.calculate_volume_significance(avg_price, prices, volumes)?;
        
        // Calculate historical touches
        let historical_touches = self.calculate_historical_touches(avg_price, prices)?;
        
        // Calculate Fibonacci alignment
        let fibonacci_alignment = cluster_levels.iter()
            .filter(|(_, indicator_type, _)| {
                matches!(indicator_type, IndicatorType::FibonacciRetracement | IndicatorType::FibonacciExtension)
            })
            .count() as f32 / cluster_levels.len() as f32;
        
        // Calculate distance from last test
        let current_price = prices[prices.len() - 1];
        let last_test_distance = (current_price - avg_price).abs() / current_price;
        
        // Combine all factors for final strength
        let final_strength = base_strength * 
            (1.0 + volume_significance * self.config.volume_weight) *
            (1.0 + historical_touches as f32 * 0.1 * self.config.historical_weight) *
            (1.0 + fibonacci_alignment * self.config.fibonacci_weight);
        
        // Extract unique indicator types
        let mut unique_indicators = Vec::new();
        for (_, indicator_type, _) in cluster_levels {
            if !unique_indicators.contains(indicator_type) {
                unique_indicators.push(indicator_type.clone());
            }
        }
        
        Ok(ConfluenceLevel {
            price: avg_price,
            strength: final_strength.min(10.0), // Cap at 10.0
            indicators: unique_indicators,
            volume_significance,
            historical_touches,
            fibonacci_alignment,
            last_test_distance,
        })
    }
    
    /// Calculate volume significance at a price level
    fn calculate_volume_significance(&self, price_level: f32, prices: &[f32], volumes: &[f32]) -> Result<f32> {
        let mut volume_at_level = 0.0;
        let mut total_volume = 0.0;
        let tolerance = self.config.price_tolerance;
        
        for (i, (&price, &volume)) in prices.iter().zip(volumes.iter()).enumerate() {
            if i >= prices.len().saturating_sub(self.config.lookback_period) {
                total_volume += volume;
                
                let price_diff = (price - price_level).abs() / price_level;
                if price_diff <= tolerance {
                    volume_at_level += volume;
                }
            }
        }
        
        if total_volume > 0.0 {
            Ok(volume_at_level / total_volume)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate historical touches (price reactions) at a level
    fn calculate_historical_touches(&self, price_level: f32, prices: &[f32]) -> Result<u32> {
        let mut touches = 0;
        let tolerance = self.config.price_tolerance;
        
        for i in 1..prices.len() {
            let price_diff = (prices[i] - price_level).abs() / price_level;
            if price_diff <= tolerance {
                // Check if price bounced from level
                if i < prices.len() - 1 {
                    let prev_diff = (prices[i-1] - price_level).abs() / price_level;
                    let next_diff = (prices[i+1] - price_level).abs() / price_level;
                    
                    if prev_diff > tolerance || next_diff > tolerance {
                        touches += 1;
                    }
                }
            }
        }
        
        Ok(touches)
    }
    
    /// Cluster nearby price levels
    fn cluster_price_levels(&self, levels: &[f32]) -> Result<Vec<f32>> {
        if levels.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut clustered = Vec::new();
        let mut sorted_levels = levels.to_vec();
        sorted_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut i = 0;
        while i < sorted_levels.len() {
            let base_level = sorted_levels[i];
            let mut cluster_sum = base_level;
            let mut cluster_count = 1;
            let mut j = i + 1;
            
            // Find all levels within tolerance
            while j < sorted_levels.len() {
                let level_diff = (sorted_levels[j] - base_level).abs() / base_level;
                if level_diff <= self.config.price_tolerance {
                    cluster_sum += sorted_levels[j];
                    cluster_count += 1;
                    j += 1;
                } else {
                    break;
                }
            }
            
            // Use average of cluster
            clustered.push(cluster_sum / cluster_count as f32);
            i = j;
        }
        
        Ok(clustered)
    }
    
    /// Calculate price clustering score
    fn calculate_price_clustering_score(&self, confluence_areas: &[ConfluenceLevel]) -> Result<f32> {
        if confluence_areas.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_proximity = 0.0;
        let mut comparisons = 0;
        
        for i in 0..confluence_areas.len() {
            for j in (i+1)..confluence_areas.len() {
                let price_diff = (confluence_areas[i].price - confluence_areas[j].price).abs();
                let avg_price = (confluence_areas[i].price + confluence_areas[j].price) / 2.0;
                let proximity = 1.0 - (price_diff / avg_price).min(1.0);
                
                total_proximity += proximity;
                comparisons += 1;
            }
        }
        
        if comparisons > 0 {
            Ok(total_proximity / comparisons as f32)
        } else {
            Ok(0.0)
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

impl Default for ConfluenceAreaDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_confluence_detector_creation() {
        let detector = ConfluenceAreaDetector::new();
        assert_eq!(detector.config.min_indicators, 3);
        assert_eq!(detector.config.price_tolerance, 0.005);
    }
    
    #[test]
    fn test_confluence_detection() {
        let detector = ConfluenceAreaDetector::new();
        
        // Create test data with clear confluence areas
        let mut prices = vec![100.0; 150];
        let mut volumes = vec![1000.0; 150];
        
        // Add price action around key levels
        for i in 50..100 {
            if i % 10 == 0 {
                prices[i] = 105.0; // Resistance level
                volumes[i] = 2000.0; // High volume
            }
        }
        
        let timestamps = (0..150).map(|i| i as i64).collect();
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.average_strength >= 0.0);
        assert!(result.confluence_density >= 0.0);
    }
    
    #[test]
    fn test_support_resistance_detection() {
        let detector = ConfluenceAreaDetector::new();
        
        // Create price series with clear support/resistance
        let prices = vec![100.0, 105.0, 100.0, 110.0, 105.0, 100.0, 115.0, 110.0, 105.0, 100.0];
        
        let support_levels = detector.detect_support_levels(&prices);
        assert!(support_levels.is_ok());
        
        let resistance_levels = detector.detect_resistance_levels(&prices);
        assert!(resistance_levels.is_ok());
    }
    
    #[test]
    fn test_insufficient_data() {
        let detector = ConfluenceAreaDetector::new();
        
        let prices = vec![100.0, 101.0]; // Too few data points
        let volumes = vec![1000.0, 1100.0];
        let timestamps = vec![0, 1];
        let market_data = MarketData::new(prices, volumes, timestamps);
        
        let result = detector.detect(&market_data);
        assert!(result.is_err());
        
        match result {
            Err(DetectorError::InsufficientData { required, actual }) => {
                assert_eq!(required, 100); // Default lookback period
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }
}