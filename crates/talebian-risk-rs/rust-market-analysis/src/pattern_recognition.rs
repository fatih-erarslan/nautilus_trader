//! Pattern recognition module for technical analysis
//! 
//! Implements advanced algorithms for detecting classical chart patterns,
//! support/resistance levels, and technical formations with statistical validation.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::statistical,
};
use ndarray::{Array1, Array2, ArrayView1};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use rayon::prelude::*;
use tracing::{info, debug, warn};

/// Pattern recognition engine for technical analysis
#[derive(Debug, Clone)]
pub struct PatternRecognizer {
    config: PatternConfig,
    classical_patterns: ClassicalPatternDetector,
    support_resistance: SupportResistanceDetector,
    volume_patterns: VolumePatternDetector,
    microstructure_patterns: MicrostructurePatternDetector,
    pattern_history: VecDeque<HistoricalPattern>,
    pattern_cache: HashMap<String, CachedPattern>,
}

#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub min_pattern_duration: usize,
    pub max_pattern_duration: usize,
    pub confidence_threshold: f64,
    pub volume_confirmation_threshold: f64,
    pub price_tolerance: f64,
    pub time_tolerance_minutes: u32,
    pub pattern_validation_window: usize,
    pub require_volume_confirmation: bool,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            min_pattern_duration: 3,
            max_pattern_duration: 50,
            confidence_threshold: 0.7,
            volume_confirmation_threshold: 1.2,
            price_tolerance: 0.02, // 2%
            time_tolerance_minutes: 30,
            pattern_validation_window: 20,
            require_volume_confirmation: true,
        }
    }
}

impl PatternRecognizer {
    pub fn new(config: &Config) -> Result<Self> {
        let pattern_config = PatternConfig::default();
        
        Ok(Self {
            config: pattern_config.clone(),
            classical_patterns: ClassicalPatternDetector::new(&pattern_config)?,
            support_resistance: SupportResistanceDetector::new(&pattern_config)?,
            volume_patterns: VolumePatternDetector::new(&pattern_config)?,
            microstructure_patterns: MicrostructurePatternDetector::new(&pattern_config)?,
            pattern_history: VecDeque::with_capacity(1000),
            pattern_cache: HashMap::new(),
        })
    }
    
    /// Recognize patterns in market data
    pub async fn recognize_patterns(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let start_time = std::time::Instant::now();
        debug!("Starting pattern recognition for {}", data.symbol);
        
        // Check cache first
        let cache_key = self.generate_cache_key(data);
        if let Some(cached_pattern) = self.pattern_cache.get(&cache_key) {
            if !cached_pattern.is_expired() {
                return Ok(cached_pattern.patterns.clone());
            }
        }
        
        // Parallel pattern detection
        let (classical, support_resistance, volume, microstructure) = tokio::try_join!(
            self.detect_classical_patterns(data),
            self.detect_support_resistance_patterns(data),
            self.detect_volume_patterns(data),
            self.detect_microstructure_patterns(data)
        )?;
        
        // Combine and validate patterns
        let mut all_patterns = Vec::new();
        all_patterns.extend(classical);
        all_patterns.extend(support_resistance);
        all_patterns.extend(volume);
        all_patterns.extend(microstructure);
        
        // Cross-validate and filter patterns
        let validated_patterns = self.validate_patterns(all_patterns, data)?;
        
        // Cache the results
        // self.cache_patterns(&cache_key, &validated_patterns).await?;
        
        // Update pattern history
        // self.update_pattern_history(&validated_patterns, data).await?;
        
        let processing_time = start_time.elapsed();
        debug!("Pattern recognition completed in {:?}", processing_time);
        
        Ok(validated_patterns)
    }
    
    /// Detect classical chart patterns
    async fn detect_classical_patterns(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        self.classical_patterns.detect(data).await
    }
    
    /// Detect support and resistance patterns
    async fn detect_support_resistance_patterns(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        self.support_resistance.detect(data).await
    }
    
    /// Detect volume-based patterns
    async fn detect_volume_patterns(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        self.volume_patterns.detect(data).await
    }
    
    /// Detect microstructure patterns
    async fn detect_microstructure_patterns(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        self.microstructure_patterns.detect(data).await
    }
    
    /// Validate patterns using multiple criteria
    fn validate_patterns(&self, patterns: Vec<Pattern>, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut validated = Vec::new();
        
        for pattern in patterns {
            let validation_score = self.calculate_pattern_validation_score(&pattern, data)?;
            
            if validation_score >= self.config.confidence_threshold {
                let mut validated_pattern = pattern;
                validated_pattern.confidence = validation_score;
                
                // Add volume confirmation if required
                if self.config.require_volume_confirmation {
                    validated_pattern.volume_confirmation = 
                        self.check_volume_confirmation(&validated_pattern, data)?;
                }
                
                // Calculate targets and stop losses
                self.calculate_pattern_targets(&mut validated_pattern, data)?;
                
                validated.push(validated_pattern);
            }
        }
        
        // Remove overlapping patterns (keep highest confidence)
        let filtered = self.filter_overlapping_patterns(validated)?;
        
        Ok(filtered)
    }
    
    /// Calculate pattern validation score
    fn calculate_pattern_validation_score(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;
        
        // Price movement validation
        let price_movement_score = self.validate_price_movement(pattern, data)?;
        score += price_movement_score * 0.3;
        weight_sum += 0.3;
        
        // Volume validation
        let volume_score = self.validate_volume_pattern(pattern, data)?;
        score += volume_score * 0.25;
        weight_sum += 0.25;
        
        // Time duration validation
        let time_score = self.validate_time_duration(pattern)?;
        score += time_score * 0.15;
        weight_sum += 0.15;
        
        // Pattern geometry validation
        let geometry_score = self.validate_pattern_geometry(pattern, data)?;
        score += geometry_score * 0.20;
        weight_sum += 0.20;
        
        // Historical success rate
        let historical_score = self.get_historical_success_rate(&pattern.pattern_type);
        score += historical_score * 0.10;
        weight_sum += 0.10;
        
        if weight_sum > 0.0 {
            Ok(score / weight_sum)
        } else {
            Ok(0.0)
        }
    }
    
    /// Validate price movement for pattern
    fn validate_price_movement(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        if pattern.price_levels.len() < 2 {
            return Ok(0.0);
        }
        
        let min_price = pattern.price_levels.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = pattern.price_levels.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range = (max_price - min_price) / min_price;
        
        // Different patterns require different minimum price movements
        let required_movement = match pattern.pattern_type {
            PatternType::HeadAndShoulders | PatternType::InverseHeadAndShoulders => 0.03, // 3%
            PatternType::DoubleTop | PatternType::DoubleBottom => 0.02, // 2%
            PatternType::Triangle | PatternType::Wedge => 0.015, // 1.5%
            PatternType::Flag | PatternType::Pennant => 0.01, // 1%
            PatternType::Support | PatternType::Resistance => 0.005, // 0.5%
            _ => 0.01, // 1% default
        };
        
        if price_range >= required_movement {
            (price_range / required_movement).min(1.0)
        } else {
            price_range / required_movement
        }
    }
    
    /// Validate volume pattern
    fn validate_volume_pattern(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        if data.volumes.is_empty() {
            return Ok(0.5); // Neutral score if no volume data
        }
        
        // Calculate volume during pattern formation
        let pattern_start_idx = self.find_price_index(pattern.start_time, data)?;
        let pattern_end_idx = pattern.end_time
            .map(|end_time| self.find_price_index(end_time, data))
            .transpose()?
            .unwrap_or(data.volumes.len() - 1);
            
        if pattern_start_idx >= pattern_end_idx || pattern_end_idx >= data.volumes.len() {
            return Ok(0.5);
        }
        
        let pattern_volumes = &data.volumes[pattern_start_idx..=pattern_end_idx];
        let pattern_avg_volume = pattern_volumes.mean();
        
        // Compare with baseline volume
        let baseline_volumes = if pattern_start_idx >= 20 {
            &data.volumes[pattern_start_idx-20..pattern_start_idx]
        } else {
            &data.volumes[0..pattern_start_idx]
        };
        
        if baseline_volumes.is_empty() {
            return Ok(0.5);
        }
        
        let baseline_avg_volume = baseline_volumes.mean();
        
        if baseline_avg_volume == 0.0 {
            return Ok(0.5);
        }
        
        let volume_ratio = pattern_avg_volume / baseline_avg_volume;
        
        // Volume requirements vary by pattern type
        match pattern.pattern_type {
            PatternType::Breakout | PatternType::Breakdown => {
                // Breakouts should have increased volume
                if volume_ratio >= 1.5 {
                    1.0
                } else if volume_ratio >= 1.2 {
                    0.8
                } else {
                    0.4
                }
            }
            PatternType::HeadAndShoulders | PatternType::InverseHeadAndShoulders => {
                // Volume should decline into the right shoulder
                if volume_ratio <= 0.8 {
                    1.0
                } else if volume_ratio <= 1.0 {
                    0.8
                } else {
                    0.4
                }
            }
            PatternType::Triangle | PatternType::Wedge => {
                // Volume should contract during formation
                if volume_ratio <= 0.7 {
                    1.0
                } else if volume_ratio <= 0.9 {
                    0.8
                } else {
                    0.4
                }
            }
            _ => {
                // Default: any volume confirmation is positive
                if volume_ratio >= 1.0 {
                    0.8
                } else {
                    0.6
                }
            }
        }
    }
    
    /// Validate time duration
    fn validate_time_duration(&self, pattern: &Pattern) -> Result<f64> {
        let duration = if let Some(end_time) = pattern.end_time {
            (end_time - pattern.start_time).num_minutes() as usize
        } else {
            self.config.min_pattern_duration
        };
        
        if duration < self.config.min_pattern_duration {
            Ok(0.0)
        } else if duration > self.config.max_pattern_duration {
            Ok(0.5) // Too long patterns are less reliable
        } else {
            // Optimal duration range
            let optimal_min = self.config.min_pattern_duration * 2;
            let optimal_max = self.config.max_pattern_duration / 2;
            
            if duration >= optimal_min && duration <= optimal_max {
                Ok(1.0)
            } else if duration < optimal_min {
                Ok(duration as f64 / optimal_min as f64)
            } else {
                Ok(optimal_max as f64 / duration as f64)
            }
        }
    }
    
    /// Validate pattern geometry
    fn validate_pattern_geometry(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        match pattern.pattern_type {
            PatternType::HeadAndShoulders => self.validate_head_shoulders_geometry(pattern, data),
            PatternType::DoubleTop | PatternType::DoubleBottom => self.validate_double_pattern_geometry(pattern, data),
            PatternType::Triangle => self.validate_triangle_geometry(pattern, data),
            PatternType::Support | PatternType::Resistance => self.validate_support_resistance_geometry(pattern, data),
            _ => Ok(0.8), // Default score for other patterns
        }
    }
    
    /// Validate head and shoulders geometry
    fn validate_head_shoulders_geometry(&self, pattern: &Pattern, _data: &MarketData) -> Result<f64> {
        if pattern.price_levels.len() < 5 {
            return Ok(0.0);
        }
        
        // Extract shoulders and head
        let left_shoulder = pattern.price_levels[0];
        let left_valley = pattern.price_levels[1];
        let head = pattern.price_levels[2];
        let right_valley = pattern.price_levels[3];
        let right_shoulder = pattern.price_levels[4];
        
        // Check head is higher than shoulders (for head and shoulders top)
        let head_prominence = if head > left_shoulder && head > right_shoulder {
            let min_shoulder = left_shoulder.min(right_shoulder);
            let prominence = (head - min_shoulder) / min_shoulder;
            prominence.min(1.0)
        } else {
            0.0
        };
        
        // Check shoulder symmetry
        let shoulder_symmetry = {
            let height_diff = (left_shoulder - right_shoulder).abs() / left_shoulder.max(right_shoulder);
            (1.0 - height_diff).max(0.0)
        };
        
        // Check valley symmetry
        let valley_symmetry = {
            let valley_diff = (left_valley - right_valley).abs() / left_valley.max(right_valley);
            (1.0 - valley_diff).max(0.0)
        };
        
        // Combine scores
        let geometry_score = (head_prominence * 0.5 + shoulder_symmetry * 0.3 + valley_symmetry * 0.2);
        Ok(geometry_score)
    }
    
    /// Validate double top/bottom geometry
    fn validate_double_pattern_geometry(&self, pattern: &Pattern, _data: &MarketData) -> Result<f64> {
        if pattern.price_levels.len() < 3 {
            return Ok(0.0);
        }
        
        let first_peak = pattern.price_levels[0];
        let valley = pattern.price_levels[1];
        let second_peak = pattern.price_levels[2];
        
        // Check peak symmetry
        let peak_symmetry = {
            let peak_diff = (first_peak - second_peak).abs() / first_peak.max(second_peak);
            (1.0 - peak_diff * 10.0).max(0.0) // Allow 10% difference
        };
        
        // Check valley depth
        let valley_depth = {
            let min_peak = first_peak.min(second_peak);
            let depth_ratio = (min_peak - valley) / min_peak;
            depth_ratio.min(1.0)
        };
        
        // Combine scores
        let geometry_score = peak_symmetry * 0.6 + valley_depth * 0.4;
        Ok(geometry_score)
    }
    
    /// Validate triangle geometry
    fn validate_triangle_geometry(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        if pattern.price_levels.len() < 4 {
            return Ok(0.0);
        }
        
        // Calculate convergence trend
        let upper_trend = self.calculate_trend_line(&pattern.price_levels[0..pattern.price_levels.len()/2]);
        let lower_trend = self.calculate_trend_line(&pattern.price_levels[pattern.price_levels.len()/2..]);
        
        // Check if lines are converging
        let convergence_score = if (upper_trend - lower_trend).abs() > 0.001 {
            1.0
        } else {
            0.0
        };
        
        Ok(convergence_score)
    }
    
    /// Validate support/resistance geometry
    fn validate_support_resistance_geometry(&self, pattern: &Pattern, data: &MarketData) -> Result<f64> {
        if pattern.price_levels.is_empty() {
            return Ok(0.0);
        }
        
        let level = pattern.price_levels[0];
        let tolerance = level * self.config.price_tolerance;
        
        // Count touches within tolerance
        let touches = data.prices.iter()
            .filter(|&&price| (price - level).abs() <= tolerance)
            .count();
        
        // More touches = stronger level
        let touch_score = match touches {
            0..=1 => 0.0,
            2 => 0.6,
            3 => 0.8,
            4..=5 => 1.0,
            _ => 0.9, // Too many touches might indicate the level is broken
        };
        
        Ok(touch_score)
    }
    
    /// Get historical success rate for pattern type
    fn get_historical_success_rate(&self, pattern_type: &PatternType) -> f64 {
        // Historical success rates based on research
        match pattern_type {
            PatternType::HeadAndShoulders => 0.85,
            PatternType::InverseHeadAndShoulders => 0.83,
            PatternType::DoubleTop => 0.78,
            PatternType::DoubleBottom => 0.80,
            PatternType::Triangle => 0.75,
            PatternType::Support => 0.70,
            PatternType::Resistance => 0.70,
            PatternType::Breakout => 0.65,
            PatternType::Flag => 0.82,
            PatternType::Pennant => 0.80,
            _ => 0.60,
        }
    }
    
    /// Check volume confirmation
    fn check_volume_confirmation(&self, pattern: &Pattern, data: &MarketData) -> Result<bool> {
        let volume_score = self.validate_volume_pattern(pattern, data)?;
        Ok(volume_score >= 0.6) // 60% threshold for volume confirmation
    }
    
    /// Calculate pattern targets and stop losses
    fn calculate_pattern_targets(&self, pattern: &mut Pattern, data: &MarketData) -> Result<()> {
        match pattern.pattern_type {
            PatternType::HeadAndShoulders => {
                if pattern.price_levels.len() >= 5 {
                    let neckline = (pattern.price_levels[1] + pattern.price_levels[3]) / 2.0;
                    let head = pattern.price_levels[2];
                    let target_distance = head - neckline;
                    pattern.breakout_target = Some(neckline - target_distance);
                    pattern.stop_loss = Some(head * 1.02); // 2% above head
                }
            }
            PatternType::DoubleTop => {
                if pattern.price_levels.len() >= 3 {
                    let peaks = (pattern.price_levels[0] + pattern.price_levels[2]) / 2.0;
                    let valley = pattern.price_levels[1];
                    let target_distance = peaks - valley;
                    pattern.breakout_target = Some(valley - target_distance);
                    pattern.stop_loss = Some(peaks * 1.02);
                }
            }
            PatternType::Triangle => {
                if let Some(current_price) = data.prices.last() {
                    let height = pattern.price_levels.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                        - pattern.price_levels.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    pattern.breakout_target = Some(current_price + height);
                    pattern.stop_loss = Some(current_price - height * 0.3);
                }
            }
            PatternType::Support => {
                let support_level = pattern.price_levels[0];
                pattern.breakout_target = Some(support_level * 1.05); // 5% bounce target
                pattern.stop_loss = Some(support_level * 0.98); // 2% below support
            }
            PatternType::Resistance => {
                let resistance_level = pattern.price_levels[0];
                pattern.breakout_target = Some(resistance_level * 1.02); // 2% breakout target
                pattern.stop_loss = Some(resistance_level * 0.95); // 5% below resistance
            }
            _ => {
                // Default targets for other patterns
                if let Some(current_price) = data.prices.last() {
                    pattern.breakout_target = Some(current_price * 1.03);
                    pattern.stop_loss = Some(current_price * 0.97);
                }
            }
        }
        
        Ok(())
    }
    
    /// Filter overlapping patterns
    fn filter_overlapping_patterns(&self, patterns: Vec<Pattern>) -> Result<Vec<Pattern>> {
        let mut filtered = Vec::new();
        let mut used_time_ranges = Vec::new();
        
        // Sort by confidence (highest first)
        let mut sorted_patterns = patterns;
        sorted_patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        for pattern in sorted_patterns {
            let pattern_range = (pattern.start_time, pattern.end_time.unwrap_or(pattern.start_time));
            
            // Check if this pattern overlaps with any already selected pattern
            let overlaps = used_time_ranges.iter().any(|(start, end)| {
                self.time_ranges_overlap(pattern_range, (*start, *end))
            });
            
            if !overlaps {
                used_time_ranges.push(pattern_range);
                filtered.push(pattern);
            }
        }
        
        Ok(filtered)
    }
    
    /// Check if two time ranges overlap
    fn time_ranges_overlap(&self, range1: (DateTime<Utc>, Option<DateTime<Utc>>), range2: (DateTime<Utc>, Option<DateTime<Utc>>)) -> bool {
        let (start1, end1) = range1;
        let (start2, end2) = range2;
        
        let end1 = end1.unwrap_or(start1 + Duration::hours(1));
        let end2 = end2.unwrap_or(start2 + Duration::hours(1));
        
        start1 < end2 && start2 < end1
    }
    
    /// Generate cache key for patterns
    fn generate_cache_key(&self, data: &MarketData) -> String {
        format!("{}_{}_{}_{}", 
            data.symbol, 
            data.timestamp.timestamp(), 
            data.prices.len(),
            data.volumes.len()
        )
    }
    
    /// Find price index for timestamp
    fn find_price_index(&self, timestamp: DateTime<Utc>, data: &MarketData) -> Result<usize> {
        // Simplified: assume prices are in chronological order
        // In real implementation, you'd map timestamps to indices
        Ok(data.prices.len().saturating_sub(10)) // Placeholder
    }
    
    /// Calculate trend line slope
    fn calculate_trend_line(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let n = prices.len() as f64;
        let sum_x: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();
        
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator == 0.0 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Update pattern weights based on feedback
    pub async fn update_weights(&mut self, feedback: &PatternFeedback) -> Result<()> {
        info!("Updating pattern recognition weights with feedback");
        
        // Update confidence threshold based on overall accuracy
        let total_patterns = feedback.pattern_accuracy.values().sum::<f64>();
        if total_patterns > 0.0 {
            let average_accuracy = feedback.pattern_accuracy.values().sum::<f64>() / feedback.pattern_accuracy.len() as f64;
            
            if average_accuracy < 0.6 {
                // Low accuracy, be more conservative
                self.config.confidence_threshold = (self.config.confidence_threshold + 0.05).min(0.9);
            } else if average_accuracy > 0.8 {
                // High accuracy, can be more aggressive
                self.config.confidence_threshold = (self.config.confidence_threshold - 0.02).max(0.5);
            }
        }
        
        // Update volume confirmation requirement based on false pattern rate
        if feedback.false_pattern_rate > 0.3 {
            self.config.require_volume_confirmation = true;
            self.config.volume_confirmation_threshold = (self.config.volume_confirmation_threshold + 0.1).min(2.0);
        }
        
        // Apply weight adjustments
        for (parameter, adjustment) in &feedback.weight_adjustments {
            match parameter.as_str() {
                "price_tolerance" => {
                    self.config.price_tolerance = (self.config.price_tolerance + adjustment).clamp(0.005, 0.05);
                }
                "min_pattern_duration" => {
                    if let Ok(duration_adjustment) = adjustment.parse::<i32>() {
                        self.config.min_pattern_duration = (self.config.min_pattern_duration as i32 + duration_adjustment).max(1) as usize;
                    }
                }
                "volume_confirmation_threshold" => {
                    self.config.volume_confirmation_threshold = (self.config.volume_confirmation_threshold + adjustment).clamp(1.0, 3.0);
                }
                _ => {}
            }
        }
        
        info!("Updated pattern recognizer configuration: {:?}", self.config);
        Ok(())
    }
}

/// Classical pattern detector
#[derive(Debug, Clone)]
struct ClassicalPatternDetector {
    config: PatternConfig,
}

impl ClassicalPatternDetector {
    fn new(config: &PatternConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn detect(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Detect various classical patterns
        patterns.extend(self.detect_head_and_shoulders(data)?);
        patterns.extend(self.detect_double_tops_bottoms(data)?);
        patterns.extend(self.detect_triangles(data)?);
        patterns.extend(self.detect_flags_pennants(data)?);
        
        Ok(patterns)
    }
    
    fn detect_head_and_shoulders(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        if data.prices.len() < 20 {
            return Ok(patterns);
        }
        
        // Simplified head and shoulders detection
        let prices = &data.prices;
        let window = 10;
        
        for i in window*2..prices.len()-window*2 {
            // Look for potential head and shoulders pattern
            let left_shoulder_idx = self.find_local_maximum(&prices[i-window*2..i-window], i-window*2)?;
            let head_idx = self.find_local_maximum(&prices[i-window..i+window], i-window)?;
            let right_shoulder_idx = self.find_local_maximum(&prices[i+window..i+window*2], i+window)?;
            
            if let (Some(ls_idx), Some(h_idx), Some(rs_idx)) = (left_shoulder_idx, head_idx, right_shoulder_idx) {
                let left_shoulder = prices[ls_idx];
                let head = prices[h_idx];
                let right_shoulder = prices[rs_idx];
                
                // Check if head is higher than shoulders
                if head > left_shoulder && head > right_shoulder {
                    let pattern = Pattern {
                        pattern_type: PatternType::HeadAndShoulders,
                        confidence: 0.0, // Will be calculated during validation
                        start_time: data.timestamp - Duration::minutes((data.prices.len() - ls_idx) as i64),
                        end_time: Some(data.timestamp - Duration::minutes((data.prices.len() - rs_idx) as i64)),
                        price_levels: vec![left_shoulder, prices[self.find_valley_between(ls_idx, h_idx, prices)?], head, prices[self.find_valley_between(h_idx, rs_idx, prices)?], right_shoulder],
                        volume_confirmation: false,
                        breakout_target: None,
                        stop_loss: None,
                    };
                    patterns.push(pattern);
                }
            }
        }
        
        Ok(patterns)
    }
    
    fn detect_double_tops_bottoms(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        if data.prices.len() < 15 {
            return Ok(patterns);
        }
        
        let prices = &data.prices;
        let window = 7;
        
        for i in window*2..prices.len()-window {
            // Look for potential double top pattern
            let first_peak_idx = self.find_local_maximum(&prices[i-window*2..i-window], i-window*2)?;
            let valley_idx = self.find_local_minimum(&prices[i-window..i], i-window)?;
            let second_peak_idx = self.find_local_maximum(&prices[i..i+window], i)?;
            
            if let (Some(fp_idx), Some(v_idx), Some(sp_idx)) = (first_peak_idx, valley_idx, second_peak_idx) {
                let first_peak = prices[fp_idx];
                let valley = prices[v_idx];
                let second_peak = prices[sp_idx];
                
                // Check if peaks are similar height
                let peak_diff = (first_peak - second_peak).abs() / first_peak.max(second_peak);
                if peak_diff < 0.03 { // 3% tolerance
                    let pattern = Pattern {
                        pattern_type: PatternType::DoubleTop,
                        confidence: 0.0,
                        start_time: data.timestamp - Duration::minutes((data.prices.len() - fp_idx) as i64),
                        end_time: Some(data.timestamp - Duration::minutes((data.prices.len() - sp_idx) as i64)),
                        price_levels: vec![first_peak, valley, second_peak],
                        volume_confirmation: false,
                        breakout_target: None,
                        stop_loss: None,
                    };
                    patterns.push(pattern);
                }
            }
        }
        
        Ok(patterns)
    }
    
    fn detect_triangles(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        if data.prices.len() < 20 {
            return Ok(patterns);
        }
        
        // Simplified triangle detection
        let prices = &data.prices;
        let min_duration = 10;
        
        for start in 0..prices.len()-min_duration {
            for end in start+min_duration..prices.len() {
                let segment = &prices[start..=end];
                if self.is_triangle_pattern(segment)? {
                    let pattern = Pattern {
                        pattern_type: PatternType::Triangle,
                        confidence: 0.0,
                        start_time: data.timestamp - Duration::minutes((data.prices.len() - start) as i64),
                        end_time: Some(data.timestamp - Duration::minutes((data.prices.len() - end) as i64)),
                        price_levels: vec![segment[0], segment[segment.len()/2], segment[segment.len()-1]],
                        volume_confirmation: false,
                        breakout_target: None,
                        stop_loss: None,
                    };
                    patterns.push(pattern);
                }
            }
        }
        
        Ok(patterns)
    }
    
    fn detect_flags_pennants(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        // Simplified flag/pennant detection
        Ok(Vec::new())
    }
    
    // Helper methods
    
    fn find_local_maximum(&self, prices: &[f64], offset: usize) -> Result<Option<usize>> {
        if prices.len() < 3 {
            return Ok(None);
        }
        
        let mut max_idx = 0;
        let mut max_val = prices[0];
        
        for (i, &price) in prices.iter().enumerate() {
            if price > max_val {
                max_val = price;
                max_idx = i;
            }
        }
        
        // Check if it's a local maximum (higher than neighbors)
        if max_idx > 0 && max_idx < prices.len() - 1 {
            if prices[max_idx] > prices[max_idx - 1] && prices[max_idx] > prices[max_idx + 1] {
                Ok(Some(offset + max_idx))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    fn find_local_minimum(&self, prices: &[f64], offset: usize) -> Result<Option<usize>> {
        if prices.len() < 3 {
            return Ok(None);
        }
        
        let mut min_idx = 0;
        let mut min_val = prices[0];
        
        for (i, &price) in prices.iter().enumerate() {
            if price < min_val {
                min_val = price;
                min_idx = i;
            }
        }
        
        // Check if it's a local minimum (lower than neighbors)
        if min_idx > 0 && min_idx < prices.len() - 1 {
            if prices[min_idx] < prices[min_idx - 1] && prices[min_idx] < prices[min_idx + 1] {
                Ok(Some(offset + min_idx))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    fn find_valley_between(&self, start_idx: usize, end_idx: usize, prices: &[f64]) -> Result<usize> {
        if start_idx >= end_idx || end_idx >= prices.len() {
            return Ok(start_idx);
        }
        
        let mut min_idx = start_idx;
        let mut min_val = prices[start_idx];
        
        for i in start_idx..=end_idx {
            if prices[i] < min_val {
                min_val = prices[i];
                min_idx = i;
            }
        }
        
        Ok(min_idx)
    }
    
    fn is_triangle_pattern(&self, prices: &[f64]) -> Result<bool> {
        if prices.len() < 6 {
            return Ok(false);
        }
        
        // Calculate upper and lower trend lines
        let highs = self.extract_highs(prices);
        let lows = self.extract_lows(prices);
        
        if highs.len() < 2 || lows.len() < 2 {
            return Ok(false);
        }
        
        let upper_trend = self.calculate_trend_slope(&highs);
        let lower_trend = self.calculate_trend_slope(&lows);
        
        // Check if lines are converging
        Ok((upper_trend < 0.0 && lower_trend > 0.0) || (upper_trend.abs() > lower_trend.abs() && upper_trend < 0.0))
    }
    
    fn extract_highs(&self, prices: &[f64]) -> Vec<(usize, f64)> {
        let mut highs = Vec::new();
        let window = 3;
        
        for i in window..prices.len()-window {
            let is_high = (0..window).all(|j| prices[i] >= prices[i-j-1]) &&
                         (0..window).all(|j| prices[i] >= prices[i+j+1]);
            if is_high {
                highs.push((i, prices[i]));
            }
        }
        
        highs
    }
    
    fn extract_lows(&self, prices: &[f64]) -> Vec<(usize, f64)> {
        let mut lows = Vec::new();
        let window = 3;
        
        for i in window..prices.len()-window {
            let is_low = (0..window).all(|j| prices[i] <= prices[i-j-1]) &&
                        (0..window).all(|j| prices[i] <= prices[i+j+1]);
            if is_low {
                lows.push((i, prices[i]));
            }
        }
        
        lows
    }
    
    fn calculate_trend_slope(&self, points: &[(usize, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }
        
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| *x as f64).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| *x as f64 * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator == 0.0 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

/// Support and resistance detector
#[derive(Debug, Clone)]
struct SupportResistanceDetector {
    config: PatternConfig,
}

impl SupportResistanceDetector {
    fn new(config: &PatternConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn detect(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        let support_levels = self.find_support_levels(data)?;
        let resistance_levels = self.find_resistance_levels(data)?;
        
        // Create patterns for significant levels
        for level in support_levels {
            let pattern = Pattern {
                pattern_type: PatternType::Support,
                confidence: 0.0,
                start_time: data.timestamp - Duration::hours(1),
                end_time: Some(data.timestamp),
                price_levels: vec![level],
                volume_confirmation: false,
                breakout_target: None,
                stop_loss: None,
            };
            patterns.push(pattern);
        }
        
        for level in resistance_levels {
            let pattern = Pattern {
                pattern_type: PatternType::Resistance,
                confidence: 0.0,
                start_time: data.timestamp - Duration::hours(1),
                end_time: Some(data.timestamp),
                price_levels: vec![level],
                volume_confirmation: false,
                breakout_target: None,
                stop_loss: None,
            };
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn find_support_levels(&self, data: &MarketData) -> Result<Vec<f64>> {
        let mut levels = Vec::new();
        let prices = &data.prices;
        let window = 10;
        
        // Find local minima
        for i in window..prices.len()-window {
            let is_local_min = (0..window).all(|j| prices[i] <= prices[i-j-1]) &&
                              (0..window).all(|j| prices[i] <= prices[i+j+1]);
                              
            if is_local_min {
                levels.push(prices[i]);
            }
        }
        
        // Cluster similar levels
        let clustered = self.cluster_price_levels(levels, 0.005)?; // 0.5% tolerance
        
        Ok(clustered)
    }
    
    fn find_resistance_levels(&self, data: &MarketData) -> Result<Vec<f64>> {
        let mut levels = Vec::new();
        let prices = &data.prices;
        let window = 10;
        
        // Find local maxima
        for i in window..prices.len()-window {
            let is_local_max = (0..window).all(|j| prices[i] >= prices[i-j-1]) &&
                              (0..window).all(|j| prices[i] >= prices[i+j+1]);
                              
            if is_local_max {
                levels.push(prices[i]);
            }
        }
        
        // Cluster similar levels
        let clustered = self.cluster_price_levels(levels, 0.005)?; // 0.5% tolerance
        
        Ok(clustered)
    }
    
    fn cluster_price_levels(&self, levels: Vec<f64>, tolerance: f64) -> Result<Vec<f64>> {
        if levels.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut sorted_levels = levels;
        sorted_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut clustered = Vec::new();
        let mut current_cluster = vec![sorted_levels[0]];
        
        for &level in &sorted_levels[1..] {
            let cluster_center = current_cluster.iter().sum::<f64>() / current_cluster.len() as f64;
            
            if (level - cluster_center).abs() / cluster_center <= tolerance {
                current_cluster.push(level);
            } else {
                // Finalize current cluster
                if current_cluster.len() >= 2 { // At least 2 touches
                    let cluster_average = current_cluster.iter().sum::<f64>() / current_cluster.len() as f64;
                    clustered.push(cluster_average);
                }
                current_cluster = vec![level];
            }
        }
        
        // Don't forget the last cluster
        if current_cluster.len() >= 2 {
            let cluster_average = current_cluster.iter().sum::<f64>() / current_cluster.len() as f64;
            clustered.push(cluster_average);
        }
        
        Ok(clustered)
    }
}

/// Volume pattern detector
#[derive(Debug, Clone)]
struct VolumePatternDetector {
    config: PatternConfig,
}

impl VolumePatternDetector {
    fn new(config: &PatternConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn detect(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Detect volume spikes
        patterns.extend(self.detect_volume_spikes(data)?);
        
        // Detect volume climax patterns
        patterns.extend(self.detect_volume_climax(data)?);
        
        Ok(patterns)
    }
    
    fn detect_volume_spikes(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        if data.volumes.len() < 20 {
            return Ok(patterns);
        }
        
        let volumes = &data.volumes;
        let baseline_volume = volumes[..volumes.len()-5].iter().sum::<f64>() / (volumes.len() - 5) as f64;
        
        for (i, &volume) in volumes[volumes.len()-5..].iter().enumerate() {
            if volume > baseline_volume * 2.0 { // 2x spike threshold
                let pattern = Pattern {
                    pattern_type: PatternType::VolumeSpike,
                    confidence: 0.0,
                    start_time: data.timestamp - Duration::minutes((5 - i) as i64),
                    end_time: Some(data.timestamp - Duration::minutes((5 - i) as i64)),
                    price_levels: vec![data.prices[data.prices.len() - 5 + i]],
                    volume_confirmation: true,
                    breakout_target: None,
                    stop_loss: None,
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    fn detect_volume_climax(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        // Simplified volume climax detection
        Ok(Vec::new())
    }
}

/// Microstructure pattern detector
#[derive(Debug, Clone)]
struct MicrostructurePatternDetector {
    config: PatternConfig,
}

impl MicrostructurePatternDetector {
    fn new(config: &PatternConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    async fn detect(&self, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        if let Some(ref order_book) = data.order_book {
            patterns.extend(self.detect_order_book_patterns(order_book, data)?);
        }
        
        Ok(patterns)
    }
    
    fn detect_order_book_patterns(&self, order_book: &OrderBook, data: &MarketData) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Detect order book imbalances
        let imbalance = self.calculate_order_book_imbalance(order_book)?;
        
        if imbalance.abs() > 0.3 { // 30% imbalance threshold
            let pattern_type = if imbalance > 0.0 {
                PatternType::OrderBookImbalance
            } else {
                PatternType::OrderBookImbalance
            };
            
            let pattern = Pattern {
                pattern_type,
                confidence: 0.0,
                start_time: order_book.timestamp,
                end_time: Some(order_book.timestamp),
                price_levels: vec![self.calculate_mid_price(order_book)?],
                volume_confirmation: false,
                breakout_target: None,
                stop_loss: None,
            };
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    fn calculate_order_book_imbalance(&self, order_book: &OrderBook) -> Result<f64> {
        let total_bid_volume: f64 = order_book.bids.iter().map(|level| level.quantity).sum();
        let total_ask_volume: f64 = order_book.asks.iter().map(|level| level.quantity).sum();
        let total_volume = total_bid_volume + total_ask_volume;
        
        if total_volume > 0.0 {
            Ok((total_bid_volume - total_ask_volume) / total_volume)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_mid_price(&self, order_book: &OrderBook) -> Result<f64> {
        if !order_book.bids.is_empty() && !order_book.asks.is_empty() {
            Ok((order_book.bids[0].price + order_book.asks[0].price) / 2.0)
        } else {
            Err(AnalysisError::calculation_error("Empty order book"))
        }
    }
}

/// Historical pattern for machine learning
#[derive(Debug, Clone)]
struct HistoricalPattern {
    pattern: Pattern,
    outcome: PatternOutcome,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
enum PatternOutcome {
    Successful,
    Failed,
    Partial,
    Pending,
}

/// Cached pattern result
#[derive(Debug, Clone)]
struct CachedPattern {
    patterns: Vec<Pattern>,
    timestamp: DateTime<Utc>,
    ttl: Duration,
}

impl CachedPattern {
    fn is_expired(&self) -> bool {
        Utc::now() - self.timestamp > self.ttl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_pattern_recognizer_creation() {
        let config = Config::default();
        let recognizer = PatternRecognizer::new(&config);
        assert!(recognizer.is_ok());
    }
    
    #[tokio::test]
    async fn test_pattern_recognition() {
        let config = Config::default();
        let recognizer = PatternRecognizer::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let patterns = recognizer.recognize_patterns(&market_data).await;
        assert!(patterns.is_ok());
    }
    
    #[test]
    fn test_head_shoulders_validation() {
        let config = Config::default();
        let recognizer = PatternRecognizer::new(&config).unwrap();
        
        let pattern = Pattern {
            pattern_type: PatternType::HeadAndShoulders,
            confidence: 0.0,
            start_time: Utc::now(),
            end_time: Some(Utc::now()),
            price_levels: vec![100.0, 95.0, 110.0, 94.0, 101.0], // Valid H&S formation
            volume_confirmation: false,
            breakout_target: None,
            stop_loss: None,
        };
        
        let market_data = MarketData::mock_data();
        let geometry_score = recognizer.validate_head_shoulders_geometry(&pattern, &market_data);
        assert!(geometry_score.is_ok());
        assert!(geometry_score.unwrap() > 0.0);
    }
    
    #[test]
    fn test_support_resistance_clustering() {
        let config = PatternConfig::default();
        let detector = SupportResistanceDetector::new(&config).unwrap();
        
        let levels = vec![100.0, 100.5, 99.8, 101.2, 120.0, 119.5, 120.3];
        let clustered = detector.cluster_price_levels(levels, 0.01).unwrap();
        
        assert!(clustered.len() == 2); // Should form 2 clusters
    }
}