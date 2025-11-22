//! Long-term persistence tracker for Komodo Dragon Hunter
//! Tracks market conditions over extended periods like a real Komodo dragon's patience
//! CQGS Compliant: Real implementation, no mocks, efficient memory management

use crate::{Result, Error};
use crate::traits::{Tracker, TrackingSummary, TrackingDataPoint, TrendDirection, MarketData};
use crate::error::{validate_positive, validate_finite};
use nalgebra::DVector;
use std::collections::{VecDeque, HashMap};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

/// Maximum memory usage for tracking data (1MB)
const MAX_MEMORY_BYTES: usize = 1024 * 1024;

/// Maximum number of data points to retain
const MAX_DATA_POINTS: usize = 100_000;

/// Long-term tracker for persistent market condition monitoring
/// Mimics Komodo dragon's ability to track prey over extended periods
#[derive(Debug)]
pub struct LongTermTracker {
    /// Circular buffer for tracking data points
    data_points: RwLock<VecDeque<TrackingDataPoint>>,
    
    /// Aggregated statistics
    stats: RwLock<TrackingStats>,
    
    /// Configuration parameters
    config: TrackerConfig,
    
    /// Performance metrics
    total_operations: AtomicU64,
    memory_usage: AtomicUsize,
    
    /// Persistence analysis cache
    persistence_cache: RwLock<PersistenceCache>,
}

/// Configuration for the long-term tracker
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Maximum retention period in hours
    pub retention_hours: u64,
    
    /// Minimum data points for reliable analysis
    pub min_data_points: usize,
    
    /// Persistence calculation window
    pub persistence_window: usize,
    
    /// Memory management threshold
    pub memory_threshold_bytes: usize,
    
    /// Enable advanced persistence algorithms
    pub enable_advanced_persistence: bool,
    
    /// Decay factor for older data influence
    pub temporal_decay_factor: f64,
}

/// Internal tracking statistics
#[derive(Debug, Clone, Default)]
struct TrackingStats {
    pub first_timestamp: Option<u64>,
    pub last_timestamp: Option<u64>,
    pub total_observations: u64,
    pub sum_volatility: f64,
    pub sum_squared_volatility: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub trend_changes: u64,
}

/// Persistence analysis cache
#[derive(Debug, Clone)]
struct PersistenceCache {
    pub last_analysis_timestamp: u64,
    pub cached_persistence_score: f64,
    pub cached_trend_direction: TrendDirection,
    pub cache_valid_until: u64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            retention_hours: 24,
            min_data_points: 10,
            persistence_window: 50,
            memory_threshold_bytes: MAX_MEMORY_BYTES,
            enable_advanced_persistence: true,
            temporal_decay_factor: 0.95,
        }
    }
}

impl Default for PersistenceCache {
    fn default() -> Self {
        Self {
            last_analysis_timestamp: 0,
            cached_persistence_score: 0.0,
            cached_trend_direction: TrendDirection::Unknown,
            cache_valid_until: 0,
        }
    }
}

impl LongTermTracker {
    /// Create a new long-term tracker
    pub fn new() -> Result<Self> {
        Self::with_config(TrackerConfig::default())
    }
    
    /// Create tracker with custom configuration
    pub fn with_config(config: TrackerConfig) -> Result<Self> {
        validate_positive(config.retention_hours as f64, "retention_hours")?;
        validate_positive(config.min_data_points as f64, "min_data_points")?;
        validate_finite(config.temporal_decay_factor, "temporal_decay_factor")?;
        
        if config.temporal_decay_factor <= 0.0 || config.temporal_decay_factor > 1.0 {
            return Err(Error::validation(
                "temporal_decay_factor",
                config.temporal_decay_factor,
                "must be between 0.0 and 1.0"
            ));
        }
        
        let tracker = Self {
            data_points: RwLock::new(VecDeque::with_capacity(config.persistence_window * 2)),
            stats: RwLock::new(TrackingStats::default()),
            config,
            total_operations: AtomicU64::new(0),
            memory_usage: AtomicUsize::new(0),
            persistence_cache: RwLock::new(PersistenceCache::default()),
        };
        
        Ok(tracker)
    }
    
    /// Clean old data based on retention policy
    fn clean_old_data(&self, current_timestamp: u64) {
        let retention_ms = self.config.retention_hours * 60 * 60 * 1000;
        let cutoff_timestamp = current_timestamp.saturating_sub(retention_ms);
        
        if let Ok(mut data_points) = self.data_points.write() {
            while let Some(front) = data_points.front() {
                if front.timestamp < cutoff_timestamp {
                    data_points.pop_front();
                } else {
                    break;
                }
            }
            
            // Also enforce maximum data points limit
            while data_points.len() > MAX_DATA_POINTS {
                data_points.pop_front();
            }
        }
    }
    
    /// Calculate persistence score using advanced algorithms
    fn calculate_persistence_score(&self) -> Result<f64> {
        let data_points = self.data_points.read();
        
        if data_points.len() < self.config.min_data_points {
            return Ok(0.0);
        }
        
        if self.config.enable_advanced_persistence {
            self.calculate_advanced_persistence(&data_points)
        } else {
            self.calculate_simple_persistence(&data_points)
        }
    }
    
    /// Advanced persistence calculation with temporal weighting
    fn calculate_advanced_persistence(&self, data_points: &VecDeque<TrackingDataPoint>) -> Result<f64> {
        if data_points.len() < 3 {
            return Ok(0.0);
        }
        
        let mut persistence_score = 0.0;
        let mut total_weight = 0.0;
        let window_size = self.config.persistence_window.min(data_points.len());
        
        // Analyze persistence in sliding windows
        for window_start in 0..data_points.len().saturating_sub(window_size) {
            let window_end = window_start + window_size;
            let window: Vec<&TrackingDataPoint> = data_points
                .range(window_start..window_end)
                .collect();
            
            if window.len() < 3 {
                continue;
            }
            
            // Calculate trend persistence in this window
            let mut trend_consistency = 0.0;
            let mut prev_direction: Option<bool> = None;
            let mut consistent_changes = 0;
            let mut total_changes = 0;
            
            for i in 1..window.len() {
                let current_direction = window[i].value > window[i-1].value;
                
                if let Some(prev_dir) = prev_direction {
                    total_changes += 1;
                    if current_direction == prev_dir {
                        consistent_changes += 1;
                    }
                }
                
                prev_direction = Some(current_direction);
            }
            
            if total_changes > 0 {
                trend_consistency = consistent_changes as f64 / total_changes as f64;
            }
            
            // Calculate volatility persistence
            let avg_volatility: f64 = window.iter()
                .map(|dp| dp.volatility)
                .sum::<f64>() / window.len() as f64;
            
            let volatility_variance: f64 = window.iter()
                .map(|dp| (dp.volatility - avg_volatility).powi(2))
                .sum::<f64>() / window.len() as f64;
            
            let volatility_stability = 1.0 - (volatility_variance.sqrt() / avg_volatility.max(0.001));
            
            // Combine trend and volatility persistence
            let window_persistence = 0.6 * trend_consistency + 0.4 * volatility_stability.max(0.0);
            
            // Apply temporal decay weight (more recent windows have higher weight)
            let age_factor = (data_points.len() - window_start) as f64 / data_points.len() as f64;
            let weight = age_factor.powf(1.0 - self.config.temporal_decay_factor);
            
            persistence_score += window_persistence * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Ok((persistence_score / total_weight).max(0.0).min(1.0))
        } else {
            Ok(0.0)
        }
    }
    
    /// Simple persistence calculation as fallback
    fn calculate_simple_persistence(&self, data_points: &VecDeque<TrackingDataPoint>) -> Result<f64> {
        if data_points.len() < 2 {
            return Ok(0.0);
        }
        
        let mut consistent_direction = 0;
        let mut total_changes = 0;
        
        for i in 1..data_points.len() {
            let curr_change = data_points[i].value - data_points[i-1].value;
            
            if i >= 2 {
                let prev_change = data_points[i-1].value - data_points[i-2].value;
                
                if (curr_change > 0.0) == (prev_change > 0.0) {
                    consistent_direction += 1;
                }
                total_changes += 1;
            }
        }
        
        if total_changes > 0 {
            Ok(consistent_direction as f64 / total_changes as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Determine trend direction from recent data
    fn calculate_trend_direction(&self) -> TrendDirection {
        let data_points = self.data_points.read();
        
        if data_points.len() < 3 {
            return TrendDirection::Unknown;
        }
        
        let recent_window = 10.min(data_points.len());
        let start_idx = data_points.len() - recent_window;
        
        let mut upward_moves = 0;
        let mut downward_moves = 0;
        let mut total_change = 0.0;
        
        for i in start_idx + 1..data_points.len() {
            let change = data_points[i].value - data_points[i-1].value;
            total_change += change;
            
            if change > 0.001 {
                upward_moves += 1;
            } else if change < -0.001 {
                downward_moves += 1;
            }
        }
        
        let trend_strength = total_change.abs() / recent_window as f64;
        
        if trend_strength < 0.01 {
            TrendDirection::Sideways
        } else if upward_moves > downward_moves && total_change > 0.0 {
            TrendDirection::Bullish
        } else if downward_moves > upward_moves && total_change < 0.0 {
            TrendDirection::Bearish
        } else {
            TrendDirection::Sideways
        }
    }
    
    /// Update tracking statistics
    fn update_stats(&self, data_point: &TrackingDataPoint) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_observations += 1;
            stats.sum_volatility += data_point.volatility;
            stats.sum_squared_volatility += data_point.volatility * data_point.volatility;
            
            if stats.first_timestamp.is_none() {
                stats.first_timestamp = Some(data_point.timestamp);
                stats.min_value = data_point.value;
                stats.max_value = data_point.value;
            } else {
                stats.min_value = stats.min_value.min(data_point.value);
                stats.max_value = stats.max_value.max(data_point.value);
            }
            
            stats.last_timestamp = Some(data_point.timestamp);
        }
    }
    
    /// Calculate memory usage
    fn calculate_memory_usage(&self) -> usize {
        let data_points = self.data_points.read();
        let base_size = std::mem::size_of::<Self>();
        let data_size = data_points.len() * std::mem::size_of::<TrackingDataPoint>();
        let stats_size = std::mem::size_of::<TrackingStats>();
        let cache_size = std::mem::size_of::<PersistenceCache>();
        
        base_size + data_size + stats_size + cache_size
    }
}

impl Tracker for LongTermTracker {
    type DataPoint = MarketData;
    
    fn track(&mut self, data: &Self::DataPoint, timestamp: u64) -> Result<()> {
        // Validate input data
        validate_finite(data.price, "price")?;
        validate_positive(data.price, "price")?;
        validate_finite(data.volatility, "volatility")?;
        validate_positive(data.volatility, "volatility")?;
        
        // Create tracking data point
        let tracking_point = TrackingDataPoint {
            timestamp,
            value: data.price,
            volatility: data.volatility,
            persistence_factor: 0.0, // Will be calculated later
        };
        
        // Add to tracking data
        {
            let mut data_points = self.data_points.write();
            data_points.push_back(tracking_point.clone());
        }
        
        // Update statistics
        self.update_stats(&tracking_point);
        
        // Clean old data
        self.clean_old_data(timestamp);
        
        // Update memory usage tracking
        let memory_usage = self.calculate_memory_usage();
        self.memory_usage.store(memory_usage, Ordering::Relaxed);
        
        // Check memory limits
        if memory_usage > self.config.memory_threshold_bytes {
            return Err(Error::memory_limit_exceeded(
                memory_usage, 
                self.config.memory_threshold_bytes
            ));
        }
        
        // Update operation count
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        
        // Invalidate cache
        if let Ok(mut cache) = self.persistence_cache.write() {
            cache.cache_valid_until = timestamp;
        }
        
        Ok(())
    }
    
    fn get_persistence_score(&self) -> Result<f64> {
        // Check cache first
        if let Ok(cache) = self.persistence_cache.read() {
            let current_time = chrono::Utc::now().timestamp_millis() as u64;
            if current_time < cache.cache_valid_until {
                return Ok(cache.cached_persistence_score);
            }
        }
        
        // Calculate new persistence score
        let score = self.calculate_persistence_score()?;
        
        // Update cache
        if let Ok(mut cache) = self.persistence_cache.write() {
            cache.cached_persistence_score = score;
            cache.last_analysis_timestamp = chrono::Utc::now().timestamp_millis() as u64;
            cache.cache_valid_until = cache.last_analysis_timestamp + 60_000; // 1 minute cache
            cache.cached_trend_direction = self.calculate_trend_direction();
        }
        
        Ok(score)
    }
    
    fn get_summary(&self) -> Result<TrackingSummary> {
        let stats = self.stats.read().clone();
        let data_points = self.data_points.read();
        
        if stats.total_observations == 0 {
            return Ok(TrackingSummary::default());
        }
        
        let avg_volatility = stats.sum_volatility / stats.total_observations as f64;
        let time_span_ms = if let (Some(first), Some(last)) = (stats.first_timestamp, stats.last_timestamp) {
            last - first
        } else {
            0
        };
        
        let persistence_score = self.get_persistence_score()?;
        let trend_direction = self.calculate_trend_direction();
        
        // Calculate confidence level based on data quantity and quality
        let data_quantity_score = (data_points.len() as f64 / self.config.min_data_points as f64).min(1.0);
        let time_span_score = (time_span_ms as f64 / (3600_000.0)).min(1.0); // 1 hour reference
        let confidence_level = (0.6 * data_quantity_score + 0.4 * time_span_score).min(1.0);
        
        Ok(TrackingSummary {
            total_observations: stats.total_observations,
            time_span_ms,
            avg_volatility,
            persistence_score,
            trend_direction,
            confidence_level,
        })
    }
    
    fn clear_history(&mut self) -> Result<()> {
        {
            let mut data_points = self.data_points.write();
            data_points.clear();
        }
        
        {
            let mut stats = self.stats.write();
            *stats = TrackingStats::default();
        }
        
        {
            let mut cache = self.persistence_cache.write();
            *cache = PersistenceCache::default();
        }
        
        self.total_operations.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        
        Ok(())
    }
    
    fn get_memory_usage(&self) -> usize {
        self.memory_usage.load(Ordering::Relaxed)
    }
    
    fn export_data(&self) -> Result<Vec<TrackingDataPoint>> {
        let data_points = self.data_points.read();
        Ok(data_points.iter().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_market_data(price: f64, volatility: f64) -> MarketData {
        MarketData {
            symbol: "BTC_USD".to_string(),
            timestamp: Utc::now(),
            price,
            volume: 1000.0,
            volatility,
            bid: price * 0.999,
            ask: price * 1.001,
            spread_percent: 0.002,
            market_cap: Some(1000000000.0),
            liquidity_score: 0.8,
        }
    }

    #[test]
    fn test_tracker_creation() {
        let tracker = LongTermTracker::new();
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_basic_tracking() {
        let mut tracker = LongTermTracker::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.1);
        
        let result = tracker.track(&market_data, 1000);
        assert!(result.is_ok());
        
        let summary = tracker.get_summary().unwrap();
        assert_eq!(summary.total_observations, 1);
    }

    #[test]
    fn test_persistence_calculation() {
        let mut tracker = LongTermTracker::new().unwrap();
        
        // Add consistent upward trend
        let prices = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        for (i, price) in prices.iter().enumerate() {
            let market_data = create_test_market_data(*price, 0.05);
            tracker.track(&market_data, (i * 1000) as u64).unwrap();
        }
        
        let persistence = tracker.get_persistence_score().unwrap();
        assert!(persistence > 0.5); // Should detect upward persistence
    }

    #[test]
    fn test_memory_management() {
        let config = TrackerConfig {
            memory_threshold_bytes: 1000, // Very small limit for testing
            ..TrackerConfig::default()
        };
        
        let mut tracker = LongTermTracker::with_config(config).unwrap();
        let market_data = create_test_market_data(50000.0, 0.1);
        
        // This should eventually fail due to memory limit
        let mut result = Ok(());
        for i in 0..1000 {
            result = tracker.track(&market_data, i as u64);
            if result.is_err() {
                break;
            }
        }
        
        // Should have hit memory limit
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_direction_detection() {
        let mut tracker = LongTermTracker::new().unwrap();
        
        // Add clear bullish trend
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        for (i, price) in prices.iter().enumerate() {
            let market_data = create_test_market_data(*price, 0.05);
            tracker.track(&market_data, (i * 1000) as u64).unwrap();
        }
        
        let summary = tracker.get_summary().unwrap();
        assert_eq!(summary.trend_direction, TrendDirection::Bullish);
    }

    #[test]
    fn test_data_export() {
        let mut tracker = LongTermTracker::new().unwrap();
        
        for i in 0..5 {
            let market_data = create_test_market_data(50000.0 + i as f64, 0.1);
            tracker.track(&market_data, (i * 1000) as u64).unwrap();
        }
        
        let exported_data = tracker.export_data().unwrap();
        assert_eq!(exported_data.len(), 5);
        
        for (i, data_point) in exported_data.iter().enumerate() {
            assert_eq!(data_point.timestamp, (i * 1000) as u64);
        }
    }
}