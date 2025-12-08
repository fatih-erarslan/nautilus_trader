//! Main regime detection engine

use crate::{
    types::{MarketRegime, RegimeConfig, RegimeDetectionResult, RegimeFeatures},
    simd_ops::calculate_features_simd,
    confidence::ConfidenceScorer,
    cache::RegimeCache,
};
use std::sync::Arc;

/// Ultra-fast regime detector with sub-100ns latency
pub struct RegimeDetector {
    config: RegimeConfig,
    scorer: ConfidenceScorer,
    cache: Option<Arc<RegimeCache>>,
}

impl RegimeDetector {
    /// Create new regime detector with default configuration
    pub fn new() -> Self {
        let config = RegimeConfig::default();
        Self::with_config(config)
    }
    
    /// Create regime detector with custom configuration
    pub fn with_config(config: RegimeConfig) -> Self {
        let cache = if config.enable_cache {
            Some(Arc::new(RegimeCache::new(config.cache_size, 60))) // 60 second TTL
        } else {
            None
        };
        
        Self {
            config,
            scorer: ConfidenceScorer::new(),
            cache,
        }
    }
    
    /// Detect market regime with sub-100ns latency
    #[inline(always)]
    pub fn detect_regime(&self, prices: &[f32], volumes: &[f32]) -> RegimeDetectionResult {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(cached_result) = cache.get(prices, volumes, self.config.window_size) {
                return cached_result;
            }
        }
        
        // Calculate features using SIMD
        let features = self.calculate_features_fast(prices, volumes);
        
        // Score all regimes
        let scores = self.scorer.calculate_scores(&features);
        
        // Find best regime
        let (regime, confidence) = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(r, c)| (*r, *c))
            .unwrap_or((MarketRegime::Ranging, 0.0));
        
        // Get transition probabilities
        let transition_probs = self.scorer.get_transition_probabilities(regime, &features);
        
        let latency_ns = start_time.elapsed().as_nanos() as u64;
        
        let result = RegimeDetectionResult {
            regime,
            confidence,
            features,
            transition_probs,
            latency_ns,
        };
        
        // Cache result
        if let Some(ref cache) = self.cache {
            cache.put(prices, volumes, self.config.window_size, result.clone());
        }
        
        result
    }
    
    /// Fast feature calculation with optimizations
    #[inline(always)]
    fn calculate_features_fast(&self, prices: &[f32], volumes: &[f32]) -> RegimeFeatures {
        let window_size = self.config.window_size.min(prices.len());
        
        if prices.len() < 2 {
            return RegimeFeatures::default();
        }
        
        // Use sliding window for efficiency
        let start_idx = if prices.len() > window_size {
            prices.len() - window_size
        } else {
            0
        };
        
        let price_window = &prices[start_idx..];
        let volume_window = if volumes.len() > start_idx {
            &volumes[start_idx..]
        } else {
            &volumes[..]
        };
        
        calculate_features_simd(price_window, volume_window)
    }
    
    /// Detect regime for streaming data (single update)
    #[inline(always)]
    pub fn detect_regime_streaming(
        &self,
        price_buffer: &[f32],
        volume_buffer: &[f32],
        new_price: f32,
        new_volume: f32,
    ) -> RegimeDetectionResult {
        // Create temporary buffers with new data
        let mut prices = Vec::with_capacity(price_buffer.len() + 1);
        let mut volumes = Vec::with_capacity(volume_buffer.len() + 1);
        
        prices.extend_from_slice(price_buffer);
        prices.push(new_price);
        
        volumes.extend_from_slice(volume_buffer);
        volumes.push(new_volume);
        
        // Maintain window size
        let window_size = self.config.window_size;
        if prices.len() > window_size {
            let start = prices.len() - window_size;
            self.detect_regime(&prices[start..], &volumes[start..])
        } else {
            self.detect_regime(&prices, &volumes)
        }
    }
    
    /// Batch detection for multiple windows
    pub fn detect_regime_batch(&self, windows: &[(&[f32], &[f32])]) -> Vec<RegimeDetectionResult> {
        // Process sequentially for now (rayon parallel processing can be added later)
        windows.iter()
            .map(|(prices, volumes)| self.detect_regime(prices, volumes))
            .collect()
    }
    
    /// Get regime persistence (how long regime has been active)
    pub fn get_regime_persistence(
        &self,
        price_history: &[f32],
        volume_history: &[f32],
        current_regime: MarketRegime,
    ) -> usize {
        let window_size = self.config.window_size;
        let mut persistence = 0;
        
        // Work backwards through history
        for i in (window_size..=price_history.len()).rev() {
            let start = i - window_size;
            let prices = &price_history[start..i];
            let volumes = if volume_history.len() >= i {
                &volume_history[start..i]
            } else {
                &volume_history[start.min(volume_history.len())..]
            };
            
            let result = self.detect_regime(prices, volumes);
            
            if result.regime == current_regime && result.confidence > self.config.min_confidence {
                persistence += 1;
            } else {
                break;
            }
        }
        
        persistence
    }
    
    /// Validate detection latency
    pub fn benchmark_latency(&self, iterations: usize) -> (u64, u64, u64) {
        let prices: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.1).collect();
        let volumes: Vec<f32> = (0..100).map(|i| 1000.0 + i as f32 * 10.0).collect();
        
        let mut latencies = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.detect_regime(&prices, &volumes);
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        
        latencies.sort_unstable();
        
        let min = latencies[0];
        let max = latencies[latencies.len() - 1];
        let median = latencies[latencies.len() / 2];
        
        (min, median, max)
    }
    
    /// Clear cache if enabled
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.cleanup();
        }
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for RegimeDetector {}
unsafe impl Sync for RegimeDetector {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regime_detection_basic() {
        let detector = RegimeDetector::new();
        
        // Bull trend test
        let bull_prices: Vec<f32> = (0..50).map(|i| 100.0 + i as f32 * 0.5).collect();
        let volumes: Vec<f32> = vec![1000.0; 50];
        
        let result = detector.detect_regime(&bull_prices, &volumes);
        assert!(matches!(result.regime, MarketRegime::Bullish));
        assert!(result.confidence > 0.5);
        assert!(result.latency_ns < 100_000); // Should be under 100μs
    }
    
    #[test]
    fn test_regime_detection_bear() {
        let detector = RegimeDetector::new();
        
        // Bear trend test
        let bear_prices: Vec<f32> = (0..50).map(|i| 150.0 - i as f32 * 0.5).collect();
        let volumes: Vec<f32> = vec![1000.0; 50];
        
        let result = detector.detect_regime(&bear_prices, &volumes);
        assert!(matches!(result.regime, MarketRegime::Bearish));
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_regime_detection_ranging() {
        let detector = RegimeDetector::new();
        
        // Ranging market test
        let ranging_prices: Vec<f32> = (0..50)
            .map(|i| 100.0 + (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let volumes: Vec<f32> = vec![1000.0; 50];
        
        let result = detector.detect_regime(&ranging_prices, &volumes);
        assert!(matches!(result.regime, MarketRegime::Ranging));
    }
    
    #[test]
    fn test_performance_requirement() {
        let detector = RegimeDetector::new();
        let (min, median, max) = detector.benchmark_latency(1000);
        
        println!("Latency stats (ns): min={}, median={}, max={}", min, median, max);
        
        // Critical requirement: median must be under 100ns
        assert!(median < 100, "Median latency {}ns exceeds 100ns requirement", median);
        
        // Most detections should be under 100ns
        let under_100ns_count = (0..100).map(|_| {
            let prices: Vec<f32> = (0..50).map(|i| 100.0 + i as f32 * 0.1).collect();
            let volumes: Vec<f32> = vec![1000.0; 50];
            
            let start = std::time::Instant::now();
            let _ = detector.detect_regime(&prices, &volumes);
            start.elapsed().as_nanos() as u64
        }).filter(|&latency| latency < 100).count();
        
        assert!(under_100ns_count >= 80, "Only {}% of detections under 100ns", under_100ns_count);
    }
    
    #[test]
    fn test_streaming_detection() {
        let detector = RegimeDetector::new();
        
        let mut price_buffer: Vec<f32> = (0..49).map(|i| 100.0 + i as f32 * 0.1).collect();
        let mut volume_buffer: Vec<f32> = vec![1000.0; 49];
        
        // Add streaming data
        let result = detector.detect_regime_streaming(&price_buffer, &volume_buffer, 105.0, 1000.0);
        
        assert!(result.latency_ns < 1000); // Should be very fast
    }
    
    #[test]
    fn test_batch_detection() {
        let detector = RegimeDetector::new();
        
        let windows: Vec<(Vec<f32>, Vec<f32>)> = (0..5).map(|i| {
            let prices: Vec<f32> = (0..50).map(|j| 100.0 + (i + j) as f32 * 0.1).collect();
            let volumes: Vec<f32> = vec![1000.0; 50];
            (prices, volumes)
        }).collect();
        
        let window_refs: Vec<(&[f32], &[f32])> = windows.iter()
            .map(|(p, v)| (p.as_slice(), v.as_slice()))
            .collect();
        
        let results = detector.detect_regime_batch(&window_refs);
        assert_eq!(results.len(), 5);
        
        for result in results {
            assert!(result.latency_ns < 10_000); // Batch should be efficient
        }
    }
}