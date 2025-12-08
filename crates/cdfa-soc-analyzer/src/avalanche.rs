//! Avalanche detection and analysis for SOC systems

use crate::{Result, SocError};
use std::cmp::Ordering;

/// Avalanche detector for identifying and analyzing avalanche events
pub struct AvalancheDetector {
    min_avalanche_size: usize,
    significance_threshold: f32,
}

impl AvalancheDetector {
    /// Create a new avalanche detector
    pub fn new() -> Self {
        Self {
            min_avalanche_size: 2,
            significance_threshold: 0.01,
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(min_size: usize, threshold: f32) -> Self {
        Self {
            min_avalanche_size: min_size,
            significance_threshold: threshold,
        }
    }
    
    /// Detect avalanches in a return series
    pub fn detect_avalanches(&self, returns: &[f32], window_size: usize) -> Result<Vec<f32>> {
        let n = returns.len();
        if n < window_size {
            return Err(SocError::InsufficientData(
                format!("Need at least {} data points, got {}", window_size, n)
            ));
        }
        
        let mut avalanche_sizes = vec![1.0f32; n];
        
        // Process each window
        for i in window_size..n {
            let window = &returns[(i - window_size)..i];
            let sizes = self.extract_avalanche_sizes(window);
            
            // Calculate average avalanche size for this point
            if !sizes.is_empty() {
                avalanche_sizes[i] = sizes.iter().sum::<f32>() / sizes.len() as f32;
            }
        }
        
        Ok(avalanche_sizes)
    }
    
    /// Extract avalanche sizes from a window of returns
    fn extract_avalanche_sizes(&self, window: &[f32]) -> Vec<f32> {
        let mut sizes = Vec::new();
        let mut current_size = 1;
        let mut current_sign = 0i8;
        
        for &ret in window {
            if ret.abs() < self.significance_threshold {
                // Near-zero return, potentially end of avalanche
                if current_size >= self.min_avalanche_size {
                    sizes.push(current_size as f32);
                }
                current_size = 1;
                current_sign = 0;
            } else {
                let sign = if ret > 0.0 { 1 } else { -1 };
                
                if sign == current_sign || current_sign == 0 {
                    // Same direction or starting new avalanche
                    current_size += 1;
                    current_sign = sign;
                } else {
                    // Direction changed, avalanche ended
                    if current_size >= self.min_avalanche_size {
                        sizes.push(current_size as f32);
                    }
                    current_size = 1;
                    current_sign = sign;
                }
            }
        }
        
        // Don't forget the last avalanche
        if current_size >= self.min_avalanche_size {
            sizes.push(current_size as f32);
        }
        
        sizes
    }
    
    /// Calculate avalanche distribution statistics
    pub fn calculate_avalanche_statistics(&self, avalanche_sizes: &[f32]) -> AvalancheStatistics {
        if avalanche_sizes.is_empty() {
            return AvalancheStatistics::default();
        }
        
        // Sort sizes for percentile calculations
        let mut sorted_sizes = avalanche_sizes.to_vec();
        sorted_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        
        let n = sorted_sizes.len() as f32;
        let mean = sorted_sizes.iter().sum::<f32>() / n;
        
        // Calculate standard deviation
        let variance = sorted_sizes.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n;
        let std_dev = variance.sqrt();
        
        // Calculate percentiles
        let p25_idx = ((n - 1.0) * 0.25) as usize;
        let p50_idx = ((n - 1.0) * 0.50) as usize;
        let p75_idx = ((n - 1.0) * 0.75) as usize;
        let p95_idx = ((n - 1.0) * 0.95) as usize;
        
        AvalancheStatistics {
            mean,
            std_dev,
            min: sorted_sizes[0],
            max: sorted_sizes[sorted_sizes.len() - 1],
            p25: sorted_sizes[p25_idx],
            p50: sorted_sizes[p50_idx],
            p75: sorted_sizes[p75_idx],
            p95: sorted_sizes[p95_idx],
            count: sorted_sizes.len(),
        }
    }
    
    /// Detect cascading avalanches (avalanches triggering more avalanches)
    pub fn detect_cascades(&self, returns: &[f32], threshold: f32) -> Vec<CascadeEvent> {
        let mut cascades = Vec::new();
        let mut in_cascade = false;
        let mut cascade_start = 0;
        let mut cascade_magnitude = 0.0f32;
        
        for (i, &ret) in returns.iter().enumerate() {
            if ret.abs() > threshold {
                if !in_cascade {
                    // Start of cascade
                    in_cascade = true;
                    cascade_start = i;
                    cascade_magnitude = ret.abs();
                } else {
                    // Continuation of cascade
                    cascade_magnitude += ret.abs();
                }
            } else if in_cascade {
                // End of cascade
                cascades.push(CascadeEvent {
                    start_idx: cascade_start,
                    end_idx: i,
                    duration: i - cascade_start,
                    total_magnitude: cascade_magnitude,
                    peak_magnitude: returns[cascade_start..i]
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f32, |a, b| a.max(b)),
                });
                in_cascade = false;
            }
        }
        
        // Handle cascade that extends to the end
        if in_cascade {
            cascades.push(CascadeEvent {
                start_idx: cascade_start,
                end_idx: returns.len(),
                duration: returns.len() - cascade_start,
                total_magnitude: cascade_magnitude,
                peak_magnitude: returns[cascade_start..]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, |a, b| a.max(b)),
            });
        }
        
        cascades
    }
    
    /// Calculate branching ratio (indicator of criticality)
    pub fn calculate_branching_ratio(&self, avalanche_sizes: &[f32], window: usize) -> Vec<f32> {
        let n = avalanche_sizes.len();
        if n < window * 2 {
            return vec![1.0; n];
        }
        
        let mut branching_ratios = vec![1.0f32; n];
        
        // Calculate branching ratio for each point
        for i in window..n - window {
            let ancestors = &avalanche_sizes[(i - window)..i];
            let descendants = &avalanche_sizes[i..(i + window)];
            
            let ancestor_sum: f32 = ancestors.iter().sum();
            let descendant_sum: f32 = descendants.iter().sum();
            
            if ancestor_sum > 0.0 {
                branching_ratios[i] = descendant_sum / ancestor_sum;
            }
        }
        
        branching_ratios
    }
}

impl Default for AvalancheDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about avalanche distributions
#[derive(Debug, Clone, Default)]
pub struct AvalancheStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
    pub count: usize,
}

/// Represents a cascade event (large avalanche)
#[derive(Debug, Clone)]
pub struct CascadeEvent {
    pub start_idx: usize,
    pub end_idx: usize,
    pub duration: usize,
    pub total_magnitude: f32,
    pub peak_magnitude: f32,
}

/// Calculate avalanche waiting times (time between avalanches)
pub fn calculate_waiting_times(avalanche_events: &[usize]) -> Vec<usize> {
    if avalanche_events.len() < 2 {
        return Vec::new();
    }
    
    let mut waiting_times = Vec::with_capacity(avalanche_events.len() - 1);
    
    for i in 1..avalanche_events.len() {
        waiting_times.push(avalanche_events[i] - avalanche_events[i - 1]);
    }
    
    waiting_times
}

/// Detect avalanche clustering (periods of increased avalanche activity)
pub fn detect_avalanche_clusters(
    avalanche_sizes: &[f32],
    threshold: f32,
    min_cluster_size: usize
) -> Vec<(usize, usize)> {
    let mut clusters = Vec::new();
    let mut in_cluster = false;
    let mut cluster_start = 0;
    let mut cluster_size = 0;
    
    for (i, &size) in avalanche_sizes.iter().enumerate() {
        if size > threshold {
            if !in_cluster {
                in_cluster = true;
                cluster_start = i;
                cluster_size = 1;
            } else {
                cluster_size += 1;
            }
        } else if in_cluster {
            if cluster_size >= min_cluster_size {
                clusters.push((cluster_start, i));
            }
            in_cluster = false;
        }
    }
    
    // Handle cluster that extends to the end
    if in_cluster && cluster_size >= min_cluster_size {
        clusters.push((cluster_start, avalanche_sizes.len()));
    }
    
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_avalanche_detection() {
        let detector = AvalancheDetector::new();
        let returns = vec![
            0.01, 0.02, -0.03, -0.02, 0.001, 0.0005, 0.04, 0.03, -0.001, -0.05
        ];
        
        let sizes = detector.extract_avalanche_sizes(&returns);
        assert!(!sizes.is_empty());
        
        // Check that all sizes are positive
        assert!(sizes.iter().all(|&s| s > 0.0));
    }
    
    #[test]
    fn test_avalanche_statistics() {
        let detector = AvalancheDetector::new();
        let sizes = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let stats = detector.calculate_avalanche_statistics(&sizes);
        
        assert_eq!(stats.mean, 5.5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.p50, 5.0);
        assert_eq!(stats.count, 10);
    }
    
    #[test]
    fn test_cascade_detection() {
        let detector = AvalancheDetector::new();
        let returns = vec![
            0.001, 0.002, 0.05, 0.06, 0.07, 0.001, 0.002, -0.08, -0.09, 0.001
        ];
        
        let cascades = detector.detect_cascades(&returns, 0.04);
        
        assert_eq!(cascades.len(), 2);
        assert_eq!(cascades[0].start_idx, 2);
        assert_eq!(cascades[0].end_idx, 5);
        assert_eq!(cascades[1].start_idx, 7);
        assert_eq!(cascades[1].end_idx, 9);
    }
    
    #[test]
    fn test_branching_ratio() {
        let detector = AvalancheDetector::new();
        let sizes = vec![1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        
        let ratios = detector.calculate_branching_ratio(&sizes, 2);
        
        assert_eq!(ratios.len(), sizes.len());
        // At criticality, branching ratio should be close to 1
        assert!(ratios[4..6].iter().all(|&r| (r - 1.0).abs() < 2.0));
    }
}