//! Fibonacci pattern detector implementation

use crate::{Result, PatternError, PatternResult, PatternParameters, DetectedPattern, PatternType};
use ndarray::{Array1, ArrayView1};
use std::time::Instant;

/// High-performance Fibonacci pattern detector
pub struct FibonacciPatternDetector {
    params: PatternParameters,
}

impl FibonacciPatternDetector {
    /// Create new pattern detector with default parameters
    pub fn new() -> Self {
        Self {
            params: PatternParameters::default(),
        }
    }
    
    /// Create pattern detector with custom parameters
    pub fn with_params(params: PatternParameters) -> Self {
        Self { params }
    }
    
    /// Get current parameters
    pub fn params(&self) -> &PatternParameters {
        &self.params
    }
    
    /// Detect patterns in market data
    pub fn detect_patterns(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        if high.len() != low.len() || high.len() != close.len() {
            return Err(PatternError::InvalidParameters {
                message: "High, low, and close arrays must have the same length".to_string(),
            });
        }
        
        if high.len() < self.params.min_pattern_bars {
            return Err(PatternError::InsufficientData {
                required: self.params.min_pattern_bars,
                actual: high.len(),
            });
        }
        
        let mut result = PatternResult::new();
        result.scan_period = high.len();
        
        // Detect swing points
        let swing_points = self.detect_swing_points(high.view(), low.view())?;
        result.swing_points_detected = swing_points.len();
        
        // Scan for patterns
        let scan_start = Instant::now();
        result.detected_patterns = self.scan_for_patterns(&swing_points, high.view(), low.view(), close.view())?;
        result.scan_time_ns = scan_start.elapsed().as_nanos() as u64;
        
        result.patterns_found = result.detected_patterns.len();
        result.computation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        Ok(result)
    }
    
    /// Detect swing points in price data
    fn detect_swing_points(
        &self,
        high: ArrayView1<f64>,
        low: ArrayView1<f64>,
    ) -> Result<Vec<crate::types::SwingPoint>> {
        let mut swing_points = Vec::new();
        let period = self.params.swing_detection_period;
        let n = high.len();
        
        for i in period..n-period {
            // Check for swing high
            let mut is_swing_high = true;
            for j in i-period..=i+period {
                if j != i && high[j] >= high[i] {
                    is_swing_high = false;
                    break;
                }
            }
            
            if is_swing_high {
                swing_points.push(crate::types::SwingPoint {
                    index: i,
                    price: high[i],
                    is_high: true,
                    strength: self.calculate_swing_strength(high, i, true),
                    confirmed: true,
                });
            }
            
            // Check for swing low
            let mut is_swing_low = true;
            for j in i-period..=i+period {
                if j != i && low[j] <= low[i] {
                    is_swing_low = false;
                    break;
                }
            }
            
            if is_swing_low {
                swing_points.push(crate::types::SwingPoint {
                    index: i,
                    price: low[i],
                    is_high: false,
                    strength: self.calculate_swing_strength(low, i, false),
                    confirmed: true,
                });
            }
        }
        
        // Sort by index
        swing_points.sort_by_key(|sp| sp.index);
        
        Ok(swing_points)
    }
    
    /// Calculate swing point strength
    fn calculate_swing_strength(&self, prices: ArrayView1<f64>, index: usize, is_high: bool) -> f64 {
        let period = self.params.swing_detection_period;
        let start = index.saturating_sub(period);
        let end = (index + period + 1).min(prices.len());
        
        let mut strength = 0.0;
        for i in start..end {
            if i != index {
                if is_high {
                    strength += (prices[index] - prices[i]).max(0.0);
                } else {
                    strength += (prices[i] - prices[index]).max(0.0);
                }
            }
        }
        
        strength / (end - start - 1) as f64
    }
    
    /// Scan for harmonic patterns
    fn scan_for_patterns(
        &self,
        swing_points: &[crate::types::SwingPoint],
        high: ArrayView1<f64>,
        low: ArrayView1<f64>,
        close: ArrayView1<f64>,
    ) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        if swing_points.len() < 5 {
            return Ok(patterns);
        }
        
        // Look for XABCD patterns
        for i in 0..swing_points.len().saturating_sub(4) {
            if patterns.len() >= self.params.max_patterns_per_scan {
                break;
            }
            
            let x = &swing_points[i];
            let a = &swing_points[i + 1];
            let b = &swing_points[i + 2];
            let c = &swing_points[i + 3];
            let d = &swing_points[i + 4];
            
            // Check if points form a valid XABCD pattern
            if x.is_high == a.is_high || a.is_high == b.is_high || 
               b.is_high == c.is_high || c.is_high == d.is_high {
                continue;
            }
            
            // Try to identify pattern type
            if let Some(pattern) = self.identify_pattern_type(x, a, b, c, d) {
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Identify the type of harmonic pattern
    fn identify_pattern_type(
        &self,
        x: &crate::types::SwingPoint,
        a: &crate::types::SwingPoint,
        b: &crate::types::SwingPoint,
        c: &crate::types::SwingPoint,
        d: &crate::types::SwingPoint,
    ) -> Option<DetectedPattern> {
        // Calculate ratios
        let ab_xa = (b.price - a.price).abs() / (a.price - x.price).abs();
        let bc_ab = (c.price - b.price).abs() / (b.price - a.price).abs();
        let cd_bc = (d.price - c.price).abs() / (c.price - b.price).abs();
        let ad_xa = (d.price - a.price).abs() / (a.price - x.price).abs();
        
        // Try different pattern types
        let pattern_configs = [
            crate::patterns::gartley_config(),
            crate::patterns::butterfly_config(),
            crate::patterns::bat_config(),
            crate::patterns::crab_config(),
            crate::patterns::shark_config(),
        ];
        
        for config in &pattern_configs {
            if self.validate_pattern_ratios(&config.ratios, ab_xa, bc_ab, cd_bc, ad_xa) {
                let points = vec![
                    crate::types::PatternPoint {
                        index: x.index,
                        price: x.price,
                        role: "X".to_string(),
                        timestamp: None,
                        confidence: 1.0,
                    },
                    crate::types::PatternPoint {
                        index: a.index,
                        price: a.price,
                        role: "A".to_string(),
                        timestamp: None,
                        confidence: 1.0,
                    },
                    crate::types::PatternPoint {
                        index: b.index,
                        price: b.price,
                        role: "B".to_string(),
                        timestamp: None,
                        confidence: 1.0,
                    },
                    crate::types::PatternPoint {
                        index: c.index,
                        price: c.price,
                        role: "C".to_string(),
                        timestamp: None,
                        confidence: 1.0,
                    },
                    crate::types::PatternPoint {
                        index: d.index,
                        price: d.price,
                        role: "D".to_string(),
                        timestamp: None,
                        confidence: 1.0,
                    },
                ];
                
                return Some(DetectedPattern {
                    pattern_type: config.pattern_type,
                    points,
                    confidence: 0.8, // Placeholder
                    completion_time: None,
                    is_bullish: x.price > d.price,
                    validation_score: 0.8,
                    ab_xa_ratio: ab_xa,
                    bc_ab_ratio: bc_ab,
                    cd_bc_ratio: cd_bc,
                    ad_xa_ratio: ad_xa,
                    pattern_height: (x.price - d.price).abs(),
                    pattern_duration: d.index - x.index,
                    volume_confirmation: None,
                });
            }
        }
        
        None
    }
    
    /// Validate pattern ratios against configuration
    fn validate_pattern_ratios(
        &self,
        ratios: &crate::types::HarmonicRatios,
        ab_xa: f64,
        bc_ab: f64,
        cd_bc: f64,
        ad_xa: f64,
    ) -> bool {
        let tolerance = self.params.ratio_tolerance;
        
        ab_xa >= ratios.ab_xa_min - tolerance && ab_xa <= ratios.ab_xa_max + tolerance &&
        bc_ab >= ratios.bc_ab_min - tolerance && bc_ab <= ratios.bc_ab_max + tolerance &&
        cd_bc >= ratios.cd_bc_min - tolerance && cd_bc <= ratios.cd_bc_max + tolerance &&
        ad_xa >= ratios.ad_xa_min - tolerance && ad_xa <= ratios.ad_xa_max + tolerance
    }
}

impl Default for FibonacciPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}