//! Fibonacci pattern detection for CDFA
//! 
//! High-performance harmonic pattern detection consolidated from 
//! cdfa-fibonacci-pattern-detector crate.

use ndarray::{Array1, ArrayView1};
use crate::error::{CdfaError, Result};
use super::{UnifiedDetector, DetectorMetrics};
use std::time::Instant;

#[cfg(feature = "simd")]
use crate::simd::unified as simd;

/// Fibonacci pattern detector with sub-microsecond performance
pub struct FibonacciPatternDetector {
    config: FibonacciConfig,
    metrics: DetectorMetrics,
}

/// Configuration for Fibonacci pattern detection
#[derive(Debug, Clone)]
pub struct FibonacciConfig {
    pub min_pattern_length: usize,
    pub max_pattern_length: usize,
    pub tolerance: f64,
    pub min_amplitude: f64,
    pub use_simd: bool,
    pub pattern_types: Vec<HarmonicPatternType>,
}

impl Default for FibonacciConfig {
    fn default() -> Self {
        Self {
            min_pattern_length: 5,
            max_pattern_length: 50,
            tolerance: 0.05, // 5% tolerance
            min_amplitude: 0.01, // 1% minimum amplitude
            use_simd: true,
            pattern_types: vec![
                HarmonicPatternType::Gartley,
                HarmonicPatternType::Butterfly,
                HarmonicPatternType::Bat,
                HarmonicPatternType::Crab,
            ],
        }
    }
}

/// Types of harmonic patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarmonicPatternType {
    Gartley,
    Butterfly,
    Bat,
    Crab,
    Shark,
    Cypher,
    ThreeDrives,
    ABCD,
}

/// Detected Fibonacci pattern
#[derive(Debug, Clone)]
pub struct FibonacciPattern {
    pub pattern_type: HarmonicPatternType,
    pub points: Vec<PatternPoint>,
    pub confidence: f64,
    pub start_index: usize,
    pub end_index: usize,
    pub ratios: HarmonicRatios,
}

/// Point in a harmonic pattern
#[derive(Debug, Clone)]
pub struct PatternPoint {
    pub index: usize,
    pub price: f64,
    pub point_type: PointType,
}

#[derive(Debug, Clone, Copy)]
pub enum PointType {
    X, A, B, C, D
}

/// Harmonic ratios for pattern validation
#[derive(Debug, Clone)]
pub struct HarmonicRatios {
    pub ab_xa: f64,  // AB retracement of XA
    pub bc_ab: f64,  // BC retracement of AB  
    pub cd_bc: f64,  // CD extension of BC
    pub ad_xa: f64,  // AD retracement of XA
}

/// Harmonic pattern type with standard ratios
#[derive(Debug, Clone)]
pub struct HarmonicPattern {
    pub name: &'static str,
    pub ab_xa_range: (f64, f64),
    pub bc_ab_range: (f64, f64),
    pub cd_bc_range: (f64, f64),
    pub ad_xa_range: (f64, f64),
}

impl FibonacciPatternDetector {
    /// Create new detector with default configuration
    pub fn new() -> Self {
        Self::with_config(FibonacciConfig::default())
    }
    
    /// Create new detector with custom configuration
    pub fn with_config(config: FibonacciConfig) -> Self {
        Self {
            config,
            metrics: DetectorMetrics::new(),
        }
    }
    
    /// Detect harmonic patterns in price data
    pub fn detect_patterns(
        &mut self,
        high: &ArrayView1<f64>,
        low: &ArrayView1<f64>,
        close: &ArrayView1<f64>,
    ) -> Result<Vec<FibonacciPattern>> {
        let start_time = Instant::now();
        
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CdfaError::invalid_input(
                "High, low, and close arrays must have same length".to_string()
            ));
        }
        
        if high.len() < self.config.min_pattern_length {
            return Ok(Vec::new());
        }
        
        // Find pivot points
        let pivots = self.find_pivot_points(high, low)?;
        
        // Detect patterns using pivot points
        let patterns = self.detect_harmonic_patterns(&pivots, high, low, close)?;
        
        // Update metrics
        let latency = start_time.elapsed().as_nanos() as u64;
        for pattern in &patterns {
            self.metrics.update(pattern.confidence > 0.7, latency);
        }
        
        Ok(patterns)
    }
    
    /// Find pivot points (peaks and troughs) in price data
    fn find_pivot_points(&self, high: &ArrayView1<f64>, low: &ArrayView1<f64>) -> Result<Vec<PivotPoint>> {
        let n = high.len();
        let mut pivots = Vec::new();
        
        if n < 3 {
            return Ok(pivots);
        }
        
        // Find local maxima and minima
        for i in 1..(n - 1) {
            // Check for peak (local maximum in high prices)
            if high[i] > high[i - 1] && high[i] > high[i + 1] {
                pivots.push(PivotPoint {
                    index: i,
                    price: high[i],
                    pivot_type: PivotType::Peak,
                });
            }
            
            // Check for trough (local minimum in low prices)
            if low[i] < low[i - 1] && low[i] < low[i + 1] {
                pivots.push(PivotPoint {
                    index: i,
                    price: low[i],
                    pivot_type: PivotType::Trough,
                });
            }
        }
        
        // Sort by index
        pivots.sort_by_key(|p| p.index);
        
        Ok(pivots)
    }
    
    /// Detect harmonic patterns from pivot points
    fn detect_harmonic_patterns(
        &self,
        pivots: &[PivotPoint],
        high: &ArrayView1<f64>,
        low: &ArrayView1<f64>,
        close: &ArrayView1<f64>,
    ) -> Result<Vec<FibonacciPattern>> {
        let mut patterns = Vec::new();
        
        if pivots.len() < 5 {
            return Ok(patterns);
        }
        
        // Look for 5-point patterns (XABCD)
        for i in 0..(pivots.len() - 4) {
            let x = &pivots[i];
            let a = &pivots[i + 1];
            let b = &pivots[i + 2];
            let c = &pivots[i + 3];
            let d = &pivots[i + 4];
            
            // Check alternating pivot types
            if !self.is_valid_pattern_sequence(x, a, b, c, d) {
                continue;
            }
            
            // Calculate ratios
            let ratios = self.calculate_harmonic_ratios(x, a, b, c, d);
            
            // Check each pattern type
            for &pattern_type in &self.config.pattern_types {
                if let Some(confidence) = self.validate_pattern(pattern_type, &ratios) {
                    let pattern = FibonacciPattern {
                        pattern_type,
                        points: vec![
                            PatternPoint { index: x.index, price: x.price, point_type: PointType::X },
                            PatternPoint { index: a.index, price: a.price, point_type: PointType::A },
                            PatternPoint { index: b.index, price: b.price, point_type: PointType::B },
                            PatternPoint { index: c.index, price: c.price, point_type: PointType::C },
                            PatternPoint { index: d.index, price: d.price, point_type: PointType::D },
                        ],
                        confidence,
                        start_index: x.index,
                        end_index: d.index,
                        ratios,
                    };
                    
                    patterns.push(pattern);
                }
            }
        }
        
        Ok(patterns)
    }
    
    /// Check if pivot sequence forms valid pattern structure
    fn is_valid_pattern_sequence(&self, x: &PivotPoint, a: &PivotPoint, b: &PivotPoint, c: &PivotPoint, d: &PivotPoint) -> bool {
        // For bullish patterns: X(peak) -> A(trough) -> B(peak) -> C(trough) -> D(peak)
        // For bearish patterns: X(trough) -> A(peak) -> B(trough) -> C(peak) -> D(trough)
        
        matches!(
            (x.pivot_type, a.pivot_type, b.pivot_type, c.pivot_type, d.pivot_type),
            (PivotType::Peak, PivotType::Trough, PivotType::Peak, PivotType::Trough, PivotType::Peak) |
            (PivotType::Trough, PivotType::Peak, PivotType::Trough, PivotType::Peak, PivotType::Trough)
        )
    }
    
    /// Calculate harmonic ratios for pattern validation
    fn calculate_harmonic_ratios(&self, x: &PivotPoint, a: &PivotPoint, b: &PivotPoint, c: &PivotPoint, d: &PivotPoint) -> HarmonicRatios {
        let xa = (a.price - x.price).abs();
        let ab = (b.price - a.price).abs();
        let bc = (c.price - b.price).abs();
        let cd = (d.price - c.price).abs();
        let ad = (d.price - a.price).abs();
        
        HarmonicRatios {
            ab_xa: if xa > 0.0 { ab / xa } else { 0.0 },
            bc_ab: if ab > 0.0 { bc / ab } else { 0.0 },
            cd_bc: if bc > 0.0 { cd / bc } else { 0.0 },
            ad_xa: if xa > 0.0 { ad / xa } else { 0.0 },
        }
    }
    
    /// Validate pattern against standard harmonic ratios
    fn validate_pattern(&self, pattern_type: HarmonicPatternType, ratios: &HarmonicRatios) -> Option<f64> {
        let pattern_spec = self.get_pattern_specification(pattern_type);
        
        let mut score = 0.0;
        let mut count = 0;
        
        // Check AB/XA ratio
        if self.is_ratio_in_range(ratios.ab_xa, pattern_spec.ab_xa_range) {
            score += 1.0;
        }
        count += 1;
        
        // Check BC/AB ratio
        if self.is_ratio_in_range(ratios.bc_ab, pattern_spec.bc_ab_range) {
            score += 1.0;
        }
        count += 1;
        
        // Check CD/BC ratio
        if self.is_ratio_in_range(ratios.cd_bc, pattern_spec.cd_bc_range) {
            score += 1.0;
        }
        count += 1;
        
        // Check AD/XA ratio
        if self.is_ratio_in_range(ratios.ad_xa, pattern_spec.ad_xa_range) {
            score += 1.0;
        }
        count += 1;
        
        let confidence = score / count as f64;
        
        // Require minimum confidence
        if confidence >= 0.7 {
            Some(confidence)
        } else {
            None
        }
    }
    
    /// Check if ratio is within acceptable range
    fn is_ratio_in_range(&self, ratio: f64, range: (f64, f64)) -> bool {
        let tolerance = self.config.tolerance;
        let (min_ratio, max_ratio) = range;
        
        ratio >= (min_ratio - tolerance) && ratio <= (max_ratio + tolerance)
    }
    
    /// Get pattern specification with standard ratios
    fn get_pattern_specification(&self, pattern_type: HarmonicPatternType) -> HarmonicPattern {
        match pattern_type {
            HarmonicPatternType::Gartley => HarmonicPattern {
                name: "Gartley",
                ab_xa_range: (0.618, 0.618),
                bc_ab_range: (0.382, 0.886),
                cd_bc_range: (1.13, 1.618),
                ad_xa_range: (0.786, 0.786),
            },
            HarmonicPatternType::Butterfly => HarmonicPattern {
                name: "Butterfly",
                ab_xa_range: (0.786, 0.786),
                bc_ab_range: (0.382, 0.886),
                cd_bc_range: (1.618, 2.618),
                ad_xa_range: (1.27, 1.27),
            },
            HarmonicPatternType::Bat => HarmonicPattern {
                name: "Bat",
                ab_xa_range: (0.382, 0.5),
                bc_ab_range: (0.382, 0.886),
                cd_bc_range: (1.618, 2.618),
                ad_xa_range: (0.886, 0.886),
            },
            HarmonicPatternType::Crab => HarmonicPattern {
                name: "Crab",
                ab_xa_range: (0.382, 0.618),
                bc_ab_range: (0.382, 0.886),
                cd_bc_range: (2.24, 3.618),
                ad_xa_range: (1.618, 1.618),
            },
            HarmonicPatternType::Shark => HarmonicPattern {
                name: "Shark",
                ab_xa_range: (0.382, 0.618),
                bc_ab_range: (1.13, 1.618),
                cd_bc_range: (1.618, 2.24),
                ad_xa_range: (0.886, 1.13),
            },
            HarmonicPatternType::Cypher => HarmonicPattern {
                name: "Cypher",
                ab_xa_range: (0.382, 0.618),
                bc_ab_range: (1.13, 1.414),
                cd_bc_range: (1.272, 2.0),
                ad_xa_range: (0.786, 0.786),
            },
            HarmonicPatternType::ThreeDrives => HarmonicPattern {
                name: "Three Drives",
                ab_xa_range: (0.618, 0.786),
                bc_ab_range: (0.618, 0.786),
                cd_bc_range: (1.272, 1.618),
                ad_xa_range: (1.272, 1.618),
            },
            HarmonicPatternType::ABCD => HarmonicPattern {
                name: "ABCD",
                ab_xa_range: (0.618, 0.786),
                bc_ab_range: (0.382, 0.886),
                cd_bc_range: (1.272, 1.618),
                ad_xa_range: (1.272, 1.618),
            },
        }
    }
}

impl UnifiedDetector<(ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>)> for FibonacciPatternDetector {
    type Config = FibonacciConfig;
    type Output = Vec<FibonacciPattern>;
    type Error = CdfaError;
    
    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        Ok(Self::with_config(config))
    }
    
    fn detect(&self, data: &(ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>)) -> Result<Self::Output, Self::Error> {
        let mut detector = self.clone();
        detector.detect_patterns(&data.0, &data.1, &data.2)
    }
    
    fn update_config(&mut self, config: Self::Config) -> Result<(), Self::Error> {
        self.config = config;
        Ok(())
    }
    
    fn metrics(&self) -> DetectorMetrics {
        self.metrics.clone()
    }
}

impl Clone for FibonacciPatternDetector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: DetectorMetrics::new(), // Reset metrics for clone
        }
    }
}

impl Default for FibonacciPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Pivot point in price data
#[derive(Debug, Clone)]
struct PivotPoint {
    index: usize,
    price: f64,
    pivot_type: PivotType,
}

#[derive(Debug, Clone, Copy)]
enum PivotType {
    Peak,
    Trough,
}

/// Configuration helpers for common patterns
pub fn gartley_config() -> FibonacciConfig {
    let mut config = FibonacciConfig::default();
    config.pattern_types = vec![HarmonicPatternType::Gartley];
    config.tolerance = 0.05;
    config
}

pub fn butterfly_config() -> FibonacciConfig {
    let mut config = FibonacciConfig::default();
    config.pattern_types = vec![HarmonicPatternType::Butterfly];
    config.tolerance = 0.05;
    config
}

pub fn bat_config() -> FibonacciConfig {
    let mut config = FibonacciConfig::default();
    config.pattern_types = vec![HarmonicPatternType::Bat];
    config.tolerance = 0.05;
    config
}

pub fn crab_config() -> FibonacciConfig {
    let mut config = FibonacciConfig::default();
    config.pattern_types = vec![HarmonicPatternType::Crab];
    config.tolerance = 0.05;
    config
}

pub fn shark_config() -> FibonacciConfig {
    let mut config = FibonacciConfig::default();
    config.pattern_types = vec![HarmonicPatternType::Shark];
    config.tolerance = 0.05;
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_fibonacci_detector_creation() {
        let detector = FibonacciPatternDetector::new();
        assert_eq!(detector.config.min_pattern_length, 5);
        assert_eq!(detector.config.tolerance, 0.05);
    }
    
    #[test]
    fn test_pivot_point_detection() {
        let detector = FibonacciPatternDetector::new();
        let high = array![1.0, 1.2, 1.1, 1.3, 1.05, 1.25];
        let low = array![0.95, 1.15, 1.05, 1.25, 1.0, 1.2];
        
        let pivots = detector.find_pivot_points(&high.view(), &low.view()).unwrap();
        assert!(!pivots.is_empty());
    }
    
    #[test]
    fn test_harmonic_ratio_calculation() {
        let detector = FibonacciPatternDetector::new();
        
        let x = PivotPoint { index: 0, price: 1.0, pivot_type: PivotType::Peak };
        let a = PivotPoint { index: 1, price: 0.5, pivot_type: PivotType::Trough };
        let b = PivotPoint { index: 2, price: 0.8, pivot_type: PivotType::Peak };
        let c = PivotPoint { index: 3, price: 0.6, pivot_type: PivotType::Trough };
        let d = PivotPoint { index: 4, price: 0.9, pivot_type: PivotType::Peak };
        
        let ratios = detector.calculate_harmonic_ratios(&x, &a, &b, &c, &d);
        
        // AB/XA = 0.3/0.5 = 0.6
        assert!((ratios.ab_xa - 0.6).abs() < 1e-10);
    }
    
    #[test]
    fn test_pattern_validation() {
        let detector = FibonacciPatternDetector::new();
        
        // Perfect Gartley ratios
        let ratios = HarmonicRatios {
            ab_xa: 0.618,
            bc_ab: 0.618,
            cd_bc: 1.272,
            ad_xa: 0.786,
        };
        
        let confidence = detector.validate_pattern(HarmonicPatternType::Gartley, &ratios);
        assert!(confidence.is_some());
        assert!(confidence.unwrap() > 0.7);
    }
    
    #[test]
    fn test_pattern_configurations() {
        let gartley = gartley_config();
        assert_eq!(gartley.pattern_types, vec![HarmonicPatternType::Gartley]);
        
        let butterfly = butterfly_config();
        assert_eq!(butterfly.pattern_types, vec![HarmonicPatternType::Butterfly]);
    }
}