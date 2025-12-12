//! Type definitions for Fibonacci pattern detection

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Types of harmonic patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    Gartley,
    Butterfly,
    Bat,
    Crab,
    Shark,
}

impl PatternType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PatternType::Gartley => "gartley",
            PatternType::Butterfly => "butterfly",
            PatternType::Bat => "bat",
            PatternType::Crab => "crab",
            PatternType::Shark => "shark",
        }
    }
}

/// Represents a significant point in a harmonic pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPoint {
    pub index: usize,
    pub price: f64,
    pub role: String, // X, A, B, C, D
    pub timestamp: Option<DateTime<Utc>>,
    pub confidence: f64,
}

/// Configuration for a specific harmonic pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    pub pattern_type: PatternType,
    pub ratios: HarmonicRatios,
    pub tolerance: f64,
    pub min_pattern_size: usize,
}

/// Harmonic ratios for pattern validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicRatios {
    // AB/XA ratios
    pub ab_xa_min: f64,
    pub ab_xa_max: f64,
    
    // BC/AB ratios  
    pub bc_ab_min: f64,
    pub bc_ab_max: f64,
    
    // CD/BC ratios
    pub cd_bc_min: f64,
    pub cd_bc_max: f64,
    
    // AD/XA ratios
    pub ad_xa_min: f64,
    pub ad_xa_max: f64,
}

/// A detected harmonic pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub points: Vec<PatternPoint>, // X, A, B, C, D
    pub confidence: f64,
    pub completion_time: Option<DateTime<Utc>>,
    pub is_bullish: bool,
    pub validation_score: f64,
    
    // Calculated ratios
    pub ab_xa_ratio: f64,
    pub bc_ab_ratio: f64,
    pub cd_bc_ratio: f64,
    pub ad_xa_ratio: f64,
    
    // Pattern metrics
    pub pattern_height: f64,
    pub pattern_duration: usize,
    pub volume_confirmation: Option<f64>,
}

/// Parameters for pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternParameters {
    pub min_pattern_bars: usize,
    pub max_pattern_bars: usize,
    pub swing_detection_period: usize,
    pub ratio_tolerance: f64,
    pub min_confidence: f64,
    pub enable_volume_confirmation: bool,
    pub max_patterns_per_scan: usize,
}

impl Default for PatternParameters {
    fn default() -> Self {
        Self {
            min_pattern_bars: 20,
            max_pattern_bars: 200,
            swing_detection_period: 5,
            ratio_tolerance: 0.05,
            min_confidence: 0.7,
            enable_volume_confirmation: false,
            max_patterns_per_scan: 10,
        }
    }
}

/// Result of pattern detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    pub detected_patterns: Vec<DetectedPattern>,
    pub scan_period: usize,
    pub computation_time_ns: u64,
    pub patterns_found: usize,
    pub swing_points_detected: usize,
    
    // Performance metrics
    pub scan_time_ns: u64,
    pub validation_time_ns: u64,
    pub ratio_calculation_time_ns: u64,
}

impl PatternResult {
    pub fn new() -> Self {
        Self {
            detected_patterns: Vec::new(),
            scan_period: 0,
            computation_time_ns: 0,
            patterns_found: 0,
            swing_points_detected: 0,
            scan_time_ns: 0,
            validation_time_ns: 0,
            ratio_calculation_time_ns: 0,
        }
    }
    
    pub fn has_patterns(&self) -> bool {
        !self.detected_patterns.is_empty()
    }
    
    pub fn get_patterns_by_type(&self, pattern_type: PatternType) -> Vec<&DetectedPattern> {
        self.detected_patterns
            .iter()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }
    
    pub fn get_bullish_patterns(&self) -> Vec<&DetectedPattern> {
        self.detected_patterns
            .iter()
            .filter(|p| p.is_bullish)
            .collect()
    }
    
    pub fn get_bearish_patterns(&self) -> Vec<&DetectedPattern> {
        self.detected_patterns
            .iter()
            .filter(|p| !p.is_bullish)
            .collect()
    }
    
    pub fn highest_confidence_pattern(&self) -> Option<&DetectedPattern> {
        self.detected_patterns
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }
}

impl Default for PatternResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Swing point identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwingPoint {
    pub index: usize,
    pub price: f64,
    pub is_high: bool,
    pub strength: f64,
    pub confirmed: bool,
}

/// Pattern validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub score: f64,
    pub ratio_scores: Vec<f64>,
    pub total_deviation: f64,
    pub failed_ratios: Vec<String>,
}

/// Market data for pattern detection
#[derive(Debug, Clone)]
pub struct MarketData {
    pub high: Array1<f64>,
    pub low: Array1<f64>,
    pub close: Array1<f64>,
    pub volume: Option<Array1<f64>>,
    pub timestamps: Option<Vec<DateTime<Utc>>>,
}

impl MarketData {
    pub fn new(high: Array1<f64>, low: Array1<f64>, close: Array1<f64>) -> Self {
        Self {
            high,
            low,
            close,
            volume: None,
            timestamps: None,
        }
    }
    
    pub fn len(&self) -> usize {
        self.high.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn with_volume(mut self, volume: Array1<f64>) -> Self {
        self.volume = Some(volume);
        self
    }
    
    pub fn with_timestamps(mut self, timestamps: Vec<DateTime<Utc>>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }
}