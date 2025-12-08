//! Type definitions for Fibonacci analyzer

use serde::{Deserialize, Serialize};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

impl Signal {
    /// Get numeric value of signal
    pub fn value(&self) -> f64 {
        match self {
            Signal::Buy => 1.0,
            Signal::Sell => -1.0,
            Signal::Hold => 0.0,
        }
    }
}

/// Confidence level for signals
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence(pub f64);

impl Confidence {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
    
    pub fn is_high(&self) -> bool {
        self.0 >= 0.8
    }
    
    pub fn is_medium(&self) -> bool {
        self.0 >= 0.5 && self.0 < 0.8
    }
    
    pub fn is_low(&self) -> bool {
        self.0 < 0.5
    }
}

/// Result from Fibonacci analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerResult {
    pub signal: Signal,
    pub confidence: Confidence,
    pub primary_level: f64,
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub volatility_bands: Option<(f64, f64)>,
    pub computation_time_ns: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

impl AnalyzerResult {
    pub fn new(signal: Signal, confidence: f64) -> Self {
        Self {
            signal,
            confidence: Confidence::new(confidence),
            primary_level: 0.0,
            support_levels: Vec::new(),
            resistance_levels: Vec::new(),
            volatility_bands: None,
            computation_time_ns: 0,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Fibonacci level with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciLevel {
    pub level: f64,
    pub price: f64,
    pub strength: f64,
    pub touches: usize,
    pub description: String,
}

/// Time period for analysis
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimePeriod {
    Minutes1,
    Minutes5,
    Minutes15,
    Minutes30,
    Hour1,
    Hour4,
    Day1,
    Week1,
    Month1,
}

impl TimePeriod {
    pub fn to_minutes(&self) -> usize {
        match self {
            TimePeriod::Minutes1 => 1,
            TimePeriod::Minutes5 => 5,
            TimePeriod::Minutes15 => 15,
            TimePeriod::Minutes30 => 30,
            TimePeriod::Hour1 => 60,
            TimePeriod::Hour4 => 240,
            TimePeriod::Day1 => 1440,
            TimePeriod::Week1 => 10080,
            TimePeriod::Month1 => 43200,
        }
    }
}