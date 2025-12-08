//! Standard market factors and data structures

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Standard market factors for signal alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StandardFactors {
    Trend,
    Volatility,
    Momentum,
    Sentiment,
    Liquidity,
    Correlation,
    Cycle,
    Anomaly,
}

impl StandardFactors {
    /// Get all factors in order
    pub fn all() -> Vec<Self> {
        vec![
            Self::Trend,
            Self::Volatility,
            Self::Momentum,
            Self::Sentiment,
            Self::Liquidity,
            Self::Correlation,
            Self::Cycle,
            Self::Anomaly,
        ]
    }
    
    /// Get default weights for factors
    pub fn default_weights() -> HashMap<Self, f64> {
        let mut weights = HashMap::new();
        weights.insert(Self::Trend, 0.60);
        weights.insert(Self::Volatility, 0.50);
        weights.insert(Self::Momentum, 0.55);
        weights.insert(Self::Sentiment, 0.45);
        weights.insert(Self::Liquidity, 0.35);
        weights.insert(Self::Correlation, 0.40);
        weights.insert(Self::Cycle, 0.50);
        weights.insert(Self::Anomaly, 0.30);
        weights
    }
    
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trend => "trend",
            Self::Volatility => "volatility",
            Self::Momentum => "momentum",
            Self::Sentiment => "sentiment",
            Self::Liquidity => "liquidity",
            Self::Correlation => "correlation",
            Self::Cycle => "cycle",
            Self::Anomaly => "anomaly",
        }
    }
    
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trend" => Some(Self::Trend),
            "volatility" => Some(Self::Volatility),
            "momentum" => Some(Self::Momentum),
            "sentiment" => Some(Self::Sentiment),
            "liquidity" => Some(Self::Liquidity),
            "correlation" => Some(Self::Correlation),
            "cycle" => Some(Self::Cycle),
            "anomaly" => Some(Self::Anomaly),
            _ => None,
        }
    }
}

/// Market data containing all standard factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub trend: f64,
    pub volatility: f64,
    pub momentum: f64,
    pub sentiment: f64,
    pub liquidity: f64,
    pub correlation: f64,
    pub cycle: f64,
    pub anomaly: f64,
    pub timestamp: DateTime<Utc>,
}

impl MarketData {
    /// Create new market data with current timestamp
    pub fn new(
        trend: f64,
        volatility: f64,
        momentum: f64,
        sentiment: f64,
        liquidity: f64,
        correlation: f64,
        cycle: f64,
        anomaly: f64,
    ) -> Self {
        Self {
            trend,
            volatility,
            momentum,
            sentiment,
            liquidity,
            correlation,
            cycle,
            anomaly,
            timestamp: Utc::now(),
        }
    }
    
    /// Get value for a specific factor
    pub fn get_factor(&self, factor: StandardFactors) -> f64 {
        match factor {
            StandardFactors::Trend => self.trend,
            StandardFactors::Volatility => self.volatility,
            StandardFactors::Momentum => self.momentum,
            StandardFactors::Sentiment => self.sentiment,
            StandardFactors::Liquidity => self.liquidity,
            StandardFactors::Correlation => self.correlation,
            StandardFactors::Cycle => self.cycle,
            StandardFactors::Anomaly => self.anomaly,
        }
    }
    
    /// Set value for a specific factor
    pub fn set_factor(&mut self, factor: StandardFactors, value: f64) {
        match factor {
            StandardFactors::Trend => self.trend = value,
            StandardFactors::Volatility => self.volatility = value,
            StandardFactors::Momentum => self.momentum = value,
            StandardFactors::Sentiment => self.sentiment = value,
            StandardFactors::Liquidity => self.liquidity = value,
            StandardFactors::Correlation => self.correlation = value,
            StandardFactors::Cycle => self.cycle = value,
            StandardFactors::Anomaly => self.anomaly = value,
        }
    }
    
    /// Convert to HashMap for easier manipulation
    pub fn to_hashmap(&self) -> HashMap<StandardFactors, f64> {
        let mut map = HashMap::new();
        for factor in StandardFactors::all() {
            map.insert(factor, self.get_factor(factor));
        }
        map
    }
    
    /// Create from HashMap
    pub fn from_hashmap(map: HashMap<StandardFactors, f64>) -> Self {
        Self {
            trend: map.get(&StandardFactors::Trend).copied().unwrap_or(0.0),
            volatility: map.get(&StandardFactors::Volatility).copied().unwrap_or(0.0),
            momentum: map.get(&StandardFactors::Momentum).copied().unwrap_or(0.0),
            sentiment: map.get(&StandardFactors::Sentiment).copied().unwrap_or(0.0),
            liquidity: map.get(&StandardFactors::Liquidity).copied().unwrap_or(0.0),
            correlation: map.get(&StandardFactors::Correlation).copied().unwrap_or(0.0),
            cycle: map.get(&StandardFactors::Cycle).copied().unwrap_or(0.0),
            anomaly: map.get(&StandardFactors::Anomaly).copied().unwrap_or(0.0),
            timestamp: Utc::now(),
        }
    }
    
    /// Normalize all factors to [-1, 1] range
    pub fn normalize(&mut self) {
        self.trend = self.trend.clamp(-1.0, 1.0);
        self.volatility = self.volatility.clamp(0.0, 1.0);
        self.momentum = self.momentum.clamp(-1.0, 1.0);
        self.sentiment = self.sentiment.clamp(-1.0, 1.0);
        self.liquidity = self.liquidity.clamp(0.0, 1.0);
        self.correlation = self.correlation.clamp(-1.0, 1.0);
        self.cycle = self.cycle.clamp(-1.0, 1.0);
        self.anomaly = self.anomaly.clamp(0.0, 1.0);
    }
    
    /// Generate random market data for testing
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        Self {
            trend: rng.gen_range(-1.0..1.0),
            volatility: rng.gen_range(0.0..1.0),
            momentum: rng.gen_range(-1.0..1.0),
            sentiment: rng.gen_range(-1.0..1.0),
            liquidity: rng.gen_range(0.0..1.0),
            correlation: rng.gen_range(-1.0..1.0),
            cycle: rng.gen_range(-1.0..1.0),
            anomaly: rng.gen_range(0.0..1.0),
            timestamp: Utc::now(),
        }
    }
}

/// Factor weights for agent decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorWeights {
    weights: HashMap<StandardFactors, f64>,
}

impl FactorWeights {
    /// Create new factor weights
    pub fn new(weights: HashMap<StandardFactors, f64>) -> Self {
        Self { weights }
    }
    
    /// Create with default weights
    pub fn default() -> Self {
        Self {
            weights: StandardFactors::default_weights(),
        }
    }
    
    /// Get weight for a factor
    pub fn get(&self, factor: StandardFactors) -> f64 {
        self.weights.get(&factor).copied().unwrap_or(0.0)
    }
    
    /// Set weight for a factor
    pub fn set(&mut self, factor: StandardFactors, weight: f64) {
        self.weights.insert(factor, weight);
    }
    
    /// Update weight with delta
    pub fn update(&mut self, factor: StandardFactors, delta: f64) {
        let current = self.get(factor);
        self.set(factor, current + delta);
    }
    
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.values().sum();
        if sum > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= sum;
            }
        }
    }
    
    /// Get all weights as vector in standard order
    pub fn as_vector(&self) -> Vec<f64> {
        StandardFactors::all()
            .iter()
            .map(|&factor| self.get(factor))
            .collect()
    }
    
    /// Create from vector in standard order
    pub fn from_vector(values: Vec<f64>) -> Self {
        let mut weights = HashMap::new();
        for (factor, value) in StandardFactors::all().into_iter().zip(values) {
            weights.insert(factor, value);
        }
        Self { weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_standard_factors() {
        let factors = StandardFactors::all();
        assert_eq!(factors.len(), 8);
        
        let weights = StandardFactors::default_weights();
        assert_eq!(weights.len(), 8);
        
        // Test string conversion
        assert_eq!(StandardFactors::Trend.as_str(), "trend");
        assert_eq!(StandardFactors::from_str("trend"), Some(StandardFactors::Trend));
    }
    
    #[test]
    fn test_market_data() {
        let mut data = MarketData::random();
        data.normalize();
        
        assert!(data.trend >= -1.0 && data.trend <= 1.0);
        assert!(data.volatility >= 0.0 && data.volatility <= 1.0);
        
        // Test factor access
        assert_eq!(data.get_factor(StandardFactors::Trend), data.trend);
        
        // Test HashMap conversion
        let map = data.to_hashmap();
        assert_eq!(map.len(), 8);
        
        let data2 = MarketData::from_hashmap(map);
        assert_eq!(data2.trend, data.trend);
    }
    
    #[test]
    fn test_factor_weights() {
        let mut weights = FactorWeights::default();
        
        // Test weight access
        assert!(weights.get(StandardFactors::Trend) > 0.0);
        
        // Test normalization
        weights.normalize();
        let sum: f64 = StandardFactors::all()
            .iter()
            .map(|&f| weights.get(f))
            .sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}