//! Market State Representation for Q* Algorithm
//! 
//! Defines market state structures optimized for high-frequency trading
//! decisions with ultra-low latency processing.

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::{QStarError, QStarAction};

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Trending market with clear directional movement
    Trending,
    
    /// Ranging market with sideways movement
    Ranging,
    
    /// High volatility market with rapid price swings
    Volatile,
    
    /// Breakout market with potential regime change
    Breakout,
    
    /// Low liquidity market with wide spreads
    IlliquidMarket,
    
    /// News-driven market with fundamental catalysts
    NewsDriven,
}

/// Comprehensive market state for Q* algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Current price
    pub price: f64,
    
    /// Trading volume
    pub volume: f64,
    
    /// Market volatility (realized)
    pub volatility: f64,
    
    /// RSI indicator (0.0 to 1.0)
    pub rsi: f64,
    
    /// MACD value
    pub macd: f64,
    
    /// Current market regime
    pub market_regime: MarketRegime,
    
    /// Timestamp of state
    pub timestamp: DateTime<Utc>,
    
    /// Additional technical features
    pub features: Vec<f64>,
}

impl Hash for MarketState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash discretized values for consistent state representation
        ((self.price * 100.0) as u64).hash(state);
        ((self.volume * 1000.0) as u64).hash(state);
        ((self.volatility * 10000.0) as u64).hash(state);
        ((self.rsi * 100.0) as u32).hash(state);
        ((self.macd * 1000.0) as i32).hash(state);
        self.market_regime.hash(state);
    }
}

impl MarketState {
    /// Create new market state
    pub fn new(
        price: f64,
        volume: f64,
        volatility: f64,
        rsi: f64,
        macd: f64,
        market_regime: MarketRegime,
        features: Vec<f64>,
    ) -> Self {
        Self {
            price,
            volume,
            volatility,
            rsi,
            macd,
            market_regime,
            timestamp: Utc::now(),
            features,
        }
    }
    
    /// Get legal actions for current state
    pub fn get_legal_actions(&self) -> Vec<QStarAction> {
        let mut actions = Vec::new();
        
        // Always allow hold action
        actions.push(QStarAction::Hold);
        
        // Buy actions with different amounts
        for &amount in &[0.1, 0.25, 0.5, 1.0] {
            actions.push(QStarAction::Buy { amount });
        }
        
        // Sell actions with different amounts
        for &amount in &[0.1, 0.25, 0.5, 1.0] {
            actions.push(QStarAction::Sell { amount });
        }
        
        // Stop loss if in position (simplified logic)
        if self.has_position() {
            actions.push(QStarAction::StopLoss { threshold: 0.02 });
        }
        
        // Take profit if in position
        if self.has_position() {
            actions.push(QStarAction::TakeProfit { threshold: 0.05 });
        }
        
        actions
    }
    
    /// Check if currently holding a position
    fn has_position(&self) -> bool {
        // Simplified position detection
        // In practice, this would check actual portfolio state
        self.features.get(0).map(|&f| f > 0.0).unwrap_or(false)
    }
    
    /// Apply action to state and get next state
    pub fn apply_action(&self, action: &QStarAction) -> Result<MarketState, QStarError> {
        let mut next_state = self.clone();
        
        // Update timestamp
        next_state.timestamp = Utc::now();
        
        // Simulate market dynamics (simplified)
        match action {
            QStarAction::Buy { amount } => {
                // Buying pressure increases price slightly
                next_state.price *= 1.0 + (amount * 0.001);
                next_state.volume += amount * 1000.0;
                
                // Update position in features
                if let Some(position) = next_state.features.get_mut(0) {
                    *position += amount;
                }
            }
            
            QStarAction::Sell { amount } => {
                // Selling pressure decreases price slightly
                next_state.price *= 1.0 - (amount * 0.001);
                next_state.volume += amount * 1000.0;
                
                // Update position in features
                if let Some(position) = next_state.features.get_mut(0) {
                    *position -= amount;
                }
            }
            
            QStarAction::Hold => {
                // No immediate price impact, slight volume decay
                next_state.volume *= 0.99;
            }
            
            QStarAction::StopLoss { .. } => {
                // Stop loss execution (close position)
                if let Some(position) = next_state.features.get_mut(0) {
                    *position = 0.0;
                }
                next_state.price *= 0.98; // Price impact from stop loss
            }
            
            QStarAction::TakeProfit { .. } => {
                // Take profit execution (close position)
                if let Some(position) = next_state.features.get_mut(0) {
                    *position = 0.0;
                }
                next_state.price *= 1.02; // Price impact from take profit
            }
            
            QStarAction::CloseAll => {
                // Close all positions
                if let Some(position) = next_state.features.get_mut(0) {
                    *position = 0.0;
                }
                next_state.volume *= 1.5; // Increased volume from closing
            }
            
            QStarAction::Rebalance { weights } => {
                // Rebalance portfolio according to weights
                let total_weight: f64 = weights.iter().sum();
                if total_weight > 0.0 {
                    next_state.volume += 1000.0; // Rebalancing activity
                }
            }
            
            QStarAction::Scale { factor } => {
                // Scale position by factor
                if let Some(position) = next_state.features.get_mut(0) {
                    *position *= factor;
                }
            }
            
            QStarAction::Hedge { ratio } => {
                // Hedge position with opposite trade
                if next_state.features.len() > 1 {
                    let position = next_state.features[0];
                    next_state.features[1] = -position * ratio;
                }
            }
            
            QStarAction::Wait => {
                // Do nothing, just advance time
                next_state.volume *= 0.95;
            }
        }
        
        // Update derived indicators
        next_state.update_indicators();
        
        Ok(next_state)
    }
    
    /// Update technical indicators based on price changes
    fn update_indicators(&mut self) {
        // Simplified indicator updates
        
        // Update RSI based on price momentum
        let price_change = (self.price - 50000.0) / 50000.0; // Relative to base price
        self.rsi = (0.5 + price_change * 0.1).clamp(0.0, 1.0);
        
        // Update MACD based on short-term price action
        self.macd = price_change * 0.001;
        
        // Update volatility based on recent price action
        self.volatility = (self.volatility * 0.9 + price_change.abs() * 0.1).clamp(0.001, 0.1);
        
        // Update market regime based on volatility and trend
        self.market_regime = self.classify_market_regime();
    }
    
    /// Classify current market regime
    fn classify_market_regime(&self) -> MarketRegime {
        if self.volatility > 0.05 {
            MarketRegime::Volatile
        } else if self.rsi > 0.7 || self.rsi < 0.3 {
            if self.volume > 1500000.0 {
                MarketRegime::Breakout
            } else {
                MarketRegime::Trending
            }
        } else if self.volume < 500000.0 {
            MarketRegime::IlliquidMarket
        } else {
            MarketRegime::Ranging
        }
    }
    
    /// Convert state to feature vector for neural networks
    pub fn to_feature_vector(&self) -> Array1<f64> {
        let mut features = vec![
            self.price / 100000.0,        // Normalized price
            self.volume / 1000000.0,      // Normalized volume
            self.volatility * 100.0,      // Volatility percentage
            self.rsi,                     // RSI (already 0-1)
            self.macd * 1000.0,          // Scaled MACD
        ];
        
        // Add market regime as one-hot encoding
        let regime_encoding = self.encode_market_regime();
        features.extend(regime_encoding);
        
        // Add custom features
        features.extend(&self.features);
        
        Array1::from_vec(features)
    }
    
    /// Encode market regime as one-hot vector
    fn encode_market_regime(&self) -> Vec<f64> {
        let mut encoding = vec![0.0; 6]; // 6 regime types
        
        let index = match self.market_regime {
            MarketRegime::Trending => 0,
            MarketRegime::Ranging => 1,
            MarketRegime::Volatile => 2,
            MarketRegime::Breakout => 3,
            MarketRegime::IlliquidMarket => 4,
            MarketRegime::NewsDriven => 5,
        };
        
        encoding[index] = 1.0;
        encoding
    }
    
    /// Create state from feature vector (inverse of to_feature_vector)
    pub fn from_feature_vector(features: &Array1<f64>) -> Result<Self, QStarError> {
        if features.len() < 11 { // 5 basic + 6 regime encoding
            return Err(QStarError::StateError(
                "Insufficient features for state reconstruction".to_string()
            ));
        }
        
        let price = features[0] * 100000.0;
        let volume = features[1] * 1000000.0;
        let volatility = features[2] / 100.0;
        let rsi = features[3];
        let macd = features[4] / 1000.0;
        
        // Decode market regime from one-hot encoding
        let regime_start = 5;
        let regime_end = regime_start + 6;
        let regime_encoding = &features.slice(ndarray::s![regime_start..regime_end]);
        
        let market_regime = Self::decode_market_regime(regime_encoding)?;
        
        // Extract custom features
        let custom_features = if features.len() > regime_end {
            features.slice(ndarray::s![regime_end..]).to_vec()
        } else {
            Vec::new()
        };
        
        Ok(Self::new(
            price,
            volume,
            volatility,
            rsi,
            macd,
            market_regime,
            custom_features,
        ))
    }
    
    /// Decode market regime from one-hot encoding
    fn decode_market_regime(encoding: &ndarray::ArrayView1<f64>) -> Result<MarketRegime, QStarError> {
        let max_index = encoding.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| QStarError::StateError("Empty regime encoding".to_string()))?;
        
        match max_index {
            0 => Ok(MarketRegime::Trending),
            1 => Ok(MarketRegime::Ranging),
            2 => Ok(MarketRegime::Volatile),
            3 => Ok(MarketRegime::Breakout),
            4 => Ok(MarketRegime::IlliquidMarket),
            5 => Ok(MarketRegime::NewsDriven),
            _ => Err(QStarError::StateError(
                format!("Invalid regime index: {}", max_index)
            )),
        }
    }
    
    /// Calculate state similarity for clustering
    pub fn similarity(&self, other: &MarketState) -> f64 {
        let self_features = self.to_feature_vector();
        let other_features = other.to_feature_vector();
        
        // Cosine similarity
        let dot_product = self_features.dot(&other_features);
        let self_norm = self_features.dot(&self_features).sqrt();
        let other_norm = other_features.dot(&other_features).sqrt();
        
        if self_norm == 0.0 || other_norm == 0.0 {
            0.0
        } else {
            dot_product / (self_norm * other_norm)
        }
    }
    
    /// Check if state is terminal (end of trading session, etc.)
    pub fn is_terminal(&self) -> bool {
        // Simple terminal conditions
        self.price <= 0.0 || 
        self.volume <= 0.0 ||
        self.volatility > 0.2 // Circuit breaker level
    }
    
    /// Get state stability score (0.0 = unstable, 1.0 = stable)
    pub fn stability_score(&self) -> f64 {
        let volatility_score = (1.0 - (self.volatility * 10.0)).clamp(0.0, 1.0);
        let volume_score = (self.volume / 1000000.0).clamp(0.0, 1.0);
        let regime_score = match self.market_regime {
            MarketRegime::Ranging => 1.0,
            MarketRegime::Trending => 0.8,
            MarketRegime::Breakout => 0.6,
            MarketRegime::Volatile => 0.3,
            MarketRegime::IlliquidMarket => 0.4,
            MarketRegime::NewsDriven => 0.2,
        };
        
        (volatility_score + volume_score + regime_score) / 3.0
    }
}

/// Market state builder for easy construction
pub struct MarketStateBuilder {
    price: Option<f64>,
    volume: Option<f64>,
    volatility: Option<f64>,
    rsi: Option<f64>,
    macd: Option<f64>,
    market_regime: Option<MarketRegime>,
    features: Vec<f64>,
}

impl MarketStateBuilder {
    pub fn new() -> Self {
        Self {
            price: None,
            volume: None,
            volatility: None,
            rsi: None,
            macd: None,
            market_regime: None,
            features: Vec::new(),
        }
    }
    
    pub fn price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }
    
    pub fn volume(mut self, volume: f64) -> Self {
        self.volume = Some(volume);
        self
    }
    
    pub fn volatility(mut self, volatility: f64) -> Self {
        self.volatility = Some(volatility);
        self
    }
    
    pub fn rsi(mut self, rsi: f64) -> Self {
        self.rsi = Some(rsi);
        self
    }
    
    pub fn macd(mut self, macd: f64) -> Self {
        self.macd = Some(macd);
        self
    }
    
    pub fn market_regime(mut self, regime: MarketRegime) -> Self {
        self.market_regime = Some(regime);
        self
    }
    
    pub fn add_feature(mut self, feature: f64) -> Self {
        self.features.push(feature);
        self
    }
    
    pub fn features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }
    
    pub fn build(self) -> Result<MarketState, QStarError> {
        Ok(MarketState::new(
            self.price.ok_or_else(|| QStarError::StateError("Price is required".to_string()))?,
            self.volume.unwrap_or(1000000.0),
            self.volatility.unwrap_or(0.02),
            self.rsi.unwrap_or(0.5),
            self.macd.unwrap_or(0.0),
            self.market_regime.unwrap_or(MarketRegime::Ranging),
            self.features,
        ))
    }
}

impl Default for MarketStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_state_creation() {
        let state = MarketState::new(
            50000.0,
            1000000.0,
            0.02,
            0.5,
            0.001,
            MarketRegime::Trending,
            vec![0.1, 0.2],
        );
        
        assert_eq!(state.price, 50000.0);
        assert_eq!(state.market_regime, MarketRegime::Trending);
        assert_eq!(state.features.len(), 2);
    }
    
    #[test]
    fn test_market_state_builder() {
        let state = MarketStateBuilder::new()
            .price(45000.0)
            .volume(800000.0)
            .volatility(0.03)
            .market_regime(MarketRegime::Volatile)
            .add_feature(0.5)
            .build()
            .unwrap();
        
        assert_eq!(state.price, 45000.0);
        assert_eq!(state.market_regime, MarketRegime::Volatile);
        assert_eq!(state.features.len(), 1);
    }
    
    #[test]
    fn test_legal_actions() {
        let state = MarketState::new(
            50000.0,
            1000000.0,
            0.02,
            0.5,
            0.001,
            MarketRegime::Trending,
            vec![0.0], // No position
        );
        
        let actions = state.get_legal_actions();
        assert!(!actions.is_empty());
        assert!(actions.contains(&QStarAction::Hold));
    }
    
    #[test]
    fn test_feature_vector_conversion() {
        let state = MarketState::new(
            50000.0,
            1000000.0,
            0.02,
            0.5,
            0.001,
            MarketRegime::Trending,
            vec![0.1, 0.2],
        );
        
        let features = state.to_feature_vector();
        assert!(features.len() >= 13); // 5 basic + 6 regime + 2 custom
        
        // Test round-trip conversion
        let reconstructed = MarketState::from_feature_vector(&features).unwrap();
        assert!((reconstructed.price - state.price).abs() < 1.0);
        assert_eq!(reconstructed.market_regime, state.market_regime);
    }
    
    #[test]
    fn test_state_similarity() {
        let state1 = MarketState::new(
            50000.0, 1000000.0, 0.02, 0.5, 0.001,
            MarketRegime::Trending, vec![0.1],
        );
        
        let state2 = MarketState::new(
            50100.0, 1010000.0, 0.021, 0.51, 0.0011,
            MarketRegime::Trending, vec![0.11],
        );
        
        let similarity = state1.similarity(&state2);
        assert!(similarity > 0.9); // Should be very similar
    }
    
    #[test]
    fn test_stability_score() {
        let stable_state = MarketState::new(
            50000.0, 1000000.0, 0.01, 0.5, 0.001,
            MarketRegime::Ranging, vec![],
        );
        
        let volatile_state = MarketState::new(
            50000.0, 500000.0, 0.08, 0.9, 0.01,
            MarketRegime::Volatile, vec![],
        );
        
        assert!(stable_state.stability_score() > volatile_state.stability_score());
    }
}