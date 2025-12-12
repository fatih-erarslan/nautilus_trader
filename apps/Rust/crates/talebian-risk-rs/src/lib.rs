//! # Talebian Risk Management - Machiavellian Parasitic Trading Engine
//!
//! This crate implements aggressive Talebian risk management principles specifically designed
//! for parasitic crypto whale trading. It recalibrates conservative parameters to capture
//! opportunities instead of blocking them.
//!
//! ## Core Philosophy
//! "Be fearful when others are greedy, but be PARASITIC when whales are moving.
//! Antifragility means profiting from volatility, not hiding from it."
//!
//! ## Key Features
//! - Aggressive antifragility detection (threshold 0.7 → 0.35)
//! - Opportunistic barbell strategy (safe ratio 85% → 65%)
//! - Black swan tolerance (0.05 → 0.18)
//! - Aggressive Kelly criterion (0.25 → 0.55)
//! - Real-time whale detection and parasitic opportunity scoring
//! - SIMD-optimized calculations for high-frequency trading

use serde::{Deserialize, Serialize};

pub mod antifragility;
pub mod barbell;
pub mod black_swan;
pub mod kelly;
pub mod whale_detection;
pub mod parasitic_opportunities;
pub mod risk_engine;
pub mod market_regime;
pub mod performance;
pub mod errors;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub use errors::TalebianRiskError;

/// Aggressive Machiavellian configuration for parasitic trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacchiavelianConfig {
    // Aggressive Antifragility Parameters
    pub antifragility_threshold: f64,          // 0.35 (was 0.7)
    pub volatility_love_factor: f64,           // 1.8 (embrace volatility)
    pub momentum_multiplier: f64,              // 2.2 (chase trends aggressively)
    
    // Opportunistic Barbell Strategy
    pub barbell_safe_ratio: f64,               // 0.65 (was 0.85)
    pub barbell_risky_ratio: f64,              // 0.35 (was 0.15)
    pub whale_detected_multiplier: f64,        // 1.5x when whales move
    pub dynamic_rebalance_threshold: f64,      // 0.1 (10% threshold)
    
    // Black Swan Tolerance
    pub black_swan_threshold: f64,             // 0.18 (was 0.05)
    pub beneficial_swan_multiplier: f64,       // 2.0x for profitable volatility
    pub destructive_swan_protection: f64,      // 0.3 (30% max loss tolerance)
    
    // Aggressive Kelly Criterion
    pub kelly_fraction: f64,                   // 0.55 (was 0.25)
    pub kelly_max_fraction: f64,               // 0.75 (maximum Kelly)
    pub kelly_whale_multiplier: f64,           // 1.3x when following whales
    
    // Whale Detection Parameters
    pub whale_volume_threshold: f64,           // 2.0x average volume
    pub whale_price_impact_threshold: f64,     // 0.02 (2% price impact)
    pub order_book_imbalance_threshold: f64,   // 0.7 (70% imbalance)
    pub smart_money_confidence: f64,           // 0.8 (80% confidence)
    
    // Parasitic Opportunity Scoring
    pub parasitic_opportunity_threshold: f64,  // 0.6 (60% opportunity score)
    pub momentum_window: usize,                // 20 periods
    pub volatility_lookback: usize,            // 50 periods
    
    // Performance Optimization
    pub calculation_frequency_ms: u64,         // 100ms for real-time
    pub risk_update_frequency_ms: u64,         // 500ms
    pub memory_pool_size: usize,               // 10000 calculations
}

impl Default for MacchiavelianConfig {
    fn default() -> Self {
        Self::aggressive_defaults()
    }
}

impl MacchiavelianConfig {
    /// Create aggressive Machiavellian defaults for parasitic trading
    pub fn aggressive_defaults() -> Self {
        Self {
            // Aggressive Antifragility - 50% more permissive
            antifragility_threshold: 0.35,        // Was 0.7 - now 50% lower
            volatility_love_factor: 1.8,          // Embrace volatility
            momentum_multiplier: 2.2,             // Chase strong trends
            
            // Opportunistic Barbell - 20% more risk allocation
            barbell_safe_ratio: 0.65,             // Was 0.85 - reduced by 20%
            barbell_risky_ratio: 0.35,            // Was 0.15 - increased by 20%
            whale_detected_multiplier: 1.5,       // 50% increase when whales move
            dynamic_rebalance_threshold: 0.1,     // Rebalance at 10% drift
            
            // Black Swan Tolerance - 3.6x more tolerant
            black_swan_threshold: 0.18,           // Was 0.05 - 3.6x increase
            beneficial_swan_multiplier: 2.0,      // Double down on good volatility
            destructive_swan_protection: 0.3,     // 30% max drawdown tolerance
            
            // Aggressive Kelly - 2.2x more aggressive
            kelly_fraction: 0.55,                 // Was 0.25 - 2.2x increase
            kelly_max_fraction: 0.75,             // Maximum Kelly fraction
            kelly_whale_multiplier: 1.3,          // 30% boost when following whales
            
            // Whale Detection - Sensitive but not paranoid
            whale_volume_threshold: 2.0,          // 2x average volume
            whale_price_impact_threshold: 0.02,   // 2% price impact
            order_book_imbalance_threshold: 0.7,  // 70% order book imbalance
            smart_money_confidence: 0.8,          // 80% confidence required
            
            // Parasitic Opportunities - Moderately aggressive
            parasitic_opportunity_threshold: 0.6, // 60% opportunity score
            momentum_window: 20,                  // 20 period momentum
            volatility_lookback: 50,              // 50 period volatility
            
            // High-Performance Real-Time
            calculation_frequency_ms: 100,        // 10 Hz calculation rate
            risk_update_frequency_ms: 500,        // 2 Hz risk updates
            memory_pool_size: 10000,              // Large calculation cache
        }
    }
    
    /// Conservative baseline for comparison (the problematic old settings)
    pub fn conservative_baseline() -> Self {
        Self {
            antifragility_threshold: 0.7,         // Original conservative
            volatility_love_factor: 0.8,          // Fear volatility
            momentum_multiplier: 1.0,             // No momentum bias
            
            barbell_safe_ratio: 0.85,             // Original conservative
            barbell_risky_ratio: 0.15,            // Original conservative
            whale_detected_multiplier: 1.0,       // No whale adjustment
            dynamic_rebalance_threshold: 0.05,    // Very tight rebalancing
            
            black_swan_threshold: 0.05,           // Original conservative
            beneficial_swan_multiplier: 1.0,      // No upside capture
            destructive_swan_protection: 0.1,     // Very tight protection
            
            kelly_fraction: 0.25,                 // Original conservative
            kelly_max_fraction: 0.3,              // Very conservative max
            kelly_whale_multiplier: 1.0,          // No whale adjustment
            
            whale_volume_threshold: 3.0,          // Too high threshold
            whale_price_impact_threshold: 0.01,   // Too sensitive
            order_book_imbalance_threshold: 0.8,  // Too strict
            smart_money_confidence: 0.9,          // Too high confidence
            
            parasitic_opportunity_threshold: 0.8, // Too high threshold
            momentum_window: 50,                  // Too long window
            volatility_lookback: 100,             // Too long lookback
            
            calculation_frequency_ms: 1000,       // Too slow
            risk_update_frequency_ms: 5000,       // Way too slow
            memory_pool_size: 1000,               // Too small cache
        }
    }
    
    /// Extreme Machiavellian settings for high-volatility markets
    pub fn extreme_machiavellian() -> Self {
        Self {
            antifragility_threshold: 0.25,        // Extremely permissive
            volatility_love_factor: 2.5,          // Love extreme volatility
            momentum_multiplier: 3.0,             // Extreme momentum chasing
            
            barbell_safe_ratio: 0.55,             // Very aggressive allocation
            barbell_risky_ratio: 0.45,            // Very high risk allocation
            whale_detected_multiplier: 2.0,       // Double when whales move
            dynamic_rebalance_threshold: 0.15,    // Wide rebalance bands
            
            black_swan_threshold: 0.25,           // Very tolerant of swans
            beneficial_swan_multiplier: 3.0,      // Triple down on good volatility
            destructive_swan_protection: 0.4,     // High drawdown tolerance
            
            kelly_fraction: 0.7,                  // Very aggressive sizing
            kelly_max_fraction: 0.9,              // Near-maximum Kelly
            kelly_whale_multiplier: 1.5,          // 50% boost for whales
            
            whale_volume_threshold: 1.5,          // Very sensitive detection
            whale_price_impact_threshold: 0.03,   // Higher impact tolerance
            order_book_imbalance_threshold: 0.6,  // More permissive
            smart_money_confidence: 0.7,          // Lower confidence barrier
            
            parasitic_opportunity_threshold: 0.5, // Lower opportunity barrier
            momentum_window: 15,                  // Shorter momentum
            volatility_lookback: 30,              // Shorter volatility
            
            calculation_frequency_ms: 50,         // Very high frequency
            risk_update_frequency_ms: 250,        // Fast risk updates
            memory_pool_size: 20000,              // Large calculation cache
        }
    }
}

/// Market data structure for real-time risk calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub volatility: f64,
    pub returns: Vec<f64>,
    pub volume_history: Vec<f64>,
}

/// Whale detection result with confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetection {
    pub is_whale_detected: bool,
    pub confidence: f64,
    pub volume_anomaly: f64,
    pub price_impact: f64,
    pub order_book_imbalance: f64,
    pub smart_money_flow: f64,
    pub direction: WhaleDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhaleDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Parasitic opportunity with comprehensive scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticOpportunity {
    pub opportunity_score: f64,
    pub momentum_factor: f64,
    pub volatility_factor: f64,
    pub whale_alignment: f64,
    pub regime_factor: f64,
    pub recommended_allocation: f64,
    pub confidence: f64,
}

/// Complete Talebian risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalebianRiskAssessment {
    pub antifragility_score: f64,
    pub barbell_allocation: (f64, f64), // (safe, risky)
    pub black_swan_probability: f64,
    pub kelly_fraction: f64,
    pub whale_detection: WhaleDetection,
    pub parasitic_opportunity: ParasiticOpportunity,
    pub overall_risk_score: f64,
    pub recommended_position_size: f64,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aggressive_vs_conservative_config() {
        let conservative = MacchiavelianConfig::conservative_baseline();
        let aggressive = MacchiavelianConfig::aggressive_defaults();
        
        // Verify aggressive configuration is more permissive
        assert!(aggressive.antifragility_threshold < conservative.antifragility_threshold);
        assert!(aggressive.barbell_safe_ratio < conservative.barbell_safe_ratio);
        assert!(aggressive.black_swan_threshold > conservative.black_swan_threshold);
        assert!(aggressive.kelly_fraction > conservative.kelly_fraction);
        
        // Verify performance improvements
        assert!(aggressive.calculation_frequency_ms < conservative.calculation_frequency_ms);
        assert!(aggressive.memory_pool_size > conservative.memory_pool_size);
    }
    
    #[test]
    fn test_extreme_machiavellian_config() {
        let extreme = MacchiavelianConfig::extreme_machiavellian();
        let aggressive = MacchiavelianConfig::aggressive_defaults();
        
        // Verify extreme is more aggressive than aggressive
        assert!(extreme.antifragility_threshold < aggressive.antifragility_threshold);
        assert!(extreme.kelly_fraction > aggressive.kelly_fraction);
        assert!(extreme.whale_detected_multiplier > aggressive.whale_detected_multiplier);
    }
}
