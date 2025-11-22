//! Market regime detection for trading strategies

use serde::{Deserialize, Serialize};

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    // Volatility-based regimes
    LowVolatility,
    HighVolatility,
    VolatilitySpike,
    VolatilityCompression,
    
    // Trend-based regimes
    StrongUptrend,
    StrongDowntrend,
    Consolidation,
    Sideways,
    
    // Liquidity-based regimes
    HighLiquidity,
    LowLiquidity,
    LiquidityDrain,
    LiquidityFlood,
    
    // Crisis regimes
    FlashCrash,
    MarketMelt,
    CircuitBreaker,
    TradingHalt,
    
    // Game theory regimes
    CooperativeMarket,
    AdversarialMarket,
    PrisonersDilemma,
    ChickenGame,
    
    // Quantum-detected regimes
    QuantumCoherent,
    QuantumEntangled,
    QuantumSuperposition,
    QuantumDecoherence,
    
    // Unknown/transitional
    RegimeTransition,
    Unknown,
    Neutral,
}

impl MarketRegime {
    /// Get the typical duration of this regime
    pub fn typical_duration(&self) -> chrono::Duration {
        match self {
            MarketRegime::FlashCrash => chrono::Duration::seconds(30),
            MarketRegime::VolatilitySpike => chrono::Duration::minutes(15),
            MarketRegime::TradingHalt => chrono::Duration::hours(1),
            MarketRegime::LowVolatility => chrono::Duration::days(7),
            MarketRegime::HighVolatility => chrono::Duration::days(3),
            MarketRegime::StrongUptrend => chrono::Duration::days(30),
            MarketRegime::StrongDowntrend => chrono::Duration::days(21),
            MarketRegime::Consolidation => chrono::Duration::days(14),
            MarketRegime::QuantumCoherent => chrono::Duration::minutes(5),
            MarketRegime::RegimeTransition => chrono::Duration::minutes(10),
            _ => chrono::Duration::hours(4),
        }
    }
    
    /// Get the volatility level associated with this regime
    pub fn volatility_level(&self) -> f64 {
        match self {
            MarketRegime::LowVolatility => 0.1,
            MarketRegime::HighVolatility => 0.3,
            MarketRegime::VolatilitySpike => 0.8,
            MarketRegime::FlashCrash => 1.5,
            MarketRegime::MarketMelt => 2.0,
            _ => 0.2,
        }
    }
    
    /// Check if this regime is considered dangerous
    pub fn is_dangerous(&self) -> bool {
        matches!(self, 
            MarketRegime::FlashCrash | 
            MarketRegime::MarketMelt | 
            MarketRegime::VolatilitySpike |
            MarketRegime::LiquidityDrain |
            MarketRegime::CircuitBreaker |
            MarketRegime::TradingHalt
        )
    }
    
    /// Check if this regime supports quantum strategies
    pub fn supports_quantum_strategies(&self) -> bool {
        matches!(self,
            MarketRegime::QuantumCoherent |
            MarketRegime::QuantumEntangled |
            MarketRegime::QuantumSuperposition |
            MarketRegime::HighVolatility |
            MarketRegime::AdversarialMarket
        )
    }
}