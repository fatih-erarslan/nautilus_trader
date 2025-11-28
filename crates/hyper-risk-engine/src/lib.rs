//! # HyperRiskEngine - Enterprise Quantitative Trading Risk Management
//!
//! State-of-the-art risk engine implementing a three-tier latency architecture
//! for professional trading firms, following techniques from Citadel, Jane Street,
//! Jump Trading, and academic research.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         HYPER-RISK-ENGINE                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  FAST PATH (<100μs)         │  MEDIUM PATH (100μs-1ms)                 │
//! │  ────────────────────       │  ────────────────────────                │
//! │  • Kill Switches            │  • Regime Detection (HMM)                │
//! │  • Position Limits          │  • VaR/CVaR Calculation                  │
//! │  • Circuit Breakers         │  • Kelly Position Sizing                 │
//! │  • Pre-trade Risk           │  • Correlation Updates (DCC)             │
//! │  • Anomaly Detection        │  • Alpha Generation                      │
//! │                             │                                          │
//! ├─────────────────────────────┼──────────────────────────────────────────┤
//! │  SLOW PATH (>1ms)           │  EVOLUTION LAYER (seconds-hours)         │
//! │  ────────────────────       │  ────────────────────────────            │
//! │  • Monte Carlo VaR          │  • Parameter Optimization                │
//! │  • Stress Testing           │  • Model Retraining                      │
//! │  • FRTB ES Calculation      │  • Regime Model Updates                  │
//! │  • Portfolio Optimization   │  • Neural Pattern Learning               │
//! └─────────────────────────────┴──────────────────────────────────────────┘
//! ```
//!
//! ## Latency Budget (Sub-100μs Target)
//!
//! | Component | Budget | Implementation |
//! |-----------|--------|----------------|
//! | Data Ingestion | 5μs | Lock-free ring buffer |
//! | Feature Computation | 10μs | SIMD vectorized |
//! | Model Inference | 30μs | Pre-fitted parameters |
//! | Risk Calculation | 20μs | Inline quantile functions |
//! | Anomaly Check | 15μs | Simple autoencoder |
//! | Decision Logic | 10μs | Lookup tables |
//! | **Total** | **90μs** | |
//!
//! ## Agent/Sentinel Taxonomy
//!
//! ### Sentinels (Passive Monitors - Fast Path)
//! - **GlobalKillSwitch**: Atomic halt across all strategies
//! - **PositionLimitSentinel**: Per-asset and portfolio limits
//! - **DrawdownSentinel**: Max drawdown enforcement
//! - **CircuitBreakerSentinel**: Volatility/loss triggered halts
//! - **VaRSentinel**: Real-time VaR monitoring
//! - **WhaleSentinel**: Large flow detection
//!
//! ### Agents (Active Processors - Medium Path)
//! - **PortfolioManagerAgent**: Position orchestration
//! - **AlphaGeneratorAgent**: Signal generation
//! - **RegimeDetectionAgent**: HMM/MS-GARCH regime
//! - **ExecutionAgent**: Order management
//! - **ResearcherAgent**: Strategy analysis
//!
//! ## Scientific References
//!
//! - McNeil & Frey (2000): "Conditional EVT" for GARCH + EVT
//! - Siffer et al. (2017): "SPOT/DSPOT" streaming anomaly detection
//! - Romano et al. (2019): "Conformalized Quantile Regression"
//! - Buehler et al. (2019): "Deep Hedging" neural approach
//! - Schulman et al. (2017): "PPO" for position sizing
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use hyper_risk_engine::{HyperRiskEngine, EngineConfig};
//! use hyper_risk_engine::sentinels::{GlobalKillSwitch, DrawdownSentinel};
//!
//! // Initialize engine with sub-100μs target
//! let config = EngineConfig::production();
//! let mut engine = HyperRiskEngine::new(config)?;
//!
//! // Register sentinels
//! engine.register_sentinel(GlobalKillSwitch::new());
//! engine.register_sentinel(DrawdownSentinel::new(0.15)); // 15% max drawdown
//!
//! // Fast-path pre-trade check (<100μs)
//! let decision = engine.pre_trade_check(&order)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod sentinels;
pub mod agents;
pub mod evt;
pub mod position_sizing;
pub mod fast_path;
pub mod medium_path;
pub mod slow_path;
pub mod simd;

// Re-export primary types
pub use crate::core::{
    EngineConfig, HyperRiskEngine, RiskDecision, RiskLevel,
    Timestamp, Price, Quantity, Symbol, PositionId,
    Order, OrderSide, Portfolio, Position, MarketRegime,
};

pub use crate::core::error::{RiskError, Result};

pub use crate::sentinels::{
    Sentinel, SentinelStatus, SentinelId,
    GlobalKillSwitch, PositionLimitSentinel, DrawdownSentinel,
    CircuitBreakerSentinel, VaRSentinel, WhaleSentinel,
    TradeSurveillanceSentinel, SurveillanceConfig, ManipulationType,
    SurveillanceAlert, OrderFlowStats, StressTestSentinel, StressConfig,
    Scenario, StressResult, Factor,
};

pub use crate::agents::{
    Agent, AgentId, AgentStatus,
    PortfolioManagerAgent, AlphaGeneratorAgent, RegimeDetectionAgent,
    MarketMakerAgent, MarketMakerConfig, Quote, InventoryState, ToxicityScore,
};

pub use crate::evt::{
    StreamingEVT, SpotConfig, DspotConfig, GPDParams,
    TailRiskEstimate, ExceedanceEvent,
};

pub use crate::position_sizing::{
    KellyCriterion, FractionalKelly, RiskTolerance,
    PPOPositionSizer, PositionSizeResult,
};

// ============================================================================
// Architecture Constants (from state-of-the-art research)
// ============================================================================

/// Fast path latency budget in nanoseconds (100μs).
pub const FAST_PATH_LATENCY_NS: u64 = 100_000;

/// Medium path latency budget in nanoseconds (1ms).
pub const MEDIUM_PATH_LATENCY_NS: u64 = 1_000_000;

/// Target pre-trade risk check latency (20μs).
pub const PRE_TRADE_CHECK_LATENCY_NS: u64 = 20_000;

/// Ring buffer size for lock-free event processing.
/// Based on LMAX Disruptor pattern for 6M events/sec.
pub const RING_BUFFER_SIZE: usize = 65536; // 2^16, cache-aligned

/// Cache line size for memory alignment (64 bytes on most architectures).
pub const CACHE_LINE_SIZE: usize = 64;

/// Default fractional Kelly (0.5x achieves 75% growth with 50% drawdown).
pub const DEFAULT_FRACTIONAL_KELLY: f64 = 0.5;

/// Basel III FRTB confidence level for Expected Shortfall.
pub const FRTB_ES_CONFIDENCE: f64 = 0.975;

/// Default VaR confidence level.
pub const DEFAULT_VAR_CONFIDENCE: f64 = 0.95;

/// Maximum Monte Carlo paths for real-time VaR.
pub const MAX_REALTIME_MC_PATHS: usize = 10_000;

// ============================================================================
// Performance Validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_hierarchy() {
        assert!(FAST_PATH_LATENCY_NS < MEDIUM_PATH_LATENCY_NS);
        assert!(PRE_TRADE_CHECK_LATENCY_NS < FAST_PATH_LATENCY_NS);
    }

    #[test]
    fn test_ring_buffer_power_of_two() {
        assert!(RING_BUFFER_SIZE.is_power_of_two());
    }

    #[test]
    fn test_cache_alignment() {
        assert_eq!(CACHE_LINE_SIZE, 64);
    }

    #[test]
    fn test_kelly_bounds() {
        assert!(DEFAULT_FRACTIONAL_KELLY > 0.0);
        assert!(DEFAULT_FRACTIONAL_KELLY <= 1.0);
    }

    #[test]
    fn test_confidence_levels() {
        assert!(DEFAULT_VAR_CONFIDENCE > 0.0 && DEFAULT_VAR_CONFIDENCE < 1.0);
        assert!(FRTB_ES_CONFIDENCE > DEFAULT_VAR_CONFIDENCE);
    }
}
