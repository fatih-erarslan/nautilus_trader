//! Sentinel framework for fast-path risk monitoring.
//!
//! Sentinels are passive monitors that run in the fast path (<100μs)
//! to detect risk violations and trigger protective actions.
//!
//! ## Sentinel Taxonomy
//!
//! | Sentinel | Latency | Function |
//! |----------|---------|----------|
//! | GlobalKillSwitch | <1μs | Atomic halt all trading |
//! | PositionLimitSentinel | <5μs | Per-asset position limits |
//! | DrawdownSentinel | <5μs | Max drawdown enforcement |
//! | CircuitBreakerSentinel | <10μs | Volatility/loss triggers |
//! | VaRSentinel | <20μs | Real-time VaR monitoring |
//! | WhaleSentinel | <15μs | Large flow detection |

pub mod base;
pub mod kill_switch;
pub mod position_limit;
pub mod drawdown;
pub mod circuit_breaker;
pub mod var_sentinel;
pub mod whale;

pub use base::{Sentinel, SentinelId, SentinelStatus, SentinelConfig};
pub use kill_switch::GlobalKillSwitch;
pub use position_limit::PositionLimitSentinel;
pub use drawdown::DrawdownSentinel;
pub use circuit_breaker::CircuitBreakerSentinel;
pub use var_sentinel::VaRSentinel;
pub use whale::WhaleSentinel;
