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
//! | GreeksSentinel | <25μs | Portfolio Greeks monitoring |
//! | CounterpartySentinel | <20μs | Counterparty credit risk (Basel III) |
//! | TradeSurveillanceSentinel | <50μs | Market manipulation detection |
//! | ChiefRiskOfficerSentinel | <50μs | Firm-wide risk orchestration |
//! | RegulatoryComplianceSentinel | <100μs | MiFID II, Dodd-Frank, CFTC compliance |
//! | DataFeedSentinel | <20μs | Market data quality monitoring |
//! | SystemHealthSentinel | <20μs | Infrastructure health monitoring |
//! | StrategyKillSwitchSentinel | <20μs | Strategy-level emergency halt |

pub mod base;
pub mod kill_switch;
pub mod position_limit;
pub mod drawdown;
pub mod circuit_breaker;
pub mod var_sentinel;
pub mod whale;
pub mod greeks;
pub mod counterparty;
pub mod surveillance;
pub mod cro;
pub mod stress_test;
pub mod compliance;
pub mod data_feed;
pub mod system_health;
pub mod strategy_kill_switch;

pub use base::{Sentinel, SentinelId, SentinelStatus, SentinelConfig, SentinelStats};
pub use kill_switch::GlobalKillSwitch;
pub use position_limit::PositionLimitSentinel;
pub use drawdown::DrawdownSentinel;
pub use circuit_breaker::CircuitBreakerSentinel;
pub use var_sentinel::VaRSentinel;
pub use whale::WhaleSentinel;
pub use greeks::GreeksSentinel;
pub use counterparty::{
    CounterpartySentinel, CounterpartyConfig, CounterpartyExposure,
    NettingSet, Trade, AssetClass, AddOnFactors, MaturityBuckets,
    ExposureAlert, AlertSeverity, ExposureType,
};
pub use surveillance::{
    TradeSurveillanceSentinel, SurveillanceConfig, ManipulationType,
    SurveillanceAlert, OrderFlowStats,
};
pub use cro::{
    ChiefRiskOfficerSentinel, CROConfig, AggregateRiskMetrics, LiquidityCrisis,
    CounterpartyReport, VetoDecision, HaltReason, PositionReductionMandate,
};
pub use stress_test::{
    StressTestSentinel, StressConfig, Scenario, StressResult, Factor,
};
pub use compliance::{
    RegulatoryComplianceSentinel, ComplianceConfig, ComplianceCheckType,
    ComplianceViolation, ViolationSeverity, RegulatoryReport, ReportType,
    ShortSaleStatus, PositionLimitConfig,
};
pub use data_feed::{DataFeedSentinel, DataFeedConfig, FeedStatus};
pub use system_health::{SystemHealthSentinel, SystemHealthConfig, HealthMetrics, HealthLevel};
pub use strategy_kill_switch::{StrategyKillSwitchSentinel, StrategyKillSwitchConfig, KillReason, KillEvent, StrategyPerformance};
