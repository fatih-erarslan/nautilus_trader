//! Main HyperRiskEngine implementation.
//!
//! Orchestrates sentinels, agents, and risk calculations across
//! the three-tier latency architecture.

use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::core::error::{Result, RiskError};
use crate::core::ring_buffer::{RingBuffer, RingBufferConfig, RiskEvent, RiskEventType, EventPayload};
use crate::core::types::{
    MarketRegime, Order, Portfolio, RiskDecision, RiskLevel, Timestamp,
};
use crate::sentinels::{Sentinel, SentinelId, SentinelStatus};
use crate::agents::{Agent, AgentId, AgentStatus};
use crate::{FAST_PATH_LATENCY_NS, PRE_TRADE_CHECK_LATENCY_NS};

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum drawdown before halt (e.g., 0.15 = 15%).
    pub max_drawdown: f64,
    /// VaR limit as fraction of portfolio.
    pub var_limit_pct: f64,
    /// VaR confidence level.
    pub var_confidence: f64,
    /// Enable fast-path latency enforcement.
    pub enforce_latency: bool,
    /// Ring buffer size.
    pub ring_buffer_size: usize,
    /// Maximum concurrent sentinels.
    pub max_sentinels: usize,
    /// Maximum concurrent agents.
    pub max_agents: usize,
    /// Enable detailed metrics.
    pub enable_metrics: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.15,
            var_limit_pct: 0.02,
            var_confidence: 0.95,
            enforce_latency: true,
            ring_buffer_size: 65536,
            max_sentinels: 16,
            max_agents: 32,
            enable_metrics: true,
        }
    }
}

impl EngineConfig {
    /// Production configuration with strict limits.
    pub fn production() -> Self {
        Self {
            max_drawdown: 0.10,
            var_limit_pct: 0.01,
            var_confidence: 0.99,
            enforce_latency: true,
            ring_buffer_size: 131072,
            max_sentinels: 32,
            max_agents: 64,
            enable_metrics: true,
        }
    }

    /// Development configuration with relaxed limits.
    pub fn development() -> Self {
        Self {
            max_drawdown: 0.25,
            var_limit_pct: 0.05,
            var_confidence: 0.95,
            enforce_latency: false,
            ring_buffer_size: 4096,
            max_sentinels: 8,
            max_agents: 16,
            enable_metrics: true,
        }
    }
}

/// Engine operational state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    /// Engine is initializing.
    Initializing,
    /// Engine is running normally.
    Running,
    /// Engine is in restricted mode (elevated risk).
    Restricted,
    /// Engine is paused (no new orders).
    Paused,
    /// Engine is in emergency halt.
    Halted,
    /// Engine is shutting down.
    ShuttingDown,
}

/// Main HyperRiskEngine implementation.
pub struct HyperRiskEngine {
    /// Engine configuration.
    config: EngineConfig,
    /// Current engine state.
    state: Arc<RwLock<EngineState>>,
    /// Current market regime.
    regime: Arc<RwLock<MarketRegime>>,
    /// Portfolio state.
    portfolio: Arc<RwLock<Portfolio>>,
    /// Registered sentinels.
    sentinels: DashMap<SentinelId, Arc<dyn Sentinel>>,
    /// Registered agents.
    agents: DashMap<AgentId, Arc<dyn Agent>>,
    /// Event ring buffer.
    event_buffer: Arc<RwLock<RingBuffer>>,
    /// Engine metrics.
    metrics: Arc<EngineMetrics>,
}

/// Engine performance metrics.
#[derive(Debug, Default)]
pub struct EngineMetrics {
    /// Total pre-trade checks performed.
    pub total_checks: std::sync::atomic::AtomicU64,
    /// Checks that exceeded latency budget.
    pub slow_checks: std::sync::atomic::AtomicU64,
    /// Total rejections.
    pub rejections: std::sync::atomic::AtomicU64,
    /// Kill switch activations.
    pub kill_switch_activations: std::sync::atomic::AtomicU64,
    /// Average latency (exponential moving average).
    pub avg_latency_ns: std::sync::atomic::AtomicU64,
    /// Maximum latency observed.
    pub max_latency_ns: std::sync::atomic::AtomicU64,
}

impl HyperRiskEngine {
    /// Create new engine with configuration.
    pub fn new(config: EngineConfig) -> Result<Self> {
        let ring_config = RingBufferConfig {
            size: config.ring_buffer_size,
            enable_stats: config.enable_metrics,
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(EngineState::Initializing)),
            regime: Arc::new(RwLock::new(MarketRegime::Unknown)),
            portfolio: Arc::new(RwLock::new(Portfolio::new(0.0))),
            sentinels: DashMap::new(),
            agents: DashMap::new(),
            event_buffer: Arc::new(RwLock::new(RingBuffer::new(ring_config))),
            metrics: Arc::new(EngineMetrics::default()),
        })
    }

    /// Initialize engine with portfolio.
    pub fn initialize(&self, initial_portfolio: Portfolio) -> Result<()> {
        let mut state = self.state.write();
        if *state != EngineState::Initializing {
            return Err(RiskError::ConfigurationError(
                "Engine already initialized".to_string(),
            ));
        }

        // Set portfolio
        {
            let mut portfolio = self.portfolio.write();
            *portfolio = initial_portfolio;
        }

        // Transition to running
        *state = EngineState::Running;
        Ok(())
    }

    /// Register a sentinel for fast-path monitoring.
    pub fn register_sentinel(&self, sentinel: Arc<dyn Sentinel>) -> Result<SentinelId> {
        if self.sentinels.len() >= self.config.max_sentinels {
            return Err(RiskError::ConfigurationError(format!(
                "Maximum sentinels ({}) reached",
                self.config.max_sentinels
            )));
        }

        let id = sentinel.id();
        self.sentinels.insert(id.clone(), sentinel);
        Ok(id)
    }

    /// Register an agent for medium-path processing.
    pub fn register_agent(&self, agent: Arc<dyn Agent>) -> Result<AgentId> {
        if self.agents.len() >= self.config.max_agents {
            return Err(RiskError::ConfigurationError(format!(
                "Maximum agents ({}) reached",
                self.config.max_agents
            )));
        }

        let id = agent.id();
        self.agents.insert(id.clone(), agent);
        Ok(id)
    }

    /// Fast-path pre-trade risk check.
    ///
    /// Target latency: <100μs (typically <20μs).
    ///
    /// # Performance
    ///
    /// This function is optimized for minimal latency:
    /// - No heap allocation
    /// - Lock-free sentinel checks where possible
    /// - Early exit on first rejection
    /// - Inline quantile functions
    #[inline]
    pub fn pre_trade_check(&self, order: &Order) -> Result<RiskDecision> {
        let start = Instant::now();

        // Check engine state first
        let state = *self.state.read();
        match state {
            EngineState::Halted => {
                return Err(RiskError::KillSwitchActivated {
                    reason: "Engine halted",
                });
            }
            EngineState::Paused => {
                return Ok(RiskDecision::reject(
                    "Engine paused",
                    RiskLevel::High,
                    start.elapsed().as_nanos() as u64,
                ));
            }
            EngineState::ShuttingDown => {
                return Ok(RiskDecision::reject(
                    "Engine shutting down",
                    RiskLevel::Emergency,
                    start.elapsed().as_nanos() as u64,
                ));
            }
            _ => {}
        }

        // Run all sentinels (fast path)
        let portfolio = self.portfolio.read();

        for entry in self.sentinels.iter() {
            let sentinel = entry.value();

            if sentinel.status() != SentinelStatus::Active {
                continue;
            }

            // Each sentinel check should be <10μs
            match sentinel.check(order, &portfolio) {
                Ok(()) => continue,
                Err(e) => {
                    let latency_ns = start.elapsed().as_nanos() as u64;
                    self.record_check_latency(latency_ns);
                    self.metrics
                        .rejections
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    // Emit event
                    self.emit_risk_event(RiskEventType::RiskCheck, EventPayload::RiskCheck {
                        allowed: false,
                        risk_level: e.severity(),
                        latency_ns,
                    });

                    return Err(e);
                }
            }
        }

        // All checks passed
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.record_check_latency(latency_ns);

        // Emit success event
        self.emit_risk_event(RiskEventType::RiskCheck, EventPayload::RiskCheck {
            allowed: true,
            risk_level: 0,
            latency_ns,
        });

        Ok(RiskDecision::approve(latency_ns))
    }

    /// Record check latency and update metrics.
    #[inline]
    fn record_check_latency(&self, latency_ns: u64) {
        use std::sync::atomic::Ordering;

        self.metrics.total_checks.fetch_add(1, Ordering::Relaxed);

        // Update max latency
        let prev_max = self.metrics.max_latency_ns.load(Ordering::Relaxed);
        if latency_ns > prev_max {
            self.metrics.max_latency_ns.store(latency_ns, Ordering::Relaxed);
        }

        // Check if slow
        if self.config.enforce_latency && latency_ns > PRE_TRADE_CHECK_LATENCY_NS {
            self.metrics.slow_checks.fetch_add(1, Ordering::Relaxed);
        }

        // Update EMA (alpha = 0.1)
        let prev_avg = self.metrics.avg_latency_ns.load(Ordering::Relaxed);
        let new_avg = ((prev_avg as f64) * 0.9 + (latency_ns as f64) * 0.1) as u64;
        self.metrics.avg_latency_ns.store(new_avg, Ordering::Relaxed);
    }

    /// Emit event to ring buffer.
    fn emit_risk_event(&self, event_type: RiskEventType, payload: EventPayload) {
        let event = RiskEvent {
            sequence: 0, // Will be set by buffer
            timestamp: Timestamp::now(),
            event_type,
            payload,
        };

        let buffer = self.event_buffer.read();
        let _ = buffer.publish(event);
    }

    /// Activate global kill switch.
    pub fn activate_kill_switch(&self, reason: &'static str) -> Result<()> {
        let mut state = self.state.write();
        *state = EngineState::Halted;

        self.metrics
            .kill_switch_activations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        self.emit_risk_event(
            RiskEventType::SentinelAction,
            EventPayload::Alert {
                severity: 4, // Emergency
                code: 1,     // Kill switch
            },
        );

        Err(RiskError::KillSwitchActivated { reason })
    }

    /// Reset kill switch and resume trading.
    pub fn reset_kill_switch(&self) -> Result<()> {
        let mut state = self.state.write();
        if *state != EngineState::Halted {
            return Err(RiskError::ConfigurationError(
                "Engine not in halted state".to_string(),
            ));
        }
        *state = EngineState::Running;
        Ok(())
    }

    /// Get current engine state.
    pub fn state(&self) -> EngineState {
        *self.state.read()
    }

    /// Get current market regime.
    pub fn regime(&self) -> MarketRegime {
        *self.regime.read()
    }

    /// Update market regime.
    pub fn update_regime(&self, regime: MarketRegime) {
        let mut current = self.regime.write();
        *current = regime;
    }

    /// Get portfolio reference.
    pub fn portfolio(&self) -> parking_lot::RwLockReadGuard<Portfolio> {
        self.portfolio.read()
    }

    /// Get mutable portfolio reference.
    pub fn portfolio_mut(&self) -> parking_lot::RwLockWriteGuard<Portfolio> {
        self.portfolio.write()
    }

    /// Get engine metrics.
    pub fn metrics(&self) -> &EngineMetrics {
        &self.metrics
    }

    /// Get configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Check if engine is operational.
    pub fn is_operational(&self) -> bool {
        matches!(
            self.state(),
            EngineState::Running | EngineState::Restricted
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = HyperRiskEngine::new(config).unwrap();
        assert_eq!(engine.state(), EngineState::Initializing);
    }

    #[test]
    fn test_engine_initialization() {
        let config = EngineConfig::default();
        let engine = HyperRiskEngine::new(config).unwrap();

        let portfolio = Portfolio::new(100_000.0);
        engine.initialize(portfolio).unwrap();

        assert_eq!(engine.state(), EngineState::Running);
    }

    #[test]
    fn test_kill_switch() {
        let config = EngineConfig::default();
        let engine = HyperRiskEngine::new(config).unwrap();
        engine.initialize(Portfolio::new(100_000.0)).unwrap();

        let result = engine.activate_kill_switch("Test halt");
        assert!(result.is_err());
        assert_eq!(engine.state(), EngineState::Halted);

        engine.reset_kill_switch().unwrap();
        assert_eq!(engine.state(), EngineState::Running);
    }

    #[test]
    fn test_production_config() {
        let config = EngineConfig::production();
        assert_eq!(config.max_drawdown, 0.10);
        assert_eq!(config.var_confidence, 0.99);
        assert!(config.enforce_latency);
    }
}
