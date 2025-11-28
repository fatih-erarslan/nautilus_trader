//! Base traits and types for agents.
//!
//! Agents are active processors in the medium path that perform
//! sophisticated analysis beyond simple threshold checks.

use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::core::types::{MarketRegime, Portfolio, RiskDecision};
use crate::core::error::Result;

/// Unique identifier for an agent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    /// Create new agent ID.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get ID as string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Agent operational status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle, waiting for work.
    Idle,
    /// Agent is actively processing.
    Processing,
    /// Agent is paused.
    Paused,
    /// Agent encountered an error.
    Error,
    /// Agent is shutting down.
    ShuttingDown,
}

/// Base configuration for agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name.
    pub name: String,
    /// Whether agent is enabled.
    pub enabled: bool,
    /// Processing priority (lower = higher priority).
    pub priority: u8,
    /// Maximum processing time in microseconds.
    pub max_latency_us: u64,
    /// Enable detailed logging.
    pub verbose: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            enabled: true,
            priority: 100,
            max_latency_us: 1000, // 1ms default
            verbose: false,
        }
    }
}

/// Core trait for all agents.
///
/// Agents operate in the medium path and can perform:
/// - Complex mathematical calculations
/// - Statistical model inference
/// - Multi-asset analysis
/// - Regime detection
pub trait Agent: Send + Sync + Debug {
    /// Get agent identifier.
    fn id(&self) -> AgentId;

    /// Get current status.
    fn status(&self) -> AgentStatus;

    /// Process a tick/update cycle.
    ///
    /// # Arguments
    /// * `portfolio` - Current portfolio state
    /// * `regime` - Current market regime
    ///
    /// # Returns
    /// Optional risk decision if agent needs to modify behavior
    fn process(&self, portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>>;

    /// Start the agent.
    fn start(&self) -> Result<()>;

    /// Stop the agent.
    fn stop(&self) -> Result<()>;

    /// Pause the agent.
    fn pause(&self);

    /// Resume the agent.
    fn resume(&self);

    /// Get processing count.
    fn process_count(&self) -> u64;

    /// Get average processing latency in nanoseconds.
    fn avg_latency_ns(&self) -> u64;
}

/// Statistics tracking for agents.
#[derive(Debug, Default)]
pub struct AgentStats {
    /// Total processing cycles.
    pub cycles: AtomicU64,
    /// Total latency (for average calculation).
    pub total_latency_ns: AtomicU64,
    /// Maximum latency observed.
    pub max_latency_ns: AtomicU64,
    /// Error count.
    pub errors: AtomicU64,
}

impl AgentStats {
    /// Create new stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a processing cycle.
    #[inline]
    pub fn record_cycle(&self, latency_ns: u64) {
        self.cycles.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);

        // Update max
        let prev_max = self.max_latency_ns.load(Ordering::Relaxed);
        if latency_ns > prev_max {
            self.max_latency_ns.store(latency_ns, Ordering::Relaxed);
        }
    }

    /// Record an error.
    #[inline]
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency.
    pub fn avg_latency_ns(&self) -> u64 {
        let cycles = self.cycles.load(Ordering::Relaxed);
        if cycles == 0 {
            return 0;
        }
        self.total_latency_ns.load(Ordering::Relaxed) / cycles
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.cycles.store(0, Ordering::Relaxed);
        self.total_latency_ns.store(0, Ordering::Relaxed);
        self.max_latency_ns.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id() {
        let id = AgentId::new("test_agent");
        assert_eq!(id.as_str(), "test_agent");
    }

    #[test]
    fn test_agent_stats() {
        let stats = AgentStats::new();

        stats.record_cycle(1000);
        stats.record_cycle(2000);
        stats.record_cycle(1500);

        assert_eq!(stats.cycles.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_latency_ns(), 1500);
        assert_eq!(stats.max_latency_ns.load(Ordering::Relaxed), 2000);
    }
}
