//! Asset allocation agent for portfolio optimization.
//!
//! Operates in the medium path (<1ms) to compute optimal asset allocations
//! based on risk-return objectives and market conditions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the asset allocation agent.
#[derive(Debug, Clone)]
pub struct AssetAllocationConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Target volatility for the portfolio.
    pub target_volatility: f64,
    /// Maximum weight for any single asset.
    pub max_single_weight: f64,
    /// Minimum weight for any single asset.
    pub min_single_weight: f64,
    /// Rebalancing threshold (deviation from target).
    pub rebalance_threshold: f64,
    /// Risk-free rate for Sharpe calculations.
    pub risk_free_rate: f64,
}

impl Default for AssetAllocationConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "asset_allocation_agent".to_string(),
                enabled: true,
                priority: 2,
                max_latency_us: 1000, // 1ms
                verbose: false,
            },
            target_volatility: 0.15,
            max_single_weight: 0.25,
            min_single_weight: 0.0,
            rebalance_threshold: 0.05,
            risk_free_rate: 0.05,
        }
    }
}

/// Target allocation for an asset.
#[derive(Debug, Clone)]
pub struct AssetTarget {
    /// Symbol of the asset.
    pub symbol: Symbol,
    /// Target weight (0.0 to 1.0).
    pub target_weight: f64,
    /// Current weight.
    pub current_weight: f64,
    /// Deviation from target.
    pub deviation: f64,
    /// Recommended action.
    pub action: AllocationAction,
}

/// Recommended allocation action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAction {
    /// Increase position.
    Increase,
    /// Decrease position.
    Decrease,
    /// Hold current position.
    Hold,
    /// Close position entirely.
    Close,
}

/// Asset allocation recommendation.
#[derive(Debug, Clone)]
pub struct AllocationRecommendation {
    /// Individual asset targets.
    pub targets: Vec<AssetTarget>,
    /// Portfolio-level metrics.
    pub expected_return: f64,
    /// Expected portfolio volatility.
    pub expected_volatility: f64,
    /// Expected Sharpe ratio.
    pub expected_sharpe: f64,
    /// Recommendation timestamp.
    pub generated_at: Timestamp,
    /// Whether rebalancing is recommended.
    pub needs_rebalance: bool,
}

/// Asset return and risk statistics.
#[derive(Debug, Clone)]
pub struct AssetStats {
    /// Expected return.
    pub expected_return: f64,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Correlation with other assets (simplified).
    pub beta: f64,
}

/// Asset allocation agent.
#[derive(Debug)]
pub struct AssetAllocationAgent {
    config: AssetAllocationConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Asset statistics.
    asset_stats: RwLock<HashMap<Symbol, AssetStats>>,
    /// Current target allocation.
    current_targets: RwLock<HashMap<Symbol, f64>>,
    /// Latest recommendation.
    latest_recommendation: RwLock<Option<AllocationRecommendation>>,
}

impl AssetAllocationAgent {
    /// Create a new asset allocation agent.
    pub fn new(config: AssetAllocationConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            asset_stats: RwLock::new(HashMap::new()),
            current_targets: RwLock::new(HashMap::new()),
            latest_recommendation: RwLock::new(None),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AssetAllocationConfig::default())
    }

    /// Update asset statistics.
    pub fn update_asset_stats(&self, symbol: Symbol, stats: AssetStats) {
        self.asset_stats.write().insert(symbol, stats);
    }

    /// Set target allocation for an asset.
    pub fn set_target(&self, symbol: Symbol, weight: f64) {
        let clamped = weight.clamp(self.config.min_single_weight, self.config.max_single_weight);
        self.current_targets.write().insert(symbol, clamped);
    }

    /// Get the latest allocation recommendation.
    pub fn get_recommendation(&self) -> Option<AllocationRecommendation> {
        self.latest_recommendation.read().clone()
    }

    /// Compute optimal allocation based on current portfolio and targets.
    fn compute_allocation(&self, portfolio: &Portfolio) -> AllocationRecommendation {
        let targets = self.current_targets.read();
        let asset_stats = self.asset_stats.read();

        // Calculate current weights
        let total_value = portfolio.total_value;
        let mut current_weights: HashMap<Symbol, f64> = HashMap::new();

        if total_value > 0.0 {
            for position in &portfolio.positions {
                let position_value = position.market_value();
                current_weights.insert(position.symbol.clone(), position_value / total_value);
            }
        }

        // Build allocation targets
        let mut allocation_targets = Vec::new();
        let mut needs_rebalance = false;

        for (symbol, &target_weight) in targets.iter() {
            let current_weight = current_weights.get(symbol).copied().unwrap_or(0.0);
            let deviation = current_weight - target_weight;

            let action = if deviation.abs() > self.config.rebalance_threshold {
                needs_rebalance = true;
                if deviation > 0.0 {
                    AllocationAction::Decrease
                } else {
                    AllocationAction::Increase
                }
            } else if target_weight == 0.0 && current_weight > 0.0 {
                needs_rebalance = true;
                AllocationAction::Close
            } else {
                AllocationAction::Hold
            };

            allocation_targets.push(AssetTarget {
                symbol: symbol.clone(),
                target_weight,
                current_weight,
                deviation,
                action,
            });
        }

        // Calculate expected portfolio metrics
        let (expected_return, expected_volatility) = self.calculate_portfolio_metrics(&targets, &asset_stats);
        let expected_sharpe = if expected_volatility > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_volatility
        } else {
            0.0
        };

        AllocationRecommendation {
            targets: allocation_targets,
            expected_return,
            expected_volatility,
            expected_sharpe,
            generated_at: Timestamp::now(),
            needs_rebalance,
        }
    }

    /// Calculate expected portfolio return and volatility.
    fn calculate_portfolio_metrics(
        &self,
        weights: &HashMap<Symbol, f64>,
        asset_stats: &HashMap<Symbol, AssetStats>,
    ) -> (f64, f64) {
        let mut expected_return = 0.0;
        let mut variance = 0.0;

        for (symbol, &weight) in weights.iter() {
            if let Some(stats) = asset_stats.get(symbol) {
                expected_return += weight * stats.expected_return;
                // Simplified variance calculation (ignoring correlations)
                variance += (weight * stats.volatility).powi(2);
            }
        }

        (expected_return, variance.sqrt())
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for AssetAllocationAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Compute allocation recommendation
        let recommendation = self.compute_allocation(portfolio);
        *self.latest_recommendation.write() = Some(recommendation);

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_allocation_agent_creation() {
        let agent = AssetAllocationAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_target_setting() {
        let agent = AssetAllocationAgent::with_defaults();

        let symbol = Symbol::new("AAPL");
        agent.set_target(symbol.clone(), 0.20);

        let targets = agent.current_targets.read();
        assert_eq!(targets.get(&symbol), Some(&0.20));
    }

    #[test]
    fn test_weight_clamping() {
        let agent = AssetAllocationAgent::with_defaults();

        let symbol = Symbol::new("AAPL");
        // Try to set weight above max (0.25)
        agent.set_target(symbol.clone(), 0.50);

        let targets = agent.current_targets.read();
        assert_eq!(targets.get(&symbol), Some(&0.25));
    }

    #[test]
    fn test_allocation_computation() {
        let agent = AssetAllocationAgent::with_defaults();
        agent.start().unwrap();

        let symbol = Symbol::new("AAPL");
        agent.set_target(symbol.clone(), 0.20);
        agent.update_asset_stats(symbol, AssetStats {
            expected_return: 0.10,
            volatility: 0.20,
            beta: 1.0,
        });

        let portfolio = Portfolio::default();
        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let recommendation = agent.get_recommendation();
        assert!(recommendation.is_some());
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = AssetAllocationAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }
}
