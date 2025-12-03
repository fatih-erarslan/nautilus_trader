//! Tiny Dancer Neural Routing
//!
//! Integrates ruvector-tiny-dancer-core for intelligent strategy routing.
//!
//! ## Features
//!
//! - **FastGRNN Routing**: Sub-millisecond neural routing decisions
//! - **Circuit Breakers**: Graceful degradation under high uncertainty
//! - **Strategy Optimization**: Route trading decisions to optimal strategies
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Kusupati et al. (2018) "FastGRNN: A fast, accurate, stable RNN" NeurIPS

use ruvector_tiny_dancer_core::{
    Router, RouterConfig,
    Candidate, RoutingRequest, RoutingResponse,
    circuit_breaker::{CircuitBreaker, CircuitState},
};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{AgentDBError, Result};
use crate::trading::{MarketRegime, MarketContext, TradeAction};

/// Trading strategy router using Tiny Dancer
pub struct TradingStrategyRouter {
    /// Core router
    router: Arc<Router>,
    /// Circuit breaker for each strategy
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    /// Registered strategies (candidates)
    strategies: Arc<RwLock<Vec<StrategyCandidate>>>,
    /// Configuration
    config: TradingRouterConfig,
    /// Statistics
    stats: Arc<RwLock<RouterStats>>,
}

/// Configuration for trading router
#[derive(Debug, Clone)]
pub struct TradingRouterConfig {
    /// Model path for FastGRNN
    pub model_path: String,
    /// Confidence threshold for routing
    pub confidence_threshold: f32,
    /// Maximum uncertainty allowed
    pub max_uncertainty: f32,
    /// Maximum consecutive failures before circuit opens
    pub max_failures: u32,
    /// Reset timeout in seconds
    pub reset_timeout_secs: u64,
    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,
}

impl Default for TradingRouterConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/trading_router.safetensors".to_string(),
            confidence_threshold: 0.85,
            max_uncertainty: 0.15,
            max_failures: 5,
            reset_timeout_secs: 60,
            enable_circuit_breaker: true,
        }
    }
}

/// A trading strategy as a routing candidate
#[derive(Debug, Clone)]
pub struct StrategyCandidate {
    /// Strategy identifier
    pub id: String,
    /// Strategy name
    pub name: String,
    /// Supported market regimes
    pub supported_regimes: Vec<MarketRegime>,
    /// Supported trade actions
    pub supported_actions: Vec<TradeAction>,
    /// Historical win rate (success_rate)
    pub win_rate: f32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Cost per trade (compute/fees)
    pub cost_per_trade: f64,
    /// Strategy embedding
    pub embedding: Vec<f32>,
    /// Access count
    pub access_count: u64,
}

impl StrategyCandidate {
    /// Convert to Tiny Dancer candidate
    fn to_candidate(&self) -> Candidate {
        Candidate {
            id: self.id.clone(),
            embedding: self.embedding.clone(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now().timestamp(),
            access_count: self.access_count,
            success_rate: self.win_rate,
        }
    }
}

/// Router statistics
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    /// Total routing decisions
    pub total_decisions: u64,
    /// Average routing latency in microseconds
    pub avg_latency_us: f64,
    /// Strategy selection counts
    pub strategy_selections: HashMap<String, u64>,
    /// Circuit breaker trips
    pub circuit_trips: u64,
    /// Fallback uses
    pub fallback_uses: u64,
    /// Average confidence
    pub avg_confidence: f64,
}

impl TradingStrategyRouter {
    /// Create new trading strategy router
    pub fn new(config: TradingRouterConfig) -> Result<Self> {
        // Create router configuration
        let router_config = RouterConfig {
            model_path: config.model_path.clone(),
            confidence_threshold: config.confidence_threshold,
            max_uncertainty: config.max_uncertainty,
            enable_circuit_breaker: config.enable_circuit_breaker,
            circuit_breaker_threshold: config.max_failures,
            enable_quantization: true,
            database_path: None,
        };

        let router = Router::new(router_config)
            .map_err(|e| AgentDBError::RoutingError(e.to_string()))?;

        Ok(Self {
            router: Arc::new(router),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            strategies: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: Arc::new(RwLock::new(RouterStats::default())),
        })
    }

    /// Register a trading strategy
    pub fn register_strategy(&self, strategy: StrategyCandidate) -> Result<()> {
        // Create circuit breaker for this strategy
        let circuit_breaker = CircuitBreaker::with_timeout(
            self.config.max_failures,
            Duration::from_secs(self.config.reset_timeout_secs),
        );

        self.circuit_breakers.write().insert(strategy.id.clone(), circuit_breaker);
        self.strategies.write().push(strategy);

        Ok(())
    }

    /// Route a trading decision to best strategy
    pub fn route(
        &self,
        context: &MarketContext,
        action: TradeAction,
        query_embedding: &[f32],
    ) -> Result<RoutingResult> {
        let start = std::time::Instant::now();

        // Get available strategies (not circuit-broken)
        let available = self.get_available_strategies(context.regime, action);
        if available.is_empty() {
            return Err(AgentDBError::RoutingError("No available strategies".into()));
        }

        // Convert to candidates
        let candidates: Vec<Candidate> = available.iter()
            .map(|s| s.to_candidate())
            .collect();

        // Create routing request
        let request = RoutingRequest {
            query_embedding: query_embedding.to_vec(),
            candidates: candidates.clone(),
            metadata: None,
        };

        // Route through FastGRNN
        let response = self.router.route(request)
            .map_err(|e| AgentDBError::RoutingError(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as f64;

        // Get best decision
        let best_decision = response.decisions.first()
            .ok_or_else(|| AgentDBError::RoutingError("No routing decisions returned".into()))?;

        // Check if we need fallback due to high uncertainty
        let mut used_fallback = false;
        let selected_strategy = if best_decision.uncertainty > self.config.max_uncertainty {
            // High uncertainty - use fallback strategy
            used_fallback = true;
            self.get_fallback_strategy(context.regime, action)
                .ok_or_else(|| AgentDBError::RoutingError("No fallback available".into()))?
        } else {
            // Use routed strategy
            available.iter()
                .find(|s| s.id == best_decision.candidate_id)
                .cloned()
                .ok_or_else(|| AgentDBError::RoutingError("Selected strategy not found".into()))?
        };

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_decisions += 1;
            stats.avg_latency_us = 0.99 * stats.avg_latency_us + 0.01 * latency_us;
            stats.avg_confidence = 0.99 * stats.avg_confidence + 0.01 * best_decision.confidence as f64;
            *stats.strategy_selections.entry(selected_strategy.id.clone()).or_default() += 1;
            if used_fallback {
                stats.fallback_uses += 1;
            }
        }

        // Collect all scores
        let all_scores: Vec<(String, f64)> = response.decisions.iter()
            .map(|d| (d.candidate_id.clone(), d.confidence as f64))
            .collect();

        Ok(RoutingResult {
            strategy: selected_strategy,
            confidence: best_decision.confidence as f64,
            uncertainty: best_decision.uncertainty as f64,
            latency_us,
            used_fallback,
            all_scores,
        })
    }

    /// Get available strategies (not circuit-broken, supports regime/action)
    fn get_available_strategies(
        &self,
        regime: MarketRegime,
        action: TradeAction,
    ) -> Vec<StrategyCandidate> {
        let strategies = self.strategies.read();
        let circuit_breakers = self.circuit_breakers.read();

        strategies.iter()
            .filter(|s| {
                // Check regime support
                let regime_ok = s.supported_regimes.contains(&regime);
                // Check action support
                let action_ok = s.supported_actions.contains(&action);
                // Check circuit breaker
                let cb_ok = circuit_breakers.get(&s.id)
                    .map(|cb| cb.state() != CircuitState::Open)
                    .unwrap_or(true);

                regime_ok && action_ok && cb_ok
            })
            .cloned()
            .collect()
    }

    /// Get fallback strategy for regime/action
    fn get_fallback_strategy(
        &self,
        regime: MarketRegime,
        action: TradeAction,
    ) -> Option<StrategyCandidate> {
        let strategies = self.strategies.read();

        // Find strategy with highest win rate that supports this regime/action
        strategies.iter()
            .filter(|s| {
                s.supported_regimes.contains(&regime) &&
                s.supported_actions.contains(&action)
            })
            .max_by(|a, b| {
                a.win_rate.partial_cmp(&b.win_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Record trade outcome for circuit breaker
    pub fn record_outcome(
        &self,
        strategy_id: &str,
        success: bool,
        _pnl: f64,
    ) -> Result<()> {
        let mut circuit_breakers = self.circuit_breakers.write();

        if let Some(cb) = circuit_breakers.get_mut(strategy_id) {
            if success {
                cb.record_success();
            } else {
                cb.record_failure();

                // Check if circuit opened
                if cb.state() == CircuitState::Open {
                    self.stats.write().circuit_trips += 1;
                }
            }
        }

        Ok(())
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStats {
        self.stats.read().clone()
    }

    /// Get circuit breaker states
    pub fn circuit_states(&self) -> HashMap<String, CircuitState> {
        self.circuit_breakers.read()
            .iter()
            .map(|(id, cb)| (id.clone(), cb.state()))
            .collect()
    }
}

/// Result of routing decision
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Selected strategy
    pub strategy: StrategyCandidate,
    /// Routing confidence (0-1)
    pub confidence: f64,
    /// Uncertainty estimate (0-1)
    pub uncertainty: f64,
    /// Routing latency in microseconds
    pub latency_us: f64,
    /// Whether fallback was used
    pub used_fallback: bool,
    /// Scores for all candidates
    pub all_scores: Vec<(String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_strategy(id: &str, name: &str, win_rate: f32) -> StrategyCandidate {
        StrategyCandidate {
            id: id.to_string(),
            name: name.to_string(),
            supported_regimes: vec![MarketRegime::BullTrend, MarketRegime::BearTrend, MarketRegime::RangeBound],
            supported_actions: vec![TradeAction::Long, TradeAction::Short, TradeAction::Close],
            win_rate,
            avg_latency_ms: 1.0,
            cost_per_trade: 0.001,
            embedding: vec![0.1f32; 384],
            access_count: 10,
        }
    }

    #[test]
    fn test_strategy_to_candidate() {
        let strategy = create_test_strategy("test", "Test Strategy", 0.75);
        let candidate = strategy.to_candidate();

        assert_eq!(candidate.id, "test");
        assert_eq!(candidate.embedding.len(), 384);
        assert_eq!(candidate.success_rate, 0.75);
        assert_eq!(candidate.access_count, 10);
    }

    #[test]
    fn test_router_config_default() {
        let config = TradingRouterConfig::default();

        assert_eq!(config.confidence_threshold, 0.85);
        assert_eq!(config.max_uncertainty, 0.15);
        assert_eq!(config.max_failures, 5);
        assert!(config.enable_circuit_breaker);
    }

    #[test]
    fn test_router_stats_default() {
        let stats = RouterStats::default();

        assert_eq!(stats.total_decisions, 0);
        assert_eq!(stats.avg_latency_us, 0.0);
        assert_eq!(stats.circuit_trips, 0);
    }
}
