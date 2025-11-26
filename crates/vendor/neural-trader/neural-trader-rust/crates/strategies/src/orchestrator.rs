//! Strategy Orchestrator
//!
//! Manages multiple strategies with:
//! - Market regime detection
//! - Adaptive strategy selection
//! - Portfolio allocation across strategies
//! - Performance monitoring and switching

use crate::{
    Result, Strategy, Signal, StrategyError, MarketData, Portfolio,
    integration::{NeuralPredictor, MarketRegime, RiskManager, StrategyExecutor, ValidationResult},
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::DateTime;

/// Strategy orchestrator
pub struct StrategyOrchestrator {
    /// Available strategies
    strategies: HashMap<String, Arc<dyn Strategy>>,
    /// Active strategy IDs
    active_strategies: Vec<String>,
    /// Neural predictor for regime detection
    neural: Arc<NeuralPredictor>,
    /// Risk manager
    risk_manager: Arc<RwLock<RiskManager>>,
    /// Strategy executor
    executor: Arc<StrategyExecutor>,
    /// Strategy performance tracking
    performance: Arc<RwLock<HashMap<String, StrategyPerformance>>>,
    /// Current market regime
    current_regime: Arc<RwLock<Option<MarketRegime>>>,
    /// Allocation mode
    allocation_mode: AllocationMode,
}

impl StrategyOrchestrator {
    /// Create new orchestrator
    pub fn new(
        neural: Arc<NeuralPredictor>,
        risk_manager: Arc<RwLock<RiskManager>>,
        executor: Arc<StrategyExecutor>,
    ) -> Self {
        Self {
            strategies: HashMap::new(),
            active_strategies: Vec::new(),
            neural,
            risk_manager,
            executor,
            performance: Arc::new(RwLock::new(HashMap::new())),
            current_regime: Arc::new(RwLock::new(None)),
            allocation_mode: AllocationMode::Adaptive,
        }
    }

    /// Register a strategy
    pub fn register_strategy(&mut self, strategy: Arc<dyn Strategy>) {
        let id = strategy.id().to_string();
        info!("Registering strategy: {}", id);

        self.strategies.insert(id.clone(), strategy);

        // Initialize performance tracking
        tokio::spawn({
            let performance = Arc::clone(&self.performance);
            let id = id.clone();
            async move {
                let mut perf = performance.write().await;
                perf.insert(id, StrategyPerformance::default());
            }
        });
    }

    /// Set active strategies
    pub fn set_active_strategies(&mut self, strategy_ids: Vec<String>) {
        info!("Setting active strategies: {:?}", strategy_ids);
        self.active_strategies = strategy_ids;
    }

    /// Set allocation mode
    pub fn set_allocation_mode(&mut self, mode: AllocationMode) {
        self.allocation_mode = mode;
    }

    /// Process market data through all active strategies
    pub async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        // Detect market regime
        let features = self.extract_features(&market_data.bars);
        let regime = self.neural.detect_regime(&market_data.symbol, &features).await?;

        {
            let mut current_regime = self.current_regime.write().await;
            *current_regime = Some(regime);
        }

        debug!("Current market regime: {:?}", regime);

        // Select appropriate strategies based on regime
        let selected_strategies = self.select_strategies_for_regime(regime).await;

        if selected_strategies.is_empty() {
            debug!("No strategies selected for current regime");
            return Ok(Vec::new());
        }

        // Generate signals from selected strategies
        let mut all_signals = Vec::new();

        for strategy_id in selected_strategies {
            if let Some(strategy) = self.strategies.get(&strategy_id) {
                match strategy.process(market_data, portfolio).await {
                    Ok(mut signals) => {
                        // Validate and size signals through risk manager
                        let validated_signals = self.validate_signals(&mut signals, portfolio).await?;
                        all_signals.extend(validated_signals);

                        // Update performance tracking
                        self.update_performance(&strategy_id, &signals).await;
                    }
                    Err(e) => {
                        warn!("Strategy {} error: {}", strategy_id, e);
                        continue;
                    }
                }
            }
        }

        // Aggregate and prioritize signals
        let final_signals = self.aggregate_signals(all_signals).await;

        Ok(final_signals)
    }

    /// Select strategies based on market regime
    async fn select_strategies_for_regime(&self, regime: MarketRegime) -> Vec<String> {
        match self.allocation_mode {
            AllocationMode::Static => {
                // Use all active strategies
                self.active_strategies.clone()
            }

            AllocationMode::RegimeBased => {
                // Select strategies suited to current regime
                self.active_strategies
                    .iter()
                    .filter(|id| self.is_strategy_suitable_for_regime(id, regime))
                    .cloned()
                    .collect()
            }

            AllocationMode::Adaptive => {
                // Select top performers with regime consideration
                let performance = self.performance.read().await;

                let mut ranked: Vec<_> = self
                    .active_strategies
                    .iter()
                    .filter(|id| self.is_strategy_suitable_for_regime(id, regime))
                    .map(|id| {
                        let perf = performance.get(id).cloned().unwrap_or_default();
                        (id.clone(), perf.sharpe_ratio)
                    })
                    .collect();

                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Take top 3 performers
                ranked.into_iter().take(3).map(|(id, _)| id).collect()
            }

            AllocationMode::Ensemble => {
                // Use all strategies and aggregate their signals
                self.active_strategies.clone()
            }
        }
    }

    /// Check if strategy is suitable for regime
    fn is_strategy_suitable_for_regime(&self, strategy_id: &str, regime: MarketRegime) -> bool {
        match (strategy_id, regime) {
            ("momentum_trader", MarketRegime::Trending) => true,
            ("mean_reversion", MarketRegime::Ranging) => true,
            ("neural_trend", MarketRegime::Trending) => true,
            ("neural_sentiment", MarketRegime::VolatileBullish | MarketRegime::VolatileBearish) => true,
            ("pairs_trading", MarketRegime::Ranging | MarketRegime::LowVolatility) => true,
            _ => true, // Allow all strategies by default
        }
    }

    /// Validate signals through risk manager
    async fn validate_signals(
        &self,
        signals: &mut [Signal],
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        let mut validated = Vec::new();
        let risk_manager = self.risk_manager.read().await;

        // Build current positions map
        let positions = portfolio
            .positions()
            .iter()
            .map(|(symbol, pos)| {
                (
                    symbol.clone(),
                    crate::integration::risk::Position {
                        symbol: symbol.clone(),
                        quantity: pos.quantity,
                        market_value: pos.market_value,
                        unrealized_pnl: Decimal::ZERO,
                    },
                )
            })
            .collect();

        for signal in signals.iter_mut() {
            match risk_manager.validate_signal(signal, portfolio.total_value(), &positions)? {
                ValidationResult::Approved => {
                    validated.push(signal.clone());
                }
                ValidationResult::Rejected(reason) => {
                    debug!("Signal rejected: {}", reason);
                }
            }
        }

        Ok(validated)
    }

    /// Aggregate signals from multiple strategies
    async fn aggregate_signals(&self, signals: Vec<Signal>) -> Vec<Signal> {
        if signals.is_empty() {
            return signals;
        }

        match self.allocation_mode {
            AllocationMode::Static | AllocationMode::RegimeBased | AllocationMode::Adaptive => {
                // Take highest confidence signal per symbol
                let mut best_signals: HashMap<String, Signal> = HashMap::new();

                for signal in signals {
                    let confidence = signal.confidence.unwrap_or(0.0);
                    best_signals
                        .entry(signal.symbol.clone())
                        .and_modify(|existing| {
                            if confidence > existing.confidence.unwrap_or(0.0) {
                                *existing = signal.clone();
                            }
                        })
                        .or_insert(signal);
                }

                best_signals.into_values().collect()
            }

            AllocationMode::Ensemble => {
                // Aggregate signals by averaging confidence and combining signals
                let mut grouped: HashMap<String, Vec<Signal>> = HashMap::new();

                for signal in signals {
                    grouped.entry(signal.symbol.clone()).or_default().push(signal);
                }

                grouped
                    .into_iter()
                    .map(|(symbol, signals)| {
                        let avg_confidence = signals
                            .iter()
                            .map(|s| s.confidence.unwrap_or(0.0))
                            .sum::<f64>()
                            / signals.len() as f64;

                        // Use most common direction
                        let mut long_count = 0;
                        let mut short_count = 0;
                        let mut close_count = 0;

                        for s in &signals {
                            match s.direction {
                                crate::Direction::Long => long_count += 1,
                                crate::Direction::Short => short_count += 1,
                                crate::Direction::Close => close_count += 1,
                            }
                        }

                        let direction = if long_count > short_count && long_count > close_count {
                            crate::Direction::Long
                        } else if short_count > long_count && short_count > close_count {
                            crate::Direction::Short
                        } else {
                            crate::Direction::Close
                        };

                        Signal::new("ensemble".to_string(), symbol, direction)
                            .with_confidence(avg_confidence)
                    })
                    .collect()
            }
        }
    }

    /// Extract features from bars
    fn extract_features(&self, bars: &[crate::Bar]) -> Vec<f64> {
        bars.iter()
            .map(|b| b.close.to_f64().unwrap())
            .collect()
    }

    /// Update strategy performance
    async fn update_performance(&self, strategy_id: &str, signals: &[Signal]) {
        let mut performance = self.performance.write().await;

        if let Some(perf) = performance.get_mut(strategy_id) {
            perf.signals_generated += signals.len();
            perf.last_signal_time = chrono::Utc::now();
        }
    }

    /// Get current regime
    pub async fn current_regime(&self) -> Option<MarketRegime> {
        *self.current_regime.read().await
    }

    /// Get strategy performance
    pub async fn get_performance(&self, strategy_id: &str) -> Option<StrategyPerformance> {
        let performance = self.performance.read().await;
        performance.get(strategy_id).cloned()
    }

    /// Get all performance metrics
    pub async fn all_performance(&self) -> HashMap<String, StrategyPerformance> {
        self.performance.read().await.clone()
    }
}

/// Allocation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    /// Use all active strategies equally
    Static,
    /// Select strategies based on market regime
    RegimeBased,
    /// Adapt based on recent performance
    Adaptive,
    /// Combine all strategy signals
    Ensemble,
}

/// Strategy performance tracking
#[derive(Debug, Clone, Default)]
pub struct StrategyPerformance {
    pub signals_generated: usize,
    pub trades_executed: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: Decimal,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub last_signal_time: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::MomentumStrategy;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let neural = Arc::new(NeuralPredictor::default());
        let risk_manager = Arc::new(RwLock::new(RiskManager::default()));
        let executor = Arc::new(StrategyExecutor::new(
            Arc::new(crate::integration::broker::MockBroker { should_fail: false }),
        ));

        let mut orchestrator = StrategyOrchestrator::new(neural, risk_manager, executor);

        let strategy = Arc::new(MomentumStrategy::new(vec!["AAPL".to_string()], 20, 2.0, 0.5));
        orchestrator.register_strategy(strategy);

        assert_eq!(orchestrator.strategies.len(), 1);
    }
}
