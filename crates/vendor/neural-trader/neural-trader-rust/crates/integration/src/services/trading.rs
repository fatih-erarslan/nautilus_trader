//! Trading service - coordinates strategies, brokers, and risk.

use crate::{
    Config, Error, Result,
    coordination::{BrokerPool, StrategyManager, MemoryCoordinator},
    services::{RiskService, NeuralService},
    types::{ExecutionResult, Portfolio, ComponentHealth, HealthStatusEnum},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use chrono::Utc;

/// High-level trading service that orchestrates the trading system.
pub struct TradingService {
    config: Arc<Config>,
    broker_pool: Arc<BrokerPool>,
    strategy_manager: Arc<StrategyManager>,
    risk_service: Arc<RiskService>,
    neural_service: Arc<NeuralService>,
    memory: Arc<MemoryCoordinator>,
    running: Arc<RwLock<bool>>,
}

impl TradingService {
    /// Creates a new trading service.
    pub async fn new(
        config: Arc<Config>,
        broker_pool: Arc<BrokerPool>,
        strategy_manager: Arc<StrategyManager>,
        risk_service: Arc<RiskService>,
        neural_service: Arc<NeuralService>,
        memory: Arc<MemoryCoordinator>,
    ) -> Result<Self> {
        info!("Initializing trading service");

        Ok(Self {
            config,
            broker_pool,
            strategy_manager,
            risk_service,
            neural_service,
            memory,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Starts the trading system.
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;

        if *running {
            return Err(Error::invalid_state("Trading system is already running"));
        }

        info!("Starting trading system");
        *running = true;

        // TODO: Start trading loop
        info!("Trading system started");

        Ok(())
    }

    /// Stops the trading system.
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;

        if !*running {
            return Ok(());
        }

        info!("Stopping trading system");
        *running = false;

        // TODO: Stop trading loop gracefully
        info!("Trading system stopped");

        Ok(())
    }

    /// Executes a specific strategy.
    pub async fn execute_strategy(&self, name: &str) -> Result<ExecutionResult> {
        info!("Executing strategy: {}", name);

        let strategy = self.strategy_manager.get(name)
            .ok_or_else(|| Error::not_found(format!("Strategy not found: {}", name)))?;

        // TODO: Implement actual strategy execution

        Ok(ExecutionResult {
            strategy_name: name.to_string(),
            timestamp: Utc::now(),
            orders: vec![],
            total_value: rust_decimal::Decimal::ZERO,
            profit_loss: rust_decimal::Decimal::ZERO,
            metadata: serde_json::json!({}),
        })
    }

    /// Gets the current portfolio state.
    pub async fn get_portfolio(&self) -> Result<Portfolio> {
        // TODO: Implement actual portfolio retrieval
        Ok(Portfolio {
            total_value: rust_decimal::Decimal::ZERO,
            cash: rust_decimal::Decimal::ZERO,
            positions: vec![],
            updated_at: Utc::now(),
        })
    }

    /// Health check for the trading service.
    pub async fn health(&self) -> Result<ComponentHealth> {
        let running = *self.running.read().await;

        Ok(ComponentHealth {
            status: if running { HealthStatusEnum::Healthy } else { HealthStatusEnum::Unhealthy },
            message: Some(if running { "Trading active".to_string() } else { "Trading stopped".to_string() }),
            last_check: Utc::now(),
            uptime: std::time::Duration::from_secs(0),
        })
    }

    /// Gracefully shuts down the trading service.
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down trading service");
        self.stop().await?;
        info!("Trading service shutdown complete");
        Ok(())
    }
}
