//! # Neural Trader Integration Layer
//!
//! This crate provides the main integration layer that unifies all 17 neural-trader crates
//! into a cohesive, production-ready trading system.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     Neural Trader                            │
//! │                   (Main Facade)                             │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         │                    │                    │
//!    ┌────▼────┐        ┌─────▼─────┐       ┌─────▼─────┐
//!    │ Trading │        │ Analytics │       │   Risk    │
//!    │ Service │        │  Service  │       │  Service  │
//!    └────┬────┘        └─────┬─────┘       └─────┬─────┘
//!         │                   │                    │
//!    ┌────▼───────────────────▼────────────────────▼─────┐
//!    │            Coordination Layer                      │
//!    │  (BrokerPool, StrategyManager, ModelRegistry)     │
//!    └────┬───────────────────┬────────────────────┬─────┘
//!         │                   │                    │
//!    ┌────▼────┐        ┌─────▼─────┐       ┌─────▼─────┐
//!    │ Brokers │        │Strategies │       │  Neural   │
//!    │ (11)    │        │   (7+)    │       │ Models(3) │
//!    └─────────┘        └───────────┘       └───────────┘
//! ```
//!
//! ## Components
//!
//! - **Services**: High-level business logic (trading, analytics, risk, neural)
//! - **Coordination**: Resource management (brokers, strategies, models)
//! - **API**: External interfaces (REST, WebSocket, CLI)
//! - **Config**: Unified configuration system
//! - **Runtime**: Async runtime management
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use neural_trader_integration::{NeuralTrader, config::Config};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load configuration
//!     let _config = Config::from_file("config.toml")?;
//!
//!     // Initialize the system
//!     let trader = NeuralTrader::new(config).await?;
//!
//!     // Start trading
//!     trader.start_trading().await?;
//!
//!     // Execute a strategy
//!     trader.execute_strategy("momentum").await?;
//!
//!     // Get portfolio status
//!     let portfolio = trader.get_portfolio().await?;
//!     println!("Portfolio value: ${}", portfolio.total_value);
//!
//!     Ok(())
//! }
//! ```

use std::sync::Arc;
use tokio::runtime::Runtime as TokioRuntime;
use tracing::{info, warn, error};

pub mod config;
pub mod runtime;
pub mod services;
pub mod coordination;
pub mod api;
pub mod error;
pub mod types;

pub use config::Config;
pub use config::Config as IntegrationConfig;
pub use error::{Error, Result};
pub use types::*;

use coordination::{BrokerPool, StrategyManager, ModelRegistry, MemoryCoordinator};
use services::{TradingService, AnalyticsService, RiskService, NeuralService};

/// Main facade for the Neural Trader system.
///
/// This is the primary entry point for interacting with the trading system.
/// It provides a high-level API that coordinates all subsystems.
pub struct NeuralTrader {
    /// System configuration
    config: Arc<IntegrationConfig>,

    /// Async runtime handle
    runtime: Arc<runtime::Runtime>,

    /// Broker connection pool (11 brokers)
    broker_pool: Arc<BrokerPool>,

    /// Strategy orchestration (7+ strategies)
    strategy_manager: Arc<StrategyManager>,

    /// Neural model lifecycle management (3+ models)
    model_registry: Arc<ModelRegistry>,

    /// Cross-service memory coordination
    memory: Arc<MemoryCoordinator>,

    /// High-level services
    trading_service: Arc<TradingService>,
    analytics_service: Arc<AnalyticsService>,
    risk_service: Arc<RiskService>,
    neural_service: Arc<NeuralService>,
}

impl NeuralTrader {
    /// Creates a new Neural Trader instance with the given configuration.
    ///
    /// This initializes all subsystems:
    /// - Broker connections
    /// - Strategy engines
    /// - Neural models
    /// - Memory systems
    /// - Services
    ///
    /// # Errors
    ///
    /// Returns an error if any subsystem fails to initialize.
    pub async fn new(integration_config: IntegrationConfig) -> Result<Self> {
        info!("Initializing Neural Trader system");

        let config = Arc::new(integration_config);

        // Initialize runtime
        let runtime = Arc::new(runtime::Runtime::new(&config)?);
        info!("Runtime initialized");

        // Initialize coordination layer
        let broker_pool = Arc::new(BrokerPool::new(&config).await?);
        info!("Broker pool initialized with {} brokers", broker_pool.size());

        let strategy_manager = Arc::new(StrategyManager::new(&config).await?);
        info!("Strategy manager initialized with {} strategies", strategy_manager.count());

        let model_registry = Arc::new(ModelRegistry::new(&config).await?);
        info!("Model registry initialized with {} models", model_registry.count());

        let memory = Arc::new(MemoryCoordinator::new(&config).await?);
        info!("Memory coordinator initialized");

        // Initialize services
        let risk_service = Arc::new(RiskService::new(
            config.clone(),
            broker_pool.clone(),
            memory.clone(),
        ).await?);

        let neural_service = Arc::new(NeuralService::new(
            config.clone(),
            model_registry.clone(),
            memory.clone(),
        ).await?);

        let trading_service = Arc::new(TradingService::new(
            config.clone(),
            broker_pool.clone(),
            strategy_manager.clone(),
            risk_service.clone(),
            neural_service.clone(),
            memory.clone(),
        ).await?);

        let analytics_service = Arc::new(AnalyticsService::new(
            config.clone(),
            broker_pool.clone(),
            memory.clone(),
        ).await?);

        info!("All services initialized successfully");

        Ok(Self {
            config,
            runtime,
            broker_pool,
            strategy_manager,
            model_registry,
            memory,
            trading_service,
            analytics_service,
            risk_service,
            neural_service,
        })
    }

    /// Starts the trading system.
    ///
    /// This begins active trading based on the configured strategies.
    pub async fn start_trading(&self) -> Result<()> {
        info!("Starting trading system");
        self.trading_service.start().await
    }

    /// Stops the trading system gracefully.
    pub async fn stop_trading(&self) -> Result<()> {
        info!("Stopping trading system");
        self.trading_service.stop().await
    }

    /// Executes a specific strategy by name.
    pub async fn execute_strategy(&self, name: &str) -> Result<ExecutionResult> {
        info!("Executing strategy: {}", name);
        self.trading_service.execute_strategy(name).await
    }

    /// Gets the current portfolio state.
    pub async fn get_portfolio(&self) -> Result<Portfolio> {
        self.trading_service.get_portfolio().await
    }

    /// Performs a comprehensive risk analysis.
    pub async fn analyze_risk(&self) -> Result<RiskReport> {
        self.risk_service.analyze().await
    }

    /// Trains a neural model with the specified configuration.
    pub async fn train_model(&self, config: ModelTrainingConfig) -> Result<String> {
        self.neural_service.train(config).await
    }

    /// Generates a performance report for the specified time period.
    pub async fn generate_report(&self, period: TimePeriod) -> Result<PerformanceReport> {
        self.analytics_service.generate_report(period).await
    }

    /// Returns the system health status.
    pub async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            broker_pool: self.broker_pool.health().await?,
            strategy_manager: self.strategy_manager.health().await?,
            model_registry: self.model_registry.health().await?,
            trading_service: self.trading_service.health().await?,
            risk_service: self.risk_service.health().await?,
            neural_service: self.neural_service.health().await?,
            analytics_service: self.analytics_service.health().await?,
        })
    }

    /// Gracefully shuts down the system.
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Neural Trader system");

        // Stop trading first
        if let Err(e) = self.stop_trading().await {
            warn!("Error stopping trading: {}", e);
        }

        // Shutdown services in reverse order of initialization
        self.analytics_service.shutdown().await?;
        self.trading_service.shutdown().await?;
        self.neural_service.shutdown().await?;
        self.risk_service.shutdown().await?;

        // Shutdown coordination layer
        self.memory.shutdown().await?;
        self.model_registry.shutdown().await?;
        self.strategy_manager.shutdown().await?;
        self.broker_pool.shutdown().await?;

        info!("Neural Trader shutdown complete");
        Ok(())
    }
}

/// Builder for constructing a NeuralTrader instance with custom configuration.
pub struct NeuralTraderBuilder {
    config: IntegrationConfig,
}

impl NeuralTraderBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
        }
    }

    /// Sets the configuration from a file.
    pub fn with_config_file(mut self, path: &str) -> Result<Self> {
        self.config = IntegrationConfig::from_file(path)?;
        Ok(self)
    }

    /// Sets the configuration directly.
    pub fn with_config(mut self, integration_config: IntegrationConfig) -> Self {
        self.config = integration_config;
        self
    }

    /// Builds the NeuralTrader instance.
    pub async fn build(self) -> Result<NeuralTrader> {
        NeuralTrader::new(self.config).await
    }
}

impl Default for NeuralTraderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_builder_pattern() {
        let integration_config = IntegrationConfig::default();
        let builder = NeuralTraderBuilder::new().with_config(integration_config);

        // Note: This will fail without proper configuration
        // but demonstrates the API usage
        assert!(builder.build().await.is_err());
    }
}
