//! HyperPhysics Strategy Implementation.
//!
//! This strategy implements a Nautilus-compatible trading strategy that
//! uses HyperPhysics physics-based modeling, biomimetic optimization,
//! and Byzantine consensus for signal generation.

use crate::adapter::{NautilusDataAdapter, NautilusExecBridge};
use crate::config::IntegrationConfig;
use crate::error::{IntegrationError, Result};
use crate::types::{
    HyperPhysicsOrderCommand, MarketSnapshot, NautilusBar, NautilusQuoteTick,
    NautilusTradeTick, NautilusOrderBookDelta,
};
use hyperphysics_hft_ecosystem::core::unified_pipeline::{
    MarketFeed, PipelineResult, PipelineStats, UnifiedPipeline,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Strategy state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyState {
    /// Strategy initialized but not started
    Initialized,
    /// Strategy is running
    Running,
    /// Strategy is stopped
    Stopped,
    /// Strategy encountered an error
    Faulted,
}

/// Strategy performance metrics
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    /// Total quotes processed
    pub quotes_processed: u64,
    /// Total trades processed
    pub trades_processed: u64,
    /// Total bars processed
    pub bars_processed: u64,
    /// Total signals generated
    pub signals_generated: u64,
    /// Total orders submitted
    pub orders_submitted: u64,
    /// Average signal latency (microseconds)
    pub avg_signal_latency_us: f64,
    /// Maximum signal latency (microseconds)
    pub max_signal_latency_us: u64,
    /// Total runtime (seconds)
    pub runtime_seconds: f64,
}

/// HyperPhysics Strategy for Nautilus Trader.
///
/// This strategy acts as a bridge between Nautilus Trader's event-driven
/// architecture and HyperPhysics's physics-based trading pipeline.
///
/// # Architecture
///
/// ```text
/// [Nautilus DataEngine] → [on_quote/on_trade/on_bar]
///                               ↓
/// [NautilusDataAdapter] → [MarketFeed]
///                               ↓
/// [UnifiedPipeline] → [Physics → Optimization → Consensus]
///                               ↓
/// [NautilusExecBridge] → [Order Commands]
///                               ↓
/// [Nautilus ExecutionEngine] → [Venue]
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let config = IntegrationConfig::default();
/// let strategy = HyperPhysicsStrategy::new(config).await?;
///
/// // Set trading instrument
/// strategy.set_instrument("BTCUSDT.BINANCE").await;
///
/// // Start strategy
/// strategy.start().await?;
///
/// // Process data events
/// strategy.on_quote(&quote_tick).await?;
/// ```
pub struct HyperPhysicsStrategy {
    /// Strategy ID
    pub strategy_id: String,

    /// Configuration
    config: IntegrationConfig,

    /// Current state
    state: Arc<RwLock<StrategyState>>,

    /// Data adapter
    data_adapter: NautilusDataAdapter,

    /// Execution bridge
    exec_bridge: NautilusExecBridge,

    /// HyperPhysics unified pipeline
    pipeline: Arc<UnifiedPipeline>,

    /// Performance metrics
    metrics: Arc<RwLock<StrategyMetrics>>,

    /// Start time
    start_time: Arc<RwLock<Option<Instant>>>,

    /// Current instrument
    current_instrument: Arc<RwLock<Option<String>>>,

    /// Order callback (for integration with Nautilus MessageBus)
    order_callback: Arc<RwLock<Option<Box<dyn Fn(HyperPhysicsOrderCommand) + Send + Sync>>>>,
}

impl HyperPhysicsStrategy {
    /// Create a new HyperPhysics strategy
    pub async fn new(config: IntegrationConfig) -> Result<Self> {
        let strategy_id = config.strategy_id.clone();

        // Create pipeline with ecosystem config
        let ecosystem_config = hyperphysics_hft_ecosystem::core::EcosystemConfig {
            target_latency_us: config.target_latency_us,
            ..Default::default()
        };

        let pipeline = UnifiedPipeline::new(ecosystem_config)?;

        info!(
            strategy_id = %strategy_id,
            confidence_threshold = config.min_confidence_threshold,
            target_latency_us = config.target_latency_us,
            "Created HyperPhysics strategy"
        );

        Ok(Self {
            strategy_id,
            config: config.clone(),
            state: Arc::new(RwLock::new(StrategyState::Initialized)),
            data_adapter: NautilusDataAdapter::new(config.clone()),
            exec_bridge: NautilusExecBridge::new(config),
            pipeline: Arc::new(pipeline),
            metrics: Arc::new(RwLock::new(StrategyMetrics::default())),
            start_time: Arc::new(RwLock::new(None)),
            current_instrument: Arc::new(RwLock::new(None)),
            order_callback: Arc::new(RwLock::new(None)),
        })
    }

    /// Set the order callback for integration with Nautilus
    pub async fn set_order_callback<F>(&self, callback: F)
    where
        F: Fn(HyperPhysicsOrderCommand) + Send + Sync + 'static,
    {
        let mut cb = self.order_callback.write().await;
        *cb = Some(Box::new(callback));
    }

    /// Set the trading instrument
    pub async fn set_instrument(&self, instrument_id: &str) {
        let mut current = self.current_instrument.write().await;
        *current = Some(instrument_id.to_string());
        self.exec_bridge.set_instrument(instrument_id).await;

        info!(
            strategy_id = %self.strategy_id,
            instrument = instrument_id,
            "Set trading instrument"
        );
    }

    /// Start the strategy
    pub async fn start(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state == StrategyState::Running {
            return Ok(());
        }

        *state = StrategyState::Running;

        let mut start_time = self.start_time.write().await;
        *start_time = Some(Instant::now());

        info!(strategy_id = %self.strategy_id, "Strategy started");
        Ok(())
    }

    /// Stop the strategy
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = StrategyState::Stopped;

        // Update runtime
        let start_time = self.start_time.read().await;
        if let Some(start) = *start_time {
            let mut metrics = self.metrics.write().await;
            metrics.runtime_seconds = start.elapsed().as_secs_f64();
        }

        info!(strategy_id = %self.strategy_id, "Strategy stopped");
        Ok(())
    }

    /// Check if strategy is running
    pub async fn is_running(&self) -> bool {
        let state = self.state.read().await;
        *state == StrategyState::Running
    }

    /// Process a quote tick event
    pub async fn on_quote(&self, quote: &NautilusQuoteTick) -> Result<Option<HyperPhysicsOrderCommand>> {
        if !self.is_running().await {
            return Ok(None);
        }

        let start = Instant::now();

        // Convert to HyperPhysics format
        let feed = self.data_adapter.on_quote(quote).await?;

        // Execute pipeline
        let result = self.execute_pipeline(&feed).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.quotes_processed += 1;
            let latency = start.elapsed().as_micros() as u64;
            metrics.max_signal_latency_us = metrics.max_signal_latency_us.max(latency);
            let n = metrics.quotes_processed as f64;
            metrics.avg_signal_latency_us = metrics.avg_signal_latency_us * ((n - 1.0) / n)
                + latency as f64 / n;
        }

        // Process result through execution bridge
        self.process_pipeline_result(result).await
    }

    /// Process a trade tick event
    pub async fn on_trade(&self, trade: &NautilusTradeTick) -> Result<Option<HyperPhysicsOrderCommand>> {
        if !self.is_running().await {
            return Ok(None);
        }

        // Update data adapter state
        let feed = self.data_adapter.on_trade(trade).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.trades_processed += 1;
        }

        // Execute pipeline
        let result = self.execute_pipeline(&feed).await?;
        self.process_pipeline_result(result).await
    }

    /// Process a bar event
    pub async fn on_bar(&self, bar: &NautilusBar) -> Result<Option<HyperPhysicsOrderCommand>> {
        if !self.is_running().await {
            return Ok(None);
        }

        // Update data adapter state
        let feed = self.data_adapter.on_bar(bar).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.bars_processed += 1;
        }

        // Execute pipeline
        let result = self.execute_pipeline(&feed).await?;
        self.process_pipeline_result(result).await
    }

    /// Process an order book delta
    pub async fn on_book_delta(&self, delta: &NautilusOrderBookDelta) -> Result<()> {
        if !self.is_running().await {
            return Ok(());
        }

        self.data_adapter.on_book_delta(delta).await
    }

    /// Execute the HyperPhysics pipeline
    async fn execute_pipeline(&self, feed: &MarketFeed) -> Result<PipelineResult> {
        self.pipeline.execute(feed).await
            .map_err(|e| IntegrationError::Pipeline(e.to_string()))
    }

    /// Process pipeline result and generate order if appropriate
    async fn process_pipeline_result(
        &self,
        result: PipelineResult,
    ) -> Result<Option<HyperPhysicsOrderCommand>> {
        // Try to generate order through exec bridge
        let order = match self.exec_bridge.process_result(&result).await {
            Ok(order) => order,
            Err(IntegrationError::ConsensusNotReached { .. }) => {
                debug!("Consensus not reached, no order generated");
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        if let Some(ref cmd) = order {
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.signals_generated += 1;
                metrics.orders_submitted += 1;
            }

            // Call order callback if set
            let callback = self.order_callback.read().await;
            if let Some(ref cb) = *callback {
                cb(cmd.clone());
            }
        }

        Ok(order)
    }

    /// Get current market snapshot
    pub async fn get_snapshot(&self) -> Option<MarketSnapshot> {
        let instrument = self.current_instrument.read().await;
        if let Some(ref id) = *instrument {
            // Use instrument hash
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            self.data_adapter.get_snapshot(hasher.finish()).await
        } else {
            None
        }
    }

    /// Get strategy metrics
    pub async fn get_metrics(&self) -> StrategyMetrics {
        let mut metrics = self.metrics.read().await.clone();

        // Update runtime
        let start_time = self.start_time.read().await;
        if let Some(start) = *start_time {
            metrics.runtime_seconds = start.elapsed().as_secs_f64();
        }

        metrics
    }

    /// Get pipeline statistics
    pub async fn get_pipeline_stats(&self) -> PipelineStats {
        self.pipeline.get_stats().await
    }

    /// Get execution bridge statistics
    pub async fn get_exec_stats(&self) -> crate::adapter::exec_bridge::ExecBridgeStats {
        self.exec_bridge.get_stats().await
    }

    /// Get current state
    pub async fn get_state(&self) -> StrategyState {
        *self.state.read().await
    }

    /// Reset strategy state
    pub async fn reset(&self) -> Result<()> {
        // Stop if running
        self.stop().await?;

        // Clear data adapter
        self.data_adapter.clear_all().await;

        // Clear exec bridge
        self.exec_bridge.clear_pending().await;
        self.exec_bridge.reset_stats().await;

        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = StrategyMetrics::default();
        }

        // Reset state
        {
            let mut state = self.state.write().await;
            *state = StrategyState::Initialized;
        }

        info!(strategy_id = %self.strategy_id, "Strategy reset");
        Ok(())
    }
}

impl Clone for HyperPhysicsStrategy {
    fn clone(&self) -> Self {
        Self {
            strategy_id: self.strategy_id.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            data_adapter: self.data_adapter.clone(),
            exec_bridge: self.exec_bridge.clone(),
            pipeline: Arc::clone(&self.pipeline),
            metrics: Arc::clone(&self.metrics),
            start_time: Arc::clone(&self.start_time),
            current_instrument: Arc::clone(&self.current_instrument),
            order_callback: Arc::clone(&self.order_callback),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_strategy_lifecycle() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        assert_eq!(strategy.get_state().await, StrategyState::Initialized);

        strategy.start().await.unwrap();
        assert_eq!(strategy.get_state().await, StrategyState::Running);
        assert!(strategy.is_running().await);

        strategy.stop().await.unwrap();
        assert_eq!(strategy.get_state().await, StrategyState::Stopped);
        assert!(!strategy.is_running().await);
    }

    #[tokio::test]
    async fn test_quote_processing() {
        let config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0, // Accept all signals for testing
            ..Default::default()
        };
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.set_instrument("BTCUSDT.BINANCE").await;
        strategy.start().await.unwrap();

        let quote = NautilusQuoteTick {
            instrument_id: 12345,
            bid_price: 5000000,
            ask_price: 5000100,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1000000000,
            ts_init: 1000000000,
        };

        let _result = strategy.on_quote(&quote).await;

        let metrics = strategy.get_metrics().await;
        assert_eq!(metrics.quotes_processed, 1);
    }

    #[tokio::test]
    async fn test_strategy_reset() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.start().await.unwrap();
        strategy.reset().await.unwrap();

        assert_eq!(strategy.get_state().await, StrategyState::Initialized);
        let metrics = strategy.get_metrics().await;
        assert_eq!(metrics.quotes_processed, 0);
    }
}
