//! Execution bridge for converting HyperPhysics signals to Nautilus orders.

use crate::config::IntegrationConfig;
use crate::error::{IntegrationError, Result};
use crate::types::{
    HyperPhysicsOrderCommand, OrderSide, OrderType, TimeInForce,
    decision_to_order_command,
};
use hyperphysics_hft_ecosystem::core::{Action, TradingDecision};
use hyperphysics_hft_ecosystem::core::unified_pipeline::{ConsensusState, PipelineResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Statistics for the execution bridge
#[derive(Debug, Clone, Default)]
pub struct ExecBridgeStats {
    /// Total signals received
    pub signals_received: u64,
    /// Signals that passed confidence threshold
    pub signals_passed: u64,
    /// Signals that failed confidence threshold
    pub signals_filtered: u64,
    /// Orders generated
    pub orders_generated: u64,
    /// Hold signals (no action)
    pub hold_signals: u64,
    /// Average confidence of passed signals
    pub avg_confidence: f64,
    /// Average latency of passed signals (microseconds)
    pub avg_latency_us: f64,
}

/// Execution bridge for converting HyperPhysics decisions to Nautilus orders.
///
/// The bridge validates signals against confidence thresholds, applies
/// position sizing rules, and formats orders for Nautilus execution.
pub struct NautilusExecBridge {
    /// Configuration
    config: IntegrationConfig,
    /// Order sequence counter
    order_sequence: AtomicU64,
    /// Current instrument being traded
    current_instrument: Arc<RwLock<Option<String>>>,
    /// Statistics
    stats: Arc<RwLock<ExecBridgeStats>>,
    /// Pending orders awaiting confirmation
    pending_orders: Arc<RwLock<Vec<HyperPhysicsOrderCommand>>>,
}

impl NautilusExecBridge {
    /// Create a new execution bridge
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            order_sequence: AtomicU64::new(0),
            current_instrument: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(ExecBridgeStats::default())),
            pending_orders: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set the current trading instrument
    pub async fn set_instrument(&self, instrument_id: &str) {
        let mut current = self.current_instrument.write().await;
        *current = Some(instrument_id.to_string());
    }

    /// Process a pipeline result and generate order command if appropriate
    pub async fn process_result(&self, result: &PipelineResult) -> Result<Option<HyperPhysicsOrderCommand>> {
        let mut stats = self.stats.write().await;
        stats.signals_received += 1;

        // Get instrument
        let instrument = self.current_instrument.read().await;
        let instrument_id = instrument.as_ref()
            .ok_or_else(|| IntegrationError::Configuration("No instrument set".into()))?
            .clone();

        // Check consensus requirement
        if self.config.enable_consensus && !result.consensus_reached {
            warn!(
                confidence = result.decision.confidence,
                "Signal filtered: consensus not reached"
            );
            stats.signals_filtered += 1;
            return Err(IntegrationError::ConsensusNotReached {
                confidence: result.decision.confidence,
                threshold: self.config.min_confidence_threshold,
            });
        }

        // Check confidence threshold
        if result.decision.confidence < self.config.min_confidence_threshold {
            debug!(
                confidence = result.decision.confidence,
                threshold = self.config.min_confidence_threshold,
                "Signal filtered: below confidence threshold"
            );
            stats.signals_filtered += 1;
            return Ok(None);
        }

        // Check for hold action
        if matches!(result.decision.action, Action::Hold) {
            stats.hold_signals += 1;
            return Ok(None);
        }

        stats.signals_passed += 1;

        // Update rolling averages
        let n = stats.signals_passed as f64;
        stats.avg_confidence = stats.avg_confidence * ((n - 1.0) / n)
            + result.decision.confidence / n;
        stats.avg_latency_us = stats.avg_latency_us * ((n - 1.0) / n)
            + result.total_latency_us as f64 / n;

        // Generate order command
        let order = self.create_order_command(
            &result.decision,
            &instrument_id,
            result.total_latency_us,
            result.consensus_state.term,
        )?;

        if let Some(ref cmd) = order {
            stats.orders_generated += 1;
            info!(
                order_id = %cmd.client_order_id,
                side = ?cmd.side,
                quantity = cmd.quantity,
                confidence = cmd.hp_confidence,
                latency_us = cmd.hp_latency_us,
                "Generated order command"
            );

            // Add to pending
            let mut pending = self.pending_orders.write().await;
            pending.push(cmd.clone());
        }

        Ok(order)
    }

    /// Create order command from trading decision
    fn create_order_command(
        &self,
        decision: &TradingDecision,
        instrument_id: &str,
        latency_us: u64,
        consensus_term: u64,
    ) -> Result<Option<HyperPhysicsOrderCommand>> {
        let seq = self.order_sequence.fetch_add(1, Ordering::SeqCst);

        let side = match decision.action {
            Action::Buy => OrderSide::Buy,
            Action::Sell => OrderSide::Sell,
            Action::Hold => return Ok(None),
        };

        // Apply position size limit
        let quantity = (decision.size * self.config.max_position_size)
            .min(self.config.max_position_size);

        let client_order_id = format!(
            "{}-{:08}-{}",
            self.config.order_id_prefix,
            seq,
            chrono::Utc::now().timestamp_micros() % 1_000_000
        );

        Ok(Some(HyperPhysicsOrderCommand {
            client_order_id,
            instrument_id: instrument_id.to_string(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
            post_only: false,
            hp_confidence: decision.confidence,
            hp_algorithm: "HyperPhysicsPipeline".to_string(),
            hp_latency_us: latency_us,
            hp_consensus_term: consensus_term,
        }))
    }

    /// Create a limit order command
    pub fn create_limit_order(
        &self,
        decision: &TradingDecision,
        instrument_id: &str,
        limit_price: f64,
        latency_us: u64,
        consensus_term: u64,
    ) -> Result<Option<HyperPhysicsOrderCommand>> {
        let mut cmd = self.create_order_command(
            decision,
            instrument_id,
            latency_us,
            consensus_term,
        )?;

        if let Some(ref mut order) = cmd {
            order.order_type = OrderType::Limit;
            order.price = Some(limit_price);
            order.time_in_force = TimeInForce::GTC;
        }

        Ok(cmd)
    }

    /// Mark an order as filled/confirmed
    pub async fn on_order_filled(&self, client_order_id: &str) {
        let mut pending = self.pending_orders.write().await;
        pending.retain(|o| o.client_order_id != client_order_id);
    }

    /// Mark an order as rejected/canceled
    pub async fn on_order_rejected(&self, client_order_id: &str, reason: &str) {
        warn!(
            order_id = client_order_id,
            reason = reason,
            "Order rejected"
        );
        let mut pending = self.pending_orders.write().await;
        pending.retain(|o| o.client_order_id != client_order_id);
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ExecBridgeStats {
        self.stats.read().await.clone()
    }

    /// Get number of pending orders
    pub async fn pending_count(&self) -> usize {
        self.pending_orders.read().await.len()
    }

    /// Clear all pending orders
    pub async fn clear_pending(&self) {
        self.pending_orders.write().await.clear();
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = ExecBridgeStats::default();
    }
}

impl Clone for NautilusExecBridge {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            order_sequence: AtomicU64::new(self.order_sequence.load(Ordering::SeqCst)),
            current_instrument: Arc::clone(&self.current_instrument),
            stats: Arc::clone(&self.stats),
            pending_orders: Arc::clone(&self.pending_orders),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_result(action: Action, confidence: f64, consensus: bool) -> PipelineResult {
        PipelineResult {
            decision: TradingDecision {
                action,
                confidence,
                size: 0.5,
            },
            total_latency_us: 500,
            market_data_latency_us: 100,
            physics_latency_us: 200,
            neural_latency_us: 0,
            quantum_latency_us: 0,
            optimization_latency_us: 150,
            consensus_latency_us: 50,
            consensus_state: ConsensusState {
                leader_id: 0,
                term: 1,
                active_nodes: 4,
                byzantine_threshold: 1,
                consensus_latency_us: 50,
            },
            consensus_reached: consensus,
        }
    }

    #[tokio::test]
    async fn test_order_generation() {
        let config = IntegrationConfig {
            min_confidence_threshold: 0.5,
            enable_consensus: false,
            ..Default::default()
        };
        let bridge = NautilusExecBridge::new(config);
        bridge.set_instrument("BTCUSDT.BINANCE").await;

        let result = make_test_result(Action::Buy, 0.8, true);
        let order = bridge.process_result(&result).await.unwrap();

        assert!(order.is_some());
        let cmd = order.unwrap();
        assert_eq!(cmd.side, OrderSide::Buy);
        assert!((cmd.quantity - 0.5).abs() < 0.01);
        assert!(cmd.hp_confidence > 0.7);
    }

    #[tokio::test]
    async fn test_confidence_filtering() {
        let config = IntegrationConfig {
            min_confidence_threshold: 0.6,
            enable_consensus: false,
            ..Default::default()
        };
        let bridge = NautilusExecBridge::new(config);
        bridge.set_instrument("BTCUSDT.BINANCE").await;

        let result = make_test_result(Action::Buy, 0.4, true);
        let order = bridge.process_result(&result).await.unwrap();

        assert!(order.is_none());

        let stats = bridge.get_stats().await;
        assert_eq!(stats.signals_filtered, 1);
    }

    #[tokio::test]
    async fn test_hold_action() {
        let config = IntegrationConfig::default();
        let bridge = NautilusExecBridge::new(config);
        bridge.set_instrument("BTCUSDT.BINANCE").await;

        let result = make_test_result(Action::Hold, 0.9, true);
        let order = bridge.process_result(&result).await.unwrap();

        assert!(order.is_none());

        let stats = bridge.get_stats().await;
        assert_eq!(stats.hold_signals, 1);
    }

    #[tokio::test]
    async fn test_consensus_requirement() {
        let config = IntegrationConfig {
            enable_consensus: true,
            ..Default::default()
        };
        let bridge = NautilusExecBridge::new(config);
        bridge.set_instrument("BTCUSDT.BINANCE").await;

        let result = make_test_result(Action::Buy, 0.9, false);
        let order = bridge.process_result(&result).await;

        assert!(order.is_err());
        assert!(matches!(order.unwrap_err(), IntegrationError::ConsensusNotReached { .. }));
    }
}
