//! Core trading strategy implementation for Tengri
//! 
//! Provides the main TengriStrategy struct that orchestrates all components
//! including data aggregation, signal generation, risk management, and execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, sleep};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::{Result, TengriError};
use crate::config::TengriConfig;
use crate::data::{DataAggregator, MarketEvent};
use crate::signals::SignalGenerator;
use crate::risk::RiskManager;
use crate::execution::ExecutionEngine;
use crate::monitoring::PerformanceMonitor;
use crate::types::{
    TradingSignal, Position, Order, PortfolioMetrics, RiskMetrics,
    MarketState, TradingSession, PositionSide, SignalType
};

/// Main trading strategy that coordinates all components
pub struct TengriStrategy {
    /// Strategy configuration
    config: TengriConfig,
    
    /// Data aggregation from multiple sources
    data_aggregator: DataAggregator,
    
    /// Signal generation engine
    signal_generator: SignalGenerator,
    
    /// Risk management system
    risk_manager: RiskManager,
    
    /// Order execution engine
    execution_engine: ExecutionEngine,
    
    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
    
    /// Current trading session
    current_session: Arc<RwLock<Option<TradingSession>>>,
    
    /// Active positions
    positions: Arc<RwLock<HashMap<String, Position>>>,
    
    /// Active orders
    orders: Arc<RwLock<HashMap<String, Order>>>,
    
    /// Portfolio metrics
    portfolio_metrics: Arc<RwLock<PortfolioMetrics>>,
    
    /// Market event receiver
    market_event_rx: Option<broadcast::Receiver<MarketEvent>>,
    
    /// Strategy state
    is_running: Arc<RwLock<bool>>,
    
    /// Strategy performance
    session_pnl: Arc<RwLock<f64>>,
}

impl TengriStrategy {
    /// Create new Tengri strategy instance
    pub async fn new(config: TengriConfig) -> Result<Self> {
        tracing::info!("Initializing Tengri trading strategy");

        // Initialize components
        let data_aggregator = DataAggregator::new(config.data_sources.clone()).await?;
        let signal_generator = SignalGenerator::new(config.strategy.parameters.clone());
        let risk_manager = RiskManager::new(config.risk.clone()).await?;
        let execution_engine = ExecutionEngine::new(config.exchanges.clone()).await?;
        let performance_monitor = PerformanceMonitor::new(config.monitoring.clone()).await?;

        Ok(Self {
            config,
            data_aggregator,
            signal_generator,
            risk_manager,
            execution_engine,
            performance_monitor,
            current_session: Arc::new(RwLock::new(None)),
            positions: Arc::new(RwLock::new(HashMap::new())),
            orders: Arc::new(RwLock::new(HashMap::new())),
            portfolio_metrics: Arc::new(RwLock::new(PortfolioMetrics::default())),
            market_event_rx: None,
            is_running: Arc::new(RwLock::new(false)),
            session_pnl: Arc::new(RwLock::new(0.0)),
        })
    }

    /// Start the trading strategy
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting Tengri trading strategy");

        // Set running state
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }

        // Start new trading session
        self.start_new_session().await?;

        // Start data streaming
        self.data_aggregator.start_streaming().await?;

        // Subscribe to market events
        self.market_event_rx = Some(self.data_aggregator.subscribe_events());

        // Start main trading loop
        self.run_trading_loop().await?;

        Ok(())
    }

    /// Stop the trading strategy
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping Tengri trading strategy");

        // Set running state to false
        {
            let mut running = self.is_running.write().await;
            *running = false;
        }

        // Close all positions
        self.close_all_positions().await?;

        // Cancel all pending orders
        self.cancel_all_orders().await?;

        // End current session
        self.end_current_session().await?;

        tracing::info!("Tengri trading strategy stopped");
        Ok(())
    }

    /// Main trading loop
    async fn run_trading_loop(&mut self) -> Result<()> {
        let mut market_events = self.market_event_rx.take()
            .ok_or_else(|| TengriError::Strategy("Market event receiver not initialized".to_string()))?;

        // Performance monitoring interval
        let mut performance_interval = interval(Duration::from_secs(60));
        
        // Portfolio metrics update interval
        let mut metrics_interval = interval(Duration::from_secs(30));

        // Risk check interval
        let mut risk_interval = interval(Duration::from_secs(10));

        loop {
            tokio::select! {
                // Process market events
                Ok(event) = market_events.recv() => {
                    if let Err(e) = self.process_market_event(event).await {
                        tracing::error!("Error processing market event: {}", e);
                    }
                }

                // Performance monitoring
                _ = performance_interval.tick() => {
                    if let Err(e) = self.update_performance_metrics().await {
                        tracing::error!("Error updating performance metrics: {}", e);
                    }
                }

                // Portfolio metrics update
                _ = metrics_interval.tick() => {
                    if let Err(e) = self.update_portfolio_metrics().await {
                        tracing::error!("Error updating portfolio metrics: {}", e);
                    }
                }

                // Risk management checks
                _ = risk_interval.tick() => {
                    if let Err(e) = self.perform_risk_checks().await {
                        tracing::error!("Error performing risk checks: {}", e);
                    }
                }
            }

            // Check if strategy should continue running
            {
                let running = self.is_running.read().await;
                if !*running {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Process incoming market events
    async fn process_market_event(&mut self, event: MarketEvent) -> Result<()> {
        match event {
            MarketEvent::Price(price_update) => {
                self.process_price_update(price_update).await?;
            }
            MarketEvent::Trade(trade_data) => {
                self.process_trade_data(trade_data).await?;
            }
            MarketEvent::OrderBook(order_book) => {
                self.process_order_book_update(order_book).await?;
            }
            MarketEvent::PolymarketOdds(polymarket_data) => {
                self.process_polymarket_update(polymarket_data).await?;
            }
            MarketEvent::DataQualityAlert(alert) => {
                self.process_data_quality_alert(alert).await?;
            }
        }

        Ok(())
    }

    /// Process price updates and generate trading signals
    async fn process_price_update(&mut self, price_update: crate::data::PriceUpdate) -> Result<()> {
        let symbol = &price_update.symbol;
        
        // Convert to PriceData format
        let price_data = crate::types::PriceData {
            symbol: price_update.symbol.clone(),
            price: price_update.price,
            timestamp: price_update.timestamp,
            source: price_update.source,
            volume_24h: price_update.volume_24h,
            bid: price_update.bid,
            ask: price_update.ask,
        };

        // Update signal generator
        if let Some(signal) = self.signal_generator.update_price(price_data).await? {
            self.process_trading_signal(signal).await?;
        }

        // Update existing positions with new prices
        self.update_position_prices(symbol, price_update.price).await?;

        Ok(())
    }

    /// Process trade data for market analysis
    async fn process_trade_data(&mut self, _trade_data: crate::types::TradeData) -> Result<()> {
        // Analyze trade flow and market microstructure
        // Update market state based on trade patterns
        // This could be used for more sophisticated signal generation
        Ok(())
    }

    /// Process order book updates
    async fn process_order_book_update(&mut self, _order_book: crate::data::OrderBookUpdate) -> Result<()> {
        // Analyze order book depth and liquidity
        // Update execution algorithms based on market depth
        // Calculate market impact estimates
        Ok(())
    }

    /// Process Polymarket prediction market updates
    async fn process_polymarket_update(&mut self, _polymarket_data: crate::data::PolymarketUpdate) -> Result<()> {
        // Analyze prediction market sentiment
        // Correlate with crypto price movements
        // Generate cross-asset signals
        Ok(())
    }

    /// Process data quality alerts
    async fn process_data_quality_alert(&mut self, alert: crate::data::DataQualityAlert) -> Result<()> {
        tracing::warn!("Data quality alert: {} - {}", alert.alert_type, alert.message);
        
        match alert.severity {
            crate::data::AlertSeverity::Critical => {
                // Consider pausing trading or switching to backup data source
                tracing::error!("Critical data quality issue detected");
            }
            crate::data::AlertSeverity::High => {
                // Reduce position sizes or increase signal thresholds
                tracing::warn!("High severity data quality issue");
            }
            _ => {
                // Log and monitor
                tracing::info!("Data quality issue noted: {}", alert.message);
            }
        }

        Ok(())
    }

    /// Process trading signals and make trading decisions
    async fn process_trading_signal(&mut self, signal: TradingSignal) -> Result<()> {
        tracing::info!("Processing signal: {:?} for {} with strength {:.3}", 
            signal.signal_type, signal.symbol, signal.strength);

        // Check if signal is still valid
        if !signal.is_valid() {
            return Ok(());
        }

        // Get current position for this symbol
        let current_position = {
            let positions = self.positions.read().await;
            positions.get(&signal.symbol).cloned()
        };

        // Risk management pre-check
        let risk_approval = self.risk_manager.evaluate_signal(&signal, current_position.as_ref()).await?;
        if !risk_approval.approved {
            tracing::info!("Signal rejected by risk management: {}", risk_approval.reason);
            return Ok(());
        }

        // Determine trade action
        let trade_action = self.determine_trade_action(&signal, current_position.as_ref()).await?;

        // Execute trade if action is determined
        if let Some(action) = trade_action {
            self.execute_trade_action(action).await?;
        }

        Ok(())
    }

    /// Determine what trading action to take based on signal and current position
    async fn determine_trade_action(&self, signal: &TradingSignal, current_position: Option<&Position>) -> Result<Option<TradeAction>> {
        let symbol = &signal.symbol;
        let signal_strength = signal.strength.abs();
        
        // Calculate position size based on risk management
        let max_position_size = self.risk_manager.calculate_position_size(signal).await?;
        
        let trade_action = match signal.signal_type {
            SignalType::Buy | SignalType::StrongBuy => {
                let target_size = if signal.signal_type == SignalType::StrongBuy {
                    max_position_size
                } else {
                    max_position_size * 0.7 // Reduce size for regular buy
                };

                match current_position {
                    None => {
                        // No position, open new long position
                        Some(TradeAction::OpenPosition {
                            symbol: symbol.clone(),
                            side: PositionSide::Long,
                            size: target_size,
                            reason: "New long position based on buy signal".to_string(),
                        })
                    }
                    Some(pos) if pos.side == PositionSide::Short => {
                        // Close short position and potentially open long
                        Some(TradeAction::ClosePosition {
                            position_id: pos.id,
                            reason: "Close short due to buy signal".to_string(),
                        })
                    }
                    Some(pos) if pos.side == PositionSide::Long => {
                        // Existing long position, consider increasing size
                        if target_size > pos.size && signal_strength > 0.8 {
                            Some(TradeAction::IncreasePosition {
                                position_id: pos.id,
                                additional_size: target_size - pos.size,
                                reason: "Increase long position on strong signal".to_string(),
                            })
                        } else {
                            None // No action needed
                        }
                    }
                    _ => None,
                }
            }

            SignalType::Sell | SignalType::StrongSell => {
                let target_size = if signal.signal_type == SignalType::StrongSell {
                    max_position_size
                } else {
                    max_position_size * 0.7
                };

                match current_position {
                    None => {
                        // No position, open new short position (if enabled)
                        if self.config.strategy.parameters.max_position_size > 0.0 {
                            Some(TradeAction::OpenPosition {
                                symbol: symbol.clone(),
                                side: PositionSide::Short,
                                size: target_size,
                                reason: "New short position based on sell signal".to_string(),
                            })
                        } else {
                            None
                        }
                    }
                    Some(pos) if pos.side == PositionSide::Long => {
                        // Close long position
                        Some(TradeAction::ClosePosition {
                            position_id: pos.id,
                            reason: "Close long due to sell signal".to_string(),
                        })
                    }
                    Some(pos) if pos.side == PositionSide::Short => {
                        // Existing short position, consider increasing size
                        if target_size > pos.size && signal_strength > 0.8 {
                            Some(TradeAction::IncreasePosition {
                                position_id: pos.id,
                                additional_size: target_size - pos.size,
                                reason: "Increase short position on strong signal".to_string(),
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }

            SignalType::Hold => {
                // Check if we should close positions due to weak signals
                if let Some(pos) = current_position {
                    if signal.confidence < 0.3 {
                        Some(TradeAction::ClosePosition {
                            position_id: pos.id,
                            reason: "Close position due to low confidence hold signal".to_string(),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        Ok(trade_action)
    }

    /// Execute the determined trade action
    async fn execute_trade_action(&mut self, action: TradeAction) -> Result<()> {
        match action {
            TradeAction::OpenPosition { symbol, side, size, reason } => {
                tracing::info!("Opening {} position for {} with size {}: {}", 
                    match side { PositionSide::Long => "long", PositionSide::Short => "short" },
                    symbol, size, reason);
                
                let order = self.execution_engine.create_market_order(
                    &symbol,
                    match side {
                        PositionSide::Long => crate::types::OrderSide::Buy,
                        PositionSide::Short => crate::types::OrderSide::Sell,
                    },
                    size,
                ).await?;

                // Store order
                {
                    let mut orders = self.orders.write().await;
                    orders.insert(order.id.clone(), order);
                }
            }

            TradeAction::ClosePosition { position_id, reason } => {
                let position = {
                    let positions = self.positions.read().await;
                    positions.get(&position_id.to_string()).cloned()
                };

                if let Some(pos) = position {
                    tracing::info!("Closing position {} for {}: {}", 
                        position_id, pos.symbol, reason);
                    
                    let order = self.execution_engine.create_market_order(
                        &pos.symbol,
                        match pos.side {
                            PositionSide::Long => crate::types::OrderSide::Sell,
                            PositionSide::Short => crate::types::OrderSide::Buy,
                        },
                        pos.size,
                    ).await?;

                    // Store order
                    {
                        let mut orders = self.orders.write().await;
                        orders.insert(order.id.clone(), order);
                    }
                }
            }

            TradeAction::IncreasePosition { position_id, additional_size, reason } => {
                let position = {
                    let positions = self.positions.read().await;
                    positions.get(&position_id.to_string()).cloned()
                };

                if let Some(pos) = position {
                    tracing::info!("Increasing position {} for {} by {}: {}", 
                        position_id, pos.symbol, additional_size, reason);
                    
                    let order = self.execution_engine.create_market_order(
                        &pos.symbol,
                        match pos.side {
                            PositionSide::Long => crate::types::OrderSide::Buy,
                            PositionSide::Short => crate::types::OrderSide::Sell,
                        },
                        additional_size,
                    ).await?;

                    // Store order
                    {
                        let mut orders = self.orders.write().await;
                        orders.insert(order.id.clone(), order);
                    }
                }
            }
        }

        Ok(())
    }

    /// Update position prices based on market data
    async fn update_position_prices(&mut self, symbol: &str, new_price: f64) -> Result<()> {
        let mut positions = self.positions.write().await;
        
        if let Some(position) = positions.get_mut(symbol) {
            let old_pnl = position.unrealized_pnl;
            position.current_price = new_price;
            position.unrealized_pnl = position.calculate_pnl();
            position.updated_at = Utc::now();
            
            // Update session P&L
            let pnl_change = position.unrealized_pnl - old_pnl;
            let mut session_pnl = self.session_pnl.write().await;
            *session_pnl += pnl_change;
        }

        Ok(())
    }

    /// Perform risk management checks
    async fn perform_risk_checks(&mut self) -> Result<()> {
        let positions = self.positions.read().await;
        let portfolio_metrics = self.portfolio_metrics.read().await;

        // Check portfolio-level risk limits
        let risk_metrics = self.risk_manager.calculate_portfolio_risk(&*positions, &*portfolio_metrics).await?;

        // Emergency position closure if risk limits exceeded
        if risk_metrics.leverage > self.config.risk.position_sizing.max_leverage * 1.1 {
            tracing::warn!("Leverage exceeded safe limits: {:.2}x", risk_metrics.leverage);
            // Consider closing some positions
        }

        let should_emergency_stop = portfolio_metrics.max_drawdown < -self.config.risk.max_portfolio_loss;
        drop(positions); // Drop the borrow before calling emergency_stop
        drop(portfolio_metrics); // Drop the borrow before calling emergency_stop

        if should_emergency_stop {
            tracing::error!("Maximum portfolio loss exceeded");
            // Emergency stop - close all positions
            self.emergency_stop().await?;
        }

        Ok(())
    }

    /// Emergency stop - close all positions immediately
    async fn emergency_stop(&mut self) -> Result<()> {
        tracing::error!("EMERGENCY STOP: Closing all positions immediately");
        
        self.close_all_positions().await?;
        self.cancel_all_orders().await?;
        
        // Set running state to false
        {
            let mut running = self.is_running.write().await;
            *running = false;
        }

        Ok(())
    }

    /// Close all open positions
    async fn close_all_positions(&mut self) -> Result<()> {
        let positions = {
            let pos = self.positions.read().await;
            pos.values().cloned().collect::<Vec<_>>()
        };

        for position in positions {
            let _order = self.execution_engine.create_market_order(
                &position.symbol,
                match position.side {
                    PositionSide::Long => crate::types::OrderSide::Sell,
                    PositionSide::Short => crate::types::OrderSide::Buy,
                },
                position.size,
            ).await?;

            tracing::info!("Closing position {} for {}", position.id, position.symbol);
        }

        Ok(())
    }

    /// Cancel all pending orders
    async fn cancel_all_orders(&mut self) -> Result<()> {
        let orders = {
            let ord = self.orders.read().await;
            ord.values().cloned().collect::<Vec<_>>()
        };

        for order in orders {
            if order.is_active() {
                self.execution_engine.cancel_order(&order.id).await?;
                tracing::info!("Cancelled order {}", order.id);
            }
        }

        Ok(())
    }

    /// Start a new trading session
    async fn start_new_session(&mut self) -> Result<()> {
        let session = TradingSession {
            id: Uuid::new_v4(),
            start_time: Utc::now(),
            end_time: None,
            session_pnl: 0.0,
            trade_count: 0,
            win_rate: 0.0,
            max_drawdown: 0.0,
            notes: Some(format!("Tengri strategy session - {}", self.config.strategy.name)),
        };

        {
            let mut current_session = self.current_session.write().await;
            *current_session = Some(session);
        }

        {
            let mut session_pnl = self.session_pnl.write().await;
            *session_pnl = 0.0;
        }

        tracing::info!("Started new trading session");
        Ok(())
    }

    /// End the current trading session
    async fn end_current_session(&mut self) -> Result<()> {
        let mut current_session = self.current_session.write().await;
        
        if let Some(ref mut session) = *current_session {
            session.end_time = Some(Utc::now());
            session.session_pnl = *self.session_pnl.read().await;
            
            tracing::info!("Ended trading session {} with P&L: {:.2}", 
                session.id, session.session_pnl);
        }

        *current_session = None;
        Ok(())
    }

    /// Update performance metrics
    async fn update_performance_metrics(&mut self) -> Result<()> {
        let positions = self.positions.read().await;
        let session_pnl = *self.session_pnl.read().await;

        // Calculate portfolio metrics
        let total_value = positions.values().map(|p| p.value()).sum::<f64>();
        let unrealized_pnl = positions.values().map(|p| p.unrealized_pnl).sum::<f64>();
        let open_positions = positions.len();

        {
            let mut metrics = self.portfolio_metrics.write().await;
            metrics.total_value = total_value;
            metrics.unrealized_pnl = unrealized_pnl;
            metrics.realized_pnl = session_pnl;
            metrics.open_positions = open_positions;
            metrics.timestamp = Utc::now();
        }

        Ok(())
    }

    /// Update portfolio metrics
    async fn update_portfolio_metrics(&mut self) -> Result<()> {
        // This would typically calculate more sophisticated metrics
        // like Sharpe ratio, maximum drawdown, etc.
        self.update_performance_metrics().await
    }

    /// Get current portfolio metrics
    pub async fn get_portfolio_metrics(&self) -> PortfolioMetrics {
        self.portfolio_metrics.read().await.clone()
    }

    /// Get current positions
    pub async fn get_positions(&self) -> HashMap<String, Position> {
        self.positions.read().await.clone()
    }

    /// Get current orders
    pub async fn get_orders(&self) -> HashMap<String, Order> {
        self.orders.read().await.clone()
    }

    /// Get current trading session
    pub async fn get_current_session(&self) -> Option<TradingSession> {
        self.current_session.read().await.clone()
    }

    /// Get strategy configuration
    pub fn get_config(&self) -> &TengriConfig {
        &self.config
    }
}

/// Trading action types
#[derive(Debug, Clone)]
enum TradeAction {
    OpenPosition {
        symbol: String,
        side: PositionSide,
        size: f64,
        reason: String,
    },
    ClosePosition {
        position_id: Uuid,
        reason: String,
    },
    IncreasePosition {
        position_id: Uuid,
        additional_size: f64,
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TengriConfig;

    #[tokio::test]
    async fn test_strategy_creation() {
        let config = TengriConfig::default();
        let strategy = TengriStrategy::new(config).await;
        assert!(strategy.is_ok());
    }

    #[tokio::test]
    async fn test_session_management() {
        let config = TengriConfig::default();
        let mut strategy = TengriStrategy::new(config).await.unwrap();
        
        strategy.start_new_session().await.unwrap();
        let session = strategy.get_current_session().await;
        assert!(session.is_some());
        
        strategy.end_current_session().await.unwrap();
        let session = strategy.get_current_session().await;
        assert!(session.is_none());
    }
}