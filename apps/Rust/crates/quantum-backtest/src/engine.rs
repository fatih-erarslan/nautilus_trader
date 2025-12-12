//! Backtesting engine core implementation

use crate::{BacktestConfig, BacktestResult, BacktestError, Result};
use crate::{MarketData, TradingSignal, SignalType};
use crate::strategy::TengriQuantumStrategy;
use crate::portfolio::{Portfolio, Trade};
use crate::metrics::{PerformanceMetrics, RiskMetrics, DrawdownMetrics};
use crate::data::DataLoader;
use crate::quantum::QuantumPatternEngine;

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug, warn};

/// Main backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    strategy: TengriQuantumStrategy,
    portfolio: Portfolio,
    quantum_engine: Option<QuantumPatternEngine>,
    data_loader: DataLoader,
}

impl BacktestEngine {
    /// Create a new backtesting engine
    pub async fn new(config: BacktestConfig) -> Result<Self> {
        info!("Initializing backtesting engine with config: {:?}", config);
        
        let strategy = TengriQuantumStrategy::new(&config);
        let portfolio = Portfolio::new(config.initial_capital);
        let data_loader = DataLoader::new(&config).await?;
        
        let quantum_engine = if config.enable_quantum {
            Some(QuantumPatternEngine::new(&config.quantum_thresholds).await?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            strategy,
            portfolio,
            quantum_engine,
            data_loader,
        })
    }
    
    /// Run the complete backtest
    pub async fn run(&mut self) -> Result<BacktestResult> {
        let start_time = Instant::now();
        
        info!("Starting backtest from {} to {}", 
              self.config.start_date, self.config.end_date);
        
        // Load historical data
        let market_data = self.load_market_data().await?;
        info!("Loaded {} data points for {} assets", 
              market_data.len(), self.config.assets.len());
        
        // Initialize tracking
        let mut trades = Vec::new();
        let mut signals = Vec::new();
        let mut portfolio_history = Vec::new();
        
        // Process each time step
        let mut processed_bars = 0;
        for (timestamp, data_points) in &market_data {
            processed_bars += 1;
            
            if processed_bars % 1000 == 0 {
                debug!("Processed {} bars, current time: {}", processed_bars, timestamp);
            }
            
            // Update portfolio with current market prices
            self.portfolio.update_market_prices(data_points);
            
            // Generate quantum signals if enabled
            let quantum_signals = if let Some(ref quantum_engine) = self.quantum_engine {
                self.generate_quantum_signals(quantum_engine, timestamp, data_points).await?
            } else {
                Vec::new()
            };
            
            // Generate trading signals from strategy
            let strategy_signals = self.strategy.generate_signals(
                timestamp, 
                data_points, 
                &quantum_signals,
                &self.portfolio
            ).await?;
            
            // Execute trades based on signals
            let new_trades = self.execute_signals(&strategy_signals, timestamp, data_points).await?;
            trades.extend(new_trades);
            
            // Store signals and portfolio value
            signals.extend(strategy_signals);
            portfolio_history.push((*timestamp, self.portfolio.total_value()));
            
            // Check risk limits
            if self.check_risk_limits()? {
                warn!("Risk limits exceeded, stopping backtest at {}", timestamp);
                break;
            }
        }
        
        let execution_time = start_time.elapsed();
        
        // Calculate final metrics
        let performance = self.calculate_performance_metrics(&trades, &portfolio_history)?;
        let risk_metrics = self.calculate_risk_metrics(&portfolio_history)?;
        let drawdown_metrics = self.calculate_drawdown_metrics(&portfolio_history)?;
        
        info!("Backtest completed in {:?}", execution_time);
        info!("Total trades: {}, Final portfolio value: ${:.2}", 
              trades.len(), self.portfolio.total_value());
        
        Ok(BacktestResult {
            config: self.config.clone(),
            performance,
            risk_metrics,
            drawdown_metrics,
            trades,
            quantum_signals: signals,
            portfolio_value_history: portfolio_history,
            execution_time_ms: execution_time.as_millis(),
        })
    }
    
    /// Load and organize market data
    async fn load_market_data(&self) -> Result<Vec<(DateTime<Utc>, HashMap<String, MarketData>)>> {
        let mut all_data = HashMap::new();
        
        // Load data for each asset
        for asset in &self.config.assets {
            let asset_data = self.data_loader.load_asset_data(
                asset,
                &self.config.timeframe,
                self.config.start_date,
                self.config.end_date,
            ).await?;
            
            all_data.insert(asset.clone(), asset_data);
        }
        
        // Combine data by timestamp
        let mut combined_data = Vec::new();
        let mut timestamps = std::collections::BTreeSet::new();
        
        // Collect all unique timestamps
        for asset_data in all_data.values() {
            for data_point in asset_data {
                timestamps.insert(data_point.timestamp);
            }
        }
        
        // Create combined data points
        for timestamp in timestamps {
            let mut data_points = HashMap::new();
            
            for (asset, asset_data) in &all_data {
                if let Some(data_point) = asset_data.iter().find(|d| d.timestamp == timestamp) {
                    data_points.insert(asset.clone(), data_point.clone());
                }
            }
            
            if !data_points.is_empty() {
                combined_data.push((timestamp, data_points));
            }
        }
        
        combined_data.sort_by_key(|(timestamp, _)| *timestamp);
        Ok(combined_data)
    }
    
    /// Generate quantum pattern signals
    async fn generate_quantum_signals(
        &self,
        quantum_engine: &QuantumPatternEngine,
        timestamp: &DateTime<Utc>,
        data_points: &HashMap<String, MarketData>,
    ) -> Result<Vec<TradingSignal>> {
        let mut signals = Vec::new();
        
        for (symbol, data) in data_points {
            // Prepare price data for quantum analysis
            let price_series = vec![data.open, data.high, data.low, data.close];
            
            // Analyze quantum patterns
            let quantum_result = quantum_engine.analyze_patterns(&price_series).await
                .map_err(|e| BacktestError::Quantum(e.to_string()))?;
            
            // Convert quantum patterns to trading signals
            if quantum_result.confidence > self.config.quantum_thresholds.superposition_threshold {
                let signal_type = if quantum_result.trend_strength > 0.7 {
                    if quantum_result.trend_direction > 0.0 {
                        SignalType::StrongBuy
                    } else {
                        SignalType::StrongSell
                    }
                } else if quantum_result.trend_strength > 0.4 {
                    if quantum_result.trend_direction > 0.0 {
                        SignalType::Buy
                    } else {
                        SignalType::Sell
                    }
                } else {
                    SignalType::Hold
                };
                
                signals.push(TradingSignal {
                    timestamp: *timestamp,
                    symbol: symbol.clone(),
                    signal_type,
                    confidence: quantum_result.confidence,
                    quantum_patterns: quantum_result.detected_patterns,
                    price_target: Some(data.close * (1.0 + quantum_result.trend_direction * 0.1)),
                    stop_loss: Some(data.close * (1.0 - quantum_result.trend_direction.abs() * 0.05)),
                });
            }
        }
        
        Ok(signals)
    }
    
    /// Execute trading signals
    async fn execute_signals(
        &mut self,
        signals: &[TradingSignal],
        timestamp: &DateTime<Utc>,
        data_points: &HashMap<String, MarketData>,
    ) -> Result<Vec<Trade>> {
        let mut executed_trades = Vec::new();
        
        for signal in signals {
            if let Some(market_data) = data_points.get(&signal.symbol) {
                let trade_result = self.portfolio.execute_signal(
                    signal,
                    market_data,
                    &self.config.risk_management,
                ).await;
                
                match trade_result {
                    Ok(Some(trade)) => {
                        executed_trades.push(trade);
                        debug!("Executed trade: {} {} at ${:.2}", 
                               signal.signal_type as u8, signal.symbol, market_data.close);
                    },
                    Ok(None) => {
                        // No trade executed (risk limits, insufficient funds, etc.)
                    },
                    Err(e) => {
                        warn!("Failed to execute trade for {}: {}", signal.symbol, e);
                    }
                }
            }
        }
        
        Ok(executed_trades)
    }
    
    /// Check if risk limits are exceeded
    fn check_risk_limits(&self) -> Result<bool> {
        let current_drawdown = self.portfolio.current_drawdown();
        
        if current_drawdown > self.config.risk_management.max_drawdown {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        trades: &[Trade],
        portfolio_history: &[(DateTime<Utc>, f64)],
    ) -> Result<PerformanceMetrics> {
        PerformanceMetrics::calculate(
            self.config.initial_capital,
            trades,
            portfolio_history,
        )
    }
    
    /// Calculate risk metrics
    fn calculate_risk_metrics(
        &self,
        portfolio_history: &[(DateTime<Utc>, f64)],
    ) -> Result<RiskMetrics> {
        RiskMetrics::calculate(portfolio_history)
    }
    
    /// Calculate drawdown metrics
    fn calculate_drawdown_metrics(
        &self,
        portfolio_history: &[(DateTime<Utc>, f64)],
    ) -> Result<DrawdownMetrics> {
        DrawdownMetrics::calculate(portfolio_history)
    }
}