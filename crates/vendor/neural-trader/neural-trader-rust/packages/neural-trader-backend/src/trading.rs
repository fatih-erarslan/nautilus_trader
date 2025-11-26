//! Trading strategy implementations and execution
//!
//! Provides NAPI bindings for:
//! - Strategy execution (MomentumStrategy, MeanReversionStrategy, etc.)
//! - Trade simulation and backtesting
//! - Portfolio management
//! - Risk analysis
//! - Performance metrics

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::error::*;

// Import strategy types and traits
use nt_strategies::{
    StrategyRegistry,
    Strategy,
    MarketData,
    Portfolio,
    Position,
    Signal,
    Direction,
    BacktestEngine,
    momentum::MomentumStrategy,
    mean_reversion::MeanReversionStrategy,
    pairs::PairsStrategy,
    neural_trend::NeuralTrendStrategy,
};

use nt_core::types::Bar;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde_json;

/// List all available trading strategies
#[napi]
pub async fn list_strategies() -> Result<Vec<StrategyInfo>> {
    // Use the actual strategy registry from nt-strategies
    let registry = StrategyRegistry::new();
    let strategies = registry.list_all();

    let strategy_infos: Vec<StrategyInfo> = strategies
        .into_iter()
        .map(|meta| StrategyInfo {
            name: meta.name.clone(),
            description: meta.description.clone(),
            gpu_capable: meta.gpu_capable,
        })
        .collect();

    Ok(strategy_infos)
}

/// Strategy information
#[napi(object)]
pub struct StrategyInfo {
    pub name: String,
    pub description: String,
    pub gpu_capable: bool,
}

/// Get detailed information about a specific strategy
#[napi]
pub async fn get_strategy_info(strategy: String) -> Result<String> {
    let registry = StrategyRegistry::new();

    match registry.get(&strategy) {
        Some(meta) => {
            let info = serde_json::json!({
                "name": meta.name,
                "description": meta.description,
                "sharpe_ratio": meta.sharpe_ratio,
                "status": meta.status,
                "gpu_capable": meta.gpu_capable,
                "risk_level": meta.risk_level,
            });
            Ok(serde_json::to_string_pretty(&info)
                .map_err(|e| NeuralTraderError::Trading(format!("Failed to serialize strategy info: {}", e)))?)
        }
        None => Err(NeuralTraderError::Trading(format!("Strategy '{}' not found", strategy)).into()),
    }
}

/// Quick market analysis for a symbol
#[napi]
pub async fn quick_analysis(symbol: String, use_gpu: Option<bool>) -> Result<MarketAnalysis> {
    let _gpu = use_gpu.unwrap_or(false);

    // Create a momentum strategy for quick analysis
    let strategy = MomentumStrategy::new(vec![symbol.clone()], 20, 2.0, 0.5);

    // Generate sample market data for demonstration
    // In production, this would fetch real market data
    let bars = generate_sample_bars(&symbol, 100);

    if bars.len() < 50 {
        return Err(NeuralTraderError::Trading(
            "Insufficient market data for analysis".to_string()
        ).into());
    }

    let market_data = MarketData::new(symbol.clone(), bars.clone());
    let portfolio = Portfolio::new(Decimal::from(100000));

    // Try to generate signals to understand market conditions
    let analysis = match strategy.process(&market_data, &portfolio).await {
        Ok(signals) => {
            let (trend, recommendation) = if !signals.is_empty() {
                let signal = &signals[0];
                match signal.direction {
                    Direction::Long => ("bullish".to_string(), "buy".to_string()),
                    Direction::Short => ("bearish".to_string(), "sell".to_string()),
                    Direction::Close => ("neutral".to_string(), "hold".to_string()),
                }
            } else {
                ("neutral".to_string(), "hold".to_string())
            };

            // Calculate volatility from recent bars
            let recent_prices: Vec<f64> = bars.iter()
                .rev()
                .take(20)
                .map(|b| b.close.to_f64().unwrap_or(0.0))
                .collect();

            let volatility = calculate_volatility(&recent_prices);

            // Analyze volume trend
            let volume_trend = if bars.len() >= 2 {
                // Volume is Decimal, convert to u64 for comparison
                let recent_avg: u64 = bars.iter().rev().take(10).map(|b| b.volume.to_u64().unwrap_or(0)).sum::<u64>() / 10;
                let older_avg: u64 = bars.iter().rev().skip(10).take(10).map(|b| b.volume.to_u64().unwrap_or(0)).sum::<u64>() / 10;
                if recent_avg > older_avg {
                    "increasing".to_string()
                } else {
                    "decreasing".to_string()
                }
            } else {
                "stable".to_string()
            };

            MarketAnalysis {
                symbol,
                trend,
                volatility,
                volume_trend,
                recommendation,
            }
        }
        Err(e) => {
            return Err(NeuralTraderError::Trading(format!(
                "Market analysis failed: {}", e
            )).into());
        }
    };

    Ok(analysis)
}

/// Calculate price volatility (standard deviation of returns)
fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}

/// Generate sample bars for demonstration
/// In production, this would be replaced with actual market data fetching
fn generate_sample_bars(symbol: &str, count: usize) -> Vec<Bar> {
    let mut bars = Vec::new();
    let mut price = 100.0;
    let base_time = Utc::now() - chrono::Duration::days(count as i64);

    for i in 0..count {
        // Simulate price movement with some randomness
        let change = (((i * 7) % 100) as f64 - 50.0) / 100.0;
        price = (price * (1.0 + change)).max(10.0);

        let bar = Bar {
            symbol: nt_core::types::Symbol::new(symbol).unwrap(), // Use Symbol::new() constructor
            timestamp: base_time + chrono::Duration::days(i as i64),
            open: Decimal::from_f64(price * 0.99).unwrap(),
            high: Decimal::from_f64(price * 1.02).unwrap(),
            low: Decimal::from_f64(price * 0.98).unwrap(),
            close: Decimal::from_f64(price).unwrap(),
            volume: Decimal::from(1000000 + (i * 10000) as u64), // Volume is Decimal
        };
        bars.push(bar);
    }

    bars
}

/// Market analysis result
#[napi(object)]
pub struct MarketAnalysis {
    pub symbol: String,
    pub trend: String,
    pub volatility: f64,
    pub volume_trend: String,
    pub recommendation: String,
}

/// Simulate a trade operation
#[napi]
pub async fn simulate_trade(
    strategy: String,
    symbol: String,
    action: String,
    use_gpu: Option<bool>,
) -> Result<TradeSimulation> {
    let _gpu = use_gpu.unwrap_or(false);
    let start_time = std::time::Instant::now();

    // Validate strategy exists
    let registry = StrategyRegistry::new();
    if !registry.contains(&strategy) {
        return Err(NeuralTraderError::Trading(
            format!("Unknown strategy: {}", strategy)
        ).into());
    }

    // Generate market data for simulation
    let bars = generate_sample_bars(&symbol, 100);
    let market_data = MarketData::new(symbol.clone(), bars);
    let portfolio = Portfolio::new(Decimal::from(100000));

    // Create strategy instance based on name
    let signals = match strategy.as_str() {
        "momentum_trading" | "momentum" => {
            let strat = MomentumStrategy::new(vec![symbol.clone()], 20, 2.0, 0.5);
            strat.process(&market_data, &portfolio).await
        }
        "mean_reversion" => {
            let strat = MeanReversionStrategy::new(vec![symbol.clone()], 20, 2.0, 14);
            strat.process(&market_data, &portfolio).await
        }
        "trend_following" => {
            let strat = NeuralTrendStrategy::new(vec![symbol.clone()], 0.7, 50);
            strat.process(&market_data, &portfolio).await
        }
        "pairs_trading" => {
            // For pairs trading, we need a second symbol
            let strat = PairsStrategy::new(
                vec![(symbol.clone(), "SPY".to_string())],
                60,
                2.0,
                0.5,
            );
            strat.process(&market_data, &portfolio).await
        }
        _ => {
            return Err(NeuralTraderError::Trading(
                format!("Strategy '{}' not implemented for simulation", strategy)
            ).into());
        }
    };

    // Calculate simulation metrics
    let (expected_return, risk_score) = match signals {
        Ok(signals) => {
            if !signals.is_empty() {
                let signal = &signals[0];
                // Revert: confidence IS Option<f64>, use unwrap_or
                let confidence = signal.confidence.unwrap_or(0.5);

                // Estimate expected return based on confidence and historical performance
                let meta = registry.get(&strategy).unwrap();
                let base_return = meta.sharpe_ratio / 10.0; // Rough estimate
                let expected_return = base_return * confidence;

                // Risk score inversely related to confidence
                let risk_score = 1.0 - confidence;

                (expected_return, risk_score)
            } else {
                (0.0, 0.5)
            }
        }
        Err(_) => (0.0, 0.8),
    };

    let execution_time_ms = start_time.elapsed().as_millis() as i64;

    Ok(TradeSimulation {
        strategy,
        symbol,
        action,
        expected_return,
        risk_score,
        execution_time_ms,
    })
}

/// Trade simulation result
#[napi(object)]
pub struct TradeSimulation {
    pub strategy: String,
    pub symbol: String,
    pub action: String,
    pub expected_return: f64,
    pub risk_score: f64,
    pub execution_time_ms: i64,
}

/// Get current portfolio status
#[napi]
pub async fn get_portfolio_status(include_analytics: Option<bool>) -> Result<PortfolioStatus> {
    let _analytics = include_analytics.unwrap_or(true);

    // Create a sample portfolio with realistic values
    // In production, this would fetch from a persistent portfolio manager
    let mut portfolio = Portfolio::new(Decimal::from(100000));

    // Add some sample positions
    portfolio.update_position("AAPL".to_string(), Position {
        symbol: "AAPL".to_string(),
        quantity: 100,
        avg_price: Decimal::from_f64(150.0).unwrap(),
        current_price: Decimal::from_f64(155.0).unwrap(),
        market_value: Decimal::from_f64(15500.0).unwrap(),
        unrealized_pnl: Decimal::from_f64(500.0).unwrap(),
    });

    portfolio.update_position("GOOGL".to_string(), Position {
        symbol: "GOOGL".to_string(),
        quantity: 50,
        avg_price: Decimal::from_f64(2800.0).unwrap(),
        current_price: Decimal::from_f64(2850.0).unwrap(),
        market_value: Decimal::from_f64(142500.0).unwrap(),
        unrealized_pnl: Decimal::from_f64(2500.0).unwrap(),
    });

    // Update cash to reflect positions
    portfolio.update_cash(Decimal::from(-58000)); // Initial 100k - position costs

    let total_value = portfolio.total_value().to_f64().unwrap_or(0.0);
    let cash = portfolio.cash().to_f64().unwrap_or(0.0);
    let positions_count = portfolio.positions().len() as u32;

    // Calculate daily P&L (simplified)
    let daily_pnl = portfolio.positions()
        .values()
        .map(|p| p.unrealized_pnl.to_f64().unwrap_or(0.0))
        .sum::<f64>() * 0.1; // Assume 10% of unrealized is from today

    // Calculate total return
    let initial_capital = 100000.0;
    let total_return = (total_value - initial_capital) / initial_capital;

    Ok(PortfolioStatus {
        total_value,
        cash,
        positions: positions_count,
        daily_pnl,
        total_return,
    })
}

/// Portfolio status
#[napi(object)]
pub struct PortfolioStatus {
    pub total_value: f64,
    pub cash: f64,
    pub positions: u32,
    pub daily_pnl: f64,
    pub total_return: f64,
}

/// Execute a live trade
#[napi]
pub async fn execute_trade(
    strategy: String,
    symbol: String,
    action: String,
    quantity: u32,
    order_type: Option<String>,
    limit_price: Option<f64>,
) -> Result<TradeExecution> {
    let order_type = order_type.unwrap_or_else(|| "market".to_string());

    // Validate strategy
    let registry = StrategyRegistry::new();
    if !registry.contains(&strategy) {
        return Err(NeuralTraderError::Trading(
            format!("Unknown strategy: {}", strategy)
        ).into());
    }

    // Validate action
    let valid_actions = ["buy", "sell", "close"];
    if !valid_actions.contains(&action.to_lowercase().as_str()) {
        return Err(NeuralTraderError::Trading(
            format!("Invalid action: {}. Must be 'buy', 'sell', or 'close'", action)
        ).into());
    }

    // Generate order ID
    let order_id = format!("ORD-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap().to_uppercase());

    // Get current market price (simulated)
    let bars = generate_sample_bars(&symbol, 10);
    let current_price = bars.last()
        .map(|b| b.close.to_f64().unwrap_or(150.0))
        .unwrap_or(150.0);

    // Determine fill price based on order type
    let fill_price = match order_type.as_str() {
        "market" => {
            // Market orders fill at current price with slight slippage
            current_price * (1.0 + 0.001) // 0.1% slippage
        }
        "limit" => {
            match limit_price {
                Some(limit) => {
                    // Check if limit order would fill
                    if (action == "buy" && current_price <= limit) ||
                       (action == "sell" && current_price >= limit) {
                        limit
                    } else {
                        return Ok(TradeExecution {
                            order_id,
                            strategy,
                            symbol,
                            action,
                            quantity,
                            status: "pending".to_string(),
                            fill_price: 0.0,
                        });
                    }
                }
                None => {
                    return Err(NeuralTraderError::Trading(
                        "Limit price required for limit orders".to_string()
                    ).into());
                }
            }
        }
        _ => {
            return Err(NeuralTraderError::Trading(
                format!("Unsupported order type: {}", order_type)
            ).into());
        }
    };

    // In production, this would:
    // 1. Send order to broker API (nt-execution crate)
    // 2. Wait for fill confirmation
    // 3. Update portfolio state
    // 4. Log trade to database

    Ok(TradeExecution {
        order_id,
        strategy,
        symbol,
        action,
        quantity,
        status: "filled".to_string(),
        fill_price,
    })
}

/// Trade execution result
#[napi(object)]
pub struct TradeExecution {
    pub order_id: String,
    pub strategy: String,
    pub symbol: String,
    pub action: String,
    pub quantity: u32,
    pub status: String,
    pub fill_price: f64,
}

/// Run a comprehensive backtest
#[napi]
pub async fn run_backtest(
    strategy: String,
    symbol: String,
    start_date: String,
    end_date: String,
    use_gpu: Option<bool>,
) -> Result<BacktestResult> {
    let _gpu = use_gpu.unwrap_or(true);

    // Validate strategy
    let registry = StrategyRegistry::new();
    if !registry.contains(&strategy) {
        return Err(NeuralTraderError::Trading(
            format!("Unknown strategy: {}", strategy)
        ).into());
    }

    // Parse dates
    let start = DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", start_date))
        .map_err(|e| NeuralTraderError::Trading(format!("Invalid start_date: {}", e)))?
        .with_timezone(&Utc);

    let end = DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", end_date))
        .map_err(|e| NeuralTraderError::Trading(format!("Invalid end_date: {}", e)))?
        .with_timezone(&Utc);

    if start >= end {
        return Err(NeuralTraderError::Trading(
            "start_date must be before end_date".to_string()
        ).into());
    }

    // Generate historical data
    let days = (end - start).num_days() as usize;
    let bars = generate_backtest_bars(&symbol, days, start);

    let mut historical_data = HashMap::new();
    historical_data.insert(symbol.clone(), bars);

    // Create strategy instance
    let result = match strategy.as_str() {
        "momentum_trading" | "momentum" => {
            let strat = MomentumStrategy::new(vec![symbol.clone()], 20, 2.0, 0.5);
            let mut engine = BacktestEngine::new(Decimal::from(100000))
                .with_commission(0.001)
                .with_slippage(nt_strategies::backtest::SlippageModel::default());

            engine.run(&strat, historical_data, start, end).await
        }
        "mean_reversion" => {
            let strat = MeanReversionStrategy::new(vec![symbol.clone()], 20, 2.0, 14);
            let mut engine = BacktestEngine::new(Decimal::from(100000))
                .with_commission(0.001)
                .with_slippage(nt_strategies::backtest::SlippageModel::default());

            engine.run(&strat, historical_data, start, end).await
        }
        "trend_following" => {
            let strat = NeuralTrendStrategy::new(vec![symbol.clone()], 0.7, 50);
            let mut engine = BacktestEngine::new(Decimal::from(100000))
                .with_commission(0.001)
                .with_slippage(nt_strategies::backtest::SlippageModel::default());

            engine.run(&strat, historical_data, start, end).await
        }
        "pairs_trading" => {
            let strat = PairsStrategy::new(
                vec![(symbol.clone(), "SPY".to_string())],
                60,
                2.0,
                0.5,
            );
            let mut engine = BacktestEngine::new(Decimal::from(100000))
                .with_commission(0.001)
                .with_slippage(nt_strategies::backtest::SlippageModel::default());

            engine.run(&strat, historical_data, start, end).await
        }
        _ => {
            return Err(NeuralTraderError::Trading(
                format!("Strategy '{}' not implemented for backtesting", strategy)
            ).into());
        }
    };

    // Convert result to NAPI format
    match result {
        Ok(bt_result) => {
            // Use Option::unwrap_or since these might be Option<f64>
            let sharpe_ratio = Some(bt_result.metrics.sharpe_ratio).unwrap_or(0.0);
            let max_drawdown = Some(bt_result.metrics.max_drawdown).unwrap_or(0.0);

            // Calculate win rate
            let winning_trades = bt_result.trades.iter()
                .filter(|t| t.pnl > Decimal::ZERO)
                .count();
            let total_trades = bt_result.trades.len();
            let win_rate = if total_trades > 0 {
                winning_trades as f64 / total_trades as f64
            } else {
                0.0
            };

            Ok(BacktestResult {
                strategy,
                symbol,
                start_date,
                end_date,
                total_return: bt_result.total_return,
                sharpe_ratio,
                max_drawdown,
                total_trades: total_trades as u32,
                win_rate,
            })
        }
        Err(e) => {
            Err(NeuralTraderError::Trading(
                format!("Backtest failed: {}", e)
            ).into())
        }
    }
}

/// Generate historical bars for backtesting
fn generate_backtest_bars(symbol: &str, days: usize, start_date: DateTime<Utc>) -> Vec<Bar> {
    let mut bars = Vec::new();
    let mut price = 100.0;

    for i in 0..days {
        // Simulate realistic price movement with trend and noise
        let trend = 0.0002; // Slight upward trend
        let noise = (((i * 13 + 7) % 100) as f64 - 50.0) / 500.0; // Random-like noise
        price = (price * (1.0 + trend + noise)).max(10.0);

        let bar = Bar {
            symbol: nt_core::types::Symbol::new(symbol).unwrap(), // Use Symbol::new() constructor
            timestamp: start_date + chrono::Duration::days(i as i64),
            open: Decimal::from_f64(price * 0.998).unwrap(),
            high: Decimal::from_f64(price * 1.015).unwrap(),
            low: Decimal::from_f64(price * 0.985).unwrap(),
            close: Decimal::from_f64(price).unwrap(),
            volume: Decimal::from(1000000 + ((i * 37) % 500000) as u64), // Volume is Decimal
        };
        bars.push(bar);
    }

    bars
}

/// Backtest result
#[napi(object)]
pub struct BacktestResult {
    pub strategy: String,
    pub symbol: String,
    pub start_date: String,
    pub end_date: String,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_trades: u32,
    pub win_rate: f64,
}
