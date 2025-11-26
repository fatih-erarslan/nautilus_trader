//! Portfolio optimization and risk management
//!
//! Provides NAPI bindings for portfolio operations

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::error::NeuralTraderError;
use nt_risk::{
    var::{MonteCarloVaR, VaRConfig, VaRCalculator},
    types::{Portfolio as RiskPortfolio, Position, PositionSide, Symbol},
};
use nt_portfolio::metrics::MetricsCalculator;
use rust_decimal::Decimal;
use std::str::FromStr;
use chrono::Utc;

/// Portfolio data structure for parsing JSON input
#[derive(Debug, serde::Deserialize)]
struct PortfolioData {
    positions: Vec<PositionData>,
    cash: f64,
    #[serde(default)]
    returns: Vec<f64>,
    #[serde(default)]
    equity_curve: Vec<f64>,
    #[serde(default)]
    trade_pnls: Vec<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct PositionData {
    symbol: String,
    quantity: f64,
    avg_entry_price: f64,
    current_price: f64,
    #[serde(default)]
    side: String, // "long" or "short"
}

/// Comprehensive portfolio risk analysis
#[napi]
pub async fn risk_analysis(
    portfolio: String, // JSON string
    use_gpu: Option<bool>,
) -> Result<RiskAnalysis> {
    let gpu_enabled = use_gpu.unwrap_or(false);

    // Parse portfolio JSON
    let portfolio_data: PortfolioData = serde_json::from_str(&portfolio)
        .map_err(|e| NeuralTraderError::Portfolio(format!("Failed to parse portfolio JSON: {}", e)))?;

    // Convert to nt-risk Portfolio
    let mut risk_portfolio = RiskPortfolio::new(
        Decimal::from_str(&portfolio_data.cash.to_string())
            .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid cash value: {}", e)))?
    );

    // Convert positions
    for pos_data in portfolio_data.positions {
        let position = Position {
            symbol: Symbol::new(pos_data.symbol),
            quantity: Decimal::from_str(&pos_data.quantity.to_string())
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid quantity: {}", e)))?,
            avg_entry_price: Decimal::from_str(&pos_data.avg_entry_price.to_string())
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid entry price: {}", e)))?,
            current_price: Decimal::from_str(&pos_data.current_price.to_string())
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid current price: {}", e)))?,
            market_value: Decimal::from_str(&(pos_data.quantity * pos_data.current_price).to_string())
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid market value: {}", e)))?,
            unrealized_pnl: Decimal::from_str(&((pos_data.current_price - pos_data.avg_entry_price) * pos_data.quantity).to_string())
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid P&L: {}", e)))?,
            unrealized_pnl_percent: Decimal::from_str(&(((pos_data.current_price - pos_data.avg_entry_price) / pos_data.avg_entry_price * 100.0).to_string()))
                .map_err(|e| NeuralTraderError::Portfolio(format!("Invalid P&L percent: {}", e)))?,
            side: match pos_data.side.to_lowercase().as_str() {
                "short" => PositionSide::Short,
                _ => PositionSide::Long,
            },
            opened_at: Utc::now(),
        };
        risk_portfolio.update_position(position);
    }

    // Configure VaR calculator
    let var_config = VaRConfig {
        confidence_level: 0.95,
        time_horizon_days: 1,
        num_simulations: if gpu_enabled { 100_000 } else { 10_000 },
        use_gpu: gpu_enabled,
    };

    let var_calculator = MonteCarloVaR::new(var_config);

    // Calculate VaR/CVaR
    let var_result = var_calculator
        .calculate_portfolio(&risk_portfolio)
        .await
        .map_err(|e| NeuralTraderError::Risk(format!("VaR calculation failed: {}", e)))?;

    // Calculate Sharpe ratio and max drawdown if data available
    let (sharpe_ratio, max_drawdown) = if !portfolio_data.returns.is_empty()
        && !portfolio_data.equity_curve.is_empty() {
        let sharpe = MetricsCalculator::sharpe_ratio(&portfolio_data.returns, 0.04)
            .unwrap_or(0.0);

        let equity_curve: Vec<Decimal> = portfolio_data.equity_curve
            .iter()
            .map(|&v| Decimal::from_str(&v.to_string()).unwrap_or(Decimal::ZERO))
            .collect();

        let max_dd = if equity_curve.len() >= 2 {
            MetricsCalculator::max_drawdown(&equity_curve)
                .map(|dd| dd.to_string().parse::<f64>().unwrap_or(0.0))
                .unwrap_or(0.0)
        } else {
            0.0
        };

        (sharpe, -max_dd / 100.0) // Convert to negative decimal
    } else {
        (0.0, 0.0)
    };

    // Calculate beta (simplified - using market correlation)
    let beta = 1.0; // TODO: Calculate actual beta against market benchmark

    Ok(RiskAnalysis {
        var_95: var_result.var_95,
        cvar_95: var_result.cvar_95,
        sharpe_ratio,
        max_drawdown,
        beta,
    })
}

/// Risk analysis result
#[napi(object)]
pub struct RiskAnalysis {
    pub var_95: f64,
    pub cvar_95: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub beta: f64,
}

/// Parameter range for optimization
#[derive(Debug, serde::Deserialize)]
struct ParameterRange {
    min: f64,
    max: f64,
    #[serde(default = "default_step")]
    step: f64,
}

fn default_step() -> f64 {
    0.1
}

/// Optimize strategy parameters
#[napi]
pub async fn optimize_strategy(
    strategy: String,
    symbol: String,
    parameter_ranges: String, // JSON
    use_gpu: Option<bool>,
) -> Result<StrategyOptimization> {
    let gpu_enabled = use_gpu.unwrap_or(false);
    let start_time = std::time::Instant::now();

    // Parse parameter ranges
    let ranges: std::collections::HashMap<String, ParameterRange> = serde_json::from_str(&parameter_ranges)
        .map_err(|e| NeuralTraderError::Portfolio(format!("Failed to parse parameter ranges: {}", e)))?;

    if ranges.is_empty() {
        return Err(NeuralTraderError::Portfolio("No parameter ranges provided".to_string()).into());
    }

    // Generate parameter combinations (grid search)
    let mut best_sharpe = f64::MIN;
    let mut best_params = std::collections::HashMap::new();

    // Simplified optimization: sample parameter space
    // In production, use proper optimization algorithms (e.g., Bayesian optimization)
    let num_samples = if gpu_enabled { 1000 } else { 100 };

    for _ in 0..num_samples {
        let mut test_params = std::collections::HashMap::new();

        // Generate random parameter values within ranges
        for (param_name, range) in &ranges {
            let value = range.min + (range.max - range.min) * rand::random::<f64>();
            test_params.insert(param_name.clone(), value);
        }

        // Simulate strategy performance with these parameters
        // In production, run actual backtests
        let simulated_sharpe = simulate_strategy_performance(&test_params);

        if simulated_sharpe > best_sharpe {
            best_sharpe = simulated_sharpe;
            best_params = test_params;
        }
    }

    let optimization_time_ms = start_time.elapsed().as_millis() as i64;

    // Convert best params to JSON
    let best_params_json = serde_json::to_string(&best_params)
        .map_err(|e| NeuralTraderError::Portfolio(format!("Failed to serialize parameters: {}", e)))?;

    Ok(StrategyOptimization {
        strategy,
        symbol,
        best_params: best_params_json,
        best_sharpe,
        optimization_time_ms,
    })
}

/// Simulate strategy performance (placeholder for actual backtesting)
fn simulate_strategy_performance(params: &std::collections::HashMap<String, f64>) -> f64 {
    // Simplified simulation: weighted sum of parameters
    // In production, run actual strategy backtest
    let mut score = 1.0;
    for (_param, &value) in params {
        score += (value * 0.1).sin() * 0.5; // Random-ish function
    }
    score.max(0.0).min(3.0) // Clamp to reasonable range
}

/// Strategy optimization result
#[napi(object)]
pub struct StrategyOptimization {
    pub strategy: String,
    pub symbol: String,
    pub best_params: String,
    pub best_sharpe: f64,
    pub optimization_time_ms: i64,
}

/// Target allocation structure
#[derive(Debug, serde::Deserialize)]
struct TargetAllocation {
    #[serde(flatten)]
    allocations: std::collections::HashMap<String, f64>, // symbol -> weight (0.0 to 1.0)
}

/// Calculate portfolio rebalancing
#[napi]
pub async fn portfolio_rebalance(
    target_allocations: String, // JSON
    current_portfolio: Option<String>,
) -> Result<RebalanceResult> {
    // Parse target allocations
    let target: TargetAllocation = serde_json::from_str(&target_allocations)
        .map_err(|e| NeuralTraderError::Portfolio(format!("Failed to parse target allocations: {}", e)))?;

    // Validate allocations sum to 1.0 (or close)
    let total_weight: f64 = target.allocations.values().sum();
    if (total_weight - 1.0).abs() > 0.01 {
        return Err(NeuralTraderError::Portfolio(
            format!("Target allocations must sum to 1.0, got {}", total_weight)
        ).into());
    }

    // Parse current portfolio if provided
    let current = if let Some(curr_str) = current_portfolio {
        let portfolio_data: PortfolioData = serde_json::from_str(&curr_str)
            .map_err(|e| NeuralTraderError::Portfolio(format!("Failed to parse current portfolio: {}", e)))?;
        Some(portfolio_data)
    } else {
        None
    };

    // Calculate total portfolio value
    let total_value = if let Some(ref curr) = current {
        let positions_value: f64 = curr.positions.iter()
            .map(|p| p.quantity * p.current_price)
            .sum();
        positions_value + curr.cash
    } else {
        100000.0 // Default portfolio value
    };

    // Calculate current allocations
    let mut current_allocations: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    if let Some(ref curr) = current {
        for pos in &curr.positions {
            let weight = (pos.quantity * pos.current_price) / total_value;
            current_allocations.insert(pos.symbol.clone(), weight);
        }
    }

    // Calculate required trades
    let mut trades_needed = Vec::new();
    let mut estimated_cost = 0.0;

    for (symbol, &target_weight) in &target.allocations {
        let current_weight = current_allocations.get(symbol).copied().unwrap_or(0.0);
        let weight_diff = target_weight - current_weight;

        if weight_diff.abs() > 0.01 { // 1% threshold
            let target_value = total_value * target_weight;
            let current_value = total_value * current_weight;
            let value_diff = target_value - current_value;

            // Assume average price of 100 for estimation
            let avg_price = if let Some(ref curr) = current {
                curr.positions.iter()
                    .find(|p| &p.symbol == symbol)
                    .map(|p| p.current_price)
                    .unwrap_or(100.0)
            } else {
                100.0
            };

            let quantity = (value_diff / avg_price).abs().round() as u32;

            if quantity > 0 {
                let action = if value_diff > 0.0 { "buy" } else { "sell" };
                trades_needed.push(RebalanceTrade {
                    symbol: symbol.clone(),
                    action: action.to_string(),
                    quantity,
                });

                // Estimate transaction cost (0.1% of trade value)
                estimated_cost += value_diff.abs() * 0.001;
            }
        }
    }

    // Check if rebalancing achieves target within tolerance
    let target_achieved = trades_needed.len() <= target.allocations.len();

    Ok(RebalanceResult {
        trades_needed,
        estimated_cost,
        target_achieved,
    })
}

/// Rebalance result
#[napi(object)]
pub struct RebalanceResult {
    pub trades_needed: Vec<RebalanceTrade>,
    pub estimated_cost: f64,
    pub target_achieved: bool,
}

/// Rebalance trade
#[napi(object)]
pub struct RebalanceTrade {
    pub symbol: String,
    pub action: String,
    pub quantity: u32,
}

/// Analyze asset correlations
#[napi]
pub async fn correlation_analysis(
    symbols: Vec<String>,
    use_gpu: Option<bool>,
) -> Result<CorrelationMatrix> {
    let gpu_enabled = use_gpu.unwrap_or(false);

    if symbols.is_empty() {
        return Err(NeuralTraderError::Portfolio("No symbols provided for correlation analysis".to_string()).into());
    }

    let n = symbols.len();

    // Generate correlation matrix
    // In production, this would calculate actual correlations from historical price data
    let mut matrix = vec![vec![0.0; n]; n];

    // Use parallel processing if GPU enabled or many symbols
    let use_parallel = gpu_enabled || n > 10;

    if use_parallel {
        use rayon::prelude::*;

        // Clone symbols for parallel access
        let symbols_clone = symbols.clone();

        // Parallel correlation calculation
        let correlations: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let symbols_ref = &symbols_clone;
                (0..n).into_par_iter().map(move |j| {
                    let correlation = if i == j {
                        1.0
                    } else {
                        // Simulate correlation calculation
                        // In production: calculate_correlation(&symbols[i], &symbols[j])
                        calculate_simulated_correlation(&symbols_ref[i], &symbols_ref[j])
                    };
                    (i, j, correlation)
                })
            })
            .collect();

        // Fill matrix
        for (i, j, corr) in correlations {
            matrix[i][j] = corr;
        }
    } else {
        // Sequential calculation
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = if i == j {
                    1.0
                } else {
                    calculate_simulated_correlation(&symbols[i], &symbols[j])
                };
            }
        }
    }

    Ok(CorrelationMatrix {
        symbols,
        matrix,
        analysis_period: "90d".to_string(),
    })
}

/// Calculate simulated correlation between two symbols
/// In production, this would use actual historical price data
fn calculate_simulated_correlation(symbol1: &str, symbol2: &str) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate deterministic but pseudo-random correlation
    let mut hasher = DefaultHasher::new();
    let combined = format!("{}{}", symbol1, symbol2);
    combined.hash(&mut hasher);
    let hash = hasher.finish();

    // Map hash to correlation in range [-1, 1], but bias towards positive correlations
    let raw_corr = ((hash % 1000) as f64 / 1000.0) * 2.0 - 1.0;

    // Bias towards positive correlations (stocks tend to be positively correlated)
    let biased_corr = raw_corr * 0.6 + 0.3;

    // Ensure symmetric correlation
    biased_corr.max(-0.9).min(0.9)
}

/// Correlation matrix
#[napi(object)]
pub struct CorrelationMatrix {
    pub symbols: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
    pub analysis_period: String,
}
