//! Performance and risk metrics calculation

use crate::{Result, BacktestError};
use crate::portfolio::Trade;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance metrics for backtesting results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return as percentage
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Win rate as percentage
    pub win_rate: f64,
    /// Average winning trade
    pub avg_winning_trade: f64,
    /// Average losing trade
    pub avg_losing_trade: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Best trade
    pub best_trade: f64,
    /// Worst trade
    pub worst_trade: f64,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk (95% confidence)
    pub var_95: f64,
    /// Conditional Value at Risk (95%)
    pub cvar_95: f64,
    /// Beta (market correlation)
    pub beta: f64,
    /// Alpha (excess return)
    pub alpha: f64,
    /// Information ratio
    pub information_ratio: f64,
    /// Tracking error
    pub tracking_error: f64,
    /// Downside deviation
    pub downside_deviation: f64,
    /// Upside potential
    pub upside_potential: f64,
}

/// Drawdown metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownMetrics {
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Maximum drawdown duration (days)
    pub max_drawdown_duration: i64,
    /// Average drawdown
    pub avg_drawdown: f64,
    /// Average drawdown duration (days)
    pub avg_drawdown_duration: f64,
    /// Recovery factor
    pub recovery_factor: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Drawdown periods
    pub drawdown_periods: Vec<DrawdownPeriod>,
}

/// Individual drawdown period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub peak_value: f64,
    pub trough_value: f64,
    pub drawdown_pct: f64,
    pub duration_days: i64,
    pub recovered: bool,
}

impl PerformanceMetrics {
    /// Calculate performance metrics from trade history and portfolio values
    pub fn calculate(
        initial_capital: f64,
        trades: &[Trade],
        portfolio_history: &[(DateTime<Utc>, f64)],
    ) -> Result<Self> {
        if portfolio_history.is_empty() {
            return Err(BacktestError::Data("Empty portfolio history".to_string()));
        }
        
        let final_value = portfolio_history.last().unwrap().1;
        let total_return = (final_value - initial_capital) / initial_capital;
        
        // Calculate returns series
        let returns = Self::calculate_returns(portfolio_history);
        
        // Time period calculations
        let start_date = portfolio_history.first().unwrap().0;
        let end_date = portfolio_history.last().unwrap().0;
        let duration = end_date.signed_duration_since(start_date);
        let years = duration.num_days() as f64 / 365.25;
        
        // Basic metrics
        let annualized_return = (1.0 + total_return).powf(1.0 / years) - 1.0;
        let volatility = Self::calculate_volatility(&returns) * (252.0_f64).sqrt(); // Annualized
        let sharpe_ratio = if volatility > 0.0 { annualized_return / volatility } else { 0.0 };
        
        // Sortino ratio (using downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_deviation = if !downside_returns.is_empty() {
            (downside_returns.iter().map(|r| r * r).sum::<f64>() / downside_returns.len() as f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_deviation > 0.0 {
            annualized_return / (downside_deviation * (252.0_f64).sqrt())
        } else {
            0.0
        };
        
        // Trade analysis
        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count();
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };
        
        let winning_pnl: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let losing_pnl: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl).sum();
        
        let avg_winning_trade = if winning_trades > 0 {
            winning_pnl / winning_trades as f64
        } else {
            0.0
        };
        
        let avg_losing_trade = if losing_trades > 0 {
            losing_pnl / losing_trades as f64
        } else {
            0.0
        };
        
        let profit_factor = if losing_pnl.abs() > 0.0 {
            winning_pnl / losing_pnl.abs()
        } else {
            f64::INFINITY
        };
        
        let best_trade = trades.iter().map(|t| t.pnl).fold(f64::NEG_INFINITY, f64::max);
        let worst_trade = trades.iter().map(|t| t.pnl).fold(f64::INFINITY, f64::min);
        
        // Calculate maximum drawdown for Calmar ratio
        let max_drawdown = Self::calculate_max_drawdown(portfolio_history);
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };
        
        Ok(Self {
            total_return: total_return * 100.0,
            annualized_return: annualized_return * 100.0,
            volatility: volatility * 100.0,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            avg_winning_trade,
            avg_losing_trade,
            profit_factor,
            best_trade,
            worst_trade,
        })
    }
    
    fn calculate_returns(portfolio_history: &[(DateTime<Utc>, f64)]) -> Vec<f64> {
        portfolio_history
            .windows(2)
            .map(|window| (window[1].1 - window[0].1) / window[0].1)
            .collect()
    }
    
    fn calculate_volatility(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    fn calculate_max_drawdown(portfolio_history: &[(DateTime<Utc>, f64)]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = portfolio_history[0].1;
        
        for (_, value) in portfolio_history.iter().skip(1) {
            if *value > peak {
                peak = *value;
            }
            
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
}

impl RiskMetrics {
    /// Calculate risk metrics from portfolio history
    pub fn calculate(portfolio_history: &[(DateTime<Utc>, f64)]) -> Result<Self> {
        let returns = PerformanceMetrics::calculate_returns(portfolio_history);
        
        if returns.is_empty() {
            return Ok(Self::default());
        }
        
        // Sort returns for VaR calculation
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Value at Risk (95% confidence)
        let var_index = (sorted_returns.len() as f64 * 0.05) as usize;
        let var_95 = if var_index < sorted_returns.len() {
            -sorted_returns[var_index] * 100.0
        } else {
            0.0
        };
        
        // Conditional VaR (Expected Shortfall)
        let cvar_returns: Vec<f64> = sorted_returns.iter().take(var_index + 1).cloned().collect();
        let cvar_95 = if !cvar_returns.is_empty() {
            -(cvar_returns.iter().sum::<f64>() / cvar_returns.len() as f64) * 100.0
        } else {
            0.0
        };
        
        // Downside deviation
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_deviation = if !negative_returns.is_empty() {
            let mean_negative = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
            (negative_returns.iter().map(|r| (r - mean_negative).powi(2)).sum::<f64>() 
             / negative_returns.len() as f64).sqrt() * 100.0
        } else {
            0.0
        };
        
        // Upside potential
        let positive_returns: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        let upside_potential = if !positive_returns.is_empty() {
            positive_returns.iter().sum::<f64>() / positive_returns.len() as f64 * 100.0
        } else {
            0.0
        };
        
        Ok(Self {
            var_95,
            cvar_95,
            beta: 0.0, // Would need market data to calculate
            alpha: 0.0, // Would need benchmark to calculate
            information_ratio: 0.0,
            tracking_error: 0.0,
            downside_deviation,
            upside_potential,
        })
    }
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var_95: 0.0,
            cvar_95: 0.0,
            beta: 0.0,
            alpha: 0.0,
            information_ratio: 0.0,
            tracking_error: 0.0,
            downside_deviation: 0.0,
            upside_potential: 0.0,
        }
    }
}

impl DrawdownMetrics {
    /// Calculate drawdown metrics from portfolio history
    pub fn calculate(portfolio_history: &[(DateTime<Utc>, f64)]) -> Result<Self> {
        let mut drawdown_periods = Vec::new();
        let mut max_drawdown = 0.0;
        let mut max_drawdown_duration = 0i64;
        let mut total_drawdown = 0.0;
        let mut total_duration = 0f64;
        
        let mut peak = portfolio_history[0].1;
        let mut peak_date = portfolio_history[0].0;
        let mut in_drawdown = false;
        let mut drawdown_start = portfolio_history[0].0;
        let mut trough_value = peak;
        
        for (date, value) in portfolio_history.iter().skip(1) {
            if *value > peak {
                // New peak - end any existing drawdown
                if in_drawdown {
                    let duration = date.signed_duration_since(drawdown_start).num_days();
                    let drawdown_pct = (peak - trough_value) / peak * 100.0;
                    
                    drawdown_periods.push(DrawdownPeriod {
                        start_date: drawdown_start,
                        end_date: Some(*date),
                        peak_value: peak,
                        trough_value,
                        drawdown_pct,
                        duration_days: duration,
                        recovered: true,
                    });
                    
                    total_drawdown += drawdown_pct;
                    total_duration += duration as f64;
                    
                    if duration > max_drawdown_duration {
                        max_drawdown_duration = duration;
                    }
                    
                    in_drawdown = false;
                }
                
                peak = *value;
                peak_date = *date;
                trough_value = *value;
            } else {
                // Value below peak
                if !in_drawdown {
                    in_drawdown = true;
                    drawdown_start = peak_date;
                }
                
                if *value < trough_value {
                    trough_value = *value;
                }
                
                let current_drawdown = (peak - value) / peak * 100.0;
                if current_drawdown > max_drawdown {
                    max_drawdown = current_drawdown;
                }
            }
        }
        
        // Handle ongoing drawdown
        let current_drawdown = if in_drawdown {
            (peak - trough_value) / peak * 100.0
        } else {
            0.0
        };
        
        if in_drawdown {
            let last_date = portfolio_history.last().unwrap().0;
            let duration = last_date.signed_duration_since(drawdown_start).num_days();
            
            drawdown_periods.push(DrawdownPeriod {
                start_date: drawdown_start,
                end_date: None,
                peak_value: peak,
                trough_value,
                drawdown_pct: current_drawdown,
                duration_days: duration,
                recovered: false,
            });
        }
        
        let avg_drawdown = if !drawdown_periods.is_empty() {
            total_drawdown / drawdown_periods.len() as f64
        } else {
            0.0
        };
        
        let avg_drawdown_duration = if !drawdown_periods.is_empty() {
            total_duration / drawdown_periods.len() as f64
        } else {
            0.0
        };
        
        let final_value = portfolio_history.last().unwrap().1;
        let initial_value = portfolio_history.first().unwrap().1;
        let total_return = (final_value - initial_value) / initial_value * 100.0;
        
        let recovery_factor = if max_drawdown > 0.0 {
            total_return / max_drawdown
        } else {
            0.0
        };
        
        Ok(Self {
            max_drawdown,
            max_drawdown_duration,
            avg_drawdown,
            avg_drawdown_duration,
            recovery_factor,
            current_drawdown,
            drawdown_periods,
        })
    }
}