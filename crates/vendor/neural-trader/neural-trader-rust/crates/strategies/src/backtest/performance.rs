//! Performance Metrics and Analysis
//!
//! Comprehensive performance attribution including:
//! - Returns (total, annual, monthly)
//! - Risk metrics (Sharpe, Sortino, max drawdown)
//! - Win rate and profit factor
//! - Trade statistics

use super::engine::{Trade, EquityPoint, TradeSide};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Serialize, Deserialize};

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Return metrics
    pub total_return: f64,
    pub annual_return: f64,
    pub monthly_returns: Vec<f64>,

    // Risk metrics
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: i64,
    pub volatility: f64,

    // Trade statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,

    // Exposure
    pub average_exposure: f64,
    pub max_exposure: f64,
}

impl PerformanceMetrics {
    /// Calculate all metrics
    pub fn calculate(equity_curve: &[EquityPoint], trades: &[Trade]) -> Self {
        let returns = Self::calculate_returns(equity_curve);
        let (sharpe, sortino, volatility) = Self::calculate_risk_metrics(&returns);
        let (max_dd, max_dd_duration) = Self::calculate_max_drawdown(equity_curve);
        let trade_stats = Self::calculate_trade_statistics(trades);

        Self {
            total_return: Self::calculate_total_return(equity_curve),
            annual_return: Self::calculate_annual_return(equity_curve),
            monthly_returns: Self::calculate_monthly_returns(&returns),
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            max_drawdown_duration_days: max_dd_duration,
            volatility,
            total_trades: trades.len(),
            winning_trades: trade_stats.0,
            losing_trades: trade_stats.1,
            win_rate: trade_stats.2,
            profit_factor: trade_stats.3,
            average_win: trade_stats.4,
            average_loss: trade_stats.5,
            largest_win: trade_stats.6,
            largest_loss: trade_stats.7,
            average_exposure: Self::calculate_average_exposure(equity_curve),
            max_exposure: Self::calculate_max_exposure(equity_curve),
        }
    }

    /// Calculate daily returns
    fn calculate_returns(equity_curve: &[EquityPoint]) -> Vec<f64> {
        equity_curve
            .windows(2)
            .map(|window| {
                let prev = window[0].equity.to_f64().unwrap();
                let curr = window[1].equity.to_f64().unwrap();
                (curr - prev) / prev
            })
            .collect()
    }

    /// Calculate total return
    fn calculate_total_return(equity_curve: &[EquityPoint]) -> f64 {
        if equity_curve.len() < 2 {
            return 0.0;
        }

        let initial = equity_curve.first().unwrap().equity.to_f64().unwrap();
        let final_equity = equity_curve.last().unwrap().equity.to_f64().unwrap();

        (final_equity - initial) / initial
    }

    /// Calculate annualized return
    fn calculate_annual_return(equity_curve: &[EquityPoint]) -> f64 {
        if equity_curve.len() < 2 {
            return 0.0;
        }

        let total_return = Self::calculate_total_return(equity_curve);

        let start = equity_curve.first().unwrap().timestamp;
        let end = equity_curve.last().unwrap().timestamp;
        let years = (end - start).num_days() as f64 / 365.25;

        if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        }
    }

    /// Calculate monthly returns
    fn calculate_monthly_returns(returns: &[f64]) -> Vec<f64> {
        // Group by month and compound
        // Simplified: just return daily returns for now
        returns.to_vec()
    }

    /// Calculate Sharpe and Sortino ratios
    fn calculate_risk_metrics(returns: &[f64]) -> (f64, f64, f64) {
        if returns.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        // Sharpe ratio (assuming 0% risk-free rate)
        let sharpe = if volatility > 0.0 {
            mean_return * (252.0_f64).sqrt() / volatility
        } else {
            0.0
        };

        // Sortino ratio (only downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let sortino = if !downside_returns.is_empty() {
            let downside_variance = downside_returns
                .iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;

            let downside_vol = downside_variance.sqrt() * (252.0_f64).sqrt();

            if downside_vol > 0.0 {
                mean_return * (252.0_f64).sqrt() / downside_vol
            } else {
                0.0
            }
        } else {
            sharpe // No downside, use Sharpe
        };

        (sharpe, sortino, volatility)
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(equity_curve: &[EquityPoint]) -> (f64, i64) {
        if equity_curve.is_empty() {
            return (0.0, 0);
        }

        let mut max_equity = equity_curve[0].equity;
        let mut max_drawdown = 0.0;
        let mut max_dd_duration = 0_i64;
        let mut dd_start_time = equity_curve[0].timestamp;
        let mut in_drawdown = false;

        for point in equity_curve {
            if point.equity > max_equity {
                max_equity = point.equity;

                if in_drawdown {
                    let duration = (point.timestamp - dd_start_time).num_days();
                    max_dd_duration = max_dd_duration.max(duration);
                    in_drawdown = false;
                }
            } else {
                let drawdown = (max_equity - point.equity).to_f64().unwrap()
                    / max_equity.to_f64().unwrap();

                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }

                if !in_drawdown {
                    in_drawdown = true;
                    dd_start_time = point.timestamp;
                }
            }
        }

        (max_drawdown, max_dd_duration)
    }

    /// Calculate trade statistics
    #[allow(clippy::type_complexity)]
    fn calculate_trade_statistics(
        trades: &[Trade],
    ) -> (usize, usize, f64, f64, f64, f64, f64, f64) {
        if trades.is_empty() {
            return (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut wins = Vec::new();
        let mut losses = Vec::new();

        // Group trades into round trips
        for trade in trades {
            let pnl = trade.pnl.to_f64().unwrap();
            if pnl > 0.0 {
                wins.push(pnl);
            } else if pnl < 0.0 {
                losses.push(pnl.abs());
            }
        }

        let winning_trades = wins.len();
        let losing_trades = losses.len();
        let total = winning_trades + losing_trades;

        let win_rate = if total > 0 {
            winning_trades as f64 / total as f64
        } else {
            0.0
        };

        let total_wins: f64 = wins.iter().sum();
        let total_losses: f64 = losses.iter().sum();

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let average_win = if !wins.is_empty() {
            total_wins / wins.len() as f64
        } else {
            0.0
        };

        let average_loss = if !losses.is_empty() {
            total_losses / losses.len() as f64
        } else {
            0.0
        };

        let largest_win = wins.iter().fold(0.0_f64, |a, &b| a.max(b));
        let largest_loss = losses.iter().fold(0.0_f64, |a, &b| a.max(b));

        (
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            average_win,
            average_loss,
            largest_win,
            largest_loss,
        )
    }

    /// Calculate average exposure
    fn calculate_average_exposure(equity_curve: &[EquityPoint]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let total_exposure: f64 = equity_curve
            .iter()
            .map(|point| {
                let equity = point.equity.to_f64().unwrap();
                let positions = point.positions_value.to_f64().unwrap();
                if equity > 0.0 {
                    positions / equity
                } else {
                    0.0
                }
            })
            .sum();

        total_exposure / equity_curve.len() as f64
    }

    /// Calculate maximum exposure
    fn calculate_max_exposure(equity_curve: &[EquityPoint]) -> f64 {
        equity_curve
            .iter()
            .map(|point| {
                let equity = point.equity.to_f64().unwrap();
                let positions = point.positions_value.to_f64().unwrap();
                if equity > 0.0 {
                    positions / equity
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, |a, b| a.max(b))
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annual_return: 0.0,
            monthly_returns: Vec::new(),
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_duration_days: 0,
            volatility: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            average_win: 0.0,
            average_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            average_exposure: 0.0,
            max_exposure: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_metrics_calculation() {
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc::now(),
                equity: Decimal::from(100000),
                cash: Decimal::from(100000),
                positions_value: Decimal::ZERO,
            },
            EquityPoint {
                timestamp: Utc::now() + chrono::Duration::days(1),
                equity: Decimal::from(105000),
                cash: Decimal::from(95000),
                positions_value: Decimal::from(10000),
            },
            EquityPoint {
                timestamp: Utc::now() + chrono::Duration::days(2),
                equity: Decimal::from(110000),
                cash: Decimal::from(90000),
                positions_value: Decimal::from(20000),
            },
        ];

        let metrics = PerformanceMetrics::calculate(&equity_curve, &[]);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
    }
}
