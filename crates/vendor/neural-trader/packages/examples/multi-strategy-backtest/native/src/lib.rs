//! NAPI-RS implementation for performance-critical backtesting operations
//! Provides 10-100x speedup for intensive calculations

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct MarketBar {
  pub timestamp: f64,
  pub open: f64,
  pub high: f64,
  pub low: f64,
  pub close: f64,
  pub volume: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct PerformanceMetrics {
  pub total_return: f64,
  pub sharpe_ratio: f64,
  pub max_drawdown: f64,
  pub win_rate: f64,
  pub profit_factor: f64,
  pub calmar_ratio: f64,
  pub sortino_ratio: f64,
}

/// Calculate Simple Moving Average (optimized)
#[napi]
pub fn calculate_sma(prices: Vec<f64>, period: u32) -> Result<Vec<f64>> {
  if prices.is_empty() || period == 0 {
    return Ok(vec![]);
  }

  let period = period as usize;
  let mut sma = Vec::with_capacity(prices.len());

  // First SMA
  if prices.len() < period {
    return Ok(vec![]);
  }

  let mut sum: f64 = prices.iter().take(period).sum();
  sma.push(sum / period as f64);

  // Rolling SMA
  for i in period..prices.len() {
    sum = sum - prices[i - period] + prices[i];
    sma.push(sum / period as f64);
  }

  Ok(sma)
}

/// Calculate Exponential Moving Average (optimized)
#[napi]
pub fn calculate_ema(prices: Vec<f64>, period: u32) -> Result<Vec<f64>> {
  if prices.is_empty() || period == 0 {
    return Ok(vec![]);
  }

  let period = period as f64;
  let multiplier = 2.0 / (period + 1.0);
  let mut ema = Vec::with_capacity(prices.len());

  // Start with SMA for first value
  let first_sma: f64 = prices.iter().take(period as usize).sum::<f64>() / period;
  ema.push(first_sma);

  // Calculate EMA
  for i in 1..prices.len() {
    let value = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    ema.push(value);
  }

  Ok(ema)
}

/// Calculate returns from price series (optimized)
#[napi]
pub fn calculate_returns(prices: Vec<f64>) -> Result<Vec<f64>> {
  if prices.len() < 2 {
    return Ok(vec![]);
  }

  let mut returns = Vec::with_capacity(prices.len() - 1);

  for i in 1..prices.len() {
    let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
    returns.push(ret);
  }

  Ok(returns)
}

/// Calculate standard deviation (optimized)
#[napi]
pub fn calculate_std_dev(values: Vec<f64>) -> Result<f64> {
  if values.is_empty() {
    return Ok(0.0);
  }

  let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
  let variance: f64 = values
    .iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / values.len() as f64;

  Ok(variance.sqrt())
}

/// Calculate Sharpe ratio (optimized)
#[napi]
pub fn calculate_sharpe_ratio(returns: Vec<f64>, risk_free_rate: f64) -> Result<f64> {
  if returns.is_empty() {
    return Ok(0.0);
  }

  let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
  let std_dev = calculate_std_dev(returns)?;

  if std_dev == 0.0 {
    return Ok(0.0);
  }

  let excess_return = mean_return - risk_free_rate / 252.0;
  let sharpe = (excess_return / std_dev) * (252.0_f64).sqrt();

  Ok(sharpe)
}

/// Calculate maximum drawdown (optimized)
#[napi]
pub fn calculate_max_drawdown(equity_curve: Vec<f64>) -> Result<f64> {
  if equity_curve.is_empty() {
    return Ok(0.0);
  }

  let mut max_drawdown = 0.0;
  let mut peak = equity_curve[0];

  for &value in equity_curve.iter() {
    if value > peak {
      peak = value;
    }

    let drawdown = (peak - value) / peak;
    if drawdown > max_drawdown {
      max_drawdown = drawdown;
    }
  }

  Ok(max_drawdown)
}

/// Calculate comprehensive performance metrics (optimized)
#[napi]
pub fn calculate_performance_metrics(
  equity_curve: Vec<f64>,
  returns: Vec<f64>,
  trades: u32,
  wins: u32,
  total_profit: f64,
  total_loss: f64,
) -> Result<PerformanceMetrics> {
  let total_return = if !equity_curve.is_empty() && equity_curve[0] != 0.0 {
    (equity_curve[equity_curve.len() - 1] - equity_curve[0]) / equity_curve[0]
  } else {
    0.0
  };

  let sharpe_ratio = calculate_sharpe_ratio(returns.clone(), 0.0)?;
  let max_drawdown = calculate_max_drawdown(equity_curve)?;

  let win_rate = if trades > 0 {
    wins as f64 / trades as f64
  } else {
    0.0
  };

  let avg_profit = if wins > 0 {
    total_profit / wins as f64
  } else {
    0.0
  };

  let avg_loss = if trades > wins {
    total_loss.abs() / (trades - wins) as f64
  } else {
    0.0
  };

  let profit_factor = if avg_loss > 0.0 {
    avg_profit / avg_loss
  } else {
    0.0
  };

  let calmar_ratio = if max_drawdown != 0.0 {
    total_return / max_drawdown
  } else {
    0.0
  };

  // Sortino ratio - downside deviation
  let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
  let downside_std = calculate_std_dev(downside_returns)?;

  let sortino_ratio = if downside_std > 0.0 {
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    (mean_return / downside_std) * (252.0_f64).sqrt()
  } else {
    0.0
  };

  Ok(PerformanceMetrics {
    total_return,
    sharpe_ratio,
    max_drawdown,
    win_rate,
    profit_factor,
    calmar_ratio,
    sortino_ratio,
  })
}

/// Calculate Bollinger Bands (optimized)
#[napi]
pub fn calculate_bollinger_bands(
  prices: Vec<f64>,
  period: u32,
  std_multiplier: f64,
) -> Result<Vec<(f64, f64, f64)>> {
  let sma = calculate_sma(prices.clone(), period)?;
  let mut bands = Vec::with_capacity(sma.len());

  let period = period as usize;

  for (i, &middle) in sma.iter().enumerate() {
    let start = i;
    let end = start + period;

    if end <= prices.len() {
      let window = &prices[start..end];
      let std_dev = calculate_std_dev(window.to_vec())?;

      let upper = middle + (std_multiplier * std_dev);
      let lower = middle - (std_multiplier * std_dev);

      bands.push((lower, middle, upper));
    }
  }

  Ok(bands)
}

/// Calculate RSI (Relative Strength Index) - optimized
#[napi]
pub fn calculate_rsi(prices: Vec<f64>, period: u32) -> Result<Vec<f64>> {
  if prices.len() < period as usize + 1 {
    return Ok(vec![]);
  }

  let mut rsi_values = Vec::with_capacity(prices.len());
  let mut gains = Vec::new();
  let mut losses = Vec::new();

  // Calculate price changes
  for i in 1..prices.len() {
    let change = prices[i] - prices[i - 1];
    gains.push(if change > 0.0 { change } else { 0.0 });
    losses.push(if change < 0.0 { -change } else { 0.0 });
  }

  // Calculate first average gain/loss
  let mut avg_gain: f64 = gains.iter().take(period as usize).sum::<f64>() / period as f64;
  let mut avg_loss: f64 = losses.iter().take(period as usize).sum::<f64>() / period as f64;

  // Calculate RSI
  for i in period as usize..gains.len() {
    avg_gain = ((avg_gain * (period as f64 - 1.0)) + gains[i]) / period as f64;
    avg_loss = ((avg_loss * (period as f64 - 1.0)) + losses[i]) / period as f64;

    let rs = if avg_loss != 0.0 {
      avg_gain / avg_loss
    } else {
      100.0
    };

    let rsi = 100.0 - (100.0 / (1.0 + rs));
    rsi_values.push(rsi);
  }

  Ok(rsi_values)
}

/// Fast correlation calculation between two price series
#[napi]
pub fn calculate_correlation(series1: Vec<f64>, series2: Vec<f64>) -> Result<f64> {
  if series1.len() != series2.len() || series1.is_empty() {
    return Ok(0.0);
  }

  let n = series1.len() as f64;
  let mean1 = series1.iter().sum::<f64>() / n;
  let mean2 = series2.iter().sum::<f64>() / n;

  let mut covariance = 0.0;
  let mut var1 = 0.0;
  let mut var2 = 0.0;

  for i in 0..series1.len() {
    let diff1 = series1[i] - mean1;
    let diff2 = series2[i] - mean2;

    covariance += diff1 * diff2;
    var1 += diff1 * diff1;
    var2 += diff2 * diff2;
  }

  let denominator = (var1 * var2).sqrt();
  if denominator == 0.0 {
    return Ok(0.0);
  }

  Ok(covariance / denominator)
}

/// Vectorized backtest execution (ultra-fast)
#[napi]
pub fn fast_backtest(
  bars: Vec<MarketBar>,
  signals: Vec<i32>, // 1 = buy, -1 = sell, 0 = hold
  initial_capital: f64,
  commission: f64,
) -> Result<Vec<f64>> {
  if bars.len() != signals.len() {
    return Err(Error::from_reason("Bars and signals length mismatch"));
  }

  let mut equity_curve = Vec::with_capacity(bars.len());
  let mut position: f64 = 0.0;
  let mut cash = initial_capital;

  for i in 0..bars.len() {
    let bar = &bars[i];
    let signal = signals[i];

    // Execute signal
    if signal > 0 && position == 0.0 {
      // Buy
      let shares = (cash * 0.95) / bar.close;
      let cost = shares * bar.close * (1.0 + commission);
      if cost <= cash {
        position = shares;
        cash -= cost;
      }
    } else if signal < 0 && position > 0.0 {
      // Sell
      let proceeds = position * bar.close * (1.0 - commission);
      cash += proceeds;
      position = 0.0;
    }

    // Calculate equity
    let equity = cash + (position * bar.close);
    equity_curve.push(equity);
  }

  Ok(equity_curve)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_calculate_sma() {
    let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sma = calculate_sma(prices, 3).unwrap();
    assert_eq!(sma.len(), 3);
    assert!((sma[0] - 2.0).abs() < 0.001);
  }

  #[test]
  fn test_calculate_returns() {
    let prices = vec![100.0, 110.0, 105.0];
    let returns = calculate_returns(prices).unwrap();
    assert_eq!(returns.len(), 2);
    assert!((returns[0] - 0.1).abs() < 0.001);
  }

  #[test]
  fn test_sharpe_ratio() {
    let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
    let sharpe = calculate_sharpe_ratio(returns, 0.0).unwrap();
    assert!(sharpe > 0.0);
  }
}
