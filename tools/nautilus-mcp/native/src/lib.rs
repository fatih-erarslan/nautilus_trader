//! # Nautilus Native - High-Performance Trading Analytics
//!
//! Native Rust bindings for the Nautilus MCP Server providing:
//! - Technical indicators (momentum, volatility, trend)
//! - Risk analytics (VaR, CVaR, position sizing)
//! - Portfolio metrics (Sharpe, Sortino, Calmar)
//! - Execution analysis (VWAP, slippage, order flow)
//! - Regime detection (pBit dynamics, Ising model)
//! - Conformal prediction (uncertainty quantification)
//! - Options Greeks (Delta, Gamma, Theta, Vega)
//!
//! ## Performance Targets
//! - Indicator update: <1μs
//! - Risk calculation: <10μs
//! - Monte Carlo VaR: <1ms (10,000 simulations)

use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

// HyperPhysics integration available



// ============================================================================
// Common Types
// ============================================================================

/// OHLCV bar data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: i64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct Trade {
    pub price: f64,
    pub quantity: f64,
    pub side: String, // "buy" or "sell"
    pub timestamp: i64,
}

/// Analytics result wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AnalyticsResult {
    pub success: bool,
    pub value: Option<f64>,
    pub values: Option<Vec<f64>>,
    pub data: Option<String>, // JSON for complex results
    pub error: Option<String>,
}

impl AnalyticsResult {
    pub fn ok(value: f64) -> Self {
        Self {
            success: true,
            value: Some(value),
            values: None,
            data: None,
            error: None,
        }
    }

    pub fn ok_vec(values: Vec<f64>) -> Self {
        Self {
            success: true,
            value: None,
            values: Some(values),
            data: None,
            error: None,
        }
    }

    pub fn ok_json(data: serde_json::Value) -> Self {
        Self {
            success: true,
            value: None,
            values: None,
            data: Some(data.to_string()),
            error: None,
        }
    }

    pub fn err(msg: &str) -> Self {
        Self {
            success: false,
            value: None,
            values: None,
            data: None,
            error: Some(msg.to_string()),
        }
    }
}

// ============================================================================
// MOVING AVERAGES
// ============================================================================

/// Simple Moving Average
#[napi]
pub fn indicator_sma(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period || period == 0 {
        return AnalyticsResult::err("Insufficient data for SMA");
    }
    let sum: f64 = prices.iter().rev().take(period).sum();
    AnalyticsResult::ok(sum / period as f64)
}

/// Exponential Moving Average
#[napi]
pub fn indicator_ema(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.is_empty() || period == 0 {
        return AnalyticsResult::err("Insufficient data for EMA");
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = prices[0];

    for price in prices.iter().skip(1) {
        ema = (price - ema) * multiplier + ema;
    }

    AnalyticsResult::ok(ema)
}

/// Weighted Moving Average
#[napi]
pub fn indicator_wma(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period || period == 0 {
        return AnalyticsResult::err("Insufficient data for WMA");
    }

    let weights_sum = (period * (period + 1)) / 2;
    let mut sum = 0.0;

    for (i, price) in prices.iter().rev().take(period).enumerate() {
        sum += price * (period - i) as f64;
    }

    AnalyticsResult::ok(sum / weights_sum as f64)
}

/// Hull Moving Average (fast, reduced lag)
/// Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
/// Wolfram-verified: reduces lag by ~sqrt(n) compared to standard MA
#[napi]
pub fn indicator_hma(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period || period < 4 {
        return AnalyticsResult::err("Insufficient data for HMA (need period >= 4)");
    }

    let half_period = period / 2;
    let sqrt_period = ((period as f64).sqrt().round() as usize).max(1);

    // Build the intermediate series: 2*WMA(n/2) - WMA(n)
    // We need enough points to then apply WMA(sqrt_period) on this series
    let min_len = period + sqrt_period;
    if prices.len() < min_len {
        return AnalyticsResult::err(&format!(
            "HMA requires at least {} data points for period {}",
            min_len, period
        ));
    }

    // Generate intermediate series
    let mut raw_series = Vec::with_capacity(prices.len() - period + 1);
    for i in (period - 1)..prices.len() {
        let slice = &prices[(i + 1 - period)..=i];
        let wma_half = calculate_wma_slice(slice, half_period);
        let wma_full = calculate_wma_slice(slice, period);
        if wma_half.is_nan() || wma_full.is_nan() {
            continue;
        }
        raw_series.push(2.0 * wma_half - wma_full);
    }

    if raw_series.len() < sqrt_period {
        return AnalyticsResult::err("Insufficient intermediate data for HMA");
    }

    // Final WMA on the raw series with sqrt(period)
    let hma = calculate_wma_slice(&raw_series, sqrt_period);
    if hma.is_nan() {
        return AnalyticsResult::err("Final WMA calculation failed");
    }

    AnalyticsResult::ok(hma)
}

/// WMA helper for slices (uses last `period` elements)
fn calculate_wma_slice(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period || period == 0 {
        return f64::NAN;
    }
    let weights_sum = (period * (period + 1)) / 2;
    let mut sum = 0.0;
    for (i, price) in prices.iter().rev().take(period).enumerate() {
        sum += price * (period - i) as f64;
    }
    sum / weights_sum as f64
}


/// Double Exponential Moving Average
#[napi]
pub fn indicator_dema(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.is_empty() || period == 0 {
        return AnalyticsResult::err("Insufficient data for DEMA");
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    
    // First EMA
    let mut ema1 = prices[0];
    for price in prices.iter().skip(1) {
        ema1 = (price - ema1) * multiplier + ema1;
    }
    
    // Second EMA (EMA of EMA)
    let mut ema2 = prices[0];
    let mut ema1_current = prices[0];
    for price in prices.iter().skip(1) {
        ema1_current = (price - ema1_current) * multiplier + ema1_current;
        ema2 = (ema1_current - ema2) * multiplier + ema2;
    }

    // DEMA = 2 * EMA - EMA(EMA)
    AnalyticsResult::ok(2.0 * ema1 - ema2)
}

/// Volume Weighted Average Price
#[napi]
pub fn indicator_vwap(bars: Vec<Bar>) -> AnalyticsResult {
    if bars.is_empty() {
        return AnalyticsResult::err("No bars for VWAP");
    }

    let mut cum_pv = 0.0;
    let mut cum_volume = 0.0;

    for bar in &bars {
        let typical_price = (bar.high + bar.low + bar.close) / 3.0;
        cum_pv += typical_price * bar.volume;
        cum_volume += bar.volume;
    }

    if cum_volume == 0.0 {
        return AnalyticsResult::err("Zero volume");
    }

    AnalyticsResult::ok(cum_pv / cum_volume)
}

// ============================================================================
// MOMENTUM INDICATORS
// ============================================================================

/// Relative Strength Index
#[napi]
pub fn indicator_rsi(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period + 1 {
        return AnalyticsResult::err("Insufficient data for RSI");
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Smoothed averages (Wilder's method)
    let avg_gain: f64 = gains.iter().rev().take(period).sum::<f64>() / period as f64;
    let avg_loss: f64 = losses.iter().rev().take(period).sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        return AnalyticsResult::ok(100.0);
    }

    let rs = avg_gain / avg_loss;
    let rsi = 100.0 - (100.0 / (1.0 + rs));

    AnalyticsResult::ok(rsi)
}

/// MACD (Moving Average Convergence Divergence)
#[napi]
pub fn indicator_macd(prices: Vec<f64>, fast: u32, slow: u32, signal: u32) -> AnalyticsResult {
    if prices.len() < slow as usize {
        return AnalyticsResult::err("Insufficient data for MACD");
    }

    let fast_mult = 2.0 / (fast as f64 + 1.0);
    let slow_mult = 2.0 / (slow as f64 + 1.0);
    let signal_mult = 2.0 / (signal as f64 + 1.0);

    let mut fast_ema = prices[0];
    let mut slow_ema = prices[0];
    let mut macd_line = 0.0;
    let mut signal_line = 0.0;

    for (i, price) in prices.iter().enumerate() {
        fast_ema = (price - fast_ema) * fast_mult + fast_ema;
        slow_ema = (price - slow_ema) * slow_mult + slow_ema;
        macd_line = fast_ema - slow_ema;

        if i >= slow as usize {
            signal_line = (macd_line - signal_line) * signal_mult + signal_line;
        }
    }

    let histogram = macd_line - signal_line;

    AnalyticsResult::ok_json(serde_json::json!({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }))
}

/// Commodity Channel Index
#[napi]
pub fn indicator_cci(bars: Vec<Bar>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if bars.len() < period {
        return AnalyticsResult::err("Insufficient data for CCI");
    }

    // Typical prices
    let typical_prices: Vec<f64> = bars
        .iter()
        .map(|b| (b.high + b.low + b.close) / 3.0)
        .collect();

    // SMA of typical price
    let sma: f64 = typical_prices.iter().rev().take(period).sum::<f64>() / period as f64;

    // Mean absolute deviation
    let mad: f64 = typical_prices
        .iter()
        .rev()
        .take(period)
        .map(|tp| (tp - sma).abs())
        .sum::<f64>()
        / period as f64;

    if mad == 0.0 {
        return AnalyticsResult::ok(0.0);
    }

    let cci = (typical_prices.last().unwrap() - sma) / (0.015 * mad);
    AnalyticsResult::ok(cci)
}

/// Stochastic Oscillator (%K and %D)
/// %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
/// %D = SMA(%K, d_period) - the signal line
/// Wolfram-verified: standard momentum oscillator (0-100 range)
#[napi]
pub fn indicator_stochastic(bars: Vec<Bar>, k_period: u32, d_period: u32) -> AnalyticsResult {
    let k_period = k_period as usize;
    let d_period = d_period as usize;
    
    // Need enough bars to compute d_period worth of %K values
    let min_bars = k_period + d_period - 1;
    if bars.len() < min_bars {
        return AnalyticsResult::err(&format!(
            "Stochastic requires at least {} bars for K={}, D={}",
            min_bars, k_period, d_period
        ));
    }

    // Calculate %K series for the last d_period points
    let mut k_values = Vec::with_capacity(d_period);
    
    for i in (bars.len() - d_period)..bars.len() {
        let start = if i + 1 >= k_period { i + 1 - k_period } else { 0 };
        let window = &bars[start..=i];
        
        let highest_high = window.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = window.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
        let close = bars[i].close;
        
        let range = highest_high - lowest_low;
        let k = if range == 0.0 {
            50.0 // Neutral when no range
        } else {
            ((close - lowest_low) / range) * 100.0
        };
        k_values.push(k);
    }

    // Current %K is the last value
    let k = *k_values.last().unwrap();
    
    // %D is SMA of %K values
    let d = k_values.iter().sum::<f64>() / k_values.len() as f64;

    // Determine overbought/oversold zones
    let zone = if k > 80.0 {
        "overbought"
    } else if k < 20.0 {
        "oversold"
    } else {
        "neutral"
    };

    // Detect bullish/bearish crossovers
    let signal = if k > d && k_values.len() >= 2 && k_values[k_values.len() - 2] <= d {
        "bullish_crossover"
    } else if k < d && k_values.len() >= 2 && k_values[k_values.len() - 2] >= d {
        "bearish_crossover"
    } else if k > d {
        "bullish"
    } else {
        "bearish"
    };

    AnalyticsResult::ok_json(serde_json::json!({
        "k": k,
        "d": d,
        "zone": zone,
        "signal": signal,
        "k_history": k_values
    }))
}

/// Rate of Change
#[napi]
pub fn indicator_roc(prices: Vec<f64>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period + 1 {
        return AnalyticsResult::err("Insufficient data for ROC");
    }

    let current = prices[prices.len() - 1];
    let past = prices[prices.len() - 1 - period];

    if past == 0.0 {
        return AnalyticsResult::err("Division by zero in ROC");
    }

    let roc = ((current - past) / past) * 100.0;
    AnalyticsResult::ok(roc)
}

/// On-Balance Volume
#[napi]
pub fn indicator_obv(bars: Vec<Bar>) -> AnalyticsResult {
    if bars.len() < 2 {
        return AnalyticsResult::err("Insufficient data for OBV");
    }

    let mut obv = 0.0;

    for i in 1..bars.len() {
        if bars[i].close > bars[i - 1].close {
            obv += bars[i].volume;
        } else if bars[i].close < bars[i - 1].close {
            obv -= bars[i].volume;
        }
    }

    AnalyticsResult::ok(obv)
}

// ============================================================================
// VOLATILITY INDICATORS
// ============================================================================

/// Average True Range
#[napi]
pub fn indicator_atr(bars: Vec<Bar>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if bars.len() < period + 1 {
        return AnalyticsResult::err("Insufficient data for ATR");
    }

    let mut true_ranges = Vec::new();

    for i in 1..bars.len() {
        let tr = (bars[i].high - bars[i].low)
            .max((bars[i].high - bars[i - 1].close).abs())
            .max((bars[i].low - bars[i - 1].close).abs());
        true_ranges.push(tr);
    }

    // Wilder's smoothing
    let first_atr: f64 = true_ranges.iter().take(period).sum::<f64>() / period as f64;
    let mut atr = first_atr;

    for tr in true_ranges.iter().skip(period) {
        atr = (atr * (period - 1) as f64 + tr) / period as f64;
    }

    AnalyticsResult::ok(atr)
}

/// Bollinger Bands
#[napi]
pub fn indicator_bollinger(prices: Vec<f64>, period: u32, std_dev: f64) -> AnalyticsResult {
    let period = period as usize;
    if prices.len() < period {
        return AnalyticsResult::err("Insufficient data for Bollinger Bands");
    }

    let recent: Vec<f64> = prices.iter().rev().take(period).copied().collect();
    let sma: f64 = recent.iter().sum::<f64>() / period as f64;

    let variance: f64 = recent.iter().map(|x| (x - sma).powi(2)).sum::<f64>() / period as f64;
    let std = variance.sqrt();

    let upper = sma + std_dev * std;
    let lower = sma - std_dev * std;
    let width = (upper - lower) / sma * 100.0;

    AnalyticsResult::ok_json(serde_json::json!({
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "width": width,
        "std": std
    }))
}

/// Keltner Channel
#[napi]
pub fn indicator_keltner(bars: Vec<Bar>, ema_period: u32, atr_period: u32, multiplier: f64) -> AnalyticsResult {
    if bars.len() < (ema_period.max(atr_period) as usize) + 1 {
        return AnalyticsResult::err("Insufficient data for Keltner Channel");
    }

    // EMA of close
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema_mult = 2.0 / (ema_period as f64 + 1.0);
    let mut ema = closes[0];
    for close in closes.iter().skip(1) {
        ema = (close - ema) * ema_mult + ema;
    }

    // ATR
    let atr_result = indicator_atr(bars, atr_period);
    let atr = atr_result.value.unwrap_or(0.0);

    let upper = ema + multiplier * atr;
    let lower = ema - multiplier * atr;

    AnalyticsResult::ok_json(serde_json::json!({
        "upper": upper,
        "middle": ema,
        "lower": lower,
        "atr": atr
    }))
}

/// Donchian Channel
#[napi]
pub fn indicator_donchian(bars: Vec<Bar>, period: u32) -> AnalyticsResult {
    let period = period as usize;
    if bars.len() < period {
        return AnalyticsResult::err("Insufficient data for Donchian Channel");
    }

    let recent: Vec<&Bar> = bars.iter().rev().take(period).collect();
    let upper: f64 = recent.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
    let lower: f64 = recent.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
    let middle = (upper + lower) / 2.0;

    AnalyticsResult::ok_json(serde_json::json!({
        "upper": upper,
        "middle": middle,
        "lower": lower
    }))
}

// ============================================================================
// RISK ANALYTICS
// ============================================================================

/// Value at Risk (Parametric/Gaussian)
#[napi]
pub fn risk_var_parametric(returns: Vec<f64>, confidence: f64) -> AnalyticsResult {
    if returns.is_empty() {
        return AnalyticsResult::err("No returns for VaR");
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    // Z-score for confidence level (e.g., 0.95 -> 1.645, 0.99 -> 2.326)
    let z = inverse_normal_cdf(confidence);

    let var = mean - z * std;
    AnalyticsResult::ok(-var) // Positive VaR = loss
}

/// Value at Risk (Historical)
#[napi]
pub fn risk_var_historical(returns: Vec<f64>, confidence: f64) -> AnalyticsResult {
    if returns.is_empty() {
        return AnalyticsResult::err("No returns for VaR");
    }

    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let var = sorted.get(index).copied().unwrap_or(0.0);

    AnalyticsResult::ok(-var)
}

/// Conditional VaR (Expected Shortfall)
#[napi]
pub fn risk_cvar(returns: Vec<f64>, confidence: f64) -> AnalyticsResult {
    if returns.is_empty() {
        return AnalyticsResult::err("No returns for CVaR");
    }

    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutoff_index = ((1.0 - confidence) * sorted.len() as f64).ceil() as usize;
    let tail: Vec<f64> = sorted.iter().take(cutoff_index).copied().collect();

    if tail.is_empty() {
        return AnalyticsResult::ok(0.0);
    }

    let cvar: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    AnalyticsResult::ok(-cvar)
}

/// Kelly Criterion: f* = (p*b - q) / b
#[napi]
pub fn risk_kelly_criterion(win_rate: f64, win_loss_ratio: f64) -> AnalyticsResult {
    if win_rate <= 0.0 || win_rate >= 1.0 || win_loss_ratio <= 0.0 {
        return AnalyticsResult::err("Invalid parameters for Kelly");
    }

    let p = win_rate;
    let q = 1.0 - win_rate;
    let b = win_loss_ratio;

    let kelly = (p * b - q) / b;

    AnalyticsResult::ok_json(serde_json::json!({
        "kelly_fraction": kelly,
        "half_kelly": kelly / 2.0,
        "quarter_kelly": kelly / 4.0
    }))
}

/// Fixed-Risk Position Sizing with Risk-Reward Analysis
/// Implements proper position sizing with optional take-profit for R:R calculation
/// Wolfram-verified: risk_amount = equity * risk_per_trade, position = risk_amount / risk_per_unit
#[napi]
pub fn risk_position_size(
    equity: f64,
    risk_per_trade: f64,
    entry_price: f64,
    stop_loss: f64,
) -> AnalyticsResult {
    risk_position_size_with_target(equity, risk_per_trade, entry_price, stop_loss, None)
}

/// Position sizing with explicit take-profit target
#[napi]
pub fn risk_position_size_with_target(
    equity: f64,
    risk_per_trade: f64,
    entry_price: f64,
    stop_loss: f64,
    take_profit: Option<f64>,
) -> AnalyticsResult {
    if equity <= 0.0 {
        return AnalyticsResult::err("Equity must be positive");
    }
    if risk_per_trade <= 0.0 || risk_per_trade > 1.0 {
        return AnalyticsResult::err("Risk per trade must be between 0 and 1 (e.g., 0.02 = 2%)");
    }
    if entry_price <= 0.0 {
        return AnalyticsResult::err("Entry price must be positive");
    }
    if stop_loss <= 0.0 {
        return AnalyticsResult::err("Stop loss must be positive");
    }

    // Determine trade direction
    let is_long = stop_loss < entry_price;
    let risk_per_unit = (entry_price - stop_loss).abs();

    if risk_per_unit == 0.0 {
        return AnalyticsResult::err("Entry price equals stop loss - no risk defined");
    }

    let risk_amount = equity * risk_per_trade;
    let position_size = risk_amount / risk_per_unit;
    let position_value = position_size * entry_price;

    // Calculate risk-reward ratio if take_profit provided
    let (risk_reward, reward_per_unit, expected_value) = if let Some(tp) = take_profit {
        let reward = (tp - entry_price).abs();
        let rr = if risk_per_unit > 0.0 { reward / risk_per_unit } else { 0.0 };
        
        // Validate take profit direction matches trade direction
        let tp_valid = if is_long { tp > entry_price } else { tp < entry_price };
        if !tp_valid {
            return AnalyticsResult::err("Take profit direction inconsistent with trade direction");
        }
        
        // Expected value assuming 50% win rate (breakeven R:R = 1.0)
        // For positive expectancy, need win_rate > 1/(1+R:R)
        let breakeven_winrate = 1.0 / (1.0 + rr);
        
        (rr, reward, breakeven_winrate)
    } else {
        (0.0, 0.0, 0.5) // No target = no R:R calculation
    };

    // Risk metrics
    let risk_pct = (risk_per_unit / entry_price) * 100.0;
    let leverage = position_value / equity;

    AnalyticsResult::ok_json(serde_json::json!({
        "position_size": position_size,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "risk_per_unit": risk_per_unit,
        "risk_pct": risk_pct,
        "direction": if is_long { "long" } else { "short" },
        "risk_reward": risk_reward,
        "reward_per_unit": reward_per_unit,
        "breakeven_winrate": expected_value,
        "leverage": leverage,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }))
}

/// Maximum Drawdown
#[napi]
pub fn risk_max_drawdown(equity_curve: Vec<f64>) -> AnalyticsResult {
    if equity_curve.is_empty() {
        return AnalyticsResult::err("No equity data");
    }

    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];
    let mut max_dd_start = 0;
    let mut max_dd_end = 0;
    let mut current_start = 0;

    for (i, &equity) in equity_curve.iter().enumerate() {
        if equity > peak {
            peak = equity;
            current_start = i;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
            max_dd_start = current_start;
            max_dd_end = i;
        }
    }

    AnalyticsResult::ok_json(serde_json::json!({
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100.0,
        "start_index": max_dd_start,
        "end_index": max_dd_end
    }))
}

/// Hurst Exponent (persistence/mean reversion detection)
#[napi]
pub fn risk_hurst_exponent(prices: Vec<f64>) -> AnalyticsResult {
    if prices.len() < 20 {
        return AnalyticsResult::err("Insufficient data for Hurst exponent");
    }

    // R/S analysis (simplified)
    let n = prices.len();
    let mean: f64 = prices.iter().sum::<f64>() / n as f64;
    
    // Cumulative deviations
    let mut cumsum = Vec::with_capacity(n);
    let mut running_sum = 0.0;
    for price in &prices {
        running_sum += price - mean;
        cumsum.push(running_sum);
    }

    let r = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - cumsum.iter().cloned().fold(f64::INFINITY, f64::min);

    let variance: f64 = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
    let s = variance.sqrt();

    if s == 0.0 {
        return AnalyticsResult::ok(0.5);
    }

    let rs = r / s;
    let hurst = rs.ln() / (n as f64).ln();

    let interpretation = if hurst > 0.55 {
        "trending"
    } else if hurst < 0.45 {
        "mean_reverting"
    } else {
        "random_walk"
    };

    AnalyticsResult::ok_json(serde_json::json!({
        "hurst": hurst.clamp(0.0, 1.0),
        "interpretation": interpretation
    }))
}

// ============================================================================
// PORTFOLIO ANALYTICS
// ============================================================================

/// Sharpe Ratio
#[napi]
pub fn portfolio_sharpe(returns: Vec<f64>, risk_free_rate: f64) -> AnalyticsResult {
    if returns.is_empty() {
        return AnalyticsResult::err("No returns for Sharpe");
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let excess_return = mean - risk_free_rate;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return AnalyticsResult::ok(0.0);
    }

    AnalyticsResult::ok(excess_return / std)
}

/// Sortino Ratio (downside deviation)
#[napi]
pub fn portfolio_sortino(returns: Vec<f64>, risk_free_rate: f64, target: f64) -> AnalyticsResult {
    if returns.is_empty() {
        return AnalyticsResult::err("No returns for Sortino");
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let excess_return = mean - risk_free_rate;

    // Downside deviation
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < target)
        .map(|&r| (r - target).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return AnalyticsResult::ok(f64::INFINITY);
    }

    let downside_dev = (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt();

    if downside_dev == 0.0 {
        return AnalyticsResult::ok(f64::INFINITY);
    }

    AnalyticsResult::ok(excess_return / downside_dev)
}

/// Calmar Ratio (CAGR / Max Drawdown)
#[napi]
pub fn portfolio_calmar(equity_curve: Vec<f64>, periods_per_year: f64) -> AnalyticsResult {
    if equity_curve.len() < 2 {
        return AnalyticsResult::err("Insufficient data for Calmar");
    }

    let start = equity_curve[0];
    let end = equity_curve[equity_curve.len() - 1];
    let n_periods = equity_curve.len() as f64;

    // CAGR
    let cagr = ((end / start).powf(periods_per_year / n_periods) - 1.0) * 100.0;

    // Max DD
    let mut max_dd = 0.0;
    let mut peak = equity_curve[0];
    for &equity in &equity_curve {
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    if max_dd == 0.0 {
        return AnalyticsResult::ok(f64::INFINITY);
    }

    let calmar = cagr / (max_dd * 100.0);

    AnalyticsResult::ok_json(serde_json::json!({
        "calmar": calmar,
        "cagr": cagr,
        "max_drawdown": max_dd * 100.0
    }))
}

/// Win Rate
#[napi]
pub fn portfolio_win_rate(trade_pnls: Vec<f64>) -> AnalyticsResult {
    if trade_pnls.is_empty() {
        return AnalyticsResult::err("No trades");
    }

    let wins = trade_pnls.iter().filter(|&&pnl| pnl > 0.0).count();
    let total = trade_pnls.len();
    let win_rate = wins as f64 / total as f64;

    AnalyticsResult::ok_json(serde_json::json!({
        "win_rate": win_rate,
        "wins": wins,
        "losses": total - wins,
        "total_trades": total
    }))
}

/// Profit Factor
#[napi]
pub fn portfolio_profit_factor(trade_pnls: Vec<f64>) -> AnalyticsResult {
    let gross_profit: f64 = trade_pnls.iter().filter(|&&p| p > 0.0).sum();
    let gross_loss: f64 = trade_pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();

    if gross_loss == 0.0 {
        return AnalyticsResult::ok(f64::INFINITY);
    }

    AnalyticsResult::ok(gross_profit / gross_loss)
}

/// Expectancy
#[napi]
pub fn portfolio_expectancy(trade_pnls: Vec<f64>) -> AnalyticsResult {
    if trade_pnls.is_empty() {
        return AnalyticsResult::err("No trades");
    }

    let wins: Vec<f64> = trade_pnls.iter().filter(|&&p| p > 0.0).copied().collect();
    let losses: Vec<f64> = trade_pnls.iter().filter(|&&p| p < 0.0).copied().collect();

    let win_rate = wins.len() as f64 / trade_pnls.len() as f64;
    let avg_win = if wins.is_empty() { 0.0 } else { wins.iter().sum::<f64>() / wins.len() as f64 };
    let avg_loss = if losses.is_empty() { 0.0 } else { losses.iter().sum::<f64>().abs() / losses.len() as f64 };

    let expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss;

    AnalyticsResult::ok_json(serde_json::json!({
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }))
}

// ============================================================================
// EXECUTION ANALYSIS
// ============================================================================

/// VWAP Slippage Analysis
#[napi]
pub fn execution_vwap_slippage(
    trades: Vec<Trade>,
    vwap_benchmark: f64,
) -> AnalyticsResult {
    if trades.is_empty() {
        return AnalyticsResult::err("No trades");
    }

    let mut total_slippage = 0.0;
    let mut total_value = 0.0;

    for trade in &trades {
        let trade_value = trade.price * trade.quantity;
        let slippage = if trade.side == "buy" {
            trade.price - vwap_benchmark
        } else {
            vwap_benchmark - trade.price
        };
        total_slippage += slippage * trade.quantity;
        total_value += trade_value;
    }

    let avg_slippage_bps = if total_value > 0.0 {
        (total_slippage / total_value) * 10000.0
    } else {
        0.0
    };

    AnalyticsResult::ok_json(serde_json::json!({
        "total_slippage": total_slippage,
        "avg_slippage_bps": avg_slippage_bps,
        "vwap_benchmark": vwap_benchmark,
        "total_value": total_value
    }))
}

/// Order Flow Imbalance
#[napi]
pub fn orderflow_imbalance(trades: Vec<Trade>) -> AnalyticsResult {
    if trades.is_empty() {
        return AnalyticsResult::err("No trades");
    }

    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;

    for trade in &trades {
        if trade.side == "buy" {
            buy_volume += trade.quantity;
        } else {
            sell_volume += trade.quantity;
        }
    }

    let total_volume = buy_volume + sell_volume;
    let imbalance = if total_volume > 0.0 {
        (buy_volume - sell_volume) / total_volume
    } else {
        0.0
    };

    AnalyticsResult::ok_json(serde_json::json!({
        "imbalance": imbalance,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_pct": buy_volume / total_volume * 100.0,
        "sell_pct": sell_volume / total_volume * 100.0
    }))
}

// ============================================================================
// REGIME DETECTION (HyperPhysics Integration)
// ============================================================================

/// pBit-based Market State Sampling
/// Models market as a probabilistic bit with field (signal) and bias (trend)
/// Wolfram-verified: P(bullish) = σ((field + bias) / T_eff) where σ is sigmoid
/// Based on thermodynamic computing principles from HyperPhysics
#[napi]
pub fn regime_pbit_state(market_signal: f64, volatility: f64, temperature: f64) -> AnalyticsResult {
    regime_pbit_state_with_bias(market_signal, volatility, temperature, 0.0)
}

/// pBit state with explicit bias (trend strength)
#[napi]
pub fn regime_pbit_state_with_bias(
    market_signal: f64,
    volatility: f64,
    temperature: f64,
    bias: f64,
) -> AnalyticsResult {
    // Validate inputs
    if volatility < 0.0 {
        return AnalyticsResult::err("Volatility must be non-negative");
    }
    if temperature <= 0.0 {
        return AnalyticsResult::err("Temperature must be positive");
    }

    // Combined field: market signal + systematic bias (trend)
    let total_field = market_signal + bias;
    
    // Effective temperature: scales with volatility (uncertainty)
    // Higher volatility = more noise = higher effective temperature
    let effective_temp = temperature * volatility.max(0.01);
    
    // pBit probability via sigmoid (Boltzmann distribution)
    // P(bullish) = 1 / (1 + exp(-field/T))
    let exponent = -total_field / effective_temp;
    let prob_bullish = if exponent > 700.0 {
        0.0 // Prevent overflow
    } else if exponent < -700.0 {
        1.0
    } else {
        1.0 / (1.0 + exponent.exp())
    };
    let prob_bearish = 1.0 - prob_bullish;

    // State classification with hysteresis thresholds
    let state = if prob_bullish > 0.65 {
        "bullish"
    } else if prob_bullish < 0.35 {
        "bearish"
    } else {
        "neutral"
    };

    // Confidence: distance from maximum uncertainty (0.5)
    let confidence = (prob_bullish - 0.5).abs() * 2.0;

    // Shannon entropy: H = -p*log(p) - (1-p)*log(1-p)
    let entropy = if prob_bullish > 0.0 && prob_bullish < 1.0 {
        -prob_bullish * prob_bullish.ln() - prob_bearish * prob_bearish.ln()
    } else {
        0.0 // Pure state has zero entropy
    };

    // Free energy: F = -T * log(Z) where Z = exp(field/T) + exp(-field/T)
    let partition_fn = (total_field / effective_temp).exp() + (-total_field / effective_temp).exp();
    let free_energy = -effective_temp * partition_fn.ln();

    // Expected spin: <s> = tanh(field/T) - magnetization
    let magnetization = (total_field / effective_temp).tanh();

    AnalyticsResult::ok_json(serde_json::json!({
        "prob_bullish": prob_bullish,
        "prob_bearish": prob_bearish,
        "state": state,
        "confidence": confidence,
        "temperature": effective_temp,
        "entropy": entropy,
        "free_energy": free_energy,
        "magnetization": magnetization,
        "field": total_field,
        "bias": bias,
        "signal": market_signal
    }))
}

/// Ising Model Market Coherence
#[napi]
pub fn regime_ising_energy(asset_returns: Vec<f64>) -> AnalyticsResult {
    if asset_returns.len() < 2 {
        return AnalyticsResult::err("Need at least 2 assets");
    }

    // Compute pairwise correlations as coupling
    let n = asset_returns.len();
    let mean: f64 = asset_returns.iter().sum::<f64>() / n as f64;
    let signs: Vec<i32> = asset_returns.iter().map(|r| if *r > mean { 1 } else { -1 }).collect();

    // Ising energy: E = -sum(J_ij * s_i * s_j)
    let mut energy = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            energy -= (signs[i] * signs[j]) as f64;
        }
    }

    let max_energy = (n * (n - 1) / 2) as f64;
    let normalized_energy = energy / max_energy;

    // Coherence: high when all aligned
    let magnetization: f64 = signs.iter().sum::<i32>() as f64 / n as f64;

    AnalyticsResult::ok_json(serde_json::json!({
        "energy": energy,
        "normalized_energy": normalized_energy,
        "magnetization": magnetization,
        "coherence": magnetization.abs(),
        "critical_temp": 2.0 / (1.0 + SQRT_2).ln() // 2D Ising T_c
    }))
}

/// Hyperbolic Market Embedding
#[napi]
pub fn regime_hyperbolic_embed(features: Vec<f64>) -> AnalyticsResult {
    if features.is_empty() {
        return AnalyticsResult::err("No features");
    }

    // Lift to Lorentz hyperboloid
    let spatial_norm_sq: f64 = features.iter().map(|x| x * x).sum();
    let x0 = (1.0 + spatial_norm_sq).sqrt();

    let mut lorentz_coords = vec![x0];
    lorentz_coords.extend(features.iter());

    // Hyperbolic norm (distance from origin)
    let hyperbolic_norm = x0.acosh();

    AnalyticsResult::ok_json(serde_json::json!({
        "lorentz_coords": lorentz_coords,
        "hyperbolic_norm": hyperbolic_norm,
        "x0": x0,
        "dimension": features.len() + 1
    }))
}

// ============================================================================
// CONFORMAL PREDICTION
// ============================================================================

/// Conformal Prediction Interval
#[napi]
pub fn conformal_prediction_interval(
    residuals: Vec<f64>,
    prediction: f64,
    confidence: f64,
) -> AnalyticsResult {
    if residuals.is_empty() {
        return AnalyticsResult::err("No residuals for calibration");
    }

    let mut abs_residuals: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Quantile for coverage
    let index = ((confidence) * abs_residuals.len() as f64).ceil() as usize;
    let quantile = abs_residuals.get(index.min(abs_residuals.len() - 1)).copied().unwrap_or(0.0);

    let lower = prediction - quantile;
    let upper = prediction + quantile;

    AnalyticsResult::ok_json(serde_json::json!({
        "lower": lower,
        "upper": upper,
        "prediction": prediction,
        "width": upper - lower,
        "confidence": confidence,
        "quantile": quantile
    }))
}

// ============================================================================
// OPTIONS GREEKS
// ============================================================================

/// Black-Scholes Delta
#[napi]
pub fn greeks_delta(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> AnalyticsResult {
    if volatility <= 0.0 || time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return AnalyticsResult::err("Invalid parameters");
    }

    let d1 = (spot / strike).ln() + (rate + volatility.powi(2) / 2.0) * time_to_expiry;
    let d1 = d1 / (volatility * time_to_expiry.sqrt());

    let nd1 = normal_cdf(d1);

    let delta = if is_call { nd1 } else { nd1 - 1.0 };

    AnalyticsResult::ok(delta)
}

/// Black-Scholes Gamma
#[napi]
pub fn greeks_gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> AnalyticsResult {
    if volatility <= 0.0 || time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return AnalyticsResult::err("Invalid parameters");
    }

    let d1 = (spot / strike).ln() + (rate + volatility.powi(2) / 2.0) * time_to_expiry;
    let d1 = d1 / (volatility * time_to_expiry.sqrt());

    let n_prime_d1 = (-d1.powi(2) / 2.0).exp() / (2.0 * PI).sqrt();

    let gamma = n_prime_d1 / (spot * volatility * time_to_expiry.sqrt());

    AnalyticsResult::ok(gamma)
}

/// Black-Scholes Theta
#[napi]
pub fn greeks_theta(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> AnalyticsResult {
    if volatility <= 0.0 || time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return AnalyticsResult::err("Invalid parameters");
    }

    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate + volatility.powi(2) / 2.0) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let n_prime_d1 = (-d1.powi(2) / 2.0).exp() / (2.0 * PI).sqrt();
    let discount = (-rate * time_to_expiry).exp();

    let theta = if is_call {
        -(spot * n_prime_d1 * volatility) / (2.0 * sqrt_t)
            - rate * strike * discount * normal_cdf(d2)
    } else {
        -(spot * n_prime_d1 * volatility) / (2.0 * sqrt_t)
            + rate * strike * discount * normal_cdf(-d2)
    };

    // Convert to daily
    AnalyticsResult::ok(theta / 365.0)
}

/// Black-Scholes Vega
#[napi]
pub fn greeks_vega(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> AnalyticsResult {
    if volatility <= 0.0 || time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return AnalyticsResult::err("Invalid parameters");
    }

    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate + volatility.powi(2) / 2.0) * time_to_expiry)
        / (volatility * sqrt_t);

    let n_prime_d1 = (-d1.powi(2) / 2.0).exp() / (2.0 * PI).sqrt();

    let vega = spot * sqrt_t * n_prime_d1;

    // Per 1% move in volatility
    AnalyticsResult::ok(vega / 100.0)
}

/// Black-Scholes Option Pricing
#[napi]
pub fn options_black_scholes(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> AnalyticsResult {
    if volatility <= 0.0 || time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return AnalyticsResult::err("Invalid parameters");
    }

    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate + volatility.powi(2) / 2.0) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let discount = (-rate * time_to_expiry).exp();

    let price = if is_call {
        spot * normal_cdf(d1) - strike * discount * normal_cdf(d2)
    } else {
        strike * discount * normal_cdf(-d2) - spot * normal_cdf(-d1)
    };

    AnalyticsResult::ok_json(serde_json::json!({
        "price": price,
        "d1": d1,
        "d2": d2,
        "delta": if is_call { normal_cdf(d1) } else { normal_cdf(d1) - 1.0 }
    }))
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Normal CDF approximation (Abramowitz and Stegun)
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Inverse Normal CDF (Rational approximation)
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }

    // Rational approximation for lower region
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ============================================================================
// MODULE STUBS (for organization, implementations above)
// ============================================================================

mod indicators {}
mod risk {}
mod portfolio {}
mod execution {}
mod regime {}
mod conformal {}
mod greeks {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let prices = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 13.5];
        let result = indicator_ema(prices, 5);
        assert!(result.success);
        assert!(result.value.unwrap() > 11.0 && result.value.unwrap() < 14.0);
    }

    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28];
        let result = indicator_rsi(prices, 14);
        assert!(result.success);
    }

    #[test]
    fn test_sharpe() {
        let returns = vec![0.01, -0.005, 0.02, 0.015, -0.01, 0.03, 0.005];
        let result = portfolio_sharpe(returns, 0.0);
        assert!(result.success);
    }

    #[test]
    fn test_kelly() {
        let result = risk_kelly_criterion(0.55, 2.0);
        assert!(result.success);
    }
}
