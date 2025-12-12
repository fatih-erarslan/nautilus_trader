//! Volatility estimation and analysis module
//!
//! This module implements multiple volatility estimators including:
//! - Yang-Zhang volatility (combines overnight and intraday volatility)
//! - GARCH-like volatility with dynamic alpha
//! - Parkinson volatility (high-low based)
//! - ATR-based volatility
//! - Robust volatility combination

use crate::{AntifragilityError, AntifragilityParameters, AntifragilityResult, VolatilityResult};
use ndarray::prelude::*;
use num_traits::Float;
use rayon::prelude::*;
use std::f64::consts::LN_2;

/// Calculate robust volatility using multiple estimators
pub fn calculate_robust_volatility(
    prices: &Array1<f64>,
    volumes: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<VolatilityResult> {
    let n = prices.len();
    if n < params.vol_period + 5 {
        return Err(AntifragilityError::InsufficientData {
            required: params.vol_period + 5,
            actual: n,
        });
    }
    
    // Calculate individual volatility components
    let yz_volatility = calculate_yang_zhang_volatility(prices, volumes, params)?;
    let garch_volatility = calculate_garch_volatility(prices, params)?;
    let parkinson_volatility = calculate_parkinson_volatility(prices, params)?;
    let atr_volatility = calculate_atr_volatility(prices, params)?;
    
    // Combine volatility estimators
    let combined_vol = combine_volatility_estimators(
        &yz_volatility,
        &garch_volatility,
        &parkinson_volatility,
        &atr_volatility,
        params,
    )?;
    
    // Calculate volatility regime
    let vol_regime = calculate_volatility_regime(&combined_vol, params)?;
    
    // Calculate volatility rate of change
    let vol_roc_smoothed = calculate_volatility_roc(&vol_regime, params)?;
    
    Ok(VolatilityResult {
        combined_vol,
        vol_regime,
        vol_roc_smoothed,
        yz_volatility,
        garch_volatility,
        parkinson_volatility,
        atr_volatility,
    })
}

/// Calculate Yang-Zhang volatility estimator
fn calculate_yang_zhang_volatility(
    prices: &Array1<f64>,
    _volumes: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = prices.len();
    let mut yz_variance = Array1::zeros(n);
    let period = params.vol_period;
    
    // Calculate overnight returns (approximated as gaps)
    let mut overnight_returns = Array1::zeros(n);
    for i in 1..n {
        overnight_returns[i] = (prices[i] / prices[i-1]).ln();
    }
    
    // Calculate Yang-Zhang variance
    for i in period..n {
        let mut overnight_var = 0.0;
        let mut rs_sum = 0.0;
        let mut count = 0;
        
        for j in (i - period)..i {
            if j > 0 {
                // Overnight variance component
                let overnight_ret = overnight_returns[j];
                overnight_var += overnight_ret * overnight_ret;
                
                // Rogers-Satchell component (simplified without intraday data)
                let log_ret = (prices[j] / prices[j-1]).ln();
                rs_sum += log_ret * log_ret;
                count += 1;
            }
        }
        
        if count > 0 {
            overnight_var /= count as f64;
            rs_sum /= count as f64;
            
            // Yang-Zhang k parameter
            let k = params.yz_volatility_k / (1.34 + (period as f64 + 1.0) / (period as f64 - 1.0));
            
            // Combined YZ variance
            yz_variance[i] = overnight_var + k * rs_sum;
        }
    }
    
    // Convert to volatility (standard deviation)
    let yz_volatility = yz_variance.mapv(|x| x.max(0.0).sqrt());
    
    Ok(yz_volatility)
}

/// Calculate GARCH-like volatility with dynamic alpha
fn calculate_garch_volatility(
    prices: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = prices.len();
    let mut garch_var = Array1::zeros(n);
    
    // Calculate returns
    let mut returns = Array1::zeros(n);
    for i in 1..n {
        returns[i] = (prices[i] / prices[i-1]).ln();
    }
    
    // Calculate initial variance
    let initial_var = if n > 1 {
        let mean_return = returns.mean().unwrap_or(0.0);
        returns.mapv(|x| (x - mean_return).powi(2)).mean().unwrap_or(1e-6)
    } else {
        1e-6
    };
    
    garch_var[0] = initial_var;
    
    // Calculate GARCH variance with dynamic alpha
    for i in 1..n {
        let ret = returns[i];
        let ret_std = returns.std(1.0);
        let ret_ratio = (ret.abs() / ret_std.max(1e-9)).min(3.0);
        let dynamic_alpha = params.garch_alpha_base * (1.0 + ret_ratio);
        
        garch_var[i] = dynamic_alpha * ret * ret + (1.0 - dynamic_alpha) * garch_var[i-1];
    }
    
    // Convert to volatility
    let garch_volatility = garch_var.mapv(|x| x.max(0.0).sqrt());
    
    Ok(garch_volatility)
}

/// Calculate Parkinson volatility (high-low based)
fn calculate_parkinson_volatility(
    prices: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = prices.len();
    let mut parkinson_vol = Array1::zeros(n);
    let period = params.vol_period;
    
    // Since we only have close prices, approximate high-low from price movements
    let mut high_low_ratios = Array1::zeros(n);
    for i in 1..n {
        let price_change = (prices[i] - prices[i-1]).abs();
        let approx_high = prices[i] + price_change * 0.5;
        let approx_low = prices[i] - price_change * 0.5;
        
        if approx_low > 0.0 && approx_high > 0.0 {
            high_low_ratios[i] = (approx_high / approx_low).ln().powi(2);
        }
    }
    
    // Calculate rolling Parkinson volatility
    for i in period..n {
        let mut sum_hl = 0.0;
        let mut count = 0;
        
        for j in (i - period)..i {
            if high_low_ratios[j] > 0.0 {
                sum_hl += high_low_ratios[j];
                count += 1;
            }
        }
        
        if count > 0 {
            let park_var = sum_hl / (count as f64 * params.parkinson_factor);
            parkinson_vol[i] = park_var.max(0.0).sqrt();
        }
    }
    
    Ok(parkinson_vol)
}

/// Calculate ATR-based volatility
fn calculate_atr_volatility(
    prices: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = prices.len();
    let mut atr_vol = Array1::zeros(n);
    let period = params.vol_period;
    
    // Calculate true range (simplified for single price series)
    let mut true_ranges = Array1::zeros(n);
    for i in 1..n {
        let price_change = (prices[i] - prices[i-1]).abs();
        true_ranges[i] = price_change;
    }
    
    // Calculate ATR using exponential moving average
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut atr = Array1::zeros(n);
    
    if n > 0 {
        atr[0] = true_ranges[0];
        for i in 1..n {
            atr[i] = alpha * true_ranges[i] + (1.0 - alpha) * atr[i-1];
        }
    }
    
    // Normalize by current price to get relative volatility
    for i in 0..n {
        if prices[i] > 0.0 {
            atr_vol[i] = atr[i] / prices[i];
        }
    }
    
    Ok(atr_vol)
}

/// Combine multiple volatility estimators
fn combine_volatility_estimators(
    yz_vol: &Array1<f64>,
    garch_vol: &Array1<f64>,
    parkinson_vol: &Array1<f64>,
    atr_vol: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = yz_vol.len();
    let mut combined_vol = Array1::zeros(n);
    
    // Define weights for each estimator
    let weights = [0.35, 0.30, 0.15, 0.20]; // YZ, GARCH, Parkinson, ATR
    
    for i in 0..n {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        // Yang-Zhang
        if yz_vol[i] > 1e-9 {
            weighted_sum += weights[0] * yz_vol[i];
            weight_sum += weights[0];
        }
        
        // GARCH
        if garch_vol[i] > 1e-9 {
            weighted_sum += weights[1] * garch_vol[i];
            weight_sum += weights[1];
        }
        
        // Parkinson
        if parkinson_vol[i] > 1e-9 {
            weighted_sum += weights[2] * parkinson_vol[i];
            weight_sum += weights[2];
        }
        
        // ATR
        if atr_vol[i] > 1e-9 {
            weighted_sum += weights[3] * atr_vol[i];
            weight_sum += weights[3];
        }
        
        // Calculate weighted average
        if weight_sum > 0.0 {
            combined_vol[i] = weighted_sum / weight_sum;
        }
    }
    
    Ok(combined_vol)
}

/// Calculate volatility regime score (0-1)
fn calculate_volatility_regime(
    combined_vol: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = combined_vol.len();
    let mut vol_regime = Array1::zeros(n);
    let long_period = (params.vol_period as f64 * params.vol_lookback_factor) as usize;
    
    // Calculate rolling historical average
    for i in long_period..n {
        let start_idx = i.saturating_sub(long_period);
        let window = &combined_vol.slice(s![start_idx..i]);
        
        if window.len() > 0 {
            let historical_avg = window.mean().unwrap_or(1e-6);
            let current_vol = combined_vol[i];
            
            // Calculate regime as ratio of current to historical
            let regime_ratio = if historical_avg > 1e-9 {
                current_vol / historical_avg
            } else {
                1.0
            };
            
            // Transform to 0-1 scale using log transformation
            let log_regime = regime_ratio.max(1e-9).ln();
            vol_regime[i] = (0.5 + log_regime / 2.0).clamp(0.0, 1.0);
        }
    }
    
    Ok(vol_regime)
}

/// Calculate volatility rate of change
fn calculate_volatility_roc(
    vol_regime: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<Array1<f64>> {
    let n = vol_regime.len();
    let mut vol_roc = Array1::zeros(n);
    
    // Calculate percentage change
    for i in 1..n {
        if vol_regime[i-1] > 1e-9 {
            vol_roc[i] = (vol_regime[i] - vol_regime[i-1]) / vol_regime[i-1];
        }
    }
    
    // Apply exponential smoothing
    let smoothing_period = params.vol_period / 3;
    let alpha = 2.0 / (smoothing_period as f64 + 1.0);
    let mut vol_roc_smoothed = Array1::zeros(n);
    
    if n > 0 {
        vol_roc_smoothed[0] = vol_roc[0];
        for i in 1..n {
            vol_roc_smoothed[i] = alpha * vol_roc[i] + (1.0 - alpha) * vol_roc_smoothed[i-1];
        }
    }
    
    Ok(vol_roc_smoothed)
}

/// Calculate volatility persistence (Hurst exponent approximation)
pub fn calculate_volatility_persistence(volatility: &Array1<f64>) -> AntifragilityResult<f64> {
    let n = volatility.len();
    if n < 20 {
        return Err(AntifragilityError::InsufficientData {
            required: 20,
            actual: n,
        });
    }
    
    let log_returns = volatility.windows(2)
        .into_iter()
        .map(|window| (window[1] / window[0]).ln())
        .collect::<Vec<_>>();
    
    let mut rs_values = Vec::new();
    let lags = [5, 10, 20, 50].iter().filter(|&&lag| lag < n / 2);
    
    for &lag in lags {
        let mut cumulative_deviations = Vec::new();
        let mut running_sum = 0.0;
        let mean_return = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        
        for i in 0..lag {
            running_sum += log_returns[i] - mean_return;
            cumulative_deviations.push(running_sum);
        }
        
        let range = cumulative_deviations.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                  - cumulative_deviations.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        let std_dev = {
            let variance = log_returns.iter().take(lag)
                .map(|&x| (x - mean_return).powi(2))
                .sum::<f64>() / (lag - 1) as f64;
            variance.sqrt()
        };
        
        if std_dev > 1e-9 {
            rs_values.push((lag as f64, range / std_dev));
        }
    }
    
    if rs_values.len() < 2 {
        return Ok(0.5); // Default to random walk assumption
    }
    
    // Calculate Hurst exponent using linear regression on log-log scale
    let n_points = rs_values.len() as f64;
    let sum_log_lag = rs_values.iter().map(|(lag, _)| lag.ln()).sum::<f64>();
    let sum_log_rs = rs_values.iter().map(|(_, rs)| rs.ln()).sum::<f64>();
    let sum_log_lag_squared = rs_values.iter().map(|(lag, _)| lag.ln().powi(2)).sum::<f64>();
    let sum_log_lag_log_rs = rs_values.iter().map(|(lag, rs)| lag.ln() * rs.ln()).sum::<f64>();
    
    let hurst = (n_points * sum_log_lag_log_rs - sum_log_lag * sum_log_rs) /
                (n_points * sum_log_lag_squared - sum_log_lag.powi(2));
    
    Ok(hurst.clamp(0.0, 1.0))
}

/// Detect volatility regimes using change point detection
pub fn detect_volatility_regimes(volatility: &Array1<f64>, min_segment_length: usize) -> Vec<usize> {
    let n = volatility.len();
    let mut change_points = Vec::new();
    
    if n < min_segment_length * 2 {
        return change_points;
    }
    
    // Simple change point detection using variance differences
    let window_size = min_segment_length;
    
    for i in window_size..(n - window_size) {
        let left_window = volatility.slice(s![(i - window_size)..i]);
        let right_window = volatility.slice(s![i..(i + window_size)]);
        
        let left_mean = left_window.mean().unwrap_or(0.0);
        let right_mean = right_window.mean().unwrap_or(0.0);
        
        let left_var = left_window.mapv(|x| (x - left_mean).powi(2)).mean().unwrap_or(0.0);
        let right_var = right_window.mapv(|x| (x - right_mean).powi(2)).mean().unwrap_or(0.0);
        
        // Test for significant difference in means and variances
        let mean_diff = (left_mean - right_mean).abs();
        let var_ratio = if right_var > 1e-9 { left_var / right_var } else { 1.0 };
        
        // Threshold for change point detection
        if mean_diff > 0.1 * (left_mean + right_mean) / 2.0 || 
           var_ratio > 2.0 || var_ratio < 0.5 {
            change_points.push(i);
        }
    }
    
    change_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn generate_test_prices(n: usize) -> Array1<f64> {
        let mut prices = Array1::zeros(n);
        let mut price = 100.0;
        
        for i in 0..n {
            let return_rate = 0.01 * ((i as f64) * 0.1).sin() + 0.005 * ((i as f64) * 0.3).cos();
            price *= 1.0 + return_rate;
            prices[i] = price;
        }
        
        prices
    }
    
    #[test]
    fn test_yang_zhang_volatility() {
        let prices = generate_test_prices(100);
        let volumes = Array1::ones(100);
        let params = AntifragilityParameters::default();
        
        let result = calculate_yang_zhang_volatility(&prices, &volumes, &params);
        assert!(result.is_ok());
        
        let vol = result.unwrap();
        assert_eq!(vol.len(), 100);
        assert!(vol.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_garch_volatility() {
        let prices = generate_test_prices(100);
        let params = AntifragilityParameters::default();
        
        let result = calculate_garch_volatility(&prices, &params);
        assert!(result.is_ok());
        
        let vol = result.unwrap();
        assert_eq!(vol.len(), 100);
        assert!(vol.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_parkinson_volatility() {
        let prices = generate_test_prices(100);
        let params = AntifragilityParameters::default();
        
        let result = calculate_parkinson_volatility(&prices, &params);
        assert!(result.is_ok());
        
        let vol = result.unwrap();
        assert_eq!(vol.len(), 100);
        assert!(vol.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_atr_volatility() {
        let prices = generate_test_prices(100);
        let params = AntifragilityParameters::default();
        
        let result = calculate_atr_volatility(&prices, &params);
        assert!(result.is_ok());
        
        let vol = result.unwrap();
        assert_eq!(vol.len(), 100);
        assert!(vol.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_robust_volatility() {
        let prices = generate_test_prices(100);
        let volumes = Array1::ones(100);
        let params = AntifragilityParameters::default();
        
        let result = calculate_robust_volatility(&prices, &volumes, &params);
        assert!(result.is_ok());
        
        let vol_result = result.unwrap();
        assert_eq!(vol_result.combined_vol.len(), 100);
        assert_eq!(vol_result.vol_regime.len(), 100);
        assert!(vol_result.vol_regime.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_volatility_persistence() {
        let volatility = generate_test_prices(100);
        let result = calculate_volatility_persistence(&volatility);
        assert!(result.is_ok());
        
        let hurst = result.unwrap();
        assert!(hurst >= 0.0 && hurst <= 1.0);
    }
    
    #[test]
    fn test_volatility_regime_detection() {
        let mut volatility = Array1::ones(100);
        
        // Create artificial regime change
        for i in 50..100 {
            volatility[i] = 2.0;
        }
        
        let change_points = detect_volatility_regimes(&volatility, 10);
        assert!(!change_points.is_empty());
        assert!(change_points.iter().any(|&cp| cp >= 40 && cp <= 60));
    }
    
    #[test]
    fn test_insufficient_data() {
        let prices = Array1::ones(10);
        let volumes = Array1::ones(10);
        let params = AntifragilityParameters::default();
        
        let result = calculate_robust_volatility(&prices, &volumes, &params);
        assert!(result.is_err());
        
        if let Err(AntifragilityError::InsufficientData { required, actual }) = result {
            assert_eq!(actual, 10);
            assert!(required > 10);
        }
    }
}