//! Recovery velocity analysis module
//!
//! This module calculates how well performance recovers after volatility spikes.
//! Antifragile systems should show strong recovery velocity, performing well
//! in the periods following volatility increases.

use crate::{AntifragilityError, AntifragilityParameters, AntifragilityResult};
use ndarray::prelude::*;
use num_traits::Float;

/// Calculate recovery velocity component
pub fn calculate_recovery_velocity(
    prices: &Array1<f64>,
    vol_roc_smoothed: &Array1<f64>,
    params: &AntifragilityParameters,
) -> AntifragilityResult<f64> {
    let n = prices.len();
    
    if n != vol_roc_smoothed.len() {
        return Err(AntifragilityError::InvalidParameters {
            message: format!("Array lengths must match: {} vs {}", n, vol_roc_smoothed.len()),
        });
    }
    
    let future_horizon = (params.vol_period as f64 * params.recovery_horizon_factor) as usize;
    
    if n < future_horizon + params.corr_window {
        return Err(AntifragilityError::InsufficientData {
            required: future_horizon + params.corr_window,
            actual: n,
        });
    }
    
    // Calculate future performance after volatility changes
    let future_performance = calculate_future_performance(prices, future_horizon)?;
    
    // Calculate correlation between current volatility change and future performance
    let recovery_correlation = calculate_recovery_correlation(
        vol_roc_smoothed,
        &future_performance,
        params.corr_window,
    )?;
    
    // Convert correlation to 0-1 scale
    Ok((recovery_correlation + 1.0) / 2.0)
}

/// Calculate future performance at specified horizon
fn calculate_future_performance(
    prices: &Array1<f64>,
    horizon: usize,
) -> AntifragilityResult<Array1<f64>> {
    let n = prices.len();
    let mut future_perf = Array1::zeros(n);
    
    for i in 0..(n - horizon) {
        if prices[i] > 0.0 {
            future_perf[i] = (prices[i + horizon] / prices[i]).ln();
        }
    }
    
    Ok(future_perf)
}

/// Calculate correlation between volatility changes and future performance
fn calculate_recovery_correlation(
    vol_roc: &Array1<f64>,
    future_perf: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = vol_roc.len();
    let mut correlations = Vec::new();
    
    for i in window..n {
        let vol_window = vol_roc.slice(s![(i - window)..i]);
        let perf_window = future_perf.slice(s![(i - window)..i]);
        
        let correlation = calculate_correlation(&vol_window, &perf_window)?;
        if correlation.is_finite() {
            correlations.push(correlation);
        }
    }
    
    if correlations.is_empty() {
        return Ok(0.0);
    }
    
    let mean_correlation = correlations.iter().sum::<f64>() / correlations.len() as f64;
    Ok(mean_correlation)
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> AntifragilityResult<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return Ok(0.0);
    }
    
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);
    
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator > 1e-9 {
        Ok(sum_xy / denominator)
    } else {
        Ok(0.0)
    }
}

/// Calculate recovery time after volatility spikes
pub fn calculate_recovery_time(
    prices: &Array1<f64>,
    volatility: &Array1<f64>,
    spike_threshold: f64,
) -> AntifragilityResult<f64> {
    let n = prices.len();
    
    if n < 10 {
        return Err(AntifragilityError::InsufficientData {
            required: 10,
            actual: n,
        });
    }
    
    // Identify volatility spikes
    let vol_mean = volatility.mean().unwrap_or(0.0);
    let vol_std = volatility.std(1.0);
    let spike_level = vol_mean + spike_threshold * vol_std;
    
    let mut recovery_times = Vec::new();
    
    for i in 1..(n - 1) {
        if volatility[i] > spike_level && volatility[i-1] <= spike_level {
            // Found a volatility spike
            let pre_spike_price = prices[i];
            
            // Find recovery time (when price returns to pre-spike level)
            for j in (i + 1)..n {
                if prices[j] >= pre_spike_price {
                    let recovery_time = (j - i) as f64;
                    recovery_times.push(recovery_time);
                    break;
                }
            }
        }
    }
    
    if recovery_times.is_empty() {
        return Ok(0.0);
    }
    
    // Calculate average recovery time
    let avg_recovery_time = recovery_times.iter().sum::<f64>() / recovery_times.len() as f64;
    
    // Convert to score (shorter recovery time = higher score)
    let max_reasonable_recovery = 20.0; // 20 periods
    let normalized_time = (avg_recovery_time / max_reasonable_recovery).min(1.0);
    
    Ok(1.0 - normalized_time)
}

/// Calculate performance resilience during stress periods
pub fn calculate_resilience(
    returns: &Array1<f64>,
    volatility: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut resilience_scores = Vec::new();
    
    for i in window..n {
        let ret_window = returns.slice(s![(i - window)..i]);
        let vol_window = volatility.slice(s![(i - window)..i]);
        
        // Calculate resilience for this window
        let resilience = calculate_window_resilience(&ret_window, &vol_window)?;
        if resilience.is_finite() {
            resilience_scores.push(resilience);
        }
    }
    
    if resilience_scores.is_empty() {
        return Ok(0.5);
    }
    
    let mean_resilience = resilience_scores.iter().sum::<f64>() / resilience_scores.len() as f64;
    Ok(mean_resilience.clamp(0.0, 1.0))
}

/// Calculate resilience for a single window
fn calculate_window_resilience(
    returns: &ArrayView1<f64>,
    volatility: &ArrayView1<f64>,
) -> AntifragilityResult<f64> {
    let n = returns.len();
    
    if n < 5 {
        return Ok(0.5);
    }
    
    // Identify high volatility periods
    let vol_mean = volatility.mean().unwrap_or(0.0);
    let vol_threshold = vol_mean * 1.5; // 1.5x average volatility
    
    let mut stress_returns = Vec::new();
    let mut normal_returns = Vec::new();
    
    for i in 0..n {
        if volatility[i] > vol_threshold {
            stress_returns.push(returns[i]);
        } else {
            normal_returns.push(returns[i]);
        }
    }
    
    if stress_returns.is_empty() || normal_returns.is_empty() {
        return Ok(0.5);
    }
    
    // Calculate average returns in each regime
    let stress_avg = stress_returns.iter().sum::<f64>() / stress_returns.len() as f64;
    let normal_avg = normal_returns.iter().sum::<f64>() / normal_returns.len() as f64;
    
    // Resilience is how well returns hold up during stress
    let resilience = if normal_avg != 0.0 {
        (stress_avg / normal_avg + 1.0) / 2.0
    } else {
        0.5
    };
    
    Ok(resilience.clamp(0.0, 1.0))
}

/// Calculate recovery patterns after different types of shocks
pub fn calculate_shock_recovery_patterns(
    prices: &Array1<f64>,
    volumes: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<ShockRecoveryAnalysis> {
    let n = prices.len();
    
    if n < window * 2 {
        return Err(AntifragilityError::InsufficientData {
            required: window * 2,
            actual: n,
        });
    }
    
    // Calculate price returns
    let mut returns = Array1::zeros(n);
    for i in 1..n {
        returns[i] = (prices[i] / prices[i-1]).ln();
    }
    
    // Calculate volume changes
    let mut vol_changes = Array1::zeros(n);
    for i in 1..n {
        if volumes[i-1] > 0.0 {
            vol_changes[i] = (volumes[i] - volumes[i-1]) / volumes[i-1];
        }
    }
    
    let mut price_shock_recovery = Vec::new();
    let mut volume_shock_recovery = Vec::new();
    
    // Identify price shocks
    let ret_std = returns.std(1.0);
    let price_shock_threshold = 2.0 * ret_std; // 2 standard deviations
    
    for i in 1..(n - window) {
        if returns[i].abs() > price_shock_threshold {
            let recovery = calculate_post_shock_recovery(&prices, i, window)?;
            price_shock_recovery.push(recovery);
        }
    }
    
    // Identify volume shocks
    let vol_std = vol_changes.std(1.0);
    let volume_shock_threshold = 2.0 * vol_std;
    
    for i in 1..(n - window) {
        if vol_changes[i].abs() > volume_shock_threshold {
            let recovery = calculate_post_shock_recovery(&prices, i, window)?;
            volume_shock_recovery.push(recovery);
        }
    }
    
    Ok(ShockRecoveryAnalysis {
        price_shock_recovery_avg: calculate_average(&price_shock_recovery),
        volume_shock_recovery_avg: calculate_average(&volume_shock_recovery),
        price_shock_count: price_shock_recovery.len(),
        volume_shock_count: volume_shock_recovery.len(),
    })
}

/// Calculate post-shock recovery
fn calculate_post_shock_recovery(
    prices: &Array1<f64>,
    shock_index: usize,
    recovery_window: usize,
) -> AntifragilityResult<f64> {
    let n = prices.len();
    
    if shock_index + recovery_window >= n {
        return Ok(0.0);
    }
    
    let pre_shock_price = prices[shock_index];
    let post_shock_prices = prices.slice(s![(shock_index + 1)..(shock_index + recovery_window + 1)]);
    
    // Calculate recovery as the maximum price achieved relative to pre-shock
    let max_recovery = post_shock_prices.iter()
        .map(|&p| p / pre_shock_price)
        .fold(0.0, |acc, x| acc.max(x));
    
    Ok(max_recovery)
}

/// Calculate average of a vector
fn calculate_average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Shock recovery analysis results
#[derive(Debug, Clone)]
pub struct ShockRecoveryAnalysis {
    pub price_shock_recovery_avg: f64,
    pub volume_shock_recovery_avg: f64,
    pub price_shock_count: usize,
    pub volume_shock_count: usize,
}

impl ShockRecoveryAnalysis {
    /// Calculate overall recovery score
    pub fn overall_score(&self) -> f64 {
        let price_score = if self.price_shock_count > 0 {
            (self.price_shock_recovery_avg - 1.0).tanh() * 0.5 + 0.5
        } else {
            0.5
        };
        
        let volume_score = if self.volume_shock_count > 0 {
            (self.volume_shock_recovery_avg - 1.0).tanh() * 0.5 + 0.5
        } else {
            0.5
        };
        
        // Weight price shocks more heavily
        (price_score * 0.7 + volume_score * 0.3).clamp(0.0, 1.0)
    }
}

/// Calculate adaptive recovery based on shock magnitude
pub fn calculate_adaptive_recovery(
    prices: &Array1<f64>,
    window: usize,
    recovery_horizons: &[usize],
) -> AntifragilityResult<f64> {
    let n = prices.len();
    
    if n < window * 2 {
        return Err(AntifragilityError::InsufficientData {
            required: window * 2,
            actual: n,
        });
    }
    
    let mut recovery_scores = Vec::new();
    
    for &horizon in recovery_horizons {
        if n > window + horizon {
            let score = calculate_horizon_recovery(prices, window, horizon)?;
            recovery_scores.push(score);
        }
    }
    
    if recovery_scores.is_empty() {
        return Ok(0.5);
    }
    
    // Weight shorter horizons more heavily
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    
    for (i, &score) in recovery_scores.iter().enumerate() {
        let weight = 1.0 / (i as f64 + 1.0); // Decreasing weights
        weighted_sum += score * weight;
        weight_sum += weight;
    }
    
    Ok(weighted_sum / weight_sum)
}

/// Calculate recovery at specific horizon
fn calculate_horizon_recovery(
    prices: &Array1<f64>,
    window: usize,
    horizon: usize,
) -> AntifragilityResult<f64> {
    let n = prices.len();
    let mut recovery_values = Vec::new();
    
    for i in window..(n - horizon) {
        let current_price = prices[i];
        let future_price = prices[i + horizon];
        
        if current_price > 0.0 {
            let recovery = (future_price / current_price).ln();
            recovery_values.push(recovery);
        }
    }
    
    if recovery_values.is_empty() {
        return Ok(0.5);
    }
    
    let mean_recovery = recovery_values.iter().sum::<f64>() / recovery_values.len() as f64;
    
    // Convert to 0-1 scale
    Ok((mean_recovery.tanh() + 1.0) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn generate_test_data() -> (Array1<f64>, Array1<f64>) {
        let n = 100;
        let mut prices = Array1::zeros(n);
        let mut vol_roc = Array1::zeros(n);
        
        let mut price = 100.0;
        for i in 0..n {
            let t = i as f64 * 0.1;
            
            // Create price movement with some recovery patterns
            let return_rate = 0.01 * t.sin() + 0.005 * (t * 2.0).cos();
            price *= 1.0 + return_rate;
            prices[i] = price;
            
            // Create volatility rate of change
            vol_roc[i] = 0.05 * (t * 0.5).sin() + 0.02 * (t * 3.0).cos();
        }
        
        (prices, vol_roc)
    }
    
    #[test]
    fn test_recovery_velocity() {
        let (prices, vol_roc) = generate_test_data();
        let params = AntifragilityParameters::default();
        
        let result = calculate_recovery_velocity(&prices, &vol_roc, &params);
        assert!(result.is_ok());
        
        let recovery = result.unwrap();
        assert!(recovery >= 0.0 && recovery <= 1.0);
    }
    
    #[test]
    fn test_future_performance() {
        let prices = Array1::from_vec(vec![100.0, 105.0, 110.0, 115.0, 120.0]);
        let result = calculate_future_performance(&prices, 2);
        assert!(result.is_ok());
        
        let future_perf = result.unwrap();
        assert_eq!(future_perf.len(), 5);
        
        // Check first value (100 -> 110)
        assert_relative_eq!(future_perf[0], (110.0 / 100.0).ln(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_recovery_time() {
        let mut prices = Array1::from_vec(vec![100.0, 90.0, 85.0, 95.0, 105.0, 110.0]);
        let mut volatility = Array1::from_vec(vec![0.1, 0.3, 0.4, 0.2, 0.1, 0.1]);
        
        let result = calculate_recovery_time(&prices, &volatility, 1.0);
        assert!(result.is_ok());
        
        let recovery_score = result.unwrap();
        assert!(recovery_score >= 0.0 && recovery_score <= 1.0);
    }
    
    #[test]
    fn test_resilience() {
        let n = 50;
        let mut returns = Array1::zeros(n);
        let mut volatility = Array1::zeros(n);
        
        for i in 0..n {
            let t = i as f64 * 0.1;
            returns[i] = 0.01 * t.sin();
            volatility[i] = 0.1 + 0.05 * (t * 2.0).sin().abs();
        }
        
        let result = calculate_resilience(&returns, &volatility, 10);
        assert!(result.is_ok());
        
        let resilience = result.unwrap();
        assert!(resilience >= 0.0 && resilience <= 1.0);
    }
    
    #[test]
    fn test_shock_recovery_patterns() {
        let (prices, _) = generate_test_data();
        let volumes = Array1::ones(100) * 1000.0;
        
        let result = calculate_shock_recovery_patterns(&prices, &volumes, 10);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.price_shock_recovery_avg >= 0.0);
        assert!(analysis.volume_shock_recovery_avg >= 0.0);
        
        let score = analysis.overall_score();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_adaptive_recovery() {
        let (prices, _) = generate_test_data();
        let horizons = vec![5, 10, 15, 20];
        
        let result = calculate_adaptive_recovery(&prices, 10, &horizons);
        assert!(result.is_ok());
        
        let recovery = result.unwrap();
        assert!(recovery >= 0.0 && recovery <= 1.0);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        
        let corr = calculate_correlation(&x.view(), &y.view()).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
        
        let y_inverse = Array1::from_vec(vec![10.0, 8.0, 6.0, 4.0, 2.0]);
        let corr_inverse = calculate_correlation(&x.view(), &y_inverse.view()).unwrap();
        assert_relative_eq!(corr_inverse, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_insufficient_data() {
        let small_prices = Array1::from_vec(vec![100.0, 101.0]);
        let small_vol_roc = Array1::from_vec(vec![0.1, 0.2]);
        let params = AntifragilityParameters::default();
        
        let result = calculate_recovery_velocity(&small_prices, &small_vol_roc, &params);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mismatched_lengths() {
        let prices = Array1::from_vec(vec![100.0, 101.0, 102.0]);
        let vol_roc = Array1::from_vec(vec![0.1, 0.2]);
        let params = AntifragilityParameters::default();
        
        let result = calculate_recovery_velocity(&prices, &vol_roc, &params);
        assert!(result.is_err());
    }
}