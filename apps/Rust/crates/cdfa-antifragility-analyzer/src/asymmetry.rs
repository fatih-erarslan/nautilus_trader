//! Asymmetry analysis module
//!
//! This module calculates the asymmetry component of antifragility, which measures
//! skewness and kurtosis under different volatility regimes.
//!
//! Systems with favorable asymmetry show positive skewness during stress periods,
//! indicating better upside potential when volatility increases.

use crate::{AntifragilityError, AntifragilityResult};
use ndarray::prelude::*;
use num_traits::Float;

/// Calculate weighted asymmetry based on skewness and kurtosis
pub fn calculate_weighted_asymmetry(
    log_returns: &Array1<f64>,
    vol_regime: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<f64> {
    let n = log_returns.len();
    
    if n != vol_regime.len() {
        return Err(AntifragilityError::InvalidParameters {
            message: format!("Array lengths must match: {} vs {}", n, vol_regime.len()),
        });
    }
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    // Calculate rolling skewness and kurtosis
    let (weighted_skew, rolling_kurt) = calculate_rolling_moments(log_returns, vol_regime, window)?;
    
    // Calculate asymmetry score
    let asymmetry_score = calculate_asymmetry_score(&weighted_skew, &rolling_kurt, vol_regime)?;
    
    Ok(asymmetry_score)
}

/// Calculate rolling statistical moments
fn calculate_rolling_moments(
    returns: &Array1<f64>,
    vol_regime: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<(Array1<f64>, Array1<f64>)> {
    let n = returns.len();
    let mut rolling_skew = Array1::zeros(n);
    let mut rolling_kurt = Array1::zeros(n);
    
    for i in window..n {
        let window_returns = returns.slice(s![(i - window)..i]);
        let window_vol_regime = vol_regime.slice(s![(i - window)..i]);
        
        // Calculate basic moments
        let mean_ret = window_returns.mean().unwrap_or(0.0);
        let variance = window_returns.mapv(|x| (x - mean_ret).powi(2)).mean().unwrap_or(0.0);
        let std_dev = variance.sqrt();
        
        if std_dev > 1e-9 {
            // Calculate skewness
            let skewness = window_returns
                .mapv(|x| ((x - mean_ret) / std_dev).powi(3))
                .mean()
                .unwrap_or(0.0);
            
            // Calculate kurtosis
            let kurtosis = window_returns
                .mapv(|x| ((x - mean_ret) / std_dev).powi(4))
                .mean()
                .unwrap_or(3.0);
            
            // Weight by volatility regime
            let avg_vol_regime = window_vol_regime.mean().unwrap_or(0.5);
            rolling_skew[i] = skewness * avg_vol_regime;
            rolling_kurt[i] = kurtosis;
        } else {
            rolling_skew[i] = 0.0;
            rolling_kurt[i] = 3.0; // Normal distribution kurtosis
        }
    }
    
    Ok((rolling_skew, rolling_kurt))
}

/// Calculate final asymmetry score
fn calculate_asymmetry_score(
    weighted_skew: &Array1<f64>,
    rolling_kurt: &Array1<f64>,
    vol_regime: &Array1<f64>,
) -> AntifragilityResult<f64> {
    let n = weighted_skew.len();
    
    if n == 0 {
        return Ok(0.5);
    }
    
    // Calculate favorable tail profile from skewness
    let mean_skew = weighted_skew.mean().unwrap_or(0.0);
    let favorable_tail_profile = (mean_skew.tanh() + 1.0) / 2.0;
    
    // Calculate kurtosis penalty
    let mean_kurt = rolling_kurt.mean().unwrap_or(3.0);
    let avg_vol_regime = vol_regime.mean().unwrap_or(0.5);
    
    let excess_kurtosis = (mean_kurt - 3.0).max(0.0);
    let kurt_penalty_factor = 1.0 - (excess_kurtosis * 0.2 * avg_vol_regime * (1.0 - favorable_tail_profile));
    let kurt_penalty_factor = kurt_penalty_factor.max(0.0);
    
    // Combine components
    let asymmetry_score = favorable_tail_profile * kurt_penalty_factor;
    
    Ok(asymmetry_score.clamp(0.0, 1.0))
}

/// Calculate regime-conditional skewness
pub fn calculate_regime_conditional_skewness(
    returns: &Array1<f64>,
    vol_regime: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<(f64, f64)> {
    let n = returns.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut high_vol_skew = Vec::new();
    let mut low_vol_skew = Vec::new();
    
    for i in window..n {
        let window_returns = returns.slice(s![(i - window)..i]);
        let window_vol_regime = vol_regime.slice(s![(i - window)..i]);
        
        // Separate high and low volatility periods
        let mut high_vol_returns = Vec::new();
        let mut low_vol_returns = Vec::new();
        
        for j in 0..window_returns.len() {
            if window_vol_regime[j] > 0.6 {
                high_vol_returns.push(window_returns[j]);
            } else if window_vol_regime[j] < 0.4 {
                low_vol_returns.push(window_returns[j]);
            }
        }
        
        // Calculate skewness for each regime
        if high_vol_returns.len() >= 10 {
            let skew = calculate_skewness(&high_vol_returns)?;
            high_vol_skew.push(skew);
        }
        
        if low_vol_returns.len() >= 10 {
            let skew = calculate_skewness(&low_vol_returns)?;
            low_vol_skew.push(skew);
        }
    }
    
    let mean_high_vol_skew = if high_vol_skew.is_empty() {
        0.0
    } else {
        high_vol_skew.iter().sum::<f64>() / high_vol_skew.len() as f64
    };
    
    let mean_low_vol_skew = if low_vol_skew.is_empty() {
        0.0
    } else {
        low_vol_skew.iter().sum::<f64>() / low_vol_skew.len() as f64
    };
    
    Ok((mean_high_vol_skew, mean_low_vol_skew))
}

/// Calculate skewness for a vector of values
fn calculate_skewness(values: &[f64]) -> AntifragilityResult<f64> {
    if values.len() < 3 {
        return Ok(0.0);
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    if std_dev < 1e-9 {
        return Ok(0.0);
    }
    
    let skewness = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / values.len() as f64;
    
    Ok(skewness)
}

/// Calculate kurtosis for a vector of values
fn calculate_kurtosis(values: &[f64]) -> AntifragilityResult<f64> {
    if values.len() < 4 {
        return Ok(3.0);
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    if std_dev < 1e-9 {
        return Ok(3.0);
    }
    
    let kurtosis = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / values.len() as f64;
    
    Ok(kurtosis)
}

/// Calculate higher moments (skewness and kurtosis) simultaneously
pub fn calculate_higher_moments(values: &[f64]) -> AntifragilityResult<(f64, f64)> {
    if values.len() < 4 {
        return Ok((0.0, 3.0));
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    if std_dev < 1e-9 {
        return Ok((0.0, 3.0));
    }
    
    let mut skewness_sum = 0.0;
    let mut kurtosis_sum = 0.0;
    
    for &value in values {
        let standardized = (value - mean) / std_dev;
        let standardized_cubed = standardized.powi(3);
        let standardized_fourth = standardized_cubed * standardized;
        
        skewness_sum += standardized_cubed;
        kurtosis_sum += standardized_fourth;
    }
    
    let n = values.len() as f64;
    let skewness = skewness_sum / n;
    let kurtosis = kurtosis_sum / n;
    
    Ok((skewness, kurtosis))
}

/// Calculate asymmetry under different market conditions
pub fn calculate_conditional_asymmetry(
    returns: &Array1<f64>,
    vol_regime: &Array1<f64>,
    price_trend: &Array1<f64>,
    window: usize,
) -> AntifragilityResult<AsymmetryBreakdown> {
    let n = returns.len();
    
    if n < window {
        return Err(AntifragilityError::InsufficientData {
            required: window,
            actual: n,
        });
    }
    
    let mut bull_high_vol = Vec::new();
    let mut bull_low_vol = Vec::new();
    let mut bear_high_vol = Vec::new();
    let mut bear_low_vol = Vec::new();
    
    for i in window..n {
        let window_returns = returns.slice(s![(i - window)..i]);
        let window_vol_regime = vol_regime.slice(s![(i - window)..i]);
        let window_trend = price_trend.slice(s![(i - window)..i]);
        
        for j in 0..window_returns.len() {
            let is_bull = window_trend[j] > 0.0;
            let is_high_vol = window_vol_regime[j] > 0.6;
            let return_val = window_returns[j];
            
            match (is_bull, is_high_vol) {
                (true, true) => bull_high_vol.push(return_val),
                (true, false) => bull_low_vol.push(return_val),
                (false, true) => bear_high_vol.push(return_val),
                (false, false) => bear_low_vol.push(return_val),
            }
        }
    }
    
    let bull_high_vol_skew = if bull_high_vol.len() >= 10 {
        calculate_skewness(&bull_high_vol)?
    } else {
        0.0
    };
    
    let bull_low_vol_skew = if bull_low_vol.len() >= 10 {
        calculate_skewness(&bull_low_vol)?
    } else {
        0.0
    };
    
    let bear_high_vol_skew = if bear_high_vol.len() >= 10 {
        calculate_skewness(&bear_high_vol)?
    } else {
        0.0
    };
    
    let bear_low_vol_skew = if bear_low_vol.len() >= 10 {
        calculate_skewness(&bear_low_vol)?
    } else {
        0.0
    };
    
    Ok(AsymmetryBreakdown {
        bull_high_vol: bull_high_vol_skew,
        bull_low_vol: bull_low_vol_skew,
        bear_high_vol: bear_high_vol_skew,
        bear_low_vol: bear_low_vol_skew,
    })
}

/// Asymmetry breakdown by market conditions
#[derive(Debug, Clone)]
pub struct AsymmetryBreakdown {
    pub bull_high_vol: f64,
    pub bull_low_vol: f64,
    pub bear_high_vol: f64,
    pub bear_low_vol: f64,
}

impl AsymmetryBreakdown {
    /// Calculate overall asymmetry score
    pub fn overall_score(&self) -> f64 {
        let weights = [0.3, 0.2, 0.3, 0.2]; // Emphasize stress periods
        let values = [self.bull_high_vol, self.bull_low_vol, self.bear_high_vol, self.bear_low_vol];
        
        let weighted_sum = values.iter().zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>();
        
        // Transform to 0-1 scale
        (weighted_sum.tanh() + 1.0) / 2.0
    }
    
    /// Check if asymmetry is favorable (positive skewness during stress)
    pub fn is_favorable(&self) -> bool {
        // Check if high volatility periods show positive skewness
        (self.bull_high_vol > 0.1) || (self.bear_high_vol > -0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn generate_test_data() -> (Array1<f64>, Array1<f64>) {
        let n = 100;
        let mut returns = Array1::zeros(n);
        let mut vol_regime = Array1::zeros(n);
        
        for i in 0..n {
            let t = i as f64 * 0.1;
            
            // Create returns with some skewness
            let base_return = 0.01 * t.sin();
            let skewed_component = if (t * 2.0).sin() > 0.0 {
                0.02 * (t * 2.0).sin().powi(2)
            } else {
                0.01 * (t * 2.0).sin()
            };
            
            returns[i] = base_return + skewed_component;
            vol_regime[i] = (0.5 + 0.3 * (t * 0.5).sin()).clamp(0.0, 1.0);
        }
        
        (returns, vol_regime)
    }
    
    #[test]
    fn test_weighted_asymmetry() {
        let (returns, vol_regime) = generate_test_data();
        
        let result = calculate_weighted_asymmetry(&returns, &vol_regime, 20);
        assert!(result.is_ok());
        
        let asymmetry = result.unwrap();
        assert!(asymmetry >= 0.0 && asymmetry <= 1.0);
    }
    
    #[test]
    fn test_rolling_moments() {
        let (returns, vol_regime) = generate_test_data();
        
        let result = calculate_rolling_moments(&returns, &vol_regime, 20);
        assert!(result.is_ok());
        
        let (skew, kurt) = result.unwrap();
        assert_eq!(skew.len(), 100);
        assert_eq!(kurt.len(), 100);
        
        // Check that kurtosis values are reasonable
        for i in 20..100 {
            assert!(kurt[i] >= 1.0); // Kurtosis should be >= 1
        }
    }
    
    #[test]
    fn test_regime_conditional_skewness() {
        let (returns, vol_regime) = generate_test_data();
        
        let result = calculate_regime_conditional_skewness(&returns, &vol_regime, 20);
        assert!(result.is_ok());
        
        let (high_vol_skew, low_vol_skew) = result.unwrap();
        assert!(high_vol_skew.is_finite());
        assert!(low_vol_skew.is_finite());
    }
    
    #[test]
    fn test_skewness_calculation() {
        // Test with known skewed data
        let symmetric_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let skew = calculate_skewness(&symmetric_data).unwrap();
        assert_relative_eq!(skew, 0.0, epsilon = 1e-10);
        
        // Test with right-skewed data
        let right_skewed = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let skew = calculate_skewness(&right_skewed).unwrap();
        assert!(skew > 0.0);
        
        // Test with left-skewed data
        let left_skewed = vec![1.0, 9.0, 10.0, 10.0, 10.0];
        let skew = calculate_skewness(&left_skewed).unwrap();
        assert!(skew < 0.0);
    }
    
    #[test]
    fn test_kurtosis_calculation() {
        // Test with normal-like data
        let normal_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kurt = calculate_kurtosis(&normal_data).unwrap();
        assert!(kurt >= 1.0);
        
        // Test with high kurtosis data (heavy tails)
        let heavy_tails = vec![3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 5.0];
        let kurt = calculate_kurtosis(&heavy_tails).unwrap();
        assert!(kurt > 3.0);
    }
    
    #[test]
    fn test_higher_moments() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = calculate_higher_moments(&data);
        assert!(result.is_ok());
        
        let (skew, kurt) = result.unwrap();
        assert!(skew.is_finite());
        assert!(kurt.is_finite());
        assert!(kurt >= 1.0);
    }
    
    #[test]
    fn test_conditional_asymmetry() {
        let (returns, vol_regime) = generate_test_data();
        let price_trend = Array1::from_iter((0..100).map(|i| (i as f64 - 50.0) / 10.0));
        
        let result = calculate_conditional_asymmetry(&returns, &vol_regime, &price_trend, 20);
        assert!(result.is_ok());
        
        let breakdown = result.unwrap();
        assert!(breakdown.bull_high_vol.is_finite());
        assert!(breakdown.bull_low_vol.is_finite());
        assert!(breakdown.bear_high_vol.is_finite());
        assert!(breakdown.bear_low_vol.is_finite());
        
        let score = breakdown.overall_score();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_insufficient_data() {
        let small_returns = Array1::from_vec(vec![1.0, 2.0]);
        let small_vol_regime = Array1::from_vec(vec![0.5, 0.6]);
        
        let result = calculate_weighted_asymmetry(&small_returns, &small_vol_regime, 10);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mismatched_lengths() {
        let returns = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let vol_regime = Array1::from_vec(vec![0.5, 0.6]);
        
        let result = calculate_weighted_asymmetry(&returns, &vol_regime, 2);
        assert!(result.is_err());
    }
}