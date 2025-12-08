//! PCR (Potential, Connectedness, Resilience) calculation module

use crate::simd::{simd_autocorrelation, simd_mean, simd_std_dev, SimdRollingWindow};
use crate::types::{PCRComponents, MarketData};
use crate::PanarchyError;
use ndarray::{Array1, ArrayView1};

/// Calculate PCR components from market data
pub fn calculate_pcr_components(
    prices: &[f64],
    returns: &[f64],
    volatilities: &[f64],
    period: usize,
    autocorr_lag: usize,
) -> Result<Vec<PCRComponents>, PanarchyError> {
    let n = prices.len();
    
    if n < period + 1 {
        return Err(PanarchyError::InsufficientData {
            required: period + 1,
            actual: n,
        });
    }
    
    let mut results = Vec::with_capacity(n);
    
    // Initialize with default values for the first `period` elements
    for _ in 0..period.min(n) {
        results.push(PCRComponents::new(0.5, 0.5, 0.5));
    }
    
    // Create rolling windows for efficient calculation
    let mut price_window = SimdRollingWindow::new(period);
    let mut return_window = SimdRollingWindow::new(period);
    
    // Fill initial windows
    for i in 0..period.min(n) {
        price_window.push(prices[i]);
        if i < returns.len() {
            return_window.push(returns[i]);
        }
    }
    
    // Calculate PCR for each point after the initial period
    for i in period..n {
        // Update windows
        price_window.push(prices[i]);
        if i < returns.len() {
            return_window.push(returns[i]);
        }
        
        // Calculate P (Potential)
        let (min, max) = price_window.min_max();
        let potential = if max - min > 1e-9 {
            (prices[i] - min) / (max - min)
        } else {
            0.5
        };
        
        // Calculate C (Connectedness)
        let connectedness = calculate_connectedness(&returns[i.saturating_sub(period)..=i.min(returns.len() - 1)], autocorr_lag);
        
        // Calculate R (Resilience)
        let resilience = if i < volatilities.len() {
            1.0 - volatilities[i].clamp(0.0, 1.0)
        } else {
            0.5
        };
        
        results.push(PCRComponents::new(potential, connectedness, resilience));
    }
    
    Ok(results)
}

/// Calculate connectedness using autocorrelation
#[inline]
fn calculate_connectedness(returns: &[f64], lag: usize) -> f64 {
    if returns.len() <= lag {
        return 0.5;
    }
    
    let autocorr = simd_autocorrelation(returns, lag);
    // Normalize autocorrelation from [-1, 1] to [0, 1]
    (autocorr + 1.0) / 2.0
}

/// High-performance PCR calculation for full market data
pub fn calculate_pcr_batch(
    market_data: &[MarketData],
    period: usize,
    autocorr_lag: usize,
) -> Result<Vec<PCRComponents>, PanarchyError> {
    let n = market_data.len();
    
    if n < period + 1 {
        return Err(PanarchyError::InsufficientData {
            required: period + 1,
            actual: n,
        });
    }
    
    // Extract price arrays
    let prices: Vec<f64> = market_data.iter().map(|d| d.close).collect();
    
    // Calculate returns
    let returns = calculate_log_returns(&prices)?;
    
    // Calculate volatility
    let volatilities = calculate_volatility(&returns, period)?;
    
    calculate_pcr_components(&prices, &returns, &volatilities, period, autocorr_lag)
}

/// Calculate log returns from prices
pub fn calculate_log_returns(prices: &[f64]) -> Result<Vec<f64>, PanarchyError> {
    if prices.is_empty() {
        return Ok(vec![]);
    }
    
    let mut returns = vec![0.0]; // First return is 0
    
    for i in 1..prices.len() {
        if prices[i - 1] > 1e-9 {
            returns.push((prices[i] / prices[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }
    
    Ok(returns)
}

/// Calculate rolling volatility
pub fn calculate_volatility(returns: &[f64], period: usize) -> Result<Vec<f64>, PanarchyError> {
    let n = returns.len();
    let mut volatilities = vec![0.0; n];
    
    if n < period {
        return Ok(volatilities);
    }
    
    // Calculate rolling standard deviation
    for i in period..n {
        let window = &returns[i - period..i];
        let mean = simd_mean(window);
        let std_dev = simd_std_dev(window, mean);
        volatilities[i] = std_dev;
    }
    
    // Normalize to [0, 1]
    let max_vol = volatilities.iter().cloned().fold(0.0, f64::max);
    if max_vol > 1e-9 {
        for vol in &mut volatilities {
            *vol /= max_vol;
        }
    }
    
    Ok(volatilities)
}

/// Calculate ADX (Average Directional Index) for potential component
pub fn calculate_adx(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> Result<Vec<f64>, PanarchyError> {
    let n = highs.len();
    
    if n < period + 1 {
        return Err(PanarchyError::InsufficientData {
            required: period + 1,
            actual: n,
        });
    }
    
    if highs.len() != lows.len() || highs.len() != closes.len() {
        return Err(PanarchyError::InvalidParameters {
            message: "Highs, lows, and closes must have the same length".to_string(),
        });
    }
    
    let mut adx = vec![0.0; n];
    
    // Calculate True Range
    let mut true_ranges = vec![0.0; n];
    for i in 1..n {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        true_ranges[i] = high_low.max(high_close).max(low_close);
    }
    
    // Calculate directional movements
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    
    for i in 1..n {
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];
        
        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }
    }
    
    // Smooth with EMA
    let alpha = 1.0 / period as f64;
    let mut atr = simd_mean(&true_ranges[1..period + 1]);
    let mut plus_di_smooth = simd_mean(&plus_dm[1..period + 1]);
    let mut minus_di_smooth = simd_mean(&minus_dm[1..period + 1]);
    
    for i in period + 1..n {
        atr = atr * (1.0 - alpha) + true_ranges[i] * alpha;
        plus_di_smooth = plus_di_smooth * (1.0 - alpha) + plus_dm[i] * alpha;
        minus_di_smooth = minus_di_smooth * (1.0 - alpha) + minus_dm[i] * alpha;
        
        if atr > 1e-9 {
            let plus_di = 100.0 * plus_di_smooth / atr;
            let minus_di = 100.0 * minus_di_smooth / atr;
            
            let di_sum = plus_di + minus_di;
            if di_sum > 1e-9 {
                let dx = 100.0 * (plus_di - minus_di).abs() / di_sum;
                adx[i] = dx;
            }
        }
    }
    
    // Smooth ADX
    for i in 2 * period..n {
        let window = &adx[i - period..i];
        adx[i] = simd_mean(window);
    }
    
    Ok(adx)
}

/// Fast PCR calculation using pre-computed indicators
pub struct FastPCRCalculator {
    period: usize,
    autocorr_lag: usize,
    price_window: SimdRollingWindow,
    return_cache: Vec<f64>,
}

impl FastPCRCalculator {
    pub fn new(period: usize, autocorr_lag: usize) -> Self {
        Self {
            period,
            autocorr_lag,
            price_window: SimdRollingWindow::new(period),
            return_cache: Vec::with_capacity(period),
        }
    }
    
    pub fn update(&mut self, price: f64, return_val: f64, volatility: f64) -> PCRComponents {
        self.price_window.push(price);
        
        // Update return cache
        if self.return_cache.len() >= self.period {
            self.return_cache.remove(0);
        }
        self.return_cache.push(return_val);
        
        // Calculate P
        let (min, max) = self.price_window.min_max();
        let potential = if max - min > 1e-9 {
            (price - min) / (max - min)
        } else {
            0.5
        };
        
        // Calculate C
        let connectedness = if self.return_cache.len() > self.autocorr_lag {
            calculate_connectedness(&self.return_cache, self.autocorr_lag)
        } else {
            0.5
        };
        
        // Calculate R
        let resilience = 1.0 - volatility.clamp(0.0, 1.0);
        
        PCRComponents::new(potential, connectedness, resilience)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_log_returns() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let returns = calculate_log_returns(&prices).unwrap();
        
        assert_eq!(returns.len(), prices.len());
        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
    }
    
    #[test]
    fn test_pcr_components_calculation() {
        let prices = vec![100.0; 50];
        let returns = vec![0.0; 50];
        let volatilities = vec![0.5; 50];
        
        let pcr = calculate_pcr_components(&prices, &returns, &volatilities, 10, 1).unwrap();
        
        assert_eq!(pcr.len(), prices.len());
        // With constant prices, potential should be around 0.5
        assert!((pcr[20].potential - 0.5).abs() < 0.1);
        // With constant returns, connectedness should be high
        assert!(pcr[20].connectedness > 0.9);
        // Resilience should be 0.5 (1 - 0.5)
        assert!((pcr[20].resilience - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_fast_pcr_calculator() {
        let mut calc = FastPCRCalculator::new(10, 1);
        
        for i in 0..20 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            let pcr = calc.update(price, 0.01, 0.2);
            
            assert!(pcr.potential >= 0.0 && pcr.potential <= 1.0);
            assert!(pcr.connectedness >= 0.0 && pcr.connectedness <= 1.0);
            assert!(pcr.resilience >= 0.0 && pcr.resilience <= 1.0);
        }
    }
}