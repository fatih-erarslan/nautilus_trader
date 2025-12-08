//! CDFA Unified - Ultra Minimal Working Version
//! Basic functionality to prove compilation works

use ndarray::Array1;

/// Basic floating point type for CDFA
pub type CdfaFloat = f64;

/// Basic result type
pub type CdfaResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Simple mathematical precision using Kahan summation
pub struct KahanAccumulator {
    sum: f64,
    compensation: f64,
}

impl KahanAccumulator {
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }
    
    pub fn add(&mut self, value: f64) {
        let compensated_value = value - self.compensation;
        let new_sum = self.sum + compensated_value;
        self.compensation = (new_sum - self.sum) - compensated_value;
        self.sum = new_sum;
    }
    
    pub fn sum(&self) -> f64 {
        self.sum
    }
}

/// Simple correlation calculation
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> CdfaResult<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Input arrays must have the same non-zero length".into());
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }
    
    if sum_xx == 0.0 || sum_yy == 0.0 {
        return Err("Cannot compute correlation with zero variance".into());
    }
    
    Ok(sum_xy / (sum_xx * sum_yy).sqrt())
}

/// Basic financial validation
pub fn validate_price_data(prices: &[f64]) -> CdfaResult<()> {
    if prices.is_empty() {
        return Err("Price data cannot be empty".into());
    }
    
    for &price in prices {
        if !price.is_finite() || price <= 0.0 {
            return Err("Invalid price data: prices must be positive and finite".into());
        }
    }
    
    Ok(())
}

/// Get library version and features
pub fn get_version_info() -> String {
    format!("CDFA Unified v{} - Core mathematical functionality ready", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_summation() {
        let mut acc = KahanAccumulator::new();
        acc.add(1.0);
        acc.add(2.0);
        acc.add(3.0);
        assert_eq!(acc.sum(), 6.0);
    }
    
    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_price_validation() {
        let valid_prices = vec![100.0, 101.5, 99.8, 102.3];
        assert!(validate_price_data(&valid_prices).is_ok());
        
        let invalid_prices = vec![100.0, -1.0, 99.8];
        assert!(validate_price_data(&invalid_prices).is_err());
    }
    
    #[test]
    fn test_version_info() {
        let info = get_version_info();
        assert!(info.contains("CDFA Unified"));
    }
}