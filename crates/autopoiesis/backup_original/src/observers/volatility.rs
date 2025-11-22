//! Volatility observer implementation

use crate::prelude::*;

/// Observer for volatility analysis
#[derive(Debug, Clone)]
pub struct VolatilityObserver {
    pub window: usize,
    pub method: VolatilityMethod,
}

#[derive(Debug, Clone)]
pub enum VolatilityMethod {
    StandardDeviation,
    GARCH,
    EWMAnnualRolling,
}

impl VolatilityObserver {
    pub fn new(window: usize, method: VolatilityMethod) -> Self {
        Self { window, method }
    }
    
    pub fn observe(&self, data: &[f64]) -> f64 {
        if data.len() < self.window {
            return 0.0;
        }
        
        match self.method {
            VolatilityMethod::StandardDeviation => self.calculate_std_dev(data),
            VolatilityMethod::GARCH => self.calculate_garch(data),
            VolatilityMethod::EWMAnnualRolling => self.calculate_ewma(data),
        }
    }
    
    fn calculate_std_dev(&self, data: &[f64]) -> f64 {
        let returns: Vec<f64> = data.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
            
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
            
        variance.sqrt()
    }
    
    fn calculate_garch(&self, data: &[f64]) -> f64 {
        // Simplified GARCH(1,1) model
        self.calculate_std_dev(data) * 1.2 // Placeholder implementation
    }
    
    fn calculate_ewma(&self, data: &[f64]) -> f64 {
        // Exponentially weighted moving average
        let alpha = 0.06; // RiskMetrics parameter
        let returns: Vec<f64> = data.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
            
        if returns.is_empty() {
            return 0.0;
        }
        
        let mut ewma_var = returns[0].powi(2);
        for ret in &returns[1..] {
            ewma_var = alpha * ret.powi(2) + (1.0 - alpha) * ewma_var;
        }
        
        ewma_var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volatility_observer() {
        let observer = VolatilityObserver::new(10, VolatilityMethod::StandardDeviation);
        let data: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volatility = observer.observe(&data);
        assert!(volatility > 0.0);
    }
}