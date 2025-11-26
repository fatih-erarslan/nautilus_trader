//! Historical VaR calculation using actual historical returns

use async_trait::async_trait;
use crate::{Result, RiskError};
use crate::types::{Portfolio, VaRResult};
use crate::var::{VaRCalculator, VaRConfigInfo};
use chrono::Utc;
use tracing::{debug, info};

/// Historical VaR calculator
pub struct HistoricalVaR {
    lookback_days: usize,
    confidence_level: f64,
}

impl HistoricalVaR {
    /// Create a new Historical VaR calculator
    pub fn new(lookback_days: usize, confidence_level: f64) -> Self {
        Self {
            lookback_days,
            confidence_level,
        }
    }

    /// Calculate VaR and CVaR from historical returns
    pub async fn calculate(&self, historical_returns: &[f64]) -> Result<VaRResult> {
        if historical_returns.is_empty() {
            return Err(RiskError::insufficient_data(
                "No historical returns provided",
                1,
                0,
            ));
        }

        if historical_returns.len() < self.lookback_days {
            return Err(RiskError::insufficient_data(
                "Insufficient historical data",
                self.lookback_days,
                historical_returns.len(),
            ));
        }

        info!(
            "Calculating Historical VaR with {} days of data",
            historical_returns.len()
        );

        // Sort returns (ascending order - worst losses first)
        let mut sorted_returns = historical_returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_returns.len();

        // Calculate VaR at different confidence levels
        let var_95_idx = ((1.0 - 0.95) * n as f64) as usize;
        let var_99_idx = ((1.0 - 0.99) * n as f64) as usize;

        let var_95 = -sorted_returns[var_95_idx];
        let var_99 = -sorted_returns[var_99_idx];

        // Calculate CVaR (average of returns below VaR threshold)
        let cvar_95 = if var_95_idx > 0 {
            -sorted_returns[0..=var_95_idx].iter().sum::<f64>() / (var_95_idx + 1) as f64
        } else {
            var_95
        };

        let cvar_99 = if var_99_idx > 0 {
            -sorted_returns[0..=var_99_idx].iter().sum::<f64>() / (var_99_idx + 1) as f64
        } else {
            var_99
        };

        // Calculate statistics
        let expected_return = sorted_returns.iter().sum::<f64>() / n as f64;
        let variance = sorted_returns
            .iter()
            .map(|r| (r - expected_return).powi(2))
            .sum::<f64>()
            / (n - 1) as f64; // Sample variance

        let volatility = variance.sqrt();
        let worst_case = -sorted_returns[0];
        let best_case = -sorted_returns[n - 1];

        debug!(
            "Historical VaR: VaR(95%)={:.4}, CVaR(95%)={:.4}, volatility={:.4}",
            var_95, cvar_95, volatility
        );

        Ok(VaRResult {
            var_95,
            var_99,
            cvar_95,
            cvar_99,
            expected_return,
            volatility,
            worst_case,
            best_case,
            num_simulations: n,
            time_horizon_days: 1, // Historical VaR is typically 1-day
            calculated_at: Utc::now(),
        })
    }

    /// Calculate VaR from portfolio historical returns
    pub async fn calculate_from_portfolio(
        &self,
        portfolio_returns: &[f64],
    ) -> Result<VaRResult> {
        self.calculate(portfolio_returns).await
    }
}

#[async_trait]
impl VaRCalculator for HistoricalVaR {
    async fn calculate_portfolio(&self, _portfolio: &Portfolio) -> Result<VaRResult> {
        // In production, fetch historical returns for each position
        // and calculate portfolio returns
        // For now, use a simplified placeholder
        let placeholder_returns: Vec<f64> = (0..self.lookback_days)
            .map(|_| rand::random::<f64>() * 0.02 - 0.01)
            .collect();

        self.calculate(&placeholder_returns).await
    }

    fn method_name(&self) -> &str {
        "Historical VaR"
    }

    fn config(&self) -> VaRConfigInfo {
        VaRConfigInfo {
            confidence_level: self.confidence_level,
            time_horizon_days: 1,
            method: "Historical".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_historical_var() {
        // Generate sample returns (100 days)
        let returns: Vec<f64> = (0..100)
            .map(|i| (i as f64 / 100.0) * 0.1 - 0.05)
            .collect();

        let calculator = HistoricalVaR::new(100, 0.95);
        let result = calculator.calculate(&returns).await;

        assert!(result.is_ok());
        let var_result = result.unwrap();
        assert!(var_result.var_95 >= 0.0);
        assert!(var_result.cvar_95 >= var_result.var_95);
    }

    #[tokio::test]
    async fn test_insufficient_data() {
        let returns = vec![0.01, 0.02]; // Only 2 data points
        let calculator = HistoricalVaR::new(100, 0.95);

        let result = calculator.calculate(&returns).await;
        assert!(result.is_err());
    }
}
