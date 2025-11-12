use crate::error::{Result, RiskError};

/// Value-at-Risk (VaR) calculator with thermodynamic entropy constraints
///
/// Uses maximum entropy principle to estimate tail risk
/// under incomplete information
pub struct ThermodynamicVaR {
    /// Confidence level (e.g., 0.95 for 95% VaR)
    confidence_level: f64,
}

impl ThermodynamicVaR {
    pub fn new(confidence_level: f64) -> Result<Self> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(RiskError::CalculationError(
                "Confidence level must be between 0 and 1".to_string()
            ));
        }

        Ok(Self { confidence_level })
    }

    /// Calculate historical VaR from return data
    ///
    /// Returns the (1 - confidence_level) quantile of losses
    pub fn calculate_historical(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Err(RiskError::InsufficientData(
                "Cannot calculate VaR with empty return data".to_string()
            ));
        }

        // Convert to losses (negative returns)
        let mut losses: Vec<f64> = returns.iter().map(|&r| -r).collect();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find quantile index
        let alpha = 1.0 - self.confidence_level;
        let index = ((losses.len() as f64) * alpha).ceil() as usize;
        let index = index.min(losses.len() - 1);

        Ok(losses[index])
    }

    /// Calculate parametric VaR assuming normal distribution
    ///
    /// VaR = μ + σ * z_α
    /// where z_α is the α-quantile of standard normal
    pub fn calculate_parametric(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Err(RiskError::InsufficientData(
                "Cannot calculate VaR with empty return data".to_string()
            ));
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        let variance: f64 = returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / n;

        let std_dev = variance.sqrt();

        // Approximate z-score for common confidence levels
        let z_alpha = match self.confidence_level {
            x if (x - 0.95).abs() < 1e-6 => 1.645,
            x if (x - 0.99).abs() < 1e-6 => 2.326,
            _ => {
                // For other levels, use rough approximation
                // In production, would use proper inverse normal CDF
                let alpha = 1.0 - self.confidence_level;
                -(-2.0 * alpha.ln()).sqrt()
            }
        };

        // VaR is negative of (mean + z*sigma) since we want loss magnitude
        Ok(-(mean + z_alpha * std_dev))
    }

    /// Calculate VaR constrained by maximum entropy principle
    ///
    /// Under incomplete information, use distribution that maximizes entropy
    /// subject to known constraints (mean, variance, entropy bound)
    ///
    /// TODO: Implement full maximum entropy optimization
    pub fn calculate_entropy_constrained(
        &self,
        returns: &[f64],
        entropy_constraint: f64,
    ) -> Result<f64> {
        if entropy_constraint < 0.0 {
            return Err(RiskError::EntropyConstraintViolation(
                "Entropy constraint must be non-negative".to_string()
            ));
        }

        // Placeholder: For now, use parametric VaR with entropy adjustment
        // Full implementation would solve constrained optimization problem
        let var_parametric = self.calculate_parametric(returns)?;

        // Entropy penalty: higher entropy → higher uncertainty → higher VaR
        let entropy_adjustment = 1.0 + 0.1 * entropy_constraint;

        Ok(var_parametric * entropy_adjustment)
    }

    /// Get confidence level
    pub fn confidence_level(&self) -> f64 {
        self.confidence_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_historical_var() {
        let var_calc = ThermodynamicVaR::new(0.95).unwrap();

        // Create return data with known 5th percentile
        let mut returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();

        let var = var_calc.calculate_historical(&returns).unwrap();

        // 95% VaR should be around the 5th percentile (approximately -4.5)
        assert!(var > 4.0 && var < 5.0);
    }

    #[test]
    fn test_parametric_var_normal() {
        let var_calc = ThermodynamicVaR::new(0.95).unwrap();

        // Simulate normal returns with mean=0, std=1
        let returns: Vec<f64> = vec![
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
            -1.2, -0.8, -0.3, 0.2, 0.7, 1.2,
        ];

        let var = var_calc.calculate_parametric(&returns).unwrap();

        // For normal(0,1), 95% VaR should be around 1.645
        assert!(var > 1.0 && var < 2.5);
    }

    #[test]
    fn test_entropy_constraint_increases_var() {
        let var_calc = ThermodynamicVaR::new(0.95).unwrap();

        let returns: Vec<f64> = vec![0.1, -0.2, 0.15, -0.1, 0.05];

        let var_base = var_calc.calculate_entropy_constrained(&returns, 0.0).unwrap();
        let var_constrained = var_calc.calculate_entropy_constrained(&returns, 2.0).unwrap();

        // Higher entropy constraint should increase VaR
        assert!(var_constrained > var_base);
    }
}
