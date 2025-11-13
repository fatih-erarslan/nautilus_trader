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
    /// Returns the confidence_level quantile of losses (negative returns).
    /// For 95% VaR, returns the loss exceeded by 5% of worst returns.
    ///
    /// Basel III: VaR is the loss level that will not be exceeded with probability α
    /// For 95% VaR: There's 5% chance the loss will exceed VaR
    pub fn calculate_historical(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Err(RiskError::InsufficientData(
                "Cannot calculate VaR with empty return data".to_string()
            ));
        }

        // Convert to losses (negative returns)
        let mut losses: Vec<f64> = returns.iter().map(|&r| -r).collect();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // For confidence_level α (e.g., 0.95), we want the α quantile of losses
        // This is the (100×α)th percentile of the loss distribution
        // After sorting losses in ascending order, we want the index at α position
        let n = losses.len();

        // Use linear interpolation for quantile calculation
        // For 95% confidence, we want 95th percentile = rank at 0.95 * (n-1)
        let rank = self.confidence_level * (n as f64 - 1.0);
        let lower_idx = rank.floor() as usize;
        let upper_idx = (rank.ceil() as usize).min(n - 1);
        let weight = rank - lower_idx as f64;

        let var = if lower_idx == upper_idx {
            losses[lower_idx]
        } else {
            losses[lower_idx] * (1.0 - weight) + losses[upper_idx] * weight
        };

        // VaR should be non-negative (magnitude of loss)
        Ok(var.max(0.0))
    }

    /// Calculate parametric VaR assuming normal distribution
    ///
    /// VaR_α = -(μ - z_α * σ)
    /// where z_α is the α-quantile of standard normal (negative for left tail)
    ///
    /// For returns R ~ N(μ, σ²), the loss L = -R ~ N(-μ, σ²)
    /// VaR_α = quantile_α(L) = -μ + z_α * σ, where z_α > 0 for confidence > 0.5
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

        // Z-score for confidence level (e.g., 1.645 for 95%, 2.326 for 99%)
        // These are positive values for the left tail of the return distribution
        // which corresponds to the right tail of the loss distribution
        let z_alpha = match self.confidence_level {
            x if (x - 0.95).abs() < 1e-6 => 1.645,
            x if (x - 0.99).abs() < 1e-6 => 2.326,
            x if (x - 0.999).abs() < 1e-6 => 3.090,
            _ => {
                // For other levels, use rough approximation
                // For α in (0.5, 1), we want positive z-score
                let alpha = 1.0 - self.confidence_level;
                // Rough inverse normal approximation
                if alpha < 0.5 {
                    (-2.0 * alpha.ln()).sqrt()
                } else {
                    0.0 // Should not happen for typical confidence levels
                }
            }
        };

        // VaR = -μ + z_α * σ (loss magnitude)
        // For losses, positive VaR means potential loss
        let var = -mean + z_alpha * std_dev;

        // Ensure VaR is non-negative (Basel III requirement)
        Ok(var.max(0.0))
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

        // 95% VaR returns negative value (actual loss), around -4.5
        assert!(var < -3.5 && var > -5.5, "VaR {} not in expected range", var);
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

        // For normal(0,1), 95% VaR returns negative value, around -1.645
        assert!(var < -0.5 && var > -3.0, "VaR {} not in expected range", var);
    }

    #[test]
    fn test_entropy_constraint_increases_var() {
        let var_calc = ThermodynamicVaR::new(0.95).unwrap();

        let returns: Vec<f64> = vec![0.1, -0.2, 0.15, -0.1, 0.05];

        let var_base = var_calc.calculate_entropy_constrained(&returns, 0.0).unwrap();
        let var_constrained = var_calc.calculate_entropy_constrained(&returns, 2.0).unwrap();

        // Higher entropy constraint should increase VaR magnitude (more negative)
        assert!(var_constrained < var_base,
            "Expected var_constrained ({}) < var_base ({})", var_constrained, var_base);
    }
}
