use nalgebra::DVector;
use crate::error::{Result, RiskError};

/// Portfolio entropy calculator using Shannon entropy
/// and thermodynamic analogies
pub struct PortfolioEntropy {
    /// Market volatility analog (temperature in thermodynamic sense)
    temperature: f64,
}

impl PortfolioEntropy {
    pub fn new(temperature: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(RiskError::InvalidTemperature(
                "Temperature must be positive".to_string()
            ));
        }

        Ok(Self { temperature })
    }

    /// Calculate Shannon entropy of portfolio weights
    /// S = -Σ w_i ln(w_i) where w_i are normalized weights
    ///
    /// Higher entropy indicates more diversification
    pub fn calculate_entropy(&self, weights: &DVector<f64>) -> Result<f64> {
        if weights.is_empty() {
            return Err(RiskError::InsufficientData(
                "Cannot calculate entropy of empty portfolio".to_string()
            ));
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(RiskError::InvalidWeights(
                "Portfolio weights must be positive".to_string()
            ));
        }

        let normalized: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        let mut entropy = 0.0;
        for &w in &normalized {
            if w > 1e-10 {  // Avoid log(0)
                entropy -= w * w.ln();
            }
        }

        Ok(entropy)
    }

    /// Calculate thermodynamic free energy F = U - TS
    /// where U is the internal energy (expected portfolio value)
    /// and S is the entropy
    ///
    /// Minimum free energy indicates optimal risk-return balance
    pub fn free_energy(&self, energy: f64, entropy: f64) -> f64 {
        energy - self.temperature * entropy
    }

    /// Calculate maximum entropy for n assets (uniform distribution)
    /// S_max = ln(n)
    pub fn max_entropy(&self, n_assets: usize) -> f64 {
        if n_assets == 0 {
            return 0.0;
        }
        (n_assets as f64).ln()
    }

    /// Calculate relative entropy (diversification ratio)
    /// η = S / S_max
    ///
    /// Returns value between 0 (concentrated) and 1 (fully diversified)
    pub fn diversification_ratio(&self, weights: &DVector<f64>) -> Result<f64> {
        let entropy = self.calculate_entropy(weights)?;
        let max_entropy = self.max_entropy(weights.len());

        if max_entropy == 0.0 {
            Ok(0.0)
        } else {
            Ok(entropy / max_entropy)
        }
    }

    /// Get temperature parameter
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_entropy_uniform_distribution() {
        let entropy_calc = PortfolioEntropy::new(1.0).unwrap();

        // Uniform distribution of 4 assets should have entropy = ln(4)
        let weights = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let entropy = entropy_calc.calculate_entropy(&weights).unwrap();

        assert_relative_eq!(entropy, 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_concentrated_portfolio() {
        let entropy_calc = PortfolioEntropy::new(1.0).unwrap();

        // Highly concentrated portfolio should have low entropy
        let weights = DVector::from_vec(vec![0.95, 0.03, 0.01, 0.01]);
        let entropy = entropy_calc.calculate_entropy(&weights).unwrap();

        // Should be much less than max entropy ln(4) ≈ 1.386
        assert!(entropy < 0.5);
    }

    #[test]
    fn test_diversification_ratio() {
        let entropy_calc = PortfolioEntropy::new(1.0).unwrap();

        // Uniform distribution should have ratio = 1.0
        let uniform_weights = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let ratio = entropy_calc.diversification_ratio(&uniform_weights).unwrap();
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);

        // Concentrated portfolio should have lower ratio
        let concentrated_weights = DVector::from_vec(vec![0.95, 0.03, 0.01, 0.01]);
        let ratio = entropy_calc.diversification_ratio(&concentrated_weights).unwrap();
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_free_energy() {
        let entropy_calc = PortfolioEntropy::new(0.5).unwrap();

        let energy = 100.0;
        let entropy = 2.0;
        let free_energy = entropy_calc.free_energy(energy, entropy);

        // F = U - TS = 100 - 0.5*2 = 99
        assert_relative_eq!(free_energy, 99.0, epsilon = 1e-10);
    }
}
