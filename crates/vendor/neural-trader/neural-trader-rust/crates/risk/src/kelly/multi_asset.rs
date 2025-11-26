//! Multi-asset Kelly optimization
//!
//! Extends Kelly Criterion to portfolios with multiple correlated assets using:
//! - Mean-variance optimization
//! - Correlation-adjusted position sizing
//! - Portfolio-level constraints

use crate::{Result, RiskError};
use crate::types::{Portfolio, Symbol};
use nalgebra::{DMatrix, DVector};
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use tracing::{debug, info};

/// Multi-asset Kelly optimizer
#[derive(Debug, Clone)]
pub struct KellyMultiAsset {
    /// Expected returns for each asset
    expected_returns: HashMap<Symbol, f64>,
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Asset symbols (ordered)
    symbols: Vec<Symbol>,
    /// Fractional Kelly multiplier
    fractional: f64,
    /// Maximum leverage constraint
    max_leverage: f64,
    /// Maximum concentration per asset
    max_concentration: f64,
}

impl KellyMultiAsset {
    /// Create new multi-asset Kelly optimizer
    ///
    /// # Arguments
    /// * `expected_returns` - Expected return for each asset
    /// * `covariance` - Covariance matrix of returns
    /// * `symbols` - Ordered list of symbols (must match covariance rows/cols)
    /// * `fractional` - Fractional Kelly (typically 0.25-0.5)
    pub fn new(
        expected_returns: HashMap<Symbol, f64>,
        covariance: DMatrix<f64>,
        symbols: Vec<Symbol>,
        fractional: f64,
    ) -> Result<Self> {
        if symbols.len() != covariance.nrows() || symbols.len() != covariance.ncols() {
            return Err(RiskError::KellyCriterionError(format!(
                "Dimension mismatch: {} symbols but covariance is {}x{}",
                symbols.len(),
                covariance.nrows(),
                covariance.ncols()
            )));
        }

        if !(0.0..=1.0).contains(&fractional) {
            return Err(RiskError::KellyCriterionError(format!(
                "Fractional must be between 0 and 1, got {}",
                fractional
            )));
        }

        for symbol in &symbols {
            if !expected_returns.contains_key(symbol) {
                return Err(RiskError::KellyCriterionError(format!(
                    "Missing expected return for symbol: {}",
                    symbol
                )));
            }
        }

        Ok(Self {
            expected_returns,
            covariance,
            symbols,
            fractional,
            max_leverage: 1.0,
            max_concentration: 0.5,
        })
    }

    /// Set maximum leverage constraint
    pub fn with_max_leverage(mut self, max_leverage: f64) -> Self {
        self.max_leverage = max_leverage;
        self
    }

    /// Set maximum concentration per asset
    pub fn with_max_concentration(mut self, max_concentration: f64) -> Self {
        self.max_concentration = max_concentration;
        self
    }

    /// Calculate optimal portfolio weights using mean-variance optimization
    ///
    /// Solves: max w'μ - (λ/2)w'Σw
    /// subject to: ||w||₁ ≤ max_leverage, |wᵢ| ≤ max_concentration
    pub fn calculate_weights(&self) -> Result<HashMap<Symbol, f64>> {
        let n = self.symbols.len();

        // Build expected return vector
        let mut mu = DVector::zeros(n);
        for (i, symbol) in self.symbols.iter().enumerate() {
            mu[i] = *self.expected_returns.get(symbol).unwrap();
        }

        // Risk aversion parameter (from Kelly criterion theory)
        let lambda = 2.0 / self.fractional;

        // Solve: w = (1/λ) * Σ⁻¹ * μ
        let sigma_inv = self.covariance.clone().try_inverse().ok_or_else(|| {
            RiskError::MatrixError("Covariance matrix is singular".to_string())
        })?;

        let w = sigma_inv * mu / lambda;

        // Apply fractional Kelly
        let w_fractional = w * self.fractional;

        // Convert to weights and apply constraints
        let mut weights = HashMap::new();
        let mut total_weight = 0.0;

        for (i, symbol) in self.symbols.iter().enumerate() {
            let mut weight = w_fractional[i];

            // Apply concentration constraint
            weight = weight.clamp(-self.max_concentration, self.max_concentration);

            weights.insert(symbol.clone(), weight);
            total_weight += weight.abs();
        }

        // Apply leverage constraint
        if total_weight > self.max_leverage {
            let scale = self.max_leverage / total_weight;
            for weight in weights.values_mut() {
                *weight *= scale;
            }
        }

        debug!(
            "Multi-asset Kelly weights calculated: total_leverage={:.3}",
            total_weight.min(self.max_leverage)
        );

        Ok(weights)
    }

    /// Calculate position sizes given total capital
    pub fn calculate_positions(&self, total_capital: f64) -> Result<HashMap<Symbol, f64>> {
        if total_capital <= 0.0 {
            return Err(RiskError::KellyCriterionError(format!(
                "Total capital must be positive, got {}",
                total_capital
            )));
        }

        let weights = self.calculate_weights()?;
        let mut positions = HashMap::new();

        for (symbol, weight) in weights {
            let position_size = total_capital * weight;
            positions.insert(symbol.clone(), position_size);
        }

        Ok(positions)
    }

    /// Calculate rebalancing needed for current portfolio
    pub fn calculate_rebalance(
        &self,
        current_portfolio: &Portfolio,
    ) -> Result<HashMap<Symbol, f64>> {
        let total_value = current_portfolio.total_value();
        let target_positions = self.calculate_positions(total_value)?;
        let mut rebalance_orders = HashMap::new();

        // Calculate what we have vs what we want
        for (symbol, target_size) in &target_positions {
            let current_size = current_portfolio
                .get_position(symbol)
                .map(|p| p.market_value.to_f64().unwrap_or(0.0))
                .unwrap_or(0.0);

            let delta = target_size - current_size;
            if delta.abs() > 0.01 * total_value {
                // Only rebalance if > 1% of portfolio
                rebalance_orders.insert(symbol.clone(), delta);
            }
        }

        // Check for positions to close (not in target)
        for (symbol, position) in &current_portfolio.positions {
            if !target_positions.contains_key(symbol) {
                let current_size = position.market_value.to_f64().unwrap_or(0.0);
                if current_size.abs() > 0.0 {
                    rebalance_orders.insert(symbol.clone(), -current_size);
                }
            }
        }

        info!(
            "Rebalancing: {} positions to adjust",
            rebalance_orders.len()
        );

        Ok(rebalance_orders)
    }

    /// Calculate expected portfolio return
    pub fn expected_return(&self, weights: &HashMap<Symbol, f64>) -> f64 {
        weights
            .iter()
            .map(|(symbol, weight)| weight * self.expected_returns.get(symbol).unwrap_or(&0.0))
            .sum()
    }

    /// Calculate portfolio variance
    pub fn portfolio_variance(&self, weights: &HashMap<Symbol, f64>) -> Result<f64> {
        let n = self.symbols.len();
        let mut w = DVector::zeros(n);

        for (i, symbol) in self.symbols.iter().enumerate() {
            w[i] = *weights.get(symbol).unwrap_or(&0.0);
        }

        let variance = (&w).transpose() * &self.covariance * &w;
        Ok(variance[0])
    }

    /// Calculate Sharpe ratio for given weights
    pub fn sharpe_ratio(
        &self,
        weights: &HashMap<Symbol, f64>,
        risk_free_rate: f64,
    ) -> Result<f64> {
        let expected_return = self.expected_return(weights);
        let variance = self.portfolio_variance(weights)?;
        let volatility = variance.sqrt();

        if volatility == 0.0 {
            return Ok(0.0);
        }

        Ok((expected_return - risk_free_rate) / volatility)
    }
}

/// Builder for multi-asset Kelly optimizer
pub struct KellyMultiAssetBuilder {
    expected_returns: HashMap<Symbol, f64>,
    returns_history: HashMap<Symbol, Vec<f64>>,
    fractional: f64,
    max_leverage: f64,
    max_concentration: f64,
}

impl KellyMultiAssetBuilder {
    pub fn new() -> Self {
        Self {
            expected_returns: HashMap::new(),
            returns_history: HashMap::new(),
            fractional: 0.25,
            max_leverage: 1.0,
            max_concentration: 0.5,
        }
    }

    pub fn add_asset(
        mut self,
        symbol: Symbol,
        expected_return: f64,
        returns: Vec<f64>,
    ) -> Self {
        self.expected_returns.insert(symbol.clone(), expected_return);
        self.returns_history.insert(symbol, returns);
        self
    }

    pub fn fractional(mut self, fractional: f64) -> Self {
        self.fractional = fractional;
        self
    }

    pub fn max_leverage(mut self, max_leverage: f64) -> Self {
        self.max_leverage = max_leverage;
        self
    }

    pub fn max_concentration(mut self, max_concentration: f64) -> Self {
        self.max_concentration = max_concentration;
        self
    }

    pub fn build(self) -> Result<KellyMultiAsset> {
        if self.returns_history.is_empty() {
            return Err(RiskError::KellyCriterionError(
                "No assets provided".to_string(),
            ));
        }

        // Calculate covariance matrix
        let symbols: Vec<Symbol> = self.returns_history.keys().cloned().collect();
        let _n = symbols.len();

        let covariance = Self::calculate_covariance(&symbols, &self.returns_history)?;

        let mut optimizer =
            KellyMultiAsset::new(self.expected_returns, covariance, symbols, self.fractional)?;
        optimizer = optimizer
            .with_max_leverage(self.max_leverage)
            .with_max_concentration(self.max_concentration);

        Ok(optimizer)
    }

    fn calculate_covariance(
        symbols: &[Symbol],
        returns_history: &HashMap<Symbol, Vec<f64>>,
    ) -> Result<DMatrix<f64>> {
        let n = symbols.len();
        let mut cov = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let returns_i = returns_history.get(&symbols[i]).unwrap();
                let returns_j = returns_history.get(&symbols[j]).unwrap();

                if returns_i.len() != returns_j.len() {
                    return Err(RiskError::KellyCriterionError(
                        "Inconsistent return history lengths".to_string(),
                    ));
                }

                let mean_i = returns_i.iter().sum::<f64>() / returns_i.len() as f64;
                let mean_j = returns_j.iter().sum::<f64>() / returns_j.len() as f64;

                let covariance: f64 = returns_i
                    .iter()
                    .zip(returns_j.iter())
                    .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                    .sum::<f64>()
                    / (returns_i.len() - 1) as f64;

                cov[(i, j)] = covariance;
            }
        }

        Ok(cov)
    }
}

impl Default for KellyMultiAssetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kelly_multi_asset_basic() {
        let symbols = vec![Symbol::new("AAPL"), Symbol::new("MSFT")];

        let mut expected_returns = HashMap::new();
        expected_returns.insert(Symbol::new("AAPL"), 0.10);
        expected_returns.insert(Symbol::new("MSFT"), 0.08);

        let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.02, 0.02, 0.03]);

        let kelly = KellyMultiAsset::new(expected_returns, covariance, symbols, 0.25).unwrap();
        let weights = kelly.calculate_weights().unwrap();

        assert_eq!(weights.len(), 2);
        let total_weight: f64 = weights.values().map(|w| w.abs()).sum();
        assert!(total_weight <= 1.0);
    }

    #[test]
    fn test_kelly_builder() {
        let aapl_returns = vec![0.01, -0.02, 0.03, 0.01, -0.01];
        let msft_returns = vec![0.02, -0.01, 0.02, 0.01, 0.00];

        let kelly = KellyMultiAssetBuilder::new()
            .add_asset(Symbol::new("AAPL"), 0.10, aapl_returns)
            .add_asset(Symbol::new("MSFT"), 0.08, msft_returns)
            .fractional(0.25)
            .max_leverage(1.0)
            .build()
            .unwrap();

        let weights = kelly.calculate_weights().unwrap();
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_leverage_constraint() {
        let symbols = vec![Symbol::new("AAPL"), Symbol::new("MSFT")];
        let mut expected_returns = HashMap::new();
        expected_returns.insert(Symbol::new("AAPL"), 0.20);
        expected_returns.insert(Symbol::new("MSFT"), 0.15);

        let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.03]);

        let kelly = KellyMultiAsset::new(expected_returns, covariance, symbols, 0.5)
            .unwrap()
            .with_max_leverage(0.8);

        let weights = kelly.calculate_weights().unwrap();
        let total_leverage: f64 = weights.values().map(|w| w.abs()).sum();
        assert!(total_leverage <= 0.81); // Allow small numerical error
    }

    #[test]
    fn test_concentration_constraint() {
        let symbols = vec![Symbol::new("AAPL")];
        let mut expected_returns = HashMap::new();
        expected_returns.insert(Symbol::new("AAPL"), 0.20);

        let covariance = DMatrix::from_row_slice(1, 1, &[0.04]);

        let kelly = KellyMultiAsset::new(expected_returns, covariance, symbols, 0.5)
            .unwrap()
            .with_max_concentration(0.3);

        let weights = kelly.calculate_weights().unwrap();
        let aapl_weight = weights.get(&Symbol::new("AAPL")).unwrap();
        assert!(aapl_weight.abs() <= 0.31); // Allow small numerical error
    }

    #[test]
    fn test_expected_return_calculation() {
        let symbols = vec![Symbol::new("AAPL"), Symbol::new("MSFT")];
        let mut expected_returns = HashMap::new();
        expected_returns.insert(Symbol::new("AAPL"), 0.10);
        expected_returns.insert(Symbol::new("MSFT"), 0.08);

        let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.02, 0.02, 0.03]);

        let kelly = KellyMultiAsset::new(expected_returns, covariance, symbols, 0.25).unwrap();

        let mut weights = HashMap::new();
        weights.insert(Symbol::new("AAPL"), 0.6);
        weights.insert(Symbol::new("MSFT"), 0.4);

        let expected = kelly.expected_return(&weights);
        assert_relative_eq!(expected, 0.092, epsilon = 0.001); // 0.6*0.10 + 0.4*0.08
    }
}
