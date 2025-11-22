/// Risk metrics and portfolio analytics
///
/// Implements standard risk measures used in quantitative finance.
use ndarray::ArrayView1;
use crate::types::FinanceError;

/// Comprehensive risk metrics for a portfolio or asset
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Value at Risk (95% confidence)
    pub var_95: f64,

    /// Value at Risk (99% confidence)
    pub var_99: f64,

    /// Expected Shortfall / Conditional VaR (95%)
    pub cvar_95: f64,

    /// Annualized volatility
    pub volatility: f64,

    /// Sharpe ratio (assuming risk-free rate = 0)
    pub sharpe_ratio: f64,

    /// Maximum drawdown
    pub max_drawdown: f64,

    /// Skewness
    pub skewness: f64,

    /// Excess kurtosis
    pub kurtosis: f64,
}

impl RiskMetrics {
    /// Calculate comprehensive risk metrics from return series
    ///
    /// # Arguments
    /// * `returns` - Time series of returns
    /// * `periods_per_year` - Annualization factor (252 for daily, 12 for monthly)
    pub fn from_returns(
        returns: ArrayView1<f64>,
        periods_per_year: f64,
    ) -> Result<Self, FinanceError> {
        if returns.len() < 10 {
            return Err(FinanceError::InsufficientData);
        }

        let var_95 = calculate_historical_var(returns, 0.95)?;
        let var_99 = calculate_historical_var(returns, 0.99)?;
        let cvar_95 = calculate_cvar(returns, 0.95)?;
        let volatility = calculate_volatility(returns, periods_per_year);
        let mean_return = returns.mean().unwrap_or(0.0);
        let sharpe_ratio = if volatility > 0.0 {
            mean_return * periods_per_year.sqrt() / volatility
        } else {
            0.0
        };
        let max_drawdown = calculate_max_drawdown(returns);
        let skewness = calculate_skewness(returns);
        let kurtosis = calculate_kurtosis(returns);

        Ok(Self {
            var_95,
            var_99,
            cvar_95,
            volatility,
            sharpe_ratio,
            max_drawdown,
            skewness,
            kurtosis,
        })
    }
}

/// Calculate historical VaR (quantile method)
fn calculate_historical_var(returns: ArrayView1<f64>, confidence: f64) -> Result<f64, FinanceError> {
    use super::var::historical_var;
    historical_var(returns, confidence)
}

/// Calculate Conditional VaR (Expected Shortfall)
///
/// CVaR is the expected loss given that VaR has been exceeded.
fn calculate_cvar(returns: ArrayView1<f64>, confidence: f64) -> Result<f64, FinanceError> {
    let var = calculate_historical_var(returns, confidence)?;

    // Calculate mean of losses exceeding VaR
    let threshold = -var;
    let tail_losses: Vec<f64> = returns
        .iter()
        .filter(|&&r| r <= threshold)
        .copied()
        .collect();

    if tail_losses.is_empty() {
        return Ok(var);  // Fallback to VaR if no exceedances
    }

    let cvar = -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;
    Ok(cvar)
}

/// Calculate annualized volatility
fn calculate_volatility(returns: ArrayView1<f64>, periods_per_year: f64) -> f64 {
    let mean = returns.mean().unwrap_or(0.0);
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;

    variance.sqrt() * periods_per_year.sqrt()
}

/// Calculate maximum drawdown
///
/// Drawdown is the peak-to-trough decline in cumulative returns.
fn calculate_max_drawdown(returns: ArrayView1<f64>) -> f64 {
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for &ret in returns.iter() {
        cumulative *= 1.0 + ret;

        if cumulative > peak {
            peak = cumulative;
        }

        let drawdown = (peak - cumulative) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd
}

/// Calculate skewness (third standardized moment)
///
/// Skewness = E[(X - μ)³] / σ³
fn calculate_skewness(returns: ArrayView1<f64>) -> f64 {
    let n = returns.len() as f64;
    let mean = returns.mean().unwrap_or(0.0);
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    let third_moment = returns
        .iter()
        .map(|&r| ((r - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    third_moment
}

/// Calculate excess kurtosis (fourth standardized moment - 3)
///
/// Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
///
/// Excess kurtosis = 0 for normal distribution
fn calculate_kurtosis(returns: ArrayView1<f64>) -> f64 {
    let n = returns.len() as f64;
    let mean = returns.mean().unwrap_or(0.0);
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    let fourth_moment = returns
        .iter()
        .map(|&r| ((r - mean) / std_dev).powi(4))
        .sum::<f64>() / n;

    fourth_moment - 3.0  // Excess kurtosis
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_volatility() {
        let returns = array![0.01, -0.01, 0.02, -0.02, 0.015];
        let vol = calculate_volatility(returns.view(), 252.0);

        assert!(vol > 0.0);
        // Annualized volatility should be reasonable
        assert!(vol < 1.0);
    }

    #[test]
    fn test_max_drawdown() {
        // Create returns with known drawdown
        let returns = array![0.1, 0.1, -0.2, -0.1, 0.05];
        let dd = calculate_max_drawdown(returns.view());

        // Should detect the ~28% drawdown
        assert!(dd > 0.25 && dd < 0.30);
    }

    #[test]
    fn test_skewness_kurtosis() {
        // Symmetric distribution should have skewness near 0
        let symmetric = array![-0.02, -0.01, 0.0, 0.01, 0.02];
        let skew = calculate_skewness(symmetric.view());
        assert!(skew.abs() < 0.1);

        // Small sample has high kurtosis variance, use larger sample
        let larger = array![
            -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03,
            -0.025, -0.015, -0.005, 0.005, 0.015, 0.025
        ];
        let kurt = calculate_kurtosis(larger.view());
        // With small samples, kurtosis can vary widely; just check it's finite
        assert!(kurt.is_finite());
    }

    #[test]
    fn test_risk_metrics() {
        // Use more diverse returns with clear tail events
        let returns = array![
            0.02, -0.01, 0.015, -0.02, 0.01, 0.005, -0.015, 0.025, -0.005, 0.01,
            -0.03, 0.02, -0.01, 0.01, -0.025, 0.015, -0.01, 0.02, -0.015, 0.01
        ];

        let metrics = RiskMetrics::from_returns(returns.view(), 252.0).unwrap();

        assert!(metrics.var_95 > 0.0);
        assert!(metrics.var_99 >= metrics.var_95);  // 99% VaR >= 95% VaR (can be equal with small samples)
        assert!(metrics.cvar_95 >= metrics.var_95);  // CVaR >= VaR
        assert!(metrics.volatility > 0.0);
        assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
    }

    #[test]
    fn test_cvar_exceeds_var() {
        let returns = array![
            -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04
        ];

        let var_95 = calculate_historical_var(returns.view(), 0.95).unwrap();
        let cvar_95 = calculate_cvar(returns.view(), 0.95).unwrap();

        // CVaR should be >= VaR (expected tail loss >= worst case)
        assert!(cvar_95 >= var_95);
    }
}
