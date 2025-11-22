/// Value at Risk (VaR) Implementations
///
/// References:
/// - Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
///   Journal of Econometrics, 31(3), 307-327.
/// - RiskMetrics (1996). "Technical Document" (4th ed.), JP Morgan/Reuters.
/// - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial Risk"
///   (3rd ed.), McGraw-Hill, ISBN: 978-0071464956
///
/// Mathematical Formulas:
///
/// **Historical Simulation VaR:**
/// ```latex
/// VaR_α = -Quantile(returns, α)
/// ```
///
/// **GARCH(1,1) VaR:**
/// ```latex
/// σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
/// VaR_α = -μ - σ_t·Φ^{-1}(α)
/// ```
///
/// **EWMA VaR (RiskMetrics):**
/// ```latex
/// σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
/// VaR_α = -σ_t·Φ^{-1}(α)
/// ```
///
/// Where:
/// - α: Confidence level (e.g., 0.01 for 99% VaR)
/// - ω, α, β: GARCH parameters
/// - λ: Decay factor (typically 0.94 for daily data)
/// - Φ^{-1}: Inverse standard normal CDF
use ndarray::{Array1, ArrayView1};
use statrs::distribution::{Normal, ContinuousCDF};
use crate::types::FinanceError;

/// VaR model type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarModel {
    /// Historical Simulation: Empirical quantile method
    HistoricalSimulation,

    /// GARCH(1,1): Conditional volatility forecasting
    Garch11,

    /// EWMA: Exponentially weighted moving average (RiskMetrics)
    EWMA,
}

/// GARCH(1,1) parameters
///
/// Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
///
/// Constraints:
/// - ω > 0
/// - α, β ≥ 0
/// - α + β < 1 (stationarity condition)
#[derive(Debug, Clone, Copy)]
pub struct GarchParams {
    pub omega: f64,  // ω: Constant term
    pub alpha: f64,  // α: ARCH coefficient
    pub beta: f64,   // β: GARCH coefficient
}

impl GarchParams {
    /// Create GARCH parameters with validation
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Result<Self, FinanceError> {
        if omega <= 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("GARCH omega must be positive: {}", omega)
            ));
        }
        if alpha < 0.0 || beta < 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("GARCH alpha and beta must be non-negative: {}, {}", alpha, beta)
            ));
        }
        if alpha + beta >= 1.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("GARCH stationarity violated: α + β = {} >= 1", alpha + beta)
            ));
        }
        Ok(Self { omega, alpha, beta })
    }

    /// Standard GARCH(1,1) parameters from RiskMetrics
    /// Based on typical equity market estimates
    pub fn default_equity() -> Self {
        Self {
            omega: 0.000001,
            alpha: 0.08,
            beta: 0.90,
        }
    }

    /// Unconditional variance: σ² = ω / (1 - α - β)
    pub fn unconditional_variance(&self) -> f64 {
        self.omega / (1.0 - self.alpha - self.beta)
    }
}

/// Calculate Historical Simulation VaR
///
/// Returns the empirical quantile of the return distribution.
///
/// # Arguments
/// * `returns` - Historical return series
/// * `confidence` - Confidence level (e.g., 0.99 for 99% VaR)
///
/// # Example
/// ```rust
/// use ndarray::array;
/// use hyperphysics_finance::risk::var::*;
///
/// let returns = array![-0.02, -0.01, 0.01, 0.02, 0.03];
/// let var = historical_var(returns.view(), 0.95).unwrap();
/// ```
pub fn historical_var(returns: ArrayView1<f64>, confidence: f64) -> Result<f64, FinanceError> {
    if returns.len() < 2 {
        return Err(FinanceError::InsufficientData);
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(FinanceError::InvalidOptionParams(
            format!("Confidence must be in (0, 1): {}", confidence)
        ));
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate quantile index
    let alpha = 1.0 - confidence;
    let n = sorted_returns.len() as f64;
    let h = alpha * (n + 1.0);

    // Linear interpolation between order statistics
    let var = if h <= 1.0 {
        -sorted_returns[0]
    } else if h >= n {
        -sorted_returns[sorted_returns.len() - 1]
    } else {
        let k = h.floor() as usize;
        let f = h - h.floor();
        -(sorted_returns[k - 1] * (1.0 - f) + sorted_returns[k] * f)
    };

    Ok(var)
}

/// Calculate GARCH(1,1) conditional volatility forecast
///
/// Uses recursive formula: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
///
/// # Arguments
/// * `returns` - Historical return series
/// * `params` - GARCH(1,1) parameters
///
/// # Returns
/// Vector of conditional volatilities (σ_t)
pub fn garch_volatility(
    returns: ArrayView1<f64>,
    params: &GarchParams,
) -> Result<Array1<f64>, FinanceError> {
    if returns.len() < 2 {
        return Err(FinanceError::InsufficientData);
    }

    let n = returns.len();
    let mut variances = Array1::zeros(n);

    // Initialize with unconditional variance
    variances[0] = params.unconditional_variance();

    // Recursive GARCH update
    for t in 1..n {
        let epsilon_sq = returns[t - 1].powi(2);  // Squared residual
        variances[t] = params.omega + params.alpha * epsilon_sq + params.beta * variances[t - 1];
    }

    Ok(variances.mapv(|v| v.sqrt()))  // Return volatilities (σ_t)
}

/// Calculate GARCH(1,1) VaR
///
/// Uses conditional volatility forecast with normal distribution assumption.
///
/// # Arguments
/// * `returns` - Historical return series
/// * `params` - GARCH(1,1) parameters
/// * `confidence` - Confidence level (e.g., 0.99 for 99% VaR)
///
/// # Example
/// ```rust
/// use ndarray::array;
/// use hyperphysics_finance::risk::var::*;
///
/// let returns = array![-0.02, -0.01, 0.01, 0.02, 0.03];
/// let params = GarchParams::default_equity();
/// let var = garch_var(returns.view(), &params, 0.95).unwrap();
/// ```
pub fn garch_var(
    returns: ArrayView1<f64>,
    params: &GarchParams,
    confidence: f64,
) -> Result<f64, FinanceError> {
    let volatilities = garch_volatility(returns, params)?;
    let forecast_vol = volatilities[volatilities.len() - 1];

    // Calculate mean return (μ)
    let mean_return = returns.mean().unwrap_or(0.0);

    // VaR = -μ - σ_t·Φ^{-1}(α)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let alpha = 1.0 - confidence;
    let z_alpha = normal.inverse_cdf(alpha);

    let var = -mean_return - forecast_vol * z_alpha;

    Ok(var)
}

/// Calculate EWMA (RiskMetrics) conditional volatility
///
/// Uses exponentially weighted moving average: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
///
/// # Arguments
/// * `returns` - Historical return series
/// * `lambda` - Decay factor (typically 0.94 for daily data, 0.97 for monthly)
///
/// # Returns
/// Vector of conditional volatilities (σ_t)
pub fn ewma_volatility(
    returns: ArrayView1<f64>,
    lambda: f64,
) -> Result<Array1<f64>, FinanceError> {
    if returns.len() < 2 {
        return Err(FinanceError::InsufficientData);
    }
    if lambda <= 0.0 || lambda >= 1.0 {
        return Err(FinanceError::InvalidOptionParams(
            format!("Lambda must be in (0, 1): {}", lambda)
        ));
    }

    let n = returns.len();
    let mut variances = Array1::zeros(n);

    // Initialize with sample variance
    let mean_return = returns.mean().unwrap_or(0.0);
    let initial_var = returns.mapv(|r| (r - mean_return).powi(2)).mean().unwrap_or(0.0);
    variances[0] = initial_var;

    // EWMA recursion
    for t in 1..n {
        let r_sq = returns[t - 1].powi(2);
        variances[t] = lambda * variances[t - 1] + (1.0 - lambda) * r_sq;
    }

    Ok(variances.mapv(|v| v.sqrt()))
}

/// Calculate EWMA VaR (RiskMetrics methodology)
///
/// # Arguments
/// * `returns` - Historical return series
/// * `lambda` - Decay factor (typically 0.94)
/// * `confidence` - Confidence level (e.g., 0.99 for 99% VaR)
///
/// # Example
/// ```rust
/// use ndarray::array;
/// use hyperphysics_finance::risk::var::*;
///
/// let returns = array![-0.02, -0.01, 0.01, 0.02, 0.03];
/// let var = ewma_var(returns.view(), 0.94, 0.95).unwrap();
/// ```
pub fn ewma_var(
    returns: ArrayView1<f64>,
    lambda: f64,
    confidence: f64,
) -> Result<f64, FinanceError> {
    let volatilities = ewma_volatility(returns, lambda)?;
    let forecast_vol = volatilities[volatilities.len() - 1];

    // VaR = -σ_t·Φ^{-1}(α)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let alpha = 1.0 - confidence;
    let z_alpha = normal.inverse_cdf(alpha);

    let var = -forecast_vol * z_alpha;

    Ok(var)
}

/// Calculate VaR using specified model
pub fn calculate_var(
    returns: ArrayView1<f64>,
    model: VarModel,
    confidence: f64,
) -> Result<f64, FinanceError> {
    match model {
        VarModel::HistoricalSimulation => historical_var(returns, confidence),
        VarModel::Garch11 => {
            let params = GarchParams::default_equity();
            garch_var(returns, &params, confidence)
        }
        VarModel::EWMA => ewma_var(returns, 0.94, confidence),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_historical_var() {
        let returns = array![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05];
        let var_95 = historical_var(returns.view(), 0.95).unwrap();

        // 95% VaR should be around the 5th percentile
        assert!(var_95 > 0.0);
        assert!(var_95 < 0.06);
    }

    #[test]
    fn test_garch_params_validation() {
        // Valid parameters
        assert!(GarchParams::new(0.000001, 0.08, 0.90).is_ok());

        // Invalid: omega <= 0
        assert!(GarchParams::new(0.0, 0.08, 0.90).is_err());

        // Invalid: alpha < 0
        assert!(GarchParams::new(0.000001, -0.1, 0.90).is_err());

        // Invalid: alpha + beta >= 1
        assert!(GarchParams::new(0.000001, 0.6, 0.5).is_err());
    }

    #[test]
    fn test_garch_unconditional_variance() {
        let params = GarchParams::new(0.000001, 0.08, 0.90).unwrap();
        let uncond_var = params.unconditional_variance();

        // ω / (1 - α - β) = 0.000001 / (1 - 0.08 - 0.90) = 0.000001 / 0.02 = 0.00005
        assert_relative_eq!(uncond_var, 0.00005, epsilon = 1e-9);
    }

    #[test]
    fn test_garch_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02];
        let params = GarchParams::default_equity();

        let volatilities = garch_volatility(returns.view(), &params).unwrap();

        assert_eq!(volatilities.len(), returns.len());
        assert!(volatilities.iter().all(|&v| v > 0.0));

        // Volatility should be persistent (beta = 0.90)
        for i in 1..volatilities.len() {
            let ratio = volatilities[i] / volatilities[i - 1];
            assert!(ratio > 0.5 && ratio < 2.0);  // Reasonable persistence
        }
    }

    #[test]
    fn test_ewma_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02];
        let lambda = 0.94;

        let volatilities = ewma_volatility(returns.view(), lambda).unwrap();

        assert_eq!(volatilities.len(), returns.len());
        assert!(volatilities.iter().all(|&v| v > 0.0));

        // EWMA should adapt to new information
        // After large shock, volatility should increase
        assert!(volatilities[2] > volatilities[0]);
    }

    #[test]
    fn test_ewma_var() {
        let returns = array![
            -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02
        ];

        let var_95 = ewma_var(returns.view(), 0.94, 0.95).unwrap();

        assert!(var_95 > 0.0);
        // For symmetric distribution, VaR should be reasonable
        assert!(var_95 < 0.05);
    }

    #[test]
    fn test_var_model_comparison() {
        // Use more data for stable estimates
        let returns = array![
            -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02,
            -0.02, -0.01, 0.0, 0.01, 0.02
        ];

        let var_hist = calculate_var(returns.view(), VarModel::HistoricalSimulation, 0.95).unwrap();
        let var_garch = calculate_var(returns.view(), VarModel::Garch11, 0.95).unwrap();
        let var_ewma = calculate_var(returns.view(), VarModel::EWMA, 0.95).unwrap();

        // All VaRs should be positive
        assert!(var_hist > 0.0);
        assert!(var_garch > 0.0);
        assert!(var_ewma > 0.0);

        // Historical should be close to GARCH/EWMA for this data
        assert!((var_hist - var_garch).abs() < 0.05);
        assert!((var_hist - var_ewma).abs() < 0.05);
    }

    #[test]
    fn test_insufficient_data() {
        let returns = array![0.01];  // Only one point
        assert!(historical_var(returns.view(), 0.95).is_err());
        assert!(garch_volatility(returns.view(), &GarchParams::default_equity()).is_err());
    }
}
