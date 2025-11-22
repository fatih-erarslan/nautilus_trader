//! Risk module - VaR, CVaR, and risk metrics

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::RiskMetricsResult;

/// Calculate risk metrics synchronously
pub fn calculate_risk_sync(returns: &Float64Array, _confidence: f64) -> Result<RiskMetricsResult> {
    use ndarray::Array1;
    use hyperphysics_finance::RiskMetrics;

    let data = returns.as_ref();
    if data.is_empty() {
        return Err(Error::new(Status::InvalidArg, "Returns array cannot be empty"));
    }

    let returns_array = Array1::from_vec(data.to_vec());
    let metrics = RiskMetrics::from_returns(returns_array.view(), 252.0)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Risk calculation failed: {:?}", e)))?;

    Ok(RiskMetricsResult {
        var_95: metrics.var_95,
        var_99: metrics.var_99,
        expected_shortfall: metrics.cvar_95,  // CVaR is the expected shortfall
        volatility: metrics.volatility,
        max_drawdown: metrics.max_drawdown,
        sharpe_ratio: metrics.sharpe_ratio,
    })
}

/// Risk Engine for portfolio analysis
#[napi]
pub struct RiskEngine {
    periods_per_year: f64,
}

#[napi]
impl RiskEngine {
    #[napi(constructor)]
    pub fn new(periods_per_year: Option<f64>) -> Self {
        Self { periods_per_year: periods_per_year.unwrap_or(252.0) }
    }

    #[napi]
    pub fn calculate_metrics(&self, returns: Float64Array) -> Result<RiskMetricsResult> {
        calculate_risk_sync(&returns, 0.95)
    }

    #[napi]
    pub fn var(&self, returns: Float64Array, confidence: f64) -> Result<f64> {
        let data = returns.as_ref();
        if data.is_empty() {
            return Err(Error::new(Status::InvalidArg, "Returns cannot be empty"));
        }
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        Ok(-sorted[idx.min(sorted.len() - 1)])
    }

    #[napi]
    pub fn volatility(&self, returns: Float64Array) -> Result<f64> {
        let data = returns.as_ref();
        if data.len() < 2 {
            return Err(Error::new(Status::InvalidArg, "Need at least 2 returns"));
        }
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        Ok(variance.sqrt() * self.periods_per_year.sqrt())
    }

    #[napi]
    pub fn sharpe_ratio(&self, returns: Float64Array, risk_free_rate: Option<f64>) -> Result<f64> {
        let data = returns.as_ref();
        let rf = risk_free_rate.unwrap_or(0.0);
        if data.len() < 2 {
            return Err(Error::new(Status::InvalidArg, "Need at least 2 returns"));
        }
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev < 1e-10 { return Ok(0.0); }
        let annual_return = mean * self.periods_per_year;
        let annual_std = std_dev * self.periods_per_year.sqrt();
        Ok((annual_return - rf) / annual_std)
    }

    #[napi]
    pub fn max_drawdown(&self, returns: Float64Array) -> Result<f64> {
        let data = returns.as_ref();
        if data.is_empty() {
            return Err(Error::new(Status::InvalidArg, "Returns cannot be empty"));
        }
        let mut cum = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        for &r in data.iter() {
            cum *= 1.0 + r;
            if cum > peak { peak = cum; }
            let dd = (peak - cum) / peak;
            if dd > max_dd { max_dd = dd; }
        }
        Ok(max_dd)
    }
}
