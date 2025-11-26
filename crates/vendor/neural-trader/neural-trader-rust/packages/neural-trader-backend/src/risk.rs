//! Risk management module - Real-time risk monitoring and VaR calculation
//! Provides GPU-accelerated risk analysis and position limits

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::error::*;

/// Calculate Value at Risk (VaR) for portfolio
#[napi]
pub async fn calculate_var(portfolio: Vec<String>, confidence: f64) -> Result<f64> {
    // Validate portfolio
    if portfolio.is_empty() {
        return Err(NeuralTraderError::Portfolio(
            "Portfolio cannot be empty for VaR calculation".to_string()
        ).into());
    }

    // Validate confidence level
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(NeuralTraderError::Validation(
            format!("Confidence level {} must be between 0 and 1 (e.g., 0.95 for 95%)", confidence)
        ).into());
    }

    // Common confidence levels
    if ![0.90, 0.95, 0.99].iter().any(|&c| (c - confidence).abs() < 0.001) {
        tracing::warn!(
            "Unusual confidence level {:.3}. Common values are 0.90, 0.95, or 0.99",
            confidence
        );
    }

    // TODO: Integrate with nt-risk
    Ok(-0.05) // VaR value (negative indicates potential loss)
}

/// Monitor real-time risk exposure
#[napi]
pub async fn monitor_risk(portfolio_id: String) -> Result<RiskMetrics> {
    // Validate portfolio ID
    if portfolio_id.is_empty() {
        return Err(NeuralTraderError::Portfolio(
            "Portfolio ID cannot be empty for risk monitoring".to_string()
        ).into());
    }

    tracing::info!("Monitoring risk for portfolio: {}", portfolio_id);

    // TODO: Integrate with nt-risk
    Ok(RiskMetrics {
        var_95: -0.05,
        cvar_95: -0.08,
        exposure: 0.75,
    })
}

#[napi(object)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub cvar_95: f64,
    pub exposure: f64,
}
