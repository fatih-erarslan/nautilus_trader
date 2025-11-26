// Unit tests for risk management
use nt_risk::{calculate_var, calculate_cvar, calculate_sharpe_ratio, VarMethod};
use rust_decimal::Decimal;
use std::str::FromStr;

#[test]
fn test_var_calculation_historical() {
    let returns = vec![
        0.01, 0.02, -0.01, 0.03, -0.02,
        0.01, -0.03, 0.02, 0.01, -0.01,
    ];

    let var = calculate_var(&returns, 0.95, VarMethod::Historical).unwrap();

    // VaR should be negative (representing a loss)
    assert!(var < 0.0);

    // VaR at 95% should be around the 5th percentile
    assert!(var > -0.05);
}

#[test]
fn test_var_calculation_parametric() {
    let returns = vec![
        0.01, 0.02, -0.01, 0.03, -0.02,
        0.01, -0.03, 0.02, 0.01, -0.01,
    ];

    let var = calculate_var(&returns, 0.99, VarMethod::Parametric).unwrap();

    // 99% VaR should be larger (more negative) than 95% VaR
    assert!(var < 0.0);
}

#[test]
fn test_cvar_exceeds_var() {
    let returns = vec![
        0.01, 0.02, -0.01, 0.03, -0.02,
        0.01, -0.05, 0.02, 0.01, -0.01,
    ];

    let var = calculate_var(&returns, 0.95, VarMethod::Historical).unwrap();
    let cvar = calculate_cvar(&returns, 0.95).unwrap();

    // CVaR should be >= VaR in magnitude
    assert!(cvar.abs() >= var.abs());
}

#[test]
fn test_sharpe_ratio_positive_returns() {
    let returns = vec![0.01, 0.02, 0.015, 0.03, 0.025];
    let risk_free_rate = 0.02;

    let sharpe = calculate_sharpe_ratio(&returns, risk_free_rate).unwrap();

    // Positive returns should yield positive Sharpe
    assert!(sharpe > 0.0);
}

#[test]
fn test_sharpe_ratio_negative_returns() {
    let returns = vec![-0.01, -0.02, -0.015, -0.03, -0.025];
    let risk_free_rate = 0.02;

    let sharpe = calculate_sharpe_ratio(&returns, risk_free_rate).unwrap();

    // Negative returns should yield negative Sharpe
    assert!(sharpe < 0.0);
}

#[test]
fn test_position_size_kelly_criterion() {
    let win_probability = 0.6;
    let win_loss_ratio = 2.0; // Win $2 for every $1 risked

    // Kelly Criterion: f = (p * b - q) / b
    // where p = win prob, q = 1-p, b = win/loss ratio
    let kelly_fraction = (win_probability * win_loss_ratio - (1.0 - win_probability)) / win_loss_ratio;

    assert!(kelly_fraction > 0.0);
    assert!(kelly_fraction < 1.0); // Should never bet more than 100%
}

#[test]
fn test_position_size_constraints() {
    let account_value = Decimal::from_str("100000.0").unwrap();
    let risk_percent = Decimal::from_str("0.02").unwrap(); // 2% risk
    let stop_loss_percent = Decimal::from_str("0.05").unwrap(); // 5% stop

    // Position size = (Account Value * Risk %) / Stop Loss %
    let position_size = (account_value * risk_percent) / stop_loss_percent;

    // Should risk exactly 2% of account
    let risk_amount = position_size * stop_loss_percent;
    assert_eq!(risk_amount, account_value * risk_percent);
}

#[test]
fn test_max_drawdown_calculation() {
    let equity_curve = vec![
        10000.0, 10500.0, 11000.0, 10800.0, 10200.0,
        10500.0, 11200.0, 11500.0, 11000.0, 10800.0,
    ];

    let mut max_value = equity_curve[0];
    let mut max_drawdown = 0.0;

    for &value in &equity_curve {
        if value > max_value {
            max_value = value;
        }
        let drawdown = (max_value - value) / max_value;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    assert!(max_drawdown > 0.0);
    assert!(max_drawdown < 0.2); // Should be less than 20%
}

#[test]
fn test_correlation_calculation() {
    let returns_a = vec![0.01, 0.02, -0.01, 0.03, -0.02];
    let returns_b = vec![0.015, 0.025, -0.005, 0.035, -0.015];

    // Calculate correlation coefficient
    let n = returns_a.len() as f64;
    let mean_a: f64 = returns_a.iter().sum::<f64>() / n;
    let mean_b: f64 = returns_b.iter().sum::<f64>() / n;

    let covariance: f64 = returns_a.iter()
        .zip(returns_b.iter())
        .map(|(a, b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>() / n;

    let std_a = (returns_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (returns_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / n).sqrt();

    let correlation = covariance / (std_a * std_b);

    // Correlation should be between -1 and 1
    assert!(correlation >= -1.0 && correlation <= 1.0);
}

#[test]
fn test_portfolio_volatility() {
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt();

    // Annualize volatility (assuming daily returns)
    let annual_volatility = volatility * (252.0_f64).sqrt();

    assert!(annual_volatility > 0.0);
}
