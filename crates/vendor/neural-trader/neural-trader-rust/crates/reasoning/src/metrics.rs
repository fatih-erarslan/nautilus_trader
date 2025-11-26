//! Financial metrics calculations for pattern evaluation

/// Calculate annualized Sharpe ratio from returns
///
/// # Arguments
/// * `returns` - Slice of return values
/// * `risk_free_rate` - Annual risk-free rate (default 0.02)
///
/// # Returns
/// Annualized Sharpe ratio (higher is better)
pub fn calculate_sharpe_ratio(returns: &[f64]) -> f64 {
    calculate_sharpe_ratio_with_rf(returns, 0.02)
}

/// Calculate Sharpe ratio with custom risk-free rate
pub fn calculate_sharpe_ratio_with_rf(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    // Annualize: assume daily returns, 252 trading days
    let excess_return = mean - (risk_free_rate / 252.0);
    excess_return / std_dev * (252.0_f64).sqrt()
}

/// Calculate Sortino ratio (downside deviation only)
///
/// # Arguments
/// * `returns` - Slice of return values
/// * `target_return` - Minimum acceptable return (default 0.0)
///
/// # Returns
/// Annualized Sortino ratio
pub fn calculate_sortino_ratio(returns: &[f64]) -> f64 {
    calculate_sortino_ratio_with_target(returns, 0.0)
}

/// Calculate Sortino ratio with custom target return
pub fn calculate_sortino_ratio_with_target(returns: &[f64], target_return: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // Calculate downside deviation
    let downside_variance = returns
        .iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 {
        return if mean > target_return { f64::INFINITY } else { 0.0 };
    }

    // Annualize
    (mean - target_return) / downside_dev * (252.0_f64).sqrt()
}

/// Calculate maximum drawdown
///
/// # Arguments
/// * `returns` - Slice of return values
///
/// # Returns
/// Maximum drawdown as a positive percentage (0-1)
pub fn calculate_max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    // Calculate cumulative returns
    let mut cumulative = vec![1.0];
    for &ret in returns {
        let last = cumulative.last().unwrap();
        cumulative.push(last * (1.0 + ret));
    }

    // Find maximum drawdown
    let mut max_dd = 0.0;
    let mut peak = cumulative[0];

    for &value in &cumulative[1..] {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Calculate Calmar ratio (return / max drawdown)
pub fn calculate_calmar_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let max_dd = calculate_max_drawdown(returns);

    if max_dd == 0.0 {
        return if mean_return > 0.0 { f64::INFINITY } else { 0.0 };
    }

    // Annualize return
    (mean_return * 252.0) / max_dd
}

/// Calculate win rate (percentage of profitable trades)
pub fn calculate_win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let profitable = returns.iter().filter(|&&r| r > 0.0).count();
    profitable as f64 / returns.len() as f64
}

/// Calculate profit factor (gross profit / gross loss)
pub fn calculate_profit_factor(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 1.0;
    }

    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if gross_loss == 0.0 {
        return if gross_profit > 0.0 { f64::INFINITY } else { 1.0 };
    }

    gross_profit / gross_loss
}

/// Calculate Value at Risk (VaR) at given confidence level
///
/// # Arguments
/// * `returns` - Slice of return values
/// * `confidence` - Confidence level (e.g., 0.95 for 95%)
pub fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    -sorted[index.min(sorted.len() - 1)]
}

/// Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR
pub fn calculate_cvar(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let var = calculate_var(returns, confidence);
    let threshold = -var;

    let tail_losses: Vec<f64> = returns
        .iter()
        .filter(|&&r| r <= threshold)
        .copied()
        .collect();

    if tail_losses.is_empty() {
        return var;
    }

    -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.03];
        let sharpe = calculate_sharpe_ratio(&returns);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.03];
        let sortino = calculate_sortino_ratio(&returns);
        assert!(sortino > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, 0.05, -0.15, -0.1, 0.2];
        let max_dd = calculate_max_drawdown(&returns);
        assert!(max_dd > 0.0 && max_dd < 1.0);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
        let win_rate = calculate_win_rate(&returns);
        assert_eq!(win_rate, 0.6); // 3 out of 5 profitable
    }

    #[test]
    fn test_profit_factor() {
        let returns = vec![0.1, 0.2, -0.05, -0.05];
        let pf = calculate_profit_factor(&returns);
        assert_eq!(pf, 3.0); // 0.3 profit / 0.1 loss
    }
}
