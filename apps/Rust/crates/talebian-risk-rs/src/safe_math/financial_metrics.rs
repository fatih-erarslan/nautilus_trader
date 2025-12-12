//! Safe financial metric calculations with comprehensive error handling

use super::{SafeMathResult, safe_arithmetic::*, validation::*};
use crate::error::TalebianError;
use std::collections::VecDeque;

/// Financial metrics calculation result
#[derive(Debug, Clone)]
pub struct FinancialMetrics {
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
    pub tail_ratio: f64,
    pub volatility: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub recovery_factor: f64,
    pub ulcer_index: f64,
    pub sample_size: usize,
    pub confidence_level: f64,
}

impl FinancialMetrics {
    pub fn new() -> Self {
        Self {
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            var_95: 0.0,
            var_99: 0.0,
            expected_shortfall: 0.0,
            tail_ratio: 0.0,
            volatility: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            recovery_factor: 0.0,
            ulcer_index: 0.0,
            sample_size: 0,
            confidence_level: 0.0,
        }
    }
}

/// Safe financial metrics calculator
pub struct SafeFinancialCalculator {
    min_sample_size: usize,
    risk_free_rate: f64,
}

impl SafeFinancialCalculator {
    pub fn new(risk_free_rate: f64) -> SafeMathResult<Self> {
        if !risk_free_rate.is_finite() {
            return Err(TalebianError::data("Invalid risk-free rate"));
        }
        
        Ok(Self {
            min_sample_size: 30,
            risk_free_rate,
        })
    }
    
    /// Calculate comprehensive financial metrics from returns
    pub fn calculate_metrics(&self, returns: &[f64]) -> SafeMathResult<FinancialMetrics> {
        if returns.len() < self.min_sample_size {
            return Err(TalebianError::insufficient_data(self.min_sample_size, returns.len()));
        }
        
        // Validate returns data
        let validation_result = validate_returns_array(returns);
        if !validation_result.is_valid {
            return Err(TalebianError::data(format!(
                "Invalid returns data: {:?}", validation_result.errors
            )));
        }
        
        let mut metrics = FinancialMetrics::new();
        metrics.sample_size = returns.len();
        metrics.confidence_level = self.calculate_confidence_level(returns.len());
        
        // Basic statistics
        let mean_return = self.calculate_mean(returns)?;
        metrics.volatility = self.calculate_volatility(returns, mean_return)?;
        
        // Risk-adjusted returns
        metrics.sharpe_ratio = self.calculate_sharpe_ratio(mean_return, metrics.volatility)?;
        metrics.sortino_ratio = self.calculate_sortino_ratio(returns, mean_return)?;
        
        // Drawdown metrics
        let cumulative_returns = self.calculate_cumulative_returns(returns)?;
        metrics.max_drawdown = self.calculate_max_drawdown(&cumulative_returns)?;
        metrics.calmar_ratio = self.calculate_calmar_ratio(mean_return, metrics.max_drawdown)?;
        metrics.ulcer_index = self.calculate_ulcer_index(&cumulative_returns)?;
        metrics.recovery_factor = self.calculate_recovery_factor(
            cumulative_returns.last().copied().unwrap_or(0.0),
            metrics.max_drawdown
        )?;
        
        // Risk metrics
        metrics.var_95 = self.calculate_var(returns, 0.95)?;
        metrics.var_99 = self.calculate_var(returns, 0.99)?;
        metrics.expected_shortfall = self.calculate_expected_shortfall(returns, 0.95)?;
        
        // Distribution moments
        metrics.skewness = self.calculate_skewness(returns, mean_return, metrics.volatility)?;
        metrics.kurtosis = self.calculate_kurtosis(returns, mean_return, metrics.volatility)?;
        
        // Trading metrics
        let (wins, losses) = self.separate_wins_losses(returns);
        metrics.win_rate = self.calculate_win_rate(&wins, &losses)?;
        metrics.avg_win = self.calculate_average(&wins)?;
        metrics.avg_loss = self.calculate_average(&losses)?;
        metrics.profit_factor = self.calculate_profit_factor(metrics.avg_win, metrics.avg_loss, metrics.win_rate)?;
        metrics.tail_ratio = self.calculate_tail_ratio(returns)?;
        
        Ok(metrics)
    }
    
    /// Calculate mean return with overflow protection
    fn calculate_mean(&self, returns: &[f64]) -> SafeMathResult<f64> {
        if returns.is_empty() {
            return Err(TalebianError::data("Cannot calculate mean of empty returns"));
        }
        
        let sum = returns.iter().try_fold(0.0, |acc, &x| {
            safe_add(acc, x)
        })?;
        
        safe_divide(sum, returns.len() as f64)
    }
    
    /// Calculate volatility (standard deviation)
    fn calculate_volatility(&self, returns: &[f64], mean: f64) -> SafeMathResult<f64> {
        if returns.len() < 2 {
            return Err(TalebianError::data("Need at least 2 returns to calculate volatility"));
        }
        
        let variance = returns.iter().try_fold(0.0, |acc, &x| {
            let diff = safe_subtract(x, mean)?;
            let squared_diff = safe_multiply(diff, diff)?;
            safe_add(acc, squared_diff)
        })?;
        
        let avg_variance = safe_divide(variance, (returns.len() - 1) as f64)?;
        safe_sqrt(avg_variance)
    }
    
    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, mean_return: f64, volatility: f64) -> SafeMathResult<f64> {
        if volatility < 1e-10 {
            return Ok(0.0); // No volatility means no risk-adjusted return
        }
        
        let excess_return = safe_subtract(mean_return, self.risk_free_rate)?;
        safe_divide(excess_return, volatility)
    }
    
    /// Calculate Sortino ratio (downside deviation)
    fn calculate_sortino_ratio(&self, returns: &[f64], mean_return: f64) -> SafeMathResult<f64> {
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        
        if downside_returns.is_empty() {
            return Ok(f64::INFINITY); // No downside means infinite Sortino
        }
        
        let downside_variance = downside_returns.iter().try_fold(0.0, |acc, &x| {
            let squared_return = safe_multiply(x, x)?;
            safe_add(acc, squared_return)
        })?;
        
        let avg_downside_variance = safe_divide(downside_variance, downside_returns.len() as f64)?;
        let downside_deviation = safe_sqrt(avg_downside_variance)?;
        
        if downside_deviation < 1e-10 {
            return Ok(f64::INFINITY);
        }
        
        let excess_return = safe_subtract(mean_return, self.risk_free_rate)?;
        safe_divide(excess_return, downside_deviation)
    }
    
    /// Calculate cumulative returns
    fn calculate_cumulative_returns(&self, returns: &[f64]) -> SafeMathResult<Vec<f64>> {
        let mut cumulative = Vec::with_capacity(returns.len());
        let mut running_total = 1.0;
        
        for &ret in returns {
            let one_plus_return = safe_add(1.0, ret)?;
            running_total = safe_multiply(running_total, one_plus_return)?;
            cumulative.push(running_total);
        }
        
        Ok(cumulative)
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, cumulative_returns: &[f64]) -> SafeMathResult<f64> {
        if cumulative_returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut max_value = cumulative_returns[0];
        let mut max_drawdown = 0.0;
        
        for &value in cumulative_returns.iter().skip(1) {
            if value > max_value {
                max_value = value;
            } else {
                let drawdown = safe_divide(safe_subtract(max_value, value)?, max_value)?;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate Calmar ratio
    fn calculate_calmar_ratio(&self, annual_return: f64, max_drawdown: f64) -> SafeMathResult<f64> {
        if max_drawdown < 1e-10 {
            return Ok(f64::INFINITY);
        }
        
        safe_divide(annual_return, max_drawdown)
    }
    
    /// Calculate Ulcer Index
    fn calculate_ulcer_index(&self, cumulative_returns: &[f64]) -> SafeMathResult<f64> {
        if cumulative_returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut running_max = cumulative_returns[0];
        let mut squared_drawdowns_sum = 0.0;
        
        for &value in cumulative_returns {
            if value > running_max {
                running_max = value;
            }
            
            let drawdown_pct = safe_divide(safe_subtract(running_max, value)?, running_max)?;
            let squared_drawdown = safe_multiply(drawdown_pct, drawdown_pct)?;
            squared_drawdowns_sum = safe_add(squared_drawdowns_sum, squared_drawdown)?;
        }
        
        let mean_squared_drawdown = safe_divide(squared_drawdowns_sum, cumulative_returns.len() as f64)?;
        safe_sqrt(mean_squared_drawdown)
    }
    
    /// Calculate Value at Risk (VaR)
    fn calculate_var(&self, returns: &[f64], confidence_level: f64) -> SafeMathResult<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let safe_index = index.min(sorted_returns.len() - 1);
        
        Ok(-sorted_returns[safe_index]) // VaR is typically reported as positive
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    fn calculate_expected_shortfall(&self, returns: &[f64], confidence_level: f64) -> SafeMathResult<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let var = self.calculate_var(returns, confidence_level)?;
        let threshold = -var;
        
        let tail_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r <= threshold)
            .copied()
            .collect();
        
        if tail_returns.is_empty() {
            return Ok(var);
        }
        
        let tail_mean = self.calculate_mean(&tail_returns)?;
        Ok(-tail_mean)
    }
    
    /// Calculate skewness
    fn calculate_skewness(&self, returns: &[f64], mean: f64, std_dev: f64) -> SafeMathResult<f64> {
        if std_dev < 1e-10 || returns.len() < 3 {
            return Ok(0.0);
        }
        
        let n = returns.len() as f64;
        let skew_sum = returns.iter().try_fold(0.0, |acc, &x| {
            let standardized = safe_divide(safe_subtract(x, mean)?, std_dev)?;
            let cubed = safe_pow(standardized, 3.0)?;
            safe_add(acc, cubed)
        })?;
        
        let skewness = safe_divide(skew_sum, n)?;
        Ok(skewness)
    }
    
    /// Calculate kurtosis
    fn calculate_kurtosis(&self, returns: &[f64], mean: f64, std_dev: f64) -> SafeMathResult<f64> {
        if std_dev < 1e-10 || returns.len() < 4 {
            return Ok(0.0);
        }
        
        let n = returns.len() as f64;
        let kurt_sum = returns.iter().try_fold(0.0, |acc, &x| {
            let standardized = safe_divide(safe_subtract(x, mean)?, std_dev)?;
            let fourth_power = safe_pow(standardized, 4.0)?;
            safe_add(acc, fourth_power)
        })?;
        
        let kurtosis = safe_divide(kurt_sum, n)?;
        safe_subtract(kurtosis, 3.0) // Excess kurtosis
    }
    
    /// Separate wins and losses
    fn separate_wins_losses(&self, returns: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut wins = Vec::new();
        let mut losses = Vec::new();
        
        for &ret in returns {
            if ret > 0.0 {
                wins.push(ret);
            } else if ret < 0.0 {
                losses.push(ret.abs());
            }
        }
        
        (wins, losses)
    }
    
    /// Calculate win rate
    fn calculate_win_rate(&self, wins: &[f64], losses: &[f64]) -> SafeMathResult<f64> {
        let total_trades = wins.len() + losses.len();
        if total_trades == 0 {
            return Ok(0.0);
        }
        
        safe_divide(wins.len() as f64, total_trades as f64)
    }
    
    /// Calculate average of array
    fn calculate_average(&self, values: &[f64]) -> SafeMathResult<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }
        
        self.calculate_mean(values)
    }
    
    /// Calculate profit factor
    fn calculate_profit_factor(&self, avg_win: f64, avg_loss: f64, win_rate: f64) -> SafeMathResult<f64> {
        if avg_loss < 1e-10 {
            return Ok(f64::INFINITY);
        }
        
        let gross_profit = safe_multiply(avg_win, win_rate)?;
        let gross_loss = safe_multiply(avg_loss, safe_subtract(1.0, win_rate)?)?;
        
        safe_divide(gross_profit, gross_loss)
    }
    
    /// Calculate tail ratio
    fn calculate_tail_ratio(&self, returns: &[f64]) -> SafeMathResult<f64> {
        let var_95 = self.calculate_var(returns, 0.95)?;
        let var_99 = self.calculate_var(returns, 0.99)?;
        
        if var_99 < 1e-10 {
            return Ok(1.0);
        }
        
        safe_divide(var_95, var_99)
    }
    
    /// Calculate recovery factor
    fn calculate_recovery_factor(&self, total_return: f64, max_drawdown: f64) -> SafeMathResult<f64> {
        if max_drawdown < 1e-10 {
            return Ok(f64::INFINITY);
        }
        
        safe_divide(total_return, max_drawdown)
    }
    
    /// Calculate confidence level based on sample size
    fn calculate_confidence_level(&self, sample_size: usize) -> f64 {
        if sample_size < self.min_sample_size {
            return 0.0;
        }
        
        // Higher sample size increases confidence
        let confidence = 1.0 - (self.min_sample_size as f64 / sample_size as f64);
        confidence.min(0.95)
    }
}

/// Rolling window calculator for real-time metrics
pub struct RollingMetricsCalculator {
    calculator: SafeFinancialCalculator,
    window_size: usize,
    returns_buffer: VecDeque<f64>,
}

impl RollingMetricsCalculator {
    pub fn new(risk_free_rate: f64, window_size: usize) -> SafeMathResult<Self> {
        if window_size < 30 {
            return Err(TalebianError::data("Window size must be at least 30"));
        }
        
        Ok(Self {
            calculator: SafeFinancialCalculator::new(risk_free_rate)?,
            window_size,
            returns_buffer: VecDeque::with_capacity(window_size),
        })
    }
    
    /// Add new return and calculate rolling metrics
    pub fn add_return(&mut self, return_value: f64) -> SafeMathResult<Option<FinancialMetrics>> {
        if !return_value.is_finite() {
            return Err(TalebianError::data(format!("Invalid return: {}", return_value)));
        }
        
        self.returns_buffer.push_back(return_value);
        
        if self.returns_buffer.len() > self.window_size {
            self.returns_buffer.pop_front();
        }
        
        if self.returns_buffer.len() < self.calculator.min_sample_size {
            return Ok(None);
        }
        
        let returns_vec: Vec<f64> = self.returns_buffer.iter().copied().collect();
        let metrics = self.calculator.calculate_metrics(&returns_vec)?;
        Ok(Some(metrics))
    }
    
    /// Get current metrics without adding new return
    pub fn get_current_metrics(&self) -> SafeMathResult<Option<FinancialMetrics>> {
        if self.returns_buffer.len() < self.calculator.min_sample_size {
            return Ok(None);
        }
        
        let returns_vec: Vec<f64> = self.returns_buffer.iter().copied().collect();
        let metrics = self.calculator.calculate_metrics(&returns_vec)?;
        Ok(Some(metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_returns() -> Vec<f64> {
        vec![
            0.01, -0.02, 0.015, -0.01, 0.03, -0.005, 0.02, -0.015,
            0.025, -0.008, 0.012, -0.018, 0.008, -0.022, 0.035,
            -0.012, 0.018, -0.007, 0.028, -0.014, 0.009, -0.025,
            0.016, -0.011, 0.021, -0.006, 0.013, -0.019, 0.024,
            -0.003, 0.017, -0.013, 0.011, -0.026, 0.019, -0.004
        ]
    }

    #[test]
    fn test_financial_calculator_creation() {
        let calculator = SafeFinancialCalculator::new(0.02);
        assert!(calculator.is_ok());
        
        let invalid_calculator = SafeFinancialCalculator::new(f64::NAN);
        assert!(invalid_calculator.is_err());
    }

    #[test]
    fn test_calculate_metrics() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        let returns = generate_test_returns();
        
        let metrics = calculator.calculate_metrics(&returns);
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_insufficient_data() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        let returns = vec![0.01, -0.02]; // Too few returns
        
        let metrics = calculator.calculate_metrics(&returns);
        assert!(metrics.is_err());
    }

    #[test]
    fn test_rolling_calculator() {
        let mut rolling_calc = RollingMetricsCalculator::new(0.02, 50).unwrap();
        
        // Add insufficient data
        for i in 0..20 {
            let result = rolling_calc.add_return(0.01 * i as f64);
            assert!(result.is_ok());
            assert!(result.unwrap().is_none());
        }
        
        // Add sufficient data
        for i in 20..60 {
            let result = rolling_calc.add_return(0.01 * (i % 10) as f64);
            assert!(result.is_ok());
            if i >= 30 {
                assert!(result.unwrap().is_some());
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        let calculator = SafeFinancialCalculator::new(0.02).unwrap();
        
        // All zero returns
        let zero_returns = vec![0.0; 50];
        let metrics = calculator.calculate_metrics(&zero_returns);
        assert!(metrics.is_ok());
        
        // All positive returns
        let positive_returns = vec![0.01; 50];
        let metrics = calculator.calculate_metrics(&positive_returns);
        assert!(metrics.is_ok());
    }
}