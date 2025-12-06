//! Utility functions and data structures for hedge algorithms

use nalgebra::{DVector, DMatrix};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::HedgeError;

// MarketData is defined in lib.rs - using that implementation

// HedgeRecommendation and RiskMetrics are defined in lib.rs - using those implementations

/// Expert trait for hedge algorithms
pub trait Expert {
    /// Update expert with new market data
    fn update(&self, market_data: &crate::MarketData) -> Result<(), HedgeError>;
    
    /// Get expert signal
    fn get_signal(&self) -> Result<f64, HedgeError>;
    
    /// Get expert name
    fn get_name(&self) -> &str;
    
    /// Get expert confidence
    fn get_confidence(&self) -> f64;
    
    /// Get expert performance history
    fn get_performance_history(&self) -> Vec<f64> {
        Vec::new()
    }
    
    /// Reset expert state
    fn reset(&self) -> Result<(), HedgeError> {
        Ok(())
    }
}

/// Performance metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Total return
    pub total_return: f64,
    /// Average return
    pub average_return: f64,
    /// Best trade
    pub best_trade: f64,
    /// Worst trade
    pub worst_trade: f64,
    /// Consecutive wins
    pub consecutive_wins: usize,
    /// Consecutive losses
    pub consecutive_losses: usize,
    /// Maximum consecutive wins
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losses
    pub max_consecutive_losses: usize,
    /// Profit factor
    pub profit_factor: f64,
    /// Recovery factor
    pub recovery_factor: f64,
    /// Payoff ratio
    pub payoff_ratio: f64,
    /// Hit rate
    pub hit_rate: f64,
    /// Standard deviation
    pub standard_deviation: f64,
    /// Downside deviation
    pub downside_deviation: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Sterling ratio
    pub sterling_ratio: f64,
    /// Burke ratio
    pub burke_ratio: f64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_return: 0.0,
            average_return: 0.0,
            best_trade: 0.0,
            worst_trade: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            profit_factor: 0.0,
            recovery_factor: 0.0,
            payoff_ratio: 0.0,
            hit_rate: 0.0,
            standard_deviation: 0.0,
            downside_deviation: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            sterling_ratio: 0.0,
            burke_ratio: 0.0,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Update metrics with new recommendation
    pub fn update(&mut self, recommendation: &crate::HedgeRecommendation) -> Result<(), HedgeError> {
        self.total_trades += 1;
        
        let trade_return = recommendation.expected_return;
        self.total_return += trade_return;
        self.average_return = self.total_return / self.total_trades as f64;
        
        if trade_return > 0.0 {
            self.winning_trades += 1;
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
            
            if self.consecutive_wins > self.max_consecutive_wins {
                self.max_consecutive_wins = self.consecutive_wins;
            }
            
            if trade_return > self.best_trade {
                self.best_trade = trade_return;
            }
        } else {
            self.losing_trades += 1;
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
            
            if self.consecutive_losses > self.max_consecutive_losses {
                self.max_consecutive_losses = self.consecutive_losses;
            }
            
            if trade_return < self.worst_trade {
                self.worst_trade = trade_return;
            }
        }
        
        self.hit_rate = self.winning_trades as f64 / self.total_trades as f64;
        
        // Calculate profit factor
        let gross_profit = self.winning_trades as f64 * self.best_trade;
        let gross_loss = self.losing_trades as f64 * self.worst_trade.abs();
        self.profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            0.0
        };
        
        // Calculate payoff ratio
        let avg_win = if self.winning_trades > 0 {
            self.best_trade
        } else {
            0.0
        };
        let avg_loss = if self.losing_trades > 0 {
            self.worst_trade.abs()
        } else {
            0.0
        };
        self.payoff_ratio = if avg_loss > 0.0 {
            avg_win / avg_loss
        } else {
            0.0
        };
        
        // Calculate Sortino ratio
        self.sortino_ratio = if recommendation.volatility > 0.0 {
            self.average_return / recommendation.volatility
        } else {
            0.0
        };
        
        // Calculate Calmar ratio
        self.calmar_ratio = if recommendation.max_drawdown > 0.0 {
            self.average_return / recommendation.max_drawdown
        } else {
            0.0
        };
        
        self.timestamp = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();
        
        summary.insert("total_trades".to_string(), self.total_trades as f64);
        summary.insert("winning_trades".to_string(), self.winning_trades as f64);
        summary.insert("losing_trades".to_string(), self.losing_trades as f64);
        summary.insert("total_return".to_string(), self.total_return);
        summary.insert("average_return".to_string(), self.average_return);
        summary.insert("best_trade".to_string(), self.best_trade);
        summary.insert("worst_trade".to_string(), self.worst_trade);
        summary.insert("hit_rate".to_string(), self.hit_rate);
        summary.insert("profit_factor".to_string(), self.profit_factor);
        summary.insert("payoff_ratio".to_string(), self.payoff_ratio);
        summary.insert("sortino_ratio".to_string(), self.sortino_ratio);
        summary.insert("calmar_ratio".to_string(), self.calmar_ratio);
        
        summary
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Mathematical utility functions
pub mod math {
    use super::*;
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64, HedgeError> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(HedgeError::Math("Invalid input for correlation calculation".to_string()));
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate covariance
    pub fn covariance(x: &[f64], y: &[f64]) -> Result<f64, HedgeError> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(HedgeError::Math("Invalid input for covariance calculation"));
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let covariance = x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / (n - 1.0);
        
        Ok(covariance)
    }
    
    /// Calculate variance
    pub fn variance(data: &[f64]) -> Result<f64, HedgeError> {
        if data.len() < 2 {
            return Err(HedgeError::Math("Insufficient data for variance calculation"));
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        Ok(variance)
    }
    
    /// Calculate standard deviation
    pub fn standard_deviation(data: &[f64]) -> Result<f64, HedgeError> {
        let var = variance(data)?;
        Ok(var.sqrt())
    }
    
    /// Calculate exponential moving average
    pub fn ema(data: &[f64], alpha: f64) -> Result<Vec<f64>, HedgeError> {
        if data.is_empty() {
            return Err(HedgeError::Math("Empty data for EMA calculation"));
        }
        
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(HedgeError::Math("Alpha must be between 0 and 1"));
        }
        
        let mut result = Vec::with_capacity(data.len());
        result.push(data[0]);
        
        for i in 1..data.len() {
            let ema_value = alpha * data[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema_value);
        }
        
        Ok(result)
    }
    
    /// Calculate simple moving average
    pub fn sma(data: &[f64], window: usize) -> Result<Vec<f64>, HedgeError> {
        if data.len() < window {
            return Err(HedgeError::Math("Insufficient data for SMA calculation"));
        }
        
        let mut result = Vec::with_capacity(data.len() - window + 1);
        
        for i in window..=data.len() {
            let sum = data[i - window..i].iter().sum::<f64>();
            result.push(sum / window as f64);
        }
        
        Ok(result)
    }
    
    /// Calculate returns
    pub fn returns(prices: &[f64]) -> Result<Vec<f64>, HedgeError> {
        if prices.len() < 2 {
            return Err(HedgeError::Math("Insufficient data for returns calculation"));
        }
        
        let mut returns = Vec::with_capacity(prices.len() - 1);
        
        for i in 1..prices.len() {
            let return_value = (prices[i] - prices[i - 1]) / prices[i - 1];
            returns.push(return_value);
        }
        
        Ok(returns)
    }
    
    /// Calculate log returns
    pub fn log_returns(prices: &[f64]) -> Result<Vec<f64>, HedgeError> {
        if prices.len() < 2 {
            return Err(HedgeError::Math("Insufficient data for log returns calculation"));
        }
        
        let mut log_returns = Vec::with_capacity(prices.len() - 1);
        
        for i in 1..prices.len() {
            if prices[i] <= 0.0 || prices[i - 1] <= 0.0 {
                return Err(HedgeError::Math("Prices must be positive for log returns"));
            }
            
            let log_return = (prices[i] / prices[i - 1]).ln();
            log_returns.push(log_return);
        }
        
        Ok(log_returns)
    }
}

/// Statistical utility functions
pub mod stats {
    use super::*;
    use statrs::distribution::{Normal, ContinuousCDF};
    
    /// Calculate percentile
    pub fn percentile(data: &[f64], p: f64) -> Result<f64, HedgeError> {
        if data.is_empty() {
            return Err(HedgeError::Math("Empty data for percentile calculation"));
        }
        
        if p < 0.0 || p > 1.0 {
            return Err(HedgeError::Math("Percentile must be between 0 and 1"));
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = p * (sorted_data.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        
        if lower_index == upper_index {
            Ok(sorted_data[lower_index])
        } else {
            let weight = index - lower_index as f64;
            Ok(sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight)
        }
    }
    
    /// Calculate Value at Risk
    pub fn var(returns: &[f64], confidence: f64) -> Result<f64, HedgeError> {
        if returns.is_empty() {
            return Err(HedgeError::Math("Empty returns for VaR calculation"));
        }
        
        let alpha = 1.0 - confidence;
        let var_value = percentile(returns, alpha)?;
        
        Ok(-var_value) // VaR is typically reported as a positive number
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    pub fn expected_shortfall(returns: &[f64], confidence: f64) -> Result<f64, HedgeError> {
        if returns.is_empty() {
            return Err(HedgeError::Math("Empty returns for ES calculation"));
        }
        
        let alpha = 1.0 - confidence;
        let var_value = percentile(returns, alpha)?;
        
        let tail_returns: Vec<f64> = returns.iter()
            .copied()
            .filter(|&r| r <= var_value)
            .collect();
        
        if tail_returns.is_empty() {
            return Ok(0.0);
        }
        
        let es = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        Ok(-es) // ES is typically reported as a positive number
    }
    
    /// Calculate skewness
    pub fn skewness(data: &[f64]) -> Result<f64, HedgeError> {
        if data.len() < 3 {
            return Err(HedgeError::Math("Insufficient data for skewness calculation"));
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(0.0);
        }
        
        let skewness = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / data.len() as f64;
        
        Ok(skewness)
    }
    
    /// Calculate kurtosis
    pub fn kurtosis(data: &[f64]) -> Result<f64, HedgeError> {
        if data.len() < 4 {
            return Err(HedgeError::Math("Insufficient data for kurtosis calculation"));
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(0.0);
        }
        
        let kurtosis = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / data.len() as f64;
        
        Ok(kurtosis - 3.0) // Excess kurtosis
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::math::*;
    use super::stats::*;

    #[test]
    fn test_market_data_creation() {
        let timestamp = chrono::Utc::now();
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            timestamp,
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        assert_eq!(market_data.symbol, "BTCUSD");
        assert_eq!(market_data.open, 100.0);
        assert_eq!(market_data.high, 105.0);
        assert_eq!(market_data.low, 95.0);
        assert_eq!(market_data.close, 102.0);
        assert_eq!(market_data.volume, 1000.0);
        assert_eq!(market_data.typical_price(), 100.66666666666667);
    }

    #[test]
    fn test_correlation_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance(&data).unwrap();
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_returns_calculation() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = returns(&prices).unwrap();
        
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-0.045454545454545456)).abs() < 1e-10);
        assert!((returns[2] - 0.09523809523809523).abs() < 1e-10);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        let recommendation = HedgeRecommendation {
            position_size: 1.0,
            hedge_ratio: 0.5,
            confidence: 0.8,
            factor_exposures: DVector::zeros(8),
            risk_metrics: RiskMetrics::default(),
            expected_return: 0.05,
            volatility: 0.15,
            max_drawdown: 0.02,
            sharpe_ratio: 0.33,
            timestamp: chrono::Utc::now(),
        };
        
        metrics.update(&recommendation).unwrap();
        
        assert_eq!(metrics.total_trades, 1);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 0);
        assert_eq!(metrics.total_return, 0.05);
        assert_eq!(metrics.hit_rate, 1.0);
    }
}