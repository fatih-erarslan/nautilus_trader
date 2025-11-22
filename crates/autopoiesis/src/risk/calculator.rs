//! Risk calculation implementation

use crate::prelude::*;
use crate::models::{Position, MarketData};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Risk calculator for portfolio risk metrics
#[derive(Debug, Clone)]
pub struct RiskCalculator {
    /// Calculator configuration
    config: RiskCalculatorConfig,
    
    /// Historical data buffer
    historical_data: Vec<MarketData>,
    
    /// Risk metrics cache
    metrics_cache: RiskMetricsCache,
}

#[derive(Debug, Clone)]
pub struct RiskCalculatorConfig {
    /// VaR confidence level
    pub var_confidence_level: f64,
    
    /// VaR lookback period in days
    pub var_lookback_days: u32,
    
    /// Risk-free rate for Sharpe ratio
    pub risk_free_rate: f64,
    
    /// Correlation calculation period
    pub correlation_period_days: u32,
}

#[derive(Debug, Clone, Default)]
struct RiskMetricsCache {
    var_95: Option<f64>,
    var_99: Option<f64>,
    expected_shortfall: Option<f64>,
    max_drawdown: Option<f64>,
    sharpe_ratio: Option<f64>,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub value_at_risk_95: f64,
    pub value_at_risk_99: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub maximum_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub beta: f64,
    pub alpha: f64,
    pub volatility: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub correlation_matrix: HashMap<String, HashMap<String, f64>>,
}

impl Default for RiskCalculatorConfig {
    fn default() -> Self {
        Self {
            var_confidence_level: 0.95,
            var_lookback_days: 252,
            risk_free_rate: 0.02,
            correlation_period_days: 90,
        }
    }
}

impl RiskCalculator {
    /// Create a new risk calculator
    pub fn new(config: RiskCalculatorConfig) -> Self {
        Self {
            config,
            historical_data: Vec::new(),
            metrics_cache: RiskMetricsCache::default(),
        }
    }

    /// Calculate risk metrics for a portfolio
    pub async fn calculate_portfolio_risk(&mut self, positions: &[Position], market_data: &[MarketData]) -> Result<RiskMetrics> {
        // Update historical data
        self.historical_data.extend_from_slice(market_data);
        
        // Keep only required lookback period
        let max_history = self.config.var_lookback_days as usize * 24 * 60; // Assuming minute data
        if self.historical_data.len() > max_history {
            self.historical_data.drain(0..self.historical_data.len() - max_history);
        }

        // Calculate VaR
        let var_95 = self.calculate_var(0.95)?;
        let var_99 = self.calculate_var(0.99)?;

        // Calculate Expected Shortfall
        let es_95 = self.calculate_expected_shortfall(0.95)?;
        let es_99 = self.calculate_expected_shortfall(0.99)?;

        // Calculate maximum drawdown
        let max_drawdown = self.calculate_maximum_drawdown(positions)?;

        // Calculate Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe_ratio(positions)?;

        // Calculate Sortino ratio
        let sortino_ratio = self.calculate_sortino_ratio(positions)?;

        // Calculate Beta and Alpha
        let (beta, alpha) = self.calculate_beta_alpha(positions)?;

        // Calculate volatility
        let volatility = self.calculate_volatility()?;

        // Calculate skewness and kurtosis
        let (skewness, kurtosis) = self.calculate_higher_moments()?;

        // Calculate correlation matrix
        let correlation_matrix = self.calculate_correlation_matrix(positions)?;

        let metrics = RiskMetrics {
            value_at_risk_95: var_95,
            value_at_risk_99: var_99,
            expected_shortfall_95: es_95,
            expected_shortfall_99: es_99,
            maximum_drawdown: max_drawdown,
            sharpe_ratio,
            sortino_ratio,
            beta,
            alpha,
            volatility,
            skewness,
            kurtosis,
            correlation_matrix,
        };

        // Update cache
        self.metrics_cache = RiskMetricsCache {
            var_95: Some(var_95),
            var_99: Some(var_99),
            expected_shortfall: Some(es_95),
            max_drawdown: Some(max_drawdown),
            sharpe_ratio: Some(sharpe_ratio),
            last_updated: Some(Utc::now()),
        };

        Ok(metrics)
    }

    fn calculate_var(&self, confidence_level: f64) -> Result<f64> {
        if self.historical_data.len() < 30 {
            return Ok(0.0);
        }

        // Calculate returns
        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var = -sorted_returns.get(percentile_index).unwrap_or(&0.0);

        Ok(var)
    }

    fn calculate_expected_shortfall(&self, confidence_level: f64) -> Result<f64> {
        if self.historical_data.len() < 30 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let tail_returns: Vec<f64> = sorted_returns.iter().take(percentile_index).cloned().collect();

        if tail_returns.is_empty() {
            return Ok(0.0);
        }

        let es = -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        Ok(es)
    }

    fn calculate_maximum_drawdown(&self, _positions: &[Position]) -> Result<f64> {
        if self.historical_data.len() < 2 {
            return Ok(0.0);
        }

        let mut peak = self.historical_data[0].mid.to_f64().unwrap_or(0.0);
        let mut max_drawdown: f64 = 0.0;

        for data in &self.historical_data[1..] {
            let current_price = data.mid.to_f64().unwrap_or(0.0);
            
            if current_price > peak {
                peak = current_price;
            } else {
                let drawdown = (peak - current_price) / peak;
                max_drawdown = max_drawdown.max(drawdown);
            }
        }

        Ok(max_drawdown)
    }

    fn calculate_sharpe_ratio(&self, _positions: &[Position]) -> Result<f64> {
        if self.historical_data.len() < 30 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            Ok((mean_return - self.config.risk_free_rate / 252.0) / std_dev * (252.0_f64).sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn calculate_sortino_ratio(&self, _positions: &[Position]) -> Result<f64> {
        if self.historical_data.len() < 30 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        
        if negative_returns.is_empty() {
            return Ok(f64::INFINITY);
        }

        let downside_variance = negative_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / negative_returns.len() as f64;
        let downside_deviation = downside_variance.sqrt();

        if downside_deviation > 0.0 {
            Ok((mean_return - self.config.risk_free_rate / 252.0) / downside_deviation * (252.0_f64).sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn calculate_beta_alpha(&self, _positions: &[Position]) -> Result<(f64, f64)> {
        // Simplified calculation - would need market benchmark data
        Ok((1.0, 0.0))
    }

    fn calculate_volatility(&self) -> Result<f64> {
        if self.historical_data.len() < 30 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt() * (252.0_f64).sqrt()) // Annualized volatility
    }

    fn calculate_higher_moments(&self) -> Result<(f64, f64)> {
        if self.historical_data.len() < 30 {
            return Ok((0.0, 0.0));
        }

        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|pair| {
                let prev_price = pair[0].mid.to_f64().unwrap_or(0.0);
                let curr_price = pair[1].mid.to_f64().unwrap_or(0.0);
                if prev_price > 0.0 {
                    (curr_price - prev_price) / prev_price
                } else {
                    0.0
                }
            })
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok((0.0, 0.0));
        }

        // Skewness
        let skewness = returns.iter()
            .map(|r| ((r - mean_return) / std_dev).powi(3))
            .sum::<f64>() / returns.len() as f64;

        // Kurtosis
        let kurtosis = returns.iter()
            .map(|r| ((r - mean_return) / std_dev).powi(4))
            .sum::<f64>() / returns.len() as f64 - 3.0; // Excess kurtosis

        Ok((skewness, kurtosis))
    }

    fn calculate_correlation_matrix(&self, positions: &[Position]) -> Result<HashMap<String, HashMap<String, f64>>> {
        let mut correlation_matrix = HashMap::new();
        
        // Simplified - would need individual asset price data
        for position in positions {
            let mut correlations = HashMap::new();
            for other_position in positions {
                let correlation = if position.symbol == other_position.symbol {
                    1.0
                } else {
                    0.3 // Simplified correlation
                };
                correlations.insert(other_position.symbol.clone(), correlation);
            }
            correlation_matrix.insert(position.symbol.clone(), correlations);
        }

        Ok(correlation_matrix)
    }

    /// Get cached risk metrics
    pub fn get_cached_metrics(&self) -> Option<&RiskMetricsCache> {
        if self.metrics_cache.last_updated.is_some() {
            Some(&self.metrics_cache)
        } else {
            None
        }
    }
}