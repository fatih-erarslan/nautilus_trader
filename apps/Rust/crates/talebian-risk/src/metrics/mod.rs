//! Metrics and performance measurement

use crate::error::{TalebianResult as Result, TalebianError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk-adjusted performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAdjustedMetrics {
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Information ratio
    pub information_ratio: f64,
    /// Treynor ratio
    pub treynor_ratio: f64,
    /// Jensen's alpha
    pub alpha: f64,
    /// Beta
    pub beta: f64,
    /// Tracking error
    pub tracking_error: f64,
}

/// Drawdown analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Average drawdown
    pub average_drawdown: f64,
    /// Maximum drawdown duration (days)
    pub max_drawdown_duration: usize,
    /// Recovery time (days)
    pub recovery_time: usize,
    /// Drawdown frequency
    pub drawdown_frequency: f64,
    /// Pain index
    pub pain_index: f64,
}

/// Tail risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskMetrics {
    /// Value at Risk (VaR) at different confidence levels
    pub var_99: f64,
    pub var_95: f64,
    pub var_90: f64,
    /// Conditional Value at Risk (CVaR)
    pub cvar_99: f64,
    pub cvar_95: f64,
    pub cvar_90: f64,
    /// Expected Shortfall
    pub expected_shortfall: f64,
    /// Tail ratio
    pub tail_ratio: f64,
    /// Extreme value index
    pub extreme_value_index: f64,
}

/// Performance attribution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionMetrics {
    /// Asset contributions
    pub asset_contributions: HashMap<String, f64>,
    /// Factor contributions
    pub factor_contributions: HashMap<String, f64>,
    /// Interaction effects
    pub interaction_effects: HashMap<String, f64>,
    /// Total return
    pub total_return: f64,
    /// Benchmark return
    pub benchmark_return: f64,
    /// Active return
    pub active_return: f64,
}

/// Metrics calculator
pub struct MetricsCalculator {
    /// Risk-free rate
    risk_free_rate: f64,
    /// Benchmark returns
    benchmark_returns: Vec<f64>,
}

impl MetricsCalculator {
    /// Create a new metrics calculator
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            benchmark_returns: Vec::new(),
        }
    }
    
    /// Set benchmark returns
    pub fn set_benchmark(&mut self, benchmark_returns: Vec<f64>) {
        self.benchmark_returns = benchmark_returns;
    }
    
    /// Calculate risk-adjusted metrics
    pub fn calculate_risk_adjusted_metrics(&self, returns: &[f64]) -> Result<RiskAdjustedMetrics> {
        if returns.len() < 2 {
            return Err(TalebianError::insufficient_data(2, returns.len()));
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - self.risk_free_rate;
        
        // Calculate volatility
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let volatility = variance.sqrt();
        
        // Sharpe ratio
        let sharpe_ratio = if volatility > 0.0 {
            excess_return / volatility
        } else {
            0.0
        };
        
        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < self.risk_free_rate)
            .map(|&r| r - self.risk_free_rate)
            .collect();
        
        let downside_deviation = if !downside_returns.is_empty() {
            let downside_variance = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            downside_variance.sqrt()
        } else {
            0.0
        };
        
        let sortino_ratio = if downside_deviation > 0.0 {
            excess_return / downside_deviation
        } else {
            0.0
        };
        
        // Calculate maximum drawdown for Calmar ratio
        let drawdown_analysis = self.calculate_drawdown_analysis(returns)?;
        let calmar_ratio = if drawdown_analysis.max_drawdown > 0.0 {
            mean_return / drawdown_analysis.max_drawdown
        } else {
            0.0
        };
        
        // Information ratio (vs benchmark)
        let (information_ratio, tracking_error) = if !self.benchmark_returns.is_empty() 
            && self.benchmark_returns.len() == returns.len() {
            let active_returns: Vec<f64> = returns.iter()
                .zip(self.benchmark_returns.iter())
                .map(|(r, b)| r - b)
                .collect();
            
            let mean_active_return = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
            let tracking_error = if active_returns.len() > 1 {
                let variance = active_returns.iter()
                    .map(|r| (r - mean_active_return).powi(2))
                    .sum::<f64>() / (active_returns.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };
            
            let info_ratio = if tracking_error > 0.0 {
                mean_active_return / tracking_error
            } else {
                0.0
            };
            
            (info_ratio, tracking_error)
        } else {
            (0.0, 0.0)
        };
        
        // Beta and Alpha (vs benchmark)
        let (beta, alpha) = if !self.benchmark_returns.is_empty() 
            && self.benchmark_returns.len() == returns.len() {
            let benchmark_mean = self.benchmark_returns.iter().sum::<f64>() / self.benchmark_returns.len() as f64;
            
            let covariance = returns.iter()
                .zip(self.benchmark_returns.iter())
                .map(|(r, b)| (r - mean_return) * (b - benchmark_mean))
                .sum::<f64>() / (returns.len() - 1) as f64;
            
            let benchmark_variance = self.benchmark_returns.iter()
                .map(|b| (b - benchmark_mean).powi(2))
                .sum::<f64>() / (self.benchmark_returns.len() - 1) as f64;
            
            let beta = if benchmark_variance > 0.0 {
                covariance / benchmark_variance
            } else {
                0.0
            };
            
            let alpha = mean_return - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate));
            
            (beta, alpha)
        } else {
            (1.0, 0.0)
        };
        
        // Treynor ratio
        let treynor_ratio = if beta != 0.0 {
            excess_return / beta
        } else {
            0.0
        };
        
        Ok(RiskAdjustedMetrics {
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            information_ratio,
            treynor_ratio,
            alpha,
            beta,
            tracking_error,
        })
    }
    
    /// Calculate drawdown analysis
    pub fn calculate_drawdown_analysis(&self, returns: &[f64]) -> Result<DrawdownAnalysis> {
        if returns.is_empty() {
            return Err(TalebianError::insufficient_data(1, 0));
        }
        
        // Calculate cumulative returns
        let mut cumulative_returns = vec![0.0];
        for &ret in returns {
            let last_cum = cumulative_returns.last().unwrap();
            cumulative_returns.push(last_cum + ret);
        }
        
        // Calculate drawdowns
        let mut drawdowns = Vec::new();
        let mut peak = cumulative_returns[0];
        
        for &cum_ret in &cumulative_returns {
            peak = peak.max(cum_ret);
            let drawdown = (peak - cum_ret) / peak.max(1e-10); // Avoid division by zero
            drawdowns.push(drawdown);
        }
        
        let max_drawdown = drawdowns.iter().fold(0.0f64, |a, &b| a.max(b));
        let average_drawdown = drawdowns.iter().sum::<f64>() / drawdowns.len() as f64;
        
        // Find maximum drawdown duration
        let mut in_drawdown = false;
        let mut current_duration = 0;
        let mut max_duration = 0;
        let mut recovery_time = 0;
        let mut drawdown_count = 0;
        
        for &dd in &drawdowns {
            if dd > 0.001 { // In drawdown (0.1% threshold)
                if !in_drawdown {
                    in_drawdown = true;
                    current_duration = 1;
                    drawdown_count += 1;
                } else {
                    current_duration += 1;
                }
                max_duration = max_duration.max(current_duration);
            } else {
                if in_drawdown {
                    recovery_time += current_duration;
                    in_drawdown = false;
                    current_duration = 0;
                }
            }
        }
        
        let drawdown_frequency = if returns.len() > 0 {
            drawdown_count as f64 / returns.len() as f64
        } else {
            0.0
        };
        
        // Pain index (average of all drawdowns)
        let pain_index = drawdowns.iter().sum::<f64>() / drawdowns.len() as f64;
        
        Ok(DrawdownAnalysis {
            max_drawdown,
            average_drawdown,
            max_drawdown_duration: max_duration,
            recovery_time: if drawdown_count > 0 { recovery_time / drawdown_count } else { 0 },
            drawdown_frequency,
            pain_index,
        })
    }
    
    /// Calculate tail risk metrics
    pub fn calculate_tail_risk_metrics(&self, returns: &[f64]) -> Result<TailRiskMetrics> {
        if returns.len() < 10 {
            return Err(TalebianError::insufficient_data(10, returns.len()));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_returns.len();
        
        // Calculate VaR at different confidence levels
        let var_99 = -sorted_returns[(0.01 * n as f64) as usize];
        let var_95 = -sorted_returns[(0.05 * n as f64) as usize];
        let var_90 = -sorted_returns[(0.10 * n as f64) as usize];
        
        // Calculate CVaR (average of returns beyond VaR)
        let cvar_99_idx = (0.01 * n as f64) as usize;
        let cvar_95_idx = (0.05 * n as f64) as usize;
        let cvar_90_idx = (0.10 * n as f64) as usize;
        
        let cvar_99 = if cvar_99_idx > 0 {
            -sorted_returns[0..cvar_99_idx].iter().sum::<f64>() / cvar_99_idx as f64
        } else {
            var_99
        };
        
        let cvar_95 = if cvar_95_idx > 0 {
            -sorted_returns[0..cvar_95_idx].iter().sum::<f64>() / cvar_95_idx as f64
        } else {
            var_95
        };
        
        let cvar_90 = if cvar_90_idx > 0 {
            -sorted_returns[0..cvar_90_idx].iter().sum::<f64>() / cvar_90_idx as f64
        } else {
            var_90
        };
        
        // Expected shortfall (same as CVaR_95 in this implementation)
        let expected_shortfall = cvar_95;
        
        // Tail ratio (average of top 10% / average of bottom 10%)
        let top_10_idx = (0.9 * n as f64) as usize;
        let bottom_10_idx = (0.1 * n as f64) as usize;
        
        let top_10_avg = sorted_returns[top_10_idx..].iter().sum::<f64>() / (n - top_10_idx) as f64;
        let bottom_10_avg = sorted_returns[0..bottom_10_idx].iter().sum::<f64>() / bottom_10_idx as f64;
        
        let tail_ratio = if bottom_10_avg != 0.0 {
            -top_10_avg / bottom_10_avg
        } else {
            0.0
        };
        
        // Extreme value index (simplified Hill estimator)
        let k = (n as f64 * 0.1) as usize; // Use top 10% for estimation
        let extreme_value_index = if k > 1 {
            let mut sum = 0.0;
            for i in 0..k {
                let ratio = sorted_returns[n - 1 - i] / sorted_returns[n - k];
                if ratio > 0.0 {
                    sum += ratio.ln();
                }
            }
            sum / k as f64
        } else {
            0.0
        };
        
        Ok(TailRiskMetrics {
            var_99,
            var_95,
            var_90,
            cvar_99,
            cvar_95,
            cvar_90,
            expected_shortfall,
            tail_ratio,
            extreme_value_index,
        })
    }
    
    /// Calculate performance attribution
    pub fn calculate_attribution(&self, 
                                 portfolio_returns: &[f64],
                                 asset_returns: &HashMap<String, Vec<f64>>,
                                 weights: &HashMap<String, f64>) -> Result<AttributionMetrics> {
        
        if portfolio_returns.is_empty() {
            return Err(TalebianError::insufficient_data(1, 0));
        }
        
        let total_return = portfolio_returns.iter().sum::<f64>();
        
        let benchmark_return = if !self.benchmark_returns.is_empty() {
            self.benchmark_returns.iter().sum::<f64>()
        } else {
            0.0
        };
        
        let active_return = total_return - benchmark_return;
        
        // Calculate asset contributions
        let mut asset_contributions = HashMap::new();
        for (asset, returns) in asset_returns {
            if let Some(&weight) = weights.get(asset) {
                let asset_return = returns.iter().sum::<f64>();
                let contribution = weight * asset_return;
                asset_contributions.insert(asset.clone(), contribution);
            }
        }
        
        // Simplified factor contributions (would be more complex in practice)
        let mut factor_contributions = HashMap::new();
        factor_contributions.insert("Market".to_string(), total_return * 0.7);
        factor_contributions.insert("Size".to_string(), total_return * 0.1);
        factor_contributions.insert("Value".to_string(), total_return * 0.1);
        factor_contributions.insert("Momentum".to_string(), total_return * 0.1);
        
        // Interaction effects (simplified)
        let mut interaction_effects = HashMap::new();
        interaction_effects.insert("Market-Size".to_string(), total_return * 0.05);
        
        Ok(AttributionMetrics {
            asset_contributions,
            factor_contributions,
            interaction_effects,
            total_return,
            benchmark_return,
            active_return,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_calculator() {
        let calculator = MetricsCalculator::new(0.02); // 2% risk-free rate
        
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.0, 0.015, -0.005, 0.02];
        
        let risk_metrics = calculator.calculate_risk_adjusted_metrics(&returns).unwrap();
        assert!(risk_metrics.sharpe_ratio.is_finite());
        assert!(risk_metrics.sortino_ratio.is_finite());
        
        let drawdown = calculator.calculate_drawdown_analysis(&returns).unwrap();
        assert!(drawdown.max_drawdown >= 0.0);
        
        let tail_metrics = calculator.calculate_tail_risk_metrics(&returns).unwrap();
        assert!(tail_metrics.var_95 >= 0.0);
        assert!(tail_metrics.cvar_95 >= tail_metrics.var_95);
    }
    
    #[test]
    fn test_with_benchmark() {
        let mut calculator = MetricsCalculator::new(0.02);
        
        let portfolio_returns = vec![0.01, 0.02, -0.01, 0.03];
        let benchmark_returns = vec![0.005, 0.015, -0.005, 0.025];
        
        calculator.set_benchmark(benchmark_returns);
        
        let metrics = calculator.calculate_risk_adjusted_metrics(&portfolio_returns).unwrap();
        assert!(metrics.information_ratio.is_finite());
        assert!(metrics.tracking_error >= 0.0);
        assert!(metrics.beta.is_finite());
    }
    
    #[test]
    fn test_attribution() {
        let calculator = MetricsCalculator::new(0.02);
        
        let portfolio_returns = vec![0.01, 0.02, -0.01, 0.03];
        
        let mut asset_returns = HashMap::new();
        asset_returns.insert("STOCK_A".to_string(), vec![0.015, 0.025, -0.005, 0.035]);
        asset_returns.insert("STOCK_B".to_string(), vec![0.005, 0.015, -0.015, 0.025]);
        
        let mut weights = HashMap::new();
        weights.insert("STOCK_A".to_string(), 0.6);
        weights.insert("STOCK_B".to_string(), 0.4);
        
        let attribution = calculator.calculate_attribution(&portfolio_returns, &asset_returns, &weights).unwrap();
        
        assert_eq!(attribution.asset_contributions.len(), 2);
        assert!(attribution.total_return.is_finite());
    }
}