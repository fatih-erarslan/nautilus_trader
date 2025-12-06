//! Performance metrics and monitoring for hedge algorithms

use std::collections::HashMap;
// These types are defined in lib.rs
use serde::{Deserialize, Serialize};

/// Advanced performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMetrics {
    /// Information ratio
    pub information_ratio: Option<f64>,
    /// Tracking error
    pub tracking_error: Option<f64>,
    /// Treynor ratio
    pub treynor_ratio: Option<f64>,
    /// Jensen's alpha
    pub jensen_alpha: Option<f64>,
    /// Calmar ratio
    pub calmar_ratio: Option<f64>,
    /// Sterling ratio
    pub sterling_ratio: Option<f64>,
    /// Burke ratio
    pub burke_ratio: Option<f64>,
    /// Ulcer index
    pub ulcer_index: Option<f64>,
    /// Pain index
    pub pain_index: Option<f64>,
    /// Gain-to-pain ratio
    pub gain_to_pain_ratio: Option<f64>,
    /// Lake ratio
    pub lake_ratio: Option<f64>,
    /// Mountain ratio
    pub mountain_ratio: Option<f64>,
}

impl AdvancedMetrics {
    /// Create new advanced metrics
    pub fn new() -> Self {
        Self {
            information_ratio: None,
            tracking_error: None,
            treynor_ratio: None,
            jensen_alpha: None,
            calmar_ratio: None,
            sterling_ratio: None,
            burke_ratio: None,
            ulcer_index: None,
            pain_index: None,
            gain_to_pain_ratio: None,
            lake_ratio: None,
            mountain_ratio: None,
        }
    }
    
    /// Update advanced metrics
    pub fn update(&mut self, returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> Result<(), crate::HedgeError> {
        if returns.len() < 2 {
            return Ok(());
        }
        
        // Calculate information ratio
        if !benchmark_returns.is_empty() && benchmark_returns.len() == returns.len() {
            self.information_ratio = Some(self.calculate_information_ratio(returns, benchmark_returns)?);
            self.tracking_error = Some(self.calculate_tracking_error(returns, benchmark_returns)?);
        }
        
        // Calculate Treynor ratio
        if !benchmark_returns.is_empty() && benchmark_returns.len() == returns.len() {
            self.treynor_ratio = Some(self.calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate)?);
        }
        
        // Calculate Jensen's alpha
        if !benchmark_returns.is_empty() && benchmark_returns.len() == returns.len() {
            self.jensen_alpha = Some(self.calculate_jensen_alpha(returns, benchmark_returns, risk_free_rate)?);
        }
        
        // Calculate Calmar ratio
        self.calmar_ratio = Some(self.calculate_calmar_ratio(returns)?);
        
        // Calculate Sterling ratio
        self.sterling_ratio = Some(self.calculate_sterling_ratio(returns)?);
        
        // Calculate Burke ratio
        self.burke_ratio = Some(self.calculate_burke_ratio(returns)?);
        
        // Calculate Ulcer index
        self.ulcer_index = Some(self.calculate_ulcer_index(returns)?);
        
        // Calculate Pain index
        self.pain_index = Some(self.calculate_pain_index(returns)?);
        
        // Calculate Gain-to-pain ratio
        self.gain_to_pain_ratio = Some(self.calculate_gain_to_pain_ratio(returns)?);
        
        // Calculate Lake ratio
        self.lake_ratio = Some(self.calculate_lake_ratio(returns)?);
        
        // Calculate Mountain ratio
        self.mountain_ratio = Some(self.calculate_mountain_ratio(returns)?);
        
        Ok(())
    }
    
    /// Calculate information ratio
    fn calculate_information_ratio(&self, returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let excess_returns: Vec<f64> = returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(r, b)| r - b)
            .collect();
        
        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let tracking_error = self.calculate_tracking_error(returns, benchmark_returns)?;
        
        if tracking_error > 0.0 {
            Ok(mean_excess / tracking_error)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate tracking error
    fn calculate_tracking_error(&self, returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let excess_returns: Vec<f64> = returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(r, b)| r - b)
            .collect();
        
        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let variance = excess_returns.iter()
            .map(|r| (r - mean_excess).powi(2))
            .sum::<f64>() / excess_returns.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Calculate Treynor ratio
    fn calculate_treynor_ratio(&self, returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let beta = self.calculate_beta(returns, benchmark_returns)?;
        
        if beta > 0.0 {
            Ok((mean_return - risk_free_rate) / beta)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Jensen's alpha
    fn calculate_jensen_alpha(&self, returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;
        let beta = self.calculate_beta(returns, benchmark_returns)?;
        
        let expected_return = risk_free_rate + beta * (mean_benchmark - risk_free_rate);
        Ok(mean_return - expected_return)
    }
    
    /// Calculate beta
    fn calculate_beta(&self, returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;
        
        let covariance = returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(r, b)| (r - mean_return) * (b - mean_benchmark))
            .sum::<f64>() / returns.len() as f64;
        
        let benchmark_variance = benchmark_returns.iter()
            .map(|b| (b - mean_benchmark).powi(2))
            .sum::<f64>() / benchmark_returns.len() as f64;
        
        if benchmark_variance > 0.0 {
            Ok(covariance / benchmark_variance)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Calmar ratio
    fn calculate_calmar_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let max_drawdown = self.calculate_max_drawdown(returns)?;
        
        if max_drawdown > 0.0 {
            Ok(mean_return / max_drawdown)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Sterling ratio
    fn calculate_sterling_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let avg_drawdown = self.calculate_average_drawdown(returns)?;
        
        if avg_drawdown > 0.0 {
            Ok(mean_return / avg_drawdown)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Burke ratio
    fn calculate_burke_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let drawdown_deviation = self.calculate_drawdown_deviation(returns)?;
        
        if drawdown_deviation > 0.0 {
            Ok(mean_return / drawdown_deviation)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Ulcer index
    fn calculate_ulcer_index(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mut cumulative_return = 0.0;
        let mut peak = 0.0;
        let mut ulcer_sum = 0.0;
        
        for &return_val in returns {
            cumulative_return += return_val;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = (peak - cumulative_return) / (1.0 + peak);
            ulcer_sum += drawdown * drawdown;
        }
        
        Ok((ulcer_sum / returns.len() as f64).sqrt())
    }
    
    /// Calculate Pain index
    fn calculate_pain_index(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mut cumulative_return = 0.0;
        let mut peak = 0.0;
        let mut pain_sum = 0.0;
        
        for &return_val in returns {
            cumulative_return += return_val;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = peak - cumulative_return;
            pain_sum += drawdown.abs();
        }
        
        Ok(pain_sum / returns.len() as f64)
    }
    
    /// Calculate Gain-to-pain ratio
    fn calculate_gain_to_pain_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let total_gain = returns.iter().filter(|&&r| r > 0.0).sum::<f64>();
        let total_pain = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum::<f64>();
        
        if total_pain > 0.0 {
            Ok(total_gain / total_pain)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Lake ratio
    fn calculate_lake_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let ulcer_index = self.calculate_ulcer_index(returns)?;
        
        if ulcer_index > 0.0 {
            Ok(mean_return / ulcer_index)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Mountain ratio
    fn calculate_mountain_ratio(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let pain_index = self.calculate_pain_index(returns)?;
        
        if pain_index > 0.0 {
            Ok(mean_return / pain_index)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mut cumulative_return = 0.0;
        let mut peak = 0.0;
        let mut max_drawdown = 0.0;
        
        for &return_val in returns {
            cumulative_return += return_val;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = peak - cumulative_return;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate average drawdown
    fn calculate_average_drawdown(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mut cumulative_return = 0.0;
        let mut peak = 0.0;
        let mut drawdown_sum = 0.0;
        let mut drawdown_count = 0;
        
        for &return_val in returns {
            cumulative_return += return_val;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = peak - cumulative_return;
            if drawdown > 0.0 {
                drawdown_sum += drawdown;
                drawdown_count += 1;
            }
        }
        
        if drawdown_count > 0 {
            Ok(drawdown_sum / drawdown_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate drawdown deviation
    fn calculate_drawdown_deviation(&self, returns: &[f64]) -> Result<f64, crate::HedgeError> {
        let mut cumulative_return = 0.0;
        let mut peak = 0.0;
        let mut drawdowns = Vec::new();
        
        for &return_val in returns {
            cumulative_return += return_val;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = peak - cumulative_return;
            drawdowns.push(drawdown);
        }
        
        let mean_drawdown = drawdowns.iter().sum::<f64>() / drawdowns.len() as f64;
        let variance = drawdowns.iter()
            .map(|d| (d - mean_drawdown).powi(2))
            .sum::<f64>() / drawdowns.len() as f64;
        
        Ok(variance.sqrt())
    }
}

impl Default for AdvancedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAttribution {
    /// Factor contributions
    pub factor_contributions: HashMap<String, f64>,
    /// Expert contributions
    pub expert_contributions: HashMap<String, f64>,
    /// Strategy contributions
    pub strategy_contributions: HashMap<String, f64>,
    /// Total attribution
    pub total_attribution: f64,
}

impl PerformanceAttribution {
    /// Create new performance attribution
    pub fn new() -> Self {
        Self {
            factor_contributions: HashMap::new(),
            expert_contributions: HashMap::new(),
            strategy_contributions: HashMap::new(),
            total_attribution: 0.0,
        }
    }
    
    /// Add factor contribution
    pub fn add_factor_contribution(&mut self, factor: String, contribution: f64) {
        self.factor_contributions.insert(factor, contribution);
        self.update_total_attribution();
    }
    
    /// Add expert contribution
    pub fn add_expert_contribution(&mut self, expert: String, contribution: f64) {
        self.expert_contributions.insert(expert, contribution);
        self.update_total_attribution();
    }
    
    /// Add strategy contribution
    pub fn add_strategy_contribution(&mut self, strategy: String, contribution: f64) {
        self.strategy_contributions.insert(strategy, contribution);
        self.update_total_attribution();
    }
    
    /// Update total attribution
    fn update_total_attribution(&mut self) {
        self.total_attribution = self.factor_contributions.values().sum::<f64>()
            + self.expert_contributions.values().sum::<f64>()
            + self.strategy_contributions.values().sum::<f64>();
    }
    
    /// Get top contributors
    pub fn get_top_contributors(&self, n: usize) -> Vec<(String, f64)> {
        let mut all_contributions = Vec::new();
        
        for (name, contribution) in &self.factor_contributions {
            all_contributions.push((format!("Factor: {}", name), *contribution));
        }
        
        for (name, contribution) in &self.expert_contributions {
            all_contributions.push((format!("Expert: {}", name), *contribution));
        }
        
        for (name, contribution) in &self.strategy_contributions {
            all_contributions.push((format!("Strategy: {}", name), *contribution));
        }
        
        all_contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_contributions.into_iter().take(n).collect()
    }
}

impl Default for PerformanceAttribution {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk-adjusted metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAdjustedMetrics {
    /// Sharpe ratio
    pub sharpe_ratio: Option<f64>,
    /// Sortino ratio
    pub sortino_ratio: Option<f64>,
    /// Omega ratio
    pub omega_ratio: Option<f64>,
    /// Kappa ratio
    pub kappa_ratio: Option<f64>,
    /// Upside potential ratio
    pub upside_potential_ratio: Option<f64>,
    /// Downside risk
    pub downside_risk: Option<f64>,
    /// Upside risk
    pub upside_risk: Option<f64>,
}

impl RiskAdjustedMetrics {
    /// Create new risk-adjusted metrics
    pub fn new() -> Self {
        Self {
            sharpe_ratio: None,
            sortino_ratio: None,
            omega_ratio: None,
            kappa_ratio: None,
            upside_potential_ratio: None,
            downside_risk: None,
            upside_risk: None,
        }
    }
    
    /// Update risk-adjusted metrics
    pub fn update(&mut self, returns: &[f64], risk_free_rate: f64, threshold: f64) -> Result<(), crate::HedgeError> {
        if returns.len() < 2 {
            return Ok(());
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - risk_free_rate;
        
        // Calculate standard deviation
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        // Sharpe ratio
        if std_dev > 0.0 {
            self.sharpe_ratio = Some(excess_return / std_dev);
        }
        
        // Downside risk
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < threshold)
            .copied()
            .collect();
        
        if !downside_returns.is_empty() {
            let downside_variance = downside_returns.iter()
                .map(|r| (r - threshold).powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            self.downside_risk = Some(downside_variance.sqrt());
            
            // Sortino ratio
            if self.downside_risk.unwrap() > 0.0 {
                self.sortino_ratio = Some(excess_return / self.downside_risk.unwrap());
            }
        }
        
        // Upside risk
        let upside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r > threshold)
            .copied()
            .collect();
        
        if !upside_returns.is_empty() {
            let upside_variance = upside_returns.iter()
                .map(|r| (r - threshold).powi(2))
                .sum::<f64>() / upside_returns.len() as f64;
            self.upside_risk = Some(upside_variance.sqrt());
            
            // Upside potential ratio
            if let Some(downside_risk) = self.downside_risk {
                if downside_risk > 0.0 {
                    let upside_potential = upside_returns.iter()
                        .map(|r| (r - threshold).max(0.0))
                        .sum::<f64>() / upside_returns.len() as f64;
                    self.upside_potential_ratio = Some(upside_potential / downside_risk);
                }
            }
        }
        
        // Omega ratio
        let gains: f64 = returns.iter()
            .map(|r| (r - threshold).max(0.0))
            .sum();
        let losses: f64 = returns.iter()
            .map(|r| (threshold - r).max(0.0))
            .sum();
        
        if losses > 0.0 {
            self.omega_ratio = Some(gains / losses);
        }
        
        // Kappa ratio (simplified)
        if let Some(downside_risk) = self.downside_risk {
            if downside_risk > 0.0 {
                self.kappa_ratio = Some(excess_return / downside_risk.powi(3));
            }
        }
        
        Ok(())
    }
}

impl Default for RiskAdjustedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_metrics_creation() {
        let metrics = AdvancedMetrics::new();
        
        assert!(metrics.information_ratio.is_none());
        assert!(metrics.calmar_ratio.is_none());
        assert!(metrics.ulcer_index.is_none());
    }
    
    #[test]
    fn test_performance_attribution() {
        let mut attribution = PerformanceAttribution::new();
        
        attribution.add_factor_contribution("momentum".to_string(), 0.05);
        attribution.add_expert_contribution("expert1".to_string(), 0.03);
        attribution.add_strategy_contribution("pairs".to_string(), 0.02);
        
        assert_eq!(attribution.total_attribution, 0.10);
        
        let top_contributors = attribution.get_top_contributors(2);
        assert_eq!(top_contributors.len(), 2);
        assert_eq!(top_contributors[0].0, "Factor: momentum");
        assert_eq!(top_contributors[0].1, 0.05);
    }
    
    #[test]
    fn test_risk_adjusted_metrics() {
        let mut metrics = RiskAdjustedMetrics::new();
        let returns = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.01];
        
        metrics.update(&returns, 0.005, 0.0).unwrap();
        
        assert!(metrics.sharpe_ratio.is_some());
        assert!(metrics.sortino_ratio.is_some());
        assert!(metrics.omega_ratio.is_some());
    }
}