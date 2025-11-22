use crate::prelude::*;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Comprehensive portfolio analytics system
#[derive(Debug, Clone)]
pub struct PortfolioAnalytics {
    returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
    periods_per_year: f64,
    risk_free_rate: f64,
}

impl PortfolioAnalytics {
    pub fn new(risk_free_rate: f64, periods_per_year: f64) -> Self {
        Self {
            returns: Vec::new(),
            benchmark_returns: Vec::new(),
            periods_per_year,
            risk_free_rate,
        }
    }
    
    pub fn add_return(&mut self, portfolio_return: f64, benchmark_return: Option<f64>) {
        self.returns.push(portfolio_return);
        if let Some(bench_return) = benchmark_return {
            self.benchmark_returns.push(bench_return);
        }
    }
    
    pub fn calculate_sharpe_ratio(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance = self.returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / self.returns.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let annualized_return = mean_return * self.periods_per_year;
        let annualized_vol = std_dev * self.periods_per_year.sqrt();
        
        (annualized_return - self.risk_free_rate) / annualized_vol
    }
    
    pub fn calculate_sortino_ratio(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let downside_variance = self.returns.iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / self.returns.len() as f64;
        let downside_deviation = downside_variance.sqrt();
        
        if downside_deviation == 0.0 {
            return 0.0;
        }
        
        let annualized_return = mean_return * self.periods_per_year;
        let annualized_downside_dev = downside_deviation * self.periods_per_year.sqrt();
        
        (annualized_return - self.risk_free_rate) / annualized_downside_dev
    }
    
    pub fn calculate_max_drawdown(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        
        for &ret in &self.returns {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let drawdown = (peak - cumulative) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
    
    pub fn calculate_alpha_beta(&self) -> (f64, f64) {
        if self.returns.len() != self.benchmark_returns.len() || self.returns.is_empty() {
            return (0.0, 1.0);
        }
        
        let n = self.returns.len() as f64;
        let portfolio_mean = self.returns.iter().sum::<f64>() / n;
        let benchmark_mean = self.benchmark_returns.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..self.returns.len() {
            let portfolio_diff = self.returns[i] - portfolio_mean;
            let benchmark_diff = self.benchmark_returns[i] - benchmark_mean;
            
            numerator += portfolio_diff * benchmark_diff;
            denominator += benchmark_diff * benchmark_diff;
        }
        
        let beta = if denominator != 0.0 { numerator / denominator } else { 1.0 };
        let alpha = (portfolio_mean - beta * benchmark_mean) * self.periods_per_year;
        
        (alpha, beta)
    }
    
    pub fn calculate_information_ratio(&self) -> f64 {
        if self.returns.len() != self.benchmark_returns.len() || self.returns.is_empty() {
            return 0.0;
        }
        
        let excess_returns: Vec<f64> = self.returns.iter()
            .zip(self.benchmark_returns.iter())
            .map(|(p, b)| p - b)
            .collect();
        
        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let tracking_error = {
            let variance = excess_returns.iter()
                .map(|r| (r - mean_excess).powi(2))
                .sum::<f64>() / excess_returns.len() as f64;
            variance.sqrt()
        };
        
        if tracking_error == 0.0 {
            return 0.0;
        }
        
        mean_excess * self.periods_per_year.sqrt() / tracking_error
    }
    
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let sharpe = self.calculate_sharpe_ratio();
        let sortino = self.calculate_sortino_ratio();
        let max_dd = self.calculate_max_drawdown();
        let (alpha, beta) = self.calculate_alpha_beta();
        let info_ratio = self.calculate_information_ratio();
        
        let total_return = self.returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
        let annualized_return = if self.returns.is_empty() { 0.0 } else {
            let periods = self.returns.len() as f64;
            ((1.0 + total_return).powf(self.periods_per_year / periods) - 1.0)
        };
        
        PerformanceReport {
            total_return,
            annualized_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            maximum_drawdown: max_dd,
            alpha,
            beta,
            information_ratio: info_ratio,
            volatility: if self.returns.is_empty() { 0.0 } else {
                let mean = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
                let variance = self.returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / self.returns.len() as f64;
                variance.sqrt() * self.periods_per_year.sqrt()
            },
            win_rate: {
                let wins = self.returns.iter().filter(|&&r| r > 0.0).count();
                if self.returns.is_empty() { 0.0 } else { wins as f64 / self.returns.len() as f64 }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub maximum_drawdown: f64,
    pub alpha: f64,
    pub beta: f64,
    pub information_ratio: f64,
    pub volatility: f64,
    pub win_rate: f64,
}