//! Core Talebian risk management implementation
//! 
//! This module provides the foundational structures and algorithms for
//! Nassim Nicholas Taleb's risk management and antifragility concepts.

pub mod market_data;
pub mod portfolio;
pub mod position;
pub mod risk_metrics;
pub mod time_series;

// Re-export common types
pub use crate::barbell::AssetType;
pub use crate::error::{TalebianError, TalebianResult};

use crate::{AntifragilityMeasurement, BlackSwanEvent, PerformanceTracker, ReturnData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Base trait for Talebian risk components
pub trait TalebianRiskComponent {
    /// Component identifier
    fn id(&self) -> &str;
    
    /// Component name
    fn name(&self) -> &str;
    
    /// Validate component state
    fn validate(&self) -> TalebianResult<()>;
    
    /// Reset component to initial state
    fn reset(&mut self);
    
    /// Get component metadata
    fn metadata(&self) -> HashMap<String, String>;
}

/// Core Talebian risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Portfolio identifier
    pub portfolio_id: String,
    /// Risk tolerance level (0.0 to 1.0)
    pub risk_tolerance: f64,
    /// Maximum allowed drawdown
    pub max_drawdown: f64,
    /// Tail risk threshold
    pub tail_risk_threshold: f64,
    /// Maximum position size
    pub max_position_size: f64,
    /// Enable antifragility measurement
    pub enable_antifragility: bool,
    /// Enable black swan detection
    pub enable_black_swan_detection: bool,
    /// Enable barbell strategy
    pub enable_barbell_strategy: bool,
    /// Volatility targeting enabled
    pub volatility_targeting: bool,
    /// Target volatility level
    pub target_volatility: f64,
    /// Rebalancing frequency in seconds
    pub rebalancing_frequency: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            portfolio_id: "default_portfolio".to_string(),
            risk_tolerance: 0.05,
            max_drawdown: 0.15,
            tail_risk_threshold: 0.01,
            max_position_size: 0.2,
            enable_antifragility: true,
            enable_black_swan_detection: true,
            enable_barbell_strategy: true,
            volatility_targeting: true,
            target_volatility: 0.12,
            rebalancing_frequency: 3600, // 1 hour
        }
    }
}

/// Main Talebian risk management system
#[derive(Debug, Clone)]
pub struct TalebianRisk {
    config: RiskConfig,
    portfolio_value_history: Vec<f64>,
    return_history: Vec<f64>,
    volatility_history: Vec<f64>,
    drawdown_history: Vec<f64>,
    last_update: Option<DateTime<Utc>>,
    performance_tracker: PerformanceTracker,
    risk_metrics_cache: Option<RiskMetrics>,
}

/// Comprehensive risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Portfolio value at risk (95% confidence)
    pub var_95: f64,
    /// Conditional value at risk (95% confidence)
    pub cvar_95: f64,
    /// Maximum historical drawdown
    pub max_drawdown: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Portfolio volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk adjusted)
    pub sortino_ratio: f64,
    /// Calmar ratio (return/max drawdown)
    pub calmar_ratio: f64,
    /// Tail ratio (95th percentile / 5th percentile)
    pub tail_ratio: f64,
    /// Skewness of returns
    pub skewness: f64,
    /// Kurtosis of returns
    pub kurtosis: f64,
    /// Antifragility score
    pub antifragility_score: f64,
    /// Black swan probability
    pub black_swan_probability: f64,
    /// Last calculation timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var_95: 0.0,
            cvar_95: 0.0,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            tail_ratio: 1.0,
            skewness: 0.0,
            kurtosis: 0.0,
            antifragility_score: 0.0,
            black_swan_probability: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level (0.0 to 1.0)
    pub risk_level: f64,
    /// Risk category
    pub risk_category: RiskCategory,
    /// Detailed risk metrics
    pub metrics: RiskMetrics,
    /// Risk warnings
    pub warnings: Vec<RiskWarning>,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Assessment timestamp
    pub timestamp: DateTime<Utc>,
}

/// Risk category classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCategory {
    Low,
    Moderate,
    High,
    Extreme,
    BlackSwan,
}

/// Risk warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskWarning {
    HighVolatility { current: f64, threshold: f64 },
    ExcessiveDrawdown { current: f64, max_allowed: f64 },
    TailRiskElevated { probability: f64 },
    BlackSwanDetected { confidence: f64 },
    VolatilityTargetMissed { current: f64, target: f64 },
    ConcentrationRisk { sector: String, allocation: f64 },
    LiquidityRisk { estimated_exit_time: f64 },
}

impl TalebianRisk {
    /// Create a new Talebian risk management system
    pub fn new(config: RiskConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            config,
            portfolio_value_history: Vec::new(),
            return_history: Vec::new(),
            volatility_history: Vec::new(),
            drawdown_history: Vec::new(),
            last_update: None,
            performance_tracker: PerformanceTracker::default(),
            risk_metrics_cache: None,
        })
    }

    /// Update system with new market data
    pub fn update_market_data(&mut self, return_data: &ReturnData) -> Result<(), Box<dyn std::error::Error>> {
        // Update return history
        let current_return = return_data.expected_return;
        self.return_history.push(current_return);
        
        // Update volatility history
        self.volatility_history.push(return_data.volatility);
        
        // Calculate portfolio value (simplified)
        let current_value = if self.portfolio_value_history.is_empty() {
            100.0 // Starting value
        } else {
            let last_value = self.portfolio_value_history.last().unwrap();
            last_value * (1.0 + current_return)
        };
        self.portfolio_value_history.push(current_value);
        
        // Update drawdown
        let peak_value = self.portfolio_value_history.iter()
            .fold(0.0f64, |acc, &val| acc.max(val));
        let current_drawdown = if peak_value > 0.0 {
            (peak_value - current_value) / peak_value
        } else {
            0.0
        };
        self.drawdown_history.push(current_drawdown);
        
        // Maintain history size
        self.maintain_history_size();
        
        // Update performance tracker
        self.performance_tracker.total_observations += 1;
        self.performance_tracker.last_update = return_data.timestamp;
        
        // Clear metrics cache
        self.risk_metrics_cache = None;
        self.last_update = Some(return_data.timestamp);
        
        Ok(())
    }

    /// Calculate comprehensive risk metrics
    pub fn calculate_risk_metrics(&mut self) -> Result<RiskMetrics, Box<dyn std::error::Error>> {
        if let Some(ref cached) = self.risk_metrics_cache {
            // Return cached metrics if recent
            if let Some(_last_update) = self.last_update {
                let cache_age = Utc::now().signed_duration_since(cached.timestamp);
                if cache_age.num_seconds() < 300 { // 5 minutes cache
                    return Ok(cached.clone());
                }
            }
        }

        let metrics = if self.return_history.len() < 30 {
            // Not enough data for meaningful statistics
            RiskMetrics::default()
        } else {
            self.compute_detailed_metrics()?
        };

        self.risk_metrics_cache = Some(metrics.clone());
        Ok(metrics)
    }

    /// Perform comprehensive risk assessment
    pub fn assess_risk(&mut self) -> Result<RiskAssessment, Box<dyn std::error::Error>> {
        let metrics = self.calculate_risk_metrics()?;
        
        // Determine risk level and category
        let risk_level = self.calculate_overall_risk_level(&metrics);
        let risk_category = self.classify_risk_level(risk_level);
        
        // Generate warnings
        let warnings = self.generate_risk_warnings(&metrics);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&metrics, &warnings);
        
        Ok(RiskAssessment {
            risk_level,
            risk_category,
            metrics,
            warnings,
            recommendations,
            timestamp: Utc::now(),
        })
    }

    /// Get current portfolio value
    pub fn get_current_portfolio_value(&self) -> f64 {
        self.portfolio_value_history.last().copied().unwrap_or(100.0)
    }

    /// Get current drawdown
    pub fn get_current_drawdown(&self) -> f64 {
        self.drawdown_history.last().copied().unwrap_or(0.0)
    }

    /// Get return history
    pub fn get_return_history(&self) -> &[f64] {
        &self.return_history
    }

    /// Get configuration
    pub fn get_config(&self) -> &RiskConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: RiskConfig) {
        self.config = new_config;
        self.risk_metrics_cache = None; // Clear cache
    }

    /// Export risk data for analysis
    pub fn export_risk_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        
        data.insert("portfolio_values".to_string(), 
                   serde_json::to_value(&self.portfolio_value_history).unwrap_or_default());
        data.insert("returns".to_string(), 
                   serde_json::to_value(&self.return_history).unwrap_or_default());
        data.insert("volatility".to_string(), 
                   serde_json::to_value(&self.volatility_history).unwrap_or_default());
        data.insert("drawdowns".to_string(), 
                   serde_json::to_value(&self.drawdown_history).unwrap_or_default());
        data.insert("config".to_string(), 
                   serde_json::to_value(&self.config).unwrap_or_default());
        
        if let Some(ref metrics) = self.risk_metrics_cache {
            data.insert("risk_metrics".to_string(), 
                       serde_json::to_value(metrics).unwrap_or_default());
        }
        
        data
    }

    // Private helper methods
    
    /// Maintain history size to prevent memory growth
    fn maintain_history_size(&mut self) {
        const MAX_HISTORY_SIZE: usize = 10000;
        
        if self.return_history.len() > MAX_HISTORY_SIZE {
            let excess = self.return_history.len() - MAX_HISTORY_SIZE;
            self.return_history.drain(0..excess);
            self.portfolio_value_history.drain(0..excess);
            self.volatility_history.drain(0..excess);
            self.drawdown_history.drain(0..excess);
        }
    }

    /// Compute detailed risk metrics
    fn compute_detailed_metrics(&self) -> Result<RiskMetrics, Box<dyn std::error::Error>> {
        let returns = &self.return_history;
        let n = returns.len() as f64;
        
        // Basic statistics
        let mean_return = returns.iter().sum::<f64>() / n;
        let variance = returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (n - 1.0);
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized
        
        // VaR and CVaR (95% confidence)
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = (0.05 * n) as usize;
        let var_95 = sorted_returns.get(var_index).copied().unwrap_or(0.0);
        
        let tail_returns: Vec<f64> = sorted_returns.iter()
            .take(var_index + 1)
            .copied()
            .collect();
        let cvar_95 = if tail_returns.is_empty() {
            var_95
        } else {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        };
        
        // Drawdown metrics
        let max_drawdown = self.drawdown_history.iter()
            .fold(0.0f64, |acc, &dd| acc.max(dd));
        let current_drawdown = self.get_current_drawdown();
        
        // Risk-adjusted returns
        let sharpe_ratio = if volatility > 0.0 {
            (mean_return * 252.0) / volatility // Annualized
        } else {
            0.0
        };
        
        // Downside deviation for Sortino ratio
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_deviation = if downside_returns.is_empty() {
            volatility
        } else {
            let downside_variance = downside_returns.iter()
                .map(|&r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            downside_variance.sqrt() * (252.0_f64).sqrt()
        };
        
        let sortino_ratio = if downside_deviation > 0.0 {
            (mean_return * 252.0) / downside_deviation
        } else {
            0.0
        };
        
        let calmar_ratio = if max_drawdown > 0.0 {
            (mean_return * 252.0) / max_drawdown
        } else {
            0.0
        };
        
        // Higher moments
        let skewness = if n > 2.0 && variance > 0.0 {
            let skew_sum = returns.iter()
                .map(|&r| ((r - mean_return) / variance.sqrt()).powi(3))
                .sum::<f64>();
            skew_sum / n
        } else {
            0.0
        };
        
        let kurtosis = if n > 3.0 && variance > 0.0 {
            let kurt_sum = returns.iter()
                .map(|&r| ((r - mean_return) / variance.sqrt()).powi(4))
                .sum::<f64>();
            kurt_sum / n - 3.0 // Excess kurtosis
        } else {
            0.0
        };
        
        // Tail ratio
        let percentile_95_index = ((0.95 * n) as usize).min(sorted_returns.len() - 1);
        let percentile_5_index = ((0.05 * n) as usize).min(sorted_returns.len() - 1);
        let tail_ratio = if sorted_returns[percentile_5_index] != 0.0 {
            sorted_returns[percentile_95_index] / sorted_returns[percentile_5_index].abs()
        } else {
            1.0
        };
        
        // Simplified antifragility and black swan metrics
        let antifragility_score = self.calculate_antifragility_score()?;
        let black_swan_probability = self.calculate_black_swan_probability()?;
        
        Ok(RiskMetrics {
            var_95,
            cvar_95,
            max_drawdown,
            current_drawdown,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            tail_ratio,
            skewness,
            kurtosis,
            antifragility_score,
            black_swan_probability,
            timestamp: Utc::now(),
        })
    }

    /// Calculate overall risk level
    fn calculate_overall_risk_level(&self, metrics: &RiskMetrics) -> f64 {
        let mut risk_factors = Vec::new();
        
        // Volatility risk
        let vol_risk = (metrics.volatility / 0.20).min(1.0); // 20% volatility as high risk
        risk_factors.push(vol_risk);
        
        // Drawdown risk
        let dd_risk = (metrics.current_drawdown / self.config.max_drawdown).min(1.0);
        risk_factors.push(dd_risk);
        
        // VaR risk
        let var_risk = (metrics.var_95.abs() / 0.05).min(1.0); // 5% daily VaR as high risk
        risk_factors.push(var_risk);
        
        // Tail risk
        let tail_risk = if metrics.tail_ratio < 1.0 { 1.0 - metrics.tail_ratio } else { 0.0 };
        risk_factors.push(tail_risk);
        
        // Black swan risk
        risk_factors.push(metrics.black_swan_probability);
        
        // Calculate weighted average
        risk_factors.iter().sum::<f64>() / risk_factors.len() as f64
    }

    /// Classify risk level into categories
    fn classify_risk_level(&self, risk_level: f64) -> RiskCategory {
        match risk_level {
            x if x < 0.2 => RiskCategory::Low,
            x if x < 0.5 => RiskCategory::Moderate,
            x if x < 0.8 => RiskCategory::High,
            x if x < 0.95 => RiskCategory::Extreme,
            _ => RiskCategory::BlackSwan,
        }
    }

    /// Generate risk warnings
    fn generate_risk_warnings(&self, metrics: &RiskMetrics) -> Vec<RiskWarning> {
        let mut warnings = Vec::new();
        
        // High volatility warning
        if metrics.volatility > self.config.target_volatility * 1.5 {
            warnings.push(RiskWarning::HighVolatility {
                current: metrics.volatility,
                threshold: self.config.target_volatility * 1.5,
            });
        }
        
        // Excessive drawdown warning
        if metrics.current_drawdown > self.config.max_drawdown {
            warnings.push(RiskWarning::ExcessiveDrawdown {
                current: metrics.current_drawdown,
                max_allowed: self.config.max_drawdown,
            });
        }
        
        // Black swan detection
        if metrics.black_swan_probability > 0.01 {
            warnings.push(RiskWarning::BlackSwanDetected {
                confidence: metrics.black_swan_probability,
            });
        }
        
        // Volatility targeting miss
        if self.config.volatility_targeting {
            let vol_diff = (metrics.volatility - self.config.target_volatility).abs();
            if vol_diff > self.config.target_volatility * 0.2 {
                warnings.push(RiskWarning::VolatilityTargetMissed {
                    current: metrics.volatility,
                    target: self.config.target_volatility,
                });
            }
        }
        
        warnings
    }

    /// Generate risk management recommendations
    fn generate_recommendations(&self, metrics: &RiskMetrics, warnings: &[RiskWarning]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if metrics.current_drawdown > self.config.max_drawdown * 0.8 {
            recommendations.push("Consider reducing position sizes to limit further drawdown".to_string());
        }
        
        if metrics.volatility > self.config.target_volatility * 1.3 {
            recommendations.push("Implement volatility targeting to reduce portfolio risk".to_string());
        }
        
        if metrics.sharpe_ratio < 0.5 {
            recommendations.push("Review strategy performance and consider diversification".to_string());
        }
        
        if metrics.black_swan_probability > 0.005 {
            recommendations.push("Increase tail risk hedging and consider barbell strategy".to_string());
        }
        
        if !warnings.is_empty() {
            recommendations.push("Monitor positions closely and prepare for risk reduction".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Continue current risk management approach".to_string());
        }
        
        recommendations
    }

    /// Calculate simplified antifragility score
    fn calculate_antifragility_score(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.return_history.len() < 50 {
            return Ok(0.0);
        }
        
        // Simplified: measure how portfolio benefits from volatility
        let volatility_periods = self.get_high_volatility_periods();
        let performance_in_vol = self.get_performance_during_periods(&volatility_periods);
        
        // Antifragile systems perform better during stress
        let baseline_performance = self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;
        let vol_performance = performance_in_vol.iter().sum::<f64>() / performance_in_vol.len().max(1) as f64;
        
        let antifragility = if baseline_performance != 0.0 {
            ((vol_performance - baseline_performance) / baseline_performance.abs()).max(-1.0).min(1.0)
        } else {
            0.0
        };
        
        Ok((antifragility + 1.0) / 2.0) // Normalize to 0-1
    }

    /// Calculate black swan probability
    fn calculate_black_swan_probability(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.return_history.len() < 100 {
            return Ok(0.001); // Default low probability
        }
        
        // Count extreme events (beyond 3 standard deviations)
        let mean = self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;
        let std_dev = {
            let variance = self.return_history.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>() / (self.return_history.len() - 1) as f64;
            variance.sqrt()
        };
        
        let extreme_events = self.return_history.iter()
            .filter(|&&r| (r - mean).abs() > 3.0 * std_dev)
            .count();
        
        let probability = extreme_events as f64 / self.return_history.len() as f64;
        Ok(probability.min(0.1)) // Cap at 10%
    }

    /// Get periods of high volatility
    fn get_high_volatility_periods(&self) -> Vec<usize> {
        if self.volatility_history.len() < 20 {
            return Vec::new();
        }
        
        let mean_vol = self.volatility_history.iter().sum::<f64>() / self.volatility_history.len() as f64;
        let high_vol_threshold = mean_vol * 1.5;
        
        self.volatility_history.iter()
            .enumerate()
            .filter_map(|(i, &vol)| if vol > high_vol_threshold { Some(i) } else { None })
            .collect()
    }

    /// Get performance during specific periods
    fn get_performance_during_periods(&self, periods: &[usize]) -> Vec<f64> {
        periods.iter()
            .filter_map(|&i| self.return_history.get(i))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_talebian_risk_creation() {
        let config = RiskConfig::default();
        let risk_system = TalebianRisk::new(config).unwrap();
        
        assert_eq!(risk_system.return_history.len(), 0);
        assert_eq!(risk_system.get_current_portfolio_value(), 100.0);
    }

    #[test]
    fn test_market_data_update() {
        let mut risk_system = TalebianRisk::new(RiskConfig::default()).unwrap();
        
        let return_data = ReturnData {
            expected_return: 0.01,
            volatility: 0.15,
            timestamp: Utc::now(),
        };
        
        risk_system.update_market_data(&return_data).unwrap();
        assert_eq!(risk_system.return_history.len(), 1);
        assert!(risk_system.get_current_portfolio_value() > 100.0);
    }

    #[test]
    fn test_risk_metrics_calculation() {
        let mut risk_system = TalebianRisk::new(RiskConfig::default()).unwrap();
        
        // Add some sample data
        for i in 0..50 {
            let return_data = ReturnData {
                expected_return: (i as f64 * 0.001) - 0.025, // Some positive, some negative
                volatility: 0.15,
                timestamp: Utc::now(),
            };
            risk_system.update_market_data(&return_data).unwrap();
        }
        
        let metrics = risk_system.calculate_risk_metrics().unwrap();
        assert!(metrics.volatility >= 0.0);
        assert!(metrics.var_95 <= 0.0); // VaR should be negative
    }

    #[test]
    fn test_risk_assessment() {
        let mut risk_system = TalebianRisk::new(RiskConfig::default()).unwrap();
        
        // Add sample data
        for i in 0..30 {
            let return_data = ReturnData {
                expected_return: if i % 10 == 0 { -0.05 } else { 0.01 }, // Some large losses
                volatility: 0.20,
                timestamp: Utc::now(),
            };
            risk_system.update_market_data(&return_data).unwrap();
        }
        
        let assessment = risk_system.assess_risk().unwrap();
        assert!(assessment.risk_level >= 0.0 && assessment.risk_level <= 1.0);
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_risk_category_classification() {
        let config = RiskConfig::default();
        let risk_system = TalebianRisk::new(config).unwrap();
        
        assert!(matches!(risk_system.classify_risk_level(0.1), RiskCategory::Low));
        assert!(matches!(risk_system.classify_risk_level(0.3), RiskCategory::Moderate));
        assert!(matches!(risk_system.classify_risk_level(0.6), RiskCategory::High));
        assert!(matches!(risk_system.classify_risk_level(0.9), RiskCategory::Extreme));
        assert!(matches!(risk_system.classify_risk_level(0.99), RiskCategory::BlackSwan));
    }

    #[test]
    fn test_export_risk_data() {
        let mut risk_system = TalebianRisk::new(RiskConfig::default()).unwrap();
        
        let return_data = ReturnData {
            expected_return: 0.01,
            volatility: 0.15,
            timestamp: Utc::now(),
        };
        risk_system.update_market_data(&return_data).unwrap();
        
        let exported_data = risk_system.export_risk_data();
        assert!(exported_data.contains_key("returns"));
        assert!(exported_data.contains_key("portfolio_values"));
        assert!(exported_data.contains_key("config"));
    }
}