//! Volatility assessment system for market readiness validation
//!
//! This module provides comprehensive volatility analysis including:
//! - Real-time volatility calculation
//! - Historical volatility comparison
//! - Volatility regime classification
//! - Risk-adjusted volatility metrics

use std::sync::Arc;
use std::collections::VecDeque;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::types::ValidationResult;
use crate::{VolatilityLevel, ValidationStatus};
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityMetrics {
    pub realized_volatility: f64,
    pub implied_volatility: f64,
    pub historical_volatility: f64,
    pub volatility_of_volatility: f64,
    pub volatility_skew: f64,
    pub volatility_term_structure: Vec<f64>,
    pub volatility_smile: Vec<f64>,
    pub garch_forecast: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityProfile {
    pub current_level: VolatilityLevel,
    pub percentile_rank: f64,
    pub z_score: f64,
    pub trend: VolatilityTrend,
    pub regime_probability: f64,
    pub risk_adjusted_vol: f64,
    pub stress_test_vol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityWindow {
    pub window_size: usize,
    pub returns: VecDeque<f64>,
    pub volatilities: VecDeque<f64>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug)]
pub struct VolatilityAssessor {
    config: Arc<MarketReadinessConfig>,
    volatility_metrics: Arc<RwLock<VolatilityMetrics>>,
    volatility_profile: Arc<RwLock<VolatilityProfile>>,
    price_history: Arc<RwLock<VecDeque<f64>>>,
    volatility_windows: Arc<RwLock<Vec<VolatilityWindow>>>,
    garch_model: Arc<RwLock<GarchModel>>,
    risk_models: Arc<RwLock<Vec<RiskModel>>>,
}

impl VolatilityAssessor {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        let garch_model = Arc::new(RwLock::new(GarchModel::new()));
        let risk_models = Arc::new(RwLock::new(Self::initialize_risk_models()));
        
        Ok(Self {
            config,
            volatility_metrics: Arc::new(RwLock::new(VolatilityMetrics {
                realized_volatility: 0.0,
                implied_volatility: 0.0,
                historical_volatility: 0.0,
                volatility_of_volatility: 0.0,
                volatility_skew: 0.0,
                volatility_term_structure: Vec::new(),
                volatility_smile: Vec::new(),
                garch_forecast: 0.0,
            })),
            volatility_profile: Arc::new(RwLock::new(VolatilityProfile {
                current_level: VolatilityLevel::Normal,
                percentile_rank: 50.0,
                z_score: 0.0,
                trend: VolatilityTrend::Stable,
                regime_probability: 0.5,
                risk_adjusted_vol: 0.0,
                stress_test_vol: 0.0,
            })),
            price_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            volatility_windows: Arc::new(RwLock::new(Vec::new())),
            garch_model,
            risk_models,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing volatility assessor...");
        
        // Initialize GARCH model
        self.garch_model.write().await.initialize().await?;
        
        // Initialize volatility windows
        self.initialize_volatility_windows().await?;
        
        // Start real-time volatility monitoring
        self.start_volatility_monitoring().await?;
        
        info!("Volatility assessor initialized successfully");
        Ok(())
    }

    pub async fn assess_volatility(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Update volatility calculations
        self.update_volatility_metrics().await?;
        
        // Assess volatility level
        let volatility_assessment = self.assess_volatility_level().await?;
        
        // Validate volatility models
        let model_validation = self.validate_volatility_models().await?;
        
        // Check volatility thresholds
        let threshold_check = self.check_volatility_thresholds().await?;
        
        // Perform stress testing
        let stress_test = self.perform_volatility_stress_test().await?;
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        // Compile validation results
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        
        if !model_validation.is_valid {
            issues.push(format!("Volatility model validation failed: {}", model_validation.message));
        }
        
        if !threshold_check.within_limits {
            if threshold_check.is_critical {
                issues.push(format!("Critical volatility threshold breach: {}", threshold_check.message));
            } else {
                warnings.push(format!("Volatility threshold warning: {}", threshold_check.message));
            }
        }
        
        if !stress_test.passed {
            warnings.push(format!("Volatility stress test concern: {}", stress_test.message));
        }
        
        let volatility_profile = self.volatility_profile.read().await;
        let volatility_metrics = self.volatility_metrics.read().await;
        
        let result = if !issues.is_empty() {
            ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Volatility assessment failed: {}", issues.join(", ")),
                details: Some(serde_json::json!({
                    "issues": issues,
                    "warnings": warnings,
                    "volatility_profile": *volatility_profile,
                    "volatility_metrics": *volatility_metrics,
                    "model_validation": model_validation,
                    "threshold_check": threshold_check,
                    "stress_test": stress_test,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.9,
            }
        } else if !warnings.is_empty() {
            ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Volatility assessment warnings: {}", warnings.join(", ")),
                details: Some(serde_json::json!({
                    "warnings": warnings,
                    "volatility_profile": *volatility_profile,
                    "volatility_metrics": *volatility_metrics,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.95,
            }
        } else {
            ValidationResult {
                status: ValidationStatus::Passed,
                message: "Volatility assessment passed all checks".to_string(),
                details: Some(serde_json::json!({
                    "volatility_profile": *volatility_profile,
                    "volatility_metrics": *volatility_metrics,
                    "assessment": volatility_assessment,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 1.0,
            }
        };
        
        Ok(result)
    }

    pub async fn get_volatility_level(&self) -> Result<VolatilityLevel> {
        let profile = self.volatility_profile.read().await;
        Ok(profile.current_level.clone())
    }

    async fn update_volatility_metrics(&self) -> Result<()> {
        let price_history = self.price_history.read().await;
        
        if price_history.len() < 20 {
            return Ok(()); // Not enough data
        }
        
        // Calculate returns
        let returns: Vec<f64> = price_history
            .iter()
            .skip(1)
            .zip(price_history.iter())
            .map(|(current, previous)| (current / previous).ln())
            .collect();
        
        if returns.is_empty() {
            return Ok(());
        }
        
        // Calculate various volatility metrics
        let realized_vol = self.calculate_realized_volatility(&returns)?;
        let historical_vol = self.calculate_historical_volatility(&returns)?;
        let vol_of_vol = self.calculate_volatility_of_volatility(&returns)?;
        let garch_forecast = self.garch_model.read().await.forecast(&returns)?;
        
        // Update metrics
        let mut metrics = self.volatility_metrics.write().await;
        metrics.realized_volatility = realized_vol;
        metrics.historical_volatility = historical_vol;
        metrics.volatility_of_volatility = vol_of_vol;
        metrics.garch_forecast = garch_forecast;
        
        // Update volatility term structure (simplified)
        metrics.volatility_term_structure = vec![
            realized_vol * 0.9,  // 1 day
            realized_vol * 0.95, // 1 week
            realized_vol,        // 1 month
            realized_vol * 1.05, // 3 months
            realized_vol * 1.1,  // 6 months
            realized_vol * 1.15, // 1 year
        ];
        
        Ok(())
    }

    async fn assess_volatility_level(&self) -> Result<VolatilityAssessment> {
        let metrics = self.volatility_metrics.read().await;
        let mut profile = self.volatility_profile.write().await;
        
        // Calculate percentile rank against historical distribution
        let percentile_rank = self.calculate_percentile_rank(metrics.realized_volatility).await?;
        
        // Calculate z-score
        let z_score = self.calculate_z_score(metrics.realized_volatility).await?;
        
        // Determine volatility level
        let volatility_level = if metrics.realized_volatility > 0.5 {
            VolatilityLevel::Extreme
        } else if metrics.realized_volatility > 0.3 {
            VolatilityLevel::High
        } else if metrics.realized_volatility > 0.1 {
            VolatilityLevel::Normal
        } else {
            VolatilityLevel::Low
        };
        
        // Determine trend
        let trend = self.calculate_volatility_trend().await?;
        
        // Calculate risk-adjusted volatility
        let risk_adjusted_vol = self.calculate_risk_adjusted_volatility(&metrics).await?;
        
        // Update profile
        profile.current_level = volatility_level.clone();
        profile.percentile_rank = percentile_rank;
        profile.z_score = z_score;
        profile.trend = trend.clone();
        profile.risk_adjusted_vol = risk_adjusted_vol;
        
        Ok(VolatilityAssessment {
            level: volatility_level,
            percentile: percentile_rank,
            z_score,
            trend,
            confidence: 0.95,
        })
    }

    async fn validate_volatility_models(&self) -> Result<ModelValidation> {
        let garch_model = self.garch_model.read().await;
        let risk_models = self.risk_models.read().await;
        
        // Validate GARCH model
        let garch_validation = garch_model.validate().await?;
        
        // Validate risk models
        let mut risk_model_validations = Vec::new();
        for model in risk_models.iter() {
            let validation = model.validate().await?;
            risk_model_validations.push(validation);
        }
        
        // Check if all models are valid
        let all_valid = garch_validation && risk_model_validations.iter().all(|v| *v);
        
        Ok(ModelValidation {
            is_valid: all_valid,
            message: if all_valid {
                "All volatility models validated successfully".to_string()
            } else {
                "Some volatility models failed validation".to_string()
            },
            garch_valid: garch_validation,
            risk_models_valid: risk_model_validations,
        })
    }

    async fn check_volatility_thresholds(&self) -> Result<ThresholdCheck> {
        let metrics = self.volatility_metrics.read().await;
        let profile = self.volatility_profile.read().await;
        
        // Define thresholds
        let warning_threshold = 0.25;
        let critical_threshold = 0.4;
        let extreme_threshold = 0.6;
        
        if metrics.realized_volatility > extreme_threshold {
            return Ok(ThresholdCheck {
                within_limits: false,
                is_critical: true,
                message: format!("Extreme volatility detected: {:.1}%", metrics.realized_volatility * 100.0),
                threshold_breached: "extreme".to_string(),
                current_value: metrics.realized_volatility,
            });
        }
        
        if metrics.realized_volatility > critical_threshold {
            return Ok(ThresholdCheck {
                within_limits: false,
                is_critical: true,
                message: format!("Critical volatility level: {:.1}%", metrics.realized_volatility * 100.0),
                threshold_breached: "critical".to_string(),
                current_value: metrics.realized_volatility,
            });
        }
        
        if metrics.realized_volatility > warning_threshold {
            return Ok(ThresholdCheck {
                within_limits: false,
                is_critical: false,
                message: format!("High volatility warning: {:.1}%", metrics.realized_volatility * 100.0),
                threshold_breached: "warning".to_string(),
                current_value: metrics.realized_volatility,
            });
        }
        
        Ok(ThresholdCheck {
            within_limits: true,
            is_critical: false,
            message: "Volatility within acceptable limits".to_string(),
            threshold_breached: "none".to_string(),
            current_value: metrics.realized_volatility,
        })
    }

    async fn perform_volatility_stress_test(&self) -> Result<StressTestResult> {
        let metrics = self.volatility_metrics.read().await;
        let profile = self.volatility_profile.read().await;
        
        // Simulate stress scenarios
        let stress_scenarios = vec![
            ("Market Crash", 2.0),
            ("Flash Crash", 3.0),
            ("Regime Change", 1.5),
            ("Liquidity Crisis", 2.5),
        ];
        
        let mut stress_results = Vec::new();
        let mut max_stress_vol = 0.0;
        
        for (scenario, multiplier) in stress_scenarios {
            let stressed_vol = metrics.realized_volatility * multiplier;
            max_stress_vol = max_stress_vol.max(stressed_vol);
            
            stress_results.push(StressScenario {
                name: scenario.to_string(),
                multiplier,
                stressed_volatility: stressed_vol,
                acceptable: stressed_vol < 0.8, // 80% volatility threshold
            });
        }
        
        // Update profile with stress test result
        {
            let mut profile_mut = self.volatility_profile.write().await;
            profile_mut.stress_test_vol = max_stress_vol;
        }
        
        let passed = stress_results.iter().all(|s| s.acceptable);
        
        Ok(StressTestResult {
            passed,
            message: if passed {
                "All stress test scenarios passed".to_string()
            } else {
                "Some stress test scenarios failed".to_string()
            },
            max_stressed_volatility: max_stress_vol,
            scenarios: stress_results,
        })
    }

    async fn initialize_volatility_windows(&self) -> Result<()> {
        let mut windows = self.volatility_windows.write().await;
        
        // Create different volatility windows
        windows.push(VolatilityWindow {
            window_size: 20,   // 20-period window
            returns: VecDeque::with_capacity(20),
            volatilities: VecDeque::with_capacity(20),
            timestamp: Utc::now(),
        });
        
        windows.push(VolatilityWindow {
            window_size: 60,   // 60-period window
            returns: VecDeque::with_capacity(60),
            volatilities: VecDeque::with_capacity(60),
            timestamp: Utc::now(),
        });
        
        windows.push(VolatilityWindow {
            window_size: 252,  // 252-period window (1 year)
            returns: VecDeque::with_capacity(252),
            volatilities: VecDeque::with_capacity(252),
            timestamp: Utc::now(),
        });
        
        Ok(())
    }

    async fn start_volatility_monitoring(&self) -> Result<()> {
        let price_history = self.price_history.clone();
        let volatility_windows = self.volatility_windows.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Simulate price updates (in real system, this would come from market data)
                let mut prices = price_history.write().await;
                let new_price = Self::simulate_price_update(&prices);
                
                prices.push_back(new_price);
                if prices.len() > 10000 {
                    prices.pop_front();
                }
                
                // Update volatility windows
                if prices.len() >= 2 {
                    let return_val = (new_price / prices[prices.len() - 2]).ln();
                    
                    let mut windows = volatility_windows.write().await;
                    for window in windows.iter_mut() {
                        window.returns.push_back(return_val);
                        if window.returns.len() > window.window_size {
                            window.returns.pop_front();
                        }
                        
                        // Calculate volatility for this window
                        if window.returns.len() >= 2 {
                            let vol = Self::calculate_window_volatility(&window.returns);
                            window.volatilities.push_back(vol);
                            if window.volatilities.len() > window.window_size {
                                window.volatilities.pop_front();
                            }
                        }
                        
                        window.timestamp = Utc::now();
                    }
                }
            }
        });
        
        Ok(())
    }

    // Helper methods
    fn calculate_realized_volatility(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        Ok(variance.sqrt() * (252.0_f64).sqrt()) // Annualized
    }

    fn calculate_historical_volatility(&self, returns: &[f64]) -> Result<f64> {
        // Use longer window for historical volatility
        let window_size = returns.len().min(252);
        let recent_returns = &returns[returns.len() - window_size..];
        
        self.calculate_realized_volatility(recent_returns)
    }

    fn calculate_volatility_of_volatility(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < 20 {
            return Ok(0.0);
        }
        
        let window_size = 20;
        let mut volatilities = Vec::new();
        
        for i in window_size..returns.len() {
            let window_returns = &returns[i - window_size..i];
            let vol = self.calculate_realized_volatility(window_returns)?;
            volatilities.push(vol);
        }
        
        if volatilities.is_empty() {
            return Ok(0.0);
        }
        
        let mean_vol = volatilities.iter().sum::<f64>() / volatilities.len() as f64;
        let vol_variance = volatilities.iter()
            .map(|v| (v - mean_vol).powi(2))
            .sum::<f64>() / (volatilities.len() - 1) as f64;
        
        Ok(vol_variance.sqrt())
    }

    async fn calculate_percentile_rank(&self, current_vol: f64) -> Result<f64> {
        let windows = self.volatility_windows.read().await;
        
        if let Some(long_window) = windows.iter().find(|w| w.window_size == 252) {
            if !long_window.volatilities.is_empty() {
                let mut sorted_vols: Vec<f64> = long_window.volatilities.iter().cloned().collect();
                sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let position = sorted_vols.iter().position(|&v| v >= current_vol).unwrap_or(sorted_vols.len());
                return Ok((position as f64 / sorted_vols.len() as f64) * 100.0);
            }
        }
        
        Ok(50.0) // Default to 50th percentile
    }

    async fn calculate_z_score(&self, current_vol: f64) -> Result<f64> {
        let windows = self.volatility_windows.read().await;
        
        if let Some(long_window) = windows.iter().find(|w| w.window_size == 252) {
            if long_window.volatilities.len() >= 2 {
                let mean = long_window.volatilities.iter().sum::<f64>() / long_window.volatilities.len() as f64;
                let variance = long_window.volatilities.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / (long_window.volatilities.len() - 1) as f64;
                
                let std_dev = variance.sqrt();
                if std_dev > 0.0 {
                    return Ok((current_vol - mean) / std_dev);
                }
            }
        }
        
        Ok(0.0) // Default to 0 z-score
    }

    async fn calculate_volatility_trend(&self) -> Result<VolatilityTrend> {
        let windows = self.volatility_windows.read().await;
        
        if let Some(short_window) = windows.iter().find(|w| w.window_size == 20) {
            if short_window.volatilities.len() >= 10 {
                let recent_vols: Vec<f64> = short_window.volatilities.iter().rev().take(10).cloned().collect();
                
                // Simple trend calculation
                let first_half_avg = recent_vols.iter().skip(5).sum::<f64>() / 5.0;
                let second_half_avg = recent_vols.iter().take(5).sum::<f64>() / 5.0;
                
                let change_ratio = (second_half_avg - first_half_avg) / first_half_avg;
                
                return Ok(if change_ratio > 0.1 {
                    VolatilityTrend::Increasing
                } else if change_ratio < -0.1 {
                    VolatilityTrend::Decreasing
                } else if change_ratio.abs() > 0.05 {
                    VolatilityTrend::Volatile
                } else {
                    VolatilityTrend::Stable
                });
            }
        }
        
        Ok(VolatilityTrend::Stable)
    }

    async fn calculate_risk_adjusted_volatility(&self, metrics: &VolatilityMetrics) -> Result<f64> {
        // Risk-adjusted volatility using Sharpe ratio concept
        let risk_free_rate = 0.02; // 2% annual risk-free rate
        let excess_return = 0.08; // Assumed 8% excess return
        
        if metrics.realized_volatility > 0.0 {
            Ok(excess_return / metrics.realized_volatility)
        } else {
            Ok(0.0)
        }
    }

    fn simulate_price_update(prices: &VecDeque<f64>) -> f64 {
        let last_price = prices.back().unwrap_or(&100.0);
        let random_change = (rand::random::<f64>() - 0.5) * 0.02; // ±1% random change
        last_price * (1.0 + random_change)
    }

    fn calculate_window_volatility(returns: &VecDeque<f64>) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        variance.sqrt()
    }

    fn initialize_risk_models() -> Vec<RiskModel> {
        vec![
            RiskModel {
                name: "VaR Model".to_string(),
                confidence_level: 0.95,
                is_valid: true,
            },
            RiskModel {
                name: "Expected Shortfall".to_string(),
                confidence_level: 0.99,
                is_valid: true,
            },
            RiskModel {
                name: "GARCH-VaR".to_string(),
                confidence_level: 0.95,
                is_valid: true,
            },
        ]
    }
}

// Supporting structures
#[derive(Debug, Clone)]
struct VolatilityAssessment {
    level: VolatilityLevel,
    percentile: f64,
    z_score: f64,
    trend: VolatilityTrend,
    confidence: f64,
}

#[derive(Debug, Clone)]
struct ModelValidation {
    is_valid: bool,
    message: String,
    garch_valid: bool,
    risk_models_valid: Vec<bool>,
}

#[derive(Debug, Clone)]
struct ThresholdCheck {
    within_limits: bool,
    is_critical: bool,
    message: String,
    threshold_breached: String,
    current_value: f64,
}

#[derive(Debug, Clone)]
struct StressTestResult {
    passed: bool,
    message: String,
    max_stressed_volatility: f64,
    scenarios: Vec<StressScenario>,
}

#[derive(Debug, Clone)]
struct StressScenario {
    name: String,
    multiplier: f64,
    stressed_volatility: f64,
    acceptable: bool,
}

#[derive(Debug, Clone)]
struct RiskModel {
    name: String,
    confidence_level: f64,
    is_valid: bool,
}

impl RiskModel {
    async fn validate(&self) -> Result<bool> {
        // Simplified validation
        Ok(self.is_valid)
    }
}

#[derive(Debug)]
struct GarchModel {
    alpha: f64,
    beta: f64,
    omega: f64,
    is_fitted: bool,
}

impl GarchModel {
    fn new() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.8,
            omega: 0.01,
            is_fitted: false,
        }
    }

    async fn initialize(&mut self) -> Result<()> {
        // Initialize GARCH model parameters
        self.is_fitted = true;
        Ok(())
    }

    async fn validate(&self) -> Result<bool> {
        // Validate GARCH model parameters
        Ok(self.is_fitted && self.alpha > 0.0 && self.beta > 0.0 && self.omega > 0.0)
    }

    fn forecast(&self, returns: &[f64]) -> Result<f64> {
        if !self.is_fitted || returns.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified GARCH forecast
        let last_return = returns.last().unwrap_or(&0.0);
        let last_variance = last_return.powi(2);
        
        let forecast_variance = self.omega + self.alpha * last_variance + self.beta * last_variance;
        Ok(forecast_variance.sqrt())
    }
}