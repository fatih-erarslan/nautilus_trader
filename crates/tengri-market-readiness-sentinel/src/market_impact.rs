//! Market impact assessment system for trading operations
//!
//! This module provides comprehensive market impact analysis including:
//! - Real-time market impact calculation
//! - Temporary vs permanent impact analysis
//! - Participation rate optimization
//! - Execution cost analysis

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::types::ValidationResult;
use crate::ValidationStatus;
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactAnalysis {
    pub symbol: String,
    pub order_size: f64,
    pub predicted_impact: f64,
    pub actual_impact: f64,
    pub temporary_impact: f64,
    pub permanent_impact: f64,
    pub participation_rate: f64,
    pub execution_cost: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactModel {
    pub model_name: String,
    pub model_type: ImpactModelType,
    pub parameters: HashMap<String, f64>,
    pub accuracy: f64,
    pub last_calibration: DateTime<Utc>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactModelType {
    LinearImpact,
    SquareRootImpact,
    AlmgrenChriss,
    PowerLaw,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityProfile {
    pub symbol: String,
    pub bid_size: f64,
    pub ask_size: f64,
    pub spread: f64,
    pub daily_volume: f64,
    pub volatility: f64,
    pub liquidity_score: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationRateOptimization {
    pub symbol: String,
    pub optimal_rate: f64,
    pub current_rate: f64,
    pub cost_savings: f64,
    pub completion_time: u64,
    pub risk_level: f64,
}

#[derive(Debug)]
pub struct MarketImpactAssessor {
    config: Arc<MarketReadinessConfig>,
    impact_models: Arc<RwLock<HashMap<String, ImpactModel>>>,
    liquidity_profiles: Arc<RwLock<HashMap<String, LiquidityProfile>>>,
    impact_analyses: Arc<RwLock<Vec<MarketImpactAnalysis>>>,
    participation_optimizations: Arc<RwLock<HashMap<String, ParticipationRateOptimization>>>,
    market_conditions: Arc<RwLock<MarketConditions>>,
}

#[derive(Debug, Clone)]
struct MarketConditions {
    pub volatility: f64,
    pub liquidity: f64,
    pub momentum: f64,
    pub correlation: f64,
}

impl MarketImpactAssessor {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        Ok(Self {
            config,
            impact_models: Arc::new(RwLock::new(HashMap::new())),
            liquidity_profiles: Arc::new(RwLock::new(HashMap::new())),
            impact_analyses: Arc::new(RwLock::new(Vec::new())),
            participation_optimizations: Arc::new(RwLock::new(HashMap::new())),
            market_conditions: Arc::new(RwLock::new(MarketConditions {
                volatility: 0.15,
                liquidity: 0.8,
                momentum: 0.0,
                correlation: 0.5,
            })),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing market impact assessor...");
        
        // Initialize impact models
        self.initialize_impact_models().await?;
        
        // Initialize liquidity profiles
        self.initialize_liquidity_profiles().await?;
        
        // Start market conditions monitoring
        self.start_market_monitoring().await?;
        
        // Start model calibration
        self.start_model_calibration().await?;
        
        info!("Market impact assessor initialized successfully");
        Ok(())
    }

    pub async fn assess_market_impact(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Validate impact models
        let model_validation = self.validate_impact_models().await?;
        
        // Check liquidity conditions
        let liquidity_check = self.check_liquidity_conditions().await?;
        
        // Assess participation rates
        let participation_assessment = self.assess_participation_rates().await?;
        
        // Validate execution costs
        let cost_validation = self.validate_execution_costs().await?;
        
        // Check model accuracy
        let accuracy_check = self.check_model_accuracy().await?;
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        // Compile results
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        
        if !model_validation.passed {
            issues.push(format!("Impact model validation failed: {}", model_validation.message));
        }
        
        if !liquidity_check.passed {
            if liquidity_check.is_critical {
                issues.push(format!("Critical liquidity issue: {}", liquidity_check.message));
            } else {
                warnings.push(format!("Liquidity warning: {}", liquidity_check.message));
            }
        }
        
        if !participation_assessment.passed {
            warnings.push(format!("Participation rate warning: {}", participation_assessment.message));
        }
        
        if !cost_validation.passed {
            issues.push(format!("Execution cost validation failed: {}", cost_validation.message));
        }
        
        if !accuracy_check.passed {
            warnings.push(format!("Model accuracy warning: {}", accuracy_check.message));
        }
        
        let result = if !issues.is_empty() {
            ValidationResult {
                status: ValidationStatus::Failed,
                message: format!("Market impact assessment failed: {}", issues.join(", ")),
                details: Some(serde_json::json!({
                    "issues": issues,
                    "warnings": warnings,
                    "model_validation": model_validation,
                    "liquidity_check": liquidity_check,
                    "participation_assessment": participation_assessment,
                    "cost_validation": cost_validation,
                    "accuracy_check": accuracy_check,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.9,
            }
        } else if !warnings.is_empty() {
            ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Market impact assessment warnings: {}", warnings.join(", ")),
                details: Some(serde_json::json!({
                    "warnings": warnings,
                    "current_impact": self.get_current_impact().await?,
                    "optimal_participation_rates": self.get_optimal_participation_rates().await?,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 0.95,
            }
        } else {
            ValidationResult {
                status: ValidationStatus::Passed,
                message: "Market impact assessment passed all checks".to_string(),
                details: Some(serde_json::json!({
                    "current_impact": self.get_current_impact().await?,
                    "liquidity_conditions": self.get_liquidity_conditions().await?,
                    "model_performance": self.get_model_performance().await?,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: 1.0,
            }
        };
        
        Ok(result)
    }

    pub async fn get_current_impact(&self) -> Result<f64> {
        let analyses = self.impact_analyses.read().await;
        
        if analyses.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate average impact over recent analyses
        let recent_analyses: Vec<_> = analyses.iter()
            .filter(|analysis| {
                let age = Utc::now().signed_duration_since(analysis.timestamp);
                age.num_hours() < 1
            })
            .collect();
        
        if recent_analyses.is_empty() {
            return Ok(0.0);
        }
        
        let avg_impact = recent_analyses.iter()
            .map(|analysis| analysis.actual_impact)
            .sum::<f64>() / recent_analyses.len() as f64;
        
        Ok(avg_impact)
    }

    async fn initialize_impact_models(&self) -> Result<()> {
        let mut models = self.impact_models.write().await;
        
        // Linear impact model
        models.insert("linear".to_string(), ImpactModel {
            model_name: "Linear Impact Model".to_string(),
            model_type: ImpactModelType::LinearImpact,
            parameters: [
                ("beta".to_string(), 0.5),
                ("alpha".to_string(), 0.01),
            ].iter().cloned().collect(),
            accuracy: 0.85,
            last_calibration: Utc::now(),
            enabled: true,
        });
        
        // Square root impact model
        models.insert("square_root".to_string(), ImpactModel {
            model_name: "Square Root Impact Model".to_string(),
            model_type: ImpactModelType::SquareRootImpact,
            parameters: [
                ("gamma".to_string(), 0.3),
                ("eta".to_string(), 0.02),
            ].iter().cloned().collect(),
            accuracy: 0.78,
            last_calibration: Utc::now(),
            enabled: true,
        });
        
        // Almgren-Chriss model
        models.insert("almgren_chriss".to_string(), ImpactModel {
            model_name: "Almgren-Chriss Model".to_string(),
            model_type: ImpactModelType::AlmgrenChriss,
            parameters: [
                ("lambda".to_string(), 0.1),
                ("kappa".to_string(), 0.05),
                ("epsilon".to_string(), 0.0625),
            ].iter().cloned().collect(),
            accuracy: 0.82,
            last_calibration: Utc::now(),
            enabled: true,
        });
        
        Ok(())
    }

    async fn initialize_liquidity_profiles(&self) -> Result<()> {
        let mut profiles = self.liquidity_profiles.write().await;
        
        // Sample liquidity profiles
        profiles.insert("BTCUSD".to_string(), LiquidityProfile {
            symbol: "BTCUSD".to_string(),
            bid_size: 10.0,
            ask_size: 10.0,
            spread: 0.01,
            daily_volume: 1_000_000.0,
            volatility: 0.05,
            liquidity_score: 0.9,
            timestamp: Utc::now(),
        });
        
        profiles.insert("ETHUSD".to_string(), LiquidityProfile {
            symbol: "ETHUSD".to_string(),
            bid_size: 50.0,
            ask_size: 50.0,
            spread: 0.005,
            daily_volume: 5_000_000.0,
            volatility: 0.06,
            liquidity_score: 0.85,
            timestamp: Utc::now(),
        });
        
        Ok(())
    }

    async fn start_market_monitoring(&self) -> Result<()> {
        let market_conditions = self.market_conditions.clone();
        let liquidity_profiles = self.liquidity_profiles.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Update market conditions
                let mut conditions = market_conditions.write().await;
                conditions.volatility = Self::calculate_market_volatility();
                conditions.liquidity = Self::calculate_market_liquidity();
                conditions.momentum = Self::calculate_market_momentum();
                conditions.correlation = Self::calculate_market_correlation();
                
                // Update liquidity profiles
                let mut profiles = liquidity_profiles.write().await;
                for (_, profile) in profiles.iter_mut() {
                    profile.timestamp = Utc::now();
                    profile.liquidity_score = Self::calculate_liquidity_score(profile);
                }
            }
        });
        
        Ok(())
    }

    async fn start_model_calibration(&self) -> Result<()> {
        let impact_models = self.impact_models.clone();
        let impact_analyses = self.impact_analyses.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Hourly
            
            loop {
                interval.tick().await;
                
                let models = impact_models.read().await;
                let analyses = impact_analyses.read().await;
                
                // Perform model calibration (simplified)
                for (_, model) in models.iter() {
                    if model.enabled {
                        let _ = Self::calibrate_model(model, &analyses).await;
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn validate_impact_models(&self) -> Result<ModelValidation> {
        let models = self.impact_models.read().await;
        
        let mut valid_models = 0;
        let mut total_models = 0;
        let mut accuracy_sum = 0.0;
        
        for (_, model) in models.iter() {
            if model.enabled {
                total_models += 1;
                accuracy_sum += model.accuracy;
                
                if model.accuracy > 0.7 {
                    valid_models += 1;
                }
            }
        }
        
        let avg_accuracy = if total_models > 0 {
            accuracy_sum / total_models as f64
        } else {
            0.0
        };
        
        Ok(ModelValidation {
            passed: valid_models == total_models && avg_accuracy > 0.75,
            message: format!("Models validated: {}/{}, average accuracy: {:.1}%", 
                valid_models, total_models, avg_accuracy * 100.0),
            valid_models,
            total_models,
            avg_accuracy,
        })
    }

    async fn check_liquidity_conditions(&self) -> Result<LiquidityCheck> {
        let profiles = self.liquidity_profiles.read().await;
        let conditions = self.market_conditions.read().await;
        
        let mut low_liquidity_symbols = Vec::new();
        let mut critical_liquidity_symbols = Vec::new();
        
        for (symbol, profile) in profiles.iter() {
            if profile.liquidity_score < 0.3 {
                critical_liquidity_symbols.push(symbol.clone());
            } else if profile.liquidity_score < 0.5 {
                low_liquidity_symbols.push(symbol.clone());
            }
        }
        
        let overall_liquidity = conditions.liquidity;
        let is_critical = overall_liquidity < 0.3 || !critical_liquidity_symbols.is_empty();
        
        Ok(LiquidityCheck {
            passed: overall_liquidity >= 0.5 && critical_liquidity_symbols.is_empty(),
            is_critical,
            message: if is_critical {
                format!("Critical liquidity conditions detected. Overall: {:.1}%, Critical symbols: {}", 
                    overall_liquidity * 100.0, critical_liquidity_symbols.join(", "))
            } else if !low_liquidity_symbols.is_empty() {
                format!("Low liquidity detected for symbols: {}", low_liquidity_symbols.join(", "))
            } else {
                "Liquidity conditions are adequate".to_string()
            },
            low_liquidity_symbols,
            critical_liquidity_symbols,
        })
    }

    async fn assess_participation_rates(&self) -> Result<ParticipationAssessment> {
        let optimizations = self.participation_optimizations.read().await;
        let mut suboptimal_symbols = Vec::new();
        let mut cost_savings = 0.0;
        
        for (symbol, optimization) in optimizations.iter() {
            let rate_diff = (optimization.optimal_rate - optimization.current_rate).abs();
            if rate_diff > 0.05 { // 5% threshold
                suboptimal_symbols.push(symbol.clone());
                cost_savings += optimization.cost_savings;
            }
        }
        
        Ok(ParticipationAssessment {
            passed: suboptimal_symbols.is_empty(),
            message: if suboptimal_symbols.is_empty() {
                "All participation rates are optimal".to_string()
            } else {
                format!("Suboptimal participation rates for {} symbols, potential savings: ${:.2}", 
                    suboptimal_symbols.len(), cost_savings)
            },
            suboptimal_symbols,
            potential_savings: cost_savings,
        })
    }

    async fn validate_execution_costs(&self) -> Result<CostValidation> {
        let analyses = self.impact_analyses.read().await;
        
        let recent_analyses: Vec<_> = analyses.iter()
            .filter(|analysis| {
                let age = Utc::now().signed_duration_since(analysis.timestamp);
                age.num_hours() < 24
            })
            .collect();
        
        if recent_analyses.is_empty() {
            return Ok(CostValidation {
                passed: true,
                message: "No recent execution data available".to_string(),
                avg_cost: 0.0,
                cost_threshold: 0.0,
            });
        }
        
        let avg_cost = recent_analyses.iter()
            .map(|analysis| analysis.execution_cost)
            .sum::<f64>() / recent_analyses.len() as f64;
        
        let cost_threshold = 0.005; // 50 bps threshold
        
        Ok(CostValidation {
            passed: avg_cost <= cost_threshold,
            message: format!("Average execution cost: {:.1} bps (threshold: {:.1} bps)", 
                avg_cost * 10000.0, cost_threshold * 10000.0),
            avg_cost,
            cost_threshold,
        })
    }

    async fn check_model_accuracy(&self) -> Result<AccuracyCheck> {
        let models = self.impact_models.read().await;
        let analyses = self.impact_analyses.read().await;
        
        let mut model_errors = HashMap::new();
        
        // Calculate prediction errors
        for analysis in analyses.iter() {
            let error = (analysis.predicted_impact - analysis.actual_impact).abs();
            for (model_name, _) in models.iter() {
                let entry = model_errors.entry(model_name.clone()).or_insert(Vec::new());
                entry.push(error);
            }
        }
        
        let mut poor_models = Vec::new();
        for (model_name, errors) in model_errors.iter() {
            if !errors.is_empty() {
                let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                if avg_error > 0.01 { // 1% threshold
                    poor_models.push(model_name.clone());
                }
            }
        }
        
        Ok(AccuracyCheck {
            passed: poor_models.is_empty(),
            message: if poor_models.is_empty() {
                "All models have acceptable accuracy".to_string()
            } else {
                format!("Poor accuracy models: {}", poor_models.join(", "))
            },
            poor_models,
        })
    }

    async fn get_liquidity_conditions(&self) -> Result<HashMap<String, f64>> {
        let profiles = self.liquidity_profiles.read().await;
        let conditions: HashMap<String, f64> = profiles.iter()
            .map(|(symbol, profile)| (symbol.clone(), profile.liquidity_score))
            .collect();
        Ok(conditions)
    }

    async fn get_model_performance(&self) -> Result<HashMap<String, f64>> {
        let models = self.impact_models.read().await;
        let performance: HashMap<String, f64> = models.iter()
            .map(|(name, model)| (name.clone(), model.accuracy))
            .collect();
        Ok(performance)
    }

    async fn get_optimal_participation_rates(&self) -> Result<HashMap<String, f64>> {
        let optimizations = self.participation_optimizations.read().await;
        let rates: HashMap<String, f64> = optimizations.iter()
            .map(|(symbol, opt)| (symbol.clone(), opt.optimal_rate))
            .collect();
        Ok(rates)
    }

    // Helper methods for calculations
    fn calculate_market_volatility() -> f64 {
        // Simplified volatility calculation
        0.15 + (rand::random::<f64>() - 0.5) * 0.05
    }

    fn calculate_market_liquidity() -> f64 {
        // Simplified liquidity calculation
        0.8 + (rand::random::<f64>() - 0.5) * 0.2
    }

    fn calculate_market_momentum() -> f64 {
        // Simplified momentum calculation
        (rand::random::<f64>() - 0.5) * 0.1
    }

    fn calculate_market_correlation() -> f64 {
        // Simplified correlation calculation
        0.5 + (rand::random::<f64>() - 0.5) * 0.3
    }

    fn calculate_liquidity_score(profile: &LiquidityProfile) -> f64 {
        // Simplified liquidity score calculation
        let volume_score = (profile.daily_volume / 1_000_000.0).min(1.0);
        let spread_score = (0.01 / profile.spread).min(1.0);
        let size_score = ((profile.bid_size + profile.ask_size) / 100.0).min(1.0);
        
        (volume_score + spread_score + size_score) / 3.0
    }

    async fn calibrate_model(model: &ImpactModel, analyses: &[MarketImpactAnalysis]) -> Result<()> {
        // Simplified model calibration
        // In reality, this would involve complex optimization
        Ok(())
    }
}

// Supporting structures
#[derive(Debug, Clone)]
struct ModelValidation {
    passed: bool,
    message: String,
    valid_models: usize,
    total_models: usize,
    avg_accuracy: f64,
}

#[derive(Debug, Clone)]
struct LiquidityCheck {
    passed: bool,
    is_critical: bool,
    message: String,
    low_liquidity_symbols: Vec<String>,
    critical_liquidity_symbols: Vec<String>,
}

#[derive(Debug, Clone)]
struct ParticipationAssessment {
    passed: bool,
    message: String,
    suboptimal_symbols: Vec<String>,
    potential_savings: f64,
}

#[derive(Debug, Clone)]
struct CostValidation {
    passed: bool,
    message: String,
    avg_cost: f64,
    cost_threshold: f64,
}

#[derive(Debug, Clone)]
struct AccuracyCheck {
    passed: bool,
    message: String,
    poor_models: Vec<String>,
}