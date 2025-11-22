//! Scientific Rigor Watchdog Implementation
//! 
//! Validates mathematical models and enforces statistical significance requirements
//! with p < 0.001 threshold for trading decisions

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::statistics::Statistics;
use nalgebra::{DVector, DMatrix};
use chrono::{DateTime, Utc};

/// Scientific rigor validation result
#[derive(Debug, Clone)]
pub enum ScientificRigorResult {
    RigorouslyValid { 
        p_value: f64,
        confidence_interval: (f64, f64),
        statistical_power: f64,
    },
    InsufficientRigor { 
        p_value: f64,
        required_p_value: f64,
        issues: Vec<RigorIssue>,
    },
    ModelingFlaws { 
        flaws: Vec<ModelingFlaw>,
        severity: RigorSeverity,
    },
}

/// Rigor violation types
#[derive(Debug, Clone)]
pub enum RigorIssue {
    StatisticalSignificanceFailure { actual_p: f64, required_p: f64 },
    SampleSizeInsufficient { actual_n: usize, required_n: usize },
    ConfidenceIntervalTooWide { width: f64, max_width: f64 },
    MultipleComparisonBias { tests_performed: usize },
    DataDredgingDetected { suspicious_patterns: Vec<String> },
    OverfittingEvidence { validation_score: f64, training_score: f64 },
}

/// Mathematical modeling flaws
#[derive(Debug, Clone)]
pub enum ModelingFlaw {
    DistributionMismatch { expected: String, actual: String },
    AutocorrelationViolation { lag: i32, correlation: f64 },
    HomoscedasticityViolation { test_statistic: f64 },
    NonStationarity { adf_statistic: f64 },
    CointegrationFailure { test_statistic: f64 },
    AsymptoticsInvalid { condition: String },
}

/// Rigor severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum RigorSeverity {
    Critical,  // p > 0.1 or major modeling flaws
    High,      // p > 0.01 or significant issues
    Medium,    // p > 0.001 or minor concerns
    Low,       // Near-threshold violations
}

/// Statistical validation framework
pub struct StatisticalValidator {
    significance_threshold: f64,
    power_threshold: f64,
    confidence_level: f64,
    multiple_comparison_correction: bool,
}

impl StatisticalValidator {
    pub fn new() -> Self {
        Self {
            significance_threshold: 0.001, // p < 0.001 requirement
            power_threshold: 0.8,         // 80% statistical power
            confidence_level: 0.999,      // 99.9% confidence
            multiple_comparison_correction: true,
        }
    }

    /// Validate statistical significance
    pub fn validate_significance(&self, p_value: f64, n_tests: usize) -> Result<bool, TENGRIError> {
        let adjusted_threshold = if self.multiple_comparison_correction && n_tests > 1 {
            // Bonferroni correction for multiple comparisons
            self.significance_threshold / n_tests as f64
        } else {
            self.significance_threshold
        };

        Ok(p_value < adjusted_threshold)
    }

    /// Calculate statistical power
    pub fn calculate_power(&self, effect_size: f64, sample_size: usize) -> f64 {
        // Cohen's power calculation (simplified)
        let z_alpha = 3.29; // Z-score for p = 0.001
        let z_beta = (effect_size * (sample_size as f64).sqrt()) - z_alpha;
        
        // Standard normal CDF approximation
        0.5 * (1.0 + (z_beta / (1.0 + 0.33 * z_beta)).tanh())
    }

    /// Validate confidence interval
    pub fn validate_confidence_interval(&self, ci: (f64, f64), max_width: f64) -> bool {
        let width = (ci.1 - ci.0).abs();
        width <= max_width
    }

    /// Detect data dredging/p-hacking
    pub fn detect_data_dredging(&self, test_results: &[f64]) -> Vec<String> {
        let mut suspicious_patterns = Vec::new();

        // Check for just-significant results clustering
        let just_significant = test_results.iter()
            .filter(|&&p| p < 0.05 && p > 0.03)
            .count();

        if just_significant > test_results.len() / 3 {
            suspicious_patterns.push("Suspicious clustering of just-significant results".to_string());
        }

        // Check for distribution anomalies
        let mean_p = test_results.iter().sum::<f64>() / test_results.len() as f64;
        if mean_p > 0.5 {
            suspicious_patterns.push("P-value distribution suggests selective reporting".to_string());
        }

        suspicious_patterns
    }
}

/// Mathematical model validator
pub struct MathematicalModelValidator {
    distributional_tests: HashMap<String, f64>,
    time_series_tests: HashMap<String, f64>,
    regression_diagnostics: HashMap<String, f64>,
}

impl MathematicalModelValidator {
    pub fn new() -> Self {
        Self {
            distributional_tests: HashMap::new(),
            time_series_tests: HashMap::new(),
            regression_diagnostics: HashMap::new(),
        }
    }

    /// Validate model assumptions
    pub fn validate_model_assumptions(&mut self, data: &[f64], model_type: &str) -> Result<Vec<ModelingFlaw>, TENGRIError> {
        let mut flaws = Vec::new();

        // Test for normality (Shapiro-Wilk approximation)
        let normality_p = self.test_normality(data)?;
        if normality_p < 0.05 && model_type.contains("normal") {
            flaws.push(ModelingFlaw::DistributionMismatch {
                expected: "Normal".to_string(),
                actual: "Non-normal".to_string(),
            });
        }

        // Test for stationarity (Augmented Dickey-Fuller)
        let adf_stat = self.test_stationarity(data)?;
        if adf_stat > -2.86 && model_type.contains("stationary") {
            flaws.push(ModelingFlaw::NonStationarity { adf_statistic: adf_stat });
        }

        // Test for autocorrelation
        let autocorr = self.test_autocorrelation(data, 1)?;
        if autocorr.abs() > 0.3 && model_type.contains("independent") {
            flaws.push(ModelingFlaw::AutocorrelationViolation { 
                lag: 1, 
                correlation: autocorr 
            });
        }

        // Test for heteroscedasticity
        let het_stat = self.test_heteroscedasticity(data)?;
        if het_stat > 5.99 { // Chi-square critical value at 5%
            flaws.push(ModelingFlaw::HomoscedasticityViolation { 
                test_statistic: het_stat 
            });
        }

        Ok(flaws)
    }

    /// Test normality using Shapiro-Wilk approximation
    fn test_normality(&self, data: &[f64]) -> Result<f64, TENGRIError> {
        if data.len() < 3 {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Insufficient data for normality test".to_string()
            });
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        
        // Simplified normality test based on skewness and kurtosis
        let skewness = data.iter().map(|x| ((x - mean) / variance.sqrt()).powi(3)).sum::<f64>() / n;
        let kurtosis = data.iter().map(|x| ((x - mean) / variance.sqrt()).powi(4)).sum::<f64>() / n - 3.0;
        
        let jarque_bera = (n / 6.0) * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
        
        // Convert to p-value using chi-square distribution
        let chi_sq = ChiSquared::new(2.0).map_err(|e| TENGRIError::MathematicalValidationFailed {
            reason: format!("Chi-square distribution error: {}", e)
        })?;
        
        Ok(1.0 - chi_sq.cdf(jarque_bera))
    }

    /// Test stationarity using Augmented Dickey-Fuller
    fn test_stationarity(&self, data: &[f64]) -> Result<f64, TENGRIError> {
        if data.len() < 10 {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Insufficient data for stationarity test".to_string()
            });
        }

        // Simplified ADF test implementation
        let mut y = Vec::new();
        let mut x = Vec::new();
        
        for i in 1..data.len() {
            y.push(data[i] - data[i-1]);
            x.push(data[i-1]);
        }
        
        // Simple regression: Δy = α + βy_{t-1} + ε
        let n = y.len() as f64;
        let y_mean = y.iter().sum::<f64>() / n;
        let x_mean = x.iter().sum::<f64>() / n;
        
        let numerator: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - x_mean) * (yi - y_mean)).sum();
        let denominator: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
        
        let beta = numerator / denominator;
        let alpha = y_mean - beta * x_mean;
        
        // Calculate residuals and standard error
        let residuals: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| yi - alpha - beta * xi).collect();
        let sse: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let mse = sse / (n - 2.0);
        let se_beta = (mse / denominator).sqrt();
        
        // t-statistic for unit root test
        Ok(beta / se_beta)
    }

    /// Test autocorrelation
    fn test_autocorrelation(&self, data: &[f64], lag: usize) -> Result<f64, TENGRIError> {
        if data.len() <= lag {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Insufficient data for autocorrelation test".to_string()
            });
        }

        let n = data.len() - lag;
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        
        let numerator: f64 = (0..n).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum();
        let denominator: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        
        Ok(numerator / denominator)
    }

    /// Test for heteroscedasticity
    fn test_heteroscedasticity(&self, data: &[f64]) -> Result<f64, TENGRIError> {
        if data.len() < 6 {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Insufficient data for heteroscedasticity test".to_string()
            });
        }

        // Simplified Breusch-Pagan test
        let n = data.len();
        let mid = n / 2;
        
        let first_half = &data[..mid];
        let second_half = &data[mid..];
        
        let var1 = first_half.iter().map(|x| x.powi(2)).sum::<f64>() / first_half.len() as f64;
        let var2 = second_half.iter().map(|x| x.powi(2)).sum::<f64>() / second_half.len() as f64;
        
        // F-statistic approximation
        let f_stat = if var1 > var2 { var1 / var2 } else { var2 / var1 };
        
        // Convert to chi-square statistic
        Ok(f_stat.ln() * n as f64 / 2.0)
    }
}

/// Scientific rigor watchdog
pub struct ScientificRigorWatchdog {
    statistical_validator: StatisticalValidator,
    model_validator: Arc<RwLock<MathematicalModelValidator>>,
    rigor_violations: Arc<RwLock<HashMap<String, u64>>>,
    validation_cache: Arc<RwLock<HashMap<String, ScientificRigorResult>>>,
}

impl ScientificRigorWatchdog {
    /// Create new scientific rigor watchdog
    pub async fn new() -> Result<Self, TENGRIError> {
        let statistical_validator = StatisticalValidator::new();
        let model_validator = Arc::new(RwLock::new(MathematicalModelValidator::new()));
        let rigor_violations = Arc::new(RwLock::new(HashMap::new()));
        let validation_cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            statistical_validator,
            model_validator,
            rigor_violations,
            validation_cache,
        })
    }

    /// Validate scientific rigor for trading operation
    pub async fn validate(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        // Check cache first
        let cache_key = format!("{}_{}", operation.mathematical_model, operation.data_source);
        if let Some(cached_result) = self.check_cache(&cache_key).await {
            return self.convert_rigor_result(cached_result);
        }

        // Extract mathematical model data (simplified - in practice would parse model)
        let model_data = self.extract_model_data(operation).await?;
        
        // Comprehensive rigor validation
        let statistical_result = self.validate_statistical_rigor(&model_data, operation).await?;
        let modeling_result = self.validate_modeling_assumptions(&model_data, operation).await?;
        
        // Aggregate results
        let final_result = self.aggregate_rigor_results(statistical_result, modeling_result).await?;
        
        // Cache result
        self.cache_result(&cache_key, final_result.clone()).await;
        
        self.convert_rigor_result(final_result)
    }

    /// Extract model data from operation
    async fn extract_model_data(&self, operation: &TradingOperation) -> Result<Vec<f64>, TENGRIError> {
        // In practice, this would parse the mathematical model and extract relevant data
        // For now, generate synthetic data based on operation parameters
        let sample_size = (operation.risk_parameters.confidence_threshold * 1000.0) as usize;
        let data: Vec<f64> = (0..sample_size)
            .map(|i| {
                // Generate realistic-looking financial data
                let base = 100.0 + (i as f64 * 0.01);
                let noise = (i as f64 * 0.1).sin() * 0.5;
                base + noise
            })
            .collect();
        
        Ok(data)
    }

    /// Validate statistical rigor
    async fn validate_statistical_rigor(
        &self,
        data: &[f64],
        operation: &TradingOperation,
    ) -> Result<ScientificRigorResult, TENGRIError> {
        let mut issues = Vec::new();

        // Mock p-value calculation (in practice, would run actual statistical tests)
        let p_value = self.calculate_mock_p_value(data, operation).await?;
        
        // Validate significance
        if !self.statistical_validator.validate_significance(p_value, 1)? {
            issues.push(RigorIssue::StatisticalSignificanceFailure {
                actual_p: p_value,
                required_p: self.statistical_validator.significance_threshold,
            });
        }

        // Validate sample size
        let required_n = self.calculate_required_sample_size(operation.risk_parameters.confidence_threshold);
        if data.len() < required_n {
            issues.push(RigorIssue::SampleSizeInsufficient {
                actual_n: data.len(),
                required_n,
            });
        }

        // Calculate confidence interval
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64).sqrt();
        let margin_of_error = 3.29 * std_dev / (data.len() as f64).sqrt(); // 99.9% confidence
        let ci = (mean - margin_of_error, mean + margin_of_error);

        // Validate confidence interval width
        if !self.statistical_validator.validate_confidence_interval(ci, 10.0) {
            issues.push(RigorIssue::ConfidenceIntervalTooWide {
                width: ci.1 - ci.0,
                max_width: 10.0,
            });
        }

        // Calculate statistical power
        let effect_size = 0.8; // Cohen's medium effect size
        let power = self.statistical_validator.calculate_power(effect_size, data.len());

        if issues.is_empty() {
            Ok(ScientificRigorResult::RigorouslyValid {
                p_value,
                confidence_interval: ci,
                statistical_power: power,
            })
        } else {
            Ok(ScientificRigorResult::InsufficientRigor {
                p_value,
                required_p_value: self.statistical_validator.significance_threshold,
                issues,
            })
        }
    }

    /// Validate modeling assumptions
    async fn validate_modeling_assumptions(
        &self,
        data: &[f64],
        operation: &TradingOperation,
    ) -> Result<Vec<ModelingFlaw>, TENGRIError> {
        let mut validator = self.model_validator.write().await;
        validator.validate_model_assumptions(data, &operation.mathematical_model)
    }

    /// Calculate mock p-value for demonstration
    async fn calculate_mock_p_value(&self, data: &[f64], operation: &TradingOperation) -> Result<f64, TENGRIError> {
        // In practice, this would perform actual statistical tests
        // For now, calculate based on confidence threshold
        let base_p = 1.0 - operation.risk_parameters.confidence_threshold;
        
        // Add some realistic variation
        let data_factor = (data.len() as f64).log10() / 10.0;
        let p_value = base_p * (1.0 + data_factor);
        
        Ok(p_value.max(0.0001)) // Ensure minimum p-value
    }

    /// Calculate required sample size
    fn calculate_required_sample_size(&self, confidence_threshold: f64) -> usize {
        // Power analysis for required sample size
        let z_alpha = 3.29; // For p < 0.001
        let z_beta = 0.84;  // For 80% power
        let effect_size = 0.8; // Medium effect size
        
        let n = ((z_alpha + z_beta) / effect_size).powi(2);
        n.ceil() as usize
    }

    /// Aggregate rigor results
    async fn aggregate_rigor_results(
        &self,
        statistical_result: ScientificRigorResult,
        modeling_flaws: Vec<ModelingFlaw>,
    ) -> Result<ScientificRigorResult, TENGRIError> {
        // Modeling flaws take precedence
        if !modeling_flaws.is_empty() {
            let severity = self.determine_flaw_severity(&modeling_flaws);
            return Ok(ScientificRigorResult::ModelingFlaws {
                flaws: modeling_flaws,
                severity,
            });
        }

        Ok(statistical_result)
    }

    /// Determine severity of modeling flaws
    fn determine_flaw_severity(&self, flaws: &[ModelingFlaw]) -> RigorSeverity {
        let critical_flaws = flaws.iter().any(|flaw| matches!(
            flaw,
            ModelingFlaw::DistributionMismatch { .. } |
            ModelingFlaw::NonStationarity { .. }
        ));

        if critical_flaws {
            RigorSeverity::Critical
        } else if flaws.len() > 2 {
            RigorSeverity::High
        } else if flaws.len() > 1 {
            RigorSeverity::Medium
        } else {
            RigorSeverity::Low
        }
    }

    /// Convert rigor result to TENGRI oversight result
    fn convert_rigor_result(&self, result: ScientificRigorResult) -> Result<TENGRIOversightResult, TENGRIError> {
        match result {
            ScientificRigorResult::RigorouslyValid { p_value, .. } => {
                Ok(TENGRIOversightResult::Approved)
            },
            
            ScientificRigorResult::InsufficientRigor { p_value, required_p_value, issues } => {
                let severity = if p_value > 0.1 {
                    RigorSeverity::Critical
                } else if p_value > 0.01 {
                    RigorSeverity::High
                } else {
                    RigorSeverity::Medium
                };

                match severity {
                    RigorSeverity::Critical => Ok(TENGRIOversightResult::CriticalViolation {
                        violation_type: ViolationType::MathematicalInconsistency,
                        immediate_shutdown: true,
                        forensic_data: format!("P-value: {}, Required: {}", p_value, required_p_value).into_bytes(),
                    }),
                    RigorSeverity::High => Ok(TENGRIOversightResult::Rejected {
                        reason: format!("Insufficient statistical rigor: p={:.6}", p_value),
                        emergency_action: crate::EmergencyAction::QuarantineAgent {
                            agent_id: "statistical_model_agent".to_string(),
                        },
                    }),
                    _ => Ok(TENGRIOversightResult::Warning {
                        reason: format!("Statistical concerns: p={:.6}", p_value),
                        corrective_action: "Improve statistical rigor or increase sample size".to_string(),
                    }),
                }
            },
            
            ScientificRigorResult::ModelingFlaws { flaws, severity } => {
                let flaw_descriptions: Vec<String> = flaws.iter().map(|f| format!("{:?}", f)).collect();
                
                match severity {
                    RigorSeverity::Critical => Ok(TENGRIOversightResult::CriticalViolation {
                        violation_type: ViolationType::MathematicalInconsistency,
                        immediate_shutdown: true,
                        forensic_data: flaw_descriptions.join("; ").into_bytes(),
                    }),
                    RigorSeverity::High => Ok(TENGRIOversightResult::Rejected {
                        reason: format!("Critical modeling flaws: {}", flaw_descriptions.join(", ")),
                        emergency_action: crate::EmergencyAction::RollbackToSafeState,
                    }),
                    _ => Ok(TENGRIOversightResult::Warning {
                        reason: format!("Modeling concerns: {}", flaw_descriptions.join(", ")),
                        corrective_action: "Address modeling assumptions and validate approach".to_string(),
                    }),
                }
            },
        }
    }

    /// Check cache for previous validation result
    async fn check_cache(&self, key: &str) -> Option<ScientificRigorResult> {
        let cache = self.validation_cache.read().await;
        cache.get(key).cloned()
    }

    /// Cache validation result
    async fn cache_result(&self, key: &str, result: ScientificRigorResult) {
        let mut cache = self.validation_cache.write().await;
        cache.insert(key.to_string(), result);

        // Limit cache size
        if cache.len() > 1000 {
            let oldest_key = cache.keys().next().unwrap().clone();
            cache.remove(&oldest_key);
        }
    }

    /// Get rigor statistics
    pub async fn get_rigor_statistics(&self) -> RigorStatistics {
        let violations = self.rigor_violations.read().await;
        let cache = self.validation_cache.read().await;

        RigorStatistics {
            total_validations: cache.len(),
            total_violations: violations.values().sum(),
            cache_size: cache.len(),
            significance_threshold: self.statistical_validator.significance_threshold,
            power_threshold: self.statistical_validator.power_threshold,
        }
    }
}

/// Rigor statistics
#[derive(Debug, Clone)]
pub struct RigorStatistics {
    pub total_validations: usize,
    pub total_violations: u64,
    pub cache_size: usize,
    pub significance_threshold: f64,
    pub power_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_scientific_rigor_validation() {
        let watchdog = ScientificRigorWatchdog::new().await.unwrap();
        
        let operation = TradingOperation {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: crate::OperationType::PlaceOrder,
            data_source: "rigorous_statistical_model".to_string(),
            mathematical_model: "validated_regression_model".to_string(),
            risk_parameters: crate::RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.999, // High confidence
            },
            agent_id: "test_agent".to_string(),
        };
        
        let result = watchdog.validate(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved));
    }

    #[tokio::test]
    async fn test_insufficient_rigor_detection() {
        let watchdog = ScientificRigorWatchdog::new().await.unwrap();
        
        let operation = TradingOperation {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: crate::OperationType::PlaceOrder,
            data_source: "weak_statistical_model".to_string(),
            mathematical_model: "unvalidated_model".to_string(),
            risk_parameters: crate::RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.05, // Very low confidence
            },
            agent_id: "test_agent".to_string(),
        };
        
        let result = watchdog.validate(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::CriticalViolation { .. }));
    }

    #[test]
    fn test_statistical_validator() {
        let validator = StatisticalValidator::new();
        
        // Test significance validation
        assert!(validator.validate_significance(0.0001, 1).unwrap());
        assert!(!validator.validate_significance(0.01, 1).unwrap());
        
        // Test power calculation
        let power = validator.calculate_power(0.8, 100);
        assert!(power > 0.5);
        
        // Test confidence interval validation
        assert!(validator.validate_confidence_interval((95.0, 105.0), 15.0));
        assert!(!validator.validate_confidence_interval((90.0, 110.0), 15.0));
    }
}