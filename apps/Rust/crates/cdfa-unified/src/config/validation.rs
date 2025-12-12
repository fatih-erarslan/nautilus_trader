//! Configuration validation and type checking
//!
//! This module provides comprehensive validation for CDFA configuration parameters,
//! including runtime type checking, range validation, and consistency checks.

use crate::config::{CdfaConfig, ProcessingConfig, AlgorithmConfig, PerformanceConfig, MlConfig};
use crate::error::{CdfaError, Result};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

/// Validation result for a single parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationIssue {
    pub parameter_path: String,
    pub severity: ValidationSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Complete validation report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
    pub errors: usize,
    pub warnings: usize,
    pub infos: usize,
    pub is_valid: bool,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            errors: 0,
            warnings: 0,
            infos: 0,
            is_valid: true,
        }
    }
    
    /// Add a validation issue
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        match issue.severity {
            ValidationSeverity::Error => {
                self.errors += 1;
                self.is_valid = false;
            },
            ValidationSeverity::Warning => self.warnings += 1,
            ValidationSeverity::Info => self.infos += 1,
        }
        self.issues.push(issue);
    }
    
    /// Add an error
    pub fn add_error(&mut self, path: &str, message: &str, suggestion: Option<&str>) {
        self.add_issue(ValidationIssue {
            parameter_path: path.to_string(),
            severity: ValidationSeverity::Error,
            message: message.to_string(),
            suggestion: suggestion.map(|s| s.to_string()),
        });
    }
    
    /// Add a warning
    pub fn add_warning(&mut self, path: &str, message: &str, suggestion: Option<&str>) {
        self.add_issue(ValidationIssue {
            parameter_path: path.to_string(),
            severity: ValidationSeverity::Warning,
            message: message.to_string(),
            suggestion: suggestion.map(|s| s.to_string()),
        });
    }
    
    /// Add an info message
    pub fn add_info(&mut self, path: &str, message: &str) {
        self.add_issue(ValidationIssue {
            parameter_path: path.to_string(),
            severity: ValidationSeverity::Info,
            message: message.to_string(),
            suggestion: None,
        });
    }
    
    /// Get all errors
    pub fn get_errors(&self) -> Vec<&ValidationIssue> {
        self.issues.iter()
            .filter(|issue| issue.severity == ValidationSeverity::Error)
            .collect()
    }
    
    /// Get all warnings
    pub fn get_warnings(&self) -> Vec<&ValidationIssue> {
        self.issues.iter()
            .filter(|issue| issue.severity == ValidationSeverity::Warning)
            .collect()
    }
    
    /// Print a summary of the validation report
    pub fn print_summary(&self) {
        println!("Validation Report:");
        println!("  Errors: {}", self.errors);
        println!("  Warnings: {}", self.warnings);
        println!("  Info: {}", self.infos);
        println!("  Valid: {}", self.is_valid);
        
        if !self.issues.is_empty() {
            println!("\nIssues:");
            for issue in &self.issues {
                let severity_str = match issue.severity {
                    ValidationSeverity::Error => "ERROR",
                    ValidationSeverity::Warning => "WARN",
                    ValidationSeverity::Info => "INFO",
                };
                println!("  [{}] {}: {}", severity_str, issue.parameter_path, issue.message);
                if let Some(ref suggestion) = issue.suggestion {
                    println!("    Suggestion: {}", suggestion);
                }
            }
        }
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration validator
pub struct ConfigValidator {
    strict_mode: bool,
    custom_validators: HashMap<String, Box<dyn Fn(&CdfaConfig) -> Vec<ValidationIssue>>>,
}

impl ConfigValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            custom_validators: HashMap::new(),
        }
    }
    
    /// Enable strict mode (more rigorous validation)
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }
    
    /// Add a custom validator function
    pub fn add_custom_validator<F>(&mut self, name: String, validator: F)
    where
        F: Fn(&CdfaConfig) -> Vec<ValidationIssue> + 'static,
    {
        self.custom_validators.insert(name, Box::new(validator));
    }
    
    /// Validate a complete configuration
    pub fn validate(&self, config: &CdfaConfig) -> ValidationReport {
        let mut report = ValidationReport::new();
        
        // Validate each configuration section
        self.validate_processing_config(&config.processing, &mut report);
        self.validate_algorithm_config(&config.algorithms, &mut report);
        self.validate_performance_config(&config.performance, &mut report);
        self.validate_ml_config(&config.ml, &mut report);
        self.validate_data_config(&config.data, &mut report);
        self.validate_analysis_config(&config.analysis, &mut report);
        self.validate_visualization_config(&config.visualization, &mut report);
        self.validate_redis_config(&config.redis, &mut report);
        self.validate_validation_config(&config.validation, &mut report);
        self.validate_logging_config(&config.logging, &mut report);
        self.validate_advanced_config(&config.advanced, &mut report);
        self.validate_hardware_config(&config.hardware, &mut report);
        
        // Cross-section validation
        self.validate_consistency(config, &mut report);
        
        // Run custom validators
        for (name, validator) in &self.custom_validators {
            let issues = validator(config);
            for issue in issues {
                report.add_issue(issue);
            }
        }
        
        report
    }
    
    /// Validate processing configuration
    fn validate_processing_config(&self, config: &ProcessingConfig, report: &mut ValidationReport) {
        let base_path = "processing";
        
        // Validate thread count
        if config.num_threads > 0 && config.num_threads > 256 {
            report.add_warning(
                &format!("{}.num_threads", base_path),
                "Very high thread count may cause performance degradation",
                Some("Consider using automatic thread detection (0)"),
            );
        }
        
        // Validate tolerance
        if config.tolerance <= 0.0 {
            report.add_error(
                &format!("{}.tolerance", base_path),
                "Tolerance must be positive",
                Some("Use a small positive value like 1e-10"),
            );
        }
        
        if config.tolerance > 1e-3 {
            report.add_warning(
                &format!("{}.tolerance", base_path),
                "Large tolerance may reduce numerical accuracy",
                Some("Consider using a smaller value like 1e-10"),
            );
        }
        
        // Validate iterations
        if config.max_iterations == 0 {
            report.add_error(
                &format!("{}.max_iterations", base_path),
                "Maximum iterations must be positive",
                Some("Use a reasonable value like 1000"),
            );
        }
        
        if config.max_iterations > 100000 {
            report.add_warning(
                &format!("{}.max_iterations", base_path),
                "Very high iteration count may cause long execution times",
                None,
            );
        }
        
        // Validate convergence threshold
        if config.convergence_threshold <= 0.0 {
            report.add_error(
                &format!("{}.convergence_threshold", base_path),
                "Convergence threshold must be positive",
                Some("Use a small positive value like 1e-6"),
            );
        }
        
        // Validate batch size
        if config.batch_size == 0 {
            report.add_error(
                &format!("{}.batch_size", base_path),
                "Batch size must be positive",
                Some("Use a reasonable value like 1000"),
            );
        }
        
        if config.batch_size > 1_000_000 {
            report.add_warning(
                &format!("{}.batch_size", base_path),
                "Very large batch size may cause memory issues",
                None,
            );
        }
        
        // Validate memory limit
        if config.memory_limit_mb == 0 {
            report.add_warning(
                &format!("{}.memory_limit_mb", base_path),
                "No memory limit set",
                Some("Consider setting a reasonable limit"),
            );
        }
        
        if config.memory_limit_mb > 100_000 {
            report.add_warning(
                &format!("{}.memory_limit_mb", base_path),
                "Very high memory limit",
                Some("Ensure system has sufficient memory"),
            );
        }
        
        // Validate process priority
        if config.process_priority < -20 || config.process_priority > 19 {
            report.add_error(
                &format!("{}.process_priority", base_path),
                "Process priority must be between -20 and 19",
                Some("Use 0 for normal priority"),
            );
        }
        
        // Logical consistency checks
        if config.tolerance >= config.convergence_threshold {
            report.add_warning(
                &format!("{}.tolerance", base_path),
                "Tolerance is larger than convergence threshold",
                Some("Tolerance should typically be smaller than convergence threshold"),
            );
        }
    }
    
    /// Validate algorithm configuration
    fn validate_algorithm_config(&self, config: &AlgorithmConfig, report: &mut ValidationReport) {
        let base_path = "algorithms";
        
        // Validate diversity methods
        if config.diversity_methods.is_empty() {
            report.add_error(
                &format!("{}.diversity_methods", base_path),
                "At least one diversity method must be specified",
                Some("Add methods like 'pearson', 'spearman', 'kendall'"),
            );
        }
        
        let valid_diversity_methods = [
            "pearson", "spearman", "kendall", "kl_divergence", "js_divergence",
            "hellinger", "wasserstein", "cosine", "euclidean", "manhattan", "dtw"
        ];
        
        for method in &config.diversity_methods {
            if !valid_diversity_methods.contains(&method.as_str()) {
                report.add_error(
                    &format!("{}.diversity_methods", base_path),
                    &format!("Unknown diversity method: {}", method),
                    Some("Use one of the supported methods"),
                );
            }
        }
        
        // Validate fusion methods
        if config.fusion_methods.is_empty() {
            report.add_error(
                &format!("{}.fusion_methods", base_path),
                "At least one fusion method must be specified",
                Some("Add methods like 'score', 'rank', 'adaptive'"),
            );
        }
        
        let valid_fusion_methods = [
            "score", "rank", "hybrid", "weighted", "layered", "adaptive"
        ];
        
        for method in &config.fusion_methods {
            if !valid_fusion_methods.contains(&method.as_str()) {
                report.add_error(
                    &format!("{}.fusion_methods", base_path),
                    &format!("Unknown fusion method: {}", method),
                    Some("Use one of the supported methods"),
                );
            }
        }
        
        // Validate wavelet configuration
        self.validate_wavelet_config(&config.wavelet, &format!("{}.wavelet", base_path), report);
        
        // Validate entropy configuration
        self.validate_entropy_config(&config.entropy, &format!("{}.entropy", base_path), report);
        
        // Validate statistics configuration
        self.validate_statistics_config(&config.statistics, &format!("{}.statistics", base_path), report);
        
        // Validate pattern detection configuration
        self.validate_pattern_detection_config(&config.pattern_detection, &format!("{}.pattern_detection", base_path), report);
    }
    
    /// Validate wavelet configuration
    fn validate_wavelet_config(&self, config: &crate::config::WaveletConfig, base_path: &str, report: &mut ValidationReport) {
        let valid_wavelets = ["haar", "daubechies", "biorthogonal", "coiflets", "dmey"];
        
        if !valid_wavelets.contains(&config.wavelet_type.as_str()) {
            report.add_error(
                &format!("{}.wavelet_type", base_path),
                &format!("Unknown wavelet type: {}", config.wavelet_type),
                Some("Use one of: haar, daubechies, biorthogonal, coiflets, dmey"),
            );
        }
        
        if config.decomposition_levels == 0 {
            report.add_error(
                &format!("{}.decomposition_levels", base_path),
                "Decomposition levels must be positive",
                Some("Use a value between 1 and 10"),
            );
        }
        
        if config.decomposition_levels > 20 {
            report.add_warning(
                &format!("{}.decomposition_levels", base_path),
                "Very high decomposition levels may be unstable",
                Some("Consider using fewer levels"),
            );
        }
    }
    
    /// Validate entropy configuration
    fn validate_entropy_config(&self, config: &crate::config::EntropyConfig, base_path: &str, report: &mut ValidationReport) {
        // Sample entropy validation
        if config.sample_entropy_m == 0 {
            report.add_error(
                &format!("{}.sample_entropy_m", base_path),
                "Sample entropy m parameter must be positive",
                Some("Use a value like 2"),
            );
        }
        
        if config.sample_entropy_r <= 0.0 {
            report.add_error(
                &format!("{}.sample_entropy_r", base_path),
                "Sample entropy r parameter must be positive",
                Some("Use a value like 0.2"),
            );
        }
        
        // Approximate entropy validation
        if config.approximate_entropy_m == 0 {
            report.add_error(
                &format!("{}.approximate_entropy_m", base_path),
                "Approximate entropy m parameter must be positive",
                Some("Use a value like 2"),
            );
        }
        
        if config.approximate_entropy_r <= 0.0 {
            report.add_error(
                &format!("{}.approximate_entropy_r", base_path),
                "Approximate entropy r parameter must be positive",
                Some("Use a value like 0.2"),
            );
        }
        
        // Permutation entropy validation
        if config.permutation_entropy_order < 3 {
            report.add_warning(
                &format!("{}.permutation_entropy_order", base_path),
                "Low permutation entropy order may be unreliable",
                Some("Use a value of 3 or higher"),
            );
        }
        
        if config.permutation_entropy_order > 8 {
            report.add_warning(
                &format!("{}.permutation_entropy_order", base_path),
                "High permutation entropy order requires large datasets",
                None,
            );
        }
    }
    
    /// Validate statistics configuration
    fn validate_statistics_config(&self, config: &crate::config::StatisticsConfig, base_path: &str, report: &mut ValidationReport) {
        // Significance threshold validation
        if config.significance_threshold <= 0.0 || config.significance_threshold >= 1.0 {
            report.add_error(
                &format!("{}.significance_threshold", base_path),
                "Significance threshold must be between 0 and 1",
                Some("Use a value like 0.05"),
            );
        }
        
        // Bootstrap iterations validation
        if config.bootstrap_iterations == 0 {
            report.add_error(
                &format!("{}.bootstrap_iterations", base_path),
                "Bootstrap iterations must be positive",
                Some("Use a value like 1000"),
            );
        }
        
        if config.bootstrap_iterations < 100 {
            report.add_warning(
                &format!("{}.bootstrap_iterations", base_path),
                "Low bootstrap iterations may give unreliable confidence intervals",
                Some("Use at least 1000 iterations"),
            );
        }
        
        if config.bootstrap_iterations > 100_000 {
            report.add_warning(
                &format!("{}.bootstrap_iterations", base_path),
                "High bootstrap iterations may be slow",
                None,
            );
        }
    }
    
    /// Validate pattern detection configuration
    fn validate_pattern_detection_config(&self, config: &crate::config::PatternDetectionConfig, base_path: &str, report: &mut ValidationReport) {
        // Pattern length validation
        if config.min_pattern_length == 0 {
            report.add_error(
                &format!("{}.min_pattern_length", base_path),
                "Minimum pattern length must be positive",
                Some("Use a value like 5"),
            );
        }
        
        if config.max_pattern_length <= config.min_pattern_length {
            report.add_error(
                &format!("{}.max_pattern_length", base_path),
                "Maximum pattern length must be greater than minimum",
                Some("Ensure max_pattern_length > min_pattern_length"),
            );
        }
        
        // Confidence threshold validation
        if config.confidence_threshold <= 0.0 || config.confidence_threshold > 1.0 {
            report.add_error(
                &format!("{}.confidence_threshold", base_path),
                "Confidence threshold must be between 0 and 1",
                Some("Use a value like 0.7"),
            );
        }
    }
    
    /// Validate performance configuration
    fn validate_performance_config(&self, config: &PerformanceConfig, report: &mut ValidationReport) {
        let base_path = "performance";
        
        // Cache size validation
        if config.cache_size_mb > 10_000 {
            report.add_warning(
                &format!("{}.cache_size_mb", base_path),
                "Very large cache size may consume excessive memory",
                None,
            );
        }
        
        // Optimization level validation
        if config.optimization_level > 3 {
            report.add_error(
                &format!("{}.optimization_level", base_path),
                "Optimization level must be between 0 and 3",
                Some("Use 0 (none), 1 (basic), 2 (moderate), or 3 (aggressive)"),
            );
        }
    }
    
    /// Validate ML configuration
    fn validate_ml_config(&self, config: &MlConfig, report: &mut ValidationReport) {
        let base_path = "ml";
        
        if config.enable_ml_processing {
            // Validate training configuration
            let training_path = format!("{}.training", base_path);
            
            if config.training.learning_rate <= 0.0 {
                report.add_error(
                    &format!("{}.learning_rate", training_path),
                    "Learning rate must be positive",
                    Some("Use a value like 0.001"),
                );
            }
            
            if config.training.learning_rate > 1.0 {
                report.add_warning(
                    &format!("{}.learning_rate", training_path),
                    "High learning rate may cause training instability",
                    Some("Consider using a smaller value"),
                );
            }
            
            if config.training.epochs == 0 {
                report.add_error(
                    &format!("{}.epochs", training_path),
                    "Number of epochs must be positive",
                    Some("Use a value like 100"),
                );
            }
            
            if config.training.batch_size == 0 {
                report.add_error(
                    &format!("{}.batch_size", training_path),
                    "Batch size must be positive",
                    Some("Use a value like 32"),
                );
            }
            
            // Validate neural network configuration
            let nn_path = format!("{}.neural_network", base_path);
            
            if config.neural_network.hidden_layers.is_empty() {
                report.add_warning(
                    &format!("{}.hidden_layers", nn_path),
                    "No hidden layers specified",
                    Some("Consider adding hidden layers for better learning capacity"),
                );
            }
            
            for (i, &size) in config.neural_network.hidden_layers.iter().enumerate() {
                if size == 0 {
                    report.add_error(
                        &format!("{}.hidden_layers[{}]", nn_path, i),
                        "Hidden layer size must be positive",
                        Some("Use a positive number of neurons"),
                    );
                }
            }
            
            if config.neural_network.dropout_rate < 0.0 || config.neural_network.dropout_rate >= 1.0 {
                report.add_error(
                    &format!("{}.dropout_rate", nn_path),
                    "Dropout rate must be between 0 and 1",
                    Some("Use a value like 0.2"),
                );
            }
        }
    }
    
    /// Validate data configuration
    fn validate_data_config(&self, config: &crate::config::DataConfig, report: &mut ValidationReport) {
        let base_path = "data";
        
        // Validate quality thresholds
        let qt = &config.quality_thresholds;
        let qt_path = format!("{}.quality_thresholds", base_path);
        
        if qt.max_missing_percentage < 0.0 || qt.max_missing_percentage > 100.0 {
            report.add_error(
                &format!("{}.max_missing_percentage", qt_path),
                "Missing percentage must be between 0 and 100",
                Some("Use a value like 10.0"),
            );
        }
        
        if qt.min_data_points == 0 {
            report.add_error(
                &format!("{}.min_data_points", qt_path),
                "Minimum data points must be positive",
                Some("Use a value like 100"),
            );
        }
        
        if qt.max_outlier_percentage < 0.0 || qt.max_outlier_percentage > 100.0 {
            report.add_error(
                &format!("{}.max_outlier_percentage", qt_path),
                "Outlier percentage must be between 0 and 100",
                Some("Use a value like 5.0"),
            );
        }
    }
    
    /// Validate analysis configuration
    fn validate_analysis_config(&self, config: &crate::config::AnalysisConfig, report: &mut ValidationReport) {
        let base_path = "analysis";
        
        if config.time_window_size == 0 {
            report.add_error(
                &format!("{}.time_window_size", base_path),
                "Time window size must be positive",
                Some("Use a value like 252"),
            );
        }
        
        if config.rolling_step_size == 0 {
            report.add_error(
                &format!("{}.rolling_step_size", base_path),
                "Rolling step size must be positive",
                Some("Use a value like 1"),
            );
        }
        
        if config.rolling_step_size > config.time_window_size {
            report.add_warning(
                &format!("{}.rolling_step_size", base_path),
                "Rolling step size larger than window size",
                Some("Consider using a smaller step size"),
            );
        }
        
        if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
            report.add_error(
                &format!("{}.confidence_level", base_path),
                "Confidence level must be between 0 and 1",
                Some("Use a value like 0.95"),
            );
        }
    }
    
    /// Validate visualization configuration
    fn validate_visualization_config(&self, config: &crate::config::VisualizationConfig, report: &mut ValidationReport) {
        let base_path = "visualization";
        
        if config.export_dpi == 0 {
            report.add_error(
                &format!("{}.export_dpi", base_path),
                "Export DPI must be positive",
                Some("Use a value like 300"),
            );
        }
        
        if config.export_dpi > 1200 {
            report.add_warning(
                &format!("{}.export_dpi", base_path),
                "Very high DPI may create large files",
                None,
            );
        }
    }
    
    /// Validate Redis configuration
    fn validate_redis_config(&self, config: &crate::config::RedisConfig, report: &mut ValidationReport) {
        let base_path = "redis";
        
        if config.enabled {
            if config.host.is_empty() {
                report.add_error(
                    &format!("{}.host", base_path),
                    "Redis host cannot be empty when enabled",
                    Some("Set a valid hostname or IP address"),
                );
            }
            
            if config.port == 0 {
                report.add_error(
                    &format!("{}.port", base_path),
                    "Redis port must be positive",
                    Some("Use the default port 6379"),
                );
            }
            
            if config.pool_size == 0 {
                report.add_error(
                    &format!("{}.pool_size", base_path),
                    "Connection pool size must be positive",
                    Some("Use a value like 10"),
                );
            }
            
            if config.timeout_ms == 0 {
                report.add_warning(
                    &format!("{}.timeout_ms", base_path),
                    "Zero timeout may cause indefinite blocking",
                    Some("Set a reasonable timeout like 5000ms"),
                );
            }
        }
    }
    
    /// Validate validation configuration
    fn validate_validation_config(&self, config: &crate::config::ValidationConfig, report: &mut ValidationReport) {
        // Validation of validation config - meta! 
        // For now, just check that thresholds are reasonable
        for (key, value) in &config.warning_thresholds {
            if *value < 0.0 {
                report.add_warning(
                    &format!("validation.warning_thresholds.{}", key),
                    "Negative threshold value",
                    None,
                );
            }
        }
        
        for (key, value) in &config.error_thresholds {
            if *value < 0.0 {
                report.add_warning(
                    &format!("validation.error_thresholds.{}", key),
                    "Negative threshold value", 
                    None,
                );
            }
        }
    }
    
    /// Validate logging configuration
    fn validate_logging_config(&self, config: &crate::config::LoggingConfig, report: &mut ValidationReport) {
        let base_path = "logging";
        
        let valid_levels = ["trace", "debug", "info", "warn", "error", "off"];
        if !valid_levels.contains(&config.level.as_str()) {
            report.add_error(
                &format!("{}.level", base_path),
                &format!("Invalid log level: {}", config.level),
                Some("Use one of: trace, debug, info, warn, error, off"),
            );
        }
        
        if config.enable_file_logging && config.log_file_path.is_empty() {
            report.add_error(
                &format!("{}.log_file_path", base_path),
                "Log file path cannot be empty when file logging is enabled",
                Some("Set a valid file path"),
            );
        }
        
        if config.enable_metrics && config.metrics_interval_seconds == 0 {
            report.add_warning(
                &format!("{}.metrics_interval_seconds", base_path),
                "Zero metrics interval may cause high overhead",
                Some("Use a reasonable interval like 60 seconds"),
            );
        }
    }
    
    /// Validate advanced configuration
    fn validate_advanced_config(&self, config: &crate::config::AdvancedConfig, report: &mut ValidationReport) {
        let base_path = "advanced";
        
        if config.enable_neuromorphic {
            let stdp_path = format!("{}.stdp_optimization", base_path);
            
            if config.stdp_optimization.learning_rate <= 0.0 {
                report.add_error(
                    &format!("{}.learning_rate", stdp_path),
                    "STDP learning rate must be positive",
                    Some("Use a value like 0.01"),
                );
            }
            
            if config.stdp_optimization.tau_positive <= 0.0 {
                report.add_error(
                    &format!("{}.tau_positive", stdp_path),
                    "STDP tau_positive must be positive",
                    Some("Use a value like 20.0"),
                );
            }
            
            if config.stdp_optimization.tau_negative <= 0.0 {
                report.add_error(
                    &format!("{}.tau_negative", stdp_path),
                    "STDP tau_negative must be positive",
                    Some("Use a value like 20.0"),
                );
            }
            
            if config.stdp_optimization.max_weight_change <= 0.0 {
                report.add_error(
                    &format!("{}.max_weight_change", stdp_path),
                    "STDP max_weight_change must be positive",
                    Some("Use a value like 0.1"),
                );
            }
        }
    }
    
    /// Validate hardware configuration
    fn validate_hardware_config(&self, config: &crate::config::HardwareSpecificConfig, report: &mut ValidationReport) {
        let base_path = "hardware";
        
        // Check for conflicting optimizations
        if config.cpu_optimizations.contains(&"avx512".to_string()) && 
           config.cpu_optimizations.contains(&"sse2".to_string()) {
            report.add_info(
                &format!("{}.cpu_optimizations", base_path),
                "Both AVX512 and SSE2 specified - AVX512 will take precedence",
            );
        }
        
        // Validate memory strategy
        let valid_strategies = ["adaptive", "conservative", "aggressive"];
        if !valid_strategies.contains(&config.memory_strategy.as_str()) {
            report.add_warning(
                &format!("{}.memory_strategy", base_path),
                &format!("Unknown memory strategy: {}", config.memory_strategy),
                Some("Use one of: adaptive, conservative, aggressive"),
            );
        }
    }
    
    /// Validate cross-section consistency
    fn validate_consistency(&self, config: &CdfaConfig, report: &mut ValidationReport) {
        // GPU consistency
        if config.processing.enable_gpu && !config.hardware.gpu_device_ids.is_empty() &&
           config.hardware.gpu_device_ids.len() > 1 && config.processing.num_threads == 1 {
            report.add_warning(
                "processing.num_threads",
                "Single thread with multiple GPUs may be inefficient",
                Some("Consider using more threads for multi-GPU workloads"),
            );
        }
        
        // ML and performance consistency
        if config.ml.enable_ml_processing && !config.performance.enable_caching {
            report.add_info(
                "performance.enable_caching",
                "ML processing without caching may be slower",
            );
        }
        
        // Memory consistency
        let total_memory_estimate = config.performance.cache_size_mb + 
                                   config.processing.memory_limit_mb;
        if total_memory_estimate > 32_000 { // 32GB
            report.add_warning(
                "memory_usage",
                "Total memory configuration exceeds 32GB",
                Some("Ensure system has sufficient memory"),
            );
        }
        
        // Redis and distributed consistency
        if config.processing.enable_distributed && !config.redis.enabled {
            report.add_warning(
                "redis.enabled",
                "Distributed processing without Redis may have limited coordination",
                Some("Consider enabling Redis for better coordination"),
            );
        }
        
        // Real-time and batch consistency
        if config.data.enable_realtime && config.processing.batch_size > 1000 {
            report.add_warning(
                "processing.batch_size",
                "Large batch size may conflict with real-time processing",
                Some("Consider using smaller batches for real-time mode"),
            );
        }
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to validate a configuration
pub fn validate_config(config: &CdfaConfig) -> Result<()> {
    let validator = ConfigValidator::new();
    let report = validator.validate(config);
    
    if !report.is_valid {
        let errors: Vec<String> = report.get_errors()
            .iter()
            .map(|issue| format!("{}: {}", issue.parameter_path, issue.message))
            .collect();
        
        return Err(CdfaError::config_error(format!(
            "Configuration validation failed with {} errors: {}",
            report.errors,
            errors.join("; ")
        )));
    }
    
    Ok(())
}

/// Convenience function to validate with detailed reporting
pub fn validate_config_detailed(config: &CdfaConfig) -> ValidationReport {
    let validator = ConfigValidator::new();
    validator.validate(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CdfaConfig;
    
    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();
        assert!(report.is_valid);
        assert_eq!(report.errors, 0);
        
        report.add_error("test.param", "Test error", Some("Fix it"));
        assert!(!report.is_valid);
        assert_eq!(report.errors, 1);
        
        report.add_warning("test.param2", "Test warning", None);
        assert_eq!(report.warnings, 1);
    }
    
    #[test]
    fn test_valid_config() {
        let config = CdfaConfig::default();
        let result = validate_config(&config);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_invalid_config() {
        let mut config = CdfaConfig::default();
        config.processing.tolerance = -1.0; // Invalid negative tolerance
        
        let result = validate_config(&config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_custom_validator() {
        let mut validator = ConfigValidator::new();
        
        validator.add_custom_validator(
            "test_validator".to_string(),
            |config| {
                let mut issues = Vec::new();
                if config.processing.num_threads > 100 {
                    issues.push(ValidationIssue {
                        parameter_path: "processing.num_threads".to_string(),
                        severity: ValidationSeverity::Warning,
                        message: "Too many threads".to_string(),
                        suggestion: Some("Use fewer threads".to_string()),
                    });
                }
                issues
            },
        );
        
        let mut config = CdfaConfig::default();
        config.processing.num_threads = 200;
        
        let report = validator.validate(&config);
        assert!(report.warnings > 0);
    }
    
    #[test]
    fn test_detailed_validation() {
        let config = CdfaConfig::default();
        let report = validate_config_detailed(&config);
        
        // Should have some info messages but be valid
        assert!(report.is_valid);
        // Default config might generate some info/warning messages
    }
}