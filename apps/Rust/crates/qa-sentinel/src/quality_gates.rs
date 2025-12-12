//! Quality Gates Implementation
//!
//! This module implements automated quality gates that enforce code quality standards
//! across the entire TENGRI trading system.

use crate::config::{QaSentinelConfig, QualityThresholds};
use crate::coverage::CoverageReport;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn, error};

/// Quality report containing all test results and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub unit_tests: TestResults,
    pub integration_tests: TestResults,
    pub property_tests: TestResults,
    pub performance_tests: TestResults,
    pub chaos_tests: TestResults,
    pub security_tests: TestResults,
    pub coverage: CoverageReport,
    pub quality_metrics: QualityMetrics,
    pub quality_score: f64,
    pub gate_results: HashMap<String, GateResult>,
}

/// Test results for a specific test category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub category: String,
    pub total_tests: u64,
    pub passed_tests: u64,
    pub failed_tests: u64,
    pub skipped_tests: u64,
    pub test_duration: Duration,
    pub individual_results: Vec<IndividualTestResult>,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualTestResult {
    pub test_name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error_message: Option<String>,
    pub metrics: TestMetrics,
}

/// Test metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub assertions_count: u64,
    pub coverage_delta: f64,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub cyclomatic_complexity: f64,
    pub code_duplication_percent: f64,
    pub technical_debt_hours: f64,
    pub maintainability_index: f64,
    pub security_score: f64,
    pub performance_score: f64,
}

/// Quality gate result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate_name: String,
    pub passed: bool,
    pub threshold: f64,
    pub actual_value: f64,
    pub severity: GateSeverity,
    pub message: String,
}

/// Gate severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Quality gate validator
pub struct QualityGateValidator {
    config: QaSentinelConfig,
    thresholds: QualityThresholds,
}

impl QualityReport {
    /// Create a new quality report
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            unit_tests: TestResults::new(),
            integration_tests: TestResults::new(),
            property_tests: TestResults::new(),
            performance_tests: TestResults::new(),
            chaos_tests: TestResults::new(),
            security_tests: TestResults::new(),
            coverage: CoverageReport::new(),
            quality_metrics: QualityMetrics::default(),
            quality_score: 0.0,
            gate_results: HashMap::new(),
        }
    }
    
    /// Add unit test results
    pub fn add_unit_test_results(&mut self, results: TestResults) {
        self.unit_tests = results;
    }
    
    /// Add integration test results
    pub fn add_integration_test_results(&mut self, results: TestResults) {
        self.integration_tests = results;
    }
    
    /// Add property test results
    pub fn add_property_test_results(&mut self, results: TestResults) {
        self.property_tests = results;
    }
    
    /// Add performance test results
    pub fn add_performance_test_results(&mut self, results: TestResults) {
        self.performance_tests = results;
    }
    
    /// Add chaos test results
    pub fn add_chaos_test_results(&mut self, results: TestResults) {
        self.chaos_tests = results;
    }
    
    /// Add security test results
    pub fn add_security_test_results(&mut self, results: TestResults) {
        self.security_tests = results;
    }
    
    /// Set coverage report
    pub fn set_coverage(&mut self, coverage: CoverageReport) {
        self.coverage = coverage;
    }
    
    /// Get total number of tests
    pub fn total_tests(&self) -> u64 {
        self.unit_tests.total_tests +
        self.integration_tests.total_tests +
        self.property_tests.total_tests +
        self.performance_tests.total_tests +
        self.chaos_tests.total_tests +
        self.security_tests.total_tests
    }
    
    /// Get total passed tests
    pub fn passed_tests(&self) -> u64 {
        self.unit_tests.passed_tests +
        self.integration_tests.passed_tests +
        self.property_tests.passed_tests +
        self.performance_tests.passed_tests +
        self.chaos_tests.passed_tests +
        self.security_tests.passed_tests
    }
    
    /// Get total failed tests
    pub fn failed_tests(&self) -> u64 {
        self.unit_tests.failed_tests +
        self.integration_tests.failed_tests +
        self.property_tests.failed_tests +
        self.performance_tests.failed_tests +
        self.chaos_tests.failed_tests +
        self.security_tests.failed_tests
    }
    
    /// Get coverage report
    pub fn coverage(&self) -> &CoverageReport {
        &self.coverage
    }
    
    /// Get quality score
    pub fn quality_score(&self) -> f64 {
        self.quality_score
    }
    
    /// Calculate overall quality score
    pub fn calculate_quality_score(&mut self) {
        let test_pass_rate = if self.total_tests() > 0 {
            (self.passed_tests() as f64 / self.total_tests() as f64) * 100.0
        } else {
            0.0
        };
        
        let coverage_score = self.coverage.line_coverage;
        let complexity_score = (100.0 - self.quality_metrics.cyclomatic_complexity).max(0.0);
        let duplication_score = (100.0 - self.quality_metrics.code_duplication_percent).max(0.0);
        let security_score = self.quality_metrics.security_score;
        let performance_score = self.quality_metrics.performance_score;
        
        // Weighted average
        self.quality_score = (
            test_pass_rate * 0.3 +
            coverage_score * 0.25 +
            complexity_score * 0.15 +
            duplication_score * 0.10 +
            security_score * 0.10 +
            performance_score * 0.10
        );
    }
    
    /// Check if all quality gates passed
    pub fn all_gates_passed(&self) -> bool {
        self.gate_results.values().all(|result| result.passed)
    }
    
    /// Get failed gates
    pub fn failed_gates(&self) -> Vec<&GateResult> {
        self.gate_results.values().filter(|result| !result.passed).collect()
    }
    
    /// Get critical failed gates
    pub fn critical_failed_gates(&self) -> Vec<&GateResult> {
        self.gate_results.values()
            .filter(|result| !result.passed && matches!(result.severity, GateSeverity::Critical))
            .collect()
    }
}

impl TestResults {
    /// Create new test results
    pub fn new() -> Self {
        Self {
            category: String::new(),
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            skipped_tests: 0,
            test_duration: Duration::ZERO,
            individual_results: Vec::new(),
        }
    }
    
    /// Add a test result
    pub fn add_result(&mut self, result: crate::zero_mock::TestResult) {
        let individual_result = IndividualTestResult {
            test_name: result.test_name,
            passed: result.passed,
            duration: result.duration,
            error_message: result.error,
            metrics: TestMetrics {
                memory_usage_mb: result.metrics.memory_usage_mb,
                cpu_usage_percent: result.metrics.cpu_usage_percent,
                assertions_count: 0,
                coverage_delta: 0.0,
            },
        };
        
        self.individual_results.push(individual_result);
        self.total_tests += 1;
        
        if result.passed {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
        
        self.test_duration += result.duration;
    }
    
    /// Merge another test results
    pub fn merge(&mut self, other: TestResults) {
        self.total_tests += other.total_tests;
        self.passed_tests += other.passed_tests;
        self.failed_tests += other.failed_tests;
        self.skipped_tests += other.skipped_tests;
        self.test_duration += other.test_duration;
        self.individual_results.extend(other.individual_results);
    }
    
    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests > 0 {
            (self.passed_tests as f64 / self.total_tests as f64) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get passed count
    pub fn passed_count(&self) -> u64 {
        self.passed_tests
    }
    
    /// Get failed count
    pub fn failed_count(&self) -> u64 {
        self.failed_tests
    }
}

impl QualityGateValidator {
    /// Create a new quality gate validator
    pub fn new(config: QaSentinelConfig) -> Self {
        let thresholds = config.quality_gates.thresholds.clone();
        Self {
            config,
            thresholds,
        }
    }
    
    /// Validate all quality gates
    pub fn validate(&self, report: &mut QualityReport) -> Result<()> {
        info!("🚪 Validating quality gates");
        
        // Coverage gate
        self.validate_coverage_gate(report)?;
        
        // Test pass rate gate
        self.validate_test_pass_rate_gate(report)?;
        
        // Complexity gate
        self.validate_complexity_gate(report)?;
        
        // Duplication gate
        self.validate_duplication_gate(report)?;
        
        // Security gate
        self.validate_security_gate(report)?;
        
        // Performance gate
        self.validate_performance_gate(report)?;
        
        // Calculate overall quality score
        report.calculate_quality_score();
        
        // Log results
        self.log_gate_results(report);
        
        Ok(())
    }
    
    fn validate_coverage_gate(&self, report: &mut QualityReport) -> Result<()> {
        let coverage = report.coverage.line_coverage;
        let threshold = self.thresholds.min_code_coverage;
        
        let passed = coverage >= threshold;
        let gate_result = GateResult {
            gate_name: "code_coverage".to_string(),
            passed,
            threshold,
            actual_value: coverage,
            severity: GateSeverity::Critical,
            message: format!("Code coverage: {:.2}% (threshold: {:.2}%)", coverage, threshold),
        };
        
        report.gate_results.insert("code_coverage".to_string(), gate_result);
        Ok(())
    }
    
    fn validate_test_pass_rate_gate(&self, report: &mut QualityReport) -> Result<()> {
        let pass_rate = if report.total_tests() > 0 {
            (report.passed_tests() as f64 / report.total_tests() as f64) * 100.0
        } else {
            0.0
        };
        
        let threshold = self.thresholds.min_test_pass_rate;
        let passed = pass_rate >= threshold;
        
        let gate_result = GateResult {
            gate_name: "test_pass_rate".to_string(),
            passed,
            threshold,
            actual_value: pass_rate,
            severity: GateSeverity::Critical,
            message: format!("Test pass rate: {:.2}% (threshold: {:.2}%)", pass_rate, threshold),
        };
        
        report.gate_results.insert("test_pass_rate".to_string(), gate_result);
        Ok(())
    }
    
    fn validate_complexity_gate(&self, report: &mut QualityReport) -> Result<()> {
        let complexity = report.quality_metrics.cyclomatic_complexity;
        let threshold = self.thresholds.max_cyclomatic_complexity as f64;
        
        let passed = complexity <= threshold;
        let gate_result = GateResult {
            gate_name: "cyclomatic_complexity".to_string(),
            passed,
            threshold,
            actual_value: complexity,
            severity: GateSeverity::High,
            message: format!("Cyclomatic complexity: {:.2} (threshold: {:.2})", complexity, threshold),
        };
        
        report.gate_results.insert("cyclomatic_complexity".to_string(), gate_result);
        Ok(())
    }
    
    fn validate_duplication_gate(&self, report: &mut QualityReport) -> Result<()> {
        let duplication = report.quality_metrics.code_duplication_percent;
        let threshold = self.thresholds.max_code_duplication;
        
        let passed = duplication <= threshold;
        let gate_result = GateResult {
            gate_name: "code_duplication".to_string(),
            passed,
            threshold,
            actual_value: duplication,
            severity: GateSeverity::Medium,
            message: format!("Code duplication: {:.2}% (threshold: {:.2}%)", duplication, threshold),
        };
        
        report.gate_results.insert("code_duplication".to_string(), gate_result);
        Ok(())
    }
    
    fn validate_security_gate(&self, report: &mut QualityReport) -> Result<()> {
        let security_score = report.quality_metrics.security_score;
        let threshold = 95.0; // High security threshold
        
        let passed = security_score >= threshold;
        let gate_result = GateResult {
            gate_name: "security_score".to_string(),
            passed,
            threshold,
            actual_value: security_score,
            severity: GateSeverity::Critical,
            message: format!("Security score: {:.2} (threshold: {:.2})", security_score, threshold),
        };
        
        report.gate_results.insert("security_score".to_string(), gate_result);
        Ok(())
    }
    
    fn validate_performance_gate(&self, report: &mut QualityReport) -> Result<()> {
        let performance_score = report.quality_metrics.performance_score;
        let threshold = 90.0; // High performance threshold
        
        let passed = performance_score >= threshold;
        let gate_result = GateResult {
            gate_name: "performance_score".to_string(),
            passed,
            threshold,
            actual_value: performance_score,
            severity: GateSeverity::High,
            message: format!("Performance score: {:.2} (threshold: {:.2})", performance_score, threshold),
        };
        
        report.gate_results.insert("performance_score".to_string(), gate_result);
        Ok(())
    }
    
    fn log_gate_results(&self, report: &QualityReport) {
        info!("📊 Quality Gate Results:");
        
        for (gate_name, result) in &report.gate_results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            let severity = match result.severity {
                GateSeverity::Critical => "🔴 CRITICAL",
                GateSeverity::High => "🟠 HIGH",
                GateSeverity::Medium => "🟡 MEDIUM",
                GateSeverity::Low => "🟢 LOW",
                GateSeverity::Info => "ℹ️ INFO",
            };
            
            info!("  {} {} [{}] {}", status, gate_name, severity, result.message);
        }
        
        let overall_status = if report.all_gates_passed() {
            "✅ ALL GATES PASSED"
        } else {
            "❌ SOME GATES FAILED"
        };
        
        info!("🎯 Overall Quality Score: {:.2}/100", report.quality_score);
        info!("🚪 {}", overall_status);
        
        // Log critical failures
        let critical_failures = report.critical_failed_gates();
        if !critical_failures.is_empty() {
            error!("🚨 CRITICAL QUALITY GATE FAILURES:");
            for failure in critical_failures {
                error!("  - {}: {}", failure.gate_name, failure.message);
            }
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic_complexity: 0.0,
            code_duplication_percent: 0.0,
            technical_debt_hours: 0.0,
            maintainability_index: 100.0,
            security_score: 100.0,
            performance_score: 100.0,
        }
    }
}

impl Default for TestMetrics {
    fn default() -> Self {
        Self {
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            assertions_count: 0,
            coverage_delta: 0.0,
        }
    }
}

/// Initialize quality gates
pub async fn initialize_quality_gates(config: &QaSentinelConfig) -> Result<()> {
    info!("🚪 Initializing quality gates");
    
    // Validate configuration
    if !config.quality_gates.enabled {
        warn!("Quality gates are disabled in configuration");
        return Ok(());
    }
    
    // Log configured thresholds
    let thresholds = &config.quality_gates.thresholds;
    info!("📊 Quality Gate Thresholds:");
    info!("  - Code Coverage: {:.2}%", thresholds.min_code_coverage);
    info!("  - Test Pass Rate: {:.2}%", thresholds.min_test_pass_rate);
    info!("  - Max Complexity: {}", thresholds.max_cyclomatic_complexity);
    info!("  - Max Duplication: {:.2}%", thresholds.max_code_duplication);
    info!("  - Max Security Vulnerabilities: {}", thresholds.max_security_vulnerabilities);
    
    Ok(())
}

/// Validate quality gates for a report
pub async fn validate_quality_gates(report: &mut QualityReport, config: &QaSentinelConfig) -> Result<()> {
    let validator = QualityGateValidator::new(config.clone());
    validator.validate(report)?;
    
    // Check if blocking failures occurred
    let blocking_failures = report.gate_results.values()
        .filter(|result| {
            !result.passed && config.quality_gates.blocking_failures.iter().any(|gate_type| {
                match gate_type {
                    crate::config::QualityGateType::Coverage => result.gate_name == "code_coverage",
                    crate::config::QualityGateType::Security => result.gate_name == "security_score",
                    crate::config::QualityGateType::Tests => result.gate_name == "test_pass_rate",
                    crate::config::QualityGateType::Complexity => result.gate_name == "cyclomatic_complexity",
                    crate::config::QualityGateType::Duplication => result.gate_name == "code_duplication",
                    crate::config::QualityGateType::Performance => result.gate_name == "performance_score",
                }
            })
        })
        .collect::<Vec<_>>();
    
    if !blocking_failures.is_empty() {
        error!("🚨 BLOCKING QUALITY GATE FAILURES DETECTED:");
        for failure in &blocking_failures {
            error!("  - {}: {}", failure.gate_name, failure.message);
        }
        return Err(anyhow::anyhow!("Quality gates validation failed with {} blocking failures", blocking_failures.len()));
    }
    
    Ok(())
}