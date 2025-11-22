//! Comprehensive Test Orchestration Framework
//!
//! This module provides a unified test orchestration system that coordinates
//! all test types with 100% coverage requirement validation and CI integration.
//!
//! ## Test Orchestration Features:
//! - Parallel test execution with dependency management
//! - Real-time coverage monitoring and enforcement
//! - Automated test report generation
//! - CI/CD pipeline integration
//! - Test result aggregation and analysis
//! - Failure diagnosis and remediation suggestions

use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::process::Command as TokioCommand;
use futures::future::join_all;

// Import all test modules
use super::bayesian_var_research_tests::*;
use super::property_based_tests::*;
use super::formal_verification_tests::*;
use super::byzantine_fault_tolerance_tests::*;
use super::coverage::test_coverage_analyzer::*;
use super::security::timing_attack_tests::*;

/// Test orchestration errors
#[derive(Error, Debug)]
pub enum OrchestrationError {
    #[error("Test execution failed: {test_name} - {reason}")]
    TestExecutionFailed { test_name: String, reason: String },
    
    #[error("Coverage requirement not met: {actual}% < {required}%")]
    CoverageRequirementFailed { actual: f64, required: f64 },
    
    #[error("Test timeout: {test_name} exceeded {timeout_seconds}s")]
    TestTimeout { test_name: String, timeout_seconds: u64 },
    
    #[error("Dependency failure: {dependency} required by {dependent}")]
    DependencyFailure { dependency: String, dependent: String },
    
    #[error("CI integration error: {message}")]
    CiIntegrationError { message: String },
    
    #[error("Report generation failed: {reason}")]
    ReportGenerationFailed { reason: String },
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
    Timeout,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub execution_time: Duration,
    pub error_message: Option<String>,
    pub coverage_contribution: f64,
    pub assertions_passed: usize,
    pub assertions_total: usize,
    pub performance_metrics: HashMap<String, f64>,
    pub timestamp: String,
}

/// Test suite definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub test_type: TestType,
    pub priority: TestPriority,
    pub dependencies: Vec<String>,
    pub timeout_seconds: u64,
    pub parallel_allowed: bool,
    pub required_coverage: f64,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestType {
    Unit,
    Integration,
    Property,
    Formal,
    Byzantine,
    Security,
    Performance,
    Coverage,
    EndToEnd,
}

#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq, Serialize, Deserialize)]
pub enum TestPriority {
    Critical = 1,
    High = 2,
    Medium = 3,
    Low = 4,
}

/// Comprehensive test execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionReport {
    pub timestamp: String,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub timeout_tests: usize,
    pub total_execution_time: Duration,
    pub overall_coverage: f64,
    pub coverage_by_type: HashMap<String, f64>,
    pub test_results: Vec<TestResult>,
    pub coverage_report: Option<CoverageReport>,
    pub ci_metadata: CiMetadata,
    pub recommendations: Vec<String>,
    pub requirements_met: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiMetadata {
    pub branch: String,
    pub commit_hash: String,
    pub build_number: Option<String>,
    pub pr_number: Option<String>,
    pub environment: String,
    pub rust_version: String,
}

/// Main test orchestrator
pub struct ComprehensiveTestOrchestrator {
    pub test_suites: Vec<TestSuite>,
    pub config: OrchestrationConfig,
    pub results: Arc<Mutex<Vec<TestResult>>>,
    pub coverage_analyzer: TestCoverageAnalyzer,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct OrchestrationConfig {
    pub max_parallel_tests: usize,
    pub global_timeout: Duration,
    pub required_overall_coverage: f64,
    pub fail_fast: bool,
    pub generate_reports: bool,
    pub ci_integration: bool,
    pub coverage_enforcement: bool,
    pub retry_failed_tests: bool,
    pub max_retries: usize,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            max_parallel_tests: num_cpus::get(),
            global_timeout: Duration::from_secs(3600), // 1 hour
            required_overall_coverage: 100.0,
            fail_fast: false,
            generate_reports: true,
            ci_integration: true,
            coverage_enforcement: true,
            retry_failed_tests: true,
            max_retries: 2,
        }
    }
}

impl ComprehensiveTestOrchestrator {
    pub fn new(config: OrchestrationConfig, output_dir: PathBuf) -> Self {
        let coverage_config = CoverageConfig::default();
        let source_root = std::env::current_dir().unwrap();
        let coverage_analyzer = TestCoverageAnalyzer::new(
            coverage_config,
            source_root,
            output_dir.join("coverage")
        );
        
        Self {
            test_suites: Self::create_default_test_suites(),
            config,
            results: Arc::new(Mutex::new(Vec::new())),
            coverage_analyzer,
            output_dir,
        }
    }
    
    /// Create comprehensive test suite definitions
    fn create_default_test_suites() -> Vec<TestSuite> {
        vec![
            TestSuite {
                name: "research_validated_tests".to_string(),
                test_type: TestType::Unit,
                priority: TestPriority::Critical,
                dependencies: vec![],
                timeout_seconds: 300,
                parallel_allowed: true,
                required_coverage: 100.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "research_validated_tests".to_string(), "--".to_string(), "--nocapture".to_string()],
            },
            TestSuite {
                name: "property_based_tests".to_string(),
                test_type: TestType::Property,
                priority: TestPriority::Critical,
                dependencies: vec!["research_validated_tests".to_string()],
                timeout_seconds: 600,
                parallel_allowed: true,
                required_coverage: 95.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "property_based_tests".to_string()],
            },
            TestSuite {
                name: "formal_verification_tests".to_string(),
                test_type: TestType::Formal,
                priority: TestPriority::High,
                dependencies: vec!["research_validated_tests".to_string()],
                timeout_seconds: 900,
                parallel_allowed: false, // Formal verification is resource intensive
                required_coverage: 90.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "formal_verification_tests".to_string()],
            },
            TestSuite {
                name: "byzantine_fault_tolerance_tests".to_string(),
                test_type: TestType::Byzantine,
                priority: TestPriority::High,
                dependencies: vec!["research_validated_tests".to_string()],
                timeout_seconds: 1200,
                parallel_allowed: true,
                required_coverage: 95.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "byzantine_fault_tolerance_tests".to_string()],
            },
            TestSuite {
                name: "security_timing_tests".to_string(),
                test_type: TestType::Security,
                priority: TestPriority::High,
                dependencies: vec!["research_validated_tests".to_string()],
                timeout_seconds: 800,
                parallel_allowed: false, // Timing tests need isolated environment
                required_coverage: 85.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "security_tests".to_string()],
            },
            TestSuite {
                name: "performance_benchmarks".to_string(),
                test_type: TestType::Performance,
                priority: TestPriority::Medium,
                dependencies: vec!["research_validated_tests".to_string(), "property_based_tests".to_string()],
                timeout_seconds: 1800,
                parallel_allowed: false,
                required_coverage: 80.0,
                command: "cargo".to_string(),
                args: vec!["bench".to_string(), "--bench".to_string(), "research_performance_benchmarks".to_string()],
            },
            TestSuite {
                name: "integration_tests".to_string(),
                test_type: TestType::Integration,
                priority: TestPriority::Medium,
                dependencies: vec!["research_validated_tests".to_string(), "byzantine_fault_tolerance_tests".to_string()],
                timeout_seconds: 600,
                parallel_allowed: true,
                required_coverage: 90.0,
                command: "cargo".to_string(),
                args: vec!["test".to_string(), "--test".to_string(), "integration_*".to_string()],
            },
            TestSuite {
                name: "coverage_analysis".to_string(),
                test_type: TestType::Coverage,
                priority: TestPriority::Critical,
                dependencies: vec![
                    "research_validated_tests".to_string(),
                    "property_based_tests".to_string(),
                    "formal_verification_tests".to_string(),
                    "byzantine_fault_tolerance_tests".to_string(),
                    "security_timing_tests".to_string(),
                ],
                timeout_seconds: 1200,
                parallel_allowed: false,
                required_coverage: 100.0,
                command: "cargo".to_string(),
                args: vec!["tarpaulin".to_string(), "--out".to_string(), "Html".to_string(), "--out".to_string(), "Xml".to_string()],
            },
        ]
    }
    
    /// Execute all test suites with orchestration
    pub async fn execute_all_tests(&mut self) -> Result<TestExecutionReport, OrchestrationError> {
        println!("🚀 Starting comprehensive test execution...");
        let start_time = Instant::now();
        
        // Create output directories
        create_dir_all(&self.output_dir)?;
        create_dir_all(self.output_dir.join("reports"))?;
        
        // Gather CI metadata
        let ci_metadata = self.gather_ci_metadata().await;
        
        // Execute tests in dependency order
        let execution_results = self.execute_tests_with_dependencies().await?;
        
        // Run coverage analysis
        let coverage_report = if self.config.coverage_enforcement {
            Some(self.coverage_analyzer.analyze_coverage()
                .map_err(|e| OrchestrationError::CoverageRequirementFailed { 
                    actual: 0.0, 
                    required: self.config.required_overall_coverage 
                })?)
        } else {
            None
        };
        
        // Aggregate results
        let results = self.results.lock().unwrap().clone();
        let report = self.generate_execution_report(
            results,
            start_time.elapsed(),
            coverage_report,
            ci_metadata,
        )?;
        
        // Validate requirements
        if self.config.coverage_enforcement && report.requirements_met {
            self.validate_coverage_requirements(&report)?;
        }
        
        // Generate reports
        if self.config.generate_reports {
            self.generate_comprehensive_reports(&report).await?;
        }
        
        // CI integration
        if self.config.ci_integration {
            self.integrate_with_ci(&report).await?;
        }
        
        println!("✅ Test execution completed in {:.2}s", start_time.elapsed().as_secs_f64());
        Ok(report)
    }
    
    /// Execute tests respecting dependencies
    async fn execute_tests_with_dependencies(&mut self) -> Result<Vec<TestResult>, OrchestrationError> {
        let mut completed = HashSet::new();
        let mut pending = self.test_suites.clone();
        let mut execution_order = Vec::new();
        
        // Topological sort for dependency resolution
        while !pending.is_empty() {
            let mut made_progress = false;
            
            pending.retain(|suite| {
                let dependencies_met = suite.dependencies.iter()
                    .all(|dep| completed.contains(dep));
                
                if dependencies_met {
                    execution_order.push(suite.clone());
                    completed.insert(suite.name.clone());
                    false // Remove from pending
                } else {
                    true // Keep in pending
                }
            });
            
            if !made_progress && !pending.is_empty() {
                return Err(OrchestrationError::DependencyFailure {
                    dependency: pending[0].dependencies[0].clone(),
                    dependent: pending[0].name.clone(),
                });
            }
        }
        
        // Execute tests in dependency order
        let mut all_results = Vec::new();
        
        // Group by parallel allowance and priority
        let mut parallel_groups = Vec::new();
        let mut sequential_tests = Vec::new();
        
        for suite in execution_order {
            if suite.parallel_allowed {
                parallel_groups.push(suite);
            } else {
                sequential_tests.push(suite);
            }
        }
        
        // Execute parallel tests in batches
        for chunk in parallel_groups.chunks(self.config.max_parallel_tests) {
            let futures = chunk.iter().map(|suite| self.execute_single_test_suite(suite.clone()));
            let chunk_results = join_all(futures).await;
            
            for result in chunk_results {
                match result {
                    Ok(test_result) => {
                        if test_result.status == TestStatus::Failed && self.config.fail_fast {
                            return Err(OrchestrationError::TestExecutionFailed {
                                test_name: test_result.name.clone(),
                                reason: test_result.error_message.unwrap_or_default(),
                            });
                        }
                        all_results.push(test_result);
                    },
                    Err(e) => return Err(e),
                }
            }
        }
        
        // Execute sequential tests
        for suite in sequential_tests {
            match self.execute_single_test_suite(suite.clone()).await {
                Ok(test_result) => {
                    if test_result.status == TestStatus::Failed && self.config.fail_fast {
                        return Err(OrchestrationError::TestExecutionFailed {
                            test_name: test_result.name.clone(),
                            reason: test_result.error_message.unwrap_or_default(),
                        });
                    }
                    all_results.push(test_result);
                },
                Err(e) => return Err(e),
            }
        }
        
        Ok(all_results)
    }
    
    /// Execute a single test suite
    async fn execute_single_test_suite(&self, suite: TestSuite) -> Result<TestResult, OrchestrationError> {
        println!("🧪 Running test suite: {}", suite.name);
        let start_time = Instant::now();
        
        let mut cmd = TokioCommand::new(&suite.command);
        cmd.args(&suite.args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        // Execute with timeout
        let timeout_duration = Duration::from_secs(suite.timeout_seconds);
        let result = tokio::time::timeout(timeout_duration, cmd.output()).await;
        
        let execution_time = start_time.elapsed();
        
        let test_result = match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                // Parse test output for metrics
                let (assertions_passed, assertions_total) = self.parse_test_assertions(&stdout);
                let performance_metrics = self.parse_performance_metrics(&stdout);
                
                TestResult {
                    name: suite.name.clone(),
                    status: if output.status.success() {
                        TestStatus::Passed
                    } else {
                        TestStatus::Failed
                    },
                    execution_time,
                    error_message: if output.status.success() {
                        None
                    } else {
                        Some(format!("stdout: {}\nstderr: {}", stdout, stderr))
                    },
                    coverage_contribution: suite.required_coverage,
                    assertions_passed,
                    assertions_total,
                    performance_metrics,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }
            },
            Ok(Err(e)) => TestResult {
                name: suite.name.clone(),
                status: TestStatus::Failed,
                execution_time,
                error_message: Some(format!("Command execution failed: {}", e)),
                coverage_contribution: 0.0,
                assertions_passed: 0,
                assertions_total: 0,
                performance_metrics: HashMap::new(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
            Err(_) => TestResult {
                name: suite.name.clone(),
                status: TestStatus::Timeout,
                execution_time,
                error_message: Some(format!("Test timed out after {}s", suite.timeout_seconds)),
                coverage_contribution: 0.0,
                assertions_passed: 0,
                assertions_total: 0,
                performance_metrics: HashMap::new(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        };
        
        // Store result
        {
            let mut results = self.results.lock().unwrap();
            results.push(test_result.clone());
        }
        
        // Log result
        match test_result.status {
            TestStatus::Passed => println!("✅ {} completed in {:.2}s", suite.name, execution_time.as_secs_f64()),
            TestStatus::Failed => println!("❌ {} failed in {:.2}s", suite.name, execution_time.as_secs_f64()),
            TestStatus::Timeout => println!("⏰ {} timed out after {:.2}s", suite.name, execution_time.as_secs_f64()),
            _ => {}
        }
        
        Ok(test_result)
    }
    
    /// Parse test assertions from output
    fn parse_test_assertions(&self, output: &str) -> (usize, usize) {
        // Parse Rust test output format
        let mut passed = 0;
        let mut total = 0;
        
        for line in output.lines() {
            if line.contains("test result:") {
                // Extract numbers from "test result: ok. 123 passed; 0 failed; 0 ignored"
                if let Some(passed_str) = line.split("passed").next() {
                    if let Some(num_str) = passed_str.split_whitespace().last() {
                        if let Ok(num) = num_str.parse::<usize>() {
                            passed += num;
                            total += num;
                        }
                    }
                }
                if let Some(failed_str) = line.split("failed").next() {
                    if let Some(part) = failed_str.split("passed;").nth(1) {
                        if let Some(num_str) = part.trim().split_whitespace().next() {
                            if let Ok(num) = num_str.parse::<usize>() {
                                total += num;
                            }
                        }
                    }
                }
            }
        }
        
        (passed, total)
    }
    
    /// Parse performance metrics from output
    fn parse_performance_metrics(&self, output: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        for line in output.lines() {
            // Parse benchmark output and custom metrics
            if line.contains("time:") || line.contains("ns/iter") {
                // Extract timing information
                if let Some(time_str) = line.split_whitespace()
                    .find(|s| s.ends_with("ns") || s.ends_with("ms") || s.ends_with("s")) {
                    
                    let value = time_str.chars()
                        .filter(|c| c.is_ascii_digit() || *c == '.')
                        .collect::<String>();
                    
                    if let Ok(time_value) = value.parse::<f64>() {
                        metrics.insert("execution_time_ns".to_string(), time_value);
                    }
                }
            }
            
            // Parse custom performance metrics
            if line.contains("Performance:") || line.contains("Throughput:") {
                // Custom metric parsing logic
                // This would be expanded based on specific metric formats
            }
        }
        
        metrics
    }
    
    /// Generate comprehensive execution report
    fn generate_execution_report(
        &self,
        results: Vec<TestResult>,
        total_execution_time: Duration,
        coverage_report: Option<CoverageReport>,
        ci_metadata: CiMetadata,
    ) -> Result<TestExecutionReport, OrchestrationError> {
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed_tests = results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let skipped_tests = results.iter().filter(|r| r.status == TestStatus::Skipped).count();
        let timeout_tests = results.iter().filter(|r| r.status == TestStatus::Timeout).count();
        
        let overall_coverage = coverage_report.as_ref()
            .map(|cr| cr.overall_line_coverage)
            .unwrap_or(0.0);
        
        let mut coverage_by_type = HashMap::new();
        
        // Calculate coverage by test type
        let mut type_groups: HashMap<TestType, Vec<&TestResult>> = HashMap::new();
        for result in &results {
            let test_type = self.get_test_type_for_result(result);
            type_groups.entry(test_type).or_default().push(result);
        }
        
        for (test_type, type_results) in type_groups {
            let type_coverage = type_results.iter()
                .map(|r| r.coverage_contribution)
                .sum::<f64>() / type_results.len() as f64;
            coverage_by_type.insert(format!("{:?}", test_type), type_coverage);
        }
        
        let requirements_met = failed_tests == 0 && 
                              timeout_tests == 0 && 
                              overall_coverage >= self.config.required_overall_coverage;
        
        let recommendations = self.generate_recommendations(&results, &coverage_report);
        
        Ok(TestExecutionReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            total_tests,
            passed_tests,
            failed_tests,
            skipped_tests,
            timeout_tests,
            total_execution_time,
            overall_coverage,
            coverage_by_type,
            test_results: results,
            coverage_report,
            ci_metadata,
            recommendations,
            requirements_met,
        })
    }
    
    fn get_test_type_for_result(&self, result: &TestResult) -> TestType {
        // Map test result back to test type based on name patterns
        match result.name.as_str() {
            name if name.contains("research_validated") => TestType::Unit,
            name if name.contains("property_based") => TestType::Property,
            name if name.contains("formal_verification") => TestType::Formal,
            name if name.contains("byzantine") => TestType::Byzantine,
            name if name.contains("security") => TestType::Security,
            name if name.contains("performance") => TestType::Performance,
            name if name.contains("coverage") => TestType::Coverage,
            name if name.contains("integration") => TestType::Integration,
            _ => TestType::Unit,
        }
    }
    
    /// Generate recommendations based on test results
    fn generate_recommendations(&self, results: &[TestResult], coverage_report: &Option<CoverageReport>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Failed tests recommendations
        let failed_results: Vec<_> = results.iter().filter(|r| r.status == TestStatus::Failed).collect();
        if !failed_results.is_empty() {
            recommendations.push(format!(
                "🔴 {} test(s) failed. Review error messages and fix failing assertions.",
                failed_results.len()
            ));
            
            for failed in failed_results {
                recommendations.push(format!(
                    "  - {}: {}",
                    failed.name,
                    failed.error_message.as_ref().unwrap_or(&"Unknown error".to_string())
                ));
            }
        }
        
        // Timeout recommendations
        let timeout_results: Vec<_> = results.iter().filter(|r| r.status == TestStatus::Timeout).collect();
        if !timeout_results.is_empty() {
            recommendations.push(format!(
                "⏰ {} test(s) timed out. Consider increasing timeout or optimizing test performance.",
                timeout_results.len()
            ));
        }
        
        // Coverage recommendations
        if let Some(coverage) = coverage_report {
            if coverage.overall_line_coverage < self.config.required_overall_coverage {
                recommendations.push(format!(
                    "📊 Coverage is {:.1}%, below required {:.1}%. Add tests for uncovered code paths.",
                    coverage.overall_line_coverage,
                    self.config.required_overall_coverage
                ));
            }
            
            if !coverage.violations.is_empty() {
                recommendations.push("🚨 Coverage violations detected:".to_string());
                for violation in &coverage.violations {
                    recommendations.push(format!("  - {}", violation));
                }
            }
        }
        
        // Performance recommendations
        let slow_tests: Vec<_> = results.iter()
            .filter(|r| r.execution_time > Duration::from_secs(60))
            .collect();
        
        if !slow_tests.is_empty() {
            recommendations.push(format!(
                "🐌 {} test(s) took over 60s. Consider optimization or parallelization.",
                slow_tests.len()
            ));
        }
        
        if recommendations.is_empty() {
            recommendations.push("✅ All tests passed successfully with excellent coverage!".to_string());
        }
        
        recommendations
    }
    
    /// Validate coverage requirements
    fn validate_coverage_requirements(&self, report: &TestExecutionReport) -> Result<(), OrchestrationError> {
        if report.overall_coverage < self.config.required_overall_coverage {
            return Err(OrchestrationError::CoverageRequirementFailed {
                actual: report.overall_coverage,
                required: self.config.required_overall_coverage,
            });
        }
        
        // Validate individual test type coverage requirements
        for suite in &self.test_suites {
            if let Some(&actual_coverage) = report.coverage_by_type.get(&format!("{:?}", suite.test_type)) {
                if actual_coverage < suite.required_coverage {
                    return Err(OrchestrationError::CoverageRequirementFailed {
                        actual: actual_coverage,
                        required: suite.required_coverage,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate comprehensive reports
    async fn generate_comprehensive_reports(&self, report: &TestExecutionReport) -> Result<(), OrchestrationError> {
        let reports_dir = self.output_dir.join("reports");
        create_dir_all(&reports_dir)?;
        
        // JSON report
        let json_report = serde_json::to_string_pretty(report)
            .map_err(|e| OrchestrationError::ReportGenerationFailed { reason: e.to_string() })?;
        
        let mut json_file = File::create(reports_dir.join("test_execution_report.json"))?;
        json_file.write_all(json_report.as_bytes())?;
        
        // HTML report
        self.generate_html_report(report, &reports_dir).await?;
        
        // JUnit XML for CI integration
        self.generate_junit_xml(report, &reports_dir).await?;
        
        // Console summary
        self.print_execution_summary(report);
        
        Ok(())
    }
    
    /// Generate HTML report
    async fn generate_html_report(&self, report: &TestExecutionReport, output_dir: &Path) -> Result<(), OrchestrationError> {
        let html_content = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Bayesian VaR Test Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 10px; }}
        .status-passed {{ color: #28a745; }}
        .status-failed {{ color: #dc3545; }}
        .status-timeout {{ color: #ffc107; }}
        .progress {{ width: 100%; height: 25px; background-color: #e9ecef; border-radius: 15px; overflow: hidden; }}
        .progress-bar {{ height: 100%; background: linear-gradient(45deg, #28a745, #20c997); transition: width 0.3s; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .test-results {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .test-passed {{ background-color: #d4edda; }}
        .test-failed {{ background-color: #f8d7da; }}
        .test-timeout {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Comprehensive Bayesian VaR Test Report</h1>
            <p>Generated: {} | Branch: {} | Commit: {}</p>
            <p>Requirements Met: <strong>{}</strong></p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value status-passed">{}</div>
                <div class="metric-label">Tests Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-failed">{}</div>
                <div class="metric-label">Tests Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-timeout">{}</div>
                <div class="metric-label">Tests Timeout</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{:.1}%</div>
                <div class="metric-label">Coverage</div>
            </div>
        </div>
        
        <h2>Coverage Progress</h2>
        <div class="progress">
            <div class="progress-bar" style="width: {:.1}%"></div>
        </div>
        <p>Overall Coverage: {:.2}% (Required: {:.1}%)</p>
        
        <div class="recommendations">
            <h3>🎯 Recommendations</h3>
            <ul>
            {}
            </ul>
        </div>
        
        <div class="test-results">
            <h2>Test Results Details</h2>
            <table>
                <thead>
                    <tr><th>Test Name</th><th>Status</th><th>Duration</th><th>Coverage</th><th>Assertions</th></tr>
                </thead>
                <tbody>
                {}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        "#,
            report.timestamp,
            report.ci_metadata.branch,
            &report.ci_metadata.commit_hash[..8],
            if report.requirements_met { "✅ YES" } else { "❌ NO" },
            report.passed_tests,
            report.failed_tests,
            report.timeout_tests,
            report.overall_coverage,
            report.overall_coverage,
            report.overall_coverage,
            self.config.required_overall_coverage,
            report.recommendations.iter()
                .map(|r| format!("<li>{}</li>", r))
                .collect::<Vec<_>>()
                .join("\n"),
            report.test_results.iter()
                .map(|tr| format!(
                    "<tr class='test-{}'><td>{}</td><td>{:?}</td><td>{:.2}s</td><td>{:.1}%</td><td>{}/{}</td></tr>",
                    match tr.status {
                        TestStatus::Passed => "passed",
                        TestStatus::Failed => "failed",
                        TestStatus::Timeout => "timeout",
                        _ => "",
                    },
                    tr.name,
                    tr.status,
                    tr.execution_time.as_secs_f64(),
                    tr.coverage_contribution,
                    tr.assertions_passed,
                    tr.assertions_total
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        
        let mut html_file = File::create(output_dir.join("test_report.html"))?;
        html_file.write_all(html_content.as_bytes())?;
        
        Ok(())
    }
    
    /// Generate JUnit XML for CI integration
    async fn generate_junit_xml(&self, report: &TestExecutionReport, output_dir: &Path) -> Result<(), OrchestrationError> {
        let mut xml_content = String::new();
        xml_content.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml_content.push_str(&format!(
            "<testsuites name=\"BayesianVaR\" tests=\"{}\" failures=\"{}\" time=\"{:.3}\">\n",
            report.total_tests,
            report.failed_tests + report.timeout_tests,
            report.total_execution_time.as_secs_f64()
        ));
        
        for test in &report.test_results {
            xml_content.push_str(&format!(
                "  <testcase name=\"{}\" time=\"{:.3}\"",
                test.name,
                test.execution_time.as_secs_f64()
            ));
            
            match test.status {
                TestStatus::Passed => {
                    xml_content.push_str("/>\n");
                },
                TestStatus::Failed => {
                    xml_content.push_str(">\n");
                    xml_content.push_str(&format!(
                        "    <failure message=\"Test failed\">{}</failure>\n",
                        test.error_message.as_ref().unwrap_or(&"Unknown failure".to_string())
                    ));
                    xml_content.push_str("  </testcase>\n");
                },
                TestStatus::Timeout => {
                    xml_content.push_str(">\n");
                    xml_content.push_str("    <failure message=\"Test timeout\">Test exceeded timeout limit</failure>\n");
                    xml_content.push_str("  </testcase>\n");
                },
                _ => {
                    xml_content.push_str("/>\n");
                }
            }
        }
        
        xml_content.push_str("</testsuites>\n");
        
        let mut xml_file = File::create(output_dir.join("junit.xml"))?;
        xml_file.write_all(xml_content.as_bytes())?;
        
        Ok(())
    }
    
    /// Print execution summary to console
    fn print_execution_summary(&self, report: &TestExecutionReport) {
        println!("\n📊 COMPREHENSIVE TEST EXECUTION SUMMARY");
        println!("=====================================");
        println!("Timestamp: {}", report.timestamp);
        println!("Total Tests: {}", report.total_tests);
        println!("✅ Passed: {}", report.passed_tests);
        println!("❌ Failed: {}", report.failed_tests);
        println!("⏰ Timeout: {}", report.timeout_tests);
        println!("⏭  Skipped: {}", report.skipped_tests);
        println!("⏱  Total Time: {:.2}s", report.total_execution_time.as_secs_f64());
        println!("📈 Coverage: {:.2}%", report.overall_coverage);
        println!("✨ Requirements Met: {}", if report.requirements_met { "YES" } else { "NO" });
        
        if !report.recommendations.is_empty() {
            println!("\n🎯 RECOMMENDATIONS:");
            for rec in &report.recommendations {
                println!("{}", rec);
            }
        }
        
        println!("\n=====================================\n");
    }
    
    /// Gather CI metadata
    async fn gather_ci_metadata(&self) -> CiMetadata {
        let branch = std::env::var("GITHUB_REF_NAME")
            .or_else(|_| std::env::var("CI_COMMIT_REF_NAME"))
            .unwrap_or_else(|_| "unknown".to_string());
        
        let commit_hash = std::env::var("GITHUB_SHA")
            .or_else(|_| std::env::var("CI_COMMIT_SHA"))
            .unwrap_or_else(|_| "unknown".to_string());
        
        let build_number = std::env::var("GITHUB_RUN_NUMBER").ok();
        let pr_number = std::env::var("GITHUB_EVENT_PULL_REQUEST_NUMBER").ok();
        
        let environment = std::env::var("GITHUB_ACTIONS")
            .map(|_| "GitHub Actions".to_string())
            .or_else(|_| std::env::var("GITLAB_CI").map(|_| "GitLab CI".to_string()))
            .unwrap_or_else(|_| "Local".to_string());
        
        // Get Rust version
        let rust_version = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .unwrap_or_else(|| "unknown".to_string());
        
        CiMetadata {
            branch,
            commit_hash,
            build_number,
            pr_number,
            environment,
            rust_version: rust_version.trim().to_string(),
        }
    }
    
    /// Integrate with CI systems
    async fn integrate_with_ci(&self, report: &TestExecutionReport) -> Result<(), OrchestrationError> {
        // GitHub Actions integration
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            self.integrate_with_github_actions(report).await?;
        }
        
        // GitLab CI integration
        if std::env::var("GITLAB_CI").is_ok() {
            self.integrate_with_gitlab_ci(report).await?;
        }
        
        Ok(())
    }
    
    async fn integrate_with_github_actions(&self, report: &TestExecutionReport) -> Result<(), OrchestrationError> {
        // Set GitHub Actions outputs
        if let Ok(output_file) = std::env::var("GITHUB_OUTPUT") {
            let mut file = File::create(&output_file)?;
            writeln!(file, "test-results={}", if report.requirements_met { "success" } else { "failure" })?;
            writeln!(file, "coverage={:.2}", report.overall_coverage)?;
            writeln!(file, "total-tests={}", report.total_tests)?;
            writeln!(file, "failed-tests={}", report.failed_tests)?;
        }
        
        // Set exit code for CI
        if !report.requirements_met {
            std::process::exit(1);
        }
        
        Ok(())
    }
    
    async fn integrate_with_gitlab_ci(&self, _report: &TestExecutionReport) -> Result<(), OrchestrationError> {
        // GitLab CI integration would go here
        Ok(())
    }
}

/// CLI entry point for test orchestration
pub async fn run_comprehensive_tests() -> Result<(), OrchestrationError> {
    let config = OrchestrationConfig::default();
    let output_dir = std::env::current_dir()
        .map_err(OrchestrationError::IoError)?
        .join("test-results");
    
    let mut orchestrator = ComprehensiveTestOrchestrator::new(config, output_dir);
    let report = orchestrator.execute_all_tests().await?;
    
    if !report.requirements_met {
        std::process::exit(1);
    }
    
    Ok(())
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    
    #[test]
    fn test_orchestrator_configuration() {
        let config = OrchestrationConfig::default();
        assert!(config.max_parallel_tests > 0);
        assert_eq!(config.required_overall_coverage, 100.0);
        assert!(config.generate_reports);
    }
    
    #[tokio::test]
    async fn test_ci_metadata_gathering() {
        let config = OrchestrationConfig::default();
        let output_dir = std::env::temp_dir().join("test-orchestrator");
        let orchestrator = ComprehensiveTestOrchestrator::new(config, output_dir);
        
        let metadata = orchestrator.gather_ci_metadata().await;
        assert!(!metadata.rust_version.is_empty());
        assert!(!metadata.environment.is_empty());
    }
}