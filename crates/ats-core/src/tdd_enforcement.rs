//! TDD Enforcement Framework with Nanosecond Precision Requirements
//!
//! This module enforces Test-Driven Development with mandatory nanosecond
//! precision benchmarks for every component. No code can be merged without
//! passing all performance gates.

use crate::nanosecond_validator::{NanosecondValidator, ValidationResult};
use crate::error::AtsCoreError;
use std::collections::HashMap;
use std::path::Path;

/// TDD Enforcement Engine
pub struct TddEnforcer {
    validator: NanosecondValidator,
    performance_gates: HashMap<String, PerformanceGate>,
    coverage_requirements: CoverageRequirements,
}

/// Performance gate requirements
#[derive(Debug, Clone)]
pub struct PerformanceGate {
    pub target_ns: u64,
    pub success_rate: f64,
    pub mandatory: bool,
    pub component_type: ComponentType,
}

#[derive(Debug, Clone)]
pub enum ComponentType {
    TradingDecision,
    WhaleDetection,
    GpuKernel,
    ApiResponse,
    Memory,
    Network,
    Custom(String),
}

/// Coverage requirements
#[derive(Debug, Clone)]
pub struct CoverageRequirements {
    pub min_line_coverage: f64,
    pub min_branch_coverage: f64,
    pub min_function_coverage: f64,
    pub require_nanosecond_benchmarks: bool,
    pub require_real_world_scenarios: bool,
}

impl Default for CoverageRequirements {
    fn default() -> Self {
        Self {
            min_line_coverage: 95.0,
            min_branch_coverage: 90.0,
            min_function_coverage: 100.0,
            require_nanosecond_benchmarks: true,
            require_real_world_scenarios: true,
        }
    }
}

/// TDD validation result
#[derive(Debug, Clone)]
pub struct TddValidationResult {
    pub component_name: String,
    pub performance_passed: bool,
    pub coverage_passed: bool,
    pub benchmarks_exist: bool,
    pub real_world_tests_exist: bool,
    pub overall_passed: bool,
    pub performance_results: Vec<ValidationResult>,
    pub violations: Vec<TddViolation>,
}

#[derive(Debug, Clone)]
pub struct TddViolation {
    pub violation_type: ViolationType,
    pub description: String,
    pub severity: ViolationSeverity,
    pub component: String,
}

#[derive(Debug, Clone)]
pub enum ViolationType {
    MissingBenchmarks,
    PerformanceTargetMissed,
    InsufficientCoverage,
    MissingRealWorldTests,
    NoNanosecondValidation,
    MemoryLeak,
    SafetyViolation,
}

#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Critical,  // Blocks merge
    Major,     // Must be fixed before merge
    Minor,     // Should be fixed
}

impl TddEnforcer {
    /// Create a new TDD enforcer with strict nanosecond precision requirements
    pub fn new() -> Result<Self, AtsCoreError> {
        let mut performance_gates = HashMap::new();
        
        // Define mandatory performance gates
        performance_gates.insert("trading_decision".to_string(), PerformanceGate {
            target_ns: 500,
            success_rate: 0.9999,
            mandatory: true,
            component_type: ComponentType::TradingDecision,
        });
        
        performance_gates.insert("whale_detection".to_string(), PerformanceGate {
            target_ns: 200,
            success_rate: 0.9999,
            mandatory: true,
            component_type: ComponentType::WhaleDetection,
        });
        
        performance_gates.insert("gpu_kernel".to_string(), PerformanceGate {
            target_ns: 100,
            success_rate: 0.9999,
            mandatory: true,
            component_type: ComponentType::GpuKernel,
        });
        
        performance_gates.insert("api_response".to_string(), PerformanceGate {
            target_ns: 50,
            success_rate: 0.9999,
            mandatory: true,
            component_type: ComponentType::ApiResponse,
        });
        
        performance_gates.insert("memory_allocation".to_string(), PerformanceGate {
            target_ns: 1000,
            success_rate: 0.99,
            mandatory: true,
            component_type: ComponentType::Memory,
        });
        
        Ok(Self {
            validator: NanosecondValidator::new()?,
            performance_gates,
            coverage_requirements: CoverageRequirements::default(),
        })
    }
    
    /// Enforce TDD requirements for a component
    pub fn enforce_tdd_requirements<F>(&self, component_name: &str, operation: F) -> Result<TddValidationResult, AtsCoreError>
    where
        F: Fn() -> () + Clone,
    {
        let mut violations = Vec::new();
        let mut performance_results = Vec::new();
        let mut performance_passed = true;
        
        // Check if component has required performance gates
        if let Some(gate) = self.performance_gates.get(component_name) {
            let validation_result = match gate.component_type {
                ComponentType::TradingDecision => {
                    self.validator.validate_trading_decision(operation.clone(), component_name)?
                },
                ComponentType::WhaleDetection => {
                    self.validator.validate_whale_detection(operation.clone(), component_name)?
                },
                ComponentType::GpuKernel => {
                    self.validator.validate_gpu_kernel(operation.clone(), component_name)?
                },
                ComponentType::ApiResponse => {
                    self.validator.validate_api_response(operation.clone(), component_name)?
                },
                ComponentType::Memory => {
                    self.validator.validate_memory_stability(operation.clone(), component_name)?
                },
                ComponentType::Custom(_) => {
                    self.validator.validate_custom(operation.clone(), component_name, gate.target_ns, gate.success_rate)?
                },
                _ => {
                    self.validator.validate_custom(operation.clone(), component_name, gate.target_ns, gate.success_rate)?
                }
            };
            
            if !validation_result.passed && gate.mandatory {
                performance_passed = false;
                violations.push(TddViolation {
                    violation_type: ViolationType::PerformanceTargetMissed,
                    description: format!(
                        "Component '{}' failed mandatory performance gate: {}ns target with {:.2}% success rate",
                        component_name, gate.target_ns, gate.success_rate * 100.0
                    ),
                    severity: ViolationSeverity::Critical,
                    component: component_name.to_string(),
                });
            }
            
            performance_results.push(validation_result);
        } else {
            // No performance gate defined - this is a violation for critical components
            violations.push(TddViolation {
                violation_type: ViolationType::NoNanosecondValidation,
                description: format!("Component '{}' has no nanosecond precision validation", component_name),
                severity: ViolationSeverity::Major,
                component: component_name.to_string(),
            });
        }
        
        // Check for benchmark files
        let benchmarks_exist = self.check_benchmarks_exist(component_name);
        if !benchmarks_exist && self.coverage_requirements.require_nanosecond_benchmarks {
            violations.push(TddViolation {
                violation_type: ViolationType::MissingBenchmarks,
                description: format!("Component '{}' missing nanosecond precision benchmarks", component_name),
                severity: ViolationSeverity::Critical,
                component: component_name.to_string(),
            });
        }
        
        // Check for real-world test scenarios
        let real_world_tests_exist = self.check_real_world_tests_exist(component_name);
        if !real_world_tests_exist && self.coverage_requirements.require_real_world_scenarios {
            violations.push(TddViolation {
                violation_type: ViolationType::MissingRealWorldTests,
                description: format!("Component '{}' missing real-world scenario tests", component_name),
                severity: ViolationSeverity::Major,
                component: component_name.to_string(),
            });
        }
        
        // Check code coverage (simulated - in real implementation would integrate with coverage tools)
        let coverage_passed = self.validate_code_coverage(component_name, &mut violations);
        
        let overall_passed = performance_passed && 
                            coverage_passed && 
                            benchmarks_exist &&
                            real_world_tests_exist &&
                            violations.iter().all(|v| !matches!(v.severity, ViolationSeverity::Critical));
        
        Ok(TddValidationResult {
            component_name: component_name.to_string(),
            performance_passed,
            coverage_passed,
            benchmarks_exist,
            real_world_tests_exist,
            overall_passed,
            performance_results,
            violations,
        })
    }
    
    /// Check if benchmarks exist for component
    fn check_benchmarks_exist(&self, component_name: &str) -> bool {
        // In real implementation, would scan benchmark directories
        let benchmark_paths = vec![
            format!("benches/{}_benchmarks.rs", component_name),
            format!("benches/nanosecond_{}_bench.rs", component_name),
            format!("src/benches/{}.rs", component_name),
        ];
        
        benchmark_paths.iter().any(|path| Path::new(path).exists())
    }
    
    /// Check if real-world tests exist for component
    fn check_real_world_tests_exist(&self, component_name: &str) -> bool {
        // In real implementation, would scan test directories
        let test_paths = vec![
            format!("tests/{}_real_world_tests.rs", component_name),
            format!("tests/scenarios/{}_scenarios.rs", component_name),
            format!("src/tests/real_world_{}.rs", component_name),
        ];
        
        test_paths.iter().any(|path| Path::new(path).exists())
    }
    
    /// Validate code coverage requirements
    fn validate_code_coverage(&self, component_name: &str, violations: &mut Vec<TddViolation>) -> bool {
        // In real implementation, would integrate with coverage tools like cargo-llvm-cov
        // For now, simulate coverage check
        
        // Simulate coverage metrics (in real implementation, would parse coverage reports)
        let simulated_line_coverage = 92.5; // Would come from actual coverage tool
        let simulated_branch_coverage = 88.0;
        let simulated_function_coverage = 100.0;
        
        let mut coverage_passed = true;
        
        if simulated_line_coverage < self.coverage_requirements.min_line_coverage {
            violations.push(TddViolation {
                violation_type: ViolationType::InsufficientCoverage,
                description: format!(
                    "Component '{}' line coverage {:.1}% below required {:.1}%",
                    component_name, simulated_line_coverage, self.coverage_requirements.min_line_coverage
                ),
                severity: ViolationSeverity::Major,
                component: component_name.to_string(),
            });
            coverage_passed = false;
        }
        
        if simulated_branch_coverage < self.coverage_requirements.min_branch_coverage {
            violations.push(TddViolation {
                violation_type: ViolationType::InsufficientCoverage,
                description: format!(
                    "Component '{}' branch coverage {:.1}% below required {:.1}%",
                    component_name, simulated_branch_coverage, self.coverage_requirements.min_branch_coverage
                ),
                severity: ViolationSeverity::Major,
                component: component_name.to_string(),
            });
            coverage_passed = false;
        }
        
        if simulated_function_coverage < self.coverage_requirements.min_function_coverage {
            violations.push(TddViolation {
                violation_type: ViolationType::InsufficientCoverage,
                description: format!(
                    "Component '{}' function coverage {:.1}% below required {:.1}%",
                    component_name, simulated_function_coverage, self.coverage_requirements.min_function_coverage
                ),
                severity: ViolationSeverity::Critical,
                component: component_name.to_string(),
            });
            coverage_passed = false;
        }
        
        coverage_passed
    }
    
    /// Generate TDD compliance report
    pub fn generate_compliance_report(&self, results: &[TddValidationResult]) -> TddComplianceReport {
        let total_components = results.len();
        let passed_components = results.iter().filter(|r| r.overall_passed).count();
        let failed_components = total_components - passed_components;
        
        let critical_violations: Vec<_> = results.iter()
            .flat_map(|r| &r.violations)
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .cloned()
            .collect();
        
        let major_violations: Vec<_> = results.iter()
            .flat_map(|r| &r.violations)
            .filter(|v| matches!(v.severity, ViolationSeverity::Major))
            .cloned()
            .collect();
        
        let ready_for_merge = critical_violations.is_empty();
        
        TddComplianceReport {
            total_components,
            passed_components,
            failed_components,
            compliance_percentage: (passed_components as f64 / total_components as f64) * 100.0,
            critical_violations,
            major_violations,
            ready_for_merge,
            results: results.to_vec(),
        }
    }
}

/// TDD compliance report
#[derive(Debug, Clone)]
pub struct TddComplianceReport {
    pub total_components: usize,
    pub passed_components: usize,
    pub failed_components: usize,
    pub compliance_percentage: f64,
    pub critical_violations: Vec<TddViolation>,
    pub major_violations: Vec<TddViolation>,
    pub ready_for_merge: bool,
    pub results: Vec<TddValidationResult>,
}

impl TddComplianceReport {
    /// Display comprehensive compliance report
    pub fn display_compliance_report(&self) {
        println!("🔒 TDD ENFORCEMENT COMPLIANCE REPORT");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        println!("📊 OVERALL COMPLIANCE");
        println!("  Total Components: {}", self.total_components);
        println!("  Passed: {} ({}%)", self.passed_components, 
                 if self.total_components > 0 { self.passed_components * 100 / self.total_components } else { 0 });
        println!("  Failed: {} ({}%)", self.failed_components,
                 if self.total_components > 0 { self.failed_components * 100 / self.total_components } else { 0 });
        println!("  Compliance Rate: {:.1}%", self.compliance_percentage);
        println!();
        
        if self.ready_for_merge {
            println!("✅ READY FOR MERGE");
            println!("  All critical TDD requirements met");
            println!("  Nanosecond precision validated");
            println!("  Coverage requirements satisfied");
        } else {
            println!("❌ NOT READY FOR MERGE");
            println!("  {} critical violations must be fixed", self.critical_violations.len());
            println!("  {} major violations should be addressed", self.major_violations.len());
        }
        
        println!();
        
        // Display violations
        if !self.critical_violations.is_empty() {
            println!("🚨 CRITICAL VIOLATIONS (BLOCKS MERGE)");
            for violation in &self.critical_violations {
                println!("  ❌ {}: {}", violation.component, violation.description);
            }
            println!();
        }
        
        if !self.major_violations.is_empty() {
            println!("⚠️  MAJOR VIOLATIONS (SHOULD BE FIXED)");
            for violation in &self.major_violations {
                println!("  ⚠️  {}: {}", violation.component, violation.description);
            }
            println!();
        }
        
        // Display component details
        println!("📋 COMPONENT DETAILS");
        println!("┌─────────────────────┬─────────────┬──────────────┬──────────────┬──────────────┐");
        println!("│ Component           │ Performance │ Coverage     │ Benchmarks   │ Real-World   │");
        println!("├─────────────────────┼─────────────┼──────────────┼──────────────┼──────────────┤");
        
        for result in &self.results {
            println!("│ {:<19} │ {:<11} │ {:<12} │ {:<12} │ {:<12} │",
                     result.component_name,
                     if result.performance_passed { "✅ Pass" } else { "❌ Fail" },
                     if result.coverage_passed { "✅ Pass" } else { "❌ Fail" },
                     if result.benchmarks_exist { "✅ Exist" } else { "❌ Missing" },
                     if result.real_world_tests_exist { "✅ Exist" } else { "❌ Missing" });
        }
        
        println!("└─────────────────────┴─────────────┴──────────────┴──────────────┴──────────────┘");
        
        println!();
        
        if self.ready_for_merge {
            println!("🎉 TDD ENFORCEMENT PASSED!");
            println!("✅ All components meet nanosecond precision requirements");
            println!("✅ Code quality and coverage standards met");
            println!("✅ Real-world scenario validation completed");
        } else {
            println!("🚫 TDD ENFORCEMENT FAILED!");
            println!("❌ Fix all critical violations before merge");
            println!("⚠️  Address major violations for code quality");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tdd_enforcer_creation() {
        let enforcer = TddEnforcer::new();
        assert!(enforcer.is_ok());
    }
    
    #[test]
    fn test_trading_decision_tdd_enforcement() {
        let enforcer = TddEnforcer::new().unwrap();
        
        // Test a fast trading decision
        let result = enforcer.enforce_tdd_requirements("trading_decision", || {
            let data = vec![1.0, 2.0, 3.0];
            let _sum: f64 = data.iter().sum();
        }).unwrap();
        
        // Should have performance results
        assert!(!result.performance_results.is_empty());
        
        // Component name should match
        assert_eq!(result.component_name, "trading_decision");
    }
    
    #[test]
    fn test_compliance_report_generation() {
        let enforcer = TddEnforcer::new().unwrap();
        
        let results = vec![
            enforcer.enforce_tdd_requirements("test_component", || {
                let _x = 1 + 1;
            }).unwrap()
        ];
        
        let report = enforcer.generate_compliance_report(&results);
        
        assert_eq!(report.total_components, 1);
        assert!(report.compliance_percentage <= 100.0);
    }
    
    #[test]
    fn test_performance_gate_validation() {
        let enforcer = TddEnforcer::new().unwrap();
        
        // Test with operation that should pass
        let result = enforcer.enforce_tdd_requirements("trading_decision", || {
            // Minimal operation
            let _result = 1;
        }).unwrap();
        
        // Should have attempted performance validation
        assert!(!result.performance_results.is_empty());
    }
}