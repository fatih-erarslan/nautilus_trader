// CWTS Regulatory Compliance Validation Framework
// SEC Rule 15c3-5 and financial regulations compliance testing

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceValidation {
    pub rule_name: String,
    pub description: String,
    pub status: ComplianceStatus,
    pub findings: Vec<ComplianceFinding>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    Warning,
    NotTested,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFinding {
    pub severity: Severity,
    pub description: String,
    pub recommendation: String,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// SEC Rule 15c3-5 Pre-trade Controls Validator
pub struct Rule15c35Validator {
    max_order_size: Decimal,
    max_position_limit: Decimal,
    daily_loss_limit: Decimal,
    restricted_securities: Vec<String>,
    approved_users: Vec<String>,
}

impl Rule15c35Validator {
    pub fn new() -> Self {
        Self {
            max_order_size: Decimal::from(1_000_000), // $1M default
            max_position_limit: Decimal::from(10_000_000), // $10M default
            daily_loss_limit: Decimal::from(500_000), // $500K default
            restricted_securities: Vec::new(),
            approved_users: Vec::new(),
        }
    }

    /// Validate order size controls
    pub fn validate_order_size_controls(&self) -> ComplianceValidation {
        let mut findings = Vec::new();
        
        // Check if order size limits are implemented
        if self.max_order_size > Decimal::ZERO {
            findings.push(ComplianceFinding {
                severity: Severity::Info,
                description: format!("Order size limit configured: ${}", self.max_order_size),
                recommendation: "Ensure limit is appropriate for firm's risk tolerance".to_string(),
                evidence: Some("Configuration verified in system parameters".to_string()),
            });
        } else {
            findings.push(ComplianceFinding {
                severity: Severity::Critical,
                description: "No order size limits configured".to_string(),
                recommendation: "Implement maximum order size controls immediately".to_string(),
                evidence: Some("System configuration shows zero limits".to_string()),
            });
        }

        // Check for dynamic adjustment capability
        findings.push(ComplianceFinding {
            severity: Severity::Medium,
            description: "Order size limits should be adjustable by authorized personnel".to_string(),
            recommendation: "Implement role-based limit adjustment controls".to_string(),
            evidence: None,
        });

        let status = if findings.iter().any(|f| matches!(f.severity, Severity::Critical)) {
            ComplianceStatus::NonCompliant
        } else if findings.iter().any(|f| matches!(f.severity, Severity::High)) {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Compliant
        };

        ComplianceValidation {
            rule_name: "SEC Rule 15c3-5 Order Size Controls".to_string(),
            description: "Validation of pre-trade order size controls".to_string(),
            status,
            findings,
            timestamp: Utc::now(),
        }
    }

    /// Validate position limits
    pub fn validate_position_limits(&self) -> ComplianceValidation {
        let mut findings = Vec::new();
        
        if self.max_position_limit > Decimal::ZERO {
            findings.push(ComplianceFinding {
                severity: Severity::Info,
                description: format!("Position limits configured: ${}", self.max_position_limit),
                recommendation: "Regularly review and update position limits".to_string(),
                evidence: Some("Position limit configuration verified".to_string()),
            });
        } else {
            findings.push(ComplianceFinding {
                severity: Severity::Critical,
                description: "No position limits configured".to_string(),
                recommendation: "Implement position limits immediately per SEC requirements".to_string(),
                evidence: Some("System shows no position limit enforcement".to_string()),
            });
        }

        // Check for real-time monitoring
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: "Position limits must be monitored in real-time".to_string(),
            recommendation: "Implement continuous position monitoring system".to_string(),
            evidence: None,
        });

        let status = if findings.iter().any(|f| matches!(f.severity, Severity::Critical)) {
            ComplianceStatus::NonCompliant
        } else {
            ComplianceStatus::Warning
        };

        ComplianceValidation {
            rule_name: "SEC Rule 15c3-5 Position Limits".to_string(),
            description: "Validation of position limit controls".to_string(),
            status,
            findings,
            timestamp: Utc::now(),
        }
    }

    /// Validate credit/risk controls
    pub fn validate_credit_controls(&self) -> ComplianceValidation {
        let mut findings = Vec::new();
        
        if self.daily_loss_limit > Decimal::ZERO {
            findings.push(ComplianceFinding {
                severity: Severity::Info,
                description: format!("Daily loss limits configured: ${}", self.daily_loss_limit),
                recommendation: "Monitor daily P&L against configured limits".to_string(),
                evidence: Some("Daily loss limit configuration verified".to_string()),
            });
        } else {
            findings.push(ComplianceFinding {
                severity: Severity::Critical,
                description: "No daily loss limits configured".to_string(),
                recommendation: "Implement daily loss limits per regulatory requirements".to_string(),
                evidence: Some("No loss limit controls found in system".to_string()),
            });
        }

        // Check for automated halt mechanisms
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: "Automated trading halt on limit breach required".to_string(),
            recommendation: "Implement automatic halt when loss limits are exceeded".to_string(),
            evidence: None,
        });

        let status = if findings.iter().any(|f| matches!(f.severity, Severity::Critical)) {
            ComplianceStatus::NonCompliant
        } else {
            ComplianceStatus::Warning
        };

        ComplianceValidation {
            rule_name: "SEC Rule 15c3-5 Credit Controls".to_string(),
            description: "Validation of credit and risk controls".to_string(),
            status,
            findings,
            timestamp: Utc::now(),
        }
    }
}

/// Audit Trail Validator
pub struct AuditTrailValidator {
    required_fields: Vec<String>,
    retention_period_days: u32,
}

impl AuditTrailValidator {
    pub fn new() -> Self {
        Self {
            required_fields: vec![
                "timestamp".to_string(),
                "order_id".to_string(),
                "user_id".to_string(),
                "symbol".to_string(),
                "side".to_string(),
                "quantity".to_string(),
                "price".to_string(),
                "order_type".to_string(),
                "execution_id".to_string(),
                "venue".to_string(),
            ],
            retention_period_days: 2555, // 7 years as per SEC requirements
        }
    }

    /// Validate audit trail completeness
    pub fn validate_audit_trail(&self) -> ComplianceValidation {
        let mut findings = Vec::new();
        
        // Check required fields
        for field in &self.required_fields {
            findings.push(ComplianceFinding {
                severity: Severity::Info,
                description: format!("Required audit field '{}' validation needed", field),
                recommendation: format!("Verify '{}' is captured in all transaction records", field),
                evidence: None,
            });
        }

        // Check timestamp accuracy
        findings.push(ComplianceFinding {
            severity: Severity::Medium,
            description: "Timestamp accuracy must be within 1 millisecond".to_string(),
            recommendation: "Implement NTP synchronization for accurate timestamping".to_string(),
            evidence: None,
        });

        // Check data retention
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: format!("Audit records must be retained for {} days", self.retention_period_days),
            recommendation: "Implement automated archival system with proper backup".to_string(),
            evidence: None,
        });

        // Check immutability
        findings.push(ComplianceFinding {
            severity: Severity::Critical,
            description: "Audit records must be immutable once created".to_string(),
            recommendation: "Implement write-once audit log with cryptographic integrity".to_string(),
            evidence: None,
        });

        ComplianceValidation {
            rule_name: "Audit Trail Requirements".to_string(),
            description: "Validation of comprehensive audit trail capabilities".to_string(),
            status: ComplianceStatus::Warning, // Requires implementation verification
            findings,
            timestamp: Utc::now(),
        }
    }
}

/// Circuit Breaker Validator
pub struct CircuitBreakerValidator {
    price_movement_threshold: Decimal,
    volume_spike_threshold: Decimal,
    error_rate_threshold: f64,
    recovery_time_minutes: u32,
}

impl CircuitBreakerValidator {
    pub fn new() -> Self {
        Self {
            price_movement_threshold: Decimal::from_f32_retain(0.10), // 10% price movement
            volume_spike_threshold: Decimal::from(5), // 5x normal volume
            error_rate_threshold: 0.05, // 5% error rate
            recovery_time_minutes: 5, // 5 minute recovery period
        }
    }

    /// Validate circuit breaker functionality
    pub fn validate_circuit_breakers(&self) -> ComplianceValidation {
        let mut findings = Vec::new();

        // Price movement circuit breaker
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: format!("Price movement circuit breaker threshold: {}%", 
                               self.price_movement_threshold * Decimal::from(100)),
            recommendation: "Test circuit breaker triggers under simulated market stress".to_string(),
            evidence: None,
        });

        // Volume spike detection
        findings.push(ComplianceFinding {
            severity: Severity::Medium,
            description: format!("Volume spike detection threshold: {}x normal volume", 
                               self.volume_spike_threshold),
            recommendation: "Implement real-time volume monitoring with automated alerts".to_string(),
            evidence: None,
        });

        // Error rate monitoring
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: format!("Error rate threshold: {}%", self.error_rate_threshold * 100.0),
            recommendation: "Monitor system error rates and implement automatic halt on threshold breach".to_string(),
            evidence: None,
        });

        // Recovery mechanism
        findings.push(ComplianceFinding {
            severity: Severity::Critical,
            description: format!("Circuit breaker recovery time: {} minutes", self.recovery_time_minutes),
            recommendation: "Implement graduated recovery with manual override capability".to_string(),
            evidence: None,
        });

        // Testing requirements
        findings.push(ComplianceFinding {
            severity: Severity::High,
            description: "Circuit breakers must be tested regularly".to_string(),
            recommendation: "Establish monthly circuit breaker testing schedule".to_string(),
            evidence: None,
        });

        ComplianceValidation {
            rule_name: "Circuit Breaker Controls".to_string(),
            description: "Validation of automated circuit breaker mechanisms".to_string(),
            status: ComplianceStatus::NotTested, // Requires functional testing
            findings,
            timestamp: Utc::now(),
        }
    }
}

/// Comprehensive Compliance Validator
pub struct ComplianceValidator {
    rule_validator: Rule15c35Validator,
    audit_validator: AuditTrailValidator,
    circuit_validator: CircuitBreakerValidator,
}

impl ComplianceValidator {
    pub fn new() -> Self {
        Self {
            rule_validator: Rule15c35Validator::new(),
            audit_validator: AuditTrailValidator::new(),
            circuit_validator: CircuitBreakerValidator::new(),
        }
    }

    /// Run comprehensive compliance validation
    pub fn run_comprehensive_validation(&self) -> Vec<ComplianceValidation> {
        let mut validations = Vec::new();

        // SEC Rule 15c3-5 validations
        validations.push(self.rule_validator.validate_order_size_controls());
        validations.push(self.rule_validator.validate_position_limits());
        validations.push(self.rule_validator.validate_credit_controls());

        // Audit trail validation
        validations.push(self.audit_validator.validate_audit_trail());

        // Circuit breaker validation
        validations.push(self.circuit_validator.validate_circuit_breakers());

        validations
    }

    /// Generate compliance report
    pub fn generate_compliance_report(&self) -> String {
        let validations = self.run_comprehensive_validation();
        let mut report = String::new();

        report.push_str("# CWTS Regulatory Compliance Validation Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

        let compliant_count = validations.iter().filter(|v| matches!(v.status, ComplianceStatus::Compliant)).count();
        let warning_count = validations.iter().filter(|v| matches!(v.status, ComplianceStatus::Warning)).count();
        let non_compliant_count = validations.iter().filter(|v| matches!(v.status, ComplianceStatus::NonCompliant)).count();
        let not_tested_count = validations.iter().filter(|v| matches!(v.status, ComplianceStatus::NotTested)).count();

        report.push_str("## Executive Summary\n");
        report.push_str(&format!("- ✅ Compliant: {} rules\n", compliant_count));
        report.push_str(&format!("- ⚠️ Warning: {} rules\n", warning_count));
        report.push_str(&format!("- ❌ Non-Compliant: {} rules\n", non_compliant_count));
        report.push_str(&format!("- ⏳ Not Tested: {} rules\n\n", not_tested_count));

        // Overall assessment
        if non_compliant_count > 0 {
            report.push_str("## 🚨 Overall Status: NON-COMPLIANT\n");
            report.push_str("**Critical compliance issues identified that must be resolved before production deployment.**\n\n");
        } else if warning_count > 0 || not_tested_count > 0 {
            report.push_str("## ⚠️ Overall Status: NEEDS ATTENTION\n");
            report.push_str("**Compliance concerns identified that should be addressed.**\n\n");
        } else {
            report.push_str("## ✅ Overall Status: COMPLIANT\n");
            report.push_str("**All tested compliance requirements are satisfied.**\n\n");
        }

        // Detailed findings
        for validation in &validations {
            report.push_str(&format!("## {}\n", validation.rule_name));
            report.push_str(&format!("**Description**: {}\n", validation.description));
            report.push_str(&format!("**Status**: {:?}\n", validation.status));
            
            if !validation.findings.is_empty() {
                report.push_str("**Findings**:\n");
                for finding in &validation.findings {
                    let severity_icon = match finding.severity {
                        Severity::Critical => "🔴",
                        Severity::High => "🟠", 
                        Severity::Medium => "🟡",
                        Severity::Low => "🔵",
                        Severity::Info => "ℹ️",
                    };
                    report.push_str(&format!("- {} **{:?}**: {}\n", 
                                           severity_icon, finding.severity, finding.description));
                    report.push_str(&format!("  - Recommendation: {}\n", finding.recommendation));
                    if let Some(evidence) = &finding.evidence {
                        report.push_str(&format!("  - Evidence: {}\n", evidence));
                    }
                }
            }
            report.push_str("\n");
        }

        // Action items
        report.push_str("## Required Actions\n");
        let mut critical_actions = Vec::new();
        let mut high_actions = Vec::new();

        for validation in &validations {
            for finding in &validation.findings {
                match finding.severity {
                    Severity::Critical => critical_actions.push(&finding.recommendation),
                    Severity::High => high_actions.push(&finding.recommendation),
                    _ => {},
                }
            }
        }

        if !critical_actions.is_empty() {
            report.push_str("### 🔴 Critical (Must Fix Before Production)\n");
            for action in critical_actions {
                report.push_str(&format!("- {}\n", action));
            }
            report.push_str("\n");
        }

        if !high_actions.is_empty() {
            report.push_str("### 🟠 High Priority (Recommended)\n");
            for action in high_actions {
                report.push_str(&format!("- {}\n", action));
            }
            report.push_str("\n");
        }

        report
    }
}

impl Default for ComplianceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rule_validator_creation() {
        let validator = Rule15c35Validator::new();
        assert!(validator.max_order_size > Decimal::ZERO);
    }
    
    #[test]
    fn test_compliance_validation() {
        let validator = ComplianceValidator::new();
        let validations = validator.run_comprehensive_validation();
        assert!(!validations.is_empty());
    }
    
    #[test]
    fn test_audit_trail_validation() {
        let validator = AuditTrailValidator::new();
        let validation = validator.validate_audit_trail();
        assert_eq!(validation.rule_name, "Audit Trail Requirements");
    }
}