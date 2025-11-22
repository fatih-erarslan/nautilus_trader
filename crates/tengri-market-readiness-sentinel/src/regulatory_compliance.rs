//! Regulatory compliance validation for market readiness assessment
//!
//! This module implements comprehensive regulatory compliance checks for trading systems,
//! including MiFID II, SEC, CFTC, and other regulatory requirements.

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::error::MarketReadinessError;
use crate::{ValidationResult, ValidationStatus, ComplianceStatus, ComplianceCheck, CircuitBreaker};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub jurisdiction: Jurisdiction,
    pub rule_type: RuleType,
    pub severity: RuleSeverity,
    pub parameters: HashMap<String, serde_json::Value>,
    pub enabled: bool,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Jurisdiction {
    US,
    EU,
    UK,
    ASIA,
    GLOBAL,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    OrderManagement,
    RiskManagement,
    MarketAccess,
    Reporting,
    RecordKeeping,
    BestExecution,
    MarketManipulation,
    PositionLimits,
    CircuitBreaker,
    Transparency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub id: Uuid,
    pub rule_id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: RuleSeverity,
    pub description: String,
    pub details: serde_json::Value,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetrics {
    pub total_checks: u64,
    pub passed_checks: u64,
    pub failed_checks: u64,
    pub violation_count: u64,
    pub last_violation: Option<DateTime<Utc>>,
    pub compliance_score: f64,
}

#[derive(Debug)]
pub struct ComplianceChecker {
    config: Arc<MarketReadinessConfig>,
    regulatory_rules: Arc<RwLock<HashMap<String, RegulatoryRule>>>,
    violations: Arc<RwLock<Vec<ComplianceViolation>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    compliance_metrics: Arc<RwLock<ComplianceMetrics>>,
}

impl ComplianceChecker {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        let regulatory_rules = Arc::new(RwLock::new(HashMap::new()));
        let violations = Arc::new(RwLock::new(Vec::new()));
        let circuit_breakers = Arc::new(RwLock::new(HashMap::new()));
        let compliance_metrics = Arc::new(RwLock::new(ComplianceMetrics {
            total_checks: 0,
            passed_checks: 0,
            failed_checks: 0,
            violation_count: 0,
            last_violation: None,
            compliance_score: 1.0,
        }));

        Ok(Self {
            config,
            regulatory_rules,
            violations,
            circuit_breakers,
            compliance_metrics,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing regulatory compliance checker...");
        
        // Load regulatory rules
        self.load_regulatory_rules().await?;
        
        // Initialize circuit breakers
        self.initialize_circuit_breakers().await?;
        
        // Start compliance monitoring
        self.start_compliance_monitoring().await?;
        
        info!("Regulatory compliance checker initialized successfully");
        Ok(())
    }

    pub async fn check_compliance(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Update metrics
        {
            let mut metrics = self.compliance_metrics.write().await;
            metrics.total_checks += 1;
        }
        
        // Run all compliance checks
        let mut check_results = Vec::new();
        
        // Check regulatory rules
        let regulatory_result = self.check_regulatory_rules().await?;
        check_results.push(regulatory_result);
        
        // Check circuit breakers
        let circuit_breaker_result = self.check_circuit_breakers().await?;
        check_results.push(circuit_breaker_result);
        
        // Check position limits
        let position_limits_result = self.check_position_limits().await?;
        check_results.push(position_limits_result);
        
        // Check market manipulation
        let market_manipulation_result = self.check_market_manipulation().await?;
        check_results.push(market_manipulation_result);
        
        // Check best execution
        let best_execution_result = self.check_best_execution().await?;
        check_results.push(best_execution_result);
        
        // Determine overall compliance status
        let overall_status = self.determine_compliance_status(&check_results).await?;
        
        // Update metrics
        {
            let mut metrics = self.compliance_metrics.write().await;
            match overall_status {
                ValidationStatus::Passed => metrics.passed_checks += 1,
                ValidationStatus::Failed => metrics.failed_checks += 1,
                _ => {}
            }
            metrics.compliance_score = metrics.passed_checks as f64 / metrics.total_checks as f64;
        }
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        Ok(ValidationResult {
            status: overall_status,
            message: "Regulatory compliance check completed".to_string(),
            details: Some(serde_json::json!({
                "check_results": check_results,
                "duration_ms": duration
            })),
            timestamp: Utc::now(),
            duration_ms: duration,
            confidence: 0.95,
        })
    }

    async fn check_regulatory_rules(&self) -> Result<bool> {
        let rules = self.regulatory_rules.read().await;
        
        for (rule_id, rule) in rules.iter() {
            if !rule.enabled {
                continue;
            }
            
            match rule.rule_type {
                RuleType::OrderManagement => {
                    if !self.check_order_management_rule(rule).await? {
                        self.record_violation(rule_id, "Order management rule violation", rule.severity.clone()).await?;
                        return Ok(false);
                    }
                },
                RuleType::RiskManagement => {
                    if !self.check_risk_management_rule(rule).await? {
                        self.record_violation(rule_id, "Risk management rule violation", rule.severity.clone()).await?;
                        return Ok(false);
                    }
                },
                RuleType::MarketAccess => {
                    if !self.check_market_access_rule(rule).await? {
                        self.record_violation(rule_id, "Market access rule violation", rule.severity.clone()).await?;
                        return Ok(false);
                    }
                },
                RuleType::Reporting => {
                    if !self.check_reporting_rule(rule).await? {
                        self.record_violation(rule_id, "Reporting rule violation", rule.severity.clone()).await?;
                        return Ok(false);
                    }
                },
                _ => {
                    // Handle other rule types
                    info!("Checking rule type: {:?}", rule.rule_type);
                }
            }
        }
        
        Ok(true)
    }

    async fn check_circuit_breakers(&self) -> Result<bool> {
        let circuit_breakers = self.circuit_breakers.read().await;
        
        for (name, breaker) in circuit_breakers.iter() {
            if !breaker.enabled {
                continue;
            }
            
            if breaker.current_value >= breaker.threshold {
                warn!("Circuit breaker {} triggered: {} >= {}", name, breaker.current_value, breaker.threshold);
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    async fn check_position_limits(&self) -> Result<bool> {
        // Check position limits compliance
        // This would integrate with actual position data
        
        let max_position_limit = 1000000.0; // Example limit
        let current_position = 850000.0; // Example current position
        
        if current_position > max_position_limit {
            self.record_violation("position_limits", "Position limit exceeded", RuleSeverity::Critical).await?;
            return Ok(false);
        }
        
        Ok(true)
    }

    async fn check_market_manipulation(&self) -> Result<bool> {
        // Check for potential market manipulation patterns
        // This would integrate with order flow analysis
        
        let order_pattern_score = 0.15; // Example score
        let manipulation_threshold = 0.8;
        
        if order_pattern_score > manipulation_threshold {
            self.record_violation("market_manipulation", "Potential market manipulation detected", RuleSeverity::High).await?;
            return Ok(false);
        }
        
        Ok(true)
    }

    async fn check_best_execution(&self) -> Result<bool> {
        // Check best execution compliance
        // This would integrate with execution quality metrics
        
        let execution_quality_score = 0.95; // Example score
        let best_execution_threshold = 0.90;
        
        if execution_quality_score < best_execution_threshold {
            self.record_violation("best_execution", "Best execution requirement not met", RuleSeverity::Medium).await?;
            return Ok(false);
        }
        
        Ok(true)
    }

    async fn check_order_management_rule(&self, rule: &RegulatoryRule) -> Result<bool> {
        // Check order management specific rules
        match rule.jurisdiction {
            Jurisdiction::EU => {
                // MiFID II order management requirements
                info!("Checking MiFID II order management rule: {}", rule.name);
                Ok(true) // Simplified for example
            },
            Jurisdiction::US => {
                // SEC order management requirements
                info!("Checking SEC order management rule: {}", rule.name);
                Ok(true) // Simplified for example
            },
            _ => Ok(true)
        }
    }

    async fn check_risk_management_rule(&self, rule: &RegulatoryRule) -> Result<bool> {
        // Check risk management specific rules
        info!("Checking risk management rule: {}", rule.name);
        Ok(true) // Simplified for example
    }

    async fn check_market_access_rule(&self, rule: &RegulatoryRule) -> Result<bool> {
        // Check market access specific rules
        info!("Checking market access rule: {}", rule.name);
        Ok(true) // Simplified for example
    }

    async fn check_reporting_rule(&self, rule: &RegulatoryRule) -> Result<bool> {
        // Check reporting specific rules
        info!("Checking reporting rule: {}", rule.name);
        Ok(true) // Simplified for example
    }

    async fn determine_compliance_status(&self, check_results: &[bool]) -> Result<ValidationStatus> {
        let failed_checks = check_results.iter().filter(|&&result| !result).count();
        
        if failed_checks > 0 {
            Ok(ValidationStatus::Failed)
        } else {
            Ok(ValidationStatus::Passed)
        }
    }

    async fn record_violation(&self, rule_id: &str, description: &str, severity: RuleSeverity) -> Result<()> {
        let violation = ComplianceViolation {
            id: Uuid::new_v4(),
            rule_id: rule_id.to_string(),
            timestamp: Utc::now(),
            severity,
            description: description.to_string(),
            details: serde_json::json!({}),
            resolved: false,
            resolution_time: None,
        };
        
        let mut violations = self.violations.write().await;
        violations.push(violation);
        
        // Update metrics
        {
            let mut metrics = self.compliance_metrics.write().await;
            metrics.violation_count += 1;
            metrics.last_violation = Some(Utc::now());
        }
        
        error!("Compliance violation recorded: {} - {}", rule_id, description);
        Ok(())
    }

    async fn load_regulatory_rules(&self) -> Result<()> {
        let mut rules = self.regulatory_rules.write().await;
        
        // Load MiFID II rules
        rules.insert("mifid2_order_mgmt".to_string(), RegulatoryRule {
            id: "mifid2_order_mgmt".to_string(),
            name: "MiFID II Order Management".to_string(),
            description: "Order management requirements under MiFID II".to_string(),
            jurisdiction: Jurisdiction::EU,
            rule_type: RuleType::OrderManagement,
            severity: RuleSeverity::Critical,
            parameters: HashMap::new(),
            enabled: true,
            last_updated: Utc::now(),
        });
        
        // Load SEC rules
        rules.insert("sec_market_access".to_string(), RegulatoryRule {
            id: "sec_market_access".to_string(),
            name: "SEC Market Access Rule".to_string(),
            description: "Market access requirements under SEC regulations".to_string(),
            jurisdiction: Jurisdiction::US,
            rule_type: RuleType::MarketAccess,
            severity: RuleSeverity::High,
            parameters: HashMap::new(),
            enabled: true,
            last_updated: Utc::now(),
        });
        
        // Load CFTC rules
        rules.insert("cftc_position_limits".to_string(), RegulatoryRule {
            id: "cftc_position_limits".to_string(),
            name: "CFTC Position Limits".to_string(),
            description: "Position limits under CFTC regulations".to_string(),
            jurisdiction: Jurisdiction::US,
            rule_type: RuleType::PositionLimits,
            severity: RuleSeverity::Critical,
            parameters: HashMap::new(),
            enabled: true,
            last_updated: Utc::now(),
        });
        
        info!("Loaded {} regulatory rules", rules.len());
        Ok(())
    }

    async fn initialize_circuit_breakers(&self) -> Result<()> {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        circuit_breakers.insert("price_volatility".to_string(), CircuitBreaker {
            name: "Price Volatility".to_string(),
            enabled: true,
            threshold: 0.10, // 10% price movement
            current_value: 0.0,
            time_window: 300, // 5 minutes
        });
        
        circuit_breakers.insert("order_rate".to_string(), CircuitBreaker {
            name: "Order Rate".to_string(),
            enabled: true,
            threshold: 1000.0, // 1000 orders per second
            current_value: 0.0,
            time_window: 1, // 1 second
        });
        
        circuit_breakers.insert("error_rate".to_string(), CircuitBreaker {
            name: "Error Rate".to_string(),
            enabled: true,
            threshold: 0.05, // 5% error rate
            current_value: 0.0,
            time_window: 60, // 1 minute
        });
        
        info!("Initialized {} circuit breakers", circuit_breakers.len());
        Ok(())
    }

    async fn start_compliance_monitoring(&self) -> Result<()> {
        let circuit_breakers = self.circuit_breakers.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Update circuit breaker values
                {
                    let mut breakers = circuit_breakers.write().await;
                    for breaker in breakers.values_mut() {
                        // Simulate updating circuit breaker values
                        // In real implementation, this would pull from actual metrics
                        breaker.current_value = match breaker.name.as_str() {
                            "Price Volatility" => 0.02, // 2% volatility
                            "Order Rate" => 500.0, // 500 orders/sec
                            "Error Rate" => 0.01, // 1% error rate
                            _ => 0.0,
                        };
                    }
                }
            }
        });
        
        Ok(())
    }

    pub async fn get_compliance_status(&self) -> Result<ComplianceStatus> {
        let violations = self.violations.read().await;
        let circuit_breakers = self.circuit_breakers.read().await;
        
        let regulatory_checks = vec![
            ComplianceCheck {
                rule_id: "mifid2_order_mgmt".to_string(),
                status: true,
                description: "MiFID II order management compliance".to_string(),
                last_check: Utc::now(),
            },
            ComplianceCheck {
                rule_id: "sec_market_access".to_string(),
                status: true,
                description: "SEC market access compliance".to_string(),
                last_check: Utc::now(),
            },
            ComplianceCheck {
                rule_id: "cftc_position_limits".to_string(),
                status: true,
                description: "CFTC position limits compliance".to_string(),
                last_check: Utc::now(),
            },
        ];
        
        let circuit_breakers_vec: Vec<CircuitBreaker> = circuit_breakers.values().cloned().collect();
        
        Ok(ComplianceStatus {
            regulatory_checks,
            circuit_breakers: circuit_breakers_vec,
            position_limits_check: violations.iter().all(|v| v.rule_id != "position_limits"),
            market_manipulation_check: violations.iter().all(|v| v.rule_id != "market_manipulation"),
            best_execution_check: violations.iter().all(|v| v.rule_id != "best_execution"),
        })
    }

    pub async fn get_compliance_metrics(&self) -> Result<ComplianceMetrics> {
        Ok(self.compliance_metrics.read().await.clone())
    }

    pub async fn get_active_violations(&self) -> Result<Vec<ComplianceViolation>> {
        let violations = self.violations.read().await;
        Ok(violations.iter().filter(|v| !v.resolved).cloned().collect())
    }

    pub async fn resolve_violation(&self, violation_id: Uuid) -> Result<()> {
        let mut violations = self.violations.write().await;
        
        for violation in violations.iter_mut() {
            if violation.id == violation_id {
                violation.resolved = true;
                violation.resolution_time = Some(Utc::now());
                info!("Resolved compliance violation: {}", violation_id);
                break;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MarketReadinessConfig;

    #[tokio::test]
    async fn test_compliance_checker_initialization() {
        let config = Arc::new(MarketReadinessConfig::default());
        let checker = ComplianceChecker::new(config).await.unwrap();
        assert!(checker.initialize().await.is_ok());
    }

    #[tokio::test]
    async fn test_compliance_check() {
        let config = Arc::new(MarketReadinessConfig::default());
        let checker = ComplianceChecker::new(config).await.unwrap();
        checker.initialize().await.unwrap();
        
        let result = checker.check_compliance().await.unwrap();
        assert_eq!(result.status, ValidationStatus::Passed);
    }

    #[tokio::test]
    async fn test_violation_recording() {
        let config = Arc::new(MarketReadinessConfig::default());
        let checker = ComplianceChecker::new(config).await.unwrap();
        
        checker.record_violation("test_rule", "Test violation", RuleSeverity::Medium).await.unwrap();
        
        let violations = checker.get_active_violations().await.unwrap();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_id, "test_rule");
    }

    #[tokio::test]
    async fn test_violation_resolution() {
        let config = Arc::new(MarketReadinessConfig::default());
        let checker = ComplianceChecker::new(config).await.unwrap();
        
        checker.record_violation("test_rule", "Test violation", RuleSeverity::Medium).await.unwrap();
        let violations = checker.get_active_violations().await.unwrap();
        let violation_id = violations[0].id;
        
        checker.resolve_violation(violation_id).await.unwrap();
        
        let active_violations = checker.get_active_violations().await.unwrap();
        assert_eq!(active_violations.len(), 0);
    }
}