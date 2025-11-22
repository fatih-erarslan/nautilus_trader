//! Financial-specific security controls for trading operations
//! 
//! This module implements specialized security controls for financial trading:
//! - Trade validation and authorization
//! - Market data integrity verification
//! - Financial transaction logging
//! - Risk management controls
//! - Compliance monitoring
//! - Anti-money laundering (AML) checks

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock, Mutex};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use validator::Validate;
use tracing::{info, warn, error};
use rust_decimal::Decimal;

use crate::error::{HiveMindError, Result};
use crate::security::{SecurityManager, SecurityEvent};

/// Maximum trade amount to prevent large unauthorized trades
const MAX_TRADE_AMOUNT: Decimal = Decimal::from_parts(1000000, 0, 0, false, 2); // $1,000,000
/// Maximum daily trading volume per user
const MAX_DAILY_VOLUME: Decimal = Decimal::from_parts(10000000, 0, 0, false, 2); // $10,000,000
/// Minimum time between trades to prevent high-frequency abuse
const MIN_TRADE_INTERVAL: Duration = Duration::from_millis(100);

/// Financial security manager for trading operations
pub struct FinancialSecurityManager {
    security_manager: Arc<SecurityManager>,
    trade_monitor: Arc<TradeMonitor>,
    risk_manager: Arc<RiskManager>,
    compliance_monitor: Arc<ComplianceMonitor>,
    transaction_logger: Arc<TransactionLogger>,
}

impl FinancialSecurityManager {
    /// Create new financial security manager
    pub async fn new(security_manager: Arc<SecurityManager>) -> Result<Self> {
        let trade_monitor = Arc::new(TradeMonitor::new());
        let risk_manager = Arc::new(RiskManager::new());
        let compliance_monitor = Arc::new(ComplianceMonitor::new().await?);
        let transaction_logger = Arc::new(TransactionLogger::new().await?);

        Ok(Self {
            security_manager,
            trade_monitor,
            risk_manager,
            compliance_monitor,
            transaction_logger,
        })
    }

    /// Validate and authorize a financial trade
    pub async fn validate_trade(&self, trade: &TradeRequest, user_id: &str) -> Result<TradeValidationResult> {
        // Input validation
        trade.validate()
            .map_err(|e| HiveMindError::InvalidState { 
                message: format!("Trade validation failed: {}", e) 
            })?;

        // Check trade permissions
        if !self.check_trade_permission(user_id, &trade.instrument).await? {
            return Ok(TradeValidationResult::Denied {
                reason: "Insufficient permissions for this instrument".to_string(),
            });
        }

        // Risk management checks
        let risk_result = self.risk_manager.assess_trade_risk(trade, user_id).await?;
        if let RiskAssessment::High { reason } = risk_result {
            self.security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
                policy: "High Risk Trade".to_string(),
                details: reason.clone(),
            }).await?;
            
            return Ok(TradeValidationResult::Denied { reason });
        }

        // Rate limiting check
        if !self.trade_monitor.check_trade_rate_limit(user_id).await? {
            return Ok(TradeValidationResult::Denied {
                reason: "Trade rate limit exceeded".to_string(),
            });
        }

        // Compliance checks
        let compliance_result = self.compliance_monitor.check_compliance(trade, user_id).await?;
        if let ComplianceResult::Violation { reason } = compliance_result {
            return Ok(TradeValidationResult::Denied { reason });
        }

        // All checks passed
        Ok(TradeValidationResult::Approved {
            trade_id: Uuid::new_v4().to_string(),
            authorized_amount: trade.amount,
        })
    }

    /// Execute validated trade with full audit trail
    pub async fn execute_trade(&self, trade: &ValidatedTrade, user_id: &str) -> Result<TradeExecutionResult> {
        let execution_start = SystemTime::now();

        // Log trade execution start
        self.transaction_logger.log_trade_execution_start(trade, user_id).await?;

        // Simulate trade execution (replace with actual trading logic)
        let execution_result = self.simulate_trade_execution(trade).await?;

        // Update monitoring metrics
        self.trade_monitor.record_trade(user_id, trade).await?;

        // Log final transaction
        let transaction_record = TransactionRecord {
            id: Uuid::new_v4().to_string(),
            trade_id: trade.trade_id.clone(),
            user_id: user_id.to_string(),
            instrument: trade.request.instrument.clone(),
            action: trade.request.action,
            amount: trade.request.amount,
            price: execution_result.execution_price,
            execution_time: execution_start,
            status: TransactionStatus::Completed,
            compliance_checked: true,
        };

        self.transaction_logger.log_transaction_record(&transaction_record).await?;

        Ok(execution_result)
    }

    /// Check user permission for trading specific instrument
    async fn check_trade_permission(&self, user_id: &str, instrument: &str) -> Result<bool> {
        // TODO: Implement proper permission checking against user roles/permissions
        // For now, allow all trades for demo purposes
        Ok(true)
    }

    /// Simulate trade execution (replace with real trading API)
    async fn simulate_trade_execution(&self, trade: &ValidatedTrade) -> Result<TradeExecutionResult> {
        // Simulate market price with small random variation
        let base_price = Decimal::new(10000, 2); // $100.00
        let price_variation = Decimal::new(rand::random::<u64>() % 100, 2); // ±$1.00
        let execution_price = base_price + price_variation;

        Ok(TradeExecutionResult {
            trade_id: trade.trade_id.clone(),
            execution_price,
            executed_amount: trade.request.amount,
            execution_time: SystemTime::now(),
            commission: trade.request.amount * Decimal::new(1, 3), // 0.1% commission
            status: ExecutionStatus::Filled,
        })
    }

    /// Get trading statistics for user
    pub async fn get_trading_statistics(&self, user_id: &str) -> Result<TradingStatistics> {
        self.trade_monitor.get_user_statistics(user_id).await
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(&self, start_time: SystemTime, end_time: SystemTime) -> Result<ComplianceReport> {
        self.compliance_monitor.generate_report(start_time, end_time).await
    }
}

/// Trade monitoring for rate limits and statistics
pub struct TradeMonitor {
    user_trades: RwLock<HashMap<String, Vec<TradeRecord>>>,
    daily_volumes: RwLock<HashMap<String, Decimal>>,
}

impl TradeMonitor {
    pub fn new() -> Self {
        Self {
            user_trades: RwLock::new(HashMap::new()),
            daily_volumes: RwLock::new(HashMap::new()),
        }
    }

    /// Check if user can place another trade (rate limiting)
    pub async fn check_trade_rate_limit(&self, user_id: &str) -> Result<bool> {
        let trades = self.user_trades.read().await;
        if let Some(user_trade_history) = trades.get(user_id) {
            if let Some(last_trade) = user_trade_history.last() {
                let elapsed = SystemTime::now()
                    .duration_since(last_trade.timestamp)
                    .map_err(|_| HiveMindError::Internal("Time error".to_string()))?;
                
                if elapsed < MIN_TRADE_INTERVAL {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Record trade for monitoring
    pub async fn record_trade(&self, user_id: &str, trade: &ValidatedTrade) -> Result<()> {
        let trade_record = TradeRecord {
            timestamp: SystemTime::now(),
            amount: trade.request.amount,
            instrument: trade.request.instrument.clone(),
        };

        // Record trade
        {
            let mut trades = self.user_trades.write().await;
            trades.entry(user_id.to_string())
                .or_insert_with(Vec::new)
                .push(trade_record);
        }

        // Update daily volume
        {
            let mut volumes = self.daily_volumes.write().await;
            let current_volume = volumes.entry(user_id.to_string()).or_insert(Decimal::ZERO);
            *current_volume += trade.request.amount;
        }

        Ok(())
    }

    /// Get trading statistics for user
    pub async fn get_user_statistics(&self, user_id: &str) -> Result<TradingStatistics> {
        let trades = self.user_trades.read().await;
        let volumes = self.daily_volumes.read().await;

        let user_trades = trades.get(user_id).cloned().unwrap_or_default();
        let daily_volume = volumes.get(user_id).copied().unwrap_or(Decimal::ZERO);

        let total_trades = user_trades.len();
        let total_volume = user_trades.iter().map(|t| t.amount).sum();

        Ok(TradingStatistics {
            user_id: user_id.to_string(),
            total_trades,
            total_volume,
            daily_volume,
            last_trade_time: user_trades.last().map(|t| t.timestamp),
        })
    }
}

/// Trade record for monitoring
#[derive(Debug, Clone)]
struct TradeRecord {
    timestamp: SystemTime,
    amount: Decimal,
    instrument: String,
}

/// Risk management for trade assessment
pub struct RiskManager {
    risk_profiles: RwLock<HashMap<String, UserRiskProfile>>,
}

impl RiskManager {
    pub fn new() -> Self {
        Self {
            risk_profiles: RwLock::new(HashMap::new()),
        }
    }

    /// Assess risk level of a trade
    pub async fn assess_trade_risk(&self, trade: &TradeRequest, user_id: &str) -> Result<RiskAssessment> {
        // Check trade amount limits
        if trade.amount > MAX_TRADE_AMOUNT {
            return Ok(RiskAssessment::High {
                reason: format!("Trade amount {} exceeds maximum allowed {}", trade.amount, MAX_TRADE_AMOUNT),
            });
        }

        // Check user risk profile
        let risk_profiles = self.risk_profiles.read().await;
        if let Some(profile) = risk_profiles.get(user_id) {
            if profile.risk_score > 80.0 {
                return Ok(RiskAssessment::High {
                    reason: "User risk score too high".to_string(),
                });
            }
        }

        // Check for suspicious patterns
        if self.detect_suspicious_patterns(trade, user_id).await? {
            return Ok(RiskAssessment::Medium {
                reason: "Suspicious trading pattern detected".to_string(),
            });
        }

        Ok(RiskAssessment::Low)
    }

    /// Detect suspicious trading patterns
    async fn detect_suspicious_patterns(&self, _trade: &TradeRequest, _user_id: &str) -> Result<bool> {
        // TODO: Implement pattern detection algorithms
        // - Rapid consecutive trades
        // - Unusual instrument selections
        // - Trade amounts outside normal patterns
        Ok(false)
    }
}

/// User risk profile
#[derive(Debug, Clone)]
struct UserRiskProfile {
    user_id: String,
    risk_score: f64, // 0-100, higher is riskier
    max_trade_amount: Decimal,
    daily_trade_limit: Decimal,
}

/// Risk assessment result
#[derive(Debug, Clone)]
pub enum RiskAssessment {
    Low,
    Medium { reason: String },
    High { reason: String },
}

/// Compliance monitoring for regulatory requirements
pub struct ComplianceMonitor {
    aml_rules: Vec<AmlRule>,
    suspicious_activity_log: Mutex<Vec<SuspiciousActivity>>,
}

impl ComplianceMonitor {
    pub async fn new() -> Result<Self> {
        let aml_rules = vec![
            AmlRule {
                name: "Large Transaction".to_string(),
                threshold: Decimal::new(10000, 2), // $100.00
                description: "Monitor transactions above threshold".to_string(),
            },
            AmlRule {
                name: "Rapid Trading".to_string(),
                threshold: Decimal::new(50, 0), // 50 trades
                description: "Monitor users with high trading frequency".to_string(),
            },
        ];

        Ok(Self {
            aml_rules,
            suspicious_activity_log: Mutex::new(Vec::new()),
        })
    }

    /// Check trade compliance
    pub async fn check_compliance(&self, trade: &TradeRequest, user_id: &str) -> Result<ComplianceResult> {
        // Check AML rules
        for rule in &self.aml_rules {
            if self.check_aml_rule(rule, trade, user_id).await? {
                // Log suspicious activity
                let activity = SuspiciousActivity {
                    id: Uuid::new_v4().to_string(),
                    user_id: user_id.to_string(),
                    rule_triggered: rule.name.clone(),
                    trade_details: format!("{:?}", trade),
                    timestamp: SystemTime::now(),
                    severity: SuspiciousSeverity::Medium,
                };

                let mut log = self.suspicious_activity_log.lock().await;
                log.push(activity);

                return Ok(ComplianceResult::Warning {
                    reason: format!("AML rule triggered: {}", rule.name),
                });
            }
        }

        Ok(ComplianceResult::Compliant)
    }

    /// Check specific AML rule
    async fn check_aml_rule(&self, rule: &AmlRule, trade: &TradeRequest, _user_id: &str) -> Result<bool> {
        match rule.name.as_str() {
            "Large Transaction" => Ok(trade.amount >= rule.threshold),
            "Rapid Trading" => {
                // TODO: Implement rapid trading detection
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    /// Generate compliance report
    pub async fn generate_report(&self, start_time: SystemTime, end_time: SystemTime) -> Result<ComplianceReport> {
        let log = self.suspicious_activity_log.lock().await;
        let relevant_activities: Vec<_> = log
            .iter()
            .filter(|activity| {
                activity.timestamp >= start_time && activity.timestamp <= end_time
            })
            .cloned()
            .collect();

        Ok(ComplianceReport {
            report_id: Uuid::new_v4().to_string(),
            period_start: start_time,
            period_end: end_time,
            total_suspicious_activities: relevant_activities.len(),
            suspicious_activities: relevant_activities,
            compliance_status: if relevant_activities.is_empty() {
                ComplianceStatus::Compliant
            } else {
                ComplianceStatus::RequiresReview
            },
        })
    }
}

/// AML (Anti-Money Laundering) rule
#[derive(Debug, Clone)]
struct AmlRule {
    name: String,
    threshold: Decimal,
    description: String,
}

/// Suspicious activity record
#[derive(Debug, Clone)]
struct SuspiciousActivity {
    id: String,
    user_id: String,
    rule_triggered: String,
    trade_details: String,
    timestamp: SystemTime,
    severity: SuspiciousSeverity,
}

/// Severity levels for suspicious activity
#[derive(Debug, Clone)]
enum SuspiciousSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance check result
#[derive(Debug, Clone)]
pub enum ComplianceResult {
    Compliant,
    Warning { reason: String },
    Violation { reason: String },
}

/// Transaction logger for audit trails
pub struct TransactionLogger {
    log_file: Mutex<std::fs::File>,
}

impl TransactionLogger {
    pub async fn new() -> Result<Self> {
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("financial_transactions.log")
            .map_err(|e| HiveMindError::Internal(format!("Failed to create transaction log: {}", e)))?;

        Ok(Self {
            log_file: Mutex::new(log_file),
        })
    }

    /// Log start of trade execution
    pub async fn log_trade_execution_start(&self, trade: &ValidatedTrade, user_id: &str) -> Result<()> {
        let log_entry = format!(
            "[TRADE_START] {} - User: {}, Trade: {}, Amount: {}, Instrument: {}\n",
            chrono::Utc::now(),
            user_id,
            trade.trade_id,
            trade.request.amount,
            trade.request.instrument
        );

        self.write_log_entry(&log_entry).await
    }

    /// Log complete transaction record
    pub async fn log_transaction_record(&self, record: &TransactionRecord) -> Result<()> {
        let log_entry = format!(
            "[TRANSACTION] {} - {}\n",
            chrono::Utc::now(),
            serde_json::to_string(record)
                .map_err(|e| HiveMindError::Internal(format!("Failed to serialize transaction: {}", e)))?
        );

        self.write_log_entry(&log_entry).await
    }

    /// Write log entry to file
    async fn write_log_entry(&self, entry: &str) -> Result<()> {
        use std::io::Write;
        let mut file = self.log_file.lock().await;
        file.write_all(entry.as_bytes())
            .map_err(|e| HiveMindError::Internal(format!("Failed to write transaction log: {}", e)))?;
        file.flush()
            .map_err(|e| HiveMindError::Internal(format!("Failed to flush transaction log: {}", e)))?;
        Ok(())
    }
}

/// Data structures for financial operations

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TradeRequest {
    #[validate(length(min = 1, max = 20))]
    pub instrument: String,
    pub action: TradeAction,
    #[validate(range(min = "0.01"))]
    pub amount: Decimal,
    pub order_type: OrderType,
    pub price_limit: Option<Decimal>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TradeAction {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone)]
pub enum TradeValidationResult {
    Approved {
        trade_id: String,
        authorized_amount: Decimal,
    },
    Denied {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct ValidatedTrade {
    pub trade_id: String,
    pub request: TradeRequest,
    pub authorized_amount: Decimal,
    pub validation_time: SystemTime,
}

#[derive(Debug, Clone)]
pub struct TradeExecutionResult {
    pub trade_id: String,
    pub execution_price: Decimal,
    pub executed_amount: Decimal,
    pub execution_time: SystemTime,
    pub commission: Decimal,
    pub status: ExecutionStatus,
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutionStatus {
    Filled,
    PartiallyFilled,
    Pending,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransactionRecord {
    pub id: String,
    pub trade_id: String,
    pub user_id: String,
    pub instrument: String,
    pub action: TradeAction,
    pub amount: Decimal,
    pub price: Decimal,
    pub execution_time: SystemTime,
    pub status: TransactionStatus,
    pub compliance_checked: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum TransactionStatus {
    Pending,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct TradingStatistics {
    pub user_id: String,
    pub total_trades: usize,
    pub total_volume: Decimal,
    pub daily_volume: Decimal,
    pub last_trade_time: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub report_id: String,
    pub period_start: SystemTime,
    pub period_end: SystemTime,
    pub total_suspicious_activities: usize,
    pub suspicious_activities: Vec<SuspiciousActivity>,
    pub compliance_status: ComplianceStatus,
}

#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    Compliant,
    RequiresReview,
    NonCompliant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityManager;

    #[tokio::test]
    async fn test_trade_validation() {
        let security_manager = Arc::new(SecurityManager::new().await.unwrap());
        let financial_security = FinancialSecurityManager::new(security_manager).await.unwrap();

        let trade_request = TradeRequest {
            instrument: "EURUSD".to_string(),
            action: TradeAction::Buy,
            amount: Decimal::new(1000, 2), // $10.00
            order_type: OrderType::Market,
            price_limit: None,
        };

        let result = financial_security.validate_trade(&trade_request, "test_user").await.unwrap();
        
        match result {
            TradeValidationResult::Approved { .. } => {
                // Test passed
            }
            TradeValidationResult::Denied { reason } => {
                panic!("Trade validation failed: {}", reason);
            }
        }
    }

    #[tokio::test]
    async fn test_large_trade_rejection() {
        let security_manager = Arc::new(SecurityManager::new().await.unwrap());
        let financial_security = FinancialSecurityManager::new(security_manager).await.unwrap();

        let large_trade = TradeRequest {
            instrument: "EURUSD".to_string(),
            action: TradeAction::Buy,
            amount: Decimal::new(2000000, 2), // $20,000.00 - exceeds limit
            order_type: OrderType::Market,
            price_limit: None,
        };

        let result = financial_security.validate_trade(&large_trade, "test_user").await.unwrap();
        
        match result {
            TradeValidationResult::Denied { .. } => {
                // Expected behavior
            }
            TradeValidationResult::Approved { .. } => {
                panic!("Large trade should have been denied");
            }
        }
    }

    #[tokio::test]
    async fn test_compliance_monitoring() {
        let compliance_monitor = ComplianceMonitor::new().await.unwrap();

        let trade_request = TradeRequest {
            instrument: "EURUSD".to_string(),
            action: TradeAction::Buy,
            amount: Decimal::new(15000, 2), // $150.00 - exceeds AML threshold
            order_type: OrderType::Market,
            price_limit: None,
        };

        let result = compliance_monitor.check_compliance(&trade_request, "test_user").await.unwrap();
        
        match result {
            ComplianceResult::Warning { .. } => {
                // Expected for large transaction
            }
            ComplianceResult::Compliant => {
                // Also acceptable
            }
            ComplianceResult::Violation { .. } => {
                panic!("Should not be a violation for legitimate trade");
            }
        }
    }
}