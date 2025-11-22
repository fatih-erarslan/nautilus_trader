//! TENGRI Compliance error types with zero tolerance

use thiserror::Error;
use rust_decimal::Decimal;
use uuid::Uuid;

pub type ComplianceResult<T> = Result<T, ComplianceError>;

#[derive(Error, Debug, Clone)]
pub enum ComplianceError {
    #[error("CRITICAL: Position limit violated - current: {current}, limit: {limit}, symbol: {symbol}")]
    PositionLimitViolation {
        symbol: String,
        current: Decimal,
        limit: Decimal,
    },

    #[error("CRITICAL: Leverage constraint violated - current: {current}, max: {max_allowed}")]
    LeverageViolation {
        current: Decimal,
        max_allowed: Decimal,
    },

    #[error("CRITICAL: Risk limit breached - metric: {metric}, value: {value}, threshold: {threshold}")]
    RiskLimitBreach {
        metric: String,
        value: Decimal,
        threshold: Decimal,
    },

    #[error("CRITICAL: Market manipulation detected - pattern: {pattern}, confidence: {confidence}%")]
    MarketManipulation {
        pattern: String,
        confidence: f64,
    },

    #[error("CRITICAL: Regulatory violation - rule: {rule}, details: {details}")]
    RegulatoryViolation {
        rule: String,
        details: String,
    },

    #[error("CRITICAL: Wash trading detected - trades: {trade_ids:?}")]
    WashTradingDetected {
        trade_ids: Vec<Uuid>,
    },

    #[error("CRITICAL: Spoofing detected - orders: {order_ids:?}, confidence: {confidence}%")]
    SpoofingDetected {
        order_ids: Vec<Uuid>,
        confidence: f64,
    },

    #[error("CRITICAL: Circuit breaker triggered - reason: {reason}")]
    CircuitBreakerTriggered {
        reason: String,
    },

    #[error("CRITICAL: Kill switch activated - trigger: {trigger}")]
    KillSwitchActivated {
        trigger: String,
    },

    #[error("CRITICAL: Compliance engine failure - component: {component}, error: {error}")]
    EngineFailure {
        component: String,
        error: String,
    },

    #[error("CRITICAL: Audit trail failure - cannot proceed without audit")]
    AuditFailure,

    #[error("CRITICAL: Rule validation failed - rule: {rule_id}, reason: {reason}")]
    RuleValidationFailed {
        rule_id: String,
        reason: String,
    },

    #[error("CRITICAL: Unauthorized trading attempt - trader: {trader_id}, reason: {reason}")]
    UnauthorizedTrading {
        trader_id: String,
        reason: String,
    },

    #[error("CRITICAL: Concentration risk exceeded - asset: {asset}, concentration: {concentration}%")]
    ConcentrationRiskExceeded {
        asset: String,
        concentration: f64,
    },

    #[error("CRITICAL: Daily loss limit exceeded - current: {current_loss}, limit: {daily_limit}")]
    DailyLossLimitExceeded {
        current_loss: Decimal,
        daily_limit: Decimal,
    },

    #[error("CRITICAL: Trade frequency violation - current: {current_rate}/min, max: {max_rate}/min")]
    TradeFrequencyViolation {
        current_rate: u32,
        max_rate: u32,
    },

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Severity levels for compliance violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational - logged but doesn't block
    Info,
    /// Warning - may require review
    Warning,
    /// Critical - blocks execution immediately
    Critical,
    /// Fatal - triggers emergency shutdown
    Fatal,
}

impl ComplianceError {
    pub fn severity(&self) -> Severity {
        match self {
            ComplianceError::Internal(_) => Severity::Warning,
            ComplianceError::CircuitBreakerTriggered { .. } => Severity::Fatal,
            ComplianceError::KillSwitchActivated { .. } => Severity::Fatal,
            ComplianceError::EngineFailure { .. } => Severity::Fatal,
            ComplianceError::AuditFailure => Severity::Fatal,
            _ => Severity::Critical,
        }
    }

    pub fn is_fatal(&self) -> bool {
        self.severity() == Severity::Fatal
    }

    pub fn requires_immediate_action(&self) -> bool {
        matches!(self.severity(), Severity::Critical | Severity::Fatal)
    }
}