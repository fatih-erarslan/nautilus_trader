//! Error types for HyperRiskEngine.
//!
//! Designed for zero-allocation in fast path through pre-allocated error variants.

use thiserror::Error;

/// Result type alias for HyperRiskEngine operations.
pub type Result<T> = std::result::Result<T, RiskError>;

/// Comprehensive error type for risk engine operations.
#[derive(Debug, Error)]
pub enum RiskError {
    // ========================================================================
    // Fast-Path Errors (Pre-allocated, no allocation)
    // ========================================================================

    /// Global kill switch activated - halt all trading immediately.
    #[error("KILL SWITCH ACTIVATED: {reason}")]
    KillSwitchActivated {
        /// Reason for kill switch activation.
        reason: &'static str,
    },

    // ========================================================================
    // Position Limit Errors
    // ========================================================================

    /// Position limit exceeded (with heap allocation for dynamic symbol).
    #[error("Position limit exceeded for {symbol}: current={current:.2}, attempted={attempted:.2}, limit={limit:.2}")]
    PositionLimitExceeded {
        /// Symbol that exceeded limit.
        symbol: String,
        /// Current position value.
        current: f64,
        /// Attempted order value.
        attempted: f64,
        /// Maximum allowed position.
        limit: f64,
    },

    /// Concentration limit exceeded.
    #[error("Concentration limit exceeded for {symbol}: {concentration:.2}% > {limit:.2}%")]
    ConcentrationLimitExceeded {
        /// Symbol.
        symbol: String,
        /// Current concentration.
        concentration: f64,
        /// Maximum allowed.
        limit: f64,
    },

    /// Total exposure limit exceeded.
    #[error("Exposure limit exceeded: current={current:.2}, attempted={attempted:.2}, limit={limit:.2}")]
    ExposureLimitExceeded {
        /// Current exposure.
        current: f64,
        /// Attempted addition.
        attempted: f64,
        /// Maximum allowed.
        limit: f64,
    },

    /// Maximum positions count exceeded.
    #[error("Maximum positions exceeded: current={current}, limit={limit}")]
    MaxPositionsExceeded {
        /// Current number of positions.
        current: usize,
        /// Maximum allowed positions.
        limit: usize,
    },

    // ========================================================================
    // Drawdown Errors
    // ========================================================================

    /// Drawdown limit exceeded.
    #[error("Drawdown limit exceeded: current={current:.2}%, limit={limit:.2}%")]
    DrawdownLimitExceeded {
        /// Current drawdown.
        current: f64,
        /// Maximum allowed drawdown.
        limit: f64,
    },

    /// Emergency drawdown threshold breached.
    #[error("EMERGENCY DRAWDOWN: {current:.2}% exceeds emergency threshold {threshold:.2}%")]
    EmergencyDrawdown {
        /// Current drawdown.
        current: f64,
        /// Emergency threshold.
        threshold: f64,
    },

    /// Daily loss limit exceeded.
    #[error("Daily loss limit exceeded: loss={loss:.2}%, limit={limit:.2}%")]
    DailyLossLimitExceeded {
        /// Current daily loss.
        loss: f64,
        /// Daily limit.
        limit: f64,
    },

    /// Weekly loss limit exceeded.
    #[error("Weekly loss limit exceeded: loss={loss:.2}%, limit={limit:.2}%")]
    WeeklyLossLimitExceeded {
        /// Current weekly loss.
        loss: f64,
        /// Weekly limit.
        limit: f64,
    },

    // ========================================================================
    // Circuit Breaker Errors
    // ========================================================================

    /// Circuit breaker tripped.
    #[error("Circuit breaker tripped: {reason}")]
    CircuitBreakerTripped {
        /// Reason for circuit breaker activation.
        reason: &'static str,
    },

    /// Circuit breaker is open (cooldown active).
    #[error("Circuit breaker open: {remaining_cooldown_ns}ns remaining")]
    CircuitBreakerOpen {
        /// Remaining cooldown in nanoseconds.
        remaining_cooldown_ns: u64,
    },

    /// Circuit breaker lockout (too many trips).
    #[error("Circuit breaker lockout: {trips} trips exceeds max {max}")]
    CircuitBreakerLockout {
        /// Number of trips.
        trips: u64,
        /// Maximum allowed trips.
        max: u32,
    },

    // ========================================================================
    // VaR Errors
    // ========================================================================

    /// VaR limit exceeded.
    #[error("VaR limit exceeded: VaR={var:.4} > limit={limit:.4} at {confidence:.0}% confidence")]
    VaRLimitExceeded {
        /// Current VaR.
        var: f64,
        /// VaR limit.
        limit: f64,
        /// Confidence level.
        confidence: f64,
    },

    /// CVaR (Expected Shortfall) limit exceeded.
    #[error("CVaR limit exceeded: CVaR={cvar:.4} > limit={limit:.4} at {confidence:.0}% confidence")]
    CVaRLimitExceeded {
        /// Current CVaR.
        cvar: f64,
        /// CVaR limit.
        limit: f64,
        /// Confidence level.
        confidence: f64,
    },

    /// Stress VaR limit exceeded.
    #[error("Stress VaR exceeded: stress_var={stress_var:.4} > limit={limit:.4}")]
    StressVaRExceeded {
        /// Stress VaR value.
        stress_var: f64,
        /// Stress limit.
        limit: f64,
    },

    // ========================================================================
    // Whale Detection Errors
    // ========================================================================

    /// Whale order detected (large relative to ADV).
    #[error("Whale order detected: size={order_size:.2} is {adv_pct:.2}% of ADV > {threshold:.2}%")]
    WhaleOrderDetected {
        /// Order size.
        order_size: f64,
        /// Percentage of ADV.
        adv_pct: f64,
        /// Threshold.
        threshold: f64,
    },

    /// Large notional order.
    #[error("Large notional order: ${notional:.2} > ${threshold:.2}")]
    LargeNotionalOrder {
        /// Order notional value.
        notional: f64,
        /// Notional threshold.
        threshold: f64,
    },

    /// Large portfolio percentage order.
    #[error("Large portfolio order: {order_pct:.2}% of portfolio > {threshold:.2}%")]
    LargePortfolioOrder {
        /// Order as percentage of portfolio.
        order_pct: f64,
        /// Threshold.
        threshold: f64,
    },

    /// Order flow imbalance detected.
    #[error("Flow imbalance detected: imbalance={imbalance:.2} > {threshold:.2}")]
    FlowImbalanceDetected {
        /// Flow imbalance.
        imbalance: f64,
        /// Threshold.
        threshold: f64,
    },

    /// Toxic flow detected (high VPIN).
    #[error("Toxic flow detected: VPIN={vpin:.3} > {threshold:.3}")]
    ToxicFlowDetected {
        /// VPIN value.
        vpin: f64,
        /// VPIN threshold.
        threshold: f64,
    },

    /// Whale activity detected (legacy).
    #[error("Whale activity detected: volume={volume:.2}, threshold={threshold:.2}")]
    WhaleActivityDetected {
        /// Detected volume.
        volume: f64,
        /// Alert threshold.
        threshold: f64,
    },

    // ========================================================================
    // Medium-Path Errors
    // ========================================================================

    /// Regime detection failed.
    #[error("Regime detection failed: {0}")]
    RegimeDetectionFailed(String),

    /// Correlation matrix computation failed.
    #[error("Correlation computation failed: {0}")]
    CorrelationComputationFailed(String),

    /// Kelly criterion calculation failed.
    #[error("Kelly criterion error: {0}")]
    KellyCriterionError(String),

    /// Position sizing failed.
    #[error("Position sizing failed: {0}")]
    PositionSizingFailed(String),

    // ========================================================================
    // Slow-Path Errors
    // ========================================================================

    /// Monte Carlo simulation failed.
    #[error("Monte Carlo simulation failed: {0}")]
    MonteCarloFailed(String),

    /// Stress test failed.
    #[error("Stress test failed: {0}")]
    StressTestFailed(String),

    /// Stress test breach (scenario exceeds loss threshold).
    #[error("Stress test breach: {0}")]
    StressTestBreach(String),

    /// FRTB calculation failed.
    #[error("FRTB calculation failed: {0}")]
    FRTBCalculationFailed(String),

    // ========================================================================
    // Data & Validation Errors
    // ========================================================================

    /// Insufficient data for calculation.
    #[error("Insufficient data: {message} (required={required}, available={available})")]
    InsufficientData {
        /// Error message.
        message: String,
        /// Required data points.
        required: usize,
        /// Available data points.
        available: usize,
    },

    /// Invalid parameter.
    #[error("Invalid parameter: {param}={value}, constraint: {constraint}")]
    InvalidParameter {
        /// Parameter name.
        param: &'static str,
        /// Invalid value.
        value: String,
        /// Constraint description.
        constraint: &'static str,
    },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    // ========================================================================
    // System Errors
    // ========================================================================

    /// Engine not initialized.
    #[error("Engine not initialized")]
    EngineNotInitialized,

    /// Sentinel not found.
    #[error("Sentinel not found: {0}")]
    SentinelNotFound(String),

    /// Agent not found.
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    /// Concurrency error.
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// Internal error (should not happen).
    #[error("Internal error: {0}")]
    InternalError(String),

    // ========================================================================
    // EVT/Tail Risk Errors
    // ========================================================================

    /// EVT parameter estimation failed.
    #[error("EVT parameter estimation failed: {0}")]
    EVTEstimationFailed(String),

    /// GPD fitting failed.
    #[error("GPD fitting failed: {0}")]
    GPDFittingFailed(String),

    /// Entropy constraint violation.
    #[error("Entropy constraint violation: {0}")]
    EntropyConstraintViolation(String),

    /// Threshold selection failed.
    #[error("Threshold selection failed: {0}")]
    ThresholdSelectionFailed(String),
}

impl RiskError {
    /// Check if this error should trigger immediate halt.
    #[inline]
    pub const fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::KillSwitchActivated { .. }
                | Self::EmergencyDrawdown { .. }
                | Self::CircuitBreakerTripped { .. }
                | Self::CircuitBreakerLockout { .. }
        )
    }

    /// Check if this error is recoverable.
    #[inline]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::PositionLimitExceeded { .. }
                | Self::ConcentrationLimitExceeded { .. }
                | Self::VaRLimitExceeded { .. }
                | Self::CVaRLimitExceeded { .. }
                | Self::InsufficientData { .. }
                | Self::CircuitBreakerOpen { .. }
        )
    }

    /// Get error severity level (0-4, higher is more severe).
    #[inline]
    pub const fn severity(&self) -> u8 {
        match self {
            // Emergency - immediate halt
            Self::KillSwitchActivated { .. } | Self::EmergencyDrawdown { .. } => 4,
            // Critical - require manual intervention
            Self::CircuitBreakerTripped { .. }
            | Self::CircuitBreakerLockout { .. }
            | Self::DrawdownLimitExceeded { .. } => 3,
            // High - restrict operations
            Self::DailyLossLimitExceeded { .. }
            | Self::WeeklyLossLimitExceeded { .. }
            | Self::VaRLimitExceeded { .. }
            | Self::CVaRLimitExceeded { .. }
            | Self::StressVaRExceeded { .. } => 2,
            // Medium - reject order but continue
            Self::PositionLimitExceeded { .. }
            | Self::ConcentrationLimitExceeded { .. }
            | Self::ExposureLimitExceeded { .. }
            | Self::MaxPositionsExceeded { .. }
            | Self::WhaleOrderDetected { .. }
            | Self::LargeNotionalOrder { .. }
            | Self::LargePortfolioOrder { .. }
            | Self::FlowImbalanceDetected { .. }
            | Self::ToxicFlowDetected { .. }
            | Self::WhaleActivityDetected { .. }
            | Self::CircuitBreakerOpen { .. } => 1,
            // Low - informational
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_errors() {
        let err = RiskError::KillSwitchActivated {
            reason: "Manual trigger",
        };
        assert!(err.is_critical());
        assert_eq!(err.severity(), 4);
    }

    #[test]
    fn test_emergency_drawdown() {
        let err = RiskError::EmergencyDrawdown {
            current: 0.20,
            threshold: 0.15,
        };
        assert!(err.is_critical());
        assert_eq!(err.severity(), 4);
    }

    #[test]
    fn test_recoverable_errors() {
        let err = RiskError::PositionLimitExceeded {
            symbol: "AAPL".to_string(),
            current: 50000.0,
            attempted: 60000.0,
            limit: 100000.0,
        };
        assert!(err.is_recoverable());
        assert!(!err.is_critical());
        assert_eq!(err.severity(), 1);
    }

    #[test]
    fn test_var_error() {
        let err = RiskError::VaRLimitExceeded {
            var: 0.025,
            limit: 0.02,
            confidence: 0.95,
        };
        assert!(!err.is_critical());
        assert!(err.is_recoverable());
        assert_eq!(err.severity(), 2);
    }

    #[test]
    fn test_circuit_breaker_open() {
        let err = RiskError::CircuitBreakerOpen {
            remaining_cooldown_ns: 30_000_000_000,
        };
        assert!(!err.is_critical());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = RiskError::DrawdownLimitExceeded {
            current: 0.185,
            limit: 0.15,
        };
        let msg = format!("{err}");
        // Format uses .2 precision so 0.185 -> "0.19" and 0.15 -> "0.15"
        assert!(msg.contains("0.19") || msg.contains("0.18"), "Expected percentage in message: {}", msg);
        assert!(msg.contains("0.15"), "Expected limit percentage in message: {}", msg);
    }
}
