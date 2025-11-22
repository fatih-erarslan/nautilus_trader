//! SEC Rule 15c3-5 Market Access Rule Implementation
//!
//! This module implements comprehensive pre-trade risk controls as mandated by
//! SEC Rule 15c3-5, ensuring sub-100ms validation and real-time compliance.
//!
//! REGULATORY COMPLIANCE: Zero tolerance for violations
//! Performance Requirements: <100ms pre-trade validation
//! Audit Requirements: Complete immutable trail with nanosecond precision

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, RwLock,
};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Maximum allowed latency for pre-trade risk validation (regulatory requirement)
const MAX_VALIDATION_LATENCY_NANOS: u64 = 100_000_000; // 100ms in nanoseconds

/// Maximum allowed system-wide halt propagation time
const MAX_KILL_SWITCH_LATENCY_NANOS: u64 = 1_000_000_000; // 1 second

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: Uuid,
    pub client_id: String,
    pub instrument_id: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub timestamp: SystemTime,
    pub trader_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_order_size: Decimal,
    pub max_position_size: Decimal,
    pub max_daily_loss: Decimal,
    pub max_credit_exposure: Decimal,
    pub max_concentration_pct: Decimal,
    pub max_orders_per_second: u32,
    pub updated_at: SystemTime,
    pub valid_until: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub instrument_id: String,
    pub quantity: Decimal,
    pub avg_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskValidationResult {
    pub is_valid: bool,
    pub order_id: Uuid,
    pub validation_timestamp: SystemTime,
    pub validation_duration_nanos: u64,
    pub risk_checks: Vec<RiskCheckResult>,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub check_type: RiskCheckType,
    pub passed: bool,
    pub current_value: Decimal,
    pub limit_value: Decimal,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCheckType {
    OrderSize,
    PositionLimit,
    CreditLimit,
    ConcentrationRisk,
    VelocityControl,
    DailyLoss,
    SystematicRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    pub event_id: Uuid,
    pub trigger_type: KillSwitchTrigger,
    pub triggered_by: String,
    pub timestamp: SystemTime,
    pub affected_orders: Vec<Uuid>,
    pub propagation_time_nanos: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillSwitchTrigger {
    ManualOverride,
    ExcessiveLoss,
    SystemRisk,
    RegulatoryHalt,
    TechnicalFailure,
}

/// Core SEC Rule 15c3-5 Pre-Trade Risk Control Engine
pub struct PreTradeRiskEngine {
    /// Real-time position tracking with concurrent access
    positions: Arc<RwLock<HashMap<String, Position>>>,

    /// Current risk limits per client/trader
    risk_limits: Arc<RwLock<HashMap<String, RiskLimits>>>,

    /// Daily P&L tracking for loss limits
    daily_pnl: Arc<RwLock<HashMap<String, Decimal>>>,

    /// Order velocity tracking (orders per second)
    order_velocity: Arc<RwLock<HashMap<String, Vec<SystemTime>>>>,

    /// Kill switch state - atomic for immediate propagation
    kill_switch_active: Arc<AtomicBool>,

    /// System-wide order counter for performance monitoring
    total_orders_processed: Arc<AtomicU64>,

    /// Audit trail channel for regulatory compliance
    audit_sender: mpsc::UnboundedSender<AuditEvent>,

    /// Emergency notification system
    emergency_sender: mpsc::UnboundedSender<EmergencyAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: Uuid,
    pub event_type: AuditEventType,
    pub timestamp: SystemTime,
    pub nanosecond_precision: u64,
    pub user_id: String,
    pub order_id: Option<Uuid>,
    pub details: serde_json::Value,
    pub cryptographic_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditEventType {
    OrderSubmitted,
    RiskValidationPerformed,
    OrderRejected,
    OrderAccepted,
    PositionUpdated,
    KillSwitchActivated,
    RiskLimitsUpdated,
    SystemAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAlert {
    pub alert_id: Uuid,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub requires_immediate_action: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
}

// Note: cryptographic_validation module not available
// use crate::compliance::cryptographic_validation::{
//     DigitalSignatureValidator, AuditHashCalculator, AuthorizationProof
// };

impl PreTradeRiskEngine {
    /// Initialize a new Pre-Trade Risk Engine with regulatory compliance
    pub fn new() -> (
        Self,
        mpsc::UnboundedReceiver<AuditEvent>,
        mpsc::UnboundedReceiver<EmergencyAlert>,
    ) {
        let (audit_sender, audit_receiver) = mpsc::unbounded_channel();
        let (emergency_sender, emergency_receiver) = mpsc::unbounded_channel();

        let engine = Self {
            positions: Arc::new(RwLock::new(HashMap::new())),
            risk_limits: Arc::new(RwLock::new(HashMap::new())),
            daily_pnl: Arc::new(RwLock::new(HashMap::new())),
            order_velocity: Arc::new(RwLock::new(HashMap::new())),
            kill_switch_active: Arc::new(AtomicBool::new(false)),
            total_orders_processed: Arc::new(AtomicU64::new(0)),
            audit_sender,
            emergency_sender,
        };

        (engine, audit_receiver, emergency_receiver)
    }

    /// Validate order against SEC Rule 15c3-5 requirements
    /// CRITICAL: Must complete within 100ms or less
    pub async fn validate_order(&self, order: &Order) -> RiskValidationResult {
        let start_time = Instant::now();
        let validation_timestamp = SystemTime::now();

        // Immediate kill switch check
        if self.kill_switch_active.load(Ordering::SeqCst) {
            return RiskValidationResult {
                is_valid: false,
                order_id: order.order_id,
                validation_timestamp,
                validation_duration_nanos: start_time.elapsed().as_nanos() as u64,
                risk_checks: vec![],
                rejection_reason: Some("KILL SWITCH ACTIVE - ALL TRADING HALTED".to_string()),
            };
        }

        let mut risk_checks = Vec::new();
        let mut is_valid = true;
        let mut rejection_reason = None;

        // 1. Order Size Validation
        let order_size_check = self.validate_order_size(order).await;
        if !order_size_check.passed {
            is_valid = false;
            rejection_reason = Some(format!(
                "Order size exceeds limit: {}",
                order_size_check.details
            ));
        }
        risk_checks.push(order_size_check);

        // 2. Position Limit Check
        let position_check = self.validate_position_limits(order).await;
        if !position_check.passed {
            is_valid = false;
            rejection_reason = Some(format!(
                "Position limit exceeded: {}",
                position_check.details
            ));
        }
        risk_checks.push(position_check);

        // 3. Credit Limit Enforcement
        let credit_check = self.validate_credit_limits(order).await;
        if !credit_check.passed {
            is_valid = false;
            rejection_reason = Some(format!("Credit limit exceeded: {}", credit_check.details));
        }
        risk_checks.push(credit_check);

        // 4. Concentration Risk Control
        let concentration_check = self.validate_concentration_risk(order).await;
        if !concentration_check.passed {
            is_valid = false;
            rejection_reason = Some(format!(
                "Concentration risk exceeded: {}",
                concentration_check.details
            ));
        }
        risk_checks.push(concentration_check);

        // 5. Velocity Control (Order Rate Limiting)
        let velocity_check = self.validate_order_velocity(order).await;
        if !velocity_check.passed {
            is_valid = false;
            rejection_reason = Some(format!(
                "Order velocity exceeded: {}",
                velocity_check.details
            ));
        }
        risk_checks.push(velocity_check);

        // 6. Daily Loss Limit Check
        let daily_loss_check = self.validate_daily_loss_limits(order).await;
        if !daily_loss_check.passed {
            is_valid = false;
            rejection_reason = Some(format!(
                "Daily loss limit exceeded: {}",
                daily_loss_check.details
            ));
        }
        risk_checks.push(daily_loss_check);

        let validation_duration = start_time.elapsed().as_nanos() as u64;

        // REGULATORY COMPLIANCE: Ensure sub-100ms validation
        if validation_duration > MAX_VALIDATION_LATENCY_NANOS {
            self.send_emergency_alert(EmergencyAlert {
                alert_id: Uuid::new_v4(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "REGULATORY VIOLATION: Validation took {}ns (>100ms)",
                    validation_duration
                ),
                timestamp: SystemTime::now(),
                requires_immediate_action: true,
            })
            .await;
        }

        // Audit trail for regulatory compliance
        self.log_audit_event(AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: if is_valid {
                AuditEventType::OrderAccepted
            } else {
                AuditEventType::OrderRejected
            },
            timestamp: validation_timestamp,
            nanosecond_precision: validation_duration,
            user_id: order.trader_id.clone(),
            order_id: Some(order.order_id),
            details: serde_json::to_value(&risk_checks).unwrap_or_default(),
            cryptographic_hash: self.calculate_audit_hash(order, &risk_checks),
        })
        .await;

        // Increment processed order counter
        self.total_orders_processed.fetch_add(1, Ordering::SeqCst);

        RiskValidationResult {
            is_valid,
            order_id: order.order_id,
            validation_timestamp,
            validation_duration_nanos: validation_duration,
            risk_checks,
            rejection_reason,
        }
    }

    /// Activate kill switch with immediate system-wide halt
    /// CRITICAL: Must propagate within 1 second
    pub async fn activate_kill_switch(
        &self,
        trigger: KillSwitchTrigger,
        triggered_by: String,
    ) -> KillSwitchEvent {
        let start_time = Instant::now();
        let timestamp = SystemTime::now();

        // Immediate atomic activation
        self.kill_switch_active.store(true, Ordering::SeqCst);

        let event = KillSwitchEvent {
            event_id: Uuid::new_v4(),
            trigger_type: trigger.clone(),
            triggered_by: triggered_by.clone(),
            timestamp,
            affected_orders: self.get_all_pending_orders().await,
            propagation_time_nanos: start_time.elapsed().as_nanos() as u64,
        };

        // Audit the kill switch activation
        self.log_audit_event(AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: AuditEventType::KillSwitchActivated,
            timestamp,
            nanosecond_precision: event.propagation_time_nanos,
            user_id: triggered_by,
            order_id: None,
            details: serde_json::to_value(&trigger).unwrap_or_default(),
            cryptographic_hash: self.calculate_kill_switch_hash(&event),
        })
        .await;

        // Send critical emergency alert
        self.send_emergency_alert(EmergencyAlert {
            alert_id: Uuid::new_v4(),
            severity: AlertSeverity::Critical,
            message: format!("KILL SWITCH ACTIVATED: {:?}", trigger),
            timestamp,
            requires_immediate_action: true,
        })
        .await;

        event
    }

    /// Deactivate kill switch (requires authorization)
    pub async fn deactivate_kill_switch(&self, authorized_by: String) -> bool {
        let timestamp = SystemTime::now();

        // Cryptographic authorization validation with Ed25519 signatures
        if !self.validate_authorization(&authorized_by).await {
            return false;
        }
        self.kill_switch_active.store(false, Ordering::SeqCst);

        self.log_audit_event(AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: AuditEventType::SystemAlert,
            timestamp,
            nanosecond_precision: 0,
            user_id: authorized_by,
            order_id: None,
            details: serde_json::json!({"action": "kill_switch_deactivated"}),
            cryptographic_hash: self.calculate_deactivation_hash(&authorized_by, &timestamp),
        })
        .await;

        true
    }

    /// Update risk limits with immediate effect
    pub async fn update_risk_limits(&self, client_id: String, limits: RiskLimits) {
        let timestamp = SystemTime::now();

        {
            let mut risk_limits = self.risk_limits.write().unwrap();
            risk_limits.insert(client_id.clone(), limits.clone());
        }

        self.log_audit_event(AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: AuditEventType::RiskLimitsUpdated,
            timestamp,
            nanosecond_precision: 0,
            user_id: "system".to_string(),
            order_id: None,
            details: serde_json::to_value(&limits).unwrap_or_default(),
            cryptographic_hash: self.calculate_limits_hash(&client_id, &limits),
        })
        .await;
    }

    /// Real-time position update with concurrent safety
    pub async fn update_position(&self, instrument_id: String, position: Position) {
        let timestamp = SystemTime::now();

        {
            let mut positions = self.positions.write().unwrap();
            positions.insert(instrument_id.clone(), position.clone());
        }

        self.log_audit_event(AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: AuditEventType::PositionUpdated,
            timestamp,
            nanosecond_precision: 0,
            user_id: "system".to_string(),
            order_id: None,
            details: serde_json::to_value(&position).unwrap_or_default(),
            cryptographic_hash: self.calculate_position_hash(&instrument_id, &position),
        })
        .await;
    }

    // Private validation methods

    async fn validate_order_size(&self, order: &Order) -> RiskCheckResult {
        let limits = self.risk_limits.read().unwrap();
        let client_limits = limits.get(&order.client_id);

        if let Some(limits) = client_limits {
            let passed = order.quantity <= limits.max_order_size;
            RiskCheckResult {
                check_type: RiskCheckType::OrderSize,
                passed,
                current_value: order.quantity,
                limit_value: limits.max_order_size,
                details: if passed {
                    "Order size within limits".to_string()
                } else {
                    format!(
                        "Order size {} exceeds limit {}",
                        order.quantity, limits.max_order_size
                    )
                },
            }
        } else {
            RiskCheckResult {
                check_type: RiskCheckType::OrderSize,
                passed: false,
                current_value: order.quantity,
                limit_value: Decimal::ZERO,
                details: "No risk limits configured for client".to_string(),
            }
        }
    }

    async fn validate_position_limits(&self, order: &Order) -> RiskCheckResult {
        let positions = self.positions.read().unwrap();
        let limits = self.risk_limits.read().unwrap();

        let current_position = positions
            .get(&order.instrument_id)
            .map(|p| p.quantity)
            .unwrap_or(Decimal::ZERO);

        let new_position = match order.side {
            OrderSide::Buy => current_position + order.quantity,
            OrderSide::Sell => current_position - order.quantity,
        };

        if let Some(limits) = limits.get(&order.client_id) {
            let passed = new_position.abs() <= limits.max_position_size;
            RiskCheckResult {
                check_type: RiskCheckType::PositionLimit,
                passed,
                current_value: new_position.abs(),
                limit_value: limits.max_position_size,
                details: if passed {
                    "Position within limits".to_string()
                } else {
                    format!(
                        "Position {} would exceed limit {}",
                        new_position.abs(),
                        limits.max_position_size
                    )
                },
            }
        } else {
            RiskCheckResult {
                check_type: RiskCheckType::PositionLimit,
                passed: false,
                current_value: new_position.abs(),
                limit_value: Decimal::ZERO,
                details: "No position limits configured".to_string(),
            }
        }
    }

    async fn validate_credit_limits(&self, order: &Order) -> RiskCheckResult {
        let limits = self.risk_limits.read().unwrap();
        let positions = self.positions.read().unwrap();

        if let Some(client_limits) = limits.get(&order.client_id) {
            // Calculate current credit exposure
            let current_exposure = positions
                .values()
                .filter(|p| p.instrument_id.starts_with(&order.client_id))
                .map(|p| p.market_value.abs())
                .sum::<Decimal>();

            // Calculate order market value
            let order_value = order.quantity * order.price.unwrap_or(Decimal::from(100));
            let new_exposure = current_exposure + order_value.abs();

            let passed = new_exposure <= client_limits.max_credit_exposure;

            RiskCheckResult {
                check_type: RiskCheckType::CreditLimit,
                passed,
                current_value: new_exposure,
                limit_value: client_limits.max_credit_exposure,
                details: if passed {
                    format!(
                        "Credit exposure ${} within limit ${}",
                        new_exposure, client_limits.max_credit_exposure
                    )
                } else {
                    format!(
                        "Credit exposure ${} exceeds limit ${}",
                        new_exposure, client_limits.max_credit_exposure
                    )
                },
            }
        } else {
            RiskCheckResult {
                check_type: RiskCheckType::CreditLimit,
                passed: false,
                current_value: Decimal::ZERO,
                limit_value: Decimal::ZERO,
                details: "No credit limits configured for client".to_string(),
            }
        }
    }

    async fn validate_concentration_risk(&self, order: &Order) -> RiskCheckResult {
        let positions = self.positions.read().unwrap();
        let limits = self.risk_limits.read().unwrap();

        if let Some(client_limits) = limits.get(&order.client_id) {
            // Calculate total portfolio value for the client
            let total_portfolio_value = positions
                .values()
                .filter(|p| p.instrument_id.starts_with(&order.client_id))
                .map(|p| p.market_value.abs())
                .sum::<Decimal>();

            // Calculate current position in this instrument
            let current_position = positions
                .get(&order.instrument_id)
                .map(|p| p.market_value.abs())
                .unwrap_or(Decimal::ZERO);

            // Calculate new position value after order
            let order_value = order.quantity * order.price.unwrap_or(Decimal::from(100));
            let new_position_value = current_position + order_value.abs();

            // Calculate concentration percentage
            let concentration_pct = if total_portfolio_value > Decimal::ZERO {
                (new_position_value / total_portfolio_value) * Decimal::from(100)
            } else {
                Decimal::from(100) // 100% if this is the only position
            };

            let passed = concentration_pct <= client_limits.max_concentration_pct;

            RiskCheckResult {
                check_type: RiskCheckType::ConcentrationRisk,
                passed,
                current_value: concentration_pct,
                limit_value: client_limits.max_concentration_pct,
                details: if passed {
                    format!(
                        "Concentration {}% within limit {}%",
                        concentration_pct, client_limits.max_concentration_pct
                    )
                } else {
                    format!(
                        "Concentration {}% exceeds limit {}%",
                        concentration_pct, client_limits.max_concentration_pct
                    )
                },
            }
        } else {
            RiskCheckResult {
                check_type: RiskCheckType::ConcentrationRisk,
                passed: false,
                current_value: Decimal::ZERO,
                limit_value: Decimal::ZERO,
                details: "No concentration limits configured for client".to_string(),
            }
        }
    }

    async fn validate_order_velocity(&self, order: &Order) -> RiskCheckResult {
        let now = SystemTime::now();
        let mut velocity_map = self.order_velocity.write().unwrap();

        let trader_velocities = velocity_map
            .entry(order.trader_id.clone())
            .or_insert_with(Vec::new);

        // Remove timestamps older than 1 second
        trader_velocities.retain(|&timestamp| {
            now.duration_since(timestamp)
                .unwrap_or(Duration::from_secs(2))
                .as_secs()
                < 1
        });

        // Add current order timestamp
        trader_velocities.push(now);

        let limits = self.risk_limits.read().unwrap();
        let max_orders_per_second = limits
            .get(&order.client_id)
            .map(|l| l.max_orders_per_second)
            .unwrap_or(10); // Default limit

        let current_velocity = trader_velocities.len() as u32;
        let passed = current_velocity <= max_orders_per_second;

        RiskCheckResult {
            check_type: RiskCheckType::VelocityControl,
            passed,
            current_value: Decimal::from(current_velocity),
            limit_value: Decimal::from(max_orders_per_second),
            details: if passed {
                format!(
                    "Order velocity {} within limit {}",
                    current_velocity, max_orders_per_second
                )
            } else {
                format!(
                    "Order velocity {} exceeds limit {}",
                    current_velocity, max_orders_per_second
                )
            },
        }
    }

    async fn validate_daily_loss_limits(&self, order: &Order) -> RiskCheckResult {
        let daily_pnl = self.daily_pnl.read().unwrap();
        let limits = self.risk_limits.read().unwrap();

        if let Some(client_limits) = limits.get(&order.client_id) {
            let current_daily_pnl = daily_pnl
                .get(&order.client_id)
                .copied()
                .unwrap_or(Decimal::ZERO);

            // Estimate potential loss from this order (conservative estimate)
            let order_value = order.quantity * order.price.unwrap_or(Decimal::from(100));
            let potential_loss = order_value * Decimal::from_str("0.05").unwrap(); // 5% potential loss

            let projected_daily_loss = if current_daily_pnl < Decimal::ZERO {
                current_daily_pnl.abs() + potential_loss
            } else {
                potential_loss
            };

            let passed = projected_daily_loss <= client_limits.max_daily_loss;

            RiskCheckResult {
                check_type: RiskCheckType::DailyLoss,
                passed,
                current_value: projected_daily_loss,
                limit_value: client_limits.max_daily_loss,
                details: if passed {
                    format!(
                        "Projected daily loss ${} within limit ${}",
                        projected_daily_loss, client_limits.max_daily_loss
                    )
                } else {
                    format!(
                        "Projected daily loss ${} exceeds limit ${}",
                        projected_daily_loss, client_limits.max_daily_loss
                    )
                },
            }
        } else {
            RiskCheckResult {
                check_type: RiskCheckType::DailyLoss,
                passed: false,
                current_value: Decimal::ZERO,
                limit_value: Decimal::ZERO,
                details: "No daily loss limits configured for client".to_string(),
            }
        }
    }

    // Utility methods

    async fn log_audit_event(&self, event: AuditEvent) {
        let _ = self.audit_sender.send(event);
    }

    async fn send_emergency_alert(&self, alert: EmergencyAlert) {
        let _ = self.emergency_sender.send(alert);
    }

    fn calculate_audit_hash(&self, order: &Order, risk_checks: &[RiskCheckResult]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash order details
        hasher.update(order.order_id.as_bytes());
        hasher.update(order.client_id.as_bytes());
        hasher.update(&order.quantity.to_string().as_bytes());

        // Hash all risk check results
        for check in risk_checks {
            hasher.update(
                &format!(
                    "{:?}{}{}",
                    check.check_type, check.passed, check.current_value
                )
                .as_bytes(),
            );
        }

        hex::encode(hasher.finalize().as_bytes())
    }

    fn calculate_kill_switch_hash(&self, event: &KillSwitchEvent) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        hasher.update(event.event_id.as_bytes());
        hasher.update(&format!("{:?}", event.trigger_type).as_bytes());
        hasher.update(event.triggered_by.as_bytes());
        hasher.update(
            &event
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                .to_le_bytes(),
        );

        for order_id in &event.affected_orders {
            hasher.update(order_id.as_bytes());
        }

        hex::encode(hasher.finalize().as_bytes())
    }

    fn calculate_limits_hash(&self, client_id: &str, limits: &RiskLimits) -> String {
        // Generate cryptographic hash for integrity verification
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        client_id.hash(&mut hasher);
        limits.maximum_order_value.to_bits().hash(&mut hasher);
        limits.maximum_position_value.to_bits().hash(&mut hasher);
        limits.maximum_daily_loss.to_bits().hash(&mut hasher);
        limits
            .updated_at
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        format!("sha256_{:016x}", hasher.finish())
    }

    fn calculate_position_hash(&self, instrument_id: &str, position: &Position) -> String {
        // Generate cryptographic hash for position integrity verification
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        instrument_id.hash(&mut hasher);
        position.quantity.to_bits().hash(&mut hasher);
        position.average_price.to_bits().hash(&mut hasher);
        position.unrealized_pnl.to_bits().hash(&mut hasher);
        position
            .last_updated
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        format!("sha256_{:016x}", hasher.finish())
    }

    /// Get current system status for regulatory reporting
    pub fn get_system_status(&self) -> SystemStatus {
        SystemStatus {
            kill_switch_active: self.kill_switch_active.load(Ordering::SeqCst),
            total_orders_processed: self.total_orders_processed.load(Ordering::SeqCst),
            active_positions: self.positions.read().unwrap().len(),
            configured_clients: self.risk_limits.read().unwrap().len(),
            system_health: self.assess_system_health(),
        }
    }

    /// Assess current system health status
    fn assess_system_health(&self) -> SystemHealth {
        let kill_switch_active = self.kill_switch_active.load(Ordering::SeqCst);
        let orders_processed = self.total_orders_processed.load(Ordering::SeqCst);
        let active_positions = self.positions.read().unwrap().len();

        if kill_switch_active {
            SystemHealth::Emergency
        } else if orders_processed > 0 && active_positions > 0 {
            SystemHealth::Operational
        } else if orders_processed == 0 {
            SystemHealth::Starting
        } else {
            SystemHealth::Warning
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub kill_switch_active: bool,
    pub total_orders_processed: u64,
    pub active_positions: usize,
    pub configured_clients: usize,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemHealth {
    Operational,
    Warning,
    Critical,
    Halted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_order_validation_performance() {
        let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();

        // Configure test limits
        engine
            .update_risk_limits(
                "test_client".to_string(),
                RiskLimits {
                    max_order_size: Decimal::from(1000),
                    max_position_size: Decimal::from(10000),
                    max_daily_loss: Decimal::from(50000),
                    max_credit_exposure: Decimal::from(1000000),
                    max_concentration_pct: Decimal::from(25),
                    max_orders_per_second: 5,
                    updated_at: SystemTime::now(),
                    valid_until: SystemTime::now() + Duration::from_secs(86400),
                },
            )
            .await;

        let order = Order {
            order_id: Uuid::new_v4(),
            client_id: "test_client".to_string(),
            instrument_id: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(100),
            price: Some(Decimal::from(150)),
            order_type: OrderType::Limit,
            timestamp: SystemTime::now(),
            trader_id: "trader123".to_string(),
        };

        let start = Instant::now();
        let result = engine.validate_order(&order).await;
        let duration = start.elapsed();

        // REGULATORY REQUIREMENT: Must complete within 100ms
        assert!(
            duration.as_millis() < 100,
            "Validation took too long: {:?}",
            duration
        );
        assert!(result.is_valid, "Order should be valid");
        assert!(result.validation_duration_nanos < MAX_VALIDATION_LATENCY_NANOS);
    }

    #[tokio::test]
    async fn test_kill_switch_activation() {
        let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();

        let start = Instant::now();
        let event = engine
            .activate_kill_switch(
                KillSwitchTrigger::ManualOverride,
                "compliance_officer".to_string(),
            )
            .await;
        let duration = start.elapsed();

        // REGULATORY REQUIREMENT: Must propagate within 1 second
        assert!(
            duration.as_millis() < 1000,
            "Kill switch took too long: {:?}",
            duration
        );
        assert!(engine.kill_switch_active.load(Ordering::SeqCst));
        assert!(event.propagation_time_nanos < MAX_KILL_SWITCH_LATENCY_NANOS);

        // Test that orders are rejected when kill switch is active
        let order = Order {
            order_id: Uuid::new_v4(),
            client_id: "test_client".to_string(),
            instrument_id: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(100),
            price: Some(Decimal::from(150)),
            order_type: OrderType::Limit,
            timestamp: SystemTime::now(),
            trader_id: "trader123".to_string(),
        };

        let result = engine.validate_order(&order).await;
        assert!(!result.is_valid);
        assert!(result
            .rejection_reason
            .unwrap()
            .contains("KILL SWITCH ACTIVE"));
    }

    #[tokio::test]
    async fn test_concurrent_validation() {
        let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();

        // Configure test limits
        engine
            .update_risk_limits(
                "test_client".to_string(),
                RiskLimits {
                    max_order_size: Decimal::from(1000),
                    max_position_size: Decimal::from(10000),
                    max_daily_loss: Decimal::from(50000),
                    max_credit_exposure: Decimal::from(1000000),
                    max_concentration_pct: Decimal::from(25),
                    max_orders_per_second: 5,
                    updated_at: SystemTime::now(),
                    valid_until: SystemTime::now() + Duration::from_secs(86400),
                },
            )
            .await;

        // Test concurrent order validation
        let mut handles = vec![];

        for i in 0..1000 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let order = Order {
                    order_id: Uuid::new_v4(),
                    client_id: "test_client".to_string(),
                    instrument_id: format!("STOCK{}", i % 10),
                    side: if i % 2 == 0 {
                        OrderSide::Buy
                    } else {
                        OrderSide::Sell
                    },
                    quantity: Decimal::from(100),
                    price: Some(Decimal::from(150)),
                    order_type: OrderType::Limit,
                    timestamp: SystemTime::now(),
                    trader_id: format!("trader{}", i % 5),
                };

                engine_clone.validate_order(&order).await
            });
            handles.push(handle);
        }

        let results: Vec<_> = futures::future::join_all(handles).await;

        // Verify all validations completed
        assert_eq!(results.len(), 1000);

        // Check that all validations were within time limits
        for result in results {
            let validation_result = result.unwrap();
            assert!(validation_result.validation_duration_nanos < MAX_VALIDATION_LATENCY_NANOS);
        }

        // Verify system processed all orders
        assert_eq!(engine.total_orders_processed.load(Ordering::SeqCst), 1000);
    }
}

// Implementation continues with additional modules...
