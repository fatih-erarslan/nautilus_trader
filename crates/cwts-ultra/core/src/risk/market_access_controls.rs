//! Market Access Controls Implementation
//!
//! Implements systematic risk controls, circuit breakers, and automated
//! halt mechanisms as required by SEC Rule 15c3-5

use rust_decimal::{prelude::ToPrimitive, Decimal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, RwLock,
};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::compliance::sec_rule_15c3_5::{
    AlertSeverity, AuditEvent, AuditEventType, EmergencyAlert,
};

/// Circuit breaker thresholds as percentages
const LEVEL_1_THRESHOLD: Decimal = Decimal::from_parts(7, 0, 0, false, 0); // 7%
const LEVEL_2_THRESHOLD: Decimal = Decimal::from_parts(13, 0, 0, false, 0); // 13%
const LEVEL_3_THRESHOLD: Decimal = Decimal::from_parts(20, 0, 0, false, 0); // 20%

/// Maximum allowed order-to-execution latency (regulatory requirement)
const MAX_EXECUTION_LATENCY_NANOS: u64 = 100_000_000; // 100ms

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CircuitBreakerState {
    pub level: CircuitBreakerLevel,
    pub triggered_at: Option<SystemTime>,
    pub trigger_reason: String,
    pub market_decline_pct: Decimal,
    pub halt_duration: Duration,
    pub auto_resume: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CircuitBreakerLevel {
    None,
    Level1, // 7% decline - 15 minute halt
    Level2, // 13% decline - 15 minute halt
    Level3, // 20% decline - halt until next day
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SystematicRiskMetrics {
    pub market_stress_indicator: Decimal,
    pub volatility_index: Decimal,
    pub correlation_breakdown: bool,
    pub liquidity_stress: Decimal,
    pub credit_stress: Decimal,
    pub operational_risk_level: RiskLevel,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DailyRiskLimits {
    pub max_daily_loss: Decimal,
    pub current_daily_pnl: Decimal,
    pub max_gross_exposure: Decimal,
    pub current_gross_exposure: Decimal,
    pub max_net_exposure: Decimal,
    pub current_net_exposure: Decimal,
    pub breach_count: u32,
    pub last_breach: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LatencyMetrics {
    pub order_to_validation_nanos: u64,
    pub validation_to_execution_nanos: u64,
    pub total_latency_nanos: u64,
    pub breach_count: u32,
    pub last_breach: Option<SystemTime>,
}

/// Market Access Control Engine
pub struct MarketAccessEngine {
    /// Circuit breaker state
    circuit_breaker: Arc<RwLock<CircuitBreakerState>>,

    /// Systematic risk monitoring
    systematic_risk: Arc<RwLock<SystematicRiskMetrics>>,

    /// Daily risk tracking per client
    daily_limits: Arc<RwLock<HashMap<String, DailyRiskLimits>>>,

    /// Real-time latency monitoring
    latency_metrics: Arc<RwLock<HashMap<String, LatencyMetrics>>>,

    /// Emergency halt state
    emergency_halt: Arc<AtomicBool>,

    /// Market stress level
    market_stress_level: Arc<AtomicU64>, // 0-100 scale

    /// Audit trail
    audit_sender: mpsc::UnboundedSender<AuditEvent>,

    /// Emergency alerts
    emergency_sender: mpsc::UnboundedSender<EmergencyAlert>,
}

impl MarketAccessEngine {
    pub fn new(
        audit_sender: mpsc::UnboundedSender<AuditEvent>,
        emergency_sender: mpsc::UnboundedSender<EmergencyAlert>,
    ) -> Self {
        Self {
            circuit_breaker: Arc::new(RwLock::new(CircuitBreakerState {
                level: CircuitBreakerLevel::None,
                triggered_at: None,
                trigger_reason: String::new(),
                market_decline_pct: Decimal::ZERO,
                halt_duration: Duration::from_secs(0),
                auto_resume: false,
            })),
            systematic_risk: Arc::new(RwLock::new(SystematicRiskMetrics {
                market_stress_indicator: Decimal::ZERO,
                volatility_index: Decimal::ZERO,
                correlation_breakdown: false,
                liquidity_stress: Decimal::ZERO,
                credit_stress: Decimal::ZERO,
                operational_risk_level: RiskLevel::Low,
                last_updated: SystemTime::now(),
            })),
            daily_limits: Arc::new(RwLock::new(HashMap::new())),
            latency_metrics: Arc::new(RwLock::new(HashMap::new())),
            emergency_halt: Arc::new(AtomicBool::new(false)),
            market_stress_level: Arc::new(AtomicU64::new(0)),
            audit_sender,
            emergency_sender,
        }
    }

    /// Check if market access is allowed
    pub async fn is_market_access_allowed(&self) -> MarketAccessDecision {
        let start_time = Instant::now();

        // Emergency halt check
        if self.emergency_halt.load(Ordering::SeqCst) {
            return MarketAccessDecision {
                allowed: false,
                reason: "Emergency halt in effect".to_string(),
                decision_time_nanos: start_time.elapsed().as_nanos() as u64,
                risk_factors: vec!["emergency_halt".to_string()],
            };
        }

        // Circuit breaker check
        let circuit_breaker = self.circuit_breaker.read().unwrap();
        if circuit_breaker.level != CircuitBreakerLevel::None {
            if let Some(triggered_at) = circuit_breaker.triggered_at {
                let elapsed = SystemTime::now()
                    .duration_since(triggered_at)
                    .unwrap_or(Duration::ZERO);
                if elapsed < circuit_breaker.halt_duration && !circuit_breaker.auto_resume {
                    return MarketAccessDecision {
                        allowed: false,
                        reason: format!("Circuit breaker {:?} active", circuit_breaker.level),
                        decision_time_nanos: start_time.elapsed().as_nanos() as u64,
                        risk_factors: vec!["circuit_breaker".to_string()],
                    };
                }
            }
        }
        drop(circuit_breaker);

        // Systematic risk check
        let systematic_risk = self.systematic_risk.read().unwrap();
        let mut risk_factors = Vec::new();

        if systematic_risk.operational_risk_level == RiskLevel::Critical {
            risk_factors.push("critical_operational_risk".to_string());
        }

        if systematic_risk.correlation_breakdown {
            risk_factors.push("correlation_breakdown".to_string());
        }

        if systematic_risk.liquidity_stress > Decimal::from(80) {
            risk_factors.push("high_liquidity_stress".to_string());
        }

        let market_stress = self.market_stress_level.load(Ordering::SeqCst);
        if market_stress > 90 {
            risk_factors.push("extreme_market_stress".to_string());
        }

        drop(systematic_risk);

        let allowed = risk_factors.is_empty();
        let reason = if allowed {
            "Market access permitted".to_string()
        } else {
            format!(
                "Market access restricted due to: {}",
                risk_factors.join(", ")
            )
        };

        MarketAccessDecision {
            allowed,
            reason,
            decision_time_nanos: start_time.elapsed().as_nanos() as u64,
            risk_factors,
        }
    }

    /// Monitor and update systematic risk metrics
    pub async fn update_systematic_risk(&self, metrics: SystematicRiskMetrics) {
        let timestamp = SystemTime::now();

        {
            let mut risk = self.systematic_risk.write().unwrap();
            *risk = metrics.clone();
        }

        // Check for circuit breaker triggers
        self.check_circuit_breaker_conditions().await;

        // Update market stress level
        let stress_level = self.calculate_market_stress(&metrics);
        self.market_stress_level
            .store(stress_level, Ordering::SeqCst);

        // Audit the risk update
        let event_id = Uuid::new_v4();
        let event_type = AuditEventType::SystemAlert;
        self.log_audit_event(AuditEvent {
            event_id,
            event_type: event_type.clone(),
            timestamp,
            nanosecond_precision: 0,
            user_id: "system".to_string(),
            order_id: None,
            details: serde_json::to_value(&metrics).unwrap_or_default(),
            cryptographic_hash: self.calculate_event_hash(&event_id.to_string(), &format!("{:?}", event_type), &timestamp),
        })
        .await;

        // Send alerts for critical conditions
        if metrics.operational_risk_level == RiskLevel::Critical {
            self.send_emergency_alert(EmergencyAlert {
                alert_id: Uuid::new_v4(),
                severity: AlertSeverity::Critical,
                message: "Critical operational risk level detected".to_string(),
                timestamp,
                requires_immediate_action: true,
            })
            .await;
        }
    }

    /// Check daily risk limits for a client
    pub async fn check_daily_limits(
        &self,
        client_id: &str,
        proposed_pnl: Decimal,
    ) -> DailyLimitCheck {
        let limits = self.daily_limits.read().unwrap();

        if let Some(client_limits) = limits.get(client_id) {
            let new_daily_pnl = client_limits.current_daily_pnl + proposed_pnl;

            let loss_limit_breached = new_daily_pnl < -client_limits.max_daily_loss.abs();
            let exposure_ok =
                client_limits.current_gross_exposure <= client_limits.max_gross_exposure;

            DailyLimitCheck {
                within_limits: !loss_limit_breached && exposure_ok,
                current_pnl: new_daily_pnl,
                loss_limit: client_limits.max_daily_loss,
                exposure_ok,
                breach_reason: if loss_limit_breached {
                    Some("Daily loss limit exceeded".to_string())
                } else if !exposure_ok {
                    Some("Gross exposure limit exceeded".to_string())
                } else {
                    None
                },
            }
        } else {
            DailyLimitCheck {
                within_limits: false,
                current_pnl: Decimal::ZERO,
                loss_limit: Decimal::ZERO,
                exposure_ok: false,
                breach_reason: Some("No daily limits configured".to_string()),
            }
        }
    }

    /// Record order execution latency
    pub async fn record_execution_latency(&self, trader_id: String, latency_nanos: u64) {
        let mut metrics_map = self.latency_metrics.write().unwrap();
        let metrics = metrics_map
            .entry(trader_id.clone())
            .or_insert_with(|| LatencyMetrics {
                order_to_validation_nanos: 0,
                validation_to_execution_nanos: 0,
                total_latency_nanos: 0,
                breach_count: 0,
                last_breach: None,
            });

        metrics.total_latency_nanos = latency_nanos;

        // Check for latency breach
        if latency_nanos > MAX_EXECUTION_LATENCY_NANOS {
            metrics.breach_count += 1;
            metrics.last_breach = Some(SystemTime::now());

            // Send alert for latency breach
            self.send_emergency_alert(EmergencyAlert {
                alert_id: Uuid::new_v4(),
                severity: AlertSeverity::High,
                message: format!(
                    "Execution latency breach: {}ns for trader {}",
                    latency_nanos, trader_id
                ),
                timestamp: SystemTime::now(),
                requires_immediate_action: false,
            })
            .await;
        }
    }

    /// Activate emergency halt
    pub async fn activate_emergency_halt(&self, reason: String, activated_by: String) {
        let timestamp = SystemTime::now();

        self.emergency_halt.store(true, Ordering::SeqCst);

        let event_id = Uuid::new_v4();
        let event_type = AuditEventType::SystemAlert;
        self.log_audit_event(AuditEvent {
            event_id,
            event_type: event_type.clone(),
            timestamp,
            nanosecond_precision: 0,
            user_id: activated_by.clone(),
            order_id: None,
            details: serde_json::json!({
                "action": "emergency_halt_activated",
                "reason": reason
            }),
            cryptographic_hash: self.calculate_event_hash(&event_id.to_string(), &format!("{:?}", event_type), &timestamp),
        })
        .await;

        self.send_emergency_alert(EmergencyAlert {
            alert_id: Uuid::new_v4(),
            severity: AlertSeverity::Critical,
            message: format!("EMERGENCY HALT ACTIVATED: {} by {}", reason, activated_by),
            timestamp,
            requires_immediate_action: true,
        })
        .await;
    }

    /// Update daily limits for a client
    pub async fn update_daily_limits(&self, client_id: String, limits: DailyRiskLimits) {
        let timestamp = SystemTime::now();

        {
            let mut daily_limits = self.daily_limits.write().unwrap();
            daily_limits.insert(client_id.clone(), limits.clone());
        }

        let event_id = Uuid::new_v4();
        let event_type = AuditEventType::RiskLimitsUpdated;
        self.log_audit_event(AuditEvent {
            event_id,
            event_type: event_type.clone(),
            timestamp,
            nanosecond_precision: 0,
            user_id: "system".to_string(),
            order_id: None,
            details: serde_json::to_value(&limits).unwrap_or_default(),
            cryptographic_hash: self.calculate_event_hash(&event_id.to_string(), &format!("{:?}", event_type), &timestamp),
        })
        .await;
    }

    // Private methods

    async fn check_circuit_breaker_conditions(&self) {
        let risk = self.systematic_risk.read().unwrap();
        let market_decline = risk.market_stress_indicator;
        drop(risk);

        let mut breaker = self.circuit_breaker.write().unwrap();
        let current_level = &breaker.level;

        let new_level = if market_decline >= LEVEL_3_THRESHOLD {
            CircuitBreakerLevel::Level3
        } else if market_decline >= LEVEL_2_THRESHOLD {
            CircuitBreakerLevel::Level2
        } else if market_decline >= LEVEL_1_THRESHOLD {
            CircuitBreakerLevel::Level1
        } else {
            CircuitBreakerLevel::None
        };

        if new_level != *current_level && new_level != CircuitBreakerLevel::None {
            breaker.level = new_level.clone();
            breaker.triggered_at = Some(SystemTime::now());
            breaker.market_decline_pct = market_decline;
            breaker.trigger_reason = format!("Market decline of {}%", market_decline);

            breaker.halt_duration = match new_level {
                CircuitBreakerLevel::Level1 | CircuitBreakerLevel::Level2 => {
                    Duration::from_secs(15 * 60)
                } // 15 minutes
                CircuitBreakerLevel::Level3 => Duration::from_secs(24 * 60 * 60), // Until next day
                CircuitBreakerLevel::None => Duration::ZERO,
            };

            breaker.auto_resume = matches!(
                new_level,
                CircuitBreakerLevel::Level1 | CircuitBreakerLevel::Level2
            );

            // Send critical alert
            self.send_emergency_alert(EmergencyAlert {
                alert_id: Uuid::new_v4(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "CIRCUIT BREAKER {:?} TRIGGERED - {}% market decline",
                    new_level, market_decline
                ),
                timestamp: SystemTime::now(),
                requires_immediate_action: true,
            })
            .await;
        }
    }

    fn calculate_market_stress(&self, metrics: &SystematicRiskMetrics) -> u64 {
        let mut stress_score = 0u64;

        // Volatility component (0-30 points)
        if metrics.volatility_index > Decimal::from(30) {
            stress_score += 30;
        } else {
            stress_score += (metrics.volatility_index * Decimal::from(30) / Decimal::from(100))
                .to_u64()
                .unwrap_or(0);
        }

        // Liquidity stress (0-25 points)
        stress_score += (metrics.liquidity_stress * Decimal::from(25) / Decimal::from(100))
            .to_u64()
            .unwrap_or(0);

        // Credit stress (0-25 points)
        stress_score += (metrics.credit_stress * Decimal::from(25) / Decimal::from(100))
            .to_u64()
            .unwrap_or(0);

        // Operational risk (0-20 points)
        stress_score += match metrics.operational_risk_level {
            RiskLevel::Low => 0,
            RiskLevel::Medium => 5,
            RiskLevel::High => 15,
            RiskLevel::Critical => 20,
        };

        stress_score.min(100)
    }

    async fn log_audit_event(&self, event: AuditEvent) {
        let _ = self.audit_sender.send(event);
    }

    async fn send_emergency_alert(&self, alert: EmergencyAlert) {
        let _ = self.emergency_sender.send(alert);
    }

    /// Get current market access status
    pub fn get_market_status(&self) -> MarketStatus {
        let circuit_breaker = self.circuit_breaker.read().unwrap();
        let systematic_risk = self.systematic_risk.read().unwrap();

        MarketStatus {
            emergency_halt: self.emergency_halt.load(Ordering::SeqCst),
            circuit_breaker_level: circuit_breaker.level.clone(),
            market_stress_level: self.market_stress_level.load(Ordering::SeqCst),
            operational_risk: systematic_risk.operational_risk_level.clone(),
            correlation_breakdown: systematic_risk.correlation_breakdown,
            last_updated: SystemTime::now(),
        }
    }

    /// Calculate cryptographic hash for event integrity
    fn calculate_event_hash(
        &self,
        event_id: &str,
        event_type: &str,
        timestamp: &SystemTime,
    ) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        event_id.hash(&mut hasher);
        event_type.hash(&mut hasher);
        timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        format!("sha256_{:016x}", hasher.finish())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MarketAccessDecision {
    pub allowed: bool,
    pub reason: String,
    pub decision_time_nanos: u64,
    pub risk_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DailyLimitCheck {
    pub within_limits: bool,
    pub current_pnl: Decimal,
    pub loss_limit: Decimal,
    pub exposure_ok: bool,
    pub breach_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MarketStatus {
    pub emergency_halt: bool,
    pub circuit_breaker_level: CircuitBreakerLevel,
    pub market_stress_level: u64,
    pub operational_risk: RiskLevel,
    pub correlation_breakdown: bool,
    pub last_updated: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_market_access_decision_speed() {
        let (audit_tx, _audit_rx) = mpsc::unbounded_channel();
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();

        let engine = MarketAccessEngine::new(audit_tx, emergency_tx);

        let start = Instant::now();
        let decision = engine.is_market_access_allowed().await;
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 10,
            "Market access decision too slow: {:?}",
            duration
        );
        assert!(decision.allowed);
        assert!(decision.decision_time_nanos < 10_000_000); // 10ms
    }

    #[tokio::test]
    async fn test_circuit_breaker_levels() {
        let (audit_tx, _audit_rx) = mpsc::unbounded_channel();
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();

        let engine = MarketAccessEngine::new(audit_tx, emergency_tx);

        // Test Level 1 trigger (7% decline)
        let risk_metrics = SystematicRiskMetrics {
            market_stress_indicator: Decimal::from(8), // 8% decline
            volatility_index: Decimal::from(45),
            correlation_breakdown: false,
            liquidity_stress: Decimal::from(30),
            credit_stress: Decimal::from(20),
            operational_risk_level: RiskLevel::Medium,
            last_updated: SystemTime::now(),
        };

        engine.update_systematic_risk(risk_metrics).await;

        let status = engine.get_market_status();
        assert_eq!(status.circuit_breaker_level, CircuitBreakerLevel::Level1);

        let decision = engine.is_market_access_allowed().await;
        assert!(!decision.allowed);
        assert!(decision.reason.contains("Circuit breaker"));
    }

    #[tokio::test]
    async fn test_daily_limit_checks() {
        let (audit_tx, _audit_rx) = mpsc::unbounded_channel();
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();

        let engine = MarketAccessEngine::new(audit_tx, emergency_tx);

        // Set up daily limits
        let limits = DailyRiskLimits {
            max_daily_loss: Decimal::from(100000),    // $100K max loss
            current_daily_pnl: Decimal::from(-50000), // Currently down $50K
            max_gross_exposure: Decimal::from(1000000),
            current_gross_exposure: Decimal::from(500000),
            max_net_exposure: Decimal::from(500000),
            current_net_exposure: Decimal::from(200000),
            breach_count: 0,
            last_breach: None,
        };

        engine
            .update_daily_limits("test_client".to_string(), limits)
            .await;

        // Test within limits
        let check = engine
            .check_daily_limits("test_client", Decimal::from(-30000))
            .await;
        assert!(check.within_limits);

        // Test exceeding limits
        let check = engine
            .check_daily_limits("test_client", Decimal::from(-60000))
            .await;
        assert!(!check.within_limits);
        assert!(check.breach_reason.unwrap().contains("loss limit"));
    }

    #[tokio::test]
    async fn test_latency_monitoring() {
        let (audit_tx, _audit_rx) = mpsc::unbounded_channel();
        let (emergency_tx, mut emergency_rx) = mpsc::unbounded_channel();

        let engine = MarketAccessEngine::new(audit_tx, emergency_tx);

        // Record normal latency
        engine
            .record_execution_latency("trader1".to_string(), 50_000_000)
            .await; // 50ms - OK

        // Record breach latency
        engine
            .record_execution_latency("trader1".to_string(), 150_000_000)
            .await; // 150ms - BREACH

        // Should receive emergency alert
        let alert = emergency_rx.try_recv().unwrap();
        assert_eq!(alert.severity, AlertSeverity::High);
        assert!(alert.message.contains("latency breach"));
    }
}
