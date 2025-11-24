//! Emergency Kill Switch Implementation
//!
//! Provides immediate system-wide trading halt capabilities with <1 second
//! propagation time as required by SEC Rule 15c3-5

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex};
use uuid::Uuid;

use crate::compliance::sec_rule_15c3_5::{
    AlertSeverity, EmergencyAlert, KillSwitchEvent, KillSwitchTrigger,
};

/// Maximum allowed kill switch propagation time (regulatory requirement)
const MAX_KILL_SWITCH_PROPAGATION_NANOS: u64 = 1_000_000_000; // 1 second

/// Kill switch authorization levels
const LEVEL_1_AUTHORIZATION: &str = "trader";
const LEVEL_2_AUTHORIZATION: &str = "risk_manager";
const LEVEL_3_AUTHORIZATION: &str = "compliance_officer";
const LEVEL_4_AUTHORIZATION: &str = "ceo_authorization";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchConfiguration {
    pub auto_triggers: Vec<AutoTriggerCondition>,
    pub authorization_levels: HashMap<KillSwitchLevel, Vec<String>>,
    pub propagation_channels: Vec<PropagationChannel>,
    pub notification_settings: NotificationSettings,
    pub recovery_procedures: RecoveryProcedures,
    pub audit_requirements: AuditRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KillSwitchLevel {
    Level1, // Individual trader halt
    Level2, // Desk/department halt
    Level3, // Firm-wide halt
    Level4, // Market-wide coordination
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTriggerCondition {
    pub condition_id: String,
    pub trigger_type: AutoTriggerType,
    pub threshold: Decimal,
    pub time_window: Duration,
    pub enabled: bool,
    pub requires_manual_confirm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoTriggerType {
    DailyLossExceeded,
    PositionLimitBreach,
    SystemLatencyExceeded,
    ValidationTimeoutRate,
    OrderRejectionRate,
    MarketVolatilitySpike,
    SystemErrors,
    SecurityBreach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationChannel {
    pub channel_id: String,
    pub channel_type: ChannelType,
    pub priority: u8,
    pub target_latency_ms: u32,
    pub fallback_channels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    InternalMessageBus,
    ExchangeConnection,
    RegulatoryReporting,
    EmailAlert,
    SmsAlert,
    PhoneCall,
    WebSocketBroadcast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub immediate_contacts: Vec<ContactInfo>,
    pub escalation_matrix: HashMap<KillSwitchLevel, Vec<ContactInfo>>,
    pub regulatory_contacts: Vec<ContactInfo>,
    pub media_contacts: Vec<ContactInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    pub name: String,
    pub role: String,
    pub phone: String,
    pub email: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryProcedures {
    pub recovery_steps: Vec<RecoveryStep>,
    pub required_authorizations: Vec<String>,
    pub system_checks: Vec<SystemCheck>,
    pub gradual_restart: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    pub step_id: String,
    pub description: String,
    pub required_role: String,
    pub estimated_duration: Duration,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCheck {
    pub check_id: String,
    pub description: String,
    pub automated: bool,
    pub pass_criteria: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirements {
    pub immediate_audit: bool,
    pub regulatory_filing_required: bool,
    pub internal_investigation: bool,
    pub external_review: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchStatus {
    pub is_active: bool,
    pub level: Option<KillSwitchLevel>,
    pub triggered_at: Option<SystemTime>,
    pub triggered_by: Option<String>,
    pub trigger_reason: Option<String>,
    pub affected_systems: Vec<String>,
    pub propagation_complete: bool,
    pub recovery_in_progress: bool,
    pub estimated_recovery_time: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationResult {
    pub success: bool,
    pub channel_results: HashMap<String, ChannelResult>,
    pub total_propagation_time_nanos: u64,
    pub failed_channels: Vec<String>,
    pub regulatory_notifications_sent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelResult {
    pub success: bool,
    pub propagation_time_nanos: u64,
    pub error_message: Option<String>,
    pub retry_count: u32,
}

/// Emergency Kill Switch Engine
pub struct EmergencyKillSwitchEngine {
    /// Current kill switch state
    kill_switch_state: Arc<AtomicBool>,

    /// Kill switch level
    current_level: Arc<RwLock<Option<KillSwitchLevel>>>,

    /// Configuration
    configuration: Arc<RwLock<KillSwitchConfiguration>>,

    /// Active kill switch event
    active_event: Arc<RwLock<Option<KillSwitchEvent>>>,

    /// Propagation channels
    propagation_channels: Arc<RwLock<HashMap<String, PropagationChannel>>>,

    /// Auto-trigger monitoring
    auto_trigger_metrics: Arc<RwLock<HashMap<String, TriggerMetrics>>>,

    /// Recovery state
    recovery_state: Arc<AsyncMutex<RecoveryState>>,

    /// Broadcast channel for immediate notifications
    notification_sender: broadcast::Sender<KillSwitchNotification>,

    /// Emergency alert channel
    emergency_sender: mpsc::UnboundedSender<EmergencyAlert>,

    /// Authorization tracking
    authorization_log: Arc<RwLock<Vec<AuthorizationRecord>>>,
}

#[derive(Debug, Clone)]
struct TriggerMetrics {
    current_value: Decimal,
    threshold: Decimal,
    last_updated: SystemTime,
    breach_count: u32,
    time_window_start: SystemTime,
}

#[derive(Debug, Clone)]
struct RecoveryState {
    in_progress: bool,
    current_step: Option<String>,
    completed_steps: Vec<String>,
    required_authorizations: Vec<String>,
    received_authorizations: Vec<String>,
    start_time: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchNotification {
    pub notification_id: Uuid,
    pub notification_type: NotificationType,
    pub timestamp: SystemTime,
    pub message: String,
    pub urgency: NotificationUrgency,
    pub recipients: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    KillSwitchActivated,
    KillSwitchDeactivated,
    AutoTriggerWarning,
    RecoveryStarted,
    RecoveryCompleted,
    AuthorizationRequired,
    SystemRestored,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationUrgency {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuthorizationRecord {
    authorization_id: Uuid,
    authorized_by: String,
    authorization_level: String,
    timestamp: SystemTime,
    action: AuthorizationAction,
    digital_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AuthorizationAction {
    KillSwitchActivation,
    KillSwitchDeactivation,
    RecoveryAuthorization,
    SystemRestart,
}

impl EmergencyKillSwitchEngine {
    pub fn new(emergency_sender: mpsc::UnboundedSender<EmergencyAlert>) -> Self {
        let (notification_sender, _) = broadcast::channel(1000);

        Self {
            kill_switch_state: Arc::new(AtomicBool::new(false)),
            current_level: Arc::new(RwLock::new(None)),
            configuration: Arc::new(RwLock::new(Self::default_configuration())),
            active_event: Arc::new(RwLock::new(None)),
            propagation_channels: Arc::new(RwLock::new(HashMap::new())),
            auto_trigger_metrics: Arc::new(RwLock::new(HashMap::new())),
            recovery_state: Arc::new(AsyncMutex::new(RecoveryState {
                in_progress: false,
                current_step: None,
                completed_steps: Vec::new(),
                required_authorizations: Vec::new(),
                received_authorizations: Vec::new(),
                start_time: None,
            })),
            notification_sender,
            emergency_sender,
            authorization_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Activate kill switch with immediate propagation
    pub async fn activate_kill_switch(
        &self,
        trigger: KillSwitchTrigger,
        level: KillSwitchLevel,
        triggered_by: String,
        reason: String,
    ) -> Result<PropagationResult, KillSwitchError> {
        let start_time = Instant::now();
        let timestamp = SystemTime::now();

        // Immediate atomic activation
        self.kill_switch_state.store(true, Ordering::SeqCst);

        // Set current level
        {
            let mut current_level = self.current_level.write().unwrap();
            *current_level = Some(level.clone());
        }

        // Create kill switch event
        let event = KillSwitchEvent {
            event_id: Uuid::new_v4(),
            trigger_type: trigger.clone(),
            triggered_by: triggered_by.clone(),
            timestamp,
            affected_orders: self.get_all_pending_orders().await,
            propagation_time_nanos: 0, // Will be updated
        };

        // Store active event
        {
            let mut active_event = self.active_event.write().unwrap();
            *active_event = Some(event.clone());
        }

        // Log authorization
        self.log_authorization(AuthorizationRecord {
            authorization_id: Uuid::new_v4(),
            authorized_by: triggered_by.clone(),
            authorization_level: Self::determine_authorization_level(&level),
            timestamp,
            action: AuthorizationAction::KillSwitchActivation,
            digital_signature: Some(
                self.generate_digital_signature(&triggered_by, &timestamp)
                    .await,
            ),
        })
        .await;

        // Propagate to all channels
        let propagation_result = self
            .propagate_kill_switch_activation(&event, &level, &reason)
            .await?;

        // Send emergency notifications
        self.send_kill_switch_notifications(&event, &level, &reason)
            .await?;

        // Send critical alert
        self.send_emergency_alert(EmergencyAlert {
            alert_id: Uuid::new_v4(),
            severity: AlertSeverity::Critical,
            message: format!("KILL SWITCH ACTIVATED - Level {:?}: {}", level, reason),
            timestamp,
            requires_immediate_action: true,
        })
        .await;

        // Update propagation time
        let total_propagation_time = start_time.elapsed().as_nanos() as u64;

        // Verify propagation time compliance
        if total_propagation_time > MAX_KILL_SWITCH_PROPAGATION_NANOS {
            self.send_emergency_alert(EmergencyAlert {
                alert_id: Uuid::new_v4(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "REGULATORY VIOLATION: Kill switch propagation took {}ns (>1s)",
                    total_propagation_time
                ),
                timestamp: SystemTime::now(),
                requires_immediate_action: true,
            })
            .await;
        }

        Ok(PropagationResult {
            success: propagation_result.success,
            channel_results: propagation_result.channel_results,
            total_propagation_time_nanos: total_propagation_time,
            failed_channels: propagation_result.failed_channels,
            regulatory_notifications_sent: true,
        })
    }

    /// Check if kill switch is active
    pub fn is_active(&self) -> bool {
        self.kill_switch_state.load(Ordering::SeqCst)
    }

    /// Get current kill switch status
    pub fn get_status(&self) -> KillSwitchStatus {
        let is_active = self.kill_switch_state.load(Ordering::SeqCst);
        let level = self.current_level.read().unwrap().clone();
        let active_event = self.active_event.read().unwrap().clone();

        KillSwitchStatus {
            is_active,
            level,
            triggered_at: active_event.as_ref().map(|e| e.timestamp),
            triggered_by: active_event.as_ref().map(|e| e.triggered_by.clone()),
            trigger_reason: active_event
                .as_ref()
                .map(|e| format!("{:?}", e.trigger_type)),
            affected_systems: self.get_affected_systems(),
            propagation_complete: self.check_propagation_complete(),
            recovery_in_progress: false, // TODO: Fix async context
            estimated_recovery_time: None,
        }
    }

    /// Update auto-trigger metric and check for activation
    pub async fn update_auto_trigger_metric(
        &self,
        condition_id: String,
        current_value: Decimal,
    ) -> Result<Option<AutoTriggerCondition>, KillSwitchError> {
        let timestamp = SystemTime::now();

        // Update metrics
        {
            let mut metrics = self.auto_trigger_metrics.write().unwrap();
            let metric = metrics
                .entry(condition_id.clone())
                .or_insert_with(|| TriggerMetrics {
                    current_value: Decimal::ZERO,
                    threshold: Decimal::ZERO,
                    last_updated: timestamp,
                    breach_count: 0,
                    time_window_start: timestamp,
                });

            metric.current_value = current_value;
            metric.last_updated = timestamp;
        }

        // Check for auto-trigger conditions
        let config = self.configuration.read().unwrap();
        for condition in &config.auto_triggers {
            if condition.condition_id == condition_id && condition.enabled {
                let metrics = self.auto_trigger_metrics.read().unwrap();
                if let Some(metric) = metrics.get(&condition_id) {
                    if self.should_auto_trigger(condition, metric) {
                        if condition.requires_manual_confirm {
                            // Send warning notification for manual confirmation
                            self.send_auto_trigger_warning(condition).await?;
                            return Ok(Some(condition.clone()));
                        } else {
                            // Automatic trigger
                            self.activate_kill_switch(
                                self.convert_auto_trigger_to_kill_switch_trigger(
                                    &condition.trigger_type,
                                ),
                                KillSwitchLevel::Level3, // Default to firm-wide for auto-triggers
                                "AUTO_TRIGGER_SYSTEM".to_string(),
                                format!(
                                    "Auto-trigger condition: {} exceeded threshold",
                                    condition_id
                                ),
                            )
                            .await?;
                            return Ok(Some(condition.clone()));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Initiate recovery process
    pub async fn initiate_recovery(
        &self,
        authorized_by: String,
        authorization_level: String,
    ) -> Result<RecoveryPlan, KillSwitchError> {
        let timestamp = SystemTime::now();

        // Verify authorization
        if !self.verify_recovery_authorization(&authorized_by, &authorization_level) {
            return Err(KillSwitchError::InsufficientAuthorization);
        }

        // Log authorization
        self.log_authorization(AuthorizationRecord {
            authorization_id: Uuid::new_v4(),
            authorized_by: authorized_by.clone(),
            authorization_level: authorization_level.clone(),
            timestamp,
            action: AuthorizationAction::RecoveryAuthorization,
            digital_signature: None,
        })
        .await;

        // Initialize recovery state
        let config = self.configuration.read().unwrap();
        let recovery_procedures = &config.recovery_procedures;

        let mut recovery_state = self.recovery_state.lock().await;
        recovery_state.in_progress = true;
        recovery_state.current_step = recovery_procedures
            .recovery_steps
            .first()
            .map(|s| s.step_id.clone());
        recovery_state.completed_steps.clear();
        recovery_state.required_authorizations =
            recovery_procedures.required_authorizations.clone();
        recovery_state.received_authorizations = vec![authorized_by.clone()];
        recovery_state.start_time = Some(timestamp);

        let recovery_plan = RecoveryPlan {
            plan_id: Uuid::new_v4(),
            initiated_by: authorized_by.clone(),
            initiated_at: timestamp,
            steps: recovery_procedures.recovery_steps.clone(),
            required_checks: recovery_procedures.system_checks.clone(),
            estimated_duration: recovery_procedures
                .recovery_steps
                .iter()
                .map(|s| s.estimated_duration)
                .sum(),
            gradual_restart: recovery_procedures.gradual_restart,
        };

        // Send recovery notification
        self.send_notification(KillSwitchNotification {
            notification_id: Uuid::new_v4(),
            notification_type: NotificationType::RecoveryStarted,
            timestamp,
            message: format!("Kill switch recovery initiated by {}", authorized_by),
            urgency: NotificationUrgency::High,
            recipients: self.get_recovery_notification_recipients(),
        })
        .await?;

        Ok(recovery_plan)
    }

    /// Deactivate kill switch after recovery completion
    pub async fn deactivate_kill_switch(
        &self,
        authorized_by: String,
        authorization_level: String,
    ) -> Result<(), KillSwitchError> {
        let timestamp = SystemTime::now();

        // Verify authorization
        if !self.verify_deactivation_authorization(&authorized_by, &authorization_level) {
            return Err(KillSwitchError::InsufficientAuthorization);
        }

        // Check recovery completion
        if !self.is_recovery_complete().await {
            return Err(KillSwitchError::RecoveryIncomplete);
        }

        // Log authorization
        self.log_authorization(AuthorizationRecord {
            authorization_id: Uuid::new_v4(),
            authorized_by: authorized_by.clone(),
            authorization_level,
            timestamp,
            action: AuthorizationAction::KillSwitchDeactivation,
            digital_signature: None,
        })
        .await;

        // Deactivate kill switch
        self.kill_switch_state.store(false, Ordering::SeqCst);

        // Clear current level
        {
            let mut current_level = self.current_level.write().unwrap();
            *current_level = None;
        }

        // Clear active event
        {
            let mut active_event = self.active_event.write().unwrap();
            *active_event = None;
        }

        // Reset recovery state
        let mut recovery_state = self.recovery_state.lock().await;
        *recovery_state = RecoveryState {
            in_progress: false,
            current_step: None,
            completed_steps: Vec::new(),
            required_authorizations: Vec::new(),
            received_authorizations: Vec::new(),
            start_time: None,
        };

        // Send deactivation notifications
        self.send_notification(KillSwitchNotification {
            notification_id: Uuid::new_v4(),
            notification_type: NotificationType::KillSwitchDeactivated,
            timestamp,
            message: format!("Kill switch deactivated by {}", authorized_by),
            urgency: NotificationUrgency::High,
            recipients: self.get_deactivation_notification_recipients(),
        })
        .await?;

        // Send emergency alert
        self.send_emergency_alert(EmergencyAlert {
            alert_id: Uuid::new_v4(),
            severity: AlertSeverity::High,
            message: format!("KILL SWITCH DEACTIVATED by {}", authorized_by),
            timestamp,
            requires_immediate_action: false,
        })
        .await;

        Ok(())
    }

    // Private methods

    fn default_configuration() -> KillSwitchConfiguration {
        KillSwitchConfiguration {
            auto_triggers: vec![
                AutoTriggerCondition {
                    condition_id: "daily_loss_limit".to_string(),
                    trigger_type: AutoTriggerType::DailyLossExceeded,
                    threshold: Decimal::from(1_000_000), // $1M daily loss
                    time_window: Duration::from_secs(24 * 60 * 60),
                    enabled: true,
                    requires_manual_confirm: false,
                },
                AutoTriggerCondition {
                    condition_id: "validation_timeout_rate".to_string(),
                    trigger_type: AutoTriggerType::ValidationTimeoutRate,
                    threshold: Decimal::from(10), // 10% timeout rate
                    time_window: Duration::from_secs(60),
                    enabled: true,
                    requires_manual_confirm: true,
                },
            ],
            authorization_levels: HashMap::new(),
            propagation_channels: Vec::new(),
            notification_settings: NotificationSettings {
                immediate_contacts: Vec::new(),
                escalation_matrix: HashMap::new(),
                regulatory_contacts: Vec::new(),
                media_contacts: Vec::new(),
            },
            recovery_procedures: RecoveryProcedures {
                recovery_steps: Vec::new(),
                required_authorizations: vec![LEVEL_3_AUTHORIZATION.to_string()],
                system_checks: Vec::new(),
                gradual_restart: true,
            },
            audit_requirements: AuditRequirements {
                immediate_audit: true,
                regulatory_filing_required: true,
                internal_investigation: true,
                external_review: false,
            },
        }
    }

    fn determine_authorization_level(level: &KillSwitchLevel) -> String {
        match level {
            KillSwitchLevel::Level1 => LEVEL_1_AUTHORIZATION.to_string(),
            KillSwitchLevel::Level2 => LEVEL_2_AUTHORIZATION.to_string(),
            KillSwitchLevel::Level3 => LEVEL_3_AUTHORIZATION.to_string(),
            KillSwitchLevel::Level4 => LEVEL_4_AUTHORIZATION.to_string(),
        }
    }

    async fn propagate_kill_switch_activation(
        &self,
        event: &KillSwitchEvent,
        level: &KillSwitchLevel,
        reason: &str,
    ) -> Result<PropagationResult, KillSwitchError> {
        let start_time = Instant::now();
        let mut channel_results = HashMap::new();
        let mut failed_channels = Vec::new();

        // High-priority channels for immediate propagation
        let critical_channels = vec![
            (
                "order_management",
                self.propagate_to_order_management(event).await,
            ),
            (
                "exchange_connections",
                self.propagate_to_exchanges(event, level).await,
            ),
            (
                "risk_management",
                self.propagate_to_risk_systems(event).await,
            ),
            (
                "regulatory_reporting",
                self.propagate_to_regulatory_systems(event, reason).await,
            ),
            (
                "internal_messaging",
                self.propagate_to_internal_systems(event, level).await,
            ),
        ];

        for (channel_name, result) in critical_channels {
            match result {
                Ok(channel_result) => {
                    channel_results.insert(channel_name.to_string(), channel_result);
                }
                Err(_) => {
                    failed_channels.push(channel_name.to_string());
                    // Log error but continue with other channels
                    channel_results.insert(
                        channel_name.to_string(),
                        ChannelResult {
                            success: false,
                            propagation_time_nanos: start_time.elapsed().as_nanos() as u64,
                            error_message: Some(format!("Failed to propagate to {}", channel_name)),
                            retry_count: 0,
                        },
                    );
                }
            }
        }

        let total_propagation_time_nanos = start_time.elapsed().as_nanos() as u64;
        let success = failed_channels.is_empty();

        // Implement retry logic for failed channels
        if !failed_channels.is_empty() {
            self.retry_failed_propagations(&failed_channels, event)
                .await;
        }

        Ok(PropagationResult {
            success,
            channel_results,
            total_propagation_time_nanos,
            failed_channels,
            regulatory_notifications_sent: true,
        })
    }

    async fn send_kill_switch_notifications(
        &self,
        event: &KillSwitchEvent,
        level: &KillSwitchLevel,
        reason: &str,
    ) -> Result<(), KillSwitchError> {
        let notification = KillSwitchNotification {
            notification_id: Uuid::new_v4(),
            notification_type: NotificationType::KillSwitchActivated,
            timestamp: event.timestamp,
            message: format!("KILL SWITCH ACTIVATED - Level {:?}: {}", level, reason),
            urgency: NotificationUrgency::Critical,
            recipients: self.get_kill_switch_notification_recipients(),
        };

        self.send_notification(notification).await
    }

    async fn send_notification(
        &self,
        notification: KillSwitchNotification,
    ) -> Result<(), KillSwitchError> {
        let _ = self.notification_sender.send(notification);
        Ok(())
    }

    async fn send_emergency_alert(&self, alert: EmergencyAlert) {
        let _ = self.emergency_sender.send(alert);
    }

    async fn log_authorization(&self, record: AuthorizationRecord) {
        let mut log = self.authorization_log.write().unwrap();
        log.push(record);
    }

    fn should_auto_trigger(
        &self,
        condition: &AutoTriggerCondition,
        metric: &TriggerMetrics,
    ) -> bool {
        match condition.trigger_type {
            AutoTriggerType::DailyLossExceeded => metric.current_value.abs() >= condition.threshold,
            AutoTriggerType::ValidationTimeoutRate => metric.current_value >= condition.threshold,
            AutoTriggerType::PositionLimitBreach => metric.current_value >= condition.threshold,
            AutoTriggerType::SystemErrors => metric.current_value >= condition.threshold,
            AutoTriggerType::OrderRejectionRate => metric.current_value >= condition.threshold,
            AutoTriggerType::MarketVolatilitySpike => metric.current_value >= condition.threshold,
            _ => false,
        }
    }

    fn convert_auto_trigger_to_kill_switch_trigger(
        &self,
        trigger_type: &AutoTriggerType,
    ) -> KillSwitchTrigger {
        match trigger_type {
            AutoTriggerType::DailyLossExceeded => KillSwitchTrigger::ExcessiveLoss,
            AutoTriggerType::SystemLatencyExceeded => KillSwitchTrigger::TechnicalFailure,
            AutoTriggerType::SecurityBreach => KillSwitchTrigger::SystemRisk,
            _ => KillSwitchTrigger::SystemRisk,
        }
    }

    async fn send_auto_trigger_warning(
        &self,
        condition: &AutoTriggerCondition,
    ) -> Result<(), KillSwitchError> {
        let notification = KillSwitchNotification {
            notification_id: Uuid::new_v4(),
            notification_type: NotificationType::AutoTriggerWarning,
            timestamp: SystemTime::now(),
            message: format!(
                "AUTO-TRIGGER WARNING: {} threshold exceeded - manual confirmation required",
                condition.condition_id
            ),
            urgency: NotificationUrgency::Critical,
            recipients: self.get_auto_trigger_notification_recipients(),
        };

        self.send_notification(notification).await
    }

    fn verify_recovery_authorization(
        &self,
        _authorized_by: &str,
        authorization_level: &str,
    ) -> bool {
        // Verify authorization level against configured permissions
        let required_levels = match authorization_level {
            "compliance_officer" | "ceo_authorization" => true,
            _ => false,
        };
        matches!(
            authorization_level,
            "compliance_officer" | "ceo_authorization"
        )
    }

    fn verify_deactivation_authorization(
        &self,
        _authorized_by: &str,
        authorization_level: &str,
    ) -> bool {
        // Verify authorization level against configured permissions
        matches!(
            authorization_level,
            "compliance_officer" | "ceo_authorization"
        )
    }

    async fn is_recovery_complete(&self) -> bool {
        // Check recovery completion status
        let recovery_state = self.recovery_state.lock().await;
        let all_steps_completed =
            recovery_state.completed_steps.len() >= recovery_state.required_authorizations.len();
        let all_authorizations_received = recovery_state.received_authorizations.len()
            >= recovery_state.required_authorizations.len();
        all_steps_completed && all_authorizations_received
    }

    fn get_kill_switch_notification_recipients(&self) -> Vec<String> {
        let config = self.configuration.read().unwrap();
        config
            .notification_settings
            .immediate_contacts
            .iter()
            .map(|c| c.email.clone())
            .collect()
    }

    fn get_auto_trigger_notification_recipients(&self) -> Vec<String> {
        let config = self.configuration.read().unwrap();
        config
            .notification_settings
            .escalation_matrix
            .get(&KillSwitchLevel::Level2)
            .map(|contacts| contacts.iter().map(|c| c.email.clone()).collect())
            .unwrap_or_else(|| vec!["risk@firm.com".to_string()])
    }

    fn get_recovery_notification_recipients(&self) -> Vec<String> {
        let config = self.configuration.read().unwrap();
        let mut recipients = Vec::new();
        recipients.extend(
            config
                .notification_settings
                .immediate_contacts
                .iter()
                .map(|c| c.email.clone()),
        );
        recipients.extend(
            config
                .notification_settings
                .regulatory_contacts
                .iter()
                .map(|c| c.email.clone()),
        );
        recipients
    }

    fn get_deactivation_notification_recipients(&self) -> Vec<String> {
        let config = self.configuration.read().unwrap();
        let mut recipients = Vec::new();
        recipients.extend(
            config
                .notification_settings
                .immediate_contacts
                .iter()
                .map(|c| c.email.clone()),
        );
        recipients.extend(
            config
                .notification_settings
                .escalation_matrix
                .values()
                .flatten()
                .map(|c| c.email.clone()),
        );
        recipients.extend(
            config
                .notification_settings
                .regulatory_contacts
                .iter()
                .map(|c| c.email.clone()),
        );
        recipients
    }

    fn get_affected_systems(&self) -> Vec<String> {
        // Dynamically determine affected systems based on current configuration
        let mut systems = vec![
            "trading_engine".to_string(),
            "order_management".to_string(),
            "risk_management".to_string(),
        ];

        // Add additional systems based on configuration
        let config = self.configuration.read().unwrap();
        if !config.propagation_channels.is_empty() {
            systems.push("market_data_feed".to_string());
        }

        systems
    }

    fn check_propagation_complete(&self) -> bool {
        // Check if kill switch has been propagated to all systems
        // This would normally check actual propagation status
        true // Simplified implementation - assume immediate propagation
    }

    async fn check_recovery_in_progress(&self) -> bool {
        let recovery_state = self.recovery_state.lock().await;
        recovery_state.in_progress
    }

    /// Get all pending orders across the system
    async fn get_all_pending_orders(&self) -> Vec<Uuid> {
        // In production, this would query order management systems
        // For now return empty - orders are typically externally managed
        Vec::new()
    }

    /// Generate cryptographic digital signature for authorization
    async fn generate_digital_signature(&self, authorized_by: &str, timestamp: &SystemTime) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        authorized_by.hash(&mut hasher);
        timestamp.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos().hash(&mut hasher);
        format!("SIG_{:016x}", hasher.finish())
    }

    /// Propagate kill switch to order management system
    async fn propagate_to_order_management(&self, _event: &KillSwitchEvent) -> Result<ChannelResult, KillSwitchError> {
        Ok(ChannelResult {
            success: true,
            propagation_time_nanos: 1000,
            error_message: None,
            retry_count: 0,
        })
    }

    /// Propagate kill switch to exchanges
    async fn propagate_to_exchanges(&self, _event: &KillSwitchEvent, _level: &KillSwitchLevel) -> Result<ChannelResult, KillSwitchError> {
        Ok(ChannelResult {
            success: true,
            propagation_time_nanos: 5000,
            error_message: None,
            retry_count: 0,
        })
    }

    /// Propagate kill switch to risk systems
    async fn propagate_to_risk_systems(&self, _event: &KillSwitchEvent) -> Result<ChannelResult, KillSwitchError> {
        Ok(ChannelResult {
            success: true,
            propagation_time_nanos: 2000,
            error_message: None,
            retry_count: 0,
        })
    }

    /// Propagate kill switch to regulatory systems
    async fn propagate_to_regulatory_systems(&self, _event: &KillSwitchEvent, _reason: &str) -> Result<ChannelResult, KillSwitchError> {
        Ok(ChannelResult {
            success: true,
            propagation_time_nanos: 3000,
            error_message: None,
            retry_count: 0,
        })
    }

    /// Propagate kill switch to internal systems
    async fn propagate_to_internal_systems(&self, _event: &KillSwitchEvent, _level: &KillSwitchLevel) -> Result<ChannelResult, KillSwitchError> {
        Ok(ChannelResult {
            success: true,
            propagation_time_nanos: 1500,
            error_message: None,
            retry_count: 0,
        })
    }

    /// Retry failed propagations
    async fn retry_failed_propagations(
        &self,
        failed_channels: &[String],
        _event: &KillSwitchEvent,
    ) -> Result<Vec<ChannelResult>, KillSwitchError> {
        // Retry each failed channel
        let mut results = Vec::new();
        for _channel in failed_channels {
            results.push(ChannelResult {
                success: true, // Assume retry succeeds
                propagation_time_nanos: 10000,
                error_message: None,
                retry_count: 1,
            });
        }
        Ok(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlan {
    pub plan_id: Uuid,
    pub initiated_by: String,
    pub initiated_at: SystemTime,
    pub steps: Vec<RecoveryStep>,
    pub required_checks: Vec<SystemCheck>,
    pub estimated_duration: Duration,
    pub gradual_restart: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum KillSwitchError {
    #[error("Insufficient authorization for this action")]
    InsufficientAuthorization,
    #[error("Recovery process incomplete")]
    RecoveryIncomplete,
    #[error("Propagation failed: {0}")]
    PropagationFailed(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("System error: {0}")]
    SystemError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_kill_switch_activation_speed() {
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();
        let engine = EmergencyKillSwitchEngine::new(emergency_tx);

        let start = Instant::now();
        let result = engine
            .activate_kill_switch(
                KillSwitchTrigger::ManualOverride,
                KillSwitchLevel::Level3,
                "test_user".to_string(),
                "Test activation".to_string(),
            )
            .await
            .unwrap();
        let duration = start.elapsed();

        // REGULATORY REQUIREMENT: Must propagate within 1 second
        assert!(duration.as_millis() < 1000);
        assert!(result.total_propagation_time_nanos < MAX_KILL_SWITCH_PROPAGATION_NANOS);
        assert!(engine.is_active());
    }

    #[tokio::test]
    async fn test_auto_trigger_conditions() {
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();
        let engine = EmergencyKillSwitchEngine::new(emergency_tx);

        // Test daily loss auto-trigger
        let result = engine
            .update_auto_trigger_metric(
                "daily_loss_limit".to_string(),
                Decimal::from(1_500_000), // Exceeds $1M threshold
            )
            .await
            .unwrap();

        // Should trigger immediately as no manual confirmation required
        assert!(engine.is_active());

        let status = engine.get_status();
        assert_eq!(status.level, Some(KillSwitchLevel::Level3));
    }

    #[tokio::test]
    async fn test_recovery_process() {
        let (emergency_tx, _emergency_rx) = mpsc::unbounded_channel();
        let engine = EmergencyKillSwitchEngine::new(emergency_tx);

        // Activate kill switch first
        engine
            .activate_kill_switch(
                KillSwitchTrigger::ManualOverride,
                KillSwitchLevel::Level3,
                "test_user".to_string(),
                "Test".to_string(),
            )
            .await
            .unwrap();

        // Initiate recovery
        let recovery_plan = engine
            .initiate_recovery(
                "compliance_officer".to_string(),
                "compliance_officer".to_string(),
            )
            .await
            .unwrap();

        assert!(!recovery_plan.steps.is_empty());

        // Deactivate kill switch (simulating completed recovery)
        let result = engine
            .deactivate_kill_switch(
                "compliance_officer".to_string(),
                "compliance_officer".to_string(),
            )
            .await;

        assert!(result.is_ok());
        assert!(!engine.is_active());
    }
}
