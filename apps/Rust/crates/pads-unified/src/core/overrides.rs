//! Decision Overrides Implementation
//!
//! System for handling decision overrides, emergency stops, and manual interventions
//! in the PADS trading system with comprehensive safety mechanisms.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Override types and priorities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OverrideType {
    /// Emergency stop - highest priority
    EmergencyStop,
    /// Manual intervention by trader
    ManualIntervention,
    /// Risk management override
    RiskManagement,
    /// Regulatory compliance override
    Compliance,
    /// System maintenance override
    Maintenance,
    /// Market condition override
    MarketCondition,
    /// Testing override
    Testing,
}

/// Override priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OverridePriority {
    /// Critical - cannot be overridden
    Critical = 0,
    /// High priority
    High = 1,
    /// Medium priority
    Medium = 2,
    /// Low priority
    Low = 3,
}

/// Override status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OverrideStatus {
    /// Active override
    Active,
    /// Pending activation
    Pending,
    /// Expired override
    Expired,
    /// Manually cancelled
    Cancelled,
    /// Override completed
    Completed,
}

/// Decision override structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOverride {
    /// Unique override ID
    pub id: Uuid,
    /// Override type
    pub override_type: OverrideType,
    /// Override priority
    pub priority: OverridePriority,
    /// Override status
    pub status: OverrideStatus,
    /// Override decision
    pub decision: TradingDecision,
    /// Override reason
    pub reason: String,
    /// Creator of the override
    pub creator: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Expiration timestamp
    pub expires_at: Option<u64>,
    /// Conditions for activation
    pub conditions: Vec<OverrideCondition>,
    /// Actions to take when activated
    pub actions: Vec<OverrideAction>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Conditions for override activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverrideCondition {
    /// Price condition
    Price {
        operator: ComparisonOperator,
        value: f64,
        symbol: String,
    },
    /// Time condition
    Time {
        operator: ComparisonOperator,
        timestamp: u64,
    },
    /// Volume condition
    Volume {
        operator: ComparisonOperator,
        value: f64,
        symbol: String,
    },
    /// Portfolio condition
    Portfolio {
        operator: ComparisonOperator,
        value: f64,
        metric: String,
    },
    /// Risk condition
    Risk {
        operator: ComparisonOperator,
        value: f64,
        risk_type: String,
    },
    /// Market condition
    Market {
        condition: String,
        parameters: HashMap<String, serde_json::Value>,
    },
    /// Custom condition
    Custom {
        condition: String,
        parameters: HashMap<String, serde_json::Value>,
    },
}

/// Comparison operators for conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Actions to take when override is activated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverrideAction {
    /// Stop all trading
    StopTrading,
    /// Close all positions
    CloseAllPositions,
    /// Close specific position
    ClosePosition { symbol: String },
    /// Limit position size
    LimitPositionSize { symbol: String, max_size: f64 },
    /// Block specific symbols
    BlockSymbols { symbols: Vec<String> },
    /// Send notification
    SendNotification { message: String, urgency: String },
    /// Log event
    LogEvent { message: String, level: String },
    /// Execute custom action
    CustomAction { action: String, parameters: HashMap<String, serde_json::Value> },
}

/// Override execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideResult {
    /// Override ID
    pub override_id: Uuid,
    /// Execution success
    pub success: bool,
    /// Applied decision
    pub applied_decision: Option<TradingDecision>,
    /// Execution time
    pub execution_time_ns: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Actions executed
    pub actions_executed: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Override configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideConfig {
    /// Enable override system
    pub enabled: bool,
    /// Maximum number of active overrides
    pub max_active_overrides: usize,
    /// Default override expiration time (seconds)
    pub default_expiry_seconds: u64,
    /// Enable emergency stop
    pub enable_emergency_stop: bool,
    /// Enable manual interventions
    pub enable_manual_interventions: bool,
    /// Require confirmation for high-priority overrides
    pub require_confirmation: bool,
    /// Audit log path
    pub audit_log_path: Option<String>,
    /// Notification settings
    pub notification_config: NotificationConfig,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<String>,
    /// Minimum priority for notifications
    pub min_priority: OverridePriority,
    /// Webhook URL
    pub webhook_url: Option<String>,
    /// Email settings
    pub email_config: Option<EmailConfig>,
}

/// Email configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// SMTP server
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// From address
    pub from_address: String,
    /// To addresses
    pub to_addresses: Vec<String>,
}

/// Override statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideStats {
    /// Total overrides created
    pub total_overrides: u64,
    /// Active overrides
    pub active_overrides: usize,
    /// Overrides by type
    pub overrides_by_type: HashMap<String, u64>,
    /// Overrides by priority
    pub overrides_by_priority: HashMap<String, u64>,
    /// Average execution time
    pub average_execution_time_ns: u64,
    /// Success rate
    pub success_rate: f64,
    /// Most recent override
    pub most_recent_override: Option<u64>,
}

/// Decision overrides manager
pub struct DecisionOverrides {
    /// Configuration
    config: OverrideConfig,
    /// Active overrides
    active_overrides: Arc<RwLock<HashMap<Uuid, DecisionOverride>>>,
    /// Override history
    override_history: Arc<RwLock<Vec<DecisionOverride>>>,
    /// Execution results
    execution_results: Arc<RwLock<Vec<OverrideResult>>>,
    /// Statistics
    stats: Arc<RwLock<OverrideStats>>,
    /// Emergency stop state
    emergency_stop: Arc<RwLock<bool>>,
    /// Blocked symbols
    blocked_symbols: Arc<RwLock<Vec<String>>>,
    /// Position limits
    position_limits: Arc<RwLock<HashMap<String, f64>>>,
}

impl Default for OverrideConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_active_overrides: 100,
            default_expiry_seconds: 3600, // 1 hour
            enable_emergency_stop: true,
            enable_manual_interventions: true,
            require_confirmation: true,
            audit_log_path: None,
            notification_config: NotificationConfig::default(),
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec!["console".to_string()],
            min_priority: OverridePriority::Medium,
            webhook_url: None,
            email_config: None,
        }
    }
}

impl Default for OverrideStats {
    fn default() -> Self {
        Self {
            total_overrides: 0,
            active_overrides: 0,
            overrides_by_type: HashMap::new(),
            overrides_by_priority: HashMap::new(),
            average_execution_time_ns: 0,
            success_rate: 0.0,
            most_recent_override: None,
        }
    }
}

impl DecisionOverrides {
    /// Create a new decision overrides manager
    pub async fn new(config: OverrideConfig) -> PadsResult<Self> {
        Ok(Self {
            config,
            active_overrides: Arc::new(RwLock::new(HashMap::new())),
            override_history: Arc::new(RwLock::new(Vec::new())),
            execution_results: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(OverrideStats::default())),
            emergency_stop: Arc::new(RwLock::new(false)),
            blocked_symbols: Arc::new(RwLock::new(Vec::new())),
            position_limits: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Check if a decision should be overridden
    pub async fn check_override(&self, original_decision: &TradingDecision, market_data: &MarketData) -> PadsResult<Option<OverrideResult>> {
        if !self.config.enabled {
            return Ok(None);
        }
        
        // Check emergency stop
        if *self.emergency_stop.read().await {
            return Ok(Some(self.create_emergency_stop_result(original_decision).await?));
        }
        
        // Check blocked symbols
        let blocked_symbols = self.blocked_symbols.read().await;
        if blocked_symbols.contains(&market_data.symbol) {
            return Ok(Some(self.create_blocked_symbol_result(original_decision, &market_data.symbol).await?));
        }
        
        // Check position limits
        let position_limits = self.position_limits.read().await;
        if let Some(&limit) = position_limits.get(&market_data.symbol) {
            if original_decision.amount > limit {
                return Ok(Some(self.create_position_limit_result(original_decision, &market_data.symbol, limit).await?));
            }
        }
        
        // Check active overrides
        let active_overrides = self.active_overrides.read().await;
        let mut matching_overrides: Vec<_> = active_overrides.values().collect();
        matching_overrides.sort_by(|a, b| a.priority.cmp(&b.priority));
        
        for override_rule in matching_overrides {
            if self.evaluate_conditions(override_rule, original_decision, market_data).await? {
                return Ok(Some(self.execute_override(override_rule, original_decision).await?));
            }
        }
        
        Ok(None)
    }
    
    /// Create a new override
    pub async fn create_override(&self, override_data: DecisionOverride) -> PadsResult<Uuid> {
        let mut active_overrides = self.active_overrides.write().await;
        
        // Check if we've reached the maximum number of active overrides
        if active_overrides.len() >= self.config.max_active_overrides {
            return Err(PadsError::validation("Maximum number of active overrides reached"));
        }
        
        // Set expiration if not provided
        let mut override_with_expiry = override_data;
        if override_with_expiry.expires_at.is_none() {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            override_with_expiry.expires_at = Some(current_time + self.config.default_expiry_seconds);
        }
        
        let override_id = override_with_expiry.id;
        active_overrides.insert(override_id, override_with_expiry.clone());
        
        // Update statistics
        self.update_stats_for_new_override(&override_with_expiry).await?;
        
        // Add to history
        let mut history = self.override_history.write().await;
        history.push(override_with_expiry.clone());
        
        // Keep history size manageable
        if history.len() > 10000 {
            history.remove(0);
        }
        
        // Send notification
        self.send_notification(&override_with_expiry).await?;
        
        // Audit log
        self.write_audit_log(&format!("Override created: {}", override_id), &override_with_expiry).await?;
        
        Ok(override_id)
    }
    
    /// Cancel an override
    pub async fn cancel_override(&self, override_id: Uuid, reason: String) -> PadsResult<()> {
        let mut active_overrides = self.active_overrides.write().await;
        
        if let Some(mut override_rule) = active_overrides.remove(&override_id) {
            override_rule.status = OverrideStatus::Cancelled;
            override_rule.metadata.insert("cancellation_reason".to_string(), serde_json::Value::String(reason.clone()));
            
            // Update history
            let mut history = self.override_history.write().await;
            if let Some(entry) = history.iter_mut().find(|o| o.id == override_id) {
                entry.status = OverrideStatus::Cancelled;
                entry.metadata = override_rule.metadata.clone();
            }
            
            // Audit log
            self.write_audit_log(&format!("Override cancelled: {} - {}", override_id, reason), &override_rule).await?;
            
            Ok(())
        } else {
            Err(PadsError::validation("Override not found"))
        }
    }
    
    /// Activate emergency stop
    pub async fn emergency_stop(&self, reason: String) -> PadsResult<()> {
        if !self.config.enable_emergency_stop {
            return Err(PadsError::validation("Emergency stop is disabled"));
        }
        
        let mut emergency_stop = self.emergency_stop.write().await;
        *emergency_stop = true;
        
        // Create emergency stop override
        let emergency_override = DecisionOverride {
            id: Uuid::new_v4(),
            override_type: OverrideType::EmergencyStop,
            priority: OverridePriority::Critical,
            status: OverrideStatus::Active,
            decision: TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: format!("Emergency stop: {}", reason),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            },
            reason: reason.clone(),
            creator: "system".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: None, // Emergency stops don't expire automatically
            conditions: Vec::new(),
            actions: vec![
                OverrideAction::StopTrading,
                OverrideAction::CloseAllPositions,
                OverrideAction::SendNotification {
                    message: format!("EMERGENCY STOP ACTIVATED: {}", reason),
                    urgency: "critical".to_string(),
                },
            ],
            metadata: HashMap::new(),
        };
        
        self.create_override(emergency_override).await?;
        
        // Execute emergency actions
        for action in &[
            OverrideAction::StopTrading,
            OverrideAction::CloseAllPositions,
            OverrideAction::SendNotification {
                message: format!("EMERGENCY STOP ACTIVATED: {}", reason),
                urgency: "critical".to_string(),
            },
        ] {
            self.execute_action(action).await?;
        }
        
        Ok(())
    }
    
    /// Deactivate emergency stop
    pub async fn deactivate_emergency_stop(&self, reason: String) -> PadsResult<()> {
        let mut emergency_stop = self.emergency_stop.write().await;
        *emergency_stop = false;
        
        // Cancel all emergency stop overrides
        let active_overrides = self.active_overrides.read().await;
        let emergency_overrides: Vec<Uuid> = active_overrides
            .values()
            .filter(|o| o.override_type == OverrideType::EmergencyStop)
            .map(|o| o.id)
            .collect();
        
        drop(active_overrides);
        
        for override_id in emergency_overrides {
            self.cancel_override(override_id, reason.clone()).await?;
        }
        
        // Audit log
        self.write_audit_log(&format!("Emergency stop deactivated: {}", reason), &DecisionOverride {
            id: Uuid::new_v4(),
            override_type: OverrideType::EmergencyStop,
            priority: OverridePriority::Critical,
            status: OverrideStatus::Cancelled,
            decision: TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: "Emergency stop deactivated".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            },
            reason: reason.clone(),
            creator: "system".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: None,
            conditions: Vec::new(),
            actions: Vec::new(),
            metadata: HashMap::new(),
        }).await?;
        
        Ok(())
    }
    
    /// Block trading for specific symbols
    pub async fn block_symbols(&self, symbols: Vec<String>, reason: String) -> PadsResult<()> {
        let mut blocked_symbols = self.blocked_symbols.write().await;
        
        for symbol in &symbols {
            if !blocked_symbols.contains(symbol) {
                blocked_symbols.push(symbol.clone());
            }
        }
        
        // Audit log
        self.write_audit_log(&format!("Symbols blocked: {:?} - {}", symbols, reason), &DecisionOverride {
            id: Uuid::new_v4(),
            override_type: OverrideType::RiskManagement,
            priority: OverridePriority::High,
            status: OverrideStatus::Active,
            decision: TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: format!("Symbols blocked: {}", reason),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            },
            reason: reason.clone(),
            creator: "system".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: None,
            conditions: Vec::new(),
            actions: Vec::new(),
            metadata: HashMap::new(),
        }).await?;
        
        Ok(())
    }
    
    /// Unblock trading for specific symbols
    pub async fn unblock_symbols(&self, symbols: Vec<String>) -> PadsResult<()> {
        let mut blocked_symbols = self.blocked_symbols.write().await;
        
        for symbol in &symbols {
            blocked_symbols.retain(|s| s != symbol);
        }
        
        Ok(())
    }
    
    /// Set position limits for symbols
    pub async fn set_position_limits(&self, limits: HashMap<String, f64>) -> PadsResult<()> {
        let mut position_limits = self.position_limits.write().await;
        
        for (symbol, limit) in limits {
            position_limits.insert(symbol, limit);
        }
        
        Ok(())
    }
    
    /// Get current override statistics
    pub async fn get_stats(&self) -> PadsResult<OverrideStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
    
    /// Get active overrides
    pub async fn get_active_overrides(&self) -> PadsResult<Vec<DecisionOverride>> {
        let active_overrides = self.active_overrides.read().await;
        Ok(active_overrides.values().cloned().collect())
    }
    
    /// Clean up expired overrides
    pub async fn cleanup_expired_overrides(&self) -> PadsResult<usize> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut active_overrides = self.active_overrides.write().await;
        let mut expired_count = 0;
        
        let expired_ids: Vec<Uuid> = active_overrides
            .values()
            .filter(|o| {
                if let Some(expires_at) = o.expires_at {
                    current_time > expires_at
                } else {
                    false
                }
            })
            .map(|o| o.id)
            .collect();
        
        for id in expired_ids {
            if let Some(mut override_rule) = active_overrides.remove(&id) {
                override_rule.status = OverrideStatus::Expired;
                expired_count += 1;
                
                // Update history
                let mut history = self.override_history.write().await;
                if let Some(entry) = history.iter_mut().find(|o| o.id == id) {
                    entry.status = OverrideStatus::Expired;
                }
            }
        }
        
        Ok(expired_count)
    }
    
    /// Evaluate conditions for an override
    async fn evaluate_conditions(&self, override_rule: &DecisionOverride, decision: &TradingDecision, market_data: &MarketData) -> PadsResult<bool> {
        if override_rule.conditions.is_empty() {
            return Ok(true); // No conditions means always active
        }
        
        for condition in &override_rule.conditions {
            if !self.evaluate_single_condition(condition, decision, market_data).await? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Evaluate a single condition
    async fn evaluate_single_condition(&self, condition: &OverrideCondition, decision: &TradingDecision, market_data: &MarketData) -> PadsResult<bool> {
        match condition {
            OverrideCondition::Price { operator, value, symbol } => {
                if symbol != &market_data.symbol {
                    return Ok(false);
                }
                
                self.compare_values(market_data.price, *value, operator)
            }
            OverrideCondition::Time { operator, timestamp } => {
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                self.compare_values(current_time as f64, *timestamp as f64, operator)
            }
            OverrideCondition::Volume { operator, value, symbol } => {
                if symbol != &market_data.symbol {
                    return Ok(false);
                }
                
                self.compare_values(market_data.volume, *value, operator)
            }
            OverrideCondition::Portfolio { operator, value, metric: _ } => {
                // Would need portfolio data - simplified for now
                Ok(true)
            }
            OverrideCondition::Risk { operator, value, risk_type: _ } => {
                // Would need risk calculation - simplified for now
                self.compare_values(decision.confidence, *value, operator)
            }
            OverrideCondition::Market { condition: _, parameters: _ } => {
                // Would need market condition evaluation - simplified for now
                Ok(true)
            }
            OverrideCondition::Custom { condition: _, parameters: _ } => {
                // Would need custom condition evaluation - simplified for now
                Ok(true)
            }
        }
    }
    
    /// Compare two values using an operator
    fn compare_values(&self, left: f64, right: f64, operator: &ComparisonOperator) -> PadsResult<bool> {
        let result = match operator {
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::LessThanOrEqual => left <= right,
        };
        
        Ok(result)
    }
    
    /// Execute an override
    async fn execute_override(&self, override_rule: &DecisionOverride, original_decision: &TradingDecision) -> PadsResult<OverrideResult> {
        let start_time = std::time::Instant::now();
        let mut actions_executed = Vec::new();
        
        // Execute actions
        for action in &override_rule.actions {
            match self.execute_action(action).await {
                Ok(action_name) => actions_executed.push(action_name),
                Err(e) => {
                    return Ok(OverrideResult {
                        override_id: override_rule.id,
                        success: false,
                        applied_decision: None,
                        execution_time_ns: start_time.elapsed().as_nanos() as u64,
                        error: Some(e.to_string()),
                        actions_executed,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                }
            }
        }
        
        let execution_time = start_time.elapsed().as_nanos() as u64;
        
        let result = OverrideResult {
            override_id: override_rule.id,
            success: true,
            applied_decision: Some(override_rule.decision.clone()),
            execution_time_ns: execution_time,
            error: None,
            actions_executed,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Store execution result
        let mut execution_results = self.execution_results.write().await;
        execution_results.push(result.clone());
        
        // Keep results history manageable
        if execution_results.len() > 1000 {
            execution_results.remove(0);
        }
        
        Ok(result)
    }
    
    /// Execute an action
    async fn execute_action(&self, action: &OverrideAction) -> PadsResult<String> {
        match action {
            OverrideAction::StopTrading => {
                // Would implement actual trading stop
                Ok("stop_trading".to_string())
            }
            OverrideAction::CloseAllPositions => {
                // Would implement position closing
                Ok("close_all_positions".to_string())
            }
            OverrideAction::ClosePosition { symbol } => {
                // Would implement specific position closing
                Ok(format!("close_position_{}", symbol))
            }
            OverrideAction::LimitPositionSize { symbol, max_size } => {
                self.set_position_limits([(symbol.clone(), *max_size)].into_iter().collect()).await?;
                Ok(format!("limit_position_{}_{}", symbol, max_size))
            }
            OverrideAction::BlockSymbols { symbols } => {
                self.block_symbols(symbols.clone(), "Override action".to_string()).await?;
                Ok(format!("block_symbols_{:?}", symbols))
            }
            OverrideAction::SendNotification { message, urgency: _ } => {
                // Would implement actual notification sending
                println!("NOTIFICATION: {}", message);
                Ok("send_notification".to_string())
            }
            OverrideAction::LogEvent { message, level: _ } => {
                // Would implement actual logging
                println!("LOG: {}", message);
                Ok("log_event".to_string())
            }
            OverrideAction::CustomAction { action, parameters: _ } => {
                // Would implement custom action execution
                Ok(format!("custom_action_{}", action))
            }
        }
    }
    
    /// Create emergency stop result
    async fn create_emergency_stop_result(&self, original_decision: &TradingDecision) -> PadsResult<OverrideResult> {
        Ok(OverrideResult {
            override_id: Uuid::new_v4(),
            success: true,
            applied_decision: Some(TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: "Emergency stop active".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            }),
            execution_time_ns: 0,
            error: None,
            actions_executed: vec!["emergency_stop".to_string()],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
    
    /// Create blocked symbol result
    async fn create_blocked_symbol_result(&self, original_decision: &TradingDecision, symbol: &str) -> PadsResult<OverrideResult> {
        Ok(OverrideResult {
            override_id: Uuid::new_v4(),
            success: true,
            applied_decision: Some(TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: format!("Symbol {} is blocked", symbol),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            }),
            execution_time_ns: 0,
            error: None,
            actions_executed: vec!["block_symbol".to_string()],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
    
    /// Create position limit result
    async fn create_position_limit_result(&self, original_decision: &TradingDecision, symbol: &str, limit: f64) -> PadsResult<OverrideResult> {
        Ok(OverrideResult {
            override_id: Uuid::new_v4(),
            success: true,
            applied_decision: Some(TradingDecision {
                decision_type: original_decision.decision_type.clone(),
                confidence: original_decision.confidence,
                amount: limit,
                reasoning: format!("Position size limited to {} for {}", limit, symbol),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            }),
            execution_time_ns: 0,
            error: None,
            actions_executed: vec!["limit_position".to_string()],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
    
    /// Update statistics for new override
    async fn update_stats_for_new_override(&self, override_rule: &DecisionOverride) -> PadsResult<()> {
        let mut stats = self.stats.write().await;
        
        stats.total_overrides += 1;
        stats.active_overrides += 1;
        
        let type_key = format!("{:?}", override_rule.override_type);
        *stats.overrides_by_type.entry(type_key).or_insert(0) += 1;
        
        let priority_key = format!("{:?}", override_rule.priority);
        *stats.overrides_by_priority.entry(priority_key).or_insert(0) += 1;
        
        stats.most_recent_override = Some(override_rule.created_at);
        
        Ok(())
    }
    
    /// Send notification for override
    async fn send_notification(&self, override_rule: &DecisionOverride) -> PadsResult<()> {
        if !self.config.notification_config.enabled {
            return Ok(());
        }
        
        if override_rule.priority > self.config.notification_config.min_priority {
            return Ok(());
        }
        
        let message = format!(
            "Override {} activated: {} (Priority: {:?})",
            override_rule.id,
            override_rule.reason,
            override_rule.priority
        );
        
        // Send to configured channels
        for channel in &self.config.notification_config.channels {
            match channel.as_str() {
                "console" => println!("NOTIFICATION: {}", message),
                "webhook" => {
                    if let Some(webhook_url) = &self.config.notification_config.webhook_url {
                        // Would implement webhook sending
                        println!("WEBHOOK to {}: {}", webhook_url, message);
                    }
                }
                "email" => {
                    if let Some(_email_config) = &self.config.notification_config.email_config {
                        // Would implement email sending
                        println!("EMAIL: {}", message);
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Write audit log entry
    async fn write_audit_log(&self, message: &str, override_rule: &DecisionOverride) -> PadsResult<()> {
        if let Some(audit_log_path) = &self.config.audit_log_path {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            let log_entry = format!(
                "{} - {} - Override: {} - Type: {:?} - Priority: {:?} - Creator: {}\n",
                timestamp,
                message,
                override_rule.id,
                override_rule.override_type,
                override_rule.priority,
                override_rule.creator
            );
            
            // Would implement actual file writing
            tokio::fs::write(audit_log_path, log_entry).await
                .map_err(|e| PadsError::io(format!("Failed to write audit log: {}", e)))?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_override_creation() {
        let config = OverrideConfig::default();
        let overrides = DecisionOverrides::new(config).await.unwrap();
        
        let override_data = DecisionOverride {
            id: Uuid::new_v4(),
            override_type: OverrideType::ManualIntervention,
            priority: OverridePriority::High,
            status: OverrideStatus::Active,
            decision: TradingDecision {
                decision_type: DecisionType::Hold,
                confidence: 1.0,
                amount: 0.0,
                reasoning: "Test override".to_string(),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            },
            reason: "Test reason".to_string(),
            creator: "test_user".to_string(),
            created_at: 1234567890,
            expires_at: Some(1234567890 + 3600),
            conditions: Vec::new(),
            actions: Vec::new(),
            metadata: HashMap::new(),
        };
        
        let override_id = overrides.create_override(override_data).await.unwrap();
        
        let active_overrides = overrides.get_active_overrides().await.unwrap();
        assert_eq!(active_overrides.len(), 1);
        assert_eq!(active_overrides[0].id, override_id);
    }
    
    #[tokio::test]
    async fn test_emergency_stop() {
        let config = OverrideConfig::default();
        let overrides = DecisionOverrides::new(config).await.unwrap();
        
        overrides.emergency_stop("Test emergency".to_string()).await.unwrap();
        
        let market_data = MarketData {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 1000.0,
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let original_decision = TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            amount: 100.0,
            reasoning: "Test decision".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let result = overrides.check_override(&original_decision, &market_data).await.unwrap();
        assert!(result.is_some());
        
        let override_result = result.unwrap();
        assert_eq!(override_result.applied_decision.unwrap().decision_type, DecisionType::Hold);
    }
    
    #[tokio::test]
    async fn test_symbol_blocking() {
        let config = OverrideConfig::default();
        let overrides = DecisionOverrides::new(config).await.unwrap();
        
        overrides.block_symbols(vec!["BTCUSDT".to_string()], "Test blocking".to_string()).await.unwrap();
        
        let market_data = MarketData {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 1000.0,
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let original_decision = TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            amount: 100.0,
            reasoning: "Test decision".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let result = overrides.check_override(&original_decision, &market_data).await.unwrap();
        assert!(result.is_some());
        
        let override_result = result.unwrap();
        assert_eq!(override_result.applied_decision.unwrap().decision_type, DecisionType::Hold);
    }
    
    #[tokio::test]
    async fn test_position_limits() {
        let config = OverrideConfig::default();
        let overrides = DecisionOverrides::new(config).await.unwrap();
        
        let mut limits = HashMap::new();
        limits.insert("BTCUSDT".to_string(), 50.0);
        overrides.set_position_limits(limits).await.unwrap();
        
        let market_data = MarketData {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 1000.0,
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let original_decision = TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            amount: 100.0, // Above limit
            reasoning: "Test decision".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        };
        
        let result = overrides.check_override(&original_decision, &market_data).await.unwrap();
        assert!(result.is_some());
        
        let override_result = result.unwrap();
        assert_eq!(override_result.applied_decision.unwrap().amount, 50.0);
    }
}
