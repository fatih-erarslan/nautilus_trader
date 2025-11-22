//! Comprehensive audit trail system - every action is logged immutably

use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rust_decimal::Decimal;
use ringbuf::{HeapRb, Rb};
use dashmap::DashMap;
use crate::error::ComplianceResult;

/// Immutable audit record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub actor: String,
    pub details: serde_json::Value,
    pub outcome: AuditOutcome,
    pub risk_score: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    // Trade events
    TradeSubmitted { order_id: Uuid },
    TradeApproved { order_id: Uuid },
    TradeRejected { order_id: Uuid, reason: String },
    TradeExecuted { order_id: Uuid, execution_price: Decimal },
    TradeCancelled { order_id: Uuid },
    
    // Compliance events
    ComplianceCheckPassed { rule_id: String },
    ComplianceCheckFailed { rule_id: String, violation: String },
    RiskLimitUpdated { limit_type: String, old_value: Decimal, new_value: Decimal },
    CircuitBreakerTriggered { reason: String },
    KillSwitchActivated { trigger: String },
    
    // Surveillance events
    SuspiciousActivityDetected { pattern: String, confidence: f64 },
    MarketManipulationAlert { type_: String, details: String },
    UnusualTradingPattern { description: String },
    
    // System events
    SystemStartup,
    SystemShutdown { reason: String },
    ConfigurationChange { component: String, changes: String },
    EmergencyOverride { authorized_by: String, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure { error: String },
    Warning { message: String },
    Critical { alert: String },
}

/// High-performance audit trail with multiple storage backends
pub struct AuditTrail {
    /// In-memory ring buffer for recent events (fast access)
    recent_events: Arc<RwLock<HeapRb<AuditRecord>>>,
    
    /// Indexed storage for quick lookups
    indexed_records: Arc<DashMap<Uuid, AuditRecord>>,
    
    /// Write-ahead log for durability
    wal_path: Option<String>,
    
    /// Metrics
    total_records: Arc<RwLock<u64>>,
    critical_events: Arc<RwLock<u64>>,
}

impl AuditTrail {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            recent_events: Arc::new(RwLock::new(HeapRb::new(buffer_size))),
            indexed_records: Arc::new(DashMap::new()),
            wal_path: None,
            total_records: Arc::new(RwLock::new(0)),
            critical_events: Arc::new(RwLock::new(0)),
        }
    }

    pub fn with_wal(buffer_size: usize, wal_path: String) -> Self {
        let mut trail = Self::new(buffer_size);
        trail.wal_path = Some(wal_path);
        trail
    }

    pub async fn record(&self, event: AuditEventType, actor: String, details: serde_json::Value) -> ComplianceResult<Uuid> {
        let record = AuditRecord {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: event.clone(),
            actor,
            details,
            outcome: AuditOutcome::Success,
            risk_score: self.calculate_risk_score(&event),
            metadata: None,
        };

        self.store_record(record.clone()).await?;
        Ok(record.id)
    }

    pub async fn record_with_outcome(
        &self,
        event: AuditEventType,
        actor: String,
        details: serde_json::Value,
        outcome: AuditOutcome,
    ) -> ComplianceResult<Uuid> {
        let record = AuditRecord {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: event.clone(),
            actor,
            details,
            outcome: outcome.clone(),
            risk_score: self.calculate_risk_score(&event),
            metadata: None,
        };

        // Track critical events
        if matches!(outcome, AuditOutcome::Critical { .. }) {
            *self.critical_events.write() += 1;
        }

        self.store_record(record.clone()).await?;
        Ok(record.id)
    }

    async fn store_record(&self, record: AuditRecord) -> ComplianceResult<()> {
        // Store in ring buffer
        self.recent_events.write().push_overwrite(record.clone());
        
        // Store in indexed map
        self.indexed_records.insert(record.id, record.clone());
        
        // Increment counter
        *self.total_records.write() += 1;
        
        // Write to WAL if configured
        if let Some(ref _wal_path) = self.wal_path {
            // TODO: Implement WAL writing
            // This would write to disk for durability
        }
        
        Ok(())
    }

    pub fn get_recent_events(&self, count: usize) -> Vec<AuditRecord> {
        let buffer = self.recent_events.read();
        buffer.iter().rev().take(count).cloned().collect()
    }

    pub fn get_by_id(&self, id: &Uuid) -> Option<AuditRecord> {
        self.indexed_records.get(id).map(|r| r.clone())
    }

    pub fn search_by_actor(&self, actor: &str) -> Vec<AuditRecord> {
        self.indexed_records
            .iter()
            .filter(|entry| entry.value().actor == actor)
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn get_critical_events(&self) -> Vec<AuditRecord> {
        self.indexed_records
            .iter()
            .filter(|entry| matches!(entry.value().outcome, AuditOutcome::Critical { .. }))
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn get_events_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AuditRecord> {
        self.indexed_records
            .iter()
            .filter(|entry| {
                let ts = entry.value().timestamp;
                ts >= start && ts <= end
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    fn calculate_risk_score(&self, event: &AuditEventType) -> Option<f64> {
        match event {
            AuditEventType::TradeRejected { .. } => Some(0.3),
            AuditEventType::ComplianceCheckFailed { .. } => Some(0.7),
            AuditEventType::CircuitBreakerTriggered { .. } => Some(1.0),
            AuditEventType::KillSwitchActivated { .. } => Some(1.0),
            AuditEventType::SuspiciousActivityDetected { confidence, .. } => Some(confidence / 100.0),
            AuditEventType::MarketManipulationAlert { .. } => Some(0.9),
            AuditEventType::UnusualTradingPattern { .. } => Some(0.6),
            AuditEventType::EmergencyOverride { .. } => Some(0.8),
            _ => None,
        }
    }

    pub fn get_statistics(&self) -> AuditStatistics {
        AuditStatistics {
            total_records: *self.total_records.read(),
            critical_events: *self.critical_events.read(),
            buffer_utilization: {
                let buffer = self.recent_events.read();
                (buffer.len() as f64 / buffer.capacity() as f64) * 100.0
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditStatistics {
    pub total_records: u64,
    pub critical_events: u64,
    pub buffer_utilization: f64,
}

/// Compliance report generator
pub struct ComplianceReporter {
    audit_trail: Arc<AuditTrail>,
}

impl ComplianceReporter {
    pub fn new(audit_trail: Arc<AuditTrail>) -> Self {
        Self { audit_trail }
    }

    pub async fn generate_daily_report(&self, date: DateTime<Utc>) -> ComplianceResult<ComplianceReport> {
        let start = date.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
        let end = date.date_naive().and_hms_opt(23, 59, 59).unwrap().and_utc();
        
        let events = self.audit_trail.get_events_in_range(start, end);
        
        let total_trades = events.iter()
            .filter(|e| matches!(e.event_type, AuditEventType::TradeExecuted { .. }))
            .count();
        
        let rejected_trades = events.iter()
            .filter(|e| matches!(e.event_type, AuditEventType::TradeRejected { .. }))
            .count();
        
        let compliance_violations = events.iter()
            .filter(|e| matches!(e.event_type, AuditEventType::ComplianceCheckFailed { .. }))
            .count();
        
        let critical_events = events.iter()
            .filter(|e| matches!(e.outcome, AuditOutcome::Critical { .. }))
            .collect::<Vec<_>>();
        
        Ok(ComplianceReport {
            date,
            total_events: events.len(),
            total_trades,
            rejected_trades,
            compliance_violations,
            critical_events: critical_events.len(),
            rejection_rate: if total_trades > 0 {
                (rejected_trades as f64 / total_trades as f64) * 100.0
            } else {
                0.0
            },
            events,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceReport {
    pub date: DateTime<Utc>,
    pub total_events: usize,
    pub total_trades: usize,
    pub rejected_trades: usize,
    pub compliance_violations: usize,
    pub critical_events: usize,
    pub rejection_rate: f64,
    pub events: Vec<AuditRecord>,
}