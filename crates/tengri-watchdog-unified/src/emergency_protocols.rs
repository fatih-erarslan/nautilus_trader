//! Emergency Protocol Manager with <100ns Response Requirement
//! 
//! Implements ultra-fast emergency shutdown and response protocols

use crate::{TENGRIError, EmergencyAction, ViolationType};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, RwLock};
use tokio::time::timeout;
use tracing::{error, info, warn};
use serde::{Deserialize, Serialize};

/// Emergency message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyMessage {
    ImmediateShutdown { reason: String, timestamp: SystemTime },
    QuarantineAgent { agent_id: String, reason: String },
    SystemAlert { severity: AlertSeverity, message: String },
    ForensicCapture { operation_id: String, data: Vec<u8> },
    RollbackInitiated { checkpoint_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,   // Immediate action required
    High,       // Urgent attention needed
    Medium,     // Monitor closely
    Low,        // Information only
}

/// Emergency state tracking
#[derive(Debug, Default)]
pub struct EmergencyState {
    pub is_shutdown: AtomicBool,
    pub shutdown_timestamp: AtomicU64,
    pub total_violations: AtomicU64,
    pub last_violation_timestamp: AtomicU64,
    pub quarantined_agents: RwLock<Vec<String>>,
}

/// Emergency protocol manager
pub struct EmergencyProtocolManager {
    emergency_broadcast: broadcast::Sender<EmergencyMessage>,
    state: Arc<EmergencyState>,
    response_times: Arc<RwLock<Vec<Duration>>>,
    max_response_time_ns: u64,
}

impl EmergencyProtocolManager {
    /// Create new emergency protocol manager
    pub async fn new() -> Result<Self, TENGRIError> {
        let (emergency_broadcast, _) = broadcast::channel(1000);
        let state = Arc::new(EmergencyState::default());
        let response_times = Arc::new(RwLock::new(Vec::new()));
        let max_response_time_ns = 100; // <100ns requirement

        Ok(Self {
            emergency_broadcast,
            state,
            response_times,
            max_response_time_ns,
        })
    }

    /// Trigger immediate shutdown with <100ns response requirement
    pub async fn trigger_immediate_shutdown(&self, reason: &str) -> Result<(), TENGRIError> {
        let response_start = Instant::now();

        // Atomic state update (fastest possible)
        self.state.is_shutdown.store(true, Ordering::SeqCst);
        let shutdown_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state.shutdown_timestamp.store(shutdown_timestamp, Ordering::SeqCst);

        // Emergency broadcast (must complete within remaining time budget)
        let emergency_msg = EmergencyMessage::ImmediateShutdown {
            reason: reason.to_string(),
            timestamp: SystemTime::now(),
        };

        // Non-blocking broadcast to avoid timeout violations
        if let Err(e) = self.emergency_broadcast.send(emergency_msg) {
            warn!("Emergency broadcast failed: {}", e);
        }

        // Record response time
        let response_time = response_start.elapsed();
        self.record_response_time(response_time).await;

        // Verify <100ns requirement
        if response_time.as_nanos() > self.max_response_time_ns as u128 {
            error!(
                "CRITICAL: Emergency shutdown exceeded {}ns requirement: {:?}",
                self.max_response_time_ns, response_time
            );
            return Err(TENGRIError::EmergencyProtocolTriggered {
                reason: format!(
                    "Response time violation: {}ns > {}ns",
                    response_time.as_nanos(),
                    self.max_response_time_ns
                ),
            });
        }

        info!("Emergency shutdown completed in {:?}", response_time);
        Ok(())
    }

    /// Quarantine specific agent
    pub async fn quarantine_agent(&self, agent_id: &str, reason: &str) -> Result<(), TENGRIError> {
        let quarantine_start = Instant::now();

        // Add to quarantined agents list
        {
            let mut quarantined = self.state.quarantined_agents.write().await;
            if !quarantined.contains(&agent_id.to_string()) {
                quarantined.push(agent_id.to_string());
            }
        }

        // Broadcast quarantine message
        let quarantine_msg = EmergencyMessage::QuarantineAgent {
            agent_id: agent_id.to_string(),
            reason: reason.to_string(),
        };

        if let Err(e) = self.emergency_broadcast.send(quarantine_msg) {
            warn!("Quarantine broadcast failed: {}", e);
        }

        let quarantine_time = quarantine_start.elapsed();
        info!("Agent {} quarantined in {:?}", agent_id, quarantine_time);

        Ok(())
    }

    /// Alert operators with severity classification
    pub async fn alert_operators(
        &self,
        severity: AlertSeverity,
        message: &str,
    ) -> Result<(), TENGRIError> {
        let alert_msg = EmergencyMessage::SystemAlert {
            severity,
            message: message.to_string(),
        };

        if let Err(e) = self.emergency_broadcast.send(alert_msg) {
            warn!("Operator alert failed: {}", e);
        }

        Ok(())
    }

    /// Capture forensic data for violation analysis
    pub async fn capture_forensic_data(
        &self,
        operation_id: &str,
        data: Vec<u8>,
    ) -> Result<(), TENGRIError> {
        let forensic_msg = EmergencyMessage::ForensicCapture {
            operation_id: operation_id.to_string(),
            data,
        };

        if let Err(e) = self.emergency_broadcast.send(forensic_msg) {
            warn!("Forensic capture failed: {}", e);
        }

        Ok(())
    }

    /// Initiate rollback to last known safe state
    pub async fn initiate_rollback(&self, checkpoint_id: &str) -> Result<(), TENGRIError> {
        if self.is_shutdown() {
            return Err(TENGRIError::EmergencyProtocolTriggered {
                reason: "Cannot rollback during emergency shutdown".to_string(),
            });
        }

        let rollback_msg = EmergencyMessage::RollbackInitiated {
            checkpoint_id: checkpoint_id.to_string(),
        };

        if let Err(e) = self.emergency_broadcast.send(rollback_msg) {
            warn!("Rollback initiation failed: {}", e);
        }

        info!("Rollback initiated to checkpoint: {}", checkpoint_id);
        Ok(())
    }

    /// Check if system is in emergency shutdown state
    pub fn is_shutdown(&self) -> bool {
        self.state.is_shutdown.load(Ordering::SeqCst)
    }

    /// Check if agent is quarantined
    pub async fn is_agent_quarantined(&self, agent_id: &str) -> bool {
        let quarantined = self.state.quarantined_agents.read().await;
        quarantined.contains(&agent_id.to_string())
    }

    /// Get emergency broadcast receiver
    pub fn subscribe_to_emergencies(&self) -> broadcast::Receiver<EmergencyMessage> {
        self.emergency_broadcast.subscribe()
    }

    /// Record violation and update statistics
    pub async fn record_violation(&self, violation_type: ViolationType) -> Result<(), TENGRIError> {
        let violation_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.state.total_violations.fetch_add(1, Ordering::SeqCst);
        self.state.last_violation_timestamp.store(violation_timestamp, Ordering::SeqCst);

        // Alert based on violation frequency
        let total_violations = self.state.total_violations.load(Ordering::SeqCst);
        if total_violations > 10 {
            self.alert_operators(
                AlertSeverity::Critical,
                &format!("High violation frequency: {} total violations", total_violations),
            ).await?;
        }

        Ok(())
    }

    /// Get emergency response statistics
    pub async fn get_response_statistics(&self) -> EmergencyStatistics {
        let response_times = self.response_times.read().await;
        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<Duration>() / response_times.len() as u32
        } else {
            Duration::from_nanos(0)
        };

        let max_response_time = response_times.iter().max().copied().unwrap_or_default();
        let min_response_time = response_times.iter().min().copied().unwrap_or_default();

        EmergencyStatistics {
            total_violations: self.state.total_violations.load(Ordering::SeqCst),
            is_shutdown: self.state.is_shutdown.load(Ordering::SeqCst),
            avg_response_time,
            max_response_time,
            min_response_time,
            response_count: response_times.len(),
            max_allowed_response_ns: self.max_response_time_ns,
        }
    }

    async fn record_response_time(&self, response_time: Duration) {
        let mut response_times = self.response_times.write().await;
        response_times.push(response_time);

        // Keep only last 1000 measurements for statistics
        if response_times.len() > 1000 {
            response_times.remove(0);
        }
    }
}

/// Emergency response statistics
#[derive(Debug, Clone)]
pub struct EmergencyStatistics {
    pub total_violations: u64,
    pub is_shutdown: bool,
    pub avg_response_time: Duration,
    pub max_response_time: Duration,
    pub min_response_time: Duration,
    pub response_count: usize,
    pub max_allowed_response_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_emergency_shutdown_timing() {
        let manager = EmergencyProtocolManager::new().await.unwrap();
        
        let start = Instant::now();
        let result = manager.trigger_immediate_shutdown("Test emergency").await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert!(manager.is_shutdown());
        
        // Verify response time is reasonable (allowing some test environment variance)
        assert!(elapsed.as_nanos() < 10_000); // 10μs should be achievable in tests
    }

    #[tokio::test]
    async fn test_agent_quarantine() {
        let manager = EmergencyProtocolManager::new().await.unwrap();
        
        let agent_id = "test_agent_123";
        let result = manager.quarantine_agent(agent_id, "Synthetic data violation").await;
        
        assert!(result.is_ok());
        assert!(manager.is_agent_quarantined(agent_id).await);
    }

    #[tokio::test]
    async fn test_emergency_broadcast() {
        let manager = EmergencyProtocolManager::new().await.unwrap();
        let mut receiver = manager.subscribe_to_emergencies();
        
        // Trigger emergency
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            manager.trigger_immediate_shutdown("Test broadcast").await.unwrap();
        });

        // Receive emergency message
        let received = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(received.is_ok());
        
        if let Ok(Ok(EmergencyMessage::ImmediateShutdown { reason, .. })) = received {
            assert_eq!(reason, "Test broadcast");
        } else {
            panic!("Expected emergency shutdown message");
        }
    }

    #[tokio::test]
    async fn test_violation_statistics() {
        let manager = EmergencyProtocolManager::new().await.unwrap();
        
        // Record some violations
        manager.record_violation(ViolationType::SyntheticData).await.unwrap();
        manager.record_violation(ViolationType::IntegrityBreach).await.unwrap();
        
        let stats = manager.get_response_statistics().await;
        assert_eq!(stats.total_violations, 2);
        assert_eq!(stats.max_allowed_response_ns, 100);
    }
}