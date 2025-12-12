// Runtime Anti-Mock Monitor - Continuous Enforcement
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{error, warn, debug};

use super::{DataSource, RuntimeViolation, ViolationType, ViolationSeverity, ValidationError};

/// Runtime monitor for continuous mock detection
#[derive(Debug)]
pub struct RuntimeMonitor {
    active_connections: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
    violation_history: Arc<RwLock<Vec<RuntimeViolation>>>,
    monitoring_config: MonitoringConfig,
    is_monitoring: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connection_id: String,
    pub connection_type: String,
    pub endpoint: String,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub data_source_type: String,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStatus {
    Valid,
    Suspicious,
    Blocked,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub scan_interval: Duration,
    pub connection_timeout: Duration,
    pub max_violations_per_hour: u32,
    pub auto_block_critical: bool,
    pub log_all_violations: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            scan_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(300),
            max_violations_per_hour: 10,
            auto_block_critical: true,
            log_all_violations: true,
        }
    }
}

impl RuntimeMonitor {
    pub fn new() -> Self {
        Self {
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            violation_history: Arc::new(RwLock::new(Vec::new())),
            monitoring_config: MonitoringConfig::default(),
            is_monitoring: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start continuous monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        debug!("Starting runtime mock detection monitoring");
        
        *self.is_monitoring.write().await = true;
        
        let active_connections = self.active_connections.clone();
        let violation_history = self.violation_history.clone();
        let config = self.monitoring_config.clone();
        let is_monitoring = self.is_monitoring.clone();
        
        tokio::spawn(async move {
            while *is_monitoring.read().await {
                // Scan for violations
                let violations = Self::scan_connections(&active_connections).await;
                
                // Process violations
                for violation in violations {
                    Self::process_violation(&violation_history, violation, &config).await;
                }
                
                // Clean up stale connections
                Self::cleanup_stale_connections(&active_connections, &config).await;
                
                // Wait for next scan
                tokio::time::sleep(config.scan_interval).await;
            }
        });
        
        Ok(())
    }
    
    /// Stop monitoring
    pub async fn stop_monitoring(&self) {
        *self.is_monitoring.write().await = false;
        debug!("Runtime monitoring stopped");
    }
    
    /// Register a new connection for monitoring
    pub async fn register_connection(&self, info: ConnectionInfo) -> Result<()> {
        debug!("Registering connection for monitoring: {}", info.connection_id);
        
        // Validate connection before registering
        let validation_status = self.validate_new_connection(&info).await?;
        
        let mut updated_info = info;
        updated_info.validation_status = validation_status;
        
        self.active_connections.write().await.insert(
            updated_info.connection_id.clone(),
            updated_info,
        );
        
        Ok(())
    }
    
    /// Unregister a connection
    pub async fn unregister_connection(&self, connection_id: &str) {
        debug!("Unregistering connection: {}", connection_id);
        self.active_connections.write().await.remove(connection_id);
    }
    
    /// Update connection activity
    pub async fn update_connection_activity(&self, connection_id: &str) {
        if let Some(conn) = self.active_connections.write().await.get_mut(connection_id) {
            conn.last_activity = chrono::Utc::now();
        }
    }
    
    /// Validate runtime patterns for a data source
    pub async fn validate_runtime_patterns<T>(&self, source: &T) -> Result<(), ValidationError>
    where
        T: DataSource + Send + Sync,
    {
        // Check if source shows signs of being synthetic
        if source.contains_synthetic_patterns().await? {
            let violation = RuntimeViolation {
                violation_type: ViolationType::MockDataSource,
                source: source.get_source_name(),
                endpoint: source.get_endpoint(),
                timestamp: chrono::Utc::now(),
                severity: ViolationSeverity::Critical,
            };
            
            self.violation_history.write().await.push(violation);
            
            return Err(ValidationError::SyntheticDataDetected(
                source.get_source_name()
            ));
        }
        
        // Check data freshness
        let age = source.last_update_age();
        if age > Duration::from_secs(300) {
            let violation = RuntimeViolation {
                violation_type: ViolationType::MockDataSource,
                source: source.get_source_name(),
                endpoint: source.get_endpoint(),
                timestamp: chrono::Utc::now(),
                severity: ViolationSeverity::High,
            };
            
            self.violation_history.write().await.push(violation);
            
            return Err(ValidationError::StaleData(source.get_source_name()));
        }
        
        Ok(())
    }
    
    /// Scan all active connections for violations
    pub async fn scan_active_connections(&self) -> Result<Vec<RuntimeViolation>> {
        debug!("Scanning active connections for violations");
        
        let connections = self.active_connections.read().await;
        let mut violations = Vec::new();
        
        for (_, conn) in connections.iter() {
            // Check for suspicious patterns in endpoint
            if self.is_suspicious_endpoint(&conn.endpoint) {
                violations.push(RuntimeViolation {
                    violation_type: ViolationType::SyntheticEndpoint,
                    source: conn.connection_id.clone(),
                    endpoint: conn.endpoint.clone(),
                    timestamp: chrono::Utc::now(),
                    severity: ViolationSeverity::Critical,
                });
            }
            
            // Check for stale connections
            let age = chrono::Utc::now() - conn.last_activity;
            if age > chrono::Duration::from_std(self.monitoring_config.connection_timeout).unwrap() {
                violations.push(RuntimeViolation {
                    violation_type: ViolationType::MockDataSource,
                    source: conn.connection_id.clone(),
                    endpoint: conn.endpoint.clone(),
                    timestamp: chrono::Utc::now(),
                    severity: ViolationSeverity::Medium,
                });
            }
        }
        
        Ok(violations)
    }
    
    /// Get violation history
    pub async fn get_violation_history(&self) -> Vec<RuntimeViolation> {
        self.violation_history.read().await.clone()
    }
    
    /// Get active connections
    pub async fn get_active_connections(&self) -> HashMap<String, ConnectionInfo> {
        self.active_connections.read().await.clone()
    }
    
    async fn validate_new_connection(&self, info: &ConnectionInfo) -> Result<ValidationStatus> {
        // Check endpoint against known patterns
        if self.is_suspicious_endpoint(&info.endpoint) {
            warn!("Suspicious endpoint detected: {}", info.endpoint);
            return Ok(ValidationStatus::Suspicious);
        }
        
        // Check data source type
        if info.data_source_type.contains("mock") || 
           info.data_source_type.contains("test") ||
           info.data_source_type.contains("fake") {
            error!("Mock data source type detected: {}", info.data_source_type);
            return Ok(ValidationStatus::Blocked);
        }
        
        Ok(ValidationStatus::Valid)
    }
    
    fn is_suspicious_endpoint(&self, endpoint: &str) -> bool {
        let suspicious_patterns = [
            "localhost", "127.0.0.1", "0.0.0.0",
            "mock", "fake", "test", "demo",
            "synthetic", "dummy", "example"
        ];
        
        for pattern in &suspicious_patterns {
            if endpoint.to_lowercase().contains(pattern) {
                return true;
            }
        }
        
        false
    }
    
    async fn scan_connections(
        connections: &Arc<RwLock<HashMap<String, ConnectionInfo>>>
    ) -> Vec<RuntimeViolation> {
        let connections = connections.read().await;
        let mut violations = Vec::new();
        
        for (_, conn) in connections.iter() {
            // Check for mock patterns in connection
            if conn.endpoint.contains("mock") || 
               conn.endpoint.contains("test") ||
               conn.endpoint.contains("fake") {
                violations.push(RuntimeViolation {
                    violation_type: ViolationType::SyntheticEndpoint,
                    source: conn.connection_id.clone(),
                    endpoint: conn.endpoint.clone(),
                    timestamp: chrono::Utc::now(),
                    severity: ViolationSeverity::Critical,
                });
            }
        }
        
        violations
    }
    
    async fn process_violation(
        violation_history: &Arc<RwLock<Vec<RuntimeViolation>>>,
        violation: RuntimeViolation,
        config: &MonitoringConfig,
    ) {
        // Log violation
        match violation.severity {
            ViolationSeverity::Critical => {
                error!("🚫 CRITICAL VIOLATION: {} at {}", violation.source, violation.endpoint);
            },
            ViolationSeverity::High => {
                warn!("⚠️ HIGH SEVERITY VIOLATION: {} at {}", violation.source, violation.endpoint);
            },
            ViolationSeverity::Medium => {
                if config.log_all_violations {
                    debug!("Medium violation: {} at {}", violation.source, violation.endpoint);
                }
            },
        }
        
        // Store violation
        violation_history.write().await.push(violation);
        
        // Auto-block if configured
        if config.auto_block_critical && 
           matches!(violation.severity, ViolationSeverity::Critical) {
            error!("🚫 AUTO-BLOCKING connection due to critical violation");
            // In a real implementation, this would terminate the connection
        }
    }
    
    async fn cleanup_stale_connections(
        connections: &Arc<RwLock<HashMap<String, ConnectionInfo>>>,
        config: &MonitoringConfig,
    ) {
        let mut connections = connections.write().await;
        let now = chrono::Utc::now();
        
        connections.retain(|_, conn| {
            let age = now - conn.last_activity;
            age < chrono::Duration::from_std(config.connection_timeout).unwrap()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_runtime_monitor_creation() {
        let monitor = RuntimeMonitor::new();
        assert!(!*monitor.is_monitoring.read().await);
    }
    
    #[tokio::test]
    async fn test_connection_registration() {
        let monitor = RuntimeMonitor::new();
        
        let conn_info = ConnectionInfo {
            connection_id: "test_conn_1".to_string(),
            connection_type: "websocket".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            established_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            data_source_type: "binance".to_string(),
            validation_status: ValidationStatus::Unknown,
        };
        
        assert!(monitor.register_connection(conn_info).await.is_ok());
        
        let connections = monitor.get_active_connections().await;
        assert!(connections.contains_key("test_conn_1"));
    }
    
    #[tokio::test]
    async fn test_suspicious_endpoint_detection() {
        let monitor = RuntimeMonitor::new();
        
        // Test legitimate endpoint
        assert!(!monitor.is_suspicious_endpoint("https://api.binance.com"));
        
        // Test suspicious endpoints
        assert!(monitor.is_suspicious_endpoint("http://localhost:8080"));
        assert!(monitor.is_suspicious_endpoint("https://mock-api.com"));
        assert!(monitor.is_suspicious_endpoint("https://test-endpoint.com"));
    }
    
    #[tokio::test]
    async fn test_violation_scanning() {
        let monitor = RuntimeMonitor::new();
        
        // Register a suspicious connection
        let suspicious_conn = ConnectionInfo {
            connection_id: "suspicious_conn".to_string(),
            connection_type: "rest".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            established_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            data_source_type: "mock_exchange".to_string(),
            validation_status: ValidationStatus::Unknown,
        };
        
        monitor.register_connection(suspicious_conn).await.unwrap();
        
        // Scan for violations
        let violations = monitor.scan_active_connections().await.unwrap();
        assert!(!violations.is_empty());
        
        // Check that violation was detected
        assert!(violations.iter().any(|v| v.violation_type == ViolationType::SyntheticEndpoint));
    }
    
    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert_eq!(config.scan_interval, Duration::from_secs(30));
        assert_eq!(config.connection_timeout, Duration::from_secs(300));
        assert!(config.auto_block_critical);
    }
}