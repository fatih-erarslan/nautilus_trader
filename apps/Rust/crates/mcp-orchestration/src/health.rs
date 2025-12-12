//! Health monitoring and system status checking for MCP orchestration.

use crate::agent::{AgentInfo, AgentRegistry};
use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, AgentType, HealthStatus, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::{interval, Instant};
use tracing::{debug, error, info, warn};

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Component being checked
    pub component: String,
    /// Health status
    pub status: HealthStatus,
    /// Response time in milliseconds
    pub response_time: f64,
    /// Error message if unhealthy
    pub error_message: Option<String>,
    /// Check timestamp
    pub timestamp: Timestamp,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl HealthCheckResult {
    /// Create a healthy result
    pub fn healthy(component: String, response_time: f64) -> Self {
        Self {
            component,
            status: HealthStatus::Healthy,
            response_time,
            error_message: None,
            timestamp: Timestamp::now(),
            metrics: HashMap::new(),
        }
    }
    
    /// Create an unhealthy result
    pub fn unhealthy(component: String, response_time: f64, error: String) -> Self {
        Self {
            component,
            status: HealthStatus::Unhealthy,
            response_time,
            error_message: Some(error),
            timestamp: Timestamp::now(),
            metrics: HashMap::new(),
        }
    }
    
    /// Create a degraded result
    pub fn degraded(component: String, response_time: f64, reason: String) -> Self {
        Self {
            component,
            status: HealthStatus::Degraded,
            response_time,
            error_message: Some(reason),
            timestamp: Timestamp::now(),
            metrics: HashMap::new(),
        }
    }
    
    /// Add a metric to the result
    pub fn with_metric<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<f64>,
    {
        self.metrics.insert(key.into(), value.into());
        self
    }
}

/// Health check trait for components
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// Perform health check
    async fn check_health(&self) -> Result<HealthCheckResult>;
    
    /// Get component name
    fn component_name(&self) -> &str;
    
    /// Get check interval
    fn check_interval(&self) -> Duration {
        Duration::from_secs(30)
    }
    
    /// Get timeout for health checks
    fn check_timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    /// Overall system status
    pub overall_status: HealthStatus,
    /// Component health statuses
    pub component_statuses: HashMap<String, HealthCheckResult>,
    /// System uptime in seconds
    pub uptime: u64,
    /// Total number of health checks performed
    pub total_checks: u64,
    /// Number of failed checks
    pub failed_checks: u64,
    /// Average response time across all components
    pub avg_response_time: f64,
    /// System metrics
    pub system_metrics: SystemMetrics,
    /// Last update timestamp
    pub last_updated: Timestamp,
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Disk usage percentage
    pub disk_usage: f64,
    /// Network connections
    pub network_connections: u64,
    /// Active threads
    pub active_threads: u64,
    /// Open file descriptors
    pub open_files: u64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            available_memory: 0,
            disk_usage: 0.0,
            network_connections: 0,
            active_threads: 0,
            open_files: 0,
        }
    }
}

/// Health event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthEvent {
    /// Component became healthy
    ComponentHealthy {
        component: String,
        timestamp: Timestamp,
    },
    /// Component became unhealthy
    ComponentUnhealthy {
        component: String,
        error: String,
        timestamp: Timestamp,
    },
    /// Component became degraded
    ComponentDegraded {
        component: String,
        reason: String,
        timestamp: Timestamp,
    },
    /// System status changed
    SystemStatusChanged {
        old_status: HealthStatus,
        new_status: HealthStatus,
        timestamp: Timestamp,
    },
    /// Health check failed
    HealthCheckFailed {
        component: String,
        error: String,
        timestamp: Timestamp,
    },
}

/// Health monitor for system-wide health tracking
#[derive(Debug)]
pub struct HealthMonitor {
    /// Registered health checkers
    checkers: Arc<DashMap<String, Arc<dyn HealthChecker>>>,
    /// Agent registry for agent health
    agent_registry: Arc<AgentRegistry>,
    /// Health check results
    results: Arc<DashMap<String, HealthCheckResult>>,
    /// System health status
    system_status: Arc<RwLock<SystemHealthStatus>>,
    /// Health event broadcaster
    event_broadcaster: broadcast::Sender<HealthEvent>,
    /// Health check counters
    check_counter: Arc<AtomicU64>,
    failure_counter: Arc<AtomicU64>,
    /// System start time
    start_time: Instant,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(agent_registry: Arc<AgentRegistry>) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);
        
        Self {
            checkers: Arc::new(DashMap::new()),
            agent_registry,
            results: Arc::new(DashMap::new()),
            system_status: Arc::new(RwLock::new(SystemHealthStatus {
                overall_status: HealthStatus::Unknown,
                component_statuses: HashMap::new(),
                uptime: 0,
                total_checks: 0,
                failed_checks: 0,
                avg_response_time: 0.0,
                system_metrics: SystemMetrics::default(),
                last_updated: Timestamp::now(),
            })),
            event_broadcaster,
            check_counter: Arc::new(AtomicU64::new(0)),
            failure_counter: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }
    
    /// Start the health monitor
    pub async fn start(&self) -> Result<()> {
        // Start health checking loop
        self.start_health_checking().await?;
        
        // Start system metrics collection
        self.start_system_metrics_collection().await?;
        
        // Start status aggregation
        self.start_status_aggregation().await?;
        
        info!("Health monitor started successfully");
        Ok(())
    }
    
    /// Register a health checker
    pub async fn register_checker(&self, checker: Arc<dyn HealthChecker>) -> Result<()> {
        let component_name = checker.component_name().to_string();
        self.checkers.insert(component_name.clone(), checker);
        
        debug!("Health checker registered for component: {}", component_name);
        Ok(())
    }
    
    /// Unregister a health checker
    pub async fn unregister_checker(&self, component_name: &str) -> Result<()> {
        self.checkers.remove(component_name);
        self.results.remove(component_name);
        
        debug!("Health checker unregistered for component: {}", component_name);
        Ok(())
    }
    
    /// Get system health status
    pub async fn get_system_status(&self) -> Result<SystemHealthStatus> {
        Ok(self.system_status.read().clone())
    }
    
    /// Get component health result
    pub async fn get_component_health(&self, component: &str) -> Result<HealthCheckResult> {
        self.results.get(component)
            .map(|result| result.clone())
            .ok_or_else(|| OrchestrationError::not_found(format!("Component {}", component)))
    }
    
    /// Get all health results
    pub async fn get_all_health_results(&self) -> Result<HashMap<String, HealthCheckResult>> {
        let mut results = HashMap::new();
        
        for entry in self.results.iter() {
            results.insert(entry.key().clone(), entry.value().clone());
        }
        
        Ok(results)
    }
    
    /// Subscribe to health events
    pub async fn subscribe_events(&self) -> Result<broadcast::Receiver<HealthEvent>> {
        Ok(self.event_broadcaster.subscribe())
    }
    
    /// Start health checking loop
    async fn start_health_checking(&self) -> Result<()> {
        let checkers = Arc::clone(&self.checkers);
        let results = Arc::clone(&self.results);
        let event_broadcaster = self.event_broadcaster.clone();
        let check_counter = Arc::clone(&self.check_counter);
        let failure_counter = Arc::clone(&self.failure_counter);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Check all registered components
                for entry in checkers.iter() {
                    let component_name = entry.key().clone();
                    let checker = entry.value().clone();
                    
                    let check_counter = Arc::clone(&check_counter);
                    let failure_counter = Arc::clone(&failure_counter);
                    let results = Arc::clone(&results);
                    let event_broadcaster = event_broadcaster.clone();
                    
                    tokio::spawn(async move {
                        check_counter.fetch_add(1, Ordering::Relaxed);
                        
                        let start_time = Instant::now();
                        let result = match tokio::time::timeout(
                            checker.check_timeout(),
                            checker.check_health()
                        ).await {
                            Ok(Ok(result)) => result,
                            Ok(Err(e)) => {
                                failure_counter.fetch_add(1, Ordering::Relaxed);
                                HealthCheckResult::unhealthy(
                                    component_name.clone(),
                                    start_time.elapsed().as_millis() as f64,
                                    e.to_string(),
                                )
                            }
                            Err(_) => {
                                failure_counter.fetch_add(1, Ordering::Relaxed);
                                HealthCheckResult::unhealthy(
                                    component_name.clone(),
                                    checker.check_timeout().as_millis() as f64,
                                    "Health check timed out".to_string(),
                                )
                            }
                        };
                        
                        // Check for status changes
                        if let Some(previous_result) = results.get(&component_name) {
                            if previous_result.status != result.status {
                                let event = match result.status {
                                    HealthStatus::Healthy => HealthEvent::ComponentHealthy {
                                        component: component_name.clone(),
                                        timestamp: result.timestamp,
                                    },
                                    HealthStatus::Unhealthy => HealthEvent::ComponentUnhealthy {
                                        component: component_name.clone(),
                                        error: result.error_message.clone().unwrap_or_default(),
                                        timestamp: result.timestamp,
                                    },
                                    HealthStatus::Degraded => HealthEvent::ComponentDegraded {
                                        component: component_name.clone(),
                                        reason: result.error_message.clone().unwrap_or_default(),
                                        timestamp: result.timestamp,
                                    },
                                    HealthStatus::Unknown => HealthEvent::HealthCheckFailed {
                                        component: component_name.clone(),
                                        error: "Unknown status".to_string(),
                                        timestamp: result.timestamp,
                                    },
                                };
                                
                                let _ = event_broadcaster.send(event);
                            }
                        }
                        
                        // Store result
                        results.insert(component_name, result);
                    });
                }
            }
        });
        
        Ok(())
    }
    
    /// Start system metrics collection
    async fn start_system_metrics_collection(&self) -> Result<()> {
        let system_status = Arc::clone(&self.system_status);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                // Collect system metrics (simplified - in practice would use system APIs)
                let metrics = SystemMetrics {
                    cpu_usage: Self::get_cpu_usage(),
                    memory_usage: Self::get_memory_usage(),
                    available_memory: Self::get_available_memory(),
                    disk_usage: Self::get_disk_usage(),
                    network_connections: Self::get_network_connections(),
                    active_threads: Self::get_active_threads(),
                    open_files: Self::get_open_files(),
                };
                
                // Update system status
                let mut status = system_status.write();
                status.system_metrics = metrics;
                status.last_updated = Timestamp::now();
            }
        });
        
        Ok(())
    }
    
    /// Start status aggregation
    async fn start_status_aggregation(&self) -> Result<()> {
        let system_status = Arc::clone(&self.system_status);
        let results = Arc::clone(&self.results);
        let agent_registry = Arc::clone(&self.agent_registry);
        let check_counter = Arc::clone(&self.check_counter);
        let failure_counter = Arc::clone(&self.failure_counter);
        let start_time = self.start_time;
        let event_broadcaster = self.event_broadcaster.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(15));
            
            loop {
                interval.tick().await;
                
                let mut component_statuses = HashMap::new();
                let mut healthy_count = 0;
                let mut unhealthy_count = 0;
                let mut degraded_count = 0;
                let mut total_response_time = 0.0;
                let mut component_count = 0;
                
                // Aggregate component health
                for entry in results.iter() {
                    let result = entry.value().clone();
                    component_statuses.insert(entry.key().clone(), result.clone());
                    
                    match result.status {
                        HealthStatus::Healthy => healthy_count += 1,
                        HealthStatus::Unhealthy => unhealthy_count += 1,
                        HealthStatus::Degraded => degraded_count += 1,
                        HealthStatus::Unknown => {}
                    }
                    
                    total_response_time += result.response_time;
                    component_count += 1;
                }
                
                // Check agent health
                if let Ok(agents) = agent_registry.get_all_agents().await {
                    for agent in agents {
                        let agent_result = HealthCheckResult {
                            component: format!("Agent-{}", agent.agent_type),
                            status: agent.health_status,
                            response_time: 0.0, // Would measure actual response time
                            error_message: None,
                            timestamp: Timestamp::now(),
                            metrics: HashMap::new(),
                        };
                        
                        component_statuses.insert(
                            format!("agent-{}", agent.id),
                            agent_result.clone(),
                        );
                        
                        match agent.health_status {
                            HealthStatus::Healthy => healthy_count += 1,
                            HealthStatus::Unhealthy => unhealthy_count += 1,
                            HealthStatus::Degraded => degraded_count += 1,
                            HealthStatus::Unknown => {}
                        }
                        
                        component_count += 1;
                    }
                }
                
                // Determine overall status
                let overall_status = if unhealthy_count > 0 {
                    HealthStatus::Unhealthy
                } else if degraded_count > 0 {
                    HealthStatus::Degraded
                } else if healthy_count > 0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unknown
                };
                
                // Calculate averages
                let avg_response_time = if component_count > 0 {
                    total_response_time / component_count as f64
                } else {
                    0.0
                };
                
                // Update system status
                let mut status = system_status.write();
                let old_status = status.overall_status;
                
                status.overall_status = overall_status;
                status.component_statuses = component_statuses;
                status.uptime = start_time.elapsed().as_secs();
                status.total_checks = check_counter.load(Ordering::Relaxed);
                status.failed_checks = failure_counter.load(Ordering::Relaxed);
                status.avg_response_time = avg_response_time;
                status.last_updated = Timestamp::now();
                
                // Broadcast system status change event
                if old_status != overall_status {
                    let event = HealthEvent::SystemStatusChanged {
                        old_status,
                        new_status: overall_status,
                        timestamp: Timestamp::now(),
                    };
                    
                    let _ = event_broadcaster.send(event);
                }
            }
        });
        
        Ok(())
    }
    
    /// Get CPU usage (simplified implementation)
    fn get_cpu_usage() -> f64 {
        // In a real implementation, this would read from /proc/stat or use system APIs
        rand::random::<f64>() * 100.0
    }
    
    /// Get memory usage (simplified implementation)
    fn get_memory_usage() -> u64 {
        // In a real implementation, this would read from /proc/meminfo or use system APIs
        rand::random::<u64>() % (8 * 1024 * 1024 * 1024) // Up to 8GB
    }
    
    /// Get available memory (simplified implementation)
    fn get_available_memory() -> u64 {
        // In a real implementation, this would read from /proc/meminfo or use system APIs
        8 * 1024 * 1024 * 1024 - Self::get_memory_usage() // 8GB total
    }
    
    /// Get disk usage (simplified implementation)
    fn get_disk_usage() -> f64 {
        // In a real implementation, this would use statvfs or similar
        rand::random::<f64>() * 100.0
    }
    
    /// Get network connections (simplified implementation)
    fn get_network_connections() -> u64 {
        // In a real implementation, this would read from /proc/net/tcp or use system APIs
        rand::random::<u64>() % 1000
    }
    
    /// Get active threads (simplified implementation)
    fn get_active_threads() -> u64 {
        // In a real implementation, this would read from /proc/self/status or use system APIs
        rand::random::<u64>() % 100 + 10
    }
    
    /// Get open files (simplified implementation)
    fn get_open_files() -> u64 {
        // In a real implementation, this would read from /proc/self/fd or use system APIs
        rand::random::<u64>() % 1000 + 100
    }
}

/// Basic health checker implementation for testing
#[derive(Debug)]
pub struct BasicHealthChecker {
    component_name: String,
    is_healthy: Arc<AtomicU64>, // 0 = unhealthy, 1 = healthy, 2 = degraded
}

impl BasicHealthChecker {
    /// Create a new basic health checker
    pub fn new(component_name: String) -> Self {
        Self {
            component_name,
            is_healthy: Arc::new(AtomicU64::new(1)), // Start healthy
        }
    }
    
    /// Set health status
    pub fn set_health_status(&self, status: HealthStatus) {
        let value = match status {
            HealthStatus::Healthy => 1,
            HealthStatus::Unhealthy => 0,
            HealthStatus::Degraded => 2,
            HealthStatus::Unknown => 3,
        };
        self.is_healthy.store(value, Ordering::Relaxed);
    }
}

#[async_trait]
impl HealthChecker for BasicHealthChecker {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start_time = Instant::now();
        
        // Simulate some work
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let response_time = start_time.elapsed().as_millis() as f64;
        let status_value = self.is_healthy.load(Ordering::Relaxed);
        
        let result = match status_value {
            1 => HealthCheckResult::healthy(self.component_name.clone(), response_time),
            0 => HealthCheckResult::unhealthy(
                self.component_name.clone(),
                response_time,
                "Component is unhealthy".to_string(),
            ),
            2 => HealthCheckResult::degraded(
                self.component_name.clone(),
                response_time,
                "Component is degraded".to_string(),
            ),
            _ => HealthCheckResult::unhealthy(
                self.component_name.clone(),
                response_time,
                "Unknown status".to_string(),
            ),
        };
        
        Ok(result)
    }
    
    fn component_name(&self) -> &str {
        &self.component_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication::MessageRouter;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_health_check_result() {
        let result = HealthCheckResult::healthy("test_component".to_string(), 100.0)
            .with_metric("cpu_usage", 50.0)
            .with_metric("memory_usage", 1024.0);
        
        assert_eq!(result.component, "test_component");
        assert_eq!(result.status, HealthStatus::Healthy);
        assert_eq!(result.response_time, 100.0);
        assert_eq!(result.metrics.get("cpu_usage"), Some(&50.0));
        assert_eq!(result.metrics.get("memory_usage"), Some(&1024.0));
    }
    
    #[tokio::test]
    async fn test_basic_health_checker() {
        let checker = BasicHealthChecker::new("test_component".to_string());
        
        // Test healthy status
        let result = checker.check_health().await.unwrap();
        assert_eq!(result.status, HealthStatus::Healthy);
        
        // Test unhealthy status
        checker.set_health_status(HealthStatus::Unhealthy);
        let result = checker.check_health().await.unwrap();
        assert_eq!(result.status, HealthStatus::Unhealthy);
        
        // Test degraded status
        checker.set_health_status(HealthStatus::Degraded);
        let result = checker.check_health().await.unwrap();
        assert_eq!(result.status, HealthStatus::Degraded);
    }
    
    #[tokio::test]
    async fn test_health_monitor() {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        let health_monitor = HealthMonitor::new(agent_registry);
        
        health_monitor.start().await.unwrap();
        
        // Register a health checker
        let checker = Arc::new(BasicHealthChecker::new("test_component".to_string()));
        health_monitor.register_checker(checker.clone()).await.unwrap();
        
        // Wait for health check to run
        sleep(Duration::from_millis(500)).await;
        
        // Get component health
        let health_result = health_monitor.get_component_health("test_component").await.unwrap();
        assert_eq!(health_result.component, "test_component");
        assert_eq!(health_result.status, HealthStatus::Healthy);
        
        // Get system status
        let system_status = health_monitor.get_system_status().await.unwrap();
        assert!(system_status.component_statuses.contains_key("test_component"));
        
        // Unregister checker
        health_monitor.unregister_checker("test_component").await.unwrap();
        
        let result = health_monitor.get_component_health("test_component").await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_health_events() {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        let health_monitor = HealthMonitor::new(agent_registry);
        
        health_monitor.start().await.unwrap();
        
        let mut event_receiver = health_monitor.subscribe_events().await.unwrap();
        
        // Register a health checker
        let checker = Arc::new(BasicHealthChecker::new("test_component".to_string()));
        health_monitor.register_checker(checker.clone()).await.unwrap();
        
        // Wait for initial health check
        sleep(Duration::from_millis(500)).await;
        
        // Change health status to unhealthy
        checker.set_health_status(HealthStatus::Unhealthy);
        
        // Wait for health check to detect change
        sleep(Duration::from_millis(500)).await;
        
        // Check for health event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            HealthEvent::ComponentUnhealthy { component, .. } => {
                assert_eq!(component, "test_component");
            }
            _ => panic!("Expected ComponentUnhealthy event"),
        }
    }
    
    #[tokio::test]
    async fn test_system_health_aggregation() {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        let health_monitor = HealthMonitor::new(agent_registry);
        
        health_monitor.start().await.unwrap();
        
        // Register multiple health checkers
        let checker1 = Arc::new(BasicHealthChecker::new("component1".to_string()));
        let checker2 = Arc::new(BasicHealthChecker::new("component2".to_string()));
        let checker3 = Arc::new(BasicHealthChecker::new("component3".to_string()));
        
        health_monitor.register_checker(checker1.clone()).await.unwrap();
        health_monitor.register_checker(checker2.clone()).await.unwrap();
        health_monitor.register_checker(checker3.clone()).await.unwrap();
        
        // Set different health statuses
        checker1.set_health_status(HealthStatus::Healthy);
        checker2.set_health_status(HealthStatus::Degraded);
        checker3.set_health_status(HealthStatus::Unhealthy);
        
        // Wait for health checks and aggregation
        sleep(Duration::from_millis(1000)).await;
        
        let system_status = health_monitor.get_system_status().await.unwrap();
        
        // System should be unhealthy due to one unhealthy component
        assert_eq!(system_status.overall_status, HealthStatus::Unhealthy);
        assert_eq!(system_status.component_statuses.len(), 3);
        
        // Fix the unhealthy component
        checker3.set_health_status(HealthStatus::Healthy);
        
        // Wait for health checks and aggregation
        sleep(Duration::from_millis(1000)).await;
        
        let system_status = health_monitor.get_system_status().await.unwrap();
        
        // System should now be degraded due to one degraded component
        assert_eq!(system_status.overall_status, HealthStatus::Degraded);
    }
}