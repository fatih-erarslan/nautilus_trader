//! Fault tolerance and recovery mechanisms for MCP orchestration.

use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, RecoveryStrategy, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::{interval, sleep, Instant};
use tracing::{debug, error, info, warn};

/// Recovery action result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    /// Recovery action ID
    pub action_id: String,
    /// Component that was recovered
    pub component: String,
    /// Recovery strategy used
    pub strategy: RecoveryStrategy,
    /// Success status
    pub success: bool,
    /// Error message if recovery failed
    pub error_message: Option<String>,
    /// Recovery duration in milliseconds
    pub duration_ms: u64,
    /// Timestamp of recovery action
    pub timestamp: Timestamp,
}

/// Recovery event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryEvent {
    /// Recovery action started
    RecoveryStarted {
        component: String,
        strategy: RecoveryStrategy,
        timestamp: Timestamp,
    },
    /// Recovery action completed successfully
    RecoverySucceeded {
        component: String,
        strategy: RecoveryStrategy,
        duration_ms: u64,
        timestamp: Timestamp,
    },
    /// Recovery action failed
    RecoveryFailed {
        component: String,
        strategy: RecoveryStrategy,
        error: String,
        timestamp: Timestamp,
    },
    /// Circuit breaker opened
    CircuitBreakerOpened {
        component: String,
        failure_count: u32,
        timestamp: Timestamp,
    },
    /// Circuit breaker closed
    CircuitBreakerClosed {
        component: String,
        timestamp: Timestamp,
    },
    /// Graceful degradation activated
    GracefulDegradationActivated {
        component: String,
        reason: String,
        timestamp: Timestamp,
    },
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

/// Circuit breaker for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    /// Component name
    pub component: String,
    /// Current state
    pub state: CircuitBreakerState,
    /// Failure count
    pub failure_count: u32,
    /// Failure threshold
    pub failure_threshold: u32,
    /// Timeout duration in milliseconds
    pub timeout_ms: u64,
    /// Last failure timestamp
    pub last_failure: Option<Timestamp>,
    /// Recovery timestamp when half-open
    pub recovery_timestamp: Option<Timestamp>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(component: String, failure_threshold: u32, timeout_ms: u64) -> Self {
        Self {
            component,
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            failure_threshold,
            timeout_ms,
            last_failure: None,
            recovery_timestamp: None,
        }
    }
    
    /// Record a success
    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                // Success in half-open state closes the circuit
                self.state = CircuitBreakerState::Closed;
                self.failure_count = 0;
                self.recovery_timestamp = None;
            }
            CircuitBreakerState::Closed => {
                // Reset failure count on success
                self.failure_count = 0;
            }
            CircuitBreakerState::Open => {
                // Ignore success in open state
            }
        }
    }
    
    /// Record a failure
    pub fn record_failure(&mut self) -> CircuitBreakerState {
        self.failure_count += 1;
        self.last_failure = Some(Timestamp::now());
        
        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Failure in half-open state reopens the circuit
                self.state = CircuitBreakerState::Open;
                self.recovery_timestamp = None;
            }
            CircuitBreakerState::Open => {
                // Already open, nothing to do
            }
        }
        
        self.state
    }
    
    /// Check if the circuit breaker allows requests
    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = self.last_failure {
                    let elapsed = last_failure.elapsed().as_millis() as u64;
                    if elapsed >= self.timeout_ms {
                        // Transition to half-open
                        self.state = CircuitBreakerState::HalfOpen;
                        self.recovery_timestamp = Some(Timestamp::now());
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }
}

/// Recovery manager trait
#[async_trait]
pub trait RecoveryManager: Send + Sync {
    /// Attempt to recover a failed component
    async fn recover_component(
        &self,
        component: String,
        strategy: RecoveryStrategy,
    ) -> Result<RecoveryResult>;
    
    /// Check circuit breaker status
    async fn check_circuit_breaker(&self, component: &str) -> Result<CircuitBreakerState>;
    
    /// Record success for circuit breaker
    async fn record_success(&self, component: &str) -> Result<()>;
    
    /// Record failure for circuit breaker
    async fn record_failure(&self, component: &str) -> Result<CircuitBreakerState>;
    
    /// Enable graceful degradation for a component
    async fn enable_graceful_degradation(&self, component: String, reason: String) -> Result<()>;
    
    /// Disable graceful degradation for a component
    async fn disable_graceful_degradation(&self, component: String) -> Result<()>;
    
    /// Get recovery statistics
    async fn get_recovery_stats(&self) -> Result<RecoveryStatistics>;
    
    /// Subscribe to recovery events
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<RecoveryEvent>>;
}

/// Recovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStatistics {
    /// Total recovery attempts
    pub total_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Failed recoveries
    pub failed_recoveries: u64,
    /// Recovery attempts by strategy
    pub attempts_by_strategy: HashMap<RecoveryStrategy, u64>,
    /// Recovery success rate by strategy
    pub success_rate_by_strategy: HashMap<RecoveryStrategy, f64>,
    /// Average recovery time by strategy
    pub avg_recovery_time_by_strategy: HashMap<RecoveryStrategy, f64>,
    /// Circuit breaker status by component
    pub circuit_breaker_status: HashMap<String, CircuitBreakerState>,
    /// Components under graceful degradation
    pub degraded_components: Vec<String>,
}

/// Adaptive recovery manager implementation
#[derive(Debug)]
pub struct AdaptiveRecoveryManager {
    /// Circuit breakers by component
    circuit_breakers: Arc<DashMap<String, CircuitBreaker>>,
    /// Components under graceful degradation
    degraded_components: Arc<DashMap<String, String>>, // component -> reason
    /// Recovery history
    recovery_history: Arc<RwLock<VecDeque<RecoveryResult>>>,
    /// Recovery event broadcaster
    event_broadcaster: broadcast::Sender<RecoveryEvent>,
    /// Recovery counters
    total_attempts: Arc<AtomicU64>,
    successful_recoveries: Arc<AtomicU64>,
    failed_recoveries: Arc<AtomicU64>,
    /// Default configuration
    default_failure_threshold: u32,
    default_timeout_ms: u64,
    max_history_size: usize,
}

impl AdaptiveRecoveryManager {
    /// Create a new adaptive recovery manager
    pub fn new(
        failure_threshold: u32,
        timeout_ms: u64,
        max_history_size: usize,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);
        
        Self {
            circuit_breakers: Arc::new(DashMap::new()),
            degraded_components: Arc::new(DashMap::new()),
            recovery_history: Arc::new(RwLock::new(VecDeque::new())),
            event_broadcaster,
            total_attempts: Arc::new(AtomicU64::new(0)),
            successful_recoveries: Arc::new(AtomicU64::new(0)),
            failed_recoveries: Arc::new(AtomicU64::new(0)),
            default_failure_threshold: failure_threshold,
            default_timeout_ms: timeout_ms,
            max_history_size,
        }
    }
    
    /// Start the recovery manager
    pub async fn start(&self) -> Result<()> {
        // Start circuit breaker monitoring
        self.start_circuit_breaker_monitoring().await?;
        
        // Start recovery history cleanup
        self.start_history_cleanup().await?;
        
        info!("Adaptive recovery manager started successfully");
        Ok(())
    }
    
    /// Get or create circuit breaker for component
    fn get_or_create_circuit_breaker(&self, component: &str) -> CircuitBreaker {
        self.circuit_breakers
            .entry(component.to_string())
            .or_insert_with(|| CircuitBreaker::new(
                component.to_string(),
                self.default_failure_threshold,
                self.default_timeout_ms,
            ))
            .clone()
    }
    
    /// Perform restart recovery
    async fn perform_restart(&self, component: &str) -> Result<RecoveryResult> {
        let start_time = Instant::now();
        
        // Simulate restart process
        info!("Restarting component: {}", component);
        
        // In a real implementation, this would:
        // 1. Stop the component gracefully
        // 2. Clean up resources
        // 3. Start the component again
        // 4. Wait for health check
        
        // Simulate restart time
        sleep(Duration::from_millis(1000)).await;
        
        let duration = start_time.elapsed();
        let success = rand::random::<f64>() > 0.2; // 80% success rate
        
        Ok(RecoveryResult {
            action_id: uuid::Uuid::new_v4().to_string(),
            component: component.to_string(),
            strategy: RecoveryStrategy::Restart,
            success,
            error_message: if !success {
                Some("Restart failed".to_string())
            } else {
                None
            },
            duration_ms: duration.as_millis() as u64,
            timestamp: Timestamp::now(),
        })
    }
    
    /// Perform failover recovery
    async fn perform_failover(&self, component: &str) -> Result<RecoveryResult> {
        let start_time = Instant::now();
        
        info!("Performing failover for component: {}", component);
        
        // In a real implementation, this would:
        // 1. Identify backup/standby instance
        // 2. Switch traffic to backup
        // 3. Update load balancer configuration
        // 4. Verify backup is healthy
        
        sleep(Duration::from_millis(500)).await;
        
        let duration = start_time.elapsed();
        let success = rand::random::<f64>() > 0.1; // 90% success rate
        
        Ok(RecoveryResult {
            action_id: uuid::Uuid::new_v4().to_string(),
            component: component.to_string(),
            strategy: RecoveryStrategy::Failover,
            success,
            error_message: if !success {
                Some("Failover failed".to_string())
            } else {
                None
            },
            duration_ms: duration.as_millis() as u64,
            timestamp: Timestamp::now(),
        })
    }
    
    /// Perform retry recovery
    async fn perform_retry(&self, component: &str) -> Result<RecoveryResult> {
        let start_time = Instant::now();
        
        info!("Retrying component: {}", component);
        
        // Exponential backoff retry
        for attempt in 1..=3 {
            let backoff = Duration::from_millis(100 * 2_u64.pow(attempt - 1));
            sleep(backoff).await;
            
            // Simulate retry attempt
            if rand::random::<f64>() > 0.5 {
                let duration = start_time.elapsed();
                return Ok(RecoveryResult {
                    action_id: uuid::Uuid::new_v4().to_string(),
                    component: component.to_string(),
                    strategy: RecoveryStrategy::Retry,
                    success: true,
                    error_message: None,
                    duration_ms: duration.as_millis() as u64,
                    timestamp: Timestamp::now(),
                });
            }
        }
        
        let duration = start_time.elapsed();
        Ok(RecoveryResult {
            action_id: uuid::Uuid::new_v4().to_string(),
            component: component.to_string(),
            strategy: RecoveryStrategy::Retry,
            success: false,
            error_message: Some("All retry attempts failed".to_string()),
            duration_ms: duration.as_millis() as u64,
            timestamp: Timestamp::now(),
        })
    }
    
    /// Start circuit breaker monitoring
    async fn start_circuit_breaker_monitoring(&self) -> Result<()> {
        let circuit_breakers = Arc::clone(&self.circuit_breakers);
        let event_broadcaster = self.event_broadcaster.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                for mut entry in circuit_breakers.iter_mut() {
                    let component = entry.key().clone();
                    let circuit_breaker = entry.value_mut();
                    
                    let old_state = circuit_breaker.state;
                    
                    // Check if circuit breaker can transition states
                    circuit_breaker.can_execute();
                    
                    let new_state = circuit_breaker.state;
                    
                    // Broadcast state change events
                    if old_state != new_state {
                        let event = match new_state {
                            CircuitBreakerState::Open => RecoveryEvent::CircuitBreakerOpened {
                                component: component.clone(),
                                failure_count: circuit_breaker.failure_count,
                                timestamp: Timestamp::now(),
                            },
                            CircuitBreakerState::Closed => RecoveryEvent::CircuitBreakerClosed {
                                component: component.clone(),
                                timestamp: Timestamp::now(),
                            },
                            CircuitBreakerState::HalfOpen => continue, // Don't broadcast half-open transitions
                        };
                        
                        let _ = event_broadcaster.send(event);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start recovery history cleanup
    async fn start_history_cleanup(&self) -> Result<()> {
        let recovery_history = Arc::clone(&self.recovery_history);
        let max_history_size = self.max_history_size;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                let mut history = recovery_history.write();
                while history.len() > max_history_size {
                    history.pop_front();
                }
            }
        });
        
        Ok(())
    }
    
    /// Add recovery result to history
    fn add_to_history(&self, result: RecoveryResult) {
        let mut history = self.recovery_history.write();
        history.push_back(result);
        
        // Limit history size
        while history.len() > self.max_history_size {
            history.pop_front();
        }
    }
}

#[async_trait]
impl RecoveryManager for AdaptiveRecoveryManager {
    async fn recover_component(
        &self,
        component: String,
        strategy: RecoveryStrategy,
    ) -> Result<RecoveryResult> {
        self.total_attempts.fetch_add(1, Ordering::Relaxed);
        
        // Broadcast recovery started event
        let _ = self.event_broadcaster.send(RecoveryEvent::RecoveryStarted {
            component: component.clone(),
            strategy,
            timestamp: Timestamp::now(),
        });
        
        // Perform recovery based on strategy
        let result = match strategy {
            RecoveryStrategy::Restart => self.perform_restart(&component).await?,
            RecoveryStrategy::Failover => self.perform_failover(&component).await?,
            RecoveryStrategy::Retry => self.perform_retry(&component).await?,
            RecoveryStrategy::CircuitBreaker => {
                // Circuit breaker is handled separately
                return Err(OrchestrationError::recovery(
                    "Circuit breaker strategy should not be called directly".to_string()
                ));
            }
            RecoveryStrategy::GracefulDegradation => {
                // Enable graceful degradation
                self.enable_graceful_degradation(component.clone(), "Recovery attempt".to_string()).await?;
                
                RecoveryResult {
                    action_id: uuid::Uuid::new_v4().to_string(),
                    component: component.clone(),
                    strategy,
                    success: true,
                    error_message: None,
                    duration_ms: 0,
                    timestamp: Timestamp::now(),
                }
            }
        };
        
        // Update counters
        if result.success {
            self.successful_recoveries.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_recoveries.fetch_add(1, Ordering::Relaxed);
        }
        
        // Add to history
        self.add_to_history(result.clone());
        
        // Broadcast recovery completed event
        let event = if result.success {
            RecoveryEvent::RecoverySucceeded {
                component: component.clone(),
                strategy,
                duration_ms: result.duration_ms,
                timestamp: result.timestamp,
            }
        } else {
            RecoveryEvent::RecoveryFailed {
                component: component.clone(),
                strategy,
                error: result.error_message.clone().unwrap_or_default(),
                timestamp: result.timestamp,
            }
        };
        
        let _ = self.event_broadcaster.send(event);
        
        Ok(result)
    }
    
    async fn check_circuit_breaker(&self, component: &str) -> Result<CircuitBreakerState> {
        let circuit_breaker = self.get_or_create_circuit_breaker(component);
        Ok(circuit_breaker.state)
    }
    
    async fn record_success(&self, component: &str) -> Result<()> {
        if let Some(mut entry) = self.circuit_breakers.get_mut(component) {
            entry.record_success();
        }
        Ok(())
    }
    
    async fn record_failure(&self, component: &str) -> Result<CircuitBreakerState> {
        let mut circuit_breaker = self.get_or_create_circuit_breaker(component);
        let new_state = circuit_breaker.record_failure();
        
        // Update the stored circuit breaker
        self.circuit_breakers.insert(component.to_string(), circuit_breaker);
        
        Ok(new_state)
    }
    
    async fn enable_graceful_degradation(&self, component: String, reason: String) -> Result<()> {
        self.degraded_components.insert(component.clone(), reason.clone());
        
        // Broadcast event
        let _ = self.event_broadcaster.send(RecoveryEvent::GracefulDegradationActivated {
            component,
            reason,
            timestamp: Timestamp::now(),
        });
        
        Ok(())
    }
    
    async fn disable_graceful_degradation(&self, component: String) -> Result<()> {
        self.degraded_components.remove(&component);
        Ok(())
    }
    
    async fn get_recovery_stats(&self) -> Result<RecoveryStatistics> {
        let history = self.recovery_history.read();
        
        let mut attempts_by_strategy = HashMap::new();
        let mut success_count_by_strategy = HashMap::new();
        let mut total_time_by_strategy = HashMap::new();
        
        for result in history.iter() {
            *attempts_by_strategy.entry(result.strategy).or_insert(0) += 1;
            
            if result.success {
                *success_count_by_strategy.entry(result.strategy).or_insert(0) += 1;
            }
            
            *total_time_by_strategy.entry(result.strategy).or_insert(0.0) += result.duration_ms as f64;
        }
        
        let mut success_rate_by_strategy = HashMap::new();
        let mut avg_recovery_time_by_strategy = HashMap::new();
        
        for (strategy, attempts) in &attempts_by_strategy {
            let successes = success_count_by_strategy.get(strategy).unwrap_or(&0);
            let success_rate = *successes as f64 / *attempts as f64;
            success_rate_by_strategy.insert(*strategy, success_rate);
            
            let total_time = total_time_by_strategy.get(strategy).unwrap_or(&0.0);
            let avg_time = total_time / *attempts as f64;
            avg_recovery_time_by_strategy.insert(*strategy, avg_time);
        }
        
        let mut circuit_breaker_status = HashMap::new();
        for entry in self.circuit_breakers.iter() {
            circuit_breaker_status.insert(entry.key().clone(), entry.value().state);
        }
        
        let degraded_components: Vec<String> = self.degraded_components.iter()
            .map(|entry| entry.key().clone())
            .collect();
        
        Ok(RecoveryStatistics {
            total_attempts: self.total_attempts.load(Ordering::Relaxed),
            successful_recoveries: self.successful_recoveries.load(Ordering::Relaxed),
            failed_recoveries: self.failed_recoveries.load(Ordering::Relaxed),
            attempts_by_strategy,
            success_rate_by_strategy,
            avg_recovery_time_by_strategy,
            circuit_breaker_status,
            degraded_components,
        })
    }
    
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<RecoveryEvent>> {
        Ok(self.event_broadcaster.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[test]
    fn test_circuit_breaker() {
        let mut cb = CircuitBreaker::new("test_component".to_string(), 3, 5000);
        
        // Initially closed
        assert_eq!(cb.state, CircuitBreakerState::Closed);
        assert!(cb.can_execute());
        
        // Record failures
        cb.record_failure();
        assert_eq!(cb.state, CircuitBreakerState::Closed);
        
        cb.record_failure();
        assert_eq!(cb.state, CircuitBreakerState::Closed);
        
        cb.record_failure();
        assert_eq!(cb.state, CircuitBreakerState::Open);
        assert!(!cb.can_execute());
        
        // Record success in closed state
        cb.state = CircuitBreakerState::Closed;
        cb.record_success();
        assert_eq!(cb.failure_count, 0);
    }
    
    #[tokio::test]
    async fn test_recovery_manager() {
        let recovery_manager = AdaptiveRecoveryManager::new(3, 5000, 100);
        recovery_manager.start().await.unwrap();
        
        // Test restart recovery
        let result = recovery_manager.recover_component(
            "test_component".to_string(),
            RecoveryStrategy::Restart,
        ).await.unwrap();
        
        assert_eq!(result.component, "test_component");
        assert_eq!(result.strategy, RecoveryStrategy::Restart);
        
        // Test circuit breaker
        recovery_manager.record_failure("test_component").await.unwrap();
        recovery_manager.record_failure("test_component").await.unwrap();
        recovery_manager.record_failure("test_component").await.unwrap();
        
        let state = recovery_manager.check_circuit_breaker("test_component").await.unwrap();
        assert_eq!(state, CircuitBreakerState::Open);
        
        // Test graceful degradation
        recovery_manager.enable_graceful_degradation(
            "test_component".to_string(),
            "Testing degradation".to_string(),
        ).await.unwrap();
        
        let stats = recovery_manager.get_recovery_stats().await.unwrap();
        assert!(stats.degraded_components.contains(&"test_component".to_string()));
        
        recovery_manager.disable_graceful_degradation("test_component".to_string()).await.unwrap();
        
        let stats = recovery_manager.get_recovery_stats().await.unwrap();
        assert!(!stats.degraded_components.contains(&"test_component".to_string()));
    }
    
    #[tokio::test]
    async fn test_recovery_events() {
        let recovery_manager = AdaptiveRecoveryManager::new(3, 5000, 100);
        recovery_manager.start().await.unwrap();
        
        let mut event_receiver = recovery_manager.subscribe_events().await.unwrap();
        
        // Trigger a recovery
        let _result = recovery_manager.recover_component(
            "test_component".to_string(),
            RecoveryStrategy::Restart,
        ).await.unwrap();
        
        // Check for recovery started event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            RecoveryEvent::RecoveryStarted { component, strategy, .. } => {
                assert_eq!(component, "test_component");
                assert_eq!(strategy, RecoveryStrategy::Restart);
            }
            _ => panic!("Expected RecoveryStarted event"),
        }
        
        // Check for recovery completed event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            RecoveryEvent::RecoverySucceeded { component, strategy, .. } |
            RecoveryEvent::RecoveryFailed { component, strategy, .. } => {
                assert_eq!(component, "test_component");
                assert_eq!(strategy, RecoveryStrategy::Restart);
            }
            _ => panic!("Expected RecoverySucceeded or RecoveryFailed event"),
        }
    }
    
    #[tokio::test]
    async fn test_recovery_statistics() {
        let recovery_manager = AdaptiveRecoveryManager::new(3, 5000, 100);
        recovery_manager.start().await.unwrap();
        
        // Perform multiple recoveries
        for i in 0..5 {
            let _result = recovery_manager.recover_component(
                format!("component_{}", i),
                RecoveryStrategy::Restart,
            ).await.unwrap();
        }
        
        let stats = recovery_manager.get_recovery_stats().await.unwrap();
        assert_eq!(stats.total_attempts, 5);
        assert!(stats.attempts_by_strategy.contains_key(&RecoveryStrategy::Restart));
        assert_eq!(stats.attempts_by_strategy[&RecoveryStrategy::Restart], 5);
    }
}