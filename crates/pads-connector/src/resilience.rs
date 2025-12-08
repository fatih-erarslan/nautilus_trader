//! Resilience and adaptive capacity mechanisms

use crate::{
    config::{PadsConfig, ResilienceConfig, CircuitBreakerConfig},
    error::{PadsError, Result},
    monitoring::PadsMonitor,
    types::*,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use std::time::{Duration, Instant};

/// Manages system resilience and recovery
pub struct ResilienceEngine {
    config: Arc<PadsConfig>,
    monitor: Arc<PadsMonitor>,
    circuit_breakers: DashMap<String, CircuitBreaker>,
    fault_detector: Arc<FaultDetector>,
    recovery_manager: Arc<RecoveryManager>,
    adaptive_capacity: Arc<RwLock<AdaptiveCapacityManager>>,
}

/// Circuit breaker for fault tolerance
struct CircuitBreaker {
    id: String,
    state: RwLock<CircuitState>,
    failure_count: RwLock<usize>,
    success_count: RwLock<usize>,
    last_failure: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
}

/// Fault detector
struct FaultDetector {
    fault_patterns: DashMap<String, FaultPattern>,
    detection_history: RwLock<Vec<DetectedFault>>,
}

/// Fault pattern
#[derive(Debug, Clone)]
struct FaultPattern {
    pattern_type: FaultType,
    threshold: f64,
    window: Duration,
    severity: FaultSeverity,
}

/// Fault type
#[derive(Debug, Clone, Copy)]
enum FaultType {
    Performance,
    Capacity,
    Communication,
    ScaleTransition,
    DecisionQuality,
}

/// Fault severity
#[derive(Debug, Clone, Copy)]
enum FaultSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detected fault
#[derive(Debug, Clone)]
struct DetectedFault {
    fault_type: FaultType,
    severity: FaultSeverity,
    timestamp: Instant,
    details: String,
}

/// Recovery manager
struct RecoveryManager {
    strategies: DashMap<FaultType, RecoveryStrategy>,
    active_recoveries: DashMap<String, ActiveRecovery>,
}

/// Recovery strategy
#[derive(Debug, Clone)]
struct RecoveryStrategy {
    strategy_type: RecoveryType,
    steps: Vec<RecoveryStep>,
    timeout: Duration,
}

/// Recovery type
#[derive(Debug, Clone, Copy)]
enum RecoveryType {
    Immediate,
    Gradual,
    Adaptive,
    Failover,
}

/// Recovery step
#[derive(Debug, Clone)]
struct RecoveryStep {
    action: String,
    parameters: serde_json::Value,
    expected_duration: Duration,
}

/// Active recovery process
#[derive(Debug, Clone)]
struct ActiveRecovery {
    id: String,
    fault: DetectedFault,
    strategy: RecoveryStrategy,
    started_at: Instant,
    current_step: usize,
    status: RecoveryStatus,
}

/// Recovery status
#[derive(Debug, Clone, Copy)]
enum RecoveryStatus {
    InProgress,
    Successful,
    Failed,
    Timeout,
}

/// Adaptive capacity manager
struct AdaptiveCapacityManager {
    total_capacity: f64,
    used_capacity: f64,
    resilience_buffer: f64,
    adaptation_rate: f64,
    capacity_history: Vec<CapacitySnapshot>,
}

/// Capacity snapshot
#[derive(Debug, Clone)]
struct CapacitySnapshot {
    timestamp: Instant,
    total: f64,
    used: f64,
    resilience_score: f64,
}

impl ResilienceEngine {
    /// Create new resilience engine
    pub async fn new(config: Arc<PadsConfig>, monitor: Arc<PadsMonitor>) -> Result<Self> {
        let circuit_breakers = DashMap::new();
        
        let fault_detector = Arc::new(FaultDetector {
            fault_patterns: Self::create_fault_patterns(),
            detection_history: RwLock::new(Vec::new()),
        });
        
        let recovery_manager = Arc::new(RecoveryManager {
            strategies: Self::create_recovery_strategies(),
            active_recoveries: DashMap::new(),
        });
        
        let adaptive_capacity = Arc::new(RwLock::new(AdaptiveCapacityManager {
            total_capacity: 1.0,
            used_capacity: 0.0,
            resilience_buffer: 0.2,
            adaptation_rate: 0.1,
            capacity_history: Vec::new(),
        }));
        
        Ok(Self {
            config,
            monitor,
            circuit_breakers,
            fault_detector,
            recovery_manager,
            adaptive_capacity,
        })
    }
    
    /// Create default fault patterns
    fn create_fault_patterns() -> DashMap<String, FaultPattern> {
        let patterns = DashMap::new();
        
        patterns.insert("high_latency".to_string(), FaultPattern {
            pattern_type: FaultType::Performance,
            threshold: 100.0, // ms
            window: Duration::from_secs(60),
            severity: FaultSeverity::Medium,
        });
        
        patterns.insert("capacity_exhaustion".to_string(), FaultPattern {
            pattern_type: FaultType::Capacity,
            threshold: 0.9, // 90% usage
            window: Duration::from_secs(30),
            severity: FaultSeverity::High,
        });
        
        patterns.insert("communication_failure".to_string(), FaultPattern {
            pattern_type: FaultType::Communication,
            threshold: 0.1, // 10% error rate
            window: Duration::from_secs(120),
            severity: FaultSeverity::High,
        });
        
        patterns.insert("scale_transition_failure".to_string(), FaultPattern {
            pattern_type: FaultType::ScaleTransition,
            threshold: 3.0, // failures
            window: Duration::from_secs(300),
            severity: FaultSeverity::Critical,
        });
        
        patterns.insert("decision_quality_degradation".to_string(), FaultPattern {
            pattern_type: FaultType::DecisionQuality,
            threshold: 0.5, // 50% success rate
            window: Duration::from_secs(180),
            severity: FaultSeverity::Medium,
        });
        
        patterns
    }
    
    /// Create recovery strategies
    fn create_recovery_strategies() -> DashMap<FaultType, RecoveryStrategy> {
        let strategies = DashMap::new();
        
        strategies.insert(FaultType::Performance, RecoveryStrategy {
            strategy_type: RecoveryType::Gradual,
            steps: vec![
                RecoveryStep {
                    action: "reduce_load".to_string(),
                    parameters: serde_json::json!({"factor": 0.5}),
                    expected_duration: Duration::from_secs(10),
                },
                RecoveryStep {
                    action: "optimize_parameters".to_string(),
                    parameters: serde_json::json!({"target": "latency"}),
                    expected_duration: Duration::from_secs(20),
                },
            ],
            timeout: Duration::from_secs(60),
        });
        
        strategies.insert(FaultType::Capacity, RecoveryStrategy {
            strategy_type: RecoveryType::Immediate,
            steps: vec![
                RecoveryStep {
                    action: "shed_load".to_string(),
                    parameters: serde_json::json!({"percentage": 30}),
                    expected_duration: Duration::from_secs(5),
                },
                RecoveryStep {
                    action: "scale_resources".to_string(),
                    parameters: serde_json::json!({"factor": 1.5}),
                    expected_duration: Duration::from_secs(15),
                },
            ],
            timeout: Duration::from_secs(30),
        });
        
        strategies.insert(FaultType::Communication, RecoveryStrategy {
            strategy_type: RecoveryType::Adaptive,
            steps: vec![
                RecoveryStep {
                    action: "reconnect_channels".to_string(),
                    parameters: serde_json::json!({}),
                    expected_duration: Duration::from_secs(10),
                },
                RecoveryStep {
                    action: "switch_protocol".to_string(),
                    parameters: serde_json::json!({"fallback": true}),
                    expected_duration: Duration::from_secs(5),
                },
            ],
            timeout: Duration::from_secs(45),
        });
        
        strategies
    }
    
    /// Configure resilience mechanisms
    pub async fn configure(&self) -> Result<()> {
        info!("Configuring resilience mechanisms");
        
        // Initialize circuit breakers
        self.initialize_circuit_breakers().await?;
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        Ok(())
    }
    
    /// Initialize circuit breakers
    async fn initialize_circuit_breakers(&self) -> Result<()> {
        let components = vec![
            "scale_manager",
            "decision_router",
            "communicator",
            "micro_processor",
            "meso_processor",
            "macro_processor",
        ];
        
        for component in components {
            let breaker = CircuitBreaker {
                id: component.to_string(),
                state: RwLock::new(CircuitState::Closed),
                failure_count: RwLock::new(0),
                success_count: RwLock::new(0),
                last_failure: RwLock::new(None),
                config: self.config.resilience_config.circuit_breaker.clone(),
            };
            
            self.circuit_breakers.insert(component.to_string(), breaker);
        }
        
        Ok(())
    }
    
    /// Start health monitoring
    async fn start_health_monitoring(&self) -> Result<()> {
        let interval = self.config.resilience_config.health_check_interval;
        let engine = Arc::downgrade(&Arc::new(self));
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Some(engine) = engine.upgrade() {
                    if let Err(e) = engine.perform_health_check().await {
                        error!("Health check failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// Perform health check
    async fn perform_health_check(&self) -> Result<()> {
        debug!("Performing health check");
        
        // Check circuit breaker states
        for breaker_ref in self.circuit_breakers.iter() {
            let breaker = breaker_ref.value();
            let state = *breaker.state.read().await;
            
            if state == CircuitState::Open {
                warn!("Circuit breaker {} is open", breaker.id);
                
                // Check if should transition to half-open
                if let Some(last_failure) = *breaker.last_failure.read().await {
                    if last_failure.elapsed() > breaker.config.half_open_interval {
                        *breaker.state.write().await = CircuitState::HalfOpen;
                        info!("Circuit breaker {} transitioned to half-open", breaker.id);
                    }
                }
            }
        }
        
        // Check adaptive capacity
        let capacity = self.adaptive_capacity.read().await;
        if capacity.used_capacity > 0.8 {
            warn!("High capacity usage: {:.2}%", capacity.used_capacity * 100.0);
        }
        
        // Detect faults
        self.detect_faults().await?;
        
        Ok(())
    }
    
    /// Detect system faults
    async fn detect_faults(&self) -> Result<()> {
        let mut detected_faults = Vec::new();
        
        // Check each fault pattern
        for pattern_ref in self.fault_detector.fault_patterns.iter() {
            let pattern = pattern_ref.value();
            
            // Check if pattern is triggered (simplified)
            let triggered = match pattern.pattern_type {
                FaultType::Performance => self.check_performance_fault(pattern).await?,
                FaultType::Capacity => self.check_capacity_fault(pattern).await?,
                FaultType::Communication => self.check_communication_fault(pattern).await?,
                FaultType::ScaleTransition => self.check_scale_transition_fault(pattern).await?,
                FaultType::DecisionQuality => self.check_decision_quality_fault(pattern).await?,
            };
            
            if triggered {
                detected_faults.push(DetectedFault {
                    fault_type: pattern.pattern_type,
                    severity: pattern.severity,
                    timestamp: Instant::now(),
                    details: format!("Fault pattern {} triggered", pattern_ref.key()),
                });
            }
        }
        
        // Process detected faults
        for fault in detected_faults {
            self.handle_fault(fault).await?;
        }
        
        Ok(())
    }
    
    /// Check performance fault
    async fn check_performance_fault(&self, pattern: &FaultPattern) -> Result<bool> {
        // Check recent performance metrics
        // Simplified implementation
        Ok(false)
    }
    
    /// Check capacity fault
    async fn check_capacity_fault(&self, pattern: &FaultPattern) -> Result<bool> {
        let capacity = self.adaptive_capacity.read().await;
        Ok(capacity.used_capacity > pattern.threshold)
    }
    
    /// Check communication fault
    async fn check_communication_fault(&self, pattern: &FaultPattern) -> Result<bool> {
        // Check communication error rate
        // Simplified implementation
        Ok(false)
    }
    
    /// Check scale transition fault
    async fn check_scale_transition_fault(&self, pattern: &FaultPattern) -> Result<bool> {
        // Check recent transition failures
        // Simplified implementation
        Ok(false)
    }
    
    /// Check decision quality fault
    async fn check_decision_quality_fault(&self, pattern: &FaultPattern) -> Result<bool> {
        // Check decision success rate
        // Simplified implementation
        Ok(false)
    }
    
    /// Handle detected fault
    async fn handle_fault(&self, fault: DetectedFault) -> Result<()> {
        error!("Handling fault: {:?} with severity {:?}", fault.fault_type, fault.severity);
        
        // Record fault
        self.fault_detector.detection_history.write().await.push(fault.clone());
        
        // Initiate recovery if needed
        match fault.severity {
            FaultSeverity::Critical => {
                self.initiate_recovery(fault).await?;
            }
            FaultSeverity::High => {
                if self.should_recover(&fault).await? {
                    self.initiate_recovery(fault).await?;
                }
            }
            _ => {
                // Log and monitor
                warn!("Fault detected but not critical: {:?}", fault.fault_type);
            }
        }
        
        Ok(())
    }
    
    /// Check if should initiate recovery
    async fn should_recover(&self, fault: &DetectedFault) -> Result<bool> {
        // Check if recovery already in progress
        for recovery_ref in self.recovery_manager.active_recoveries.iter() {
            let recovery = recovery_ref.value();
            if recovery.fault.fault_type as u8 == fault.fault_type as u8 {
                return Ok(false); // Already recovering
            }
        }
        
        Ok(true)
    }
    
    /// Initiate recovery process
    async fn initiate_recovery(&self, fault: DetectedFault) -> Result<()> {
        info!("Initiating recovery for fault: {:?}", fault.fault_type);
        
        let strategy = self.recovery_manager.strategies
            .get(&fault.fault_type)
            .ok_or_else(|| PadsError::resilience("No recovery strategy found"))?
            .clone();
        
        let recovery = ActiveRecovery {
            id: uuid::Uuid::new_v4().to_string(),
            fault: fault.clone(),
            strategy: strategy.clone(),
            started_at: Instant::now(),
            current_step: 0,
            status: RecoveryStatus::InProgress,
        };
        
        self.recovery_manager.active_recoveries
            .insert(recovery.id.clone(), recovery.clone());
        
        // Execute recovery
        self.execute_recovery(recovery).await?;
        
        Ok(())
    }
    
    /// Execute recovery process
    async fn execute_recovery(&self, mut recovery: ActiveRecovery) -> Result<()> {
        for (i, step) in recovery.strategy.steps.iter().enumerate() {
            recovery.current_step = i;
            
            info!("Executing recovery step: {}", step.action);
            
            // Execute step action
            match step.action.as_str() {
                "reduce_load" => self.reduce_load(&step.parameters).await?,
                "optimize_parameters" => self.optimize_parameters(&step.parameters).await?,
                "shed_load" => self.shed_load(&step.parameters).await?,
                "scale_resources" => self.scale_resources(&step.parameters).await?,
                "reconnect_channels" => self.reconnect_channels(&step.parameters).await?,
                "switch_protocol" => self.switch_protocol(&step.parameters).await?,
                _ => warn!("Unknown recovery action: {}", step.action),
            }
            
            // Wait for step completion
            tokio::time::sleep(step.expected_duration).await;
        }
        
        // Mark recovery as successful
        recovery.status = RecoveryStatus::Successful;
        self.recovery_manager.active_recoveries.remove(&recovery.id);
        
        info!("Recovery completed successfully");
        
        Ok(())
    }
    
    /// Reduce system load
    async fn reduce_load(&self, params: &serde_json::Value) -> Result<()> {
        let factor = params["factor"].as_f64().unwrap_or(0.5);
        info!("Reducing load by factor: {}", factor);
        
        // Implement load reduction
        self.monitor.record_recovery_action("reduce_load", factor);
        
        Ok(())
    }
    
    /// Optimize parameters
    async fn optimize_parameters(&self, params: &serde_json::Value) -> Result<()> {
        let target = params["target"].as_str().unwrap_or("general");
        info!("Optimizing parameters for: {}", target);
        
        // Implement parameter optimization
        self.monitor.record_recovery_action("optimize_parameters", 1.0);
        
        Ok(())
    }
    
    /// Shed load immediately
    async fn shed_load(&self, params: &serde_json::Value) -> Result<()> {
        let percentage = params["percentage"].as_f64().unwrap_or(30.0);
        warn!("Shedding {}% of load", percentage);
        
        // Implement load shedding
        self.monitor.record_recovery_action("shed_load", percentage);
        
        Ok(())
    }
    
    /// Scale resources
    async fn scale_resources(&self, params: &serde_json::Value) -> Result<()> {
        let factor = params["factor"].as_f64().unwrap_or(1.5);
        info!("Scaling resources by factor: {}", factor);
        
        // Update adaptive capacity
        let mut capacity = self.adaptive_capacity.write().await;
        capacity.total_capacity *= factor;
        
        self.monitor.record_recovery_action("scale_resources", factor);
        
        Ok(())
    }
    
    /// Reconnect communication channels
    async fn reconnect_channels(&self, _params: &serde_json::Value) -> Result<()> {
        info!("Reconnecting communication channels");
        
        // Trigger channel reconnection
        self.monitor.record_recovery_action("reconnect_channels", 1.0);
        
        Ok(())
    }
    
    /// Switch to fallback protocol
    async fn switch_protocol(&self, params: &serde_json::Value) -> Result<()> {
        let fallback = params["fallback"].as_bool().unwrap_or(true);
        info!("Switching to {} protocol", if fallback { "fallback" } else { "primary" });
        
        self.monitor.record_recovery_action("switch_protocol", 1.0);
        
        Ok(())
    }
    
    /// Activate recovery mechanisms
    pub async fn activate_recovery(&self) -> Result<()> {
        warn!("Activating all recovery mechanisms");
        
        // Reset circuit breakers
        for breaker_ref in self.circuit_breakers.iter() {
            let breaker = breaker_ref.value();
            *breaker.state.write().await = CircuitState::Closed;
            *breaker.failure_count.write().await = 0;
            *breaker.success_count.write().await = 0;
        }
        
        // Clear fault history
        self.fault_detector.detection_history.write().await.clear();
        
        // Reset adaptive capacity
        let mut capacity = self.adaptive_capacity.write().await;
        capacity.used_capacity = 0.0;
        capacity.resilience_buffer = 0.2;
        
        Ok(())
    }
    
    /// Update resilience thresholds
    pub async fn update_thresholds(&self, feedback: AdaptiveFeedback) -> Result<()> {
        let mut capacity = self.adaptive_capacity.write().await;
        
        // Adjust based on performance
        if feedback.performance_score > 0.8 {
            capacity.resilience_buffer = (capacity.resilience_buffer - 0.05).max(0.1);
        } else if feedback.performance_score < 0.5 {
            capacity.resilience_buffer = (capacity.resilience_buffer + 0.05).min(0.5);
        }
        
        // Update adaptation rate
        capacity.adaptation_rate = capacity.adaptation_rate * 0.9 + 
                                  feedback.performance_score * 0.1;
        
        Ok(())
    }
    
    /// Get resilience capacity
    pub async fn get_capacity(&self) -> Result<f64> {
        let capacity = self.adaptive_capacity.read().await;
        let available = capacity.total_capacity - capacity.used_capacity;
        let resilience = available * (1.0 + capacity.resilience_buffer);
        
        Ok(resilience.clamp(0.0, 1.0))
    }
    
    /// Get resilience status
    pub async fn get_status(&self) -> Result<ResilienceStatus> {
        let mut circuit_state = CircuitState::Closed;
        
        // Check circuit breakers
        for breaker_ref in self.circuit_breakers.iter() {
            let state = *breaker_ref.value().state.read().await;
            if state == CircuitState::Open {
                circuit_state = CircuitState::Open;
                break;
            } else if state == CircuitState::HalfOpen && circuit_state == CircuitState::Closed {
                circuit_state = CircuitState::HalfOpen;
            }
        }
        
        let capacity = self.adaptive_capacity.read().await;
        let health_score = 1.0 - capacity.used_capacity;
        let recovery_capacity = self.get_capacity().await?;
        let fault_tolerance_level = capacity.resilience_buffer;
        
        Ok(ResilienceStatus {
            circuit_breaker_state: circuit_state,
            health_score,
            recovery_capacity,
            fault_tolerance_level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_resilience_configuration() {
        let config = Arc::new(PadsConfig::default());
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await.unwrap());
        let engine = ResilienceEngine::new(config, monitor).await.unwrap();
        
        assert!(engine.configure().await.is_ok());
        assert!(!engine.circuit_breakers.is_empty());
    }
}