//! Circuit breakers and kill switches for emergency situations

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tokio::sync::Notify;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use rust_decimal::{Decimal, prelude::FromStr};
use crate::error::{ComplianceError, ComplianceResult};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Partially restricted - some operations blocked
    HalfOpen,
    /// Fully open - all operations blocked
    Open,
}

/// Trigger conditions for circuit breakers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Too many failures in time window
    FailureRate {
        threshold: f64,
        window: Duration,
    },
    /// Consecutive failures
    ConsecutiveFailures {
        count: u32,
    },
    /// Loss threshold exceeded
    LossThreshold {
        max_loss: Decimal,
        window: Duration,
    },
    /// Volatility spike
    VolatilitySpike {
        threshold: f64,
    },
    /// Manual trigger
    Manual {
        reason: String,
    },
    /// System overload
    SystemOverload {
        cpu_threshold: f64,
        memory_threshold: f64,
    },
}

/// Individual circuit breaker
pub struct CircuitBreaker {
    name: String,
    state: Arc<RwLock<CircuitState>>,
    trigger_conditions: Vec<TriggerCondition>,
    failure_count: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
    cool_down_period: Duration,
    state_changed: Arc<Notify>,
}

impl CircuitBreaker {
    pub fn new(name: String, trigger_conditions: Vec<TriggerCondition>, cool_down_period: Duration) -> Self {
        Self {
            name,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            trigger_conditions,
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            cool_down_period,
            state_changed: Arc::new(Notify::new()),
        }
    }

    pub fn check(&self) -> ComplianceResult<()> {
        let state = *self.state.read();
        match state {
            CircuitState::Closed => Ok(()),
            CircuitState::HalfOpen => {
                // Allow some operations through for testing
                Ok(())
            }
            CircuitState::Open => {
                Err(ComplianceError::CircuitBreakerTriggered {
                    reason: format!("Circuit breaker '{}' is open", self.name),
                })
            }
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.write();
        let mut failure_count = self.failure_count.write();
        
        match *state {
            CircuitState::HalfOpen => {
                // Success in half-open state -> close the circuit
                *state = CircuitState::Closed;
                *failure_count = 0;
                self.state_changed.notify_waiters();
            }
            _ => {
                // Reset failure count on success
                *failure_count = 0;
            }
        }
    }

    pub fn record_failure(&self) {
        let mut failure_count = self.failure_count.write();
        *failure_count += 1;
        *self.last_failure.write() = Some(Instant::now());
        
        // Check if we should trip the breaker
        self.evaluate_triggers();
    }

    fn evaluate_triggers(&self) {
        let failure_count = *self.failure_count.read();
        let mut should_trip = false;
        
        for condition in &self.trigger_conditions {
            match condition {
                TriggerCondition::ConsecutiveFailures { count } => {
                    if failure_count >= *count {
                        should_trip = true;
                        break;
                    }
                }
                // Other conditions would be evaluated here
                _ => {}
            }
        }
        
        if should_trip {
            self.trip();
        }
    }

    fn trip(&self) {
        let mut state = self.state.write();
        if *state != CircuitState::Open {
            *state = CircuitState::Open;
            self.state_changed.notify_waiters();
            
            // Schedule automatic half-open transition
            let state_clone = self.state.clone();
            let notify_clone = self.state_changed.clone();
            let cool_down = self.cool_down_period;
            
            tokio::spawn(async move {
                tokio::time::sleep(cool_down).await;
                let mut state = state_clone.write();
                if *state == CircuitState::Open {
                    *state = CircuitState::HalfOpen;
                    notify_clone.notify_waiters();
                }
            });
        }
    }

    pub fn force_open(&self, reason: &str) {
        let mut state = self.state.write();
        *state = CircuitState::Open;
        self.state_changed.notify_waiters();
        tracing::error!("Circuit breaker '{}' force opened: {}", self.name, reason);
    }

    pub fn reset(&self) {
        let mut state = self.state.write();
        *state = CircuitState::Closed;
        *self.failure_count.write() = 0;
        self.state_changed.notify_waiters();
    }

    pub fn get_state(&self) -> CircuitState {
        *self.state.read()
    }
}

/// Master kill switch for emergency shutdown
pub struct KillSwitch {
    activated: Arc<RwLock<bool>>,
    activation_time: Arc<RwLock<Option<Instant>>>,
    trigger_reason: Arc<RwLock<Option<String>>>,
    notifier: Arc<Notify>,
}

impl KillSwitch {
    pub fn new() -> Self {
        Self {
            activated: Arc::new(RwLock::new(false)),
            activation_time: Arc::new(RwLock::new(None)),
            trigger_reason: Arc::new(RwLock::new(None)),
            notifier: Arc::new(Notify::new()),
        }
    }

    pub fn activate(&self, reason: String) -> ComplianceResult<()> {
        let mut activated = self.activated.write();
        if !*activated {
            *activated = true;
            *self.activation_time.write() = Some(Instant::now());
            *self.trigger_reason.write() = Some(reason.clone());
            self.notifier.notify_waiters();
            
            tracing::error!("KILL SWITCH ACTIVATED: {}", reason);
            
            Err(ComplianceError::KillSwitchActivated {
                trigger: reason,
            })
        } else {
            Err(ComplianceError::KillSwitchActivated {
                trigger: self.trigger_reason.read().clone().unwrap_or_default(),
            })
        }
    }

    pub fn check(&self) -> ComplianceResult<()> {
        if *self.activated.read() {
            Err(ComplianceError::KillSwitchActivated {
                trigger: self.trigger_reason.read().clone().unwrap_or_default(),
            })
        } else {
            Ok(())
        }
    }

    pub fn is_activated(&self) -> bool {
        *self.activated.read()
    }

    pub fn deactivate(&self, authorized_by: &str) -> ComplianceResult<()> {
        let mut activated = self.activated.write();
        if *activated {
            *activated = false;
            *self.activation_time.write() = None;
            *self.trigger_reason.write() = None;
            
            tracing::warn!("Kill switch deactivated by: {}", authorized_by);
            Ok(())
        } else {
            Ok(())
        }
    }

    pub async fn wait_for_activation(&self) {
        self.notifier.notified().await;
    }
}

/// Circuit breaker manager
pub struct CircuitBreakerManager {
    breakers: Arc<DashMap<String, Arc<CircuitBreaker>>>,
    kill_switch: Arc<KillSwitch>,
    global_failure_rate: Arc<RwLock<f64>>,
}

impl CircuitBreakerManager {
    pub fn new() -> Self {
        Self {
            breakers: Arc::new(DashMap::new()),
            kill_switch: Arc::new(KillSwitch::new()),
            global_failure_rate: Arc::new(RwLock::new(0.0)),
        }
    }

    pub fn register_breaker(&self, name: String, breaker: CircuitBreaker) {
        self.breakers.insert(name.clone(), Arc::new(breaker));
    }

    pub fn check_breaker(&self, name: &str) -> ComplianceResult<()> {
        // First check kill switch
        self.kill_switch.check()?;
        
        // Then check specific breaker
        if let Some(breaker) = self.breakers.get(name) {
            breaker.check()
        } else {
            Ok(())
        }
    }

    pub fn check_all(&self) -> ComplianceResult<()> {
        // Check kill switch first
        self.kill_switch.check()?;
        
        // Check all breakers
        for entry in self.breakers.iter() {
            entry.value().check()?;
        }
        
        Ok(())
    }

    pub fn record_success(&self, breaker_name: &str) {
        if let Some(breaker) = self.breakers.get(breaker_name) {
            breaker.record_success();
        }
    }

    pub fn record_failure(&self, breaker_name: &str) {
        if let Some(breaker) = self.breakers.get(breaker_name) {
            breaker.record_failure();
        }
        
        // Update global failure rate
        self.update_global_failure_rate();
    }

    fn update_global_failure_rate(&self) {
        // Simple implementation - would be more sophisticated in production
        let mut open_count = 0;
        let total_count = self.breakers.len();
        
        for entry in self.breakers.iter() {
            if entry.value().get_state() == CircuitState::Open {
                open_count += 1;
            }
        }
        
        if total_count > 0 {
            *self.global_failure_rate.write() = (open_count as f64 / total_count as f64) * 100.0;
        }
    }

    pub fn activate_kill_switch(&self, reason: String) -> ComplianceResult<()> {
        self.kill_switch.activate(reason)
    }

    pub fn get_status(&self) -> CircuitBreakerStatus {
        let mut breaker_states = vec![];
        
        for entry in self.breakers.iter() {
            breaker_states.push(BreakerStatus {
                name: entry.key().clone(),
                state: entry.value().get_state(),
            });
        }
        
        CircuitBreakerStatus {
            kill_switch_active: self.kill_switch.is_activated(),
            global_failure_rate: *self.global_failure_rate.read(),
            breaker_states,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CircuitBreakerStatus {
    pub kill_switch_active: bool,
    pub global_failure_rate: f64,
    pub breaker_states: Vec<BreakerStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BreakerStatus {
    pub name: String,
    pub state: CircuitState,
}