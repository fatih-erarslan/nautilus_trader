//! Emergency protocols and circuit breakers
//!
//! Automated risk protection mechanisms:
//! - Circuit breakers (rapid loss, drawdown triggers)
//! - Position flattening
//! - Trading halts
//! - Alert escalation
//! - System shutdown protocols

pub mod circuit_breakers;
pub mod protocols;

pub use circuit_breakers::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerEvent,
    CircuitBreakerState, TriggerCondition
};
pub use protocols::{
    EmergencyProtocol, EmergencyConfig, EmergencyActionRecord, ProtocolStatus
};
