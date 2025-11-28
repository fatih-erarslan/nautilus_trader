//! Core types and engine implementation for HyperRiskEngine.
//!
//! This module provides the fundamental types and the main engine
//! that orchestrates all risk management components.

pub mod error;
pub mod types;
pub mod engine;
pub mod ring_buffer;

pub use error::{RiskError, Result};
pub use types::{
    Timestamp, Price, Quantity, Symbol, PositionId,
    Position, Order, OrderSide, Portfolio,
    RiskLevel, RiskDecision, MarketRegime,
};
pub use engine::{HyperRiskEngine, EngineConfig, EngineState};
pub use ring_buffer::{RingBuffer, RingBufferConfig};
