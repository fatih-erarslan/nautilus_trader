//! Common types shared across CWTS modules to avoid duplication
//!
//! This module provides unified type definitions used throughout the trading system
//! to prevent type conflicts and ensure consistency.

use serde::{Deserialize, Serialize};

/// Unified TradeSide enum used across all trading modules
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "Buy"),
            TradeSide::Sell => write!(f, "Sell"),
        }
    }
}

/// Unified OrderType enum used across all trading modules
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Market => write!(f, "Market"),
            OrderType::Limit => write!(f, "Limit"),
            OrderType::Stop => write!(f, "Stop"),
            OrderType::StopLimit => write!(f, "StopLimit"),
        }
    }
}

/// Unified trade identifier
pub type TradeId = u64;

/// Unified position size type
pub type Size = f64;

/// Unified price type
pub type Price = f64;

/// Unified fee type
pub type Fee = f64;

/// Unified timestamp type (Unix timestamp in milliseconds)
pub type Timestamp = u64;
