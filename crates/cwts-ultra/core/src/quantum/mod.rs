//! Quantum Computing Module for CWTS
//!
//! This module implements quantum-inspired probabilistic computing
//! with GPU acceleration for ultra-high-frequency trading.

pub mod pbit_engine;
pub mod pbit_orderbook_integration;

pub use pbit_engine::*;
pub use pbit_orderbook_integration::*;