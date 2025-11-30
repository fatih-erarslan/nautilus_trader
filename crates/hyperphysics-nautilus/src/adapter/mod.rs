//! Adapter modules for Nautilus-HyperPhysics bridge.
//!
//! This module provides the core adapters that transform data between
//! Nautilus Trader and HyperPhysics systems.

mod data_adapter;
mod exec_bridge;

pub use data_adapter::NautilusDataAdapter;
pub use exec_bridge::{NautilusExecBridge, ExecBridgeStats};
