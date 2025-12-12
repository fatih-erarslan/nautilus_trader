//! # Unified Risk Management System

pub mod core;
pub mod quantum;
pub mod portfolio;
pub mod behavioral;
pub mod monitoring;
pub mod antifragile;
pub mod simple_manager;

// Advanced risk management from standalone files
pub mod crisis_management;
pub mod emergency_override;

// Re-export simplified manager for main system
pub use simple_manager::RiskManager;
pub use crisis_management::*;
pub use emergency_override::*;