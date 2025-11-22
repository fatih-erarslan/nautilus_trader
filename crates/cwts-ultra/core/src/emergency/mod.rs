//! Emergency Systems Module
//!
//! This module contains kill switch functionality and emergency
//! halt mechanisms for immediate trading suspension.

pub mod kill_switch;

// Re-export main types for convenience
pub use kill_switch::*;
