//! MCP Server Implementation
//!
//! This crate provides a complete MCP (Model Context Protocol) server
//! with support for multiple transport layers and all 87 neural-trader tools.

#![recursion_limit = "512"]

pub mod transport;
pub mod tools;
pub mod handlers;

pub use neural_trader_mcp_protocol::*;

/// MCP Server version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
