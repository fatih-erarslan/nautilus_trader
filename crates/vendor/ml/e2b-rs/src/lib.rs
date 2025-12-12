//! # E2B Rust Client
//!
//! A Rust client for E2B - secure cloud sandboxes for AI-generated code execution.
//!
//! ## Overview
//!
//! E2B is an open-source infrastructure for running AI-generated code in secure
//! isolated sandboxes in the cloud. This crate provides a Rust interface to the
//! E2B API.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use e2b_rs::{E2BClient, SandboxConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), e2b_rs::Error> {
//!     let client = E2BClient::new("your-api-key")?;
//!     
//!     // Create a sandbox
//!     let sandbox = client.create_sandbox(SandboxConfig::default()).await?;
//!     
//!     // Execute code
//!     let result = sandbox.execute("print('Hello, World!')").await?;
//!     println!("Output: {}", result.stdout);
//!     
//!     // Clean up
//!     sandbox.close().await?;
//!     Ok(())
//! }
//! ```

mod client;
mod error;
mod sandbox;
mod types;

pub use client::E2BClient;
pub use error::{Error, Result};
pub use sandbox::{Sandbox, SandboxConfig, SandboxStatus};
pub use types::*;

/// Default E2B API base URL
pub const DEFAULT_API_URL: &str = "https://api.e2b.dev/v1";

/// Default sandbox template
pub const DEFAULT_TEMPLATE: &str = "base";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = E2BClient::new("test-api-key");
        assert!(client.is_ok());
    }

    #[test]
    fn test_sandbox_config_default() {
        let config = SandboxConfig::default();
        assert_eq!(config.template, DEFAULT_TEMPLATE);
    }
}
