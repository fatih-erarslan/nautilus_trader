//! Post-Quantum Secure MCP Authentication Layer
//!
//! This crate provides cryptographic authentication for Model Context Protocol (MCP)
//! communications, protecting against injection attacks through Dilithium-based signatures.
//!
//! # Security Architecture
//!
//! ## Threat Model
//!
//! MCP injection attacks occur when malicious prompts manipulate LLMs into making
//! unauthorized tool calls. This crate prevents such attacks by:
//!
//! 1. **Request Signing**: All MCP requests must be signed with Dilithium signatures
//! 2. **Origin Validation**: Requests must originate from authenticated endpoints
//! 3. **Replay Protection**: Nonce-based protection against replay attacks
//! 4. **Content Integrity**: BLAKE3 hashing ensures request tampering is detected
//!
//! ## Research Foundation
//!
//! - Greshake et al. (2023): "Not What You've Signed Up For: Compromising LLM-Integrated Applications"
//! - Perez & Ribeiro (2022): "Ignore This Title and HackAPrompt: Exposing Systematic Vulnerabilities"
//! - NIST FIPS 204 (2024): ML-DSA (Dilithium) post-quantum digital signature standard
//!
//! # Example
//!
//! ```rust,no_run
//! use hyperphysics_mcp_auth::*;
//!
//! # async fn example() -> Result<(), McpAuthError> {
//! // Create authenticated MCP guard
//! let guard = McpAuthGuard::new(SecurityConfig::default())?;
//!
//! // Register authorized client
//! let client_id = guard.register_client("trusted-client")?;
//!
//! // Create signed request
//! let request = McpRequest::new("vector_db_search", serde_json::json!({
//!     "query": [0.1, 0.2, 0.3],
//!     "k": 10
//! }));
//!
//! let signed = guard.sign_request(&client_id, request)?;
//!
//! // Verify and execute
//! guard.verify_and_authorize(&signed)?;
//! # Ok(())
//! # }
//! ```

pub mod auth_guard;
pub mod request;
pub mod client;
pub mod nonce;
pub mod config;
pub mod audit;
pub mod transport;

// Re-exports
pub use auth_guard::McpAuthGuard;
pub use request::{McpRequest, SignedMcpRequest, McpResponse};
pub use client::{AuthenticatedClient, ClientCredentials, ClientPermissions, PublicClientCredentials};
pub use nonce::NonceManager;
pub use config::SecurityConfig;
pub use audit::{AuditLog, AuditEntry, AuditLevel};
pub use transport::{SecureMcpTransport, McpHandler, McpRouter, SecureStdioAdapter, SecureTransportBuilder};

use hyperphysics_dilithium::{DilithiumError, SecurityLevel};

/// MCP authentication error types
#[derive(Debug, thiserror::Error)]
pub enum McpAuthError {
    #[error("Cryptographic operation failed: {0}")]
    CryptoError(#[from] DilithiumError),

    #[error("Request signature invalid")]
    InvalidSignature,

    #[error("Request expired (timestamp: {timestamp}, max_age: {max_age_secs}s)")]
    RequestExpired {
        timestamp: i64,
        max_age_secs: u64,
    },

    #[error("Nonce already used: {nonce}")]
    NonceReused { nonce: String },

    #[error("Client not authorized: {client_id}")]
    ClientNotAuthorized { client_id: String },

    #[error("Client not found: {client_id}")]
    ClientNotFound { client_id: String },

    #[error("Tool not permitted for client: {tool} (client: {client_id})")]
    ToolNotPermitted { tool: String, client_id: String },

    #[error("Request rate limit exceeded: {requests_per_minute}/min (limit: {limit})")]
    RateLimitExceeded {
        requests_per_minute: u32,
        limit: u32,
    },

    #[error("Malformed request: {reason}")]
    MalformedRequest { reason: String },

    #[error("Injection pattern detected: {pattern}")]
    InjectionDetected { pattern: String },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for MCP authentication operations
pub type McpAuthResult<T> = std::result::Result<T, McpAuthError>;

/// Default security level for MCP authentication
pub const DEFAULT_SECURITY_LEVEL: SecurityLevel = SecurityLevel::High;

/// Maximum request age in seconds (prevents replay attacks)
pub const DEFAULT_MAX_REQUEST_AGE_SECS: u64 = 30;

/// Default rate limit (requests per minute per client)
pub const DEFAULT_RATE_LIMIT: u32 = 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = McpAuthError::ClientNotAuthorized {
            client_id: "test-client".to_string(),
        };
        assert!(err.to_string().contains("test-client"));
    }
}
