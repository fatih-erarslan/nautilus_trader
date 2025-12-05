//! Authenticated client management

use hyperphysics_dilithium::{DilithiumKeypair, DilithiumResult, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use chrono::{DateTime, Utc};

/// Authenticated MCP client with cryptographic credentials
pub struct AuthenticatedClient {
    /// Unique client identifier
    pub id: String,

    /// Human-readable client name
    pub name: String,

    /// Client's Dilithium keypair for signing requests
    pub keypair: DilithiumKeypair,

    /// Permissions assigned to this client
    pub permissions: ClientPermissions,

    /// Client creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activity timestamp
    pub last_active: DateTime<Utc>,

    /// Whether the client is currently active
    pub is_active: bool,
}

impl AuthenticatedClient {
    /// Create new authenticated client
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        security_level: SecurityLevel,
        permissions: ClientPermissions,
    ) -> DilithiumResult<Self> {
        let keypair = DilithiumKeypair::generate(security_level)?;
        let now = Utc::now();

        Ok(Self {
            id: id.into(),
            name: name.into(),
            keypair,
            permissions,
            created_at: now,
            last_active: now,
            is_active: true,
        })
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }

    /// Check if client can access a specific tool
    pub fn can_access_tool(&self, tool: &str) -> bool {
        if !self.is_active {
            return false;
        }

        // Check blacklist first
        if self.permissions.blocked_tools.contains(tool) {
            return false;
        }

        // Check whitelist (if defined)
        if let Some(ref allowed) = self.permissions.allowed_tools {
            return allowed.contains(tool);
        }

        // If no whitelist, allow all non-blocked tools
        true
    }

    /// Get public credentials (safe to share)
    pub fn public_credentials(&self) -> PublicClientCredentials {
        PublicClientCredentials {
            id: self.id.clone(),
            name: self.name.clone(),
            public_key_bytes: self.keypair.public_key_bytes().to_vec(),
            created_at: self.created_at,
        }
    }
}

/// Client permissions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientPermissions {
    /// Tools this client can access (whitelist, None = all allowed)
    pub allowed_tools: Option<HashSet<String>>,

    /// Tools this client cannot access (blacklist, checked before whitelist)
    pub blocked_tools: HashSet<String>,

    /// Maximum requests per minute
    pub rate_limit: u32,

    /// Can this client sign responses (server mode)
    pub can_sign_responses: bool,

    /// Custom permission flags
    pub flags: HashSet<String>,
}

impl Default for ClientPermissions {
    fn default() -> Self {
        Self {
            allowed_tools: None,
            blocked_tools: HashSet::new(),
            rate_limit: 100,
            can_sign_responses: false,
            flags: HashSet::new(),
        }
    }
}

impl ClientPermissions {
    /// Create permissions allowing specific tools only
    pub fn allow_only(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_tools: Some(tools.into_iter().map(|t| t.into()).collect()),
            ..Default::default()
        }
    }

    /// Create permissions blocking specific tools
    pub fn block(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            blocked_tools: tools.into_iter().map(|t| t.into()).collect(),
            ..Default::default()
        }
    }

    /// Create read-only permissions (common safe tools)
    pub fn read_only() -> Self {
        Self::allow_only([
            "vector_db_search",
            "vector_db_list",
            "gnn_forward",
            "gnn_attention_scores",
            "get_status",
            "get_metrics",
        ])
    }

    /// Create full access permissions
    pub fn full_access() -> Self {
        Self {
            allowed_tools: None,
            blocked_tools: HashSet::new(),
            rate_limit: 1000,
            can_sign_responses: true,
            flags: HashSet::new(),
        }
    }

    /// Add a permission flag
    pub fn with_flag(mut self, flag: impl Into<String>) -> Self {
        self.flags.insert(flag.into());
        self
    }

    /// Set custom rate limit
    pub fn with_rate_limit(mut self, limit: u32) -> Self {
        self.rate_limit = limit;
        self
    }
}

/// Client credentials for registration/authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCredentials {
    /// Client identifier
    pub id: String,

    /// Client name
    pub name: String,

    /// Security level to use
    pub security_level: SecurityLevel,

    /// Requested permissions
    pub permissions: ClientPermissions,
}

impl ClientCredentials {
    /// Create new client credentials
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            security_level: SecurityLevel::High,
            permissions: ClientPermissions::default(),
        }
    }

    /// Set security level
    pub fn with_security_level(mut self, level: SecurityLevel) -> Self {
        self.security_level = level;
        self
    }

    /// Set permissions
    pub fn with_permissions(mut self, permissions: ClientPermissions) -> Self {
        self.permissions = permissions;
        self
    }
}

/// Public client credentials (safe to transmit)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicClientCredentials {
    /// Client identifier
    pub id: String,

    /// Client name
    pub name: String,

    /// Dilithium public key bytes
    pub public_key_bytes: Vec<u8>,

    /// Registration timestamp
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_client() {
        let client = AuthenticatedClient::new(
            "test-client",
            "Test Client",
            SecurityLevel::Standard,
            ClientPermissions::default(),
        ).expect("Failed to create client");

        assert_eq!(client.id, "test-client");
        assert!(client.is_active);
    }

    #[test]
    fn test_tool_access() {
        let permissions = ClientPermissions::allow_only(["tool_a", "tool_b"]);
        let client = AuthenticatedClient::new(
            "restricted",
            "Restricted Client",
            SecurityLevel::Standard,
            permissions,
        ).expect("Failed to create client");

        assert!(client.can_access_tool("tool_a"));
        assert!(client.can_access_tool("tool_b"));
        assert!(!client.can_access_tool("tool_c"));
    }

    #[test]
    fn test_blacklist_precedence() {
        let mut permissions = ClientPermissions::default();
        permissions.allowed_tools = Some(["tool_a", "tool_b"].iter().map(|s| s.to_string()).collect());
        permissions.blocked_tools.insert("tool_a".to_string());

        let client = AuthenticatedClient::new(
            "test",
            "Test",
            SecurityLevel::Standard,
            permissions,
        ).expect("Failed to create client");

        // Blacklist takes precedence
        assert!(!client.can_access_tool("tool_a"));
        assert!(client.can_access_tool("tool_b"));
    }
}
