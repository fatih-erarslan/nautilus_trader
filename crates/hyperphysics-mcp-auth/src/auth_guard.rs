//! Core MCP authentication guard

use crate::{
    McpAuthError, McpAuthResult, SecurityConfig,
    McpRequest, SignedMcpRequest, McpResponse,
    AuthenticatedClient, ClientCredentials, ClientPermissions,
    NonceManager, AuditLog, AuditEntry, AuditLevel,
};
use hyperphysics_dilithium::DilithiumKeypair;
use dashmap::DashMap;
use chrono::Utc;
use regex::Regex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Rate limiter entry
struct RateLimitEntry {
    count: u32,
    window_start: Instant,
}

/// Core MCP authentication guard
///
/// Provides cryptographic authentication, injection detection, and
/// authorization for all MCP tool calls.
pub struct McpAuthGuard {
    /// Security configuration
    config: SecurityConfig,

    /// Server keypair for signing responses
    server_keypair: DilithiumKeypair,

    /// Registered clients
    clients: DashMap<String, AuthenticatedClient>,

    /// Nonce manager for replay protection
    nonce_manager: NonceManager,

    /// Rate limiter per client
    rate_limiter: DashMap<String, RateLimitEntry>,

    /// Compiled injection patterns
    injection_patterns: Vec<Regex>,

    /// Audit log
    audit_log: Arc<AuditLog>,
}

impl McpAuthGuard {
    /// Create new authentication guard
    pub fn new(config: SecurityConfig) -> McpAuthResult<Self> {
        let server_keypair = DilithiumKeypair::generate(config.security_level)?;

        let nonce_manager = NonceManager::new(
            config.nonce_cache_ttl_secs,
            config.max_nonce_cache_size,
        );

        // Compile injection patterns
        let injection_patterns: Vec<Regex> = config.injection_patterns
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect();

        let audit_log = if config.audit_logging {
            Arc::new(AuditLog::new(10_000, AuditLevel::Info))
        } else {
            Arc::new(AuditLog::disabled())
        };

        Ok(Self {
            config,
            server_keypair,
            clients: DashMap::new(),
            nonce_manager,
            rate_limiter: DashMap::new(),
            injection_patterns,
            audit_log,
        })
    }

    /// Create with default security configuration
    pub fn default_security() -> McpAuthResult<Self> {
        Self::new(SecurityConfig::default())
    }

    /// Create with maximum security configuration
    pub fn maximum_security() -> McpAuthResult<Self> {
        Self::new(SecurityConfig::maximum_security())
    }

    /// Get the audit log
    pub fn audit_log(&self) -> &Arc<AuditLog> {
        &self.audit_log
    }

    // ==================== Client Management ====================

    /// Register a new client
    pub fn register_client(&self, name: impl Into<String>) -> McpAuthResult<String> {
        let name = name.into();
        let id = generate_client_id(&name);

        let client = AuthenticatedClient::new(
            &id,
            &name,
            self.config.security_level,
            ClientPermissions::default(),
        )?;

        self.audit_log.log_sync(
            AuditEntry::new(AuditLevel::Security, "client", format!("Client registered: {}", name))
                .with_client(&id)
        );

        self.clients.insert(id.clone(), client);
        Ok(id)
    }

    /// Register client with specific credentials
    pub fn register_client_with_credentials(
        &self,
        credentials: ClientCredentials,
    ) -> McpAuthResult<String> {
        let client = AuthenticatedClient::new(
            &credentials.id,
            &credentials.name,
            credentials.security_level,
            credentials.permissions,
        )?;

        self.audit_log.log_sync(
            AuditEntry::new(
                AuditLevel::Security,
                "client",
                format!("Client registered with credentials: {}", credentials.name)
            ).with_client(&credentials.id)
        );

        self.clients.insert(credentials.id.clone(), client);
        Ok(credentials.id)
    }

    /// Get client by ID
    pub fn get_client(&self, client_id: &str) -> Option<dashmap::mapref::one::Ref<String, AuthenticatedClient>> {
        self.clients.get(client_id)
    }

    /// Revoke client access
    pub fn revoke_client(&self, client_id: &str) -> McpAuthResult<()> {
        if let Some(mut client) = self.clients.get_mut(client_id) {
            client.is_active = false;

            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::Security, "client", "Client access revoked")
                    .with_client(client_id)
            );

            Ok(())
        } else {
            Err(McpAuthError::ClientNotFound {
                client_id: client_id.to_string(),
            })
        }
    }

    /// Remove client entirely
    pub fn remove_client(&self, client_id: &str) -> McpAuthResult<()> {
        if self.clients.remove(client_id).is_some() {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::Security, "client", "Client removed")
                    .with_client(client_id)
            );
            Ok(())
        } else {
            Err(McpAuthError::ClientNotFound {
                client_id: client_id.to_string(),
            })
        }
    }

    // ==================== Request Signing ====================

    /// Sign a request for a client
    pub fn sign_request(
        &self,
        client_id: &str,
        request: McpRequest,
    ) -> McpAuthResult<SignedMcpRequest> {
        let client = self.clients.get(client_id)
            .ok_or_else(|| McpAuthError::ClientNotFound {
                client_id: client_id.to_string(),
            })?;

        if !client.is_active {
            return Err(McpAuthError::ClientNotAuthorized {
                client_id: client_id.to_string(),
            });
        }

        // Get canonical bytes for signing
        let canonical_bytes = request.canonical_bytes();

        // Sign with client's keypair
        let signature = client.keypair.sign(&canonical_bytes)?;

        Ok(SignedMcpRequest::new(
            request,
            client_id.to_string(),
            signature,
        ))
    }

    // ==================== Request Verification ====================

    /// Verify and authorize a signed request
    pub fn verify_and_authorize(&self, signed_request: &SignedMcpRequest) -> McpAuthResult<()> {
        let request_id = &signed_request.request.id;
        let client_id = &signed_request.client_id;

        // 1. Check content hash integrity
        if !signed_request.verify_content_hash() {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::Attack, "integrity", "Content hash mismatch")
                    .with_client(client_id)
                    .with_request(request_id)
            );
            return Err(McpAuthError::MalformedRequest {
                reason: "Content hash verification failed".to_string(),
            });
        }

        // 2. Get client and verify active
        let client = self.clients.get(client_id)
            .ok_or_else(|| {
                self.audit_log.log_sync(
                    AuditEntry::new(AuditLevel::AuthFailure, "auth", "Unknown client")
                        .with_client(client_id)
                        .with_request(request_id)
                );
                McpAuthError::ClientNotFound {
                    client_id: client_id.to_string(),
                }
            })?;

        if !client.is_active {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::AuthFailure, "auth", "Client revoked")
                    .with_client(client_id)
                    .with_request(request_id)
            );
            return Err(McpAuthError::ClientNotAuthorized {
                client_id: client_id.to_string(),
            });
        }

        // 3. Verify signature
        let canonical_bytes = signed_request.request.canonical_bytes();
        let signature_valid = client.keypair.verify(&canonical_bytes, &signed_request.signature)?;

        if !signature_valid {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::Attack, "auth", "Invalid signature")
                    .with_client(client_id)
                    .with_request(request_id)
            );
            return Err(McpAuthError::InvalidSignature);
        }

        // 4. Check request timestamp (replay protection)
        let request_age = Utc::now()
            .signed_duration_since(signed_request.request.timestamp)
            .num_seconds()
            .abs() as u64;

        if request_age > self.config.max_request_age_secs {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::Warning, "replay", "Request expired")
                    .with_client(client_id)
                    .with_request(request_id)
            );
            return Err(McpAuthError::RequestExpired {
                timestamp: signed_request.request.timestamp.timestamp(),
                max_age_secs: self.config.max_request_age_secs,
            });
        }

        // 5. Check nonce (replay protection)
        self.nonce_manager.consume(&signed_request.request.nonce)?;

        // 6. Check rate limit
        self.check_rate_limit(client_id, client.permissions.rate_limit)?;

        // 7. Check tool permissions
        let tool = &signed_request.request.method;
        if !self.is_tool_allowed(client_id, tool, &client) {
            self.audit_log.log_sync(
                AuditEntry::new(AuditLevel::AuthFailure, "access", format!("Tool not permitted: {}", tool))
                    .with_client(client_id)
                    .with_request(request_id)
                    .with_tool(tool)
            );
            return Err(McpAuthError::ToolNotPermitted {
                tool: tool.to_string(),
                client_id: client_id.to_string(),
            });
        }

        // 8. Scan for injection patterns
        self.scan_for_injection(&signed_request.request, client_id)?;

        // Log successful access
        self.audit_log.log_sync(
            AuditEntry::new(AuditLevel::Info, "access", format!("Tool access granted: {}", tool))
                .with_client(client_id)
                .with_request(request_id)
                .with_tool(tool)
        );

        // Update client activity
        drop(client);
        if let Some(mut client) = self.clients.get_mut(client_id) {
            client.touch();
        }

        Ok(())
    }

    /// Quick verification without full authorization (for performance)
    pub fn quick_verify(&self, signed_request: &SignedMcpRequest) -> McpAuthResult<bool> {
        // Just verify signature, skip other checks
        let client = self.clients.get(&signed_request.client_id)
            .ok_or_else(|| McpAuthError::ClientNotFound {
                client_id: signed_request.client_id.clone(),
            })?;

        let canonical_bytes = signed_request.request.canonical_bytes();
        client.keypair.verify(&canonical_bytes, &signed_request.signature)
            .map_err(McpAuthError::from)
    }

    // ==================== Response Signing ====================

    /// Sign a response from the server
    pub fn sign_response(&self, mut response: McpResponse) -> McpAuthResult<McpResponse> {
        let response_bytes = serde_json::to_vec(&response)
            .map_err(|e| McpAuthError::SerializationError(e.to_string()))?;

        let signature = self.server_keypair.sign(&response_bytes)?;
        response.signature = Some(signature);

        Ok(response)
    }

    // ==================== Helper Methods ====================

    /// Check if a tool is allowed for this client
    fn is_tool_allowed(&self, _client_id: &str, tool: &str, client: &AuthenticatedClient) -> bool {
        // Check global blacklist first
        if self.config.blocked_tools.contains(tool) {
            return false;
        }

        // Check global whitelist (if defined)
        if let Some(ref allowed) = self.config.allowed_tools {
            if !allowed.contains(tool) {
                return false;
            }
        }

        // Check client-specific permissions if enabled
        if self.config.per_client_tool_permissions {
            return client.can_access_tool(tool);
        }

        true
    }

    /// Check rate limit for client
    fn check_rate_limit(&self, client_id: &str, limit: u32) -> McpAuthResult<()> {
        let now = Instant::now();
        let window = Duration::from_secs(60);

        let mut entry = self.rate_limiter.entry(client_id.to_string())
            .or_insert(RateLimitEntry {
                count: 0,
                window_start: now,
            });

        // Reset window if expired
        if now.duration_since(entry.window_start) > window {
            entry.count = 0;
            entry.window_start = now;
        }

        entry.count += 1;

        if entry.count > limit {
            self.audit_log.log_sync(
                AuditEntry::new(
                    AuditLevel::Warning,
                    "rate_limit",
                    format!("Rate limit exceeded: {}/{}", entry.count, limit)
                ).with_client(client_id)
            );
            return Err(McpAuthError::RateLimitExceeded {
                requests_per_minute: entry.count,
                limit,
            });
        }

        Ok(())
    }

    /// Scan request for injection patterns
    fn scan_for_injection(&self, request: &McpRequest, client_id: &str) -> McpAuthResult<()> {
        // Convert params to string for scanning
        let params_str = serde_json::to_string(&request.params)
            .unwrap_or_default();

        // Also check method name
        let combined = format!("{} {}", request.method, params_str);

        for pattern in &self.injection_patterns {
            if pattern.is_match(&combined) {
                self.audit_log.log_sync(
                    AuditEntry::new(
                        AuditLevel::Attack,
                        "injection",
                        format!("Injection pattern detected: {}", pattern.as_str())
                    )
                        .with_client(client_id)
                        .with_request(&request.id)
                        .with_context(serde_json::json!({
                            "pattern": pattern.as_str(),
                            "method": &request.method,
                        }))
                );

                return Err(McpAuthError::InjectionDetected {
                    pattern: pattern.as_str().to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get server public key bytes
    pub fn server_public_key(&self) -> Vec<u8> {
        self.server_keypair.public_key_bytes().to_vec()
    }

    /// Get number of registered clients
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Get security configuration
    pub fn config(&self) -> &SecurityConfig {
        &self.config
    }
}

/// Generate deterministic client ID from name
fn generate_client_id(name: &str) -> String {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let hash = blake3::hash(format!("{}:{}", name, timestamp).as_bytes());
    format!("client-{}", &hash.to_hex()[..16])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_guard() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        assert!(guard.client_count() == 0);
    }

    #[test]
    fn test_register_client() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        let client_id = guard.register_client("test-client")
            .expect("Failed to register client");

        assert!(client_id.starts_with("client-"));
        assert_eq!(guard.client_count(), 1);
    }

    #[test]
    fn test_sign_and_verify() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        let client_id = guard.register_client("test-client")
            .expect("Failed to register client");

        let request = McpRequest::new("test_tool", serde_json::json!({"key": "value"}));
        let signed = guard.sign_request(&client_id, request)
            .expect("Failed to sign request");

        // Verification should pass
        assert!(guard.verify_and_authorize(&signed).is_ok());
    }

    #[test]
    fn test_unknown_client_rejected() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        let client_id = guard.register_client("test-client")
            .expect("Failed to register client");

        let request = McpRequest::new("test_tool", serde_json::json!({}));
        let mut signed = guard.sign_request(&client_id, request)
            .expect("Failed to sign request");

        // Change client ID
        signed.client_id = "unknown-client".to_string();

        // Should be rejected
        assert!(matches!(
            guard.verify_and_authorize(&signed),
            Err(McpAuthError::ClientNotFound { .. })
        ));
    }

    #[test]
    fn test_revoked_client_rejected() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        let client_id = guard.register_client("test-client")
            .expect("Failed to register client");

        // Revoke the client
        guard.revoke_client(&client_id).expect("Failed to revoke");

        // Try to sign - should fail
        let request = McpRequest::new("test_tool", serde_json::json!({}));
        assert!(matches!(
            guard.sign_request(&client_id, request),
            Err(McpAuthError::ClientNotAuthorized { .. })
        ));
    }

    #[test]
    fn test_nonce_replay_rejected() {
        let guard = McpAuthGuard::default_security()
            .expect("Failed to create guard");

        let client_id = guard.register_client("test-client")
            .expect("Failed to register client");

        let request = McpRequest::new("test_tool", serde_json::json!({}));
        let signed = guard.sign_request(&client_id, request)
            .expect("Failed to sign request");

        // First verification should pass
        assert!(guard.verify_and_authorize(&signed).is_ok());

        // Second verification with same nonce should fail
        assert!(matches!(
            guard.verify_and_authorize(&signed),
            Err(McpAuthError::NonceReused { .. })
        ));
    }
}
