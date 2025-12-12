//! # Dilithium ML-DSA-65 Authentication Layer
//!
//! Post-quantum cryptographic authentication for CQGS MCP plugin.
//! Security-first architecture following HyperPhysics dilithium_mcp pattern.
//!
//! ## Features
//!
//! - **Post-Quantum Security**: Dilithium ML-DSA-65 (NIST FIPS 204)
//! - **Nonce-Based Replay Protection**: Prevents message replay attacks
//! - **Client Quota Management**: Rate limiting and resource allocation
//! - **BLAKE3 Hashing**: Fast, secure cryptographic hashing
//!
//! ## References
//!
//! - NIST FIPS 204: Module-Lattice-Based Digital Signature Standard
//! - Ducas et al. (2018): CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme
//! - Avanzi et al. (2020): CRYSTALS-Kyber Algorithm Specifications

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use parking_lot::RwLock;

#[cfg(feature = "dilithium")]
use pqc_dilithium::{Keypair, PublicKey, DetachedSignature, PUBLICKEYBYTES, SECRETKEYBYTES, SIGNATUREBYTES};

#[cfg(feature = "dilithium")]
use blake3::Hash;

// ============================================================================
// Key Pair Management
// ============================================================================

/// Dilithium ML-DSA-65 key pair wrapper
#[derive(Clone)]
pub struct DilithiumKeyPair {
    #[cfg(feature = "dilithium")]
    public_key: [u8; PUBLICKEYBYTES],

    #[cfg(feature = "dilithium")]
    secret_key: [u8; SECRETKEYBYTES],

    /// Public key ID for tracking
    pub public_key_id: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Expiration timestamp (optional)
    pub expires_at: Option<DateTime<Utc>>,
}

impl DilithiumKeyPair {
    /// Generate new Dilithium ML-DSA-65 key pair
    #[cfg(feature = "dilithium")]
    pub fn generate() -> Result<Self> {
        let keypair = Keypair::generate();
        let public_key_id = hex::encode(&keypair.public[..32]); // First 32 bytes as ID

        Ok(Self {
            public_key: keypair.public,
            secret_key: keypair.secret,
            public_key_id,
            created_at: Utc::now(),
            expires_at: None,
        })
    }

    /// Generate with expiration
    #[cfg(feature = "dilithium")]
    pub fn generate_with_expiration(days: i64) -> Result<Self> {
        let mut key = Self::generate()?;
        key.expires_at = Some(Utc::now() + Duration::days(days));
        Ok(key)
    }

    /// Get public key hex
    #[cfg(feature = "dilithium")]
    pub fn public_key_hex(&self) -> String {
        hex::encode(&self.public_key[..])
    }

    /// Get secret key hex
    #[cfg(feature = "dilithium")]
    pub fn secret_key_hex(&self) -> String {
        hex::encode(&self.secret_key[..])
    }

    /// Check if key pair is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// Sign message with Dilithium
    #[cfg(feature = "dilithium")]
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let sig = pqc_dilithium::detached_sign(message, &self.secret_key);
        Ok(sig.to_vec())
    }

    /// Verify signature with Dilithium
    #[cfg(feature = "dilithium")]
    pub fn verify(public_key: &[u8; PUBLICKEYBYTES], signature: &[u8], message: &[u8]) -> Result<bool> {
        if signature.len() != SIGNATUREBYTES {
            return Ok(false);
        }

        let mut sig_array = [0u8; SIGNATUREBYTES];
        sig_array.copy_from_slice(signature);
        let sig = DetachedSignature::new(sig_array);

        Ok(pqc_dilithium::verify_detached_signature(&sig, message, public_key).is_ok())
    }
}

// Stub implementation when dilithium feature is disabled
#[cfg(not(feature = "dilithium"))]
impl DilithiumKeyPair {
    pub fn generate() -> Result<Self> {
        Ok(Self {
            public_key_id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            expires_at: None,
        })
    }

    pub fn generate_with_expiration(days: i64) -> Result<Self> {
        let mut key = Self::generate()?;
        key.expires_at = Some(Utc::now() + Duration::days(days));
        Ok(key)
    }

    pub fn sign(&self, _message: &[u8]) -> Result<Vec<u8>> {
        Ok(vec![0; 64]) // Stub signature
    }
}

// ============================================================================
// Client Credentials
// ============================================================================

/// Client authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCredentials {
    /// Unique client ID
    pub client_id: String,

    /// Client name
    pub name: String,

    /// Public key (hex-encoded)
    pub public_key: String,

    /// Requested capabilities
    pub capabilities: Vec<String>,

    /// Resource quotas
    pub quotas: ClientQuotas,

    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
}

/// Client resource quotas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientQuotas {
    /// Maximum requests per day
    pub daily_requests: u32,

    /// Maximum tokens per day
    pub daily_tokens: u64,

    /// Maximum concurrent requests
    pub max_concurrent: u32,

    /// Rate limit per minute
    pub rate_limit_per_minute: u32,
}

impl Default for ClientQuotas {
    fn default() -> Self {
        Self {
            daily_requests: 10_000,
            daily_tokens: 1_000_000,
            max_concurrent: 10,
            rate_limit_per_minute: 60,
        }
    }
}

// ============================================================================
// Authentication Token
// ============================================================================

/// Authenticated session token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Unique token ID
    pub token_id: Uuid,

    /// Client ID
    pub client_id: String,

    /// Authorized capabilities
    pub capabilities: Vec<String>,

    /// Issue timestamp
    pub issued_at: DateTime<Utc>,

    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,

    /// Nonce for replay protection
    pub nonce: String,
}

impl AuthToken {
    /// Create new auth token
    pub fn new(client_id: String, capabilities: Vec<String>, duration_hours: i64) -> Self {
        let now = Utc::now();
        Self {
            token_id: Uuid::new_v4(),
            client_id,
            capabilities,
            issued_at: now,
            expires_at: now + Duration::hours(duration_hours),
            nonce: Uuid::new_v4().to_string(),
        }
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if token has capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.contains(&capability.to_string())
    }

    /// Validate token
    pub fn validate(&self) -> Result<()> {
        if self.is_expired() {
            anyhow::bail!("Token expired");
        }
        Ok(())
    }
}

// ============================================================================
// Dilithium Authentication System
// ============================================================================

/// Dilithium ML-DSA-65 authentication system
pub struct DilithiumAuth {
    /// Server key pair
    server_keypair: DilithiumKeyPair,

    /// Registered clients
    clients: RwLock<HashMap<String, ClientCredentials>>,

    /// Active tokens
    tokens: RwLock<HashMap<Uuid, AuthToken>>,

    /// Used nonces (replay protection)
    nonces: RwLock<HashMap<String, DateTime<Utc>>>,
}

impl DilithiumAuth {
    /// Create new authentication system
    pub fn new() -> Result<Self> {
        let server_keypair = DilithiumKeyPair::generate()
            .context("Failed to generate server key pair")?;

        Ok(Self {
            server_keypair,
            clients: RwLock::new(HashMap::new()),
            tokens: RwLock::new(HashMap::new()),
            nonces: RwLock::new(HashMap::new()),
        })
    }

    /// Register new client
    pub fn register_client(
        &self,
        name: String,
        public_key: String,
        capabilities: Vec<String>,
        quotas: Option<ClientQuotas>,
    ) -> Result<ClientCredentials> {
        let client_id = Uuid::new_v4().to_string();

        let credentials = ClientCredentials {
            client_id: client_id.clone(),
            name,
            public_key,
            capabilities,
            quotas: quotas.unwrap_or_default(),
            registered_at: Utc::now(),
        };

        self.clients.write().insert(client_id, credentials.clone());

        Ok(credentials)
    }

    /// Authenticate client and issue token
    #[cfg(feature = "dilithium")]
    pub fn authenticate(
        &self,
        client_id: &str,
        public_key_hex: &str,
        signature_hex: &str,
        message: &[u8],
        nonce: &str,
    ) -> Result<AuthToken> {
        // Check if nonce was already used
        if self.nonces.read().contains_key(nonce) {
            anyhow::bail!("Nonce already used (replay attack detected)");
        }

        // Get client credentials
        let clients = self.clients.read();
        let client = clients.get(client_id)
            .context("Client not found")?;

        // Verify public key matches
        if client.public_key != public_key_hex {
            anyhow::bail!("Public key mismatch");
        }

        // Decode public key and signature
        let public_key_bytes = hex::decode(public_key_hex)?;
        let signature_bytes = hex::decode(signature_hex)?;

        if public_key_bytes.len() != PUBLICKEYBYTES {
            anyhow::bail!("Invalid public key length");
        }

        let mut public_key = [0u8; PUBLICKEYBYTES];
        public_key.copy_from_slice(&public_key_bytes);

        // Verify signature
        let valid = DilithiumKeyPair::verify(&public_key, &signature_bytes, message)?;
        if !valid {
            anyhow::bail!("Invalid signature");
        }

        // Mark nonce as used
        self.nonces.write().insert(nonce.to_string(), Utc::now());

        // Create auth token
        let token = AuthToken::new(
            client_id.to_string(),
            client.capabilities.clone(),
            24, // 24 hour expiration
        );

        self.tokens.write().insert(token.token_id, token.clone());

        Ok(token)
    }

    /// Validate auth token
    pub fn validate_token(&self, token_id: Uuid) -> Result<AuthToken> {
        let tokens = self.tokens.read();
        let token = tokens.get(&token_id)
            .context("Token not found")?;

        token.validate()?;

        Ok(token.clone())
    }

    /// Revoke auth token
    pub fn revoke_token(&self, token_id: Uuid) -> Result<()> {
        self.tokens.write().remove(&token_id);
        Ok(())
    }

    /// Clean up expired tokens and nonces
    pub fn cleanup_expired(&self) {
        let now = Utc::now();

        // Remove expired tokens
        self.tokens.write().retain(|_, token| {
            token.expires_at > now
        });

        // Remove old nonces (keep for 24 hours)
        self.nonces.write().retain(|_, timestamp| {
            now.signed_duration_since(*timestamp).num_hours() < 24
        });
    }

    /// Get server public key
    #[cfg(feature = "dilithium")]
    pub fn server_public_key(&self) -> String {
        self.server_keypair.public_key_hex()
    }

    /// Get client count
    pub fn client_count(&self) -> usize {
        self.clients.read().len()
    }

    /// Get active token count
    pub fn active_token_count(&self) -> usize {
        self.tokens.read().len()
    }
}

impl Default for DilithiumAuth {
    fn default() -> Self {
        Self::new().expect("Failed to create DilithiumAuth")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = DilithiumKeyPair::generate().unwrap();
        assert!(!keypair.public_key_id.is_empty());
        assert!(!keypair.is_expired());
    }

    #[test]
    fn test_keypair_expiration() {
        let keypair = DilithiumKeyPair::generate_with_expiration(30).unwrap();
        assert!(!keypair.is_expired());
        assert!(keypair.expires_at.is_some());
    }

    #[test]
    fn test_auth_token_creation() {
        let token = AuthToken::new(
            "client123".to_string(),
            vec!["read".to_string(), "write".to_string()],
            24,
        );

        assert_eq!(token.client_id, "client123");
        assert!(token.has_capability("read"));
        assert!(token.has_capability("write"));
        assert!(!token.has_capability("admin"));
        assert!(!token.is_expired());
    }

    #[test]
    fn test_dilithium_auth_registration() {
        let auth = DilithiumAuth::new().unwrap();

        let credentials = auth.register_client(
            "Test Client".to_string(),
            "public_key_hex".to_string(),
            vec!["sentinel_execute".to_string()],
            None,
        ).unwrap();

        assert_eq!(credentials.name, "Test Client");
        assert_eq!(auth.client_count(), 1);
    }
}
