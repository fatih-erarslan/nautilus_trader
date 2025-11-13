//! # Ed25519 Agent Identity Module
//!
//! Provides cryptographic identity management for agents using Ed25519 signatures.
//! Each agent maintains a keypair for signing payment mandates and messages.

use ed25519_dalek::{Signer, Verifier, SigningKey, VerifyingKey, Signature};
use serde::{Deserialize, Serialize};

/// Agent cryptographic identity with Ed25519 keypair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Unique agent identifier
    pub agent_id: String,

    /// Hex-encoded Ed25519 public key (32 bytes)
    pub public_key: String,

    /// Secret key (only available for identity owner)
    #[serde(skip)]
    secret_key: Option<SigningKey>,
}

impl AgentIdentity {
    /// Generate new Ed25519 keypair for agent
    ///
    /// # Arguments
    /// * `agent_id` - Unique identifier for the agent
    ///
    /// # Example
    /// ```rust,no_run
    /// use hyperphysics_core::crypto::identity::AgentIdentity;
    ///
    /// let identity = AgentIdentity::generate("agent-001".to_string());
    /// ```
    pub fn generate(agent_id: String) -> Self {
        let signing_key = SigningKey::from_bytes(&rand::random::<[u8; 32]>());
        let verifying_key = signing_key.verifying_key();

        Self {
            agent_id,
            public_key: hex::encode(verifying_key.to_bytes()),
            secret_key: Some(signing_key),
        }
    }

    /// Create identity from existing public key (for verification only)
    ///
    /// # Arguments
    /// * `agent_id` - Agent identifier
    /// * `public_key` - Hex-encoded public key
    pub fn from_public_key(agent_id: String, public_key: String) -> Result<Self, String> {
        // Validate public key format
        hex::decode(&public_key)
            .map_err(|e| format!("Invalid hex encoding: {}", e))?;

        Ok(Self {
            agent_id,
            public_key,
            secret_key: None,
        })
    }

    /// Sign a message using agent's secret key
    ///
    /// # Arguments
    /// * `message` - Bytes to sign
    ///
    /// # Returns
    /// Ed25519 signature or error if no secret key available
    pub fn sign(&self, message: &[u8]) -> Result<Signature, String> {
        let signing_key = self.secret_key.as_ref()
            .ok_or("No secret key available for signing")?;

        Ok(signing_key.sign(message))
    }

    /// Verify signature against a public key
    ///
    /// # Arguments
    /// * `public_key` - Hex-encoded public key
    /// * `message` - Original message bytes
    /// * `signature` - Ed25519 signature to verify
    ///
    /// # Returns
    /// Ok(()) if signature is valid, Err otherwise
    pub fn verify(
        public_key: &str,
        message: &[u8],
        signature: &Signature,
    ) -> Result<(), String> {
        let public_bytes = hex::decode(public_key)
            .map_err(|e| format!("Invalid public key hex: {}", e))?;

        let public_key_array: [u8; 32] = public_bytes
            .try_into()
            .map_err(|_| "Public key must be exactly 32 bytes".to_string())?;

        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|e| format!("Invalid public key: {}", e))?;

        verifying_key.verify(message, signature)
            .map_err(|e| format!("Signature verification failed: {}", e))
    }

    /// Export public key in hex format
    pub fn export_public_key(&self) -> String {
        self.public_key.clone()
    }

    /// Get agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Check if identity has signing capability
    pub fn can_sign(&self) -> bool {
        self.secret_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_identity() {
        let identity = AgentIdentity::generate("test-agent".to_string());
        assert_eq!(identity.agent_id, "test-agent");
        assert!(identity.can_sign());
        assert_eq!(hex::decode(&identity.public_key).unwrap().len(), 32);
    }

    #[test]
    fn test_sign_and_verify() {
        let identity = AgentIdentity::generate("signer".to_string());
        let message = b"payment mandate data";

        let signature = identity.sign(message).unwrap();
        let result = AgentIdentity::verify(&identity.public_key, message, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_invalid_signature() {
        let identity1 = AgentIdentity::generate("agent1".to_string());
        let identity2 = AgentIdentity::generate("agent2".to_string());

        let message = b"test message";
        let signature = identity1.sign(message).unwrap();

        // Verify with wrong public key should fail
        let result = AgentIdentity::verify(&identity2.public_key, message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_public_key() {
        let identity = AgentIdentity::generate("original".to_string());
        let public_key = identity.export_public_key();

        let public_only = AgentIdentity::from_public_key(
            "replica".to_string(),
            public_key.clone()
        ).unwrap();

        assert_eq!(public_only.public_key, public_key);
        assert!(!public_only.can_sign());
    }
}
