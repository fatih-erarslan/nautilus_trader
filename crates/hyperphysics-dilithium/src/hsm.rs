//! Hardware Security Module (HSM) Integration
//!
//! Provides trait-based abstraction for hardware security module backends,
//! enabling production deployments with FIPS 140-3 compliant key storage.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   HSM ABSTRACTION LAYER                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  Application Code                                           │
//! │       │                                                     │
//! │       ▼                                                     │
//! │  ┌──────────────┐                                          │
//! │  │ HsmProvider  │◄─── Trait-based abstraction              │
//! │  └──────────────┘                                          │
//! │       │                                                     │
//! │       ├───────────────┬───────────────┬──────────────┐     │
//! │       ▼               ▼               ▼              ▼     │
//! │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
//! │  │Software │    │  TPM    │    │  HSM    │    │  Cloud  │ │
//! │  │ Backend │    │ Backend │    │ Backend │    │   KMS   │ │
//! │  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security
//!
//! - Keys never leave HSM in plaintext
//! - All signing operations performed inside HSM
//! - Audit logging for all key operations
//! - Support for key attestation
//!
//! # References
//!
//! - FIPS 140-3: Security Requirements for Cryptographic Modules
//! - PKCS#11: Cryptographic Token Interface Standard

use crate::{DilithiumError, DilithiumResult, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Key handle for HSM-stored keys
///
/// Opaque identifier that references a key stored in the HSM.
/// The actual key material never leaves the secure boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyHandle {
    /// Unique identifier within the HSM
    pub id: String,
    /// Key type indicator
    pub key_type: KeyType,
    /// Security level of the key
    pub security_level: SecurityLevel,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// Optional label for human identification
    pub label: Option<String>,
}

impl KeyHandle {
    /// Create a new key handle
    pub fn new(id: impl Into<String>, key_type: KeyType, security_level: SecurityLevel) -> Self {
        Self {
            id: id.into(),
            key_type,
            security_level,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            label: None,
        }
    }

    /// Set optional label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Type of cryptographic key
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyType {
    /// Dilithium signing key pair
    DilithiumSigning,
    /// Kyber key encapsulation key pair
    KyberKem,
    /// Symmetric encryption key (ChaCha20-Poly1305)
    SymmetricAead,
    /// Key derivation master key
    KeyDerivation,
}

impl fmt::Display for KeyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyType::DilithiumSigning => write!(f, "Dilithium-Signing"),
            KeyType::KyberKem => write!(f, "Kyber-KEM"),
            KeyType::SymmetricAead => write!(f, "Symmetric-AEAD"),
            KeyType::KeyDerivation => write!(f, "Key-Derivation"),
        }
    }
}

/// Key attributes for HSM operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyAttributes {
    /// Whether key can be exported (should be false for HSM security)
    pub extractable: bool,
    /// Whether key can be used for signing
    pub sign: bool,
    /// Whether key can be used for verification
    pub verify: bool,
    /// Whether key can be used for encryption
    pub encrypt: bool,
    /// Whether key can be used for decryption
    pub decrypt: bool,
    /// Whether key can derive other keys
    pub derive: bool,
    /// Maximum number of uses (None = unlimited)
    pub max_uses: Option<u64>,
    /// Expiration timestamp (None = never expires)
    pub expires_at: Option<u64>,
}

impl Default for KeyAttributes {
    fn default() -> Self {
        Self {
            extractable: false, // HSM keys should not be extractable
            sign: true,
            verify: true,
            encrypt: false,
            decrypt: false,
            derive: false,
            max_uses: None,
            expires_at: None,
        }
    }
}

impl KeyAttributes {
    /// Create attributes for signing keys
    pub fn signing() -> Self {
        Self {
            sign: true,
            verify: true,
            ..Default::default()
        }
    }

    /// Create attributes for KEM keys
    pub fn kem() -> Self {
        Self {
            sign: false,
            verify: false,
            encrypt: true,
            decrypt: true,
            ..Default::default()
        }
    }

    /// Create attributes for derivation keys
    pub fn derivation() -> Self {
        Self {
            sign: false,
            verify: false,
            derive: true,
            ..Default::default()
        }
    }
}

/// HSM operation result with timing information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HsmOperationResult<T> {
    /// Operation result
    pub result: T,
    /// Operation duration in microseconds
    pub duration_us: u64,
    /// HSM-specific operation ID for audit
    pub operation_id: String,
}

/// HSM Provider trait
///
/// Implement this trait to integrate with different HSM backends:
/// - Software (for development/testing)
/// - TPM 2.0
/// - PKCS#11 HSMs
/// - Cloud KMS (AWS, GCP, Azure)
///
/// # Security Considerations
///
/// All implementations must ensure:
/// 1. Private keys never leave the secure boundary in plaintext
/// 2. All operations are logged for audit
/// 3. Side-channel resistance where applicable
/// 4. Proper zeroization of any temporary key material
#[allow(async_fn_in_trait)]
pub trait HsmProvider: Send + Sync {
    /// Generate a new Dilithium key pair in the HSM
    ///
    /// # Arguments
    ///
    /// * `security_level` - Dilithium security level
    /// * `attributes` - Key usage attributes
    /// * `label` - Optional human-readable label
    ///
    /// # Returns
    ///
    /// Key handle for the generated key pair. The private key
    /// remains inside the HSM and cannot be extracted.
    async fn generate_dilithium_key(
        &self,
        security_level: SecurityLevel,
        attributes: KeyAttributes,
        label: Option<&str>,
    ) -> DilithiumResult<KeyHandle>;

    /// Sign data using HSM-stored key
    ///
    /// # Arguments
    ///
    /// * `key_handle` - Handle to the signing key
    /// * `message` - Data to sign
    ///
    /// # Returns
    ///
    /// Dilithium signature bytes
    async fn sign(
        &self,
        key_handle: &KeyHandle,
        message: &[u8],
    ) -> DilithiumResult<HsmOperationResult<Vec<u8>>>;

    /// Verify signature using HSM (optional - can be done locally)
    ///
    /// Some HSMs provide hardware-accelerated verification.
    /// Default implementation extracts public key and verifies locally.
    async fn verify(
        &self,
        key_handle: &KeyHandle,
        message: &[u8],
        signature: &[u8],
    ) -> DilithiumResult<HsmOperationResult<bool>>;

    /// Export public key from HSM
    ///
    /// Public keys can always be exported safely.
    async fn export_public_key(&self, key_handle: &KeyHandle) -> DilithiumResult<Vec<u8>>;

    /// Import public key for verification
    ///
    /// Creates a key handle for verification-only operations.
    async fn import_public_key(
        &self,
        public_key: &[u8],
        security_level: SecurityLevel,
        label: Option<&str>,
    ) -> DilithiumResult<KeyHandle>;

    /// Delete key from HSM
    ///
    /// # Warning
    ///
    /// This operation is irreversible. Ensure proper backup
    /// procedures are in place before deletion.
    async fn delete_key(&self, key_handle: &KeyHandle) -> DilithiumResult<()>;

    /// List all keys in the HSM
    async fn list_keys(&self) -> DilithiumResult<Vec<KeyHandle>>;

    /// Get key attributes
    async fn get_key_attributes(&self, key_handle: &KeyHandle) -> DilithiumResult<KeyAttributes>;

    /// Update key attributes (if supported)
    async fn update_key_attributes(
        &self,
        key_handle: &KeyHandle,
        attributes: KeyAttributes,
    ) -> DilithiumResult<()>;

    /// Get HSM status and capabilities
    fn capabilities(&self) -> HsmCapabilities;

    /// Perform HSM health check
    async fn health_check(&self) -> DilithiumResult<HsmHealth>;
}

/// HSM capabilities descriptor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HsmCapabilities {
    /// HSM provider name
    pub provider: String,
    /// Provider version
    pub version: String,
    /// Supported key types
    pub supported_key_types: Vec<KeyType>,
    /// Supported Dilithium security levels
    pub supported_security_levels: Vec<SecurityLevel>,
    /// Maximum keys that can be stored
    pub max_keys: Option<u64>,
    /// Whether batch operations are supported
    pub batch_operations: bool,
    /// Whether key attestation is supported
    pub key_attestation: bool,
    /// FIPS 140 certification level (if any)
    pub fips_level: Option<u8>,
}

/// HSM health status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HsmHealth {
    /// Overall health status
    pub healthy: bool,
    /// Number of stored keys
    pub key_count: u64,
    /// Available key slots
    pub available_slots: Option<u64>,
    /// Last operation timestamp
    pub last_operation: Option<u64>,
    /// Error messages (if any)
    pub errors: Vec<String>,
}

/// Software HSM implementation for development/testing
///
/// # Warning
///
/// This implementation stores keys in memory and should ONLY be used
/// for development and testing. Use a real HSM for production.
pub struct SoftwareHsm {
    keys: std::sync::RwLock<std::collections::HashMap<String, SoftwareKey>>,
    operation_counter: std::sync::atomic::AtomicU64,
}

struct SoftwareKey {
    handle: KeyHandle,
    attributes: KeyAttributes,
    /// Public key bytes for export
    public_key_bytes: Vec<u8>,
    /// Full keypair for signing operations (None for public-key-only entries)
    keypair: Option<crate::DilithiumKeypair>,
}

impl SoftwareHsm {
    /// Create new software HSM instance
    pub fn new() -> Self {
        Self {
            keys: std::sync::RwLock::new(std::collections::HashMap::new()),
            operation_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn next_operation_id(&self) -> String {
        let id = self
            .operation_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("soft-hsm-op-{}", id)
    }
}

impl Default for SoftwareHsm {
    fn default() -> Self {
        Self::new()
    }
}

impl HsmProvider for SoftwareHsm {
    async fn generate_dilithium_key(
        &self,
        security_level: SecurityLevel,
        attributes: KeyAttributes,
        label: Option<&str>,
    ) -> DilithiumResult<KeyHandle> {
        use crate::DilithiumKeypair;

        let keypair = DilithiumKeypair::generate(security_level)?;

        let id = format!(
            "dilithium-{}-{}",
            security_level as u8,
            uuid::Uuid::new_v4()
        );
        let mut handle = KeyHandle::new(&id, KeyType::DilithiumSigning, security_level);
        if let Some(lbl) = label {
            handle = handle.with_label(lbl);
        }

        let software_key = SoftwareKey {
            handle: handle.clone(),
            attributes,
            public_key_bytes: keypair.public_key_bytes().to_vec(),
            keypair: Some(keypair),
        };

        self.keys
            .write()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?
            .insert(id, software_key);

        Ok(handle)
    }

    async fn sign(
        &self,
        key_handle: &KeyHandle,
        message: &[u8],
    ) -> DilithiumResult<HsmOperationResult<Vec<u8>>> {
        use std::time::Instant;

        let start = Instant::now();

        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        let key = keys
            .get(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        if !key.attributes.sign {
            return Err(DilithiumError::SignatureFailed(
                "Key not authorized for signing".to_string(),
            ));
        }

        let keypair = key
            .keypair
            .as_ref()
            .ok_or_else(|| DilithiumError::SignatureFailed("No signing key available".to_string()))?;

        let signature = keypair.sign(message)?;

        let duration = start.elapsed();

        Ok(HsmOperationResult {
            result: signature.signature_bytes.clone(),
            duration_us: duration.as_micros() as u64,
            operation_id: self.next_operation_id(),
        })
    }

    async fn verify(
        &self,
        key_handle: &KeyHandle,
        message: &[u8],
        signature: &[u8],
    ) -> DilithiumResult<HsmOperationResult<bool>> {
        use crate::DilithiumSignature;
        use std::time::Instant;

        let start = Instant::now();

        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        let key = keys
            .get(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        if !key.attributes.verify {
            return Err(DilithiumError::VerificationFailed(
                "Key not authorized for verification".to_string(),
            ));
        }

        let sig = DilithiumSignature::decode(signature, key_handle.security_level)?;

        // Use verify_standalone which takes public key bytes directly
        let result = sig.verify_standalone(
            message,
            &key.public_key_bytes,
            key_handle.security_level,
        )?;

        let duration = start.elapsed();

        Ok(HsmOperationResult {
            result,
            duration_us: duration.as_micros() as u64,
            operation_id: self.next_operation_id(),
        })
    }

    async fn export_public_key(&self, key_handle: &KeyHandle) -> DilithiumResult<Vec<u8>> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        let key = keys
            .get(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        Ok(key.public_key_bytes.clone())
    }

    async fn import_public_key(
        &self,
        public_key: &[u8],
        security_level: SecurityLevel,
        label: Option<&str>,
    ) -> DilithiumResult<KeyHandle> {
        let id = format!(
            "dilithium-pub-{}-{}",
            security_level as u8,
            uuid::Uuid::new_v4()
        );
        let mut handle = KeyHandle::new(&id, KeyType::DilithiumSigning, security_level);
        if let Some(lbl) = label {
            handle = handle.with_label(lbl);
        }

        let software_key = SoftwareKey {
            handle: handle.clone(),
            attributes: KeyAttributes {
                sign: false,
                verify: true,
                ..Default::default()
            },
            public_key_bytes: public_key.to_vec(),
            keypair: None, // Public-key only, no signing capability
        };

        self.keys
            .write()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?
            .insert(id, software_key);

        Ok(handle)
    }

    async fn delete_key(&self, key_handle: &KeyHandle) -> DilithiumResult<()> {
        self.keys
            .write()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?
            .remove(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        Ok(())
    }

    async fn list_keys(&self) -> DilithiumResult<Vec<KeyHandle>> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        Ok(keys.values().map(|k| k.handle.clone()).collect())
    }

    async fn get_key_attributes(&self, key_handle: &KeyHandle) -> DilithiumResult<KeyAttributes> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        let key = keys
            .get(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        Ok(key.attributes.clone())
    }

    async fn update_key_attributes(
        &self,
        key_handle: &KeyHandle,
        attributes: KeyAttributes,
    ) -> DilithiumResult<()> {
        let mut keys = self
            .keys
            .write()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        let key = keys
            .get_mut(&key_handle.id)
            .ok_or_else(|| DilithiumError::KeyGenerationFailed("Key not found".to_string()))?;

        key.attributes = attributes;
        Ok(())
    }

    fn capabilities(&self) -> HsmCapabilities {
        HsmCapabilities {
            provider: "SoftwareHSM".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_key_types: vec![KeyType::DilithiumSigning],
            supported_security_levels: vec![
                SecurityLevel::Standard,
                SecurityLevel::High,
                SecurityLevel::Maximum,
            ],
            max_keys: None,
            batch_operations: false,
            key_attestation: false,
            fips_level: None, // Software HSM is not FIPS certified
        }
    }

    async fn health_check(&self) -> DilithiumResult<HsmHealth> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DilithiumError::GPUError("Lock poisoned".to_string()))?;

        Ok(HsmHealth {
            healthy: true,
            key_count: keys.len() as u64,
            available_slots: None,
            last_operation: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
            errors: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_software_hsm_key_generation() {
        let hsm = SoftwareHsm::new();

        let handle = hsm
            .generate_dilithium_key(
                SecurityLevel::Standard,
                KeyAttributes::signing(),
                Some("test-key"),
            )
            .await
            .expect("Key generation failed");

        assert_eq!(handle.key_type, KeyType::DilithiumSigning);
        assert_eq!(handle.security_level, SecurityLevel::Standard);
        assert_eq!(handle.label, Some("test-key".to_string()));
    }

    #[tokio::test]
    async fn test_software_hsm_sign_verify() {
        let hsm = SoftwareHsm::new();

        let handle = hsm
            .generate_dilithium_key(SecurityLevel::Standard, KeyAttributes::signing(), None)
            .await
            .expect("Key generation failed");

        let message = b"Test message for HSM signing";

        let sign_result = hsm
            .sign(&handle, message)
            .await
            .expect("Signing failed");

        assert!(!sign_result.result.is_empty());
        assert!(sign_result.duration_us > 0);

        let verify_result = hsm
            .verify(&handle, message, &sign_result.result)
            .await
            .expect("Verification failed");

        assert!(verify_result.result);
    }

    #[tokio::test]
    async fn test_software_hsm_key_lifecycle() {
        let hsm = SoftwareHsm::new();

        // Generate key
        let handle = hsm
            .generate_dilithium_key(SecurityLevel::High, KeyAttributes::signing(), None)
            .await
            .expect("Key generation failed");

        // List keys
        let keys = hsm.list_keys().await.expect("List keys failed");
        assert_eq!(keys.len(), 1);

        // Delete key
        hsm.delete_key(&handle).await.expect("Delete failed");

        // Verify deletion
        let keys = hsm.list_keys().await.expect("List keys failed");
        assert_eq!(keys.len(), 0);
    }

    #[tokio::test]
    async fn test_software_hsm_capabilities() {
        let hsm = SoftwareHsm::new();
        let caps = hsm.capabilities();

        assert_eq!(caps.provider, "SoftwareHSM");
        assert!(caps.supported_key_types.contains(&KeyType::DilithiumSigning));
        assert_eq!(caps.fips_level, None); // Software HSM is not FIPS certified
    }

    #[tokio::test]
    async fn test_software_hsm_health() {
        let hsm = SoftwareHsm::new();

        let health = hsm.health_check().await.expect("Health check failed");

        assert!(health.healthy);
        assert_eq!(health.key_count, 0);
        assert!(health.errors.is_empty());
    }
}
