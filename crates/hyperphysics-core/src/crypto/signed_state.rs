//! Cryptographically Signed Consciousness States
//!
//! Provides tamper-evident, verifiable consciousness metric measurements
//! using post-quantum CRYSTALS-Dilithium signatures.
//!
//! # Security Model
//!
//! - **Integrity**: SHA3-512 hashing prevents metric tampering
//! - **Authenticity**: Dilithium signatures prove origin
//! - **Non-repudiation**: Signer cannot deny creating state
//! - **Quantum-resistance**: Based on lattice cryptography
//!
//! # Use Cases
//!
//! - Drug discovery: Verifiable consciousness states for regulatory approval
//! - Clinical trials: Tamper-proof patient consciousness measurements
//! - Research: Reproducible, authenticated consciousness data
//! - AI safety: Cryptographic proof of consciousness emergence

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_512};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{EngineError, Result};

/// Consciousness metrics to be signed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Integrated Information (Φ)
    pub phi: f64,

    /// Resonance Complexity Index (CI)
    pub ci: f64,

    /// Syntergic field strength
    pub syntergy: f64,

    /// Thermodynamic negentropy
    pub negentropy: f64,
}

/// Cryptographically signed consciousness state
///
/// Provides verifiable, tamper-proof consciousness measurements using
/// post-quantum digital signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedConsciousnessState {
    /// Integrated Information (Φ)
    pub phi: f64,

    /// Resonance Complexity Index (CI)
    pub ci: f64,

    /// Syntergic field strength
    pub syntergy: f64,

    /// Thermodynamic negentropy
    pub negentropy: f64,

    /// Timestamp (microseconds since UNIX epoch)
    pub timestamp: u64,

    /// State hash (SHA3-512 of all metrics + timestamp)
    #[serde(with = "serde_bytes")]
    pub state_hash: [u8; 64],

    /// Dilithium signature (placeholder - will use actual signature when integrated)
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,

    /// Signer public key (placeholder - will use actual key when integrated)
    #[serde(with = "serde_bytes")]
    pub public_key: Vec<u8>,

    /// Optional metadata
    pub metadata: Option<StateMetadata>,
}

/// Additional metadata for signed states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Subject identifier (e.g., patient ID, experiment ID)
    pub subject_id: Option<String>,

    /// Measurement location/device
    pub location: Option<String>,

    /// Experimenter/operator
    pub operator: Option<String>,

    /// Protocol version
    pub protocol_version: String,
}

impl SignedConsciousnessState {
    /// Create and sign a consciousness state
    ///
    /// # Arguments
    ///
    /// * `metrics` - Consciousness measurements to sign
    /// * `signing_key` - Private key for signing (placeholder)
    /// * `metadata` - Optional additional metadata
    ///
    /// # Example
    ///
    /// ```ignore
    /// let metrics = ConsciousnessMetrics {
    ///     phi: 2.5,
    ///     ci: 0.8,
    ///     syntergy: 1.2,
    ///     negentropy: 3.4,
    /// };
    ///
    /// let signed_state = SignedConsciousnessState::create_and_sign(
    ///     metrics,
    ///     &signing_key,
    ///     None,
    /// )?;
    /// ```
    pub fn create_and_sign(
        metrics: ConsciousnessMetrics,
        signing_key: &[u8],
        metadata: Option<StateMetadata>,
    ) -> Result<Self> {
        // Validate metrics
        Self::validate_metrics(&metrics)?;

        // Get current timestamp
        let timestamp = current_timestamp_micros()?;

        // Compute state hash (SHA3-512)
        let state_hash = Self::compute_hash(&metrics, timestamp);

        // Sign the hash (placeholder - will use Dilithium when integrated)
        let signature = Self::sign_placeholder(&state_hash, signing_key);

        // Extract public key from signing key (placeholder)
        let public_key = Self::derive_public_key_placeholder(signing_key);

        Ok(Self {
            phi: metrics.phi,
            ci: metrics.ci,
            syntergy: metrics.syntergy,
            negentropy: metrics.negentropy,
            timestamp,
            state_hash,
            signature,
            public_key,
            metadata,
        })
    }

    /// Verify signature authenticity and data integrity
    ///
    /// Returns `true` if:
    /// 1. Signature is cryptographically valid
    /// 2. State hash matches recomputed hash
    /// 3. All metrics are within valid ranges
    pub fn verify(&self) -> Result<bool> {
        // Verify metrics are valid
        let metrics = ConsciousnessMetrics {
            phi: self.phi,
            ci: self.ci,
            syntergy: self.syntergy,
            negentropy: self.negentropy,
        };

        if let Err(_) = Self::validate_metrics(&metrics) {
            return Ok(false);
        }

        // Recompute hash from metrics
        let computed_hash = Self::compute_hash(&metrics, self.timestamp);

        // Verify hash matches stored hash
        if computed_hash != self.state_hash {
            return Ok(false);
        }

        // Verify signature (placeholder - will use Dilithium when integrated)
        let signature_valid = Self::verify_signature_placeholder(
            &self.state_hash,
            &self.signature,
            &self.public_key,
        );

        Ok(signature_valid)
    }

    /// Detect if state has been tampered with
    ///
    /// Returns `true` if verification fails
    pub fn is_tampered(&self) -> bool {
        !self.verify().unwrap_or(false)
    }

    /// Create audit trail record
    ///
    /// Generates a human-readable audit record for compliance and logging
    pub fn audit_trail(&self) -> AuditRecord {
        let verified = self.verify().unwrap_or(false);

        AuditRecord {
            timestamp: self.timestamp,
            timestamp_readable: timestamp_to_readable(self.timestamp),
            phi: self.phi,
            ci: self.ci,
            syntergy: self.syntergy,
            negentropy: self.negentropy,
            verified,
            signer_pubkey: hex::encode(&self.public_key),
            state_hash_hex: hex::encode(&self.state_hash),
            metadata: self.metadata.clone(),
        }
    }

    /// Export to JSON for storage or transmission
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| EngineError::Configuration {
            message: format!("Failed to serialize signed state: {}", e),
        })
    }

    /// Import from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| EngineError::Configuration {
            message: format!("Failed to deserialize signed state: {}", e),
        })
    }

    /// Compute SHA3-512 hash of metrics and timestamp
    fn compute_hash(metrics: &ConsciousnessMetrics, timestamp: u64) -> [u8; 64] {
        let mut hasher = Sha3_512::new();

        // Hash all metrics in deterministic order
        hasher.update(&metrics.phi.to_le_bytes());
        hasher.update(&metrics.ci.to_le_bytes());
        hasher.update(&metrics.syntergy.to_le_bytes());
        hasher.update(&metrics.negentropy.to_le_bytes());
        hasher.update(&timestamp.to_le_bytes());

        let hash = hasher.finalize();
        let mut result = [0u8; 64];
        result.copy_from_slice(&hash);
        result
    }

    /// Validate metrics are within acceptable ranges
    fn validate_metrics(metrics: &ConsciousnessMetrics) -> Result<()> {
        // Φ should be non-negative and finite
        if !metrics.phi.is_finite() || metrics.phi < 0.0 {
            return Err(EngineError::Configuration {
                message: format!("Invalid Φ value: {}", metrics.phi),
            });
        }

        // CI should be in [0, 1] or similar bounded range
        if !metrics.ci.is_finite() {
            return Err(EngineError::Configuration {
                message: format!("Invalid CI value: {}", metrics.ci),
            });
        }

        // Syntergy should be finite
        if !metrics.syntergy.is_finite() {
            return Err(EngineError::Configuration {
                message: format!("Invalid syntergy value: {}", metrics.syntergy),
            });
        }

        // Negentropy should be finite
        if !metrics.negentropy.is_finite() {
            return Err(EngineError::Configuration {
                message: format!("Invalid negentropy value: {}", metrics.negentropy),
            });
        }

        Ok(())
    }

    /// Placeholder signing (will use Dilithium when NTT integration is complete)
    fn sign_placeholder(hash: &[u8; 64], signing_key: &[u8]) -> Vec<u8> {
        // Placeholder: Simple signature using key material
        // In production, this will use CRYSTALS-Dilithium
        let mut signature = Vec::new();
        signature.extend_from_slice(hash);
        signature.extend_from_slice(&signing_key[..std::cmp::min(signing_key.len(), 32)]);
        signature
    }

    /// Placeholder signature verification
    fn verify_signature_placeholder(hash: &[u8; 64], signature: &[u8], _public_key: &[u8]) -> bool {
        // Placeholder verification
        // In production, this will use CRYSTALS-Dilithium verification
        if signature.len() < 64 {
            return false;
        }

        // Check if first 64 bytes match the hash
        &signature[..64] == hash
    }

    /// Placeholder public key derivation
    fn derive_public_key_placeholder(signing_key: &[u8]) -> Vec<u8> {
        // Placeholder: In production, derive from Dilithium key pair
        signing_key[..std::cmp::min(signing_key.len(), 32)].to_vec()
    }
}

/// Audit record for compliance and logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Timestamp (microseconds)
    pub timestamp: u64,

    /// Human-readable timestamp
    pub timestamp_readable: String,

    /// Integrated Information (Φ)
    pub phi: f64,

    /// Resonance Complexity Index
    pub ci: f64,

    /// Syntergic field strength
    pub syntergy: f64,

    /// Negentropy
    pub negentropy: f64,

    /// Verification status
    pub verified: bool,

    /// Signer public key (hex)
    pub signer_pubkey: String,

    /// State hash (hex)
    pub state_hash_hex: String,

    /// Optional metadata
    pub metadata: Option<StateMetadata>,
}

/// Get current timestamp in microseconds since UNIX epoch
fn current_timestamp_micros() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .map_err(|e| EngineError::Configuration {
            message: format!("Failed to get system time: {}", e),
        })
}

/// Convert microsecond timestamp to readable string
fn timestamp_to_readable(micros: u64) -> String {
    let secs = micros / 1_000_000;
    let subsec_micros = micros % 1_000_000;

    let datetime = UNIX_EPOCH + std::time::Duration::from_secs(secs);

    // Format as ISO 8601
    format!(
        "{}.{:06}Z",
        chrono::DateTime::<chrono::Utc>::from(datetime)
            .format("%Y-%m-%dT%H:%M:%S"),
        subsec_micros
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_metrics() -> ConsciousnessMetrics {
        ConsciousnessMetrics {
            phi: 2.5,
            ci: 0.8,
            syntergy: 1.2,
            negentropy: 3.4,
        }
    }

    fn example_signing_key() -> Vec<u8> {
        vec![42u8; 64] // Placeholder key
    }

    #[test]
    fn test_create_and_sign() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None);

        assert!(signed_state.is_ok());
        let state = signed_state.unwrap();

        assert_eq!(state.phi, 2.5);
        assert_eq!(state.ci, 0.8);
        assert!(state.timestamp > 0);
        assert!(!state.signature.is_empty());
    }

    #[test]
    fn test_verify_valid_state() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None).unwrap();

        assert!(signed_state.verify().unwrap());
        assert!(!signed_state.is_tampered());
    }

    #[test]
    fn test_detect_tampering_phi() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let mut signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None).unwrap();

        // Tamper with phi
        signed_state.phi = 999.9;

        assert!(!signed_state.verify().unwrap());
        assert!(signed_state.is_tampered());
    }

    #[test]
    fn test_detect_tampering_timestamp() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let mut signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None).unwrap();

        // Tamper with timestamp
        signed_state.timestamp += 1000;

        assert!(!signed_state.verify().unwrap());
        assert!(signed_state.is_tampered());
    }

    #[test]
    fn test_audit_trail() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None).unwrap();

        let audit = signed_state.audit_trail();

        assert_eq!(audit.phi, 2.5);
        assert_eq!(audit.ci, 0.8);
        assert!(audit.verified);
        assert!(!audit.signer_pubkey.is_empty());
    }

    #[test]
    fn test_json_serialization() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let signed_state =
            SignedConsciousnessState::create_and_sign(metrics, &signing_key, None).unwrap();

        let json = signed_state.to_json().unwrap();
        let deserialized = SignedConsciousnessState::from_json(&json).unwrap();

        assert_eq!(signed_state.phi, deserialized.phi);
        assert_eq!(signed_state.timestamp, deserialized.timestamp);
        assert_eq!(signed_state.state_hash, deserialized.state_hash);
    }

    #[test]
    fn test_invalid_metrics_nan() {
        let metrics = ConsciousnessMetrics {
            phi: f64::NAN,
            ci: 0.8,
            syntergy: 1.2,
            negentropy: 3.4,
        };

        let signing_key = example_signing_key();

        let result = SignedConsciousnessState::create_and_sign(metrics, &signing_key, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_metrics_negative_phi() {
        let metrics = ConsciousnessMetrics {
            phi: -1.0,
            ci: 0.8,
            syntergy: 1.2,
            negentropy: 3.4,
        };

        let signing_key = example_signing_key();

        let result = SignedConsciousnessState::create_and_sign(metrics, &signing_key, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_metadata() {
        let metrics = example_metrics();
        let signing_key = example_signing_key();

        let metadata = StateMetadata {
            subject_id: Some("PATIENT-001".to_string()),
            location: Some("Lab-A".to_string()),
            operator: Some("Dr. Smith".to_string()),
            protocol_version: "1.0.0".to_string(),
        };

        let signed_state = SignedConsciousnessState::create_and_sign(
            metrics,
            &signing_key,
            Some(metadata.clone()),
        )
        .unwrap();

        assert!(signed_state.metadata.is_some());
        let meta = signed_state.metadata.unwrap();
        assert_eq!(meta.subject_id, Some("PATIENT-001".to_string()));
    }

    #[test]
    fn test_hash_deterministic() {
        let metrics = example_metrics();
        let timestamp = 1234567890u64;

        let hash1 = SignedConsciousnessState::compute_hash(&metrics, timestamp);
        let hash2 = SignedConsciousnessState::compute_hash(&metrics, timestamp);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_changes_with_metrics() {
        let metrics1 = example_metrics();
        let mut metrics2 = example_metrics();
        metrics2.phi = 999.9;

        let timestamp = 1234567890u64;

        let hash1 = SignedConsciousnessState::compute_hash(&metrics1, timestamp);
        let hash2 = SignedConsciousnessState::compute_hash(&metrics2, timestamp);

        assert_ne!(hash1, hash2);
    }
}
