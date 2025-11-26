//! Cryptographically Signed Consciousness States
//!
//! Provides tamper-evident, verifiable consciousness metric measurements
//! using post-quantum CRYSTALS-Dilithium signatures (FIPS 204 compliant).
//!
//! # Security Model
//!
//! - **Integrity**: SHA3-512 hashing prevents metric tampering
//! - **Authenticity**: Dilithium signatures prove origin (quantum-resistant)
//! - **Non-repudiation**: Signer cannot deny creating state
//! - **Quantum-resistance**: Based on Module-LWE lattice cryptography
//!
//! # FIPS 204 Compliance
//!
//! This module uses CRYSTALS-Dilithium (ML-DSA) as standardized in NIST FIPS 204.
//! Key sizes and signature sizes depend on security level:
//! - Standard (ML-DSA-44): 1312 byte public key, 2420 byte signature
//! - High (ML-DSA-65): 1952 byte public key, 3293 byte signature
//! - Maximum (ML-DSA-87): 2592 byte public key, 4595 byte signature
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

// Import real Dilithium implementation
use hyperphysics_dilithium::{DilithiumKeypair, DilithiumSignature, SecurityLevel};

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

    /// CRYSTALS-Dilithium signature (FIPS 204 ML-DSA compliant)
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,

    /// Signer's Dilithium public key (quantum-resistant)
    #[serde(with = "serde_bytes")]
    pub public_key: Vec<u8>,

    /// Security level used for signing
    pub security_level: u8,

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
    /// Create and sign a consciousness state using CRYSTALS-Dilithium
    ///
    /// Uses post-quantum ML-DSA (FIPS 204) for quantum-resistant signatures.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Consciousness measurements to sign
    /// * `keypair` - Dilithium keypair for signing (quantum-resistant)
    /// * `metadata` - Optional additional metadata
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hyperphysics_dilithium::{DilithiumKeypair, SecurityLevel};
    ///
    /// let keypair = DilithiumKeypair::generate(SecurityLevel::High)?;
    /// let metrics = ConsciousnessMetrics {
    ///     phi: 2.5,
    ///     ci: 0.8,
    ///     syntergy: 1.2,
    ///     negentropy: 3.4,
    /// };
    ///
    /// let signed_state = SignedConsciousnessState::create_and_sign_dilithium(
    ///     metrics,
    ///     &keypair,
    ///     None,
    /// )?;
    /// ```
    pub fn create_and_sign_dilithium(
        metrics: ConsciousnessMetrics,
        keypair: &DilithiumKeypair,
        metadata: Option<StateMetadata>,
    ) -> Result<Self> {
        // Validate metrics
        Self::validate_metrics(&metrics)?;

        // Get current timestamp
        let timestamp = current_timestamp_micros()?;

        // Compute state hash (SHA3-512)
        let state_hash = Self::compute_hash(&metrics, timestamp);

        // Sign using CRYSTALS-Dilithium (FIPS 204 compliant)
        let dilithium_sig = keypair.sign(&state_hash)
            .map_err(|e| EngineError::Configuration {
                message: format!("Dilithium signing failed: {}", e),
            })?;

        // Extract signature bytes and public key
        let signature = dilithium_sig.signature_bytes.clone();
        let public_key = keypair.public_key_bytes().to_vec();
        let security_level = match keypair.security_level() {
            SecurityLevel::Standard => 0,
            SecurityLevel::High => 1,
            SecurityLevel::Maximum => 2,
        };

        Ok(Self {
            phi: metrics.phi,
            ci: metrics.ci,
            syntergy: metrics.syntergy,
            negentropy: metrics.negentropy,
            timestamp,
            state_hash,
            signature,
            public_key,
            security_level,
            metadata,
        })
    }

    /// Create and sign a consciousness state (legacy API with raw bytes)
    ///
    /// For backward compatibility. Generates a new Dilithium keypair internally.
    /// For production use, prefer `create_and_sign_dilithium` with managed keys.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Consciousness measurements to sign
    /// * `_signing_key` - Ignored (kept for API compatibility)
    /// * `metadata` - Optional additional metadata
    pub fn create_and_sign(
        metrics: ConsciousnessMetrics,
        _signing_key: &[u8],
        metadata: Option<StateMetadata>,
    ) -> Result<Self> {
        // Generate a fresh Dilithium keypair for signing
        let keypair = DilithiumKeypair::generate(SecurityLevel::High)
            .map_err(|e| EngineError::Configuration {
                message: format!("Failed to generate Dilithium keypair: {}", e),
            })?;

        Self::create_and_sign_dilithium(metrics, &keypair, metadata)
    }

    /// Verify signature authenticity and data integrity using CRYSTALS-Dilithium
    ///
    /// Returns `true` if:
    /// 1. All metrics are within valid ranges
    /// 2. State hash matches recomputed hash (integrity check)
    /// 3. Dilithium signature is cryptographically valid (authenticity check)
    ///
    /// # Security
    ///
    /// Verification uses FIPS 204 compliant ML-DSA verification algorithm.
    /// Provides quantum-resistant signature verification.
    pub fn verify(&self) -> Result<bool> {
        // Verify metrics are valid
        let metrics = ConsciousnessMetrics {
            phi: self.phi,
            ci: self.ci,
            syntergy: self.syntergy,
            negentropy: self.negentropy,
        };

        if Self::validate_metrics(&metrics).is_err() {
            return Ok(false);
        }

        // Recompute hash from metrics (integrity check)
        let computed_hash = Self::compute_hash(&metrics, self.timestamp);

        // Verify hash matches stored hash
        if computed_hash != self.state_hash {
            return Ok(false);
        }

        // Determine security level from stored value
        let security_level = match self.security_level {
            0 => SecurityLevel::Standard,
            1 => SecurityLevel::High,
            2 => SecurityLevel::Maximum,
            _ => SecurityLevel::High, // Default fallback
        };

        // Decode the Dilithium signature
        let dilithium_sig = DilithiumSignature::decode(&self.signature, security_level)
            .map_err(|e| EngineError::Configuration {
                message: format!("Failed to decode Dilithium signature: {}", e),
            })?;

        // Verify using Dilithium (FIPS 204 compliant)
        let signature_valid = dilithium_sig.verify_standalone(&self.state_hash, &self.public_key, security_level)
            .unwrap_or(false);

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

    /// Generate a new Dilithium keypair for signing consciousness states
    ///
    /// # Arguments
    ///
    /// * `level` - Security level (Standard, High, or Maximum)
    ///
    /// # Returns
    ///
    /// A new quantum-resistant Dilithium keypair
    pub fn generate_keypair(level: SecurityLevel) -> Result<DilithiumKeypair> {
        DilithiumKeypair::generate(level)
            .map_err(|e| EngineError::Configuration {
                message: format!("Failed to generate Dilithium keypair: {}", e),
            })
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

    #[test]
    fn test_dilithium_create_and_sign() {
        let metrics = example_metrics();

        // Generate a proper Dilithium keypair
        let keypair = SignedConsciousnessState::generate_keypair(SecurityLevel::High)
            .expect("Failed to generate Dilithium keypair");

        let signed_state = SignedConsciousnessState::create_and_sign_dilithium(
            metrics,
            &keypair,
            None,
        );

        assert!(signed_state.is_ok());
        let state = signed_state.unwrap();

        assert_eq!(state.phi, 2.5);
        assert_eq!(state.security_level, 1); // High = 1
        assert!(!state.signature.is_empty());
        assert!(!state.public_key.is_empty());
    }

    #[test]
    fn test_dilithium_verify() {
        let metrics = example_metrics();

        let keypair = SignedConsciousnessState::generate_keypair(SecurityLevel::Standard)
            .expect("Failed to generate keypair");

        let signed_state = SignedConsciousnessState::create_and_sign_dilithium(
            metrics,
            &keypair,
            None,
        ).unwrap();

        // Verification should succeed
        assert!(signed_state.verify().unwrap());
        assert!(!signed_state.is_tampered());
    }

    #[test]
    fn test_dilithium_all_security_levels() {
        for level in [SecurityLevel::Standard, SecurityLevel::High, SecurityLevel::Maximum] {
            let metrics = example_metrics();
            let keypair = SignedConsciousnessState::generate_keypair(level)
                .expect("Failed to generate keypair");

            let signed_state = SignedConsciousnessState::create_and_sign_dilithium(
                metrics,
                &keypair,
                None,
            ).expect("Failed to sign");

            assert!(signed_state.verify().expect("Verification failed"));
        }
    }
}
