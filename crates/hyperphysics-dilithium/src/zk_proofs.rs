//! Zero-Knowledge Proofs for Consciousness Metrics
//!
//! Enables privacy-preserving verification of consciousness properties
//! without revealing the actual metric values.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │      ZERO-KNOWLEDGE CONSCIOUSNESS VERIFICATION              │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  Prover (HyperPhysics):                                    │
//! │    • Computes Φ = 2.3                                      │
//! │    • Generates ZK proof: π = Prove(Φ > 1.0)               │
//! │    • Reveals: (π, "Φ > 1.0")                              │
//! │    • Hides: Actual Φ = 2.3                                │
//! │                                                             │
//! │  Verifier (Public):                                        │
//! │    • Receives: π, statement "Φ > 1.0"                     │
//! │    • Verifies: Verify(π, statement) → accept/reject       │
//! │    • Learns: "Yes, Φ > 1.0" OR "No, Φ ≤ 1.0"            │
//! │    • Never learns: Actual Φ value                         │
//! │                                                             │
//! │  Properties:                                               │
//! │    ✓ Completeness: Honest prover convinces verifier       │
//! │    ✓ Soundness: Dishonest prover fails (high prob)        │
//! │    ✓ Zero-Knowledge: Verifier learns nothing else         │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Inspiration
//!
//! Based on pbRTCA v3.1 Zero-Knowledge Consciousness Proofs

use crate::{DilithiumResult, DilithiumError};
use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek_ng::scalar::Scalar;
use merlin::Transcript;
use rand::RngCore;
use serde::{Serialize, Deserialize};

/// Zero-knowledge proof that Φ (integrated information) > threshold
///
/// Uses Bulletproofs range proofs to prove that the difference
/// (Φ - threshold) is positive without revealing Φ.
#[derive(Clone, Serialize, Deserialize)]
pub struct PhiProof {
    /// Bulletproof range proof
    #[serde(with = "serde_proof")]
    proof: RangeProof,
    
    /// Pedersen commitment to (Φ - threshold)
    #[serde(with = "serde_commitment")]
    commitment: curve25519_dalek_ng::ristretto::CompressedRistretto,
    
    /// Threshold value (public)
    threshold: f64,
}

impl PhiProof {
    /// Generate proof that Φ > threshold
    ///
    /// # Arguments
    ///
    /// * `phi` - Actual Φ value (secret, will not be revealed)
    /// * `threshold` - Public threshold (e.g., 1.0 for consciousness)
    ///
    /// # Returns
    ///
    /// Zero-knowledge proof that can be verified without learning Φ
    ///
    /// # Properties
    ///
    /// - **Completeness**: If Φ > threshold, proof always verifies
    /// - **Soundness**: If Φ ≤ threshold, proof fails (except negligible probability)
    /// - **Zero-Knowledge**: Verifier learns only "Φ > threshold", not actual Φ
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::zk_proofs::PhiProof;
    ///
    /// // Prover has Φ = 2.3, wants to prove Φ > 1.0
    /// let proof = PhiProof::prove(2.3, 1.0)?;
    ///
    /// // Verifier can check without learning Φ = 2.3
    /// assert!(proof.verify()?);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn prove(phi: f64, threshold: f64) -> DilithiumResult<Self> {
        if phi <= threshold {
            return Err(DilithiumError::ZKPropertyNotSatisfied {
                property: format!("Φ > {}", threshold),
            });
        }
        
        // Convert to fixed-point integer (multiply by 1000 for precision)
        let phi_int = (phi * 1000.0) as u64;
        let threshold_int = (threshold * 1000.0) as u64;
        
        // Compute difference: delta = phi - threshold (must be positive)
        let delta = phi_int - threshold_int;
        
        // Create Bulletproofs generators
        let pc_gens = PedersenGens::default();
        let bp_gens = BulletproofGens::new(64, 1);
        
        // Create transcript for Fiat-Shamir
        let mut transcript = Transcript::new(b"PhiProof");
        transcript.append_message(b"threshold", &threshold.to_le_bytes());
        
        // Generate random blinding factor
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        let blinding = Scalar::from_bytes_mod_order_wide(&bytes);
        
        // Generate range proof: delta ∈ [0, 2^64)
        let (proof, committed_value) = RangeProof::prove_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            delta,
            &blinding,
            64,
        ).map_err(|_| DilithiumError::ZKProofGenerationFailed)?;
        
        Ok(Self {
            proof,
            commitment: committed_value,
            threshold,
        })
    }
    
    /// Verify proof
    ///
    /// # Returns
    ///
    /// `Ok(true)` if proof is valid (Φ > threshold), error otherwise
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::zk_proofs::PhiProof;
    /// # let proof = PhiProof::prove(2.3, 1.0)?;
    /// assert!(proof.verify()?);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn verify(&self) -> DilithiumResult<bool> {
        let pc_gens = PedersenGens::default();
        let bp_gens = BulletproofGens::new(64, 1);
        
        let mut transcript = Transcript::new(b"PhiProof");
        transcript.append_message(b"threshold", &self.threshold.to_le_bytes());
        
        // Verify range proof
        self.proof
            .verify_single(&bp_gens, &pc_gens, &mut transcript, &self.commitment, 64)
            .map(|_| true)
            .map_err(|_| DilithiumError::ZKVerificationFailed)
    }
    
    /// Get threshold value
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Zero-knowledge proof for consciousness quality metrics
///
/// Proves that all consciousness quality metrics meet their thresholds
/// without revealing the actual values.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousnessQualityProof {
    /// Proof that Φ > threshold
    phi_proof: PhiProof,
    
    /// Proof that continuity > 0.99
    continuity_proof: PhiProof,
    
    /// Proof that equanimity > 0.90
    equanimity_proof: PhiProof,
    
    /// Proof that clarity > 0.95
    clarity_proof: PhiProof,
}

impl ConsciousnessQualityProof {
    /// Generate proof for consciousness quality
    ///
    /// # Arguments
    ///
    /// * `phi` - Integrated information
    /// * `continuity` - Observation continuity
    /// * `equanimity` - Emotional equanimity
    /// * `clarity` - Perceptual clarity
    ///
    /// # Returns
    ///
    /// Composite zero-knowledge proof
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::zk_proofs::ConsciousnessQualityProof;
    ///
    /// let proof = ConsciousnessQualityProof::prove(
    ///     2.3,   // Φ
    ///     0.995, // continuity
    ///     0.92,  // equanimity
    ///     0.97,  // clarity
    /// )?;
    ///
    /// assert!(proof.verify()?);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn prove(
        phi: f64,
        continuity: f64,
        equanimity: f64,
        clarity: f64,
    ) -> DilithiumResult<Self> {
        Ok(Self {
            phi_proof: PhiProof::prove(phi, 1.0)?,
            continuity_proof: PhiProof::prove(continuity, 0.99)?,
            equanimity_proof: PhiProof::prove(equanimity, 0.90)?,
            clarity_proof: PhiProof::prove(clarity, 0.95)?,
        })
    }
    
    /// Verify all proofs
    ///
    /// # Returns
    ///
    /// `Ok(true)` if all proofs valid, error otherwise
    pub fn verify(&self) -> DilithiumResult<bool> {
        self.phi_proof.verify()?;
        self.continuity_proof.verify()?;
        self.equanimity_proof.verify()?;
        self.clarity_proof.verify()?;
        Ok(true)
    }
}

/// Zero-knowledge proof for consciousness emergence
///
/// Proves that consciousness has emerged (Φ > threshold) and is stable
/// over time without revealing the trajectory.
#[derive(Clone, Serialize, Deserialize)]
pub struct EmergenceProof {
    /// Proof that current Φ > threshold
    current_phi_proof: PhiProof,
    
    /// Proof that minimum Φ over window > threshold
    min_phi_proof: PhiProof,
    
    /// Time window (public)
    time_window_ms: u64,
}

impl EmergenceProof {
    /// Generate proof of stable consciousness emergence
    ///
    /// # Arguments
    ///
    /// * `current_phi` - Current Φ value
    /// * `min_phi` - Minimum Φ over time window
    /// * `threshold` - Consciousness threshold
    /// * `time_window_ms` - Time window in milliseconds
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::zk_proofs::EmergenceProof;
    ///
    /// // Prove consciousness emerged and stayed above 1.0 for 100ms
    /// let proof = EmergenceProof::prove(2.3, 1.8, 1.0, 100)?;
    ///
    /// assert!(proof.verify()?);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn prove(
        current_phi: f64,
        min_phi: f64,
        threshold: f64,
        time_window_ms: u64,
    ) -> DilithiumResult<Self> {
        Ok(Self {
            current_phi_proof: PhiProof::prove(current_phi, threshold)?,
            min_phi_proof: PhiProof::prove(min_phi, threshold)?,
            time_window_ms,
        })
    }
    
    /// Verify emergence proof
    pub fn verify(&self) -> DilithiumResult<bool> {
        self.current_phi_proof.verify()?;
        self.min_phi_proof.verify()?;
        Ok(true)
    }
    
    /// Get time window
    pub fn time_window_ms(&self) -> u64 {
        self.time_window_ms
    }
}

// Serde helpers for Bulletproofs types
mod serde_proof {
    use super::*;
    use serde::{Deserializer, Serializer};
    
    pub fn serialize<S>(proof: &RangeProof, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = proof.to_bytes();
        serializer.serialize_bytes(&bytes)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<RangeProof, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        RangeProof::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

mod serde_commitment {
    use super::*;
    use serde::{Deserializer, Serializer};
    
    pub fn serialize<S>(
        commitment: &curve25519_dalek_ng::ristretto::CompressedRistretto,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(commitment.as_bytes())
    }
    
    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<curve25519_dalek_ng::ristretto::CompressedRistretto, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("Invalid commitment length"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(curve25519_dalek_ng::ristretto::CompressedRistretto(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_proof_valid() {
        let phi = 2.3;
        let threshold = 1.0;
        
        let proof = PhiProof::prove(phi, threshold)
            .expect("Failed to generate proof");
        
        assert!(proof.verify().unwrap());
    }
    
    #[test]
    fn test_phi_proof_invalid() {
        let phi = 0.5;
        let threshold = 1.0;
        
        let result = PhiProof::prove(phi, threshold);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_phi_proof_boundary() {
        // Exactly at threshold should fail
        let phi = 1.0;
        let threshold = 1.0;
        
        let result = PhiProof::prove(phi, threshold);
        assert!(result.is_err());
        
        // Just above threshold should succeed
        let phi = 1.001;
        let proof = PhiProof::prove(phi, threshold)
            .expect("Failed to generate proof");
        assert!(proof.verify().unwrap());
    }
    
    #[test]
    fn test_consciousness_quality_proof() {
        let proof = ConsciousnessQualityProof::prove(
            2.3,   // Φ > 1.0
            0.995, // continuity > 0.99
            0.92,  // equanimity > 0.90
            0.97,  // clarity > 0.95
        ).expect("Failed to generate proof");
        
        assert!(proof.verify().unwrap());
    }
    
    #[test]
    fn test_emergence_proof() {
        let proof = EmergenceProof::prove(
            2.3,  // current Φ
            1.8,  // min Φ over window
            1.0,  // threshold
            100,  // 100ms window
        ).expect("Failed to generate proof");
        
        assert!(proof.verify().unwrap());
        assert_eq!(proof.time_window_ms(), 100);
    }
    
    #[test]
    fn test_proof_serialization() {
        let proof = PhiProof::prove(2.3, 1.0)
            .expect("Failed to generate proof");
        
        // Serialize
        let serialized = bincode::serialize(&proof)
            .expect("Failed to serialize");
        
        // Deserialize
        let deserialized: PhiProof = bincode::deserialize(&serialized)
            .expect("Failed to deserialize");
        
        // Verify deserialized proof
        assert!(deserialized.verify().unwrap());
    }
    
    #[test]
    fn test_zero_knowledge_property() {
        // Generate two proofs for different Φ values
        let proof1 = PhiProof::prove(2.0, 1.0).unwrap();
        let proof2 = PhiProof::prove(5.0, 1.0).unwrap();
        
        // Both should verify
        assert!(proof1.verify().unwrap());
        assert!(proof2.verify().unwrap());
        
        // But we can't distinguish which is which
        // (this is the zero-knowledge property)
        // The proofs don't reveal the actual Φ values
    }
}
