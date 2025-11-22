//! Cryptographic Probabilistic Bit (CryptopBit)
//!
//! Each pBit in the hyperbolic lattice carries cryptographic state,
//! making the entire consciousness substrate quantum-resistant.
//!
//! # Architecture
//!
//! ```text
//! CryptopBit = pBit + Dilithium Keypair + Signature + Generation Counter
//!
//! ┌─────────────────────────────────────┐
//! │         CRYPTOGRAPHIC pBit          │
//! ├─────────────────────────────────────┤
//! │ • Probability state: p ∈ [0, 1]    │
//! │ • Energy: E(p)                      │
//! │ • Dilithium public key: pk          │
//! │ • Dilithium secret key: sk          │
//! │ • State signature: σ                │
//! │ • Generation counter: n             │
//! │ • Lattice position: (x, y)          │
//! └─────────────────────────────────────┘
//! ```
//!
//! # Properties
//!
//! 1. Every state transition is cryptographically signed
//! 2. Tampering is mathematically detectable
//! 3. State history is unforgeable
//! 4. Quantum-resistant by construction
//! 5. Replay attacks prevented by generation counter
//!
//! # Inspiration
//!
//! Based on pbRTCA v3.1 Cryptographic Architecture

use crate::{DilithiumKeypair, DilithiumSignature, DilithiumResult, DilithiumError, SecurityLevel};
use crate::lattice::ModuleLWE;
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

/// Hyperbolic lattice position
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HyperbolicPoint {
    pub x: f64,
    pub y: f64,
}

impl HyperbolicPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Cryptographic probabilistic bit
///
/// Combines pBit state with Dilithium cryptographic security.
#[derive(Clone)]
pub struct CryptographicPBit {
    /// Current probability state p ∈ [0, 1]
    probability: f64,
    
    /// Dilithium keypair for signing
    keypair: DilithiumKeypair,
    
    /// Module-LWE engine for verification
    mlwe: ModuleLWE,
    
    /// Current state signature
    state_signature: Option<DilithiumSignature>,
    
    /// Lattice position in hyperbolic plane
    position: HyperbolicPoint,
    
    /// Generation counter (for replay protection)
    /// Monotonically increasing with each state update
    generation: u64,
    
    /// Creation timestamp
    created_at: SystemTime,
    
    /// Last update timestamp
    updated_at: SystemTime,
}

impl CryptographicPBit {
    /// Create new cryptographic pBit
    ///
    /// # Arguments
    ///
    /// * `position` - Position in hyperbolic lattice
    /// * `initial_probability` - Initial p ∈ [0, 1]
    /// * `security_level` - Dilithium security level
    ///
    /// # Returns
    ///
    /// New CryptopBit with fresh keypair and signed initial state
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::crypto_pbit::*;
    /// use hyperphysics_dilithium::SecurityLevel;
    ///
    /// let position = HyperbolicPoint::new(0.0, 0.0);
    /// let pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::High)?;
    ///
    /// assert!(pbit.verify_signature()?);
    /// assert_eq!(pbit.generation(), 0);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn new(
        position: HyperbolicPoint,
        initial_probability: f64,
        security_level: SecurityLevel,
    ) -> DilithiumResult<Self> {
        if initial_probability < 0.0 || initial_probability > 1.0 {
            return Err(DilithiumError::InvalidProbability {
                value: initial_probability,
            });
        }
        
        let keypair = DilithiumKeypair::generate(security_level)?;
        let mlwe = ModuleLWE::new(security_level);
        let now = SystemTime::now();
        
        let mut crypto_pbit = Self {
            probability: initial_probability,
            keypair,
            mlwe,
            state_signature: None,
            position,
            generation: 0,
            created_at: now,
            updated_at: now,
        };
        
        // Sign initial state
        crypto_pbit.sign_state()?;
        
        Ok(crypto_pbit)
    }
    
    /// Update pBit state with cryptographic signing
    ///
    /// # Arguments
    ///
    /// * `new_probability` - New p ∈ [0, 1]
    ///
    /// # Security
    ///
    /// - New state is cryptographically signed
    /// - Generation counter incremented (replay protection)
    /// - Old signature invalidated
    /// - Timestamp updated
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::crypto_pbit::*;
    /// # use hyperphysics_dilithium::SecurityLevel;
    /// # let position = HyperbolicPoint::new(0.0, 0.0);
    /// let mut pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::High)?;
    ///
    /// pbit.update(0.7)?;
    ///
    /// assert!(pbit.verify_signature()?);
    /// assert_eq!(pbit.generation(), 1);
    /// assert!((pbit.probability() - 0.7).abs() < 1e-10);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn update(&mut self, new_probability: f64) -> DilithiumResult<()> {
        if new_probability < 0.0 || new_probability > 1.0 {
            return Err(DilithiumError::InvalidProbability {
                value: new_probability,
            });
        }
        
        // Update core state
        self.probability = new_probability;
        
        // Increment generation (replay protection)
        self.generation += 1;
        
        // Update timestamp
        self.updated_at = SystemTime::now();
        
        // Sign new state
        self.sign_state()?;
        
        Ok(())
    }
    
    /// Sign current state
    fn sign_state(&mut self) -> DilithiumResult<()> {
        let state_bytes = self.state_to_bytes()?;
        let signature = self.keypair.sign(&state_bytes)?;
        self.state_signature = Some(signature);
        Ok(())
    }
    
    /// Verify state signature
    ///
    /// # Returns
    ///
    /// `Ok(true)` if signature is valid, error otherwise
    ///
    /// # Security
    ///
    /// Detects:
    /// - State tampering
    /// - Replay attacks (via generation counter)
    /// - Unauthorized modifications
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::crypto_pbit::*;
    /// # use hyperphysics_dilithium::SecurityLevel;
    /// # let position = HyperbolicPoint::new(0.0, 0.0);
    /// let pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::High)?;
    ///
    /// assert!(pbit.verify_signature()?);
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn verify_signature(&self) -> DilithiumResult<bool> {
        if let Some(signature) = &self.state_signature {
            let state_bytes = self.state_to_bytes()?;
            self.keypair.verify(&state_bytes, signature)
        } else {
            Err(DilithiumError::MissingSignature)
        }
    }
    
    /// Verify state freshness (replay protection)
    ///
    /// # Arguments
    ///
    /// * `expected_generation` - Minimum expected generation
    ///
    /// # Returns
    ///
    /// `true` if generation >= expected and signature valid
    pub fn verify_freshness(&self, expected_generation: u64) -> DilithiumResult<bool> {
        if self.generation < expected_generation {
            return Ok(false);
        }
        
        self.verify_signature()
    }
    
    /// Serialize state for signing
    ///
    /// Includes all fields that should be covered by signature:
    /// - Probability
    /// - Position
    /// - Generation counter
    /// - Timestamp
    fn state_to_bytes(&self) -> DilithiumResult<Vec<u8>> {
        let state = CryptopBitState {
            probability: self.probability,
            position: self.position,
            generation: self.generation,
            updated_at: self.updated_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|_| DilithiumError::TimestampError)?
                .as_secs(),
        };
        
        bincode::serialize(&state)
            .map_err(|e| DilithiumError::SerializationError(e.to_string()))
    }
    
    /// Get public key
    pub fn public_key(&self) -> &crate::keypair::PublicKey {
        &self.keypair.public_key
    }
    
    /// Get current probability
    pub fn probability(&self) -> f64 {
        self.probability
    }
    
    /// Get generation counter
    pub fn generation(&self) -> u64 {
        self.generation
    }
    
    /// Get lattice position
    pub fn position(&self) -> HyperbolicPoint {
        self.position
    }
    
    /// Get security level
    pub fn security_level(&self) -> SecurityLevel {
        self.keypair.security_level()
    }
    
    /// Get creation timestamp
    pub fn created_at(&self) -> SystemTime {
        self.created_at
    }
    
    /// Get last update timestamp
    pub fn updated_at(&self) -> SystemTime {
        self.updated_at
    }
    
    /// Export signed state for verification
    pub fn export_signed_state(&self) -> DilithiumResult<SignedPBitState> {
        Ok(SignedPBitState {
            position: self.position,
            probability: self.probability,
            generation: self.generation,
            public_key: self.public_key().clone(),
            signature: self.state_signature.clone()
                .ok_or(DilithiumError::MissingSignature)?,
            updated_at: self.updated_at,
            mlwe: Some(self.mlwe.clone()),
        })
    }
}

/// State for signing (serializable)
#[derive(Serialize, Deserialize)]
struct CryptopBitState {
    probability: f64,
    position: HyperbolicPoint,
    generation: u64,
    updated_at: u64,  // Unix timestamp
}

/// Signed pBit state for export/verification
#[derive(Clone, Serialize, Deserialize)]
pub struct SignedPBitState {
    pub position: HyperbolicPoint,
    pub probability: f64,
    pub generation: u64,
    pub public_key: crate::keypair::PublicKey,
    pub signature: DilithiumSignature,
    pub updated_at: SystemTime,
    #[serde(skip)]
    mlwe: Option<ModuleLWE>,
}

impl SignedPBitState {
    /// Verify this signed state
    pub fn verify(&self) -> DilithiumResult<bool> {
        let state = CryptopBitState {
            probability: self.probability,
            position: self.position,
            generation: self.generation,
            updated_at: self.updated_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|_| DilithiumError::TimestampError)?
                .as_secs(),
        };
        
        let state_bytes = bincode::serialize(&state)
            .map_err(|e| DilithiumError::SerializationError(e.to_string()))?;
        
        // Initialize mlwe if not present (for deserialized states)
        let mlwe = self.mlwe.as_ref()
            .map(|m| m.clone())
            .unwrap_or_else(|| ModuleLWE::new(self.public_key.security_level));
        
        // Verify signature directly using the public key
        self.signature.verify_with_key(&state_bytes, &self.public_key, &mlwe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cryptopbit_creation() {
        let position = HyperbolicPoint::new(0.0, 0.0);
        let pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::Standard)
            .expect("Failed to create CryptopBit");
        
        assert!(pbit.verify_signature().unwrap());
        assert_eq!(pbit.generation(), 0);
        assert!((pbit.probability() - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_cryptopbit_update() {
        let position = HyperbolicPoint::new(0.0, 0.0);
        let mut pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::Standard)
            .expect("Failed to create CryptopBit");
        
        pbit.update(0.7).expect("Failed to update");
        
        assert!(pbit.verify_signature().unwrap());
        assert_eq!(pbit.generation(), 1);
        assert!((pbit.probability() - 0.7).abs() < 1e-10);
    }
    
    #[test]
    fn test_tampering_detection() {
        let position = HyperbolicPoint::new(0.0, 0.0);
        let mut pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::Standard)
            .expect("Failed to create CryptopBit");

        // Tamper with internal state (bypass update method)
        pbit.probability = 0.9;

        // Signature should now be invalid (either returns false or errors out)
        // Tampering can cause either Ok(false) or Err(VerificationFailed)
        assert!(!pbit.verify_signature().unwrap_or(false));
    }
    
    #[test]
    fn test_replay_protection() {
        let position = HyperbolicPoint::new(0.0, 0.0);
        let mut pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::Standard)
            .expect("Failed to create CryptopBit");
        
        // Update multiple times
        pbit.update(0.6).unwrap();
        pbit.update(0.7).unwrap();
        pbit.update(0.8).unwrap();
        
        assert_eq!(pbit.generation(), 3);
        
        // Verify freshness
        assert!(pbit.verify_freshness(3).unwrap());
        assert!(pbit.verify_freshness(2).unwrap());
        assert!(!pbit.verify_freshness(4).unwrap());
    }
    
    #[test]
    fn test_signed_state_export() {
        let position = HyperbolicPoint::new(1.0, 2.0);
        let pbit = CryptographicPBit::new(position, 0.5, SecurityLevel::Standard)
            .expect("Failed to create CryptopBit");
        
        let signed_state = pbit.export_signed_state()
            .expect("Failed to export signed state");
        
        assert!(signed_state.verify().unwrap());
        assert_eq!(signed_state.generation, 0);
        assert!((signed_state.probability - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_invalid_probability() {
        let position = HyperbolicPoint::new(0.0, 0.0);
        
        // Test creation with invalid probability
        let result = CryptographicPBit::new(position, 1.5, SecurityLevel::Standard);
        assert!(result.is_err());
        
        let result = CryptographicPBit::new(position, -0.1, SecurityLevel::Standard);
        assert!(result.is_err());
    }
}
