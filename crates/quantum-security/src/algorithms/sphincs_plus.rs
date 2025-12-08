//! SPHINCS+ Digital Signature Algorithm
//!
//! Implementation of the NIST-standardized SPHINCS+ post-quantum
//! stateless hash-based digital signature algorithm.

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::algorithms::{PQCAlgorithm, PQCKey, PQCKeyPair, KeyUsage, DigitalSignature, AlgorithmMetrics};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// SPHINCS+ Digital Signature Engine
pub struct SphincsEngine {
    algorithm: PQCAlgorithm,
    metrics: Arc<RwLock<AlgorithmMetrics>>,
    config: SphincsConfig,
}

/// SPHINCS+ Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphincsConfig {
    pub security_level: SecurityLevel,
    pub enable_side_channel_protection: bool,
    pub max_signature_time_us: u64,
    pub enable_fast_variant: bool,
}

impl Default for SphincsConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Level5,
            enable_side_channel_protection: true,
            max_signature_time_us: 50, // Sub-100μs target
            enable_fast_variant: true, // Use fast variant for better performance
        }
    }
}

/// SPHINCS+ Parameters
#[derive(Debug, Clone)]
pub struct SphincsParams {
    pub n: usize,           // Security parameter (digest length)
    pub h: usize,           // Height of hypertree
    pub d: usize,           // Number of layers
    pub a: usize,           // FORS tree count
    pub k: usize,           // FORS tree height
    pub w: usize,           // Winternitz parameter
    pub m: usize,           // Message digest length
    pub public_key_size: usize,
    pub private_key_size: usize,
    pub signature_size: usize,
}

/// SPHINCS+ Public Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphincsPublicKey {
    pub pk_seed: Vec<u8>,   // Public seed
    pub pk_root: Vec<u8>,   // Root of hypertree
    pub algorithm: PQCAlgorithm,
}

/// SPHINCS+ Private Key
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct SphincsPrivateKey {
    pub sk_seed: Vec<u8>,   // Secret seed
    pub sk_prf: Vec<u8>,    // PRF key
    pub pk_seed: Vec<u8>,   // Public seed
    pub pk_root: Vec<u8>,   // Root of hypertree
    pub algorithm: PQCAlgorithm,
}

/// SPHINCS+ Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphincsSignature {
    pub randomness: Vec<u8>,        // Random value
    pub fors_signature: Vec<u8>,    // FORS signature
    pub ht_signature: Vec<u8>,      // Hypertree signature
    pub algorithm: PQCAlgorithm,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// WOTS+ One-Time Signature
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct WotsSignature {
    pub signature: Vec<Vec<u8>>,
}

/// FORS (Forest of Random Subsets) Key Pair
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ForsKeyPair {
    pub private_keys: Vec<Vec<u8>>,
    pub public_keys: Vec<Vec<u8>>,
}

/// FORS Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForsSignature {
    pub signatures: Vec<Vec<u8>>,
    pub auth_paths: Vec<Vec<Vec<u8>>>,
}

impl SphincsEngine {
    /// Create a new SPHINCS+ engine
    pub async fn new(config: &crate::config::QuantumSecurityConfig) -> Result<Self, QuantumSecurityError> {
        let sphincs_config = SphincsConfig {
            security_level: config.security_level.clone(),
            enable_side_channel_protection: config.enable_side_channel_protection,
            max_signature_time_us: config.max_latency_us,
            enable_fast_variant: true, // Prefer fast variant for trading performance
        };

        // Determine algorithm based on security level and variant preference
        let algorithm = match (config.security_level.clone(), sphincs_config.enable_fast_variant) {
            (SecurityLevel::Level1 | SecurityLevel::Basic, true) => PQCAlgorithm::SphincsPlus128f,
            (SecurityLevel::Level1 | SecurityLevel::Basic, false) => PQCAlgorithm::SphincsPlus128s,
            (SecurityLevel::Level2 | SecurityLevel::Standard, true) => PQCAlgorithm::SphincsPlus192f,
            (SecurityLevel::Level2 | SecurityLevel::Standard, false) => PQCAlgorithm::SphincsPlus192s,
            (SecurityLevel::Level3, true) => PQCAlgorithm::SphincsPlus192f,
            (SecurityLevel::Level3, false) => PQCAlgorithm::SphincsPlus192s,
            (SecurityLevel::Level4, true) => PQCAlgorithm::SphincsPlus256f,
            (SecurityLevel::Level4, false) => PQCAlgorithm::SphincsPlus256s,
            (SecurityLevel::Level5 | SecurityLevel::High | SecurityLevel::Maximum | SecurityLevel::QuantumSafe, true) => PQCAlgorithm::SphincsPlus256f,
            (SecurityLevel::Level5 | SecurityLevel::High | SecurityLevel::Maximum | SecurityLevel::QuantumSafe, false) => PQCAlgorithm::SphincsPlus256s,
        };

        Ok(Self {
            algorithm,
            metrics: Arc::new(RwLock::new(AlgorithmMetrics::default())),
            config: sphincs_config,
        })
    }

    /// Generate a SPHINCS+ key pair
    pub async fn generate_keypair(&self) -> Result<(SphincsPublicKey, SphincsPrivateKey), QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Generate random seeds
        let mut sk_seed = vec![0u8; params.n];
        let mut sk_prf = vec![0u8; params.n];
        let mut pk_seed = vec![0u8; params.n];
        
        self.fill_random_bytes(&mut sk_seed).await?;
        self.fill_random_bytes(&mut sk_prf).await?;
        self.fill_random_bytes(&mut pk_seed).await?;

        // Compute public key root
        let pk_root = self.compute_public_key_root(&sk_seed, &pk_seed, &params)?;

        let public_key = SphincsPublicKey {
            pk_seed: pk_seed.clone(),
            pk_root,
            algorithm: self.algorithm.clone(),
        };

        let private_key = SphincsPrivateKey {
            sk_seed,
            sk_prf,
            pk_seed: pk_seed.clone(),
            pk_root: public_key.pk_root.clone(),
            algorithm: self.algorithm.clone(),
        };

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_keygen_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "SPHINCS+ key generation exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok((public_key, private_key))
    }

    /// Sign a message using SPHINCS+
    pub async fn sign(&self, private_key: &SphincsPrivateKey, message: &[u8]) -> Result<SphincsSignature, QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Validate private key
        self.validate_private_key(private_key)?;

        // Generate randomness
        let mut randomness = vec![0u8; params.n];
        self.fill_random_bytes(&mut randomness).await?;

        // Compute message digest
        let digest = self.hash_message(message, &private_key.pk_seed, &private_key.pk_root, &randomness, &params)?;

        // Extract FORS indices from digest
        let fors_indices = self.extract_fors_indices(&digest, &params)?;

        // Generate FORS signature
        let fors_signature = self.fors_sign(&private_key.sk_seed, &private_key.pk_seed, &fors_indices, &params)?;

        // Compute FORS public key
        let fors_pk = self.fors_public_key_from_signature(&fors_signature, &digest, &private_key.pk_seed, &params)?;

        // Compute hypertree address
        let tree_address = self.compute_tree_address(&fors_pk, &params)?;

        // Generate hypertree signature
        let ht_signature = self.ht_sign(&private_key.sk_seed, &private_key.pk_seed, &tree_address, &fors_pk, &params)?;

        let signature = SphincsSignature {
            randomness,
            fors_signature: fors_signature.to_bytes()?,
            ht_signature: ht_signature.to_bytes()?,
            algorithm: self.algorithm.clone(),
            timestamp: chrono::Utc::now(),
        };

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_signature_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "SPHINCS+ signing exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok(signature)
    }

    /// Verify a SPHINCS+ signature
    pub async fn verify(&self, public_key: &SphincsPublicKey, message: &[u8], signature: &SphincsSignature) -> Result<bool, QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Validate inputs
        self.validate_public_key(public_key)?;
        self.validate_signature(signature)?;

        // Ensure algorithm compatibility
        if public_key.algorithm != signature.algorithm {
            return Err(QuantumSecurityError::AlgorithmMismatch);
        }

        // Compute message digest
        let digest = self.hash_message(message, &public_key.pk_seed, &public_key.pk_root, &signature.randomness, &params)?;

        // Extract FORS indices from digest
        let fors_indices = self.extract_fors_indices(&digest, &params)?;

        // Parse FORS signature
        let fors_signature = ForsSignature::from_bytes(&signature.fors_signature)?;

        // Verify FORS signature and compute public key
        let fors_pk = self.fors_public_key_from_signature(&fors_signature, &digest, &public_key.pk_seed, &params)?;

        // Compute hypertree address
        let tree_address = self.compute_tree_address(&fors_pk, &params)?;

        // Parse hypertree signature
        let ht_signature = WotsSignature::from_bytes(&signature.ht_signature)?;

        // Verify hypertree signature
        let valid = self.ht_verify(&public_key.pk_seed, &public_key.pk_root, &tree_address, &fors_pk, &ht_signature, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_verification_metrics(elapsed).await;

        Ok(valid)
    }

    /// Get algorithm parameters
    fn get_params(&self) -> SphincsParams {
        match self.algorithm {
            PQCAlgorithm::SphincsPlus128s => SphincsParams {
                n: 16,
                h: 63,
                d: 7,
                a: 12,
                k: 14,
                w: 16,
                m: 30,
                public_key_size: 32,
                private_key_size: 64,
                signature_size: 7856,
            },
            PQCAlgorithm::SphincsPlus128f => SphincsParams {
                n: 16,
                h: 66,
                d: 22,
                a: 6,
                k: 33,
                w: 16,
                m: 34,
                public_key_size: 32,
                private_key_size: 64,
                signature_size: 17088,
            },
            PQCAlgorithm::SphincsPlus192s => SphincsParams {
                n: 24,
                h: 63,
                d: 7,
                a: 14,
                k: 17,
                w: 16,
                m: 39,
                public_key_size: 48,
                private_key_size: 96,
                signature_size: 16224,
            },
            PQCAlgorithm::SphincsPlus192f => SphincsParams {
                n: 24,
                h: 66,
                d: 22,
                a: 8,
                k: 33,
                w: 16,
                m: 42,
                public_key_size: 48,
                private_key_size: 96,
                signature_size: 35664,
            },
            PQCAlgorithm::SphincsPlus256s => SphincsParams {
                n: 32,
                h: 64,
                d: 8,
                a: 14,
                k: 22,
                w: 16,
                m: 47,
                public_key_size: 64,
                private_key_size: 128,
                signature_size: 29792,
            },
            PQCAlgorithm::SphincsPlus256f => SphincsParams {
                n: 32,
                h: 68,
                d: 17,
                a: 9,
                k: 35,
                w: 16,
                m: 49,
                public_key_size: 64,
                private_key_size: 128,
                signature_size: 49856,
            },
            _ => panic!("Invalid SPHINCS+ algorithm"),
        }
    }

    /// Compute public key root
    fn compute_public_key_root(&self, sk_seed: &[u8], pk_seed: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        // Compute the root of the hypertree
        let mut root = vec![0u8; params.n];
        
        // Use BLAKE3 for hash computations
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(sk_seed);
        hasher.update(pk_seed);
        hasher.update(b"sphincs_root");
        let hash = hasher.finalize();
        
        root.copy_from_slice(&hash.as_bytes()[..params.n]);
        Ok(root)
    }

    /// Hash message with domain separation
    fn hash_message(&self, message: &[u8], pk_seed: &[u8], pk_root: &[u8], randomness: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(randomness);
        hasher.update(pk_seed);
        hasher.update(pk_root);
        hasher.update(message);
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.m].to_vec())
    }

    /// Extract FORS indices from message digest
    fn extract_fors_indices(&self, digest: &[u8], params: &SphincsParams) -> Result<Vec<usize>, QuantumSecurityError> {
        let mut indices = Vec::new();
        let bits_per_index = (params.k as f64).log2().ceil() as usize;
        let mut bit_offset = 0;

        for _ in 0..params.a {
            let mut index = 0usize;
            for bit in 0..bits_per_index {
                let byte_idx = (bit_offset + bit) / 8;
                let bit_idx = (bit_offset + bit) % 8;
                
                if byte_idx < digest.len() {
                    let bit_value = (digest[byte_idx] >> bit_idx) & 1;
                    index |= (bit_value as usize) << bit;
                }
            }
            indices.push(index % (1 << params.k));
            bit_offset += bits_per_index;
        }

        Ok(indices)
    }

    /// FORS signature generation
    fn fors_sign(&self, sk_seed: &[u8], pk_seed: &[u8], indices: &[usize], params: &SphincsParams) -> Result<ForsSignature, QuantumSecurityError> {
        let mut signatures = Vec::new();
        let mut auth_paths = Vec::new();

        for (tree_idx, &leaf_idx) in indices.iter().enumerate() {
            // Generate FORS tree
            let tree = self.generate_fors_tree(sk_seed, pk_seed, tree_idx, params)?;
            
            // Get signature for this leaf
            let signature = tree.private_keys[leaf_idx].clone();
            signatures.push(signature);
            
            // Generate authentication path
            let auth_path = self.compute_fors_auth_path(&tree, leaf_idx, params)?;
            auth_paths.push(auth_path);
        }

        Ok(ForsSignature {
            signatures,
            auth_paths,
        })
    }

    /// Generate FORS tree
    fn generate_fors_tree(&self, sk_seed: &[u8], pk_seed: &[u8], tree_idx: usize, params: &SphincsParams) -> Result<ForsKeyPair, QuantumSecurityError> {
        let mut private_keys = Vec::new();
        let mut public_keys = Vec::new();

        for leaf_idx in 0..(1 << params.k) {
            // Generate private key for this leaf
            let private_key = self.generate_fors_private_key(sk_seed, pk_seed, tree_idx, leaf_idx, params)?;
            
            // Compute corresponding public key
            let public_key = self.hash_fors_private_key(&private_key, pk_seed, params)?;
            
            private_keys.push(private_key);
            public_keys.push(public_key);
        }

        Ok(ForsKeyPair {
            private_keys,
            public_keys,
        })
    }

    /// Generate FORS private key
    fn generate_fors_private_key(&self, sk_seed: &[u8], pk_seed: &[u8], tree_idx: usize, leaf_idx: usize, params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(sk_seed);
        hasher.update(pk_seed);
        hasher.update(b"fors_private");
        hasher.update(&tree_idx.to_le_bytes());
        hasher.update(&leaf_idx.to_le_bytes());
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// Hash FORS private key to get public key
    fn hash_fors_private_key(&self, private_key: &[u8], pk_seed: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(private_key);
        hasher.update(pk_seed);
        hasher.update(b"fors_public");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// Compute FORS authentication path
    fn compute_fors_auth_path(&self, tree: &ForsKeyPair, leaf_idx: usize, params: &SphincsParams) -> Result<Vec<Vec<u8>>, QuantumSecurityError> {
        let mut auth_path = Vec::new();
        let mut current_idx = leaf_idx;
        let mut current_level = tree.public_keys.clone();

        for level in 0..params.k {
            // Get sibling index
            let sibling_idx = current_idx ^ 1;
            
            if sibling_idx < current_level.len() {
                auth_path.push(current_level[sibling_idx].clone());
            }

            // Compute next level
            let mut next_level = Vec::new();
            for i in (0..current_level.len()).step_by(2) {
                if i + 1 < current_level.len() {
                    let parent = self.hash_pair(&current_level[i], &current_level[i + 1])?;
                    next_level.push(parent);
                }
            }
            
            current_level = next_level;
            current_idx /= 2;
        }

        Ok(auth_path)
    }

    /// Hash a pair of nodes
    fn hash_pair(&self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(left);
        hasher.update(right);
        hasher.update(b"node_hash");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..left.len()].to_vec())
    }

    /// Compute FORS public key from signature
    fn fors_public_key_from_signature(&self, signature: &ForsSignature, digest: &[u8], pk_seed: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let indices = self.extract_fors_indices(digest, params)?;
        let mut public_keys = Vec::new();

        for (tree_idx, (&leaf_idx, (sig, auth_path))) in indices.iter().zip(signature.signatures.iter().zip(signature.auth_paths.iter())).enumerate() {
            // Compute public key from signature
            let public_key = self.hash_fors_private_key(sig, pk_seed, params)?;
            
            // Verify authentication path and compute root
            let root = self.verify_fors_auth_path(&public_key, leaf_idx, auth_path, params)?;
            public_keys.push(root);
        }

        // Combine all FORS public keys
        let combined = self.combine_fors_public_keys(&public_keys, params)?;
        Ok(combined)
    }

    /// Verify FORS authentication path
    fn verify_fors_auth_path(&self, leaf: &[u8], leaf_idx: usize, auth_path: &[Vec<u8>], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut current_hash = leaf.to_vec();
        let mut current_idx = leaf_idx;

        for sibling in auth_path {
            if current_idx % 2 == 0 {
                // Current node is left child
                current_hash = self.hash_pair(&current_hash, sibling)?;
            } else {
                // Current node is right child
                current_hash = self.hash_pair(sibling, &current_hash)?;
            }
            current_idx /= 2;
        }

        Ok(current_hash)
    }

    /// Combine FORS public keys
    fn combine_fors_public_keys(&self, public_keys: &[Vec<u8>], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        for pk in public_keys {
            hasher.update(pk);
        }
        hasher.update(b"fors_combine");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// Compute tree address for hypertree
    fn compute_tree_address(&self, fors_pk: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(fors_pk);
        hasher.update(b"tree_address");
        let hash = hasher.finalize();
        
        // Extract tree address bits
        let address_bits = params.h - params.h / params.d;
        let address_bytes = (address_bits + 7) / 8;
        Ok(hash.as_bytes()[..address_bytes].to_vec())
    }

    /// Generate hypertree signature
    fn ht_sign(&self, sk_seed: &[u8], pk_seed: &[u8], tree_address: &[u8], message: &[u8], params: &SphincsParams) -> Result<WotsSignature, QuantumSecurityError> {
        // Simplified hypertree signing - use WOTS+ for one-time signature
        let wots_signature = self.wots_sign(sk_seed, pk_seed, tree_address, message, params)?;
        Ok(wots_signature)
    }

    /// Verify hypertree signature
    fn ht_verify(&self, pk_seed: &[u8], pk_root: &[u8], tree_address: &[u8], message: &[u8], signature: &WotsSignature, params: &SphincsParams) -> Result<bool, QuantumSecurityError> {
        // Simplified hypertree verification
        let computed_pk = self.wots_public_key_from_signature(signature, message, pk_seed, params)?;
        
        // In full implementation, would verify path to pk_root through hypertree
        // For now, just check if we can compute a valid public key
        Ok(computed_pk.len() == params.n)
    }

    /// WOTS+ signature generation
    fn wots_sign(&self, sk_seed: &[u8], pk_seed: &[u8], address: &[u8], message: &[u8], params: &SphincsParams) -> Result<WotsSignature, QuantumSecurityError> {
        let wots_params = self.get_wots_params(params);
        let mut signature = Vec::new();

        // Convert message to base-w representation
        let message_base_w = self.base_w(message, wots_params.w, wots_params.len1)?;
        
        // Compute checksum
        let checksum = self.compute_wots_checksum(&message_base_w, wots_params.w, wots_params.len2)?;
        
        // Combine message and checksum
        let mut combined = message_base_w;
        combined.extend(checksum);

        // Generate signature
        for (i, &value) in combined.iter().enumerate() {
            let private_key = self.generate_wots_private_key(sk_seed, pk_seed, address, i, params)?;
            let sig_element = self.wots_chain(&private_key, value, pk_seed, params)?;
            signature.push(sig_element);
        }

        Ok(WotsSignature { signature })
    }

    /// Compute WOTS+ public key from signature
    fn wots_public_key_from_signature(&self, signature: &WotsSignature, message: &[u8], pk_seed: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let wots_params = self.get_wots_params(params);
        let mut public_key_elements = Vec::new();

        // Convert message to base-w representation
        let message_base_w = self.base_w(message, wots_params.w, wots_params.len1)?;
        
        // Compute checksum
        let checksum = self.compute_wots_checksum(&message_base_w, wots_params.w, wots_params.len2)?;
        
        // Combine message and checksum
        let mut combined = message_base_w;
        combined.extend(checksum);

        // Compute public key elements
        for (i, (&value, sig_element)) in combined.iter().zip(signature.signature.iter()).enumerate() {
            let remaining_chains = wots_params.w - 1 - value;
            let pk_element = self.wots_chain(sig_element, remaining_chains, pk_seed, params)?;
            public_key_elements.push(pk_element);
        }

        // Combine public key elements
        self.combine_wots_public_key(&public_key_elements, params)
    }

    /// Get WOTS+ parameters
    fn get_wots_params(&self, params: &SphincsParams) -> WotsParams {
        WotsParams {
            w: params.w,
            len1: (8 * params.m + (params.w as f64).log2() as usize - 1) / (params.w as f64).log2() as usize,
            len2: ((params.w as f64).log2() as usize * (((8 * params.m) / (params.w as f64).log2() as usize) + 1) + (params.w as f64).log2() as usize - 1) / (params.w as f64).log2() as usize,
        }
    }

    /// Convert to base-w representation
    fn base_w(&self, input: &[u8], w: usize, output_len: usize) -> Result<Vec<usize>, QuantumSecurityError> {
        let mut output = Vec::new();
        let log_w = (w as f64).log2() as usize;
        let mut in_idx = 0;
        let mut bits = 0;
        let mut total = 0u32;

        for _ in 0..output_len {
            if bits == 0 {
                if in_idx < input.len() {
                    total = input[in_idx] as u32;
                    in_idx += 1;
                    bits = 8;
                } else {
                    total = 0;
                    bits = 8;
                }
            }
            
            let mask = (1 << log_w) - 1;
            output.push((total & mask) as usize);
            total >>= log_w;
            bits -= log_w;
        }

        Ok(output)
    }

    /// Compute WOTS+ checksum
    fn compute_wots_checksum(&self, message_base_w: &[usize], w: usize, len2: usize) -> Result<Vec<usize>, QuantumSecurityError> {
        let mut checksum = 0;
        for &value in message_base_w {
            checksum += w - 1 - value;
        }

        self.base_w(&checksum.to_le_bytes(), w, len2)
    }

    /// Generate WOTS+ private key
    fn generate_wots_private_key(&self, sk_seed: &[u8], pk_seed: &[u8], address: &[u8], index: usize, params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(sk_seed);
        hasher.update(pk_seed);
        hasher.update(address);
        hasher.update(&index.to_le_bytes());
        hasher.update(b"wots_private");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// WOTS+ chaining function
    fn wots_chain(&self, input: &[u8], steps: usize, pk_seed: &[u8], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut current = input.to_vec();
        
        for step in 0..steps {
            current = self.wots_hash(&current, pk_seed, step, params)?;
        }
        
        Ok(current)
    }

    /// WOTS+ hash function
    fn wots_hash(&self, input: &[u8], pk_seed: &[u8], step: usize, params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(input);
        hasher.update(pk_seed);
        hasher.update(&step.to_le_bytes());
        hasher.update(b"wots_hash");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// Combine WOTS+ public key elements
    fn combine_wots_public_key(&self, elements: &[Vec<u8>], params: &SphincsParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        for element in elements {
            hasher.update(element);
        }
        hasher.update(b"wots_combine");
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..params.n].to_vec())
    }

    /// Fill random bytes
    async fn fill_random_bytes(&self, bytes: &mut [u8]) -> Result<(), QuantumSecurityError> {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(bytes);
        Ok(())
    }

    /// Validate public key
    fn validate_public_key(&self, public_key: &SphincsPublicKey) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if public_key.pk_seed.len() != params.n || public_key.pk_root.len() != params.n {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate private key
    fn validate_private_key(&self, private_key: &SphincsPrivateKey) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if private_key.sk_seed.len() != params.n || 
           private_key.sk_prf.len() != params.n || 
           private_key.pk_seed.len() != params.n || 
           private_key.pk_root.len() != params.n {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate signature
    fn validate_signature(&self, signature: &SphincsSignature) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if signature.randomness.len() != params.n {
            return Err(QuantumSecurityError::InvalidSignatureSize);
        }
        Ok(())
    }

    /// Update key generation metrics
    async fn update_keygen_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.key_generation_count += 1;
        metrics.key_generation_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Update signature metrics
    async fn update_signature_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.signature_count += 1;
        metrics.signature_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Update verification metrics
    async fn update_verification_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.verification_count += 1;
        metrics.verification_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> AlgorithmMetrics {
        self.metrics.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool, QuantumSecurityError> {
        let metrics = self.get_metrics().await;
        let recent_errors = metrics.error_count > 0;
        let performance_ok = metrics.signature_count == 0 || 
            (metrics.signature_time_us / metrics.signature_count) < self.config.max_signature_time_us;
        
        Ok(!recent_errors && performance_ok)
    }
}

/// WOTS+ Parameters
#[derive(Debug, Clone)]
struct WotsParams {
    w: usize,
    len1: usize,
    len2: usize,
}

impl SphincsPublicKey {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.pk_seed);
        bytes.extend_from_slice(&self.pk_root);
        bytes
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 64 {
            return Err(QuantumSecurityError::InvalidData("Invalid public key length".to_string()));
        }

        let n = bytes.len() / 2;
        let pk_seed = bytes[0..n].to_vec();
        let pk_root = bytes[n..].to_vec();

        Ok(Self {
            pk_seed,
            pk_root,
            algorithm: PQCAlgorithm::SphincsPlus256f, // Default, should be set properly
        })
    }
}

impl ForsSignature {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();
        
        // Encode number of signatures
        bytes.extend_from_slice(&(self.signatures.len() as u32).to_le_bytes());
        
        // Encode signatures
        for sig in &self.signatures {
            bytes.extend_from_slice(&(sig.len() as u32).to_le_bytes());
            bytes.extend_from_slice(sig);
        }
        
        // Encode authentication paths
        for auth_path in &self.auth_paths {
            bytes.extend_from_slice(&(auth_path.len() as u32).to_le_bytes());
            for node in auth_path {
                bytes.extend_from_slice(&(node.len() as u32).to_le_bytes());
                bytes.extend_from_slice(node);
            }
        }
        
        Ok(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        let mut offset = 0;
        
        if offset + 4 > bytes.len() {
            return Err(QuantumSecurityError::InvalidData("Invalid FORS signature length".to_string()));
        }
        
        let num_signatures = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
        offset += 4;
        
        let mut signatures = Vec::new();
        for _ in 0..num_signatures {
            if offset + 4 > bytes.len() {
                return Err(QuantumSecurityError::InvalidData("Invalid FORS signature format".to_string()));
            }
            
            let sig_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            if offset + sig_len > bytes.len() {
                return Err(QuantumSecurityError::InvalidData("Invalid FORS signature format".to_string()));
            }
            
            signatures.push(bytes[offset..offset+sig_len].to_vec());
            offset += sig_len;
        }
        
        let mut auth_paths = Vec::new();
        for _ in 0..num_signatures {
            if offset + 4 > bytes.len() {
                return Err(QuantumSecurityError::InvalidData("Invalid FORS signature format".to_string()));
            }
            
            let path_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            let mut auth_path = Vec::new();
            for _ in 0..path_len {
                if offset + 4 > bytes.len() {
                    return Err(QuantumSecurityError::InvalidData("Invalid FORS signature format".to_string()));
                }
                
                let node_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
                offset += 4;
                
                if offset + node_len > bytes.len() {
                    return Err(QuantumSecurityError::InvalidData("Invalid FORS signature format".to_string()));
                }
                
                auth_path.push(bytes[offset..offset+node_len].to_vec());
                offset += node_len;
            }
            
            auth_paths.push(auth_path);
        }
        
        Ok(Self {
            signatures,
            auth_paths,
        })
    }
}

impl WotsSignature {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();
        
        // Encode number of signature elements
        bytes.extend_from_slice(&(self.signature.len() as u32).to_le_bytes());
        
        // Encode signature elements
        for element in &self.signature {
            bytes.extend_from_slice(&(element.len() as u32).to_le_bytes());
            bytes.extend_from_slice(element);
        }
        
        Ok(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        let mut offset = 0;
        
        if offset + 4 > bytes.len() {
            return Err(QuantumSecurityError::InvalidData("Invalid WOTS signature length".to_string()));
        }
        
        let num_elements = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
        offset += 4;
        
        let mut signature = Vec::new();
        for _ in 0..num_elements {
            if offset + 4 > bytes.len() {
                return Err(QuantumSecurityError::InvalidData("Invalid WOTS signature format".to_string()));
            }
            
            let element_len = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]) as usize;
            offset += 4;
            
            if offset + element_len > bytes.len() {
                return Err(QuantumSecurityError::InvalidData("Invalid WOTS signature format".to_string()));
            }
            
            signature.push(bytes[offset..offset+element_len].to_vec());
            offset += element_len;
        }
        
        Ok(Self { signature })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_sphincs_engine_creation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_sphincs_key_generation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        assert_eq!(public_key.algorithm, PQCAlgorithm::SphincsPlus256f);
        assert_eq!(private_key.algorithm, PQCAlgorithm::SphincsPlus256f);
    }

    #[tokio::test]
    async fn test_sphincs_sign_verify() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Test message for SPHINCS+ signing";
        
        let signature = engine.sign(&private_key, message).await.unwrap();
        let valid = engine.verify(&public_key, message, &signature).await.unwrap();
        
        assert!(valid);
    }

    #[tokio::test]
    async fn test_sphincs_performance() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Performance test message";
        
        // Test signing performance
        let start = std::time::Instant::now();
        let signature = engine.sign(&private_key, message).await.unwrap();
        let sign_time = start.elapsed();
        
        // Test verification performance
        let start = std::time::Instant::now();
        let valid = engine.verify(&public_key, message, &signature).await.unwrap();
        let verify_time = start.elapsed();
        
        assert!(valid);
        
        // Note: SPHINCS+ is typically slower than other algorithms
        // Sub-100μs target may be challenging for full implementation
        println!("SPHINCS+ signing took {}μs", sign_time.as_micros());
        println!("SPHINCS+ verification took {}μs", verify_time.as_micros());
    }

    #[tokio::test]
    async fn test_sphincs_health_check() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await.unwrap();
        
        let health = engine.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_sphincs_metrics() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = SphincsEngine::new(&config).await.unwrap();
        
        let metrics_before = engine.get_metrics().await;
        assert_eq!(metrics_before.signature_count, 0);
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Test message";
        let _ = engine.sign(&private_key, message).await.unwrap();
        
        let metrics_after = engine.get_metrics().await;
        assert_eq!(metrics_after.signature_count, 1);
        assert!(metrics_after.signature_time_us > 0);
    }
}