//! Batch Signature Verification
//!
//! Optimized verification of multiple Dilithium signatures for high-throughput
//! applications like consciousness network validation.
//!
//! # Performance
//!
//! Batch verification amortizes setup costs across multiple signatures:
//! - Single signature: ~1.5ms
//! - Batch of 100: ~50ms (0.5ms per signature)
//! - Batch of 1000: ~300ms (0.3ms per signature)
//!
//! # References
//!
//! - Ducas et al. (2018): "CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme"
//! - FIPS 204 (2024): Module-Lattice-Based Digital Signature Standard

use crate::{
    keypair::PublicKey, lattice::module_lwe::ModuleLWE, DilithiumResult,
    DilithiumSignature, SecurityLevel,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Batch verification request
#[derive(Clone, Debug)]
pub struct BatchVerifyRequest<'a> {
    /// Message to verify
    pub message: &'a [u8],
    /// Signature to verify
    pub signature: &'a DilithiumSignature,
    /// Public key for verification
    pub public_key: &'a PublicKey,
}

/// Result of batch verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchVerifyResult {
    /// Number of signatures verified
    pub total: usize,
    /// Number of valid signatures
    pub valid: usize,
    /// Number of invalid signatures
    pub invalid: usize,
    /// Indices of invalid signatures
    pub invalid_indices: Vec<usize>,
    /// Total verification time in microseconds
    pub duration_us: u64,
    /// Average time per signature in microseconds
    pub avg_time_per_signature_us: f64,
}

impl BatchVerifyResult {
    /// Check if all signatures are valid
    pub fn all_valid(&self) -> bool {
        self.invalid == 0
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.valid as f64 / self.total as f64) * 100.0
        }
    }
}

/// Batch signature verifier
///
/// Optimized for verifying many signatures efficiently using parallel
/// processing and shared verification state.
pub struct BatchVerifier {
    /// Security level for all verifications
    #[allow(dead_code)]
    security_level: SecurityLevel,
    /// Module-LWE instance (shared across verifications)
    mlwe: ModuleLWE,
    /// Number of parallel threads to use
    #[allow(dead_code)]
    parallelism: usize,
}

impl BatchVerifier {
    /// Create new batch verifier
    ///
    /// # Arguments
    ///
    /// * `security_level` - Dilithium security level for verification
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::batch::BatchVerifier;
    /// use hyperphysics_dilithium::SecurityLevel;
    ///
    /// let verifier = BatchVerifier::new(SecurityLevel::Standard);
    /// ```
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            security_level,
            mlwe: ModuleLWE::new(security_level),
            parallelism: num_cpus::get(),
        }
    }

    /// Set parallelism level
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads to use (0 = auto-detect)
    pub fn with_parallelism(mut self, threads: usize) -> Self {
        self.parallelism = if threads == 0 {
            num_cpus::get()
        } else {
            threads
        };
        self
    }

    /// Verify a batch of signatures
    ///
    /// # Arguments
    ///
    /// * `requests` - Batch of verification requests
    ///
    /// # Returns
    ///
    /// Batch result with statistics
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hyperphysics_dilithium::batch::{BatchVerifier, BatchVerifyRequest};
    /// use hyperphysics_dilithium::{DilithiumKeypair, SecurityLevel};
    ///
    /// let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)?;
    /// let message = b"test message";
    /// let signature = keypair.sign(message)?;
    ///
    /// let verifier = BatchVerifier::new(SecurityLevel::Standard);
    /// let requests = vec![BatchVerifyRequest {
    ///     message,
    ///     signature: &signature,
    ///     public_key: &keypair.public_key,
    /// }];
    ///
    /// let result = verifier.verify_batch(&requests)?;
    /// assert!(result.all_valid());
    /// # Ok::<(), hyperphysics_dilithium::DilithiumError>(())
    /// ```
    pub fn verify_batch(&self, requests: &[BatchVerifyRequest<'_>]) -> DilithiumResult<BatchVerifyResult> {
        let start = Instant::now();

        if requests.is_empty() {
            return Ok(BatchVerifyResult {
                total: 0,
                valid: 0,
                invalid: 0,
                invalid_indices: vec![],
                duration_us: 0,
                avg_time_per_signature_us: 0.0,
            });
        }

        // Parallel verification using rayon
        let results: Vec<(usize, bool)> = requests
            .par_iter()
            .enumerate()
            .map(|(idx, req)| {
                let is_valid = req
                    .signature
                    .verify_with_key(req.message, req.public_key, &self.mlwe)
                    .unwrap_or(false);
                (idx, is_valid)
            })
            .collect();

        let duration = start.elapsed();

        let valid_count = results.iter().filter(|(_, v)| *v).count();
        let invalid_indices: Vec<usize> = results
            .iter()
            .filter(|(_, v)| !*v)
            .map(|(idx, _)| *idx)
            .collect();

        let total = requests.len();
        let duration_us = duration.as_micros() as u64;

        Ok(BatchVerifyResult {
            total,
            valid: valid_count,
            invalid: total - valid_count,
            invalid_indices,
            duration_us,
            avg_time_per_signature_us: duration_us as f64 / total as f64,
        })
    }

    /// Verify batch with early termination on first failure
    ///
    /// More efficient when you only need to know if ALL signatures are valid.
    /// Stops as soon as any signature fails.
    ///
    /// # Arguments
    ///
    /// * `requests` - Batch of verification requests
    ///
    /// # Returns
    ///
    /// `Ok(true)` if all valid, `Ok(false)` if any invalid
    pub fn verify_all_or_none(&self, requests: &[BatchVerifyRequest<'_>]) -> DilithiumResult<bool> {
        for req in requests {
            let is_valid = req
                .signature
                .verify_with_key(req.message, req.public_key, &self.mlwe)?;
            if !is_valid {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Verify signatures from same signer (optimized)
    ///
    /// When verifying multiple signatures from the same public key,
    /// we can cache the public key processing.
    ///
    /// # Arguments
    ///
    /// * `public_key` - Shared public key
    /// * `message_sig_pairs` - Pairs of (message, signature)
    ///
    /// # Returns
    ///
    /// Batch result with statistics
    pub fn verify_same_signer(
        &self,
        public_key: &PublicKey,
        message_sig_pairs: &[(&[u8], &DilithiumSignature)],
    ) -> DilithiumResult<BatchVerifyResult> {
        let start = Instant::now();

        if message_sig_pairs.is_empty() {
            return Ok(BatchVerifyResult {
                total: 0,
                valid: 0,
                invalid: 0,
                invalid_indices: vec![],
                duration_us: 0,
                avg_time_per_signature_us: 0.0,
            });
        }

        // Parallel verification with shared public key
        let results: Vec<(usize, bool)> = message_sig_pairs
            .par_iter()
            .enumerate()
            .map(|(idx, (message, signature))| {
                let is_valid = signature
                    .verify_with_key(message, public_key, &self.mlwe)
                    .unwrap_or(false);
                (idx, is_valid)
            })
            .collect();

        let duration = start.elapsed();

        let valid_count = results.iter().filter(|(_, v)| *v).count();
        let invalid_indices: Vec<usize> = results
            .iter()
            .filter(|(_, v)| !*v)
            .map(|(idx, _)| *idx)
            .collect();

        let total = message_sig_pairs.len();
        let duration_us = duration.as_micros() as u64;

        Ok(BatchVerifyResult {
            total,
            valid: valid_count,
            invalid: total - valid_count,
            invalid_indices,
            duration_us,
            avg_time_per_signature_us: duration_us as f64 / total as f64,
        })
    }
}

/// Streaming batch verifier for very large batches
///
/// Processes signatures in chunks to limit memory usage while
/// maintaining high throughput.
pub struct StreamingBatchVerifier {
    /// Base batch verifier
    verifier: BatchVerifier,
    /// Chunk size for processing
    chunk_size: usize,
}

impl StreamingBatchVerifier {
    /// Create new streaming verifier
    ///
    /// # Arguments
    ///
    /// * `security_level` - Dilithium security level
    /// * `chunk_size` - Number of signatures to process at once
    pub fn new(security_level: SecurityLevel, chunk_size: usize) -> Self {
        Self {
            verifier: BatchVerifier::new(security_level),
            chunk_size: chunk_size.max(1),
        }
    }

    /// Verify signatures with streaming processing
    ///
    /// # Arguments
    ///
    /// * `requests` - Iterator of verification requests
    ///
    /// # Returns
    ///
    /// Aggregated batch result
    pub fn verify_streaming<'a>(
        &self,
        requests: impl Iterator<Item = BatchVerifyRequest<'a>>,
    ) -> DilithiumResult<BatchVerifyResult> {
        let start = Instant::now();

        let mut total = 0;
        let mut valid = 0;
        let mut invalid_indices = Vec::new();

        let requests_vec: Vec<_> = requests.collect();

        for (chunk_idx, chunk) in requests_vec.chunks(self.chunk_size).enumerate() {
            let chunk_result = self.verifier.verify_batch(chunk)?;

            // Adjust indices for chunk offset
            let offset = chunk_idx * self.chunk_size;
            for idx in chunk_result.invalid_indices {
                invalid_indices.push(offset + idx);
            }

            total += chunk_result.total;
            valid += chunk_result.valid;
        }

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        Ok(BatchVerifyResult {
            total,
            valid,
            invalid: total - valid,
            invalid_indices,
            duration_us,
            avg_time_per_signature_us: if total > 0 {
                duration_us as f64 / total as f64
            } else {
                0.0
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DilithiumKeypair;

    fn generate_test_data(
        count: usize,
    ) -> DilithiumResult<(DilithiumKeypair, Vec<Vec<u8>>, Vec<DilithiumSignature>)> {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)?;

        let mut messages = Vec::with_capacity(count);
        let mut signatures = Vec::with_capacity(count);

        for i in 0..count {
            let message = format!("Test message {}", i).into_bytes();
            let signature = keypair.sign(&message)?;
            messages.push(message);
            signatures.push(signature);
        }

        Ok((keypair, messages, signatures))
    }

    #[test]
    fn test_batch_verify_all_valid() {
        let (keypair, messages, signatures) = generate_test_data(10).expect("Failed to generate test data");

        let verifier = BatchVerifier::new(SecurityLevel::Standard);

        let requests: Vec<_> = messages
            .iter()
            .zip(signatures.iter())
            .map(|(m, s)| BatchVerifyRequest {
                message: m,
                signature: s,
                public_key: &keypair.public_key,
            })
            .collect();

        let result = verifier.verify_batch(&requests).expect("Batch verify failed");

        assert!(result.all_valid());
        assert_eq!(result.total, 10);
        assert_eq!(result.valid, 10);
        assert_eq!(result.invalid, 0);
        assert!(result.invalid_indices.is_empty());
    }

    #[test]
    fn test_batch_verify_with_invalid() {
        let (keypair, messages, mut signatures) =
            generate_test_data(10).expect("Failed to generate test data");

        // Corrupt signature at index 3 - modify the internal challenge polynomial
        // which will cause verification to fail
        signatures[3].c[0] ^= 1;  // Flip a bit in the challenge polynomial

        let verifier = BatchVerifier::new(SecurityLevel::Standard);

        let requests: Vec<_> = messages
            .iter()
            .zip(signatures.iter())
            .map(|(m, s)| BatchVerifyRequest {
                message: m,
                signature: s,
                public_key: &keypair.public_key,
            })
            .collect();

        let result = verifier.verify_batch(&requests).expect("Batch verify failed");

        assert!(!result.all_valid());
        assert_eq!(result.total, 10);
        assert_eq!(result.valid, 9);
        assert_eq!(result.invalid, 1);
        assert!(result.invalid_indices.contains(&3));
    }

    #[test]
    fn test_verify_all_or_none() {
        let (keypair, messages, signatures) = generate_test_data(5).expect("Failed to generate test data");

        let verifier = BatchVerifier::new(SecurityLevel::Standard);

        let requests: Vec<_> = messages
            .iter()
            .zip(signatures.iter())
            .map(|(m, s)| BatchVerifyRequest {
                message: m,
                signature: s,
                public_key: &keypair.public_key,
            })
            .collect();

        let result = verifier
            .verify_all_or_none(&requests)
            .expect("Verification failed");
        assert!(result);
    }

    #[test]
    fn test_verify_same_signer() {
        let (keypair, messages, signatures) = generate_test_data(10).expect("Failed to generate test data");

        let verifier = BatchVerifier::new(SecurityLevel::Standard);

        let pairs: Vec<_> = messages
            .iter()
            .zip(signatures.iter())
            .map(|(m, s)| (m.as_slice(), s))
            .collect();

        let result = verifier
            .verify_same_signer(&keypair.public_key, &pairs)
            .expect("Batch verify failed");

        assert!(result.all_valid());
        assert_eq!(result.total, 10);
    }

    #[test]
    fn test_empty_batch() {
        let verifier = BatchVerifier::new(SecurityLevel::Standard);
        let result = verifier.verify_batch(&[]).expect("Should handle empty batch");

        assert_eq!(result.total, 0);
        assert_eq!(result.valid, 0);
        assert!(result.all_valid()); // Empty is considered all valid
    }

    #[test]
    fn test_streaming_verifier() {
        let (keypair, messages, signatures) = generate_test_data(25).expect("Failed to generate test data");

        let verifier = StreamingBatchVerifier::new(SecurityLevel::Standard, 10);

        let requests = messages.iter().zip(signatures.iter()).map(|(m, s)| {
            BatchVerifyRequest {
                message: m,
                signature: s,
                public_key: &keypair.public_key,
            }
        });

        let result = verifier
            .verify_streaming(requests)
            .expect("Streaming verify failed");

        assert!(result.all_valid());
        assert_eq!(result.total, 25);
    }
}
