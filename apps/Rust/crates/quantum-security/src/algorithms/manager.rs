//! Algorithm Manager
//!
//! This module provides management capabilities for post-quantum cryptographic algorithms.

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Algorithm Manager for coordinating post-quantum cryptographic operations
#[derive(Debug, Clone)]
pub struct AlgorithmManager {
    pub id: Uuid,
    pub name: String,
    pub config: AlgorithmConfig,
    pub algorithms: Arc<RwLock<HashMap<PQCAlgorithm, AlgorithmInstance>>>,
    pub metrics: Arc<RwLock<HashMap<PQCAlgorithm, AlgorithmMetrics>>>,
    pub enabled: bool,
}

/// Algorithm Instance
#[derive(Debug, Clone)]
pub struct AlgorithmInstance {
    pub algorithm: PQCAlgorithm,
    pub initialized: bool,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub error_count: u64,
    pub performance_data: AlgorithmMetrics,
}

/// Algorithm Operation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmOperationResult {
    pub success: bool,
    pub algorithm: PQCAlgorithm,
    pub operation_type: OperationType,
    pub duration_us: u64,
    pub error: Option<String>,
}

/// Operation Type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OperationType {
    KeyGeneration,
    Encapsulation,
    Decapsulation,
    Signature,
    Verification,
    KeyDerivation,
}

impl AlgorithmManager {
    /// Create a new algorithm manager
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            config: AlgorithmConfig::default(),
            algorithms: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Initialize algorithm
    pub async fn initialize_algorithm(&self, algorithm: PQCAlgorithm) -> Result<(), QuantumSecurityError> {
        let mut algorithms = self.algorithms.write().await;
        
        let instance = AlgorithmInstance {
            algorithm: algorithm.clone(),
            initialized: true,
            last_used: None,
            error_count: 0,
            performance_data: AlgorithmMetrics::default(),
        };

        algorithms.insert(algorithm.clone(), instance);
        
        // Initialize metrics
        let mut metrics = self.metrics.write().await;
        metrics.insert(algorithm, AlgorithmMetrics::default());
        
        Ok(())
    }

    /// Generate key pair
    pub async fn generate_key_pair(&self, algorithm: PQCAlgorithm, usage: KeyUsage) -> Result<PQCKeyPair, QuantumSecurityError> {
        let start_time = std::time::Instant::now();
        
        // Check if algorithm is enabled
        if !self.config.enabled_algorithms.contains(&algorithm) {
            return Err(QuantumSecurityError::AlgorithmNotEnabled(algorithm.to_string()));
        }

        // Update metrics
        self.update_operation_metrics(algorithm.clone(), OperationType::KeyGeneration, start_time.elapsed().as_micros() as u64, true).await?;

        // Generate mock key pair (in real implementation, this would use actual PQC libraries)
        let (pub_size, priv_size) = algorithm.key_sizes();
        let now = chrono::Utc::now();
        
        let (public_key, private_key) = match algorithm {
            PQCAlgorithm::Kyber512 | PQCAlgorithm::Kyber768 | PQCAlgorithm::Kyber1024 => {
                (
                    PQCKey::KyberPublicKey {
                        algorithm: algorithm.clone(),
                        key_data: vec![0u8; pub_size],
                        created_at: now,
                    },
                    PQCKey::KyberPrivateKey {
                        algorithm: algorithm.clone(),
                        key_data: SecureBytes::new(vec![0u8; priv_size]),
                        created_at: now,
                    },
                )
            }
            PQCAlgorithm::Dilithium2 | PQCAlgorithm::Dilithium3 | PQCAlgorithm::Dilithium5 => {
                (
                    PQCKey::DilithiumPublicKey {
                        algorithm: algorithm.clone(),
                        key_data: vec![0u8; pub_size],
                        created_at: now,
                    },
                    PQCKey::DilithiumPrivateKey {
                        algorithm: algorithm.clone(),
                        key_data: SecureBytes::new(vec![0u8; priv_size]),
                        created_at: now,
                    },
                )
            }
            PQCAlgorithm::Falcon512 | PQCAlgorithm::Falcon1024 => {
                (
                    PQCKey::FalconPublicKey {
                        algorithm: algorithm.clone(),
                        key_data: vec![0u8; pub_size],
                        created_at: now,
                    },
                    PQCKey::FalconPrivateKey {
                        algorithm: algorithm.clone(),
                        key_data: SecureBytes::new(vec![0u8; priv_size]),
                        created_at: now,
                    },
                )
            }
            _ => {
                (
                    PQCKey::SphincsPlusPublicKey {
                        algorithm: algorithm.clone(),
                        key_data: vec![0u8; pub_size],
                        created_at: now,
                    },
                    PQCKey::SphincsPlusPrivateKey {
                        algorithm: algorithm.clone(),
                        key_data: SecureBytes::new(vec![0u8; priv_size]),
                        created_at: now,
                    },
                )
            }
        };

        Ok(PQCKeyPair::new(public_key, private_key, algorithm, usage))
    }

    /// Encapsulate key
    pub async fn encapsulate(&self, algorithm: PQCAlgorithm, public_key: &PQCKey) -> Result<EncapsulatedKey, QuantumSecurityError> {
        let start_time = std::time::Instant::now();
        
        if !algorithm.is_kem() {
            return Err(QuantumSecurityError::InvalidOperation("Algorithm is not a KEM".to_string()));
        }

        self.update_operation_metrics(algorithm.clone(), OperationType::Encapsulation, start_time.elapsed().as_micros() as u64, true).await?;

        // Mock encapsulation
        Ok(EncapsulatedKey {
            ciphertext: vec![0u8; 32],
            shared_secret: SecureBytes::new(vec![0u8; 32]),
            algorithm,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Sign data
    pub async fn sign(&self, algorithm: PQCAlgorithm, private_key: &PQCKey, data: &[u8]) -> Result<DigitalSignature, QuantumSecurityError> {
        let start_time = std::time::Instant::now();
        
        if !algorithm.is_signature() {
            return Err(QuantumSecurityError::InvalidOperation("Algorithm is not a signature algorithm".to_string()));
        }

        self.update_operation_metrics(algorithm.clone(), OperationType::Signature, start_time.elapsed().as_micros() as u64, true).await?;

        // Mock signature
        Ok(DigitalSignature {
            signature: vec![0u8; algorithm.signature_size()],
            algorithm,
            signer_id: "mock-signer".to_string(),
            timestamp: chrono::Utc::now(),
            message_hash: data.to_vec(),
        })
    }

    /// Get algorithm metrics
    pub async fn get_metrics(&self, algorithm: PQCAlgorithm) -> Result<AlgorithmMetrics, QuantumSecurityError> {
        let metrics = self.metrics.read().await;
        metrics.get(&algorithm).cloned().ok_or_else(|| {
            QuantumSecurityError::AlgorithmNotFound(algorithm.to_string())
        })
    }

    /// Update operation metrics
    async fn update_operation_metrics(
        &self,
        algorithm: PQCAlgorithm,
        operation: OperationType,
        duration_us: u64,
        success: bool,
    ) -> Result<(), QuantumSecurityError> {
        let mut metrics = self.metrics.write().await;
        let metric = metrics.entry(algorithm).or_insert_with(AlgorithmMetrics::default);

        match operation {
            OperationType::KeyGeneration => {
                metric.key_generation_count += 1;
                metric.key_generation_time_us += duration_us;
            }
            OperationType::Encapsulation => {
                metric.encapsulation_count += 1;
                metric.encapsulation_time_us += duration_us;
            }
            OperationType::Decapsulation => {
                metric.decapsulation_count += 1;
                metric.decapsulation_time_us += duration_us;
            }
            OperationType::Signature => {
                metric.signature_count += 1;
                metric.signature_time_us += duration_us;
            }
            OperationType::Verification => {
                metric.verification_count += 1;
                metric.verification_time_us += duration_us;
            }
            OperationType::KeyDerivation => {
                // Additional operation type for key derivation
            }
        }

        if !success {
            metric.error_count += 1;
        }

        metric.last_operation = Some(chrono::Utc::now());
        Ok(())
    }

    /// Get all enabled algorithms
    pub fn get_enabled_algorithms(&self) -> Vec<PQCAlgorithm> {
        self.config.enabled_algorithms.clone()
    }

    /// Check if algorithm is enabled
    pub fn is_algorithm_enabled(&self, algorithm: &PQCAlgorithm) -> bool {
        self.config.enabled_algorithms.contains(algorithm)
    }

    /// Encrypt data using post-quantum cryptography
    pub async fn encrypt_data(
        &self,
        key_material: &SecureBytes,
        data: &[u8],
        metadata: Option<EncryptionMetadata>,
    ) -> Result<EncryptedData, QuantumSecurityError> {
        // This is a simplified implementation for compilation
        let algorithm = self.config.default_kem_algorithm.to_string();
        let nonce = vec![0u8; 12]; // Mock nonce
        let tag = vec![0u8; 16]; // Mock authentication tag
        
        Ok(EncryptedData {
            algorithm,
            ciphertext: data.to_vec(), // Mock encryption
            nonce: Some(nonce),
            tag: Some(tag),
            key_id: Some("mock-key-id".to_string()),
            metadata,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Decrypt data using post-quantum cryptography
    pub async fn decrypt_data(
        &self,
        key_material: &SecureBytes,
        encrypted_data: &EncryptedData,
    ) -> Result<Vec<u8>, QuantumSecurityError> {
        // This is a simplified implementation for compilation
        // In a real implementation, this would use the appropriate PQC algorithm
        Ok(encrypted_data.ciphertext.clone()) // Mock decryption
    }

    /// Sign data using post-quantum digital signatures
    pub async fn sign_data(
        &self,
        key_material: &SecureBytes,
        data: &[u8],
        algorithm: Option<PQCAlgorithm>,
    ) -> Result<DigitalSignature, QuantumSecurityError> {
        let alg = algorithm.unwrap_or(self.config.default_signature_algorithm.clone());
        
        // Mock signature generation
        let signature_data = vec![0u8; alg.signature_size()];
        
        Ok(DigitalSignature {
            signature: signature_data,
            algorithm: alg,
            signer_id: "mock-signer".to_string(),
            timestamp: chrono::Utc::now(),
            message_hash: data.to_vec(),
        })
    }

    /// Verify digital signature using post-quantum algorithms
    pub async fn verify_signature(
        &self,
        public_key: &PQCKey,
        data: &[u8],
        signature: &DigitalSignature,
    ) -> Result<bool, QuantumSecurityError> {
        // This is a simplified implementation for compilation
        // In a real implementation, this would verify the signature properly
        Ok(true) // Mock verification
    }
}

impl Default for AlgorithmManager {
    fn default() -> Self {
        Self::new("default-algorithm-manager".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_algorithm_manager_creation() {
        let manager = AlgorithmManager::new("test".to_string());
        assert_eq!(manager.name, "test");
        assert!(manager.enabled);
    }

    #[tokio::test]
    async fn test_initialize_algorithm() {
        let manager = AlgorithmManager::new("test".to_string());
        let result = manager.initialize_algorithm(PQCAlgorithm::Kyber1024).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_generate_key_pair() {
        let manager = AlgorithmManager::new("test".to_string());
        manager.initialize_algorithm(PQCAlgorithm::Kyber1024).await.unwrap();
        
        let result = manager.generate_key_pair(PQCAlgorithm::Kyber1024, KeyUsage::KeyEncapsulation).await;
        assert!(result.is_ok());
        
        let key_pair = result.unwrap();
        assert_eq!(key_pair.algorithm, PQCAlgorithm::Kyber1024);
        assert_eq!(key_pair.usage, KeyUsage::KeyEncapsulation);
    }

    #[tokio::test]
    async fn test_algorithm_enablement() {
        let manager = AlgorithmManager::new("test".to_string());
        assert!(manager.is_algorithm_enabled(&PQCAlgorithm::Kyber1024));
        
        let enabled = manager.get_enabled_algorithms();
        assert!(enabled.contains(&PQCAlgorithm::Kyber1024));
    }
}