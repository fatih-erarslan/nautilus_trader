//! HSM Manager
//!
//! This module provides centralized management of Hardware Security Modules.

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// HSM Manager for coordinating multiple HSM providers
#[derive(Debug, Clone)]
pub struct HSMManager {
    pub id: Uuid,
    pub name: String,
    pub providers: Arc<RwLock<HashMap<HSMProvider, HSMInstance>>>,
    pub sessions: Arc<RwLock<HashMap<Uuid, HSMSession>>>,
    pub key_handles: Arc<RwLock<HashMap<String, HSMKeyHandle>>>,
    pub metrics: Arc<RwLock<HSMPerformanceMetrics>>,
    pub configuration: HSMConfiguration,
    pub enabled: bool,
}

/// HSM Instance
#[derive(Debug, Clone)]
pub struct HSMInstance {
    pub provider: HSMProvider,
    pub status: HSMStatus,
    pub configuration: HSMConfiguration,
    pub connection_pool: ConnectionPool,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub operation_count: u64,
    pub error_count: u64,
}

/// Connection Pool for HSM connections
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    pub active_connections: u32,
    pub available_connections: u32,
    pub max_connections: u32,
    pub total_created: u64,
    pub total_destroyed: u64,
}

impl HSMManager {
    /// Create new HSM manager
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            providers: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            key_handles: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HSMPerformanceMetrics::default())),
            configuration: HSMConfiguration::default(),
            enabled: true,
        }
    }

    /// Initialize HSM provider
    pub async fn initialize_provider(
        &self,
        provider: HSMProvider,
        config: HSMConfiguration,
    ) -> Result<(), QuantumSecurityError> {
        let mut providers = self.providers.write().await;

        // Create HSM instance
        let instance = HSMInstance {
            provider: provider.clone(),
            status: HSMStatus {
                provider: provider.clone(),
                available: true,
                authenticated: false,
                firmware_version: "1.0.0".to_string(),
                hardware_version: "1.0.0".to_string(),
                serial_number: "SN123456789".to_string(),
                total_memory: 1024 * 1024, // 1MB
                free_memory: 512 * 1024,   // 512KB
                key_count: 0,
                max_sessions: 100,
                active_sessions: 0,
                temperature: Some(35.0),
                last_check: chrono::Utc::now(),
            },
            configuration: config,
            connection_pool: ConnectionPool {
                active_connections: 0,
                available_connections: 0,
                max_connections: 10,
                total_created: 0,
                total_destroyed: 0,
            },
            last_heartbeat: chrono::Utc::now(),
            operation_count: 0,
            error_count: 0,
        };

        providers.insert(provider, instance);
        Ok(())
    }

    /// Open HSM session
    pub async fn open_session(
        &self,
        provider: HSMProvider,
        read_write: bool,
    ) -> Result<HSMSession, QuantumSecurityError> {
        let providers = self.providers.read().await;
        let instance = providers.get(&provider)
            .ok_or_else(|| QuantumSecurityError::HSMNotAvailable("HSM provider not available".to_string()))?;

        if !instance.status.available {
            return Err(QuantumSecurityError::HSMNotAvailable("HSM not available".to_string()));
        }

        // Check session limits
        if instance.status.active_sessions >= instance.status.max_sessions {
            return Err(QuantumSecurityError::HSMSessionLimitReached("Session limit reached".to_string()));
        }

        let session = HSMSession {
            session_id: Uuid::new_v4(),
            provider,
            slot_id: instance.configuration.connection_config.slot_id,
            authenticated: false,
            read_write,
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            operation_count: 0,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session.session_id, session.clone());

        Ok(session)
    }

    /// Close HSM session
    pub async fn close_session(&self, session_id: Uuid) -> Result<(), QuantumSecurityError> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(&session_id)
            .ok_or_else(|| QuantumSecurityError::SessionNotFound("Session not found".to_string()))?;
        Ok(())
    }

    /// Authenticate session
    pub async fn authenticate_session(
        &self,
        session_id: Uuid,
        user_pin: SecureBytes,
    ) -> Result<(), QuantumSecurityError> {
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| QuantumSecurityError::SessionNotFound("Session not found".to_string()))?;

        // Mock authentication - in real implementation, this would interact with HSM
        if user_pin.as_bytes().is_empty() {
            return Err(QuantumSecurityError::AuthenticationFailed("Invalid PIN".to_string()));
        }

        session.authenticated = true;
        session.last_activity = chrono::Utc::now();

        Ok(())
    }

    /// Execute HSM operation
    pub async fn execute_operation(
        &self,
        session_id: Uuid,
        operation: HSMOperation,
    ) -> Result<HSMOperationResult, QuantumSecurityError> {
        let start_time = std::time::Instant::now();

        // Verify session
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| QuantumSecurityError::SessionNotFound("Session not found".to_string()))?;

        if !session.authenticated {
            return Ok(HSMOperationResult::failure(
                "operation".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Session not authenticated".to_string(),
            ));
        }

        // Update session activity
        session.last_activity = chrono::Utc::now();
        session.operation_count += 1;

        // Execute operation based on type
        let result = match operation {
            HSMOperation::GenerateKey { key_type, key_size, attributes } => {
                self.generate_key(session.provider, key_type, key_size, attributes).await
            }
            HSMOperation::ImportKey { key_data, key_type, attributes } => {
                self.import_key(session.provider, key_data, key_type, attributes).await
            }
            HSMOperation::ExportKey { key_handle, format } => {
                self.export_key(key_handle, format).await
            }
            HSMOperation::DeleteKey { key_handle } => {
                self.delete_key(key_handle).await
            }
            HSMOperation::Encrypt { key_handle, plaintext, algorithm, parameters } => {
                self.encrypt(key_handle, plaintext, algorithm, parameters).await
            }
            HSMOperation::Decrypt { key_handle, ciphertext, algorithm, parameters } => {
                self.decrypt(key_handle, ciphertext, algorithm, parameters).await
            }
            HSMOperation::Sign { key_handle, data, algorithm, parameters } => {
                self.sign(key_handle, data, algorithm, parameters).await
            }
            HSMOperation::Verify { key_handle, data, signature, algorithm, parameters } => {
                self.verify(key_handle, data, signature, algorithm, parameters).await
            }
            HSMOperation::GenerateRandom { length } => {
                self.generate_random(length).await
            }
            HSMOperation::DeriveKey { base_key_handle, derivation_data, algorithm, parameters } => {
                self.derive_key(base_key_handle, derivation_data, algorithm, parameters).await
            }
        };

        let execution_time = start_time.elapsed().as_micros() as u64;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_operations += 1;
        if result.success {
            metrics.successful_operations += 1;
        } else {
            metrics.failed_operations += 1;
        }

        // Update average latency
        let total_time = metrics.average_latency_us * (metrics.total_operations - 1) as f64 + execution_time as f64;
        metrics.average_latency_us = total_time / metrics.total_operations as f64;

        // Update min/max latency
        if execution_time > metrics.max_latency_us {
            metrics.max_latency_us = execution_time;
        }
        if metrics.min_latency_us == 0 || execution_time < metrics.min_latency_us {
            metrics.min_latency_us = execution_time;
        }

        Ok(result)
    }

    /// Generate key
    async fn generate_key(
        &self,
        provider: HSMProvider,
        key_type: HSMKeyType,
        key_size: u32,
        attributes: HashMap<String, String>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Create key handle
        let key_handle = HSMKeyHandle::new(
            key_type,
            provider,
            format!("key_{}", Uuid::new_v4()),
        );

        // Store key handle
        let mut key_handles = self.key_handles.write().await;
        key_handles.insert(key_handle.handle_id.clone(), key_handle.clone());

        HSMOperationResult::success("generate_key".to_string(), start_time.elapsed().as_micros() as u64)
            .with_key_handle(key_handle)
            .with_metadata("key_size".to_string(), key_size.to_string())
    }

    /// Import key
    async fn import_key(
        &self,
        provider: HSMProvider,
        key_data: SecureBytes,
        key_type: HSMKeyType,
        attributes: HashMap<String, String>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        if key_data.as_bytes().is_empty() {
            return HSMOperationResult::failure(
                "import_key".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Empty key data".to_string(),
            );
        }

        // Create key handle
        let key_handle = HSMKeyHandle::new(
            key_type,
            provider,
            format!("imported_key_{}", Uuid::new_v4()),
        );

        // Store key handle
        let mut key_handles = self.key_handles.write().await;
        key_handles.insert(key_handle.handle_id.clone(), key_handle.clone());

        HSMOperationResult::success("import_key".to_string(), start_time.elapsed().as_micros() as u64)
            .with_key_handle(key_handle)
    }

    /// Export key
    async fn export_key(
        &self,
        key_handle: HSMKeyHandle,
        format: KeyExportFormat,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&key_handle.handle_id) {
            return HSMOperationResult::failure(
                "export_key".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        // Mock key export (would be actual key data in real implementation)
        let exported_data = vec![0u8; 32]; // Mock exported key

        HSMOperationResult::success("export_key".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(exported_data)
            .with_metadata("format".to_string(), format!("{:?}", format))
    }

    /// Delete key
    async fn delete_key(&self, key_handle: HSMKeyHandle) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        let mut key_handles = self.key_handles.write().await;
        if key_handles.remove(&key_handle.handle_id).is_none() {
            return HSMOperationResult::failure(
                "delete_key".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        HSMOperationResult::success("delete_key".to_string(), start_time.elapsed().as_micros() as u64)
    }

    /// Encrypt data
    async fn encrypt(
        &self,
        key_handle: HSMKeyHandle,
        plaintext: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&key_handle.handle_id) {
            return HSMOperationResult::failure(
                "encrypt".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        // Mock encryption
        let ciphertext = plaintext.iter().map(|b| b ^ 0xFF).collect::<Vec<u8>>();

        HSMOperationResult::success("encrypt".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(ciphertext)
            .with_metadata("algorithm".to_string(), algorithm)
    }

    /// Decrypt data
    async fn decrypt(
        &self,
        key_handle: HSMKeyHandle,
        ciphertext: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&key_handle.handle_id) {
            return HSMOperationResult::failure(
                "decrypt".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        // Mock decryption (reverse of mock encryption)
        let plaintext = ciphertext.iter().map(|b| b ^ 0xFF).collect::<Vec<u8>>();

        HSMOperationResult::success("decrypt".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(plaintext)
            .with_metadata("algorithm".to_string(), algorithm)
    }

    /// Sign data
    async fn sign(
        &self,
        key_handle: HSMKeyHandle,
        data: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&key_handle.handle_id) {
            return HSMOperationResult::failure(
                "sign".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        // Mock signature generation
        let signature = vec![0u8; 64]; // Mock 64-byte signature

        HSMOperationResult::success("sign".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(signature)
            .with_metadata("algorithm".to_string(), algorithm)
    }

    /// Verify signature
    async fn verify(
        &self,
        key_handle: HSMKeyHandle,
        data: Vec<u8>,
        signature: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&key_handle.handle_id) {
            return HSMOperationResult::failure(
                "verify".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Key not found".to_string(),
            );
        }

        // Mock verification (always succeeds for non-empty signature)
        let verification_result = !signature.is_empty();
        let result_data = vec![if verification_result { 1u8 } else { 0u8 }];

        HSMOperationResult::success("verify".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(result_data)
            .with_metadata("algorithm".to_string(), algorithm)
            .with_metadata("verified".to_string(), verification_result.to_string())
    }

    /// Generate random bytes
    async fn generate_random(&self, length: usize) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        if length == 0 || length > 1024 * 1024 {
            return HSMOperationResult::failure(
                "generate_random".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Invalid length".to_string(),
            );
        }

        // Mock random generation
        let random_data = (0..length).map(|_| rand::random::<u8>()).collect::<Vec<u8>>();

        HSMOperationResult::success("generate_random".to_string(), start_time.elapsed().as_micros() as u64)
            .with_data(random_data)
            .with_metadata("length".to_string(), length.to_string())
    }

    /// Derive key
    async fn derive_key(
        &self,
        base_key_handle: HSMKeyHandle,
        derivation_data: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    ) -> HSMOperationResult {
        let start_time = std::time::Instant::now();

        // Check if base key exists
        let key_handles = self.key_handles.read().await;
        if !key_handles.contains_key(&base_key_handle.handle_id) {
            return HSMOperationResult::failure(
                "derive_key".to_string(),
                start_time.elapsed().as_micros() as u64,
                "Base key not found".to_string(),
            );
        }

        // Create derived key handle
        let derived_key_handle = HSMKeyHandle::new(
            base_key_handle.key_type.clone(),
            base_key_handle.provider.clone(),
            format!("derived_key_{}", Uuid::new_v4()),
        );

        drop(key_handles);
        let mut key_handles = self.key_handles.write().await;
        key_handles.insert(derived_key_handle.handle_id.clone(), derived_key_handle.clone());

        HSMOperationResult::success("derive_key".to_string(), start_time.elapsed().as_micros() as u64)
            .with_key_handle(derived_key_handle)
            .with_metadata("algorithm".to_string(), algorithm)
    }

    /// Get HSM status
    pub async fn get_status(&self, provider: HSMProvider) -> Result<HSMStatus, QuantumSecurityError> {
        let providers = self.providers.read().await;
        let instance = providers.get(&provider)
            .ok_or_else(|| QuantumSecurityError::HSMNotAvailable("HSM provider not available".to_string()))?;

        Ok(instance.status.clone())
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> Result<HSMPerformanceMetrics, QuantumSecurityError> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    /// List all key handles
    pub async fn list_keys(&self) -> Result<Vec<HSMKeyHandle>, QuantumSecurityError> {
        let key_handles = self.key_handles.read().await;
        Ok(key_handles.values().cloned().collect())
    }

    /// Health check for all providers
    pub async fn health_check(&self) -> Result<HashMap<HSMProvider, bool>, QuantumSecurityError> {
        let providers = self.providers.read().await;
        let mut health_status = HashMap::new();

        for (provider, instance) in providers.iter() {
            health_status.insert(provider.clone(), instance.status.available);
        }

        Ok(health_status)
    }
}

impl Default for HSMManager {
    fn default() -> Self {
        Self::new("default-hsm-manager".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hsm_manager_creation() {
        let manager = HSMManager::new("test".to_string());
        assert_eq!(manager.name, "test");
        assert!(manager.enabled);
    }

    #[tokio::test]
    async fn test_provider_initialization() {
        let manager = HSMManager::new("test".to_string());
        let config = HSMConfiguration::default();
        
        let result = manager.initialize_provider(HSMProvider::SoftwareEmulation, config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_management() {
        let manager = HSMManager::new("test".to_string());
        let config = HSMConfiguration::default();
        
        manager.initialize_provider(HSMProvider::SoftwareEmulation, config).await.unwrap();
        
        let session = manager.open_session(HSMProvider::SoftwareEmulation, true).await;
        assert!(session.is_ok());
        
        let session = session.unwrap();
        assert_eq!(session.provider, HSMProvider::SoftwareEmulation);
        assert!(session.read_write);
        assert!(!session.authenticated);
        
        let close_result = manager.close_session(session.session_id).await;
        assert!(close_result.is_ok());
    }

    #[tokio::test]
    async fn test_key_generation() {
        let manager = HSMManager::new("test".to_string());
        let config = HSMConfiguration::default();
        
        manager.initialize_provider(HSMProvider::SoftwareEmulation, config).await.unwrap();
        
        let session = manager.open_session(HSMProvider::SoftwareEmulation, true).await.unwrap();
        
        let pin = SecureBytes::new(b"1234".to_vec());
        manager.authenticate_session(session.session_id, pin).await.unwrap();
        
        let operation = HSMOperation::GenerateKey {
            key_type: HSMKeyType::AES,
            key_size: 256,
            attributes: HashMap::new(),
        };
        
        let result = manager.execute_operation(session.session_id, operation).await;
        assert!(result.is_ok());
        
        let operation_result = result.unwrap();
        assert!(operation_result.success);
        assert!(operation_result.key_handle.is_some());
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let manager = HSMManager::new("test".to_string());
        let config = HSMConfiguration::default();
        
        manager.initialize_provider(HSMProvider::SoftwareEmulation, config).await.unwrap();
        
        let session = manager.open_session(HSMProvider::SoftwareEmulation, true).await.unwrap();
        let pin = SecureBytes::new(b"1234".to_vec());
        manager.authenticate_session(session.session_id, pin).await.unwrap();
        
        // Perform operation
        let operation = HSMOperation::GenerateRandom { length: 32 };
        manager.execute_operation(session.session_id, operation).await.unwrap();
        
        let metrics = manager.get_metrics().await.unwrap();
        assert_eq!(metrics.total_operations, 1);
        assert_eq!(metrics.successful_operations, 1);
        assert!(metrics.average_latency_us > 0.0);
    }
}