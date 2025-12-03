// Model Persistence - Checkpointing and State Management
// Enterprise-grade model storage and versioning system

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::fs;
use serde::{Deserialize, Serialize};

use super::{IntegrationError, NeuralModel, ModelId};

/// Model persistence manager with versioning and checkpointing
pub struct ModelPersistence {
    storage_path: PathBuf,
    models: Arc<RwLock<HashMap<ModelId, ModelMetadata>>>,
    checkpoints: Arc<RwLock<HashMap<String, CheckpointMetadata>>>,
    compression_enabled: bool,
    encryption_enabled: bool,
}

impl ModelPersistence {
    pub fn new() -> Self {
        let storage_path = std::env::var("MODEL_STORAGE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models"));
        
        Self {
            storage_path,
            models: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            compression_enabled: true,
            encryption_enabled: false, // Enable for production
        }
    }
    
    pub fn with_config(config: PersistenceConfig) -> Self {
        Self {
            storage_path: config.storage_path,
            models: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            compression_enabled: config.compression_enabled,
            encryption_enabled: config.encryption_enabled,
        }
    }
    
    /// Save a model and return its unique ID
    pub async fn save_model(&self, name: String, model: Arc<dyn NeuralModel>) -> Result<ModelId, IntegrationError> {
        let model_id = self.generate_model_id(&name);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Create model directory
        let model_dir = self.storage_path.join(&model_id);
        fs::create_dir_all(&model_dir).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to create model directory: {}", e)))?;
        
        // Serialize model state
        let model_state = model.save_state()
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to serialize model: {}", e)))?;
        
        // Apply compression if enabled
        let processed_state = if self.compression_enabled {
            self.compress_data(&model_state)?
        } else {
            model_state
        };
        
        // Apply encryption if enabled
        let final_state = if self.encryption_enabled {
            self.encrypt_data(&processed_state)?
        } else {
            processed_state
        };
        
        // Save model state to file
        let state_path = model_dir.join("model.bin");
        fs::write(&state_path, &final_state).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write model state: {}", e)))?;
        
        // Create metadata
        let metadata = ModelMetadata {
            id: model_id.clone(),
            name: name.clone(),
            created_at: timestamp,
            updated_at: timestamp,
            version: 1,
            size_bytes: final_state.len(),
            compressed: self.compression_enabled,
            encrypted: self.encryption_enabled,
            architecture: "unknown".to_string(), // Would be determined from model
            hyperparameters: HashMap::new(),
            training_history: Vec::new(),
            tags: Vec::new(),
        };
        
        // Save metadata
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to serialize metadata: {}", e)))?;
        
        fs::write(&metadata_path, metadata_json).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write metadata: {}", e)))?;
        
        // Update in-memory registry
        let mut models = self.models.write().await;
        models.insert(model_id.clone(), metadata);
        
        Ok(model_id)
    }
    
    /// Load a model by its ID
    pub async fn load_model(&self, model_id: &str) -> Result<Arc<dyn NeuralModel>, IntegrationError> {
        let model_dir = self.storage_path.join(model_id);
        
        // Check if model exists
        if !model_dir.exists() {
            return Err(IntegrationError::ModelPersistenceFailed(format!("Model {} not found", model_id)));
        }
        
        // Load metadata
        let metadata = self.load_metadata(model_id).await?;
        
        // Load model state
        let state_path = model_dir.join("model.bin");
        let raw_state = fs::read(&state_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read model state: {}", e)))?;
        
        // Apply decryption if needed
        let decrypted_state = if metadata.encrypted {
            self.decrypt_data(&raw_state)?
        } else {
            raw_state
        };
        
        // Apply decompression if needed
        let final_state = if metadata.compressed {
            self.decompress_data(&decrypted_state)?
        } else {
            decrypted_state
        };
        
        // TODO: Create model instance based on architecture and load state
        // This is a placeholder - would need factory pattern for different architectures
        Err(IntegrationError::ModelPersistenceFailed("Model loading not fully implemented".to_string()))
    }
    
    /// Update an existing model
    pub async fn update_model(&self, model_id: ModelId, model: Arc<dyn NeuralModel>) -> Result<(), IntegrationError> {
        let mut models = self.models.write().await;
        
        if let Some(metadata) = models.get_mut(&model_id) {
            // Increment version
            metadata.version += 1;
            metadata.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Save new model state (similar to save_model but update existing)
            let model_dir = self.storage_path.join(&model_id);
            
            let model_state = model.save_state()
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to serialize model: {}", e)))?;
            
            let processed_state = if self.compression_enabled {
                self.compress_data(&model_state)?
            } else {
                model_state
            };
            
            let final_state = if self.encryption_enabled {
                self.encrypt_data(&processed_state)?
            } else {
                processed_state
            };
            
            metadata.size_bytes = final_state.len();
            
            // Save updated model state
            let state_path = model_dir.join("model.bin");
            fs::write(&state_path, &final_state).await
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write updated model state: {}", e)))?;
            
            // Save updated metadata
            let metadata_path = model_dir.join("metadata.json");
            let metadata_json = serde_json::to_string_pretty(metadata)
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to serialize metadata: {}", e)))?;
            
            fs::write(&metadata_path, metadata_json).await
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write metadata: {}", e)))?;
            
            Ok(())
        } else {
            Err(IntegrationError::ModelPersistenceFailed(format!("Model {} not found", model_id)))
        }
    }
    
    /// Create a checkpoint of a model
    pub async fn create_checkpoint(&self, model_id: &str, name: String, description: Option<String>) -> Result<String, IntegrationError> {
        let checkpoint_id = self.generate_checkpoint_id(&model_id, &name);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Load current model state
        let model_dir = self.storage_path.join(model_id);
        let state_path = model_dir.join("model.bin");
        
        if !state_path.exists() {
            return Err(IntegrationError::ModelPersistenceFailed(format!("Model {} not found", model_id)));
        }
        
        let model_state = fs::read(&state_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read model state: {}", e)))?;
        
        // Create checkpoint directory
        let checkpoint_dir = self.storage_path.join("checkpoints").join(&checkpoint_id);
        fs::create_dir_all(&checkpoint_dir).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to create checkpoint directory: {}", e)))?;
        
        // Save checkpoint state
        let checkpoint_state_path = checkpoint_dir.join("checkpoint.bin");
        fs::write(&checkpoint_state_path, &model_state).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write checkpoint state: {}", e)))?;
        
        // Create checkpoint metadata
        let checkpoint_metadata = CheckpointMetadata {
            id: checkpoint_id.clone(),
            model_id: model_id.to_string(),
            name,
            description: description.unwrap_or_else(|| format!("Checkpoint of model {}", model_id)),
            created_at: timestamp,
            size_bytes: model_state.len(),
            parent_model_version: 1, // Would get from model metadata
        };
        
        // Save checkpoint metadata
        let checkpoint_metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&checkpoint_metadata)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to serialize checkpoint metadata: {}", e)))?;
        
        fs::write(&checkpoint_metadata_path, metadata_json).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to write checkpoint metadata: {}", e)))?;
        
        // Update in-memory registry
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(checkpoint_id.clone(), checkpoint_metadata);
        
        Ok(checkpoint_id)
    }
    
    /// Load model from checkpoint
    pub async fn load_from_checkpoint(&self, checkpoint_id: &str) -> Result<Arc<dyn NeuralModel>, IntegrationError> {
        let checkpoint_dir = self.storage_path.join("checkpoints").join(checkpoint_id);
        
        if !checkpoint_dir.exists() {
            return Err(IntegrationError::ModelPersistenceFailed(format!("Checkpoint {} not found", checkpoint_id)));
        }
        
        // Load checkpoint metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read checkpoint metadata: {}", e)))?;
        
        let _checkpoint_metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to parse checkpoint metadata: {}", e)))?;
        
        // Load checkpoint state
        let state_path = checkpoint_dir.join("checkpoint.bin");
        let _checkpoint_state = fs::read(&state_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read checkpoint state: {}", e)))?;
        
        // TODO: Create model instance and load state
        Err(IntegrationError::ModelPersistenceFailed("Checkpoint loading not fully implemented".to_string()))
    }
    
    /// List all saved models
    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>, IntegrationError> {
        // Refresh from disk
        self.refresh_model_registry().await?;
        
        let models = self.models.read().await;
        Ok(models.values().cloned().collect())
    }
    
    /// List all checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointMetadata>, IntegrationError> {
        self.refresh_checkpoint_registry().await?;
        
        let checkpoints = self.checkpoints.read().await;
        Ok(checkpoints.values().cloned().collect())
    }
    
    /// List checkpoints for a specific model
    pub async fn list_model_checkpoints(&self, model_id: &str) -> Result<Vec<CheckpointMetadata>, IntegrationError> {
        self.refresh_checkpoint_registry().await?;
        
        let checkpoints = self.checkpoints.read().await;
        let model_checkpoints: Vec<CheckpointMetadata> = checkpoints
            .values()
            .filter(|checkpoint| checkpoint.model_id == model_id)
            .cloned()
            .collect();
        
        Ok(model_checkpoints)
    }
    
    /// Delete a model
    pub async fn delete_model(&self, model_id: &str) -> Result<(), IntegrationError> {
        let model_dir = self.storage_path.join(model_id);
        
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir).await
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to delete model directory: {}", e)))?;
        }
        
        // Remove from in-memory registry
        let mut models = self.models.write().await;
        models.remove(model_id);
        
        Ok(())
    }
    
    /// Delete a checkpoint
    pub async fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<(), IntegrationError> {
        let checkpoint_dir = self.storage_path.join("checkpoints").join(checkpoint_id);
        
        if checkpoint_dir.exists() {
            fs::remove_dir_all(&checkpoint_dir).await
                .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to delete checkpoint directory: {}", e)))?;
        }
        
        // Remove from in-memory registry
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.remove(checkpoint_id);
        
        Ok(())
    }
    
    /// Export model to external format
    pub async fn export_model(&self, model_id: &str, format: ExportFormat, output_path: &Path) -> Result<(), IntegrationError> {
        let model_dir = self.storage_path.join(model_id);
        
        if !model_dir.exists() {
            return Err(IntegrationError::ModelPersistenceFailed(format!("Model {} not found", model_id)));
        }
        
        match format {
            ExportFormat::ONNX => {
                // TODO: Implement ONNX export
                Err(IntegrationError::ModelPersistenceFailed("ONNX export not implemented".to_string()))
            },
            ExportFormat::TensorFlow => {
                // TODO: Implement TensorFlow export
                Err(IntegrationError::ModelPersistenceFailed("TensorFlow export not implemented".to_string()))
            },
            ExportFormat::PyTorch => {
                // TODO: Implement PyTorch export
                Err(IntegrationError::ModelPersistenceFailed("PyTorch export not implemented".to_string()))
            },
            ExportFormat::Native => {
                // Copy native format
                self.copy_model_files(&model_dir, output_path).await
            },
        }
    }
    
    /// Import model from external format
    pub async fn import_model(&self, name: String, format: ExportFormat, input_path: &Path) -> Result<ModelId, IntegrationError> {
        match format {
            ExportFormat::ONNX => {
                // TODO: Implement ONNX import
                Err(IntegrationError::ModelPersistenceFailed("ONNX import not implemented".to_string()))
            },
            ExportFormat::TensorFlow => {
                // TODO: Implement TensorFlow import
                Err(IntegrationError::ModelPersistenceFailed("TensorFlow import not implemented".to_string()))
            },
            ExportFormat::PyTorch => {
                // TODO: Implement PyTorch import
                Err(IntegrationError::ModelPersistenceFailed("PyTorch import not implemented".to_string()))
            },
            ExportFormat::Native => {
                // Copy native format
                let model_id = self.generate_model_id(&name);
                let model_dir = self.storage_path.join(&model_id);
                
                self.copy_model_files(input_path, &model_dir).await?;
                
                // Refresh registry to include imported model
                self.refresh_model_registry().await?;
                
                Ok(model_id)
            },
        }
    }
    
    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> Result<StorageStats, IntegrationError> {
        self.refresh_model_registry().await?;
        self.refresh_checkpoint_registry().await?;
        
        let models = self.models.read().await;
        let checkpoints = self.checkpoints.read().await;
        
        let total_models = models.len();
        let total_checkpoints = checkpoints.len();
        let total_model_size: usize = models.values().map(|m| m.size_bytes).sum();
        let total_checkpoint_size: usize = checkpoints.values().map(|c| c.size_bytes).sum();
        
        Ok(StorageStats {
            total_models,
            total_checkpoints,
            total_model_size_bytes: total_model_size,
            total_checkpoint_size_bytes: total_checkpoint_size,
            total_size_bytes: total_model_size + total_checkpoint_size,
            storage_path: self.storage_path.clone(),
            compression_enabled: self.compression_enabled,
            encryption_enabled: self.encryption_enabled,
        })
    }
    
    // Private helper methods
    
    fn generate_model_id(&self, name: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let sanitized_name = name.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>();
        
        format!("{}_{}", sanitized_name, timestamp)
    }
    
    fn generate_checkpoint_id(&self, model_id: &str, name: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let sanitized_name = name.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>();
        
        format!("{}_{}_checkpoint_{}", model_id, sanitized_name, timestamp)
    }
    
    async fn load_metadata(&self, model_id: &str) -> Result<ModelMetadata, IntegrationError> {
        let metadata_path = self.storage_path.join(model_id).join("metadata.json");
        
        let metadata_json = fs::read_to_string(&metadata_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read metadata: {}", e)))?;
        
        serde_json::from_str(&metadata_json)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to parse metadata: {}", e)))
    }
    
    async fn refresh_model_registry(&self) -> Result<(), IntegrationError> {
        // Ensure storage directory exists
        fs::create_dir_all(&self.storage_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to create storage directory: {}", e)))?;
        
        let mut models = self.models.write().await;
        models.clear();
        
        let mut entries = fs::read_dir(&self.storage_path).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read storage directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            
            if path.is_dir() && path.file_name().unwrap_or_default() != "checkpoints" {
                let model_id = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                
                if let Ok(metadata) = self.load_metadata(&model_id).await {
                    models.insert(model_id, metadata);
                }
            }
        }
        
        Ok(())
    }
    
    async fn refresh_checkpoint_registry(&self) -> Result<(), IntegrationError> {
        let checkpoints_dir = self.storage_path.join("checkpoints");
        
        if !checkpoints_dir.exists() {
            return Ok(()); // No checkpoints directory yet
        }
        
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.clear();
        
        let mut entries = fs::read_dir(&checkpoints_dir).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read checkpoints directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read checkpoint entry: {}", e)))? {
            
            let path = entry.path();
            
            if path.is_dir() {
                let checkpoint_id = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                
                let metadata_path = path.join("metadata.json");
                
                if let Ok(metadata_json) = fs::read_to_string(&metadata_path).await {
                    if let Ok(metadata) = serde_json::from_str::<CheckpointMetadata>(&metadata_json) {
                        checkpoints.insert(checkpoint_id, metadata);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, IntegrationError> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Compression failed: {}", e)))?;
        
        encoder.finish()
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Compression finalization failed: {}", e)))
    }
    
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, IntegrationError> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Decompression failed: {}", e)))?;
        
        Ok(decompressed)
    }
    
    fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, IntegrationError> {
        // TODO: Implement actual encryption
        // For now, just return the data unchanged
        Ok(data.to_vec())
    }
    
    fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, IntegrationError> {
        // TODO: Implement actual decryption
        // For now, just return the data unchanged
        Ok(data.to_vec())
    }
    
    async fn copy_model_files(&self, source: &Path, destination: &Path) -> Result<(), IntegrationError> {
        fs::create_dir_all(destination).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to create destination directory: {}", e)))?;
        
        let mut entries = fs::read_dir(source).await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read source directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to read source entry: {}", e)))? {
            
            let source_path = entry.path();
            let file_name = source_path.file_name().unwrap();
            let dest_path = destination.join(file_name);
            
            if source_path.is_file() {
                fs::copy(&source_path, &dest_path).await
                    .map_err(|e| IntegrationError::ModelPersistenceFailed(format!("Failed to copy file: {}", e)))?;
            }
        }
        
        Ok(())
    }
}

// Supporting types and structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub version: u32,
    pub size_bytes: usize,
    pub compressed: bool,
    pub encrypted: bool,
    pub architecture: String,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub training_history: Vec<TrainingEpochInfo>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub id: String,
    pub model_id: String,
    pub name: String,
    pub description: String,
    pub created_at: u64,
    pub size_bytes: usize,
    pub parent_model_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEpochInfo {
    pub epoch: u32,
    pub loss: f32,
    pub accuracy: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    pub storage_path: PathBuf,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_models: usize,
    pub total_checkpoints: usize,
    pub total_model_size_bytes: usize,
    pub total_checkpoint_size_bytes: usize,
    pub total_size_bytes: usize,
    pub storage_path: PathBuf,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    ONNX,
    TensorFlow,
    PyTorch,
    Native,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_model_persistence_creation() {
        // Use temp directory to avoid picking up leftover models from previous runs
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            storage_path: temp_dir.path().to_path_buf(),
            compression_enabled: true,
            encryption_enabled: false,
        };
        let persistence = ModelPersistence::with_config(config);
        let stats = persistence.get_storage_stats().await.unwrap();
        assert_eq!(stats.total_models, 0);
    }
    
    #[tokio::test]
    async fn test_model_id_generation() {
        let persistence = ModelPersistence::new();
        let id1 = persistence.generate_model_id("test_model");

        // Verify format: name_timestamp
        assert!(id1.starts_with("test_model_"));
        // Verify timestamp component is numeric
        let timestamp_part = id1.strip_prefix("test_model_").unwrap();
        assert!(timestamp_part.chars().all(|c| c.is_ascii_digit()));

        // Test different names produce different IDs
        let id2 = persistence.generate_model_id("other_model");
        assert!(id2.starts_with("other_model_"));
        assert_ne!(id1, id2);
    }
    
    #[tokio::test]
    async fn test_checkpoint_id_generation() {
        let persistence = ModelPersistence::new();
        let checkpoint_id = persistence.generate_checkpoint_id("model_123", "epoch_100");
        
        assert!(checkpoint_id.contains("model_123"));
        assert!(checkpoint_id.contains("epoch_100"));
        assert!(checkpoint_id.contains("checkpoint"));
    }
    
    #[tokio::test]
    async fn test_data_compression() {
        let persistence = ModelPersistence::new();
        let original_data = b"This is test data for compression. ".repeat(100);
        
        let compressed = persistence.compress_data(&original_data).unwrap();
        let decompressed = persistence.decompress_data(&compressed).unwrap();
        
        assert_eq!(original_data, decompressed);
        assert!(compressed.len() < original_data.len()); // Should be compressed
    }
    
    #[tokio::test]
    async fn test_storage_stats() {
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            storage_path: temp_dir.path().to_path_buf(),
            compression_enabled: true,
            encryption_enabled: false,
        };
        
        let persistence = ModelPersistence::with_config(config);
        let stats = persistence.get_storage_stats().await.unwrap();
        
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert!(stats.compression_enabled);
        assert!(!stats.encryption_enabled);
    }
}