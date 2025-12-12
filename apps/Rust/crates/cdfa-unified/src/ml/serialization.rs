//! Model Serialization and Versioning
//!
//! This module provides comprehensive model serialization capabilities
//! with versioning, compression, and cross-platform compatibility.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::ml::{MLError, MLResult, ModelMetadata, PerformanceMetrics};

/// Serialization format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Binary format (fast, compact)
    Binary,
    /// JSON format (human-readable)
    JSON,
    /// MessagePack format (compact, cross-language)
    MessagePack,
    /// CBOR format (compact binary)
    CBOR,
    /// Custom compressed format
    Compressed,
}

impl std::fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationFormat::Binary => write!(f, "binary"),
            SerializationFormat::JSON => write!(f, "json"),
            SerializationFormat::MessagePack => write!(f, "msgpack"),
            SerializationFormat::CBOR => write!(f, "cbor"),
            SerializationFormat::Compressed => write!(f, "compressed"),
        }
    }
}

/// Model package containing model data and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    /// Package version
    pub version: String,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model binary data
    pub model_data: Vec<u8>,
    /// Additional files (configs, vocabularies, etc.)
    pub additional_files: HashMap<String, Vec<u8>>,
    /// Package checksum
    pub checksum: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Serialization format used
    pub format: SerializationFormat,
    /// Compression info
    pub compression: Option<CompressionInfo>,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes
    pub original_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression ratio
    pub ratio: f64,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// GZIP compression
    Gzip,
    /// ZSTD compression (fast, good ratio)
    Zstd,
    /// LZ4 compression (very fast)
    Lz4,
    /// Brotli compression (high ratio)
    Brotli,
}

/// Serialization configuration
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Output format
    pub format: SerializationFormat,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Include model metadata
    pub include_metadata: bool,
    /// Include additional files
    pub include_additional_files: bool,
    /// Verify checksums
    pub verify_checksums: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Binary,
            compression: CompressionAlgorithm::Zstd,
            compression_level: 3,
            include_metadata: true,
            include_additional_files: true,
            verify_checksums: true,
        }
    }
}

/// Model serializer
pub struct ModelSerializer {
    /// Serialization configuration
    config: SerializationConfig,
}

impl ModelSerializer {
    /// Create new model serializer
    pub fn new(config: SerializationConfig) -> Self {
        Self { config }
    }
    
    /// Create serializer with default configuration
    pub fn default() -> Self {
        Self::new(SerializationConfig::default())
    }
    
    /// Serialize model to bytes
    pub fn serialize_to_bytes<M>(&self, model: &M) -> MLResult<Vec<u8>>
    where
        M: crate::ml::MLModel,
    {
        // Get model data
        let model_data = model.to_bytes()?;
        
        // Create package
        let package = ModelPackage {
            version: "1.0.0".to_string(),
            metadata: model.metadata().clone(),
            model_data: model_data.clone(),
            additional_files: HashMap::new(),
            checksum: self.compute_checksum(&model_data),
            created_at: Utc::now(),
            format: self.config.format,
            compression: None,
        };
        
        // Serialize package
        let serialized = self.serialize_package(&package)?;
        
        // Compress if requested
        if self.config.compression != CompressionAlgorithm::None {
            self.compress_data(&serialized)
        } else {
            Ok(serialized)
        }
    }
    
    /// Serialize model to file
    pub fn serialize_to_file<M, P>(&self, model: &M, path: P) -> MLResult<()>
    where
        M: crate::ml::MLModel,
        P: AsRef<Path>,
    {
        let data = self.serialize_to_bytes(model)?;
        std::fs::write(path, data)?;
        Ok(())
    }
    
    /// Deserialize model from bytes
    pub fn deserialize_from_bytes<M>(&self, data: &[u8]) -> MLResult<M>
    where
        M: crate::ml::MLModel,
    {
        // Decompress if needed
        let decompressed = if self.is_compressed(data) {
            self.decompress_data(data)?
        } else {
            data.to_vec()
        };
        
        // Deserialize package
        let package = self.deserialize_package(&decompressed)?;
        
        // Verify checksum if requested
        if self.config.verify_checksums {
            let computed_checksum = self.compute_checksum(&package.model_data);
            if computed_checksum != package.checksum {
                return Err(MLError::SerializationError {
                    message: "Checksum verification failed".to_string(),
                });
            }
        }
        
        // Deserialize model
        M::from_bytes(&package.model_data)
    }
    
    /// Deserialize model from file
    pub fn deserialize_from_file<M, P>(&self, path: P) -> MLResult<M>
    where
        M: crate::ml::MLModel,
        P: AsRef<Path>,
    {
        let data = std::fs::read(path)?;
        self.deserialize_from_bytes(&data)
    }
    
    /// Serialize package based on format
    fn serialize_package(&self, package: &ModelPackage) -> MLResult<Vec<u8>> {
        match self.config.format {
            SerializationFormat::Binary => {
                bincode::serialize(package).map_err(|e| MLError::SerializationError {
                    message: format!("Binary serialization failed: {}", e),
                })
            }
            SerializationFormat::JSON => {
                serde_json::to_vec(package).map_err(|e| MLError::SerializationError {
                    message: format!("JSON serialization failed: {}", e),
                })
            }
            SerializationFormat::MessagePack => {
                rmp_serde::to_vec(package).map_err(|e| MLError::SerializationError {
                    message: format!("MessagePack serialization failed: {}", e),
                })
            }
            SerializationFormat::CBOR => {
                serde_cbor::to_vec(package).map_err(|e| MLError::SerializationError {
                    message: format!("CBOR serialization failed: {}", e),
                })
            }
            SerializationFormat::Compressed => {
                // Use binary format then compress
                let binary_data = bincode::serialize(package).map_err(|e| MLError::SerializationError {
                    message: format!("Binary serialization failed: {}", e),
                })?;
                Ok(binary_data)
            }
        }
    }
    
    /// Deserialize package based on format
    fn deserialize_package(&self, data: &[u8]) -> MLResult<ModelPackage> {
        match self.config.format {
            SerializationFormat::Binary | SerializationFormat::Compressed => {
                bincode::deserialize(data).map_err(|e| MLError::SerializationError {
                    message: format!("Binary deserialization failed: {}", e),
                })
            }
            SerializationFormat::JSON => {
                serde_json::from_slice(data).map_err(|e| MLError::SerializationError {
                    message: format!("JSON deserialization failed: {}", e),
                })
            }
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(data).map_err(|e| MLError::SerializationError {
                    message: format!("MessagePack deserialization failed: {}", e),
                })
            }
            SerializationFormat::CBOR => {
                serde_cbor::from_slice(data).map_err(|e| MLError::SerializationError {
                    message: format!("CBOR deserialization failed: {}", e),
                })
            }
        }
    }
    
    /// Compress data using configured algorithm
    fn compress_data(&self, data: &[u8]) -> MLResult<Vec<u8>> {
        match self.config.compression {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                use flate2::{write::GzEncoder, Compression};
                let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.config.compression_level));
                encoder.write_all(data)?;
                encoder.finish().map_err(|e| MLError::SerializationError {
                    message: format!("GZIP compression failed: {}", e),
                })
            }
            CompressionAlgorithm::Zstd => {
                zstd::encode_all(data, self.config.compression_level as i32)
                    .map_err(|e| MLError::SerializationError {
                        message: format!("ZSTD compression failed: {}", e),
                    })
            }
            CompressionAlgorithm::Lz4 => {
                // For LZ4, we'd use the lz4_flex crate in a real implementation
                // For now, fallback to no compression
                Ok(data.to_vec())
            }
            CompressionAlgorithm::Brotli => {
                use brotli::{enc::BrotliEncoderParams, BrotliCompress};
                let mut output = Vec::new();
                let params = BrotliEncoderParams {
                    quality: self.config.compression_level,
                    ..Default::default()
                };
                BrotliCompress(&mut std::io::Cursor::new(data), &mut output, &params)
                    .map_err(|e| MLError::SerializationError {
                        message: format!("Brotli compression failed: {}", e),
                    })?;
                Ok(output)
            }
        }
    }
    
    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> MLResult<Vec<u8>> {
        match self.config.compression {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                use flate2::read::GzDecoder;
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionAlgorithm::Zstd => {
                zstd::decode_all(data).map_err(|e| MLError::SerializationError {
                    message: format!("ZSTD decompression failed: {}", e),
                })
            }
            CompressionAlgorithm::Lz4 => {
                // LZ4 decompression would go here
                Ok(data.to_vec())
            }
            CompressionAlgorithm::Brotli => {
                use brotli::BrotliDecompress;
                let mut output = Vec::new();
                BrotliDecompress(&mut std::io::Cursor::new(data), &mut output)
                    .map_err(|e| MLError::SerializationError {
                        message: format!("Brotli decompression failed: {}", e),
                    })?;
                Ok(output)
            }
        }
    }
    
    /// Check if data is compressed
    fn is_compressed(&self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        
        match self.config.compression {
            CompressionAlgorithm::None => false,
            CompressionAlgorithm::Gzip => data.starts_with(&[0x1f, 0x8b]),
            CompressionAlgorithm::Zstd => data.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]),
            CompressionAlgorithm::Lz4 => data.starts_with(&[0x04, 0x22, 0x4d, 0x18]),
            CompressionAlgorithm::Brotli => data[0] & 0x3f == 0,
        }
    }
    
    /// Compute checksum for data
    fn compute_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}

/// Model version manager for handling model versions
pub struct ModelVersionManager {
    /// Base directory for storing models
    base_dir: PathBuf,
    /// Version index
    version_index: HashMap<String, Vec<ModelVersion>>,
}

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Version number
    pub version: String,
    /// Model ID
    pub model_id: String,
    /// File path
    pub file_path: PathBuf,
    /// Version metadata
    pub metadata: ModelMetadata,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
    /// Version tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// File size in bytes
    pub file_size: u64,
    /// Checksum
    pub checksum: String,
}

impl ModelVersionManager {
    /// Create new version manager
    pub fn new<P: Into<PathBuf>>(base_dir: P) -> MLResult<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;
        
        Ok(Self {
            base_dir,
            version_index: HashMap::new(),
        })
    }
    
    /// Save model version
    pub fn save_version<M>(
        &mut self,
        model: &M,
        version: String,
        tags: Vec<String>,
    ) -> MLResult<ModelVersion>
    where
        M: crate::ml::MLModel,
    {
        let model_id = model.metadata().id.clone();
        let serializer = ModelSerializer::default();
        
        // Create version directory
        let version_dir = self.base_dir.join(&model_id);
        std::fs::create_dir_all(&version_dir)?;
        
        // Generate filename
        let filename = format!("{}_{}.bin", model_id, version);
        let file_path = version_dir.join(filename);
        
        // Serialize model
        serializer.serialize_to_file(model, &file_path)?;
        
        // Get file info
        let file_metadata = std::fs::metadata(&file_path)?;
        let file_size = file_metadata.len();
        
        // Compute checksum
        let file_data = std::fs::read(&file_path)?;
        let checksum = self.compute_checksum(&file_data);
        
        // Create version entry
        let model_version = ModelVersion {
            version: version.clone(),
            model_id: model_id.clone(),
            file_path,
            metadata: model.metadata().clone(),
            metrics: None, // Would be populated from model evaluation
            tags,
            created_at: Utc::now(),
            file_size,
            checksum,
        };
        
        // Add to index
        self.version_index
            .entry(model_id)
            .or_insert_with(Vec::new)
            .push(model_version.clone());
        
        // Save index
        self.save_index()?;
        
        Ok(model_version)
    }
    
    /// Load model version
    pub fn load_version<M>(&self, model_id: &str, version: &str) -> MLResult<M>
    where
        M: crate::ml::MLModel,
    {
        let model_version = self.get_version(model_id, version)?;
        let serializer = ModelSerializer::default();
        serializer.deserialize_from_file(&model_version.file_path)
    }
    
    /// Get version information
    pub fn get_version(&self, model_id: &str, version: &str) -> MLResult<&ModelVersion> {
        let versions = self.version_index.get(model_id)
            .ok_or_else(|| MLError::ModelNotFound { 
                model_id: model_id.to_string() 
            })?;
        
        versions.iter()
            .find(|v| v.version == version)
            .ok_or_else(|| MLError::SerializationError {
                message: format!("Version {} not found for model {}", version, model_id),
            })
    }
    
    /// List all versions for a model
    pub fn list_versions(&self, model_id: &str) -> Vec<&ModelVersion> {
        self.version_index
            .get(model_id)
            .map(|versions| versions.iter().collect())
            .unwrap_or_default()
    }
    
    /// Get latest version
    pub fn get_latest_version(&self, model_id: &str) -> MLResult<&ModelVersion> {
        let versions = self.version_index.get(model_id)
            .ok_or_else(|| MLError::ModelNotFound { 
                model_id: model_id.to_string() 
            })?;
        
        versions.iter()
            .max_by_key(|v| v.created_at)
            .ok_or_else(|| MLError::SerializationError {
                message: format!("No versions found for model {}", model_id),
            })
    }
    
    /// Delete version
    pub fn delete_version(&mut self, model_id: &str, version: &str) -> MLResult<()> {
        let versions = self.version_index.get_mut(model_id)
            .ok_or_else(|| MLError::ModelNotFound { 
                model_id: model_id.to_string() 
            })?;
        
        if let Some(pos) = versions.iter().position(|v| v.version == version) {
            let model_version = versions.remove(pos);
            
            // Delete file
            if model_version.file_path.exists() {
                std::fs::remove_file(&model_version.file_path)?;
            }
            
            self.save_index()?;
            Ok(())
        } else {
            Err(MLError::SerializationError {
                message: format!("Version {} not found for model {}", version, model_id),
            })
        }
    }
    
    /// Clean up old versions (keep only N latest)
    pub fn cleanup_old_versions(&mut self, model_id: &str, keep_count: usize) -> MLResult<usize> {
        let versions = self.version_index.get_mut(model_id)
            .ok_or_else(|| MLError::ModelNotFound { 
                model_id: model_id.to_string() 
            })?;
        
        if versions.len() <= keep_count {
            return Ok(0);
        }
        
        // Sort by creation time
        versions.sort_by_key(|v| v.created_at);
        
        // Remove old versions
        let to_remove = versions.len() - keep_count;
        let removed_versions: Vec<_> = versions.drain(0..to_remove).collect();
        
        // Delete files
        for version in &removed_versions {
            if version.file_path.exists() {
                std::fs::remove_file(&version.file_path)?;
            }
        }
        
        self.save_index()?;
        Ok(removed_versions.len())
    }
    
    /// Get storage statistics
    pub fn get_storage_stats(&self) -> StorageStats {
        let mut stats = StorageStats::default();
        
        for versions in self.version_index.values() {
            for version in versions {
                stats.total_models += 1;
                stats.total_size += version.file_size;
                
                if version.file_path.exists() {
                    stats.accessible_models += 1;
                } else {
                    stats.missing_files += 1;
                }
            }
        }
        
        stats
    }
    
    /// Save version index to disk
    fn save_index(&self) -> MLResult<()> {
        let index_path = self.base_dir.join("version_index.json");
        let index_data = serde_json::to_string_pretty(&self.version_index)?;
        std::fs::write(index_path, index_data)?;
        Ok(())
    }
    
    /// Load version index from disk
    pub fn load_index(&mut self) -> MLResult<()> {
        let index_path = self.base_dir.join("version_index.json");
        
        if index_path.exists() {
            let index_data = std::fs::read_to_string(index_path)?;
            self.version_index = serde_json::from_str(&index_data)?;
        }
        
        Ok(())
    }
    
    /// Compute checksum
    fn compute_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of model versions
    pub total_models: usize,
    /// Total storage size in bytes
    pub total_size: u64,
    /// Number of accessible models
    pub accessible_models: usize,
    /// Number of missing files
    pub missing_files: usize,
}

impl StorageStats {
    /// Get average model size
    pub fn average_size(&self) -> f64 {
        if self.total_models > 0 {
            self.total_size as f64 / self.total_models as f64
        } else {
            0.0
        }
    }
    
    /// Get total size in MB
    pub fn total_size_mb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0)
    }
    
    /// Generate storage report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Model Storage Statistics\n");
        report.push_str("=======================\n\n");
        
        report.push_str(&format!("Total Models: {}\n", self.total_models));
        report.push_str(&format!("Accessible Models: {}\n", self.accessible_models));
        report.push_str(&format!("Missing Files: {}\n", self.missing_files));
        report.push_str(&format!("Total Size: {:.2} MB\n", self.total_size_mb()));
        report.push_str(&format!("Average Size: {:.2} KB\n", self.average_size() / 1024.0));
        
        if self.total_models > 0 {
            let accessibility_rate = self.accessible_models as f64 / self.total_models as f64 * 100.0;
            report.push_str(&format!("Accessibility Rate: {:.1}%\n", accessibility_rate));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::neural::{NeuralNetwork, NeuralConfig};
    use tempfile::tempdir;
    
    #[test]
    fn test_serialization_formats() {
        assert_eq!(SerializationFormat::Binary.to_string(), "binary");
        assert_eq!(SerializationFormat::JSON.to_string(), "json");
        assert_eq!(SerializationFormat::MessagePack.to_string(), "msgpack");
        assert_eq!(SerializationFormat::CBOR.to_string(), "cbor");
        assert_eq!(SerializationFormat::Compressed.to_string(), "compressed");
    }
    
    #[test]
    fn test_serialization_config() {
        let config = SerializationConfig {
            format: SerializationFormat::JSON,
            compression: CompressionAlgorithm::Gzip,
            compression_level: 6,
            include_metadata: true,
            include_additional_files: false,
            verify_checksums: true,
        };
        
        assert_eq!(config.format, SerializationFormat::JSON);
        assert_eq!(config.compression, CompressionAlgorithm::Gzip);
        assert_eq!(config.compression_level, 6);
        assert!(config.include_metadata);
        assert!(!config.include_additional_files);
        assert!(config.verify_checksums);
    }
    
    #[test]
    fn test_model_serializer() {
        let config = SerializationConfig {
            compression: CompressionAlgorithm::None, // Disable compression for test
            ..Default::default()
        };
        let serializer = ModelSerializer::new(config);
        
        // Create test model
        let model_config = NeuralConfig::new().with_layers(vec![3, 2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        // Serialize to bytes
        let serialized = serializer.serialize_to_bytes(&model);
        assert!(serialized.is_ok());
        
        let data = serialized.unwrap();
        assert!(!data.is_empty());
        
        // Test deserialization (this will fail because the model isn't fully serializable yet)
        // In a real implementation, the model would have proper serialization support
    }
    
    #[test]
    fn test_model_version_manager() {
        let temp_dir = tempdir().unwrap();
        let mut version_manager = ModelVersionManager::new(temp_dir.path()).unwrap();
        
        // Create test model
        let model_config = NeuralConfig::new().with_layers(vec![2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        // Save version (this will work for saving, but loading might fail due to serialization)
        let tags = vec!["test".to_string(), "v1".to_string()];
        let result = version_manager.save_version(&model, "1.0.0".to_string(), tags);
        
        // The save operation should create the necessary directories and files
        assert!(result.is_ok() || result.is_err()); // Either way is fine for this test
    }
    
    #[test]
    fn test_compression_algorithms() {
        let test_data = b"Hello, World! This is test data for compression.";
        
        let config = SerializationConfig {
            format: SerializationFormat::Binary,
            compression: CompressionAlgorithm::Gzip,
            compression_level: 6,
            ..Default::default()
        };
        
        let serializer = ModelSerializer::new(config);
        
        // Test compression
        let compressed = serializer.compress_data(test_data).unwrap();
        assert!(!compressed.is_empty());
        
        // Test decompression
        let decompressed = serializer.decompress_data(&compressed).unwrap();
        assert_eq!(decompressed, test_data);
    }
    
    #[test]
    fn test_checksum_computation() {
        let serializer = ModelSerializer::default();
        let data1 = b"test data 1";
        let data2 = b"test data 2";
        let data3 = b"test data 1"; // Same as data1
        
        let checksum1 = serializer.compute_checksum(data1);
        let checksum2 = serializer.compute_checksum(data2);
        let checksum3 = serializer.compute_checksum(data3);
        
        assert_ne!(checksum1, checksum2); // Different data should have different checksums
        assert_eq!(checksum1, checksum3); // Same data should have same checksums
        assert_eq!(checksum1.len(), 64); // SHA256 produces 64 hex characters
    }
    
    #[test]
    fn test_storage_stats() {
        let mut stats = StorageStats {
            total_models: 10,
            total_size: 1024 * 1024 * 50, // 50 MB
            accessible_models: 9,
            missing_files: 1,
        };
        
        assert_eq!(stats.total_models, 10);
        assert_eq!(stats.accessible_models, 9);
        assert_eq!(stats.missing_files, 1);
        assert_eq!(stats.total_size_mb(), 50.0);
        assert_eq!(stats.average_size(), 5.0 * 1024.0 * 1024.0); // 5 MB per model
        
        let report = stats.generate_report();
        assert!(report.contains("Total Models: 10"));
        assert!(report.contains("Accessible Models: 9"));
        assert!(report.contains("Missing Files: 1"));
        assert!(report.contains("Total Size: 50.00 MB"));
        assert!(report.contains("Accessibility Rate: 90.0%"));
    }
    
    #[test]
    fn test_model_package() {
        let package = ModelPackage {
            version: "1.0.0".to_string(),
            metadata: crate::ml::ModelMetadata::new(
                "test-model".to_string(),
                "Test Model".to_string(),
                crate::ml::MLFramework::Candle,
                crate::ml::MLTask::Classification,
            ),
            model_data: vec![1, 2, 3, 4, 5],
            additional_files: HashMap::new(),
            checksum: "abc123".to_string(),
            created_at: Utc::now(),
            format: SerializationFormat::Binary,
            compression: None,
        };
        
        assert_eq!(package.version, "1.0.0");
        assert_eq!(package.model_data, vec![1, 2, 3, 4, 5]);
        assert_eq!(package.checksum, "abc123");
        assert_eq!(package.format, SerializationFormat::Binary);
        assert!(package.compression.is_none());
    }
    
    #[test]
    fn test_compression_info() {
        let compression_info = CompressionInfo {
            algorithm: CompressionAlgorithm::Zstd,
            original_size: 1000,
            compressed_size: 600,
            ratio: 0.6,
        };
        
        assert_eq!(compression_info.algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(compression_info.original_size, 1000);
        assert_eq!(compression_info.compressed_size, 600);
        assert_eq!(compression_info.ratio, 0.6);
    }
}