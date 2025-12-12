//! Memory-mapped model storage

use crate::config::StorageConfig;
use crate::models::{Model, ModelParameters};
use crate::{Result, NeuralForecastError};
use std::path::Path;
#[cfg(feature = "memory-efficient")]
use memmap2::MmapOptions;
use std::fs::File;

/// Memory-mapped model storage system
#[derive(Debug)]
pub struct ModelStorage {
    config: StorageConfig,
}

impl ModelStorage {
    /// Create a new model storage instance with the given configuration
    pub fn new(config: StorageConfig) -> Self {
        Self { config }
    }
    
    /// Save a model to the specified path
    pub async fn save_model<M>(&self, model: &M, path: &Path) -> Result<()>
    where
        M: Model,
    {
        // Serialize model parameters
        let parameters = model.parameters();
        let serialized = bincode::serialize(parameters)
            .map_err(|e| NeuralForecastError::StorageError(e.to_string()))?;
        
        // Write to file
        std::fs::write(path, serialized)?;
        
        Ok(())
    }
    
    /// Load model parameters from the specified path
    pub async fn load_model_parameters(&self, path: &Path) -> Result<ModelParameters> {
        #[cfg(feature = "memory-efficient")]
        {
            if self.config.memory_mapping {
                self.load_with_mmap(path).await
            } else {
                self.load_traditional(path).await
            }
        }
        #[cfg(not(feature = "memory-efficient"))]
        {
            self.load_traditional(path).await
        }
    }
    
    #[cfg(feature = "memory-efficient")]
    async fn load_with_mmap(&self, path: &Path) -> Result<ModelParameters> {
        let file = File::open(path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| NeuralForecastError::MemoryMappingError(e.to_string()))?
        };
        
        let parameters: ModelParameters = bincode::deserialize(&mmap)
            .map_err(|e| NeuralForecastError::StorageError(e.to_string()))?;
        
        Ok(parameters)
    }
    
    async fn load_traditional(&self, path: &Path) -> Result<ModelParameters> {
        let data = std::fs::read(path)?;
        let parameters: ModelParameters = bincode::deserialize(&data)
            .map_err(|e| NeuralForecastError::StorageError(e.to_string()))?;
        
        Ok(parameters)
    }
}

// Re-export is already handled in the use statement above