//! ML Model Management and Registry
//!
//! This module provides comprehensive model management capabilities including
//! model registry, versioning, deployment, and lifecycle management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::ml::{MLError, MLResult, MLModel, MLFramework, MLTask, ModelMetadata, PerformanceMetrics};

/// Model registry for managing trained models
pub struct ModelRegistry {
    /// Registered models
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,
    /// Base path for model storage
    base_path: PathBuf,
    /// Registry metadata
    metadata: RegistryMetadata,
}

/// Model entry in the registry
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model binary data
    pub data: Vec<u8>,
    /// Model checksum
    pub checksum: String,
    /// Storage location
    pub storage_path: Option<PathBuf>,
    /// Model status
    pub status: ModelStatus,
    /// Deployment information
    pub deployment: Option<DeploymentInfo>,
}

/// Model status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is training
    Training,
    /// Model is trained and ready
    Ready,
    /// Model is deployed
    Deployed,
    /// Model is deprecated
    Deprecated,
    /// Model has failed
    Failed,
    /// Model is archived
    Archived,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStatus::Training => write!(f, "Training"),
            ModelStatus::Ready => write!(f, "Ready"),
            ModelStatus::Deployed => write!(f, "Deployed"),
            ModelStatus::Deprecated => write!(f, "Deprecated"),
            ModelStatus::Failed => write!(f, "Failed"),
            ModelStatus::Archived => write!(f, "Archived"),
        }
    }
}

/// Deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    /// Deployment ID
    pub deployment_id: String,
    /// Environment (dev, staging, prod)
    pub environment: String,
    /// Deployment timestamp
    pub deployed_at: DateTime<Utc>,
    /// Deployment configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Health status
    pub health_status: HealthStatus,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
}

/// Health status for deployed models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Model is healthy
    Healthy,
    /// Model has warnings
    Warning,
    /// Model is degraded
    Degraded,
    /// Model is unhealthy
    Unhealthy,
    /// Health is unknown
    Unknown,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry version
    pub version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Total models
    pub total_models: usize,
    /// Registry statistics
    pub statistics: RegistryStatistics,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Models by framework
    pub by_framework: HashMap<String, usize>,
    /// Models by task
    pub by_task: HashMap<String, usize>,
    /// Models by status
    pub by_status: HashMap<String, usize>,
    /// Total storage size in bytes
    pub total_storage_size: u64,
}

impl Default for RegistryStatistics {
    fn default() -> Self {
        Self {
            by_framework: HashMap::new(),
            by_task: HashMap::new(),
            by_status: HashMap::new(),
            total_storage_size: 0,
        }
    }
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new<P: Into<PathBuf>>(base_path: P) -> MLResult<Self> {
        let base_path = base_path.into();
        
        // Create directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)
                .map_err(|e| MLError::IoError(e))?;
        }
        
        let metadata = RegistryMetadata {
            version: "1.0.0".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            total_models: 0,
            statistics: RegistryStatistics::default(),
        };
        
        Ok(Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            base_path,
            metadata,
        })
    }
    
    /// Register a new model
    pub fn register_model<M>(&mut self, model: &M) -> MLResult<String>
    where
        M: MLModel,
    {
        let model_id = model.metadata().id.clone();
        let model_data = model.to_bytes()?;
        let checksum = self.compute_checksum(&model_data);
        
        let entry = ModelEntry {
            metadata: model.metadata().clone(),
            data: model_data,
            checksum,
            storage_path: None,
            status: ModelStatus::Ready,
            deployment: None,
        };
        
        self.models.write().insert(model_id.clone(), entry);
        self.update_statistics();
        
        Ok(model_id)
    }
    
    /// Get model by ID
    pub fn get_model(&self, model_id: &str) -> Option<ModelEntry> {
        self.models.read().get(model_id).cloned()
    }
    
    /// List all models
    pub fn list_models(&self) -> Vec<String> {
        self.models.read().keys().cloned().collect()
    }
    
    /// List models by framework
    pub fn list_models_by_framework(&self, framework: MLFramework) -> Vec<String> {
        self.models.read()
            .iter()
            .filter(|(_, entry)| entry.metadata.framework == framework.to_string())
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// List models by task
    pub fn list_models_by_task(&self, task: MLTask) -> Vec<String> {
        self.models.read()
            .iter()
            .filter(|(_, entry)| entry.metadata.task == task.to_string())
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// List models by status
    pub fn list_models_by_status(&self, status: ModelStatus) -> Vec<String> {
        self.models.read()
            .iter()
            .filter(|(_, entry)| entry.status == status)
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Update model status
    pub fn update_model_status(&mut self, model_id: &str, status: ModelStatus) -> MLResult<()> {
        let mut models = self.models.write();
        if let Some(entry) = models.get_mut(model_id) {
            entry.status = status;
            entry.metadata.touch();
            self.update_statistics();
            Ok(())
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Deploy model
    pub fn deploy_model(
        &mut self,
        model_id: &str,
        environment: String,
        config: HashMap<String, serde_json::Value>,
    ) -> MLResult<String> {
        let deployment_id = format!("deploy-{}-{}", model_id, uuid::Uuid::new_v4());
        
        let deployment_info = DeploymentInfo {
            deployment_id: deployment_id.clone(),
            environment,
            deployed_at: Utc::now(),
            config,
            health_status: HealthStatus::Unknown,
            metrics: None,
        };
        
        let mut models = self.models.write();
        if let Some(entry) = models.get_mut(model_id) {
            entry.deployment = Some(deployment_info);
            entry.status = ModelStatus::Deployed;
            entry.metadata.touch();
            self.update_statistics();
            Ok(deployment_id)
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Undeploy model
    pub fn undeploy_model(&mut self, model_id: &str) -> MLResult<()> {
        let mut models = self.models.write();
        if let Some(entry) = models.get_mut(model_id) {
            entry.deployment = None;
            entry.status = ModelStatus::Ready;
            entry.metadata.touch();
            self.update_statistics();
            Ok(())
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Update deployment health
    pub fn update_deployment_health(
        &mut self,
        model_id: &str,
        health_status: HealthStatus,
        metrics: Option<PerformanceMetrics>,
    ) -> MLResult<()> {
        let mut models = self.models.write();
        if let Some(entry) = models.get_mut(model_id) {
            if let Some(ref mut deployment) = entry.deployment {
                deployment.health_status = health_status;
                deployment.metrics = metrics;
                entry.metadata.touch();
                Ok(())
            } else {
                Err(MLError::ConfigurationError {
                    message: "Model is not deployed".to_string(),
                })
            }
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Archive model
    pub fn archive_model(&mut self, model_id: &str) -> MLResult<()> {
        self.update_model_status(model_id, ModelStatus::Archived)
    }
    
    /// Delete model
    pub fn delete_model(&mut self, model_id: &str) -> MLResult<()> {
        let mut models = self.models.write();
        if models.remove(model_id).is_some() {
            self.update_statistics();
            Ok(())
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Save model to disk
    pub fn save_model_to_disk(&mut self, model_id: &str) -> MLResult<PathBuf> {
        let models = self.models.read();
        if let Some(entry) = models.get(model_id) {
            let filename = format!("{}.bin", model_id);
            let file_path = self.base_path.join(filename);
            
            std::fs::write(&file_path, &entry.data)
                .map_err(|e| MLError::IoError(e))?;
            
            drop(models);
            
            // Update storage path
            let mut models = self.models.write();
            if let Some(entry) = models.get_mut(model_id) {
                entry.storage_path = Some(file_path.clone());
            }
            
            Ok(file_path)
        } else {
            Err(MLError::ModelNotFound {
                model_id: model_id.to_string(),
            })
        }
    }
    
    /// Load model from disk
    pub fn load_model_from_disk<P: AsRef<Path>>(&mut self, model_id: String, file_path: P) -> MLResult<()> {
        let data = std::fs::read(file_path.as_ref())
            .map_err(|e| MLError::IoError(e))?;
        
        let checksum = self.compute_checksum(&data);
        
        // Try to deserialize metadata from the model data
        // This is a simplified approach - in practice, you'd have a more robust format
        let metadata = ModelMetadata::new(
            model_id.clone(),
            "Loaded Model".to_string(),
            MLFramework::Hybrid, // Default
            MLTask::Classification, // Default
        );
        
        let entry = ModelEntry {
            metadata,
            data,
            checksum,
            storage_path: Some(file_path.as_ref().to_path_buf()),
            status: ModelStatus::Ready,
            deployment: None,
        };
        
        self.models.write().insert(model_id, entry);
        self.update_statistics();
        
        Ok(())
    }
    
    /// Export registry metadata
    pub fn export_metadata(&self) -> MLResult<String> {
        serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| MLError::SerdeError(e))
    }
    
    /// Get registry statistics
    pub fn get_statistics(&self) -> RegistryStatistics {
        self.metadata.statistics.clone()
    }
    
    /// Search models by criteria
    pub fn search_models(&self, criteria: SearchCriteria) -> Vec<String> {
        let models = self.models.read();
        
        models.iter()
            .filter(|(_, entry)| {
                let mut matches = true;
                
                if let Some(ref framework) = criteria.framework {
                    matches = matches && entry.metadata.framework == framework.to_string();
                }
                
                if let Some(ref task) = criteria.task {
                    matches = matches && entry.metadata.task == task.to_string();
                }
                
                if let Some(ref status) = criteria.status {
                    matches = matches && entry.status == *status;
                }
                
                if let Some(ref name_pattern) = criteria.name_pattern {
                    matches = matches && entry.metadata.name.contains(name_pattern);
                }
                
                if let Some(ref tags) = criteria.tags {
                    matches = matches && tags.iter().any(|tag| entry.metadata.tags.contains(tag));
                }
                
                if let Some(min_score) = criteria.min_performance_score {
                    if let Some(primary_metric) = entry.metadata.metrics.values().next() {
                        matches = matches && *primary_metric >= min_score;
                    } else {
                        matches = false;
                    }
                }
                
                matches
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Compare models
    pub fn compare_models(&self, model_ids: &[String]) -> MLResult<ModelComparison> {
        let models = self.models.read();
        let mut comparison = ModelComparison {
            models: Vec::new(),
            comparison_matrix: HashMap::new(),
        };
        
        for model_id in model_ids {
            if let Some(entry) = models.get(model_id) {
                comparison.models.push(entry.metadata.clone());
            } else {
                return Err(MLError::ModelNotFound {
                    model_id: model_id.clone(),
                });
            }
        }
        
        // Compute comparison metrics
        for i in 0..comparison.models.len() {
            for j in (i + 1)..comparison.models.len() {
                let model1 = &comparison.models[i];
                let model2 = &comparison.models[j];
                
                let mut metrics = HashMap::new();
                
                // Compare performance metrics
                for (metric_name, value1) in &model1.metrics {
                    if let Some(value2) = model2.metrics.get(metric_name) {
                        let diff = (value1 - value2).abs();
                        metrics.insert(format!("{}_diff", metric_name), diff);
                    }
                }
                
                // Compare parameter counts (if available)
                // This would require access to the actual models
                
                let key = format!("{}_{}", model1.id, model2.id);
                comparison.comparison_matrix.insert(key, metrics);
            }
        }
        
        Ok(comparison)
    }
    
    /// Update statistics
    fn update_statistics(&mut self) {
        let models = self.models.read();
        let mut stats = RegistryStatistics::default();
        
        for entry in models.values() {
            // Count by framework
            *stats.by_framework.entry(entry.metadata.framework.clone()).or_insert(0) += 1;
            
            // Count by task
            *stats.by_task.entry(entry.metadata.task.clone()).or_insert(0) += 1;
            
            // Count by status
            *stats.by_status.entry(entry.status.to_string()).or_insert(0) += 1;
            
            // Sum storage size
            stats.total_storage_size += entry.data.len() as u64;
        }
        
        self.metadata.statistics = stats;
        self.metadata.total_models = models.len();
        self.metadata.updated_at = Utc::now();
    }
    
    /// Compute checksum for model data
    fn compute_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}

/// Search criteria for finding models
#[derive(Debug, Clone, Default)]
pub struct SearchCriteria {
    /// Filter by framework
    pub framework: Option<MLFramework>,
    /// Filter by task
    pub task: Option<MLTask>,
    /// Filter by status
    pub status: Option<ModelStatus>,
    /// Filter by name pattern
    pub name_pattern: Option<String>,
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    /// Minimum performance score
    pub min_performance_score: Option<f64>,
    /// Date range
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

impl SearchCriteria {
    /// Create new search criteria
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Filter by framework
    pub fn framework(mut self, framework: MLFramework) -> Self {
        self.framework = Some(framework);
        self
    }
    
    /// Filter by task
    pub fn task(mut self, task: MLTask) -> Self {
        self.task = Some(task);
        self
    }
    
    /// Filter by status
    pub fn status(mut self, status: ModelStatus) -> Self {
        self.status = Some(status);
        self
    }
    
    /// Filter by name pattern
    pub fn name_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.name_pattern = Some(pattern.into());
        self
    }
    
    /// Filter by tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }
    
    /// Filter by minimum performance score
    pub fn min_performance_score(mut self, score: f64) -> Self {
        self.min_performance_score = Some(score);
        self
    }
}

/// Model comparison result
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Models being compared
    pub models: Vec<ModelMetadata>,
    /// Comparison metrics matrix
    pub comparison_matrix: HashMap<String, HashMap<String, f64>>,
}

impl ModelComparison {
    /// Get best model by metric
    pub fn best_model_by_metric(&self, metric_name: &str) -> Option<&ModelMetadata> {
        self.models.iter()
            .max_by(|a, b| {
                let a_score = a.metrics.get(metric_name).unwrap_or(&0.0);
                let b_score = b.metrics.get(metric_name).unwrap_or(&0.0);
                a_score.partial_cmp(b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Generate comparison report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Model Comparison Report\n");
        report.push_str("======================\n\n");
        
        // Model summary
        for model in &self.models {
            report.push_str(&format!("Model: {} ({})\n", model.name, model.id));
            report.push_str(&format!("Framework: {}\n", model.framework));
            report.push_str(&format!("Task: {}\n", model.task));
            report.push_str(&format!("Created: {}\n", model.created_at));
            
            if !model.metrics.is_empty() {
                report.push_str("Metrics:\n");
                for (metric, value) in &model.metrics {
                    report.push_str(&format!("  {}: {:.4}\n", metric, value));
                }
            }
            report.push_str("\n");
        }
        
        // Comparison matrix
        if !self.comparison_matrix.is_empty() {
            report.push_str("Comparison Matrix\n");
            report.push_str("-----------------\n");
            for (pair, metrics) in &self.comparison_matrix {
                report.push_str(&format!("Pair: {}\n", pair));
                for (metric, value) in metrics {
                    report.push_str(&format!("  {}: {:.4}\n", metric, value));
                }
                report.push_str("\n");
            }
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::ml::neural::{NeuralNetwork, NeuralConfig};
    
    #[test]
    fn test_model_registry_creation() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path());
        
        assert!(registry.is_ok());
        let registry = registry.unwrap();
        assert_eq!(registry.list_models().len(), 0);
    }
    
    #[test]
    fn test_model_registration() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Create a test model
        let config = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let model = NeuralNetwork::new(config).unwrap();
        
        // Register the model
        let model_id = registry.register_model(&model).unwrap();
        
        assert!(!model_id.is_empty());
        assert_eq!(registry.list_models().len(), 1);
        assert!(registry.get_model(&model_id).is_some());
    }
    
    #[test]
    fn test_model_status_updates() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let config = NeuralConfig::new().with_layers(vec![3, 2, 1]);
        let model = NeuralNetwork::new(config).unwrap();
        let model_id = registry.register_model(&model).unwrap();
        
        // Update status
        registry.update_model_status(&model_id, ModelStatus::Deployed).unwrap();
        
        let entry = registry.get_model(&model_id).unwrap();
        assert_eq!(entry.status, ModelStatus::Deployed);
        
        // Test non-existent model
        let result = registry.update_model_status("non-existent", ModelStatus::Failed);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_model_deployment() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let config = NeuralConfig::new().with_layers(vec![4, 3, 1]);
        let model = NeuralNetwork::new(config).unwrap();
        let model_id = registry.register_model(&model).unwrap();
        
        let mut deploy_config = HashMap::new();
        deploy_config.insert("replicas".to_string(), serde_json::Value::Number(serde_json::Number::from(3)));
        
        let deployment_id = registry.deploy_model(
            &model_id,
            "production".to_string(),
            deploy_config,
        ).unwrap();
        
        assert!(!deployment_id.is_empty());
        
        let entry = registry.get_model(&model_id).unwrap();
        assert_eq!(entry.status, ModelStatus::Deployed);
        assert!(entry.deployment.is_some());
        
        let deployment = entry.deployment.unwrap();
        assert_eq!(deployment.environment, "production");
        assert_eq!(deployment.health_status, HealthStatus::Unknown);
        
        // Test undeploy
        registry.undeploy_model(&model_id).unwrap();
        let entry = registry.get_model(&model_id).unwrap();
        assert_eq!(entry.status, ModelStatus::Ready);
        assert!(entry.deployment.is_none());
    }
    
    #[test]
    fn test_model_search() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Register multiple models
        let config1 = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let mut model1 = NeuralNetwork::new(config1).unwrap();
        model1.metadata_mut().name = "Regression Model".to_string();
        model1.metadata_mut().add_tag("production".to_string());
        let id1 = registry.register_model(&model1).unwrap();
        
        let config2 = NeuralConfig::new().with_layers(vec![10, 5, 2]);
        let mut model2 = NeuralNetwork::new(config2).unwrap();
        model2.metadata_mut().name = "Classification Model".to_string();
        model2.metadata_mut().add_tag("experimental".to_string());
        let id2 = registry.register_model(&model2).unwrap();
        
        registry.update_model_status(&id2, ModelStatus::Deployed).unwrap();
        
        // Search by framework
        let results = registry.list_models_by_framework(MLFramework::Candle);
        assert_eq!(results.len(), 2);
        
        // Search by status
        let deployed_models = registry.list_models_by_status(ModelStatus::Deployed);
        assert_eq!(deployed_models.len(), 1);
        assert_eq!(deployed_models[0], id2);
        
        // Search by criteria
        let criteria = SearchCriteria::new()
            .name_pattern("Regression")
            .status(ModelStatus::Ready);
        let search_results = registry.search_models(criteria);
        assert_eq!(search_results.len(), 1);
        assert_eq!(search_results[0], id1);
    }
    
    #[test]
    fn test_model_comparison() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Register models with metrics
        let config1 = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let mut model1 = NeuralNetwork::new(config1).unwrap();
        model1.metadata_mut().add_metric("accuracy".to_string(), 0.85);
        model1.metadata_mut().add_metric("f1_score".to_string(), 0.82);
        let id1 = registry.register_model(&model1).unwrap();
        
        let config2 = NeuralConfig::new().with_layers(vec![7, 4, 1]);
        let mut model2 = NeuralNetwork::new(config2).unwrap();
        model2.metadata_mut().add_metric("accuracy".to_string(), 0.90);
        model2.metadata_mut().add_metric("f1_score".to_string(), 0.88);
        let id2 = registry.register_model(&model2).unwrap();
        
        let comparison = registry.compare_models(&[id1, id2]).unwrap();
        
        assert_eq!(comparison.models.len(), 2);
        assert!(!comparison.comparison_matrix.is_empty());
        
        let best_accuracy = comparison.best_model_by_metric("accuracy").unwrap();
        assert_eq!(best_accuracy.id, id2);
        
        let report = comparison.generate_report();
        assert!(report.contains("Model Comparison Report"));
        assert!(report.contains("accuracy"));
    }
    
    #[test]
    fn test_model_serialization() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let config = NeuralConfig::new().with_layers(vec![3, 2, 1]);
        let model = NeuralNetwork::new(config).unwrap();
        let model_id = registry.register_model(&model).unwrap();
        
        // Save to disk
        let file_path = registry.save_model_to_disk(&model_id).unwrap();
        assert!(file_path.exists());
        
        // Verify entry has storage path
        let entry = registry.get_model(&model_id).unwrap();
        assert!(entry.storage_path.is_some());
        assert_eq!(entry.storage_path.unwrap(), file_path);
    }
    
    #[test]
    fn test_registry_statistics() {
        let temp_dir = tempdir().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        // Register models with different frameworks and tasks
        let config1 = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let model1 = NeuralNetwork::new(config1).unwrap();
        registry.register_model(&model1).unwrap();
        
        let config2 = NeuralConfig::new().with_layers(vec![7, 4, 2]);
        let model2 = NeuralNetwork::new(config2).unwrap();
        registry.register_model(&model2).unwrap();
        
        let stats = registry.get_statistics();
        assert_eq!(stats.by_framework.get("Candle"), Some(&2));
        assert_eq!(stats.by_status.get("Ready"), Some(&2));
        assert!(stats.total_storage_size > 0);
    }
}