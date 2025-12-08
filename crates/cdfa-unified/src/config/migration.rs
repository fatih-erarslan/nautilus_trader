//! Configuration migration utilities
//!
//! This module provides utilities for migrating configuration files between versions,
//! ensuring backward compatibility and smooth upgrades.

use crate::config::CdfaConfig;
use crate::error::{CdfaError, Result};
// HashMap available through std if needed

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration version information
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfigVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ConfigVersion {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
    
    /// Current configuration version
    pub fn current() -> Self {
        Self::new(1, 0, 0)
    }
    
    /// Parse version from string
    pub fn parse(version_str: &str) -> Result<Self> {
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() != 3 {
            return Err(CdfaError::config_error("Version must be in format major.minor.patch"));
        }
        
        let major = parts[0].parse::<u32>()
            .map_err(|_| CdfaError::config_error("Invalid major version"))?;
        let minor = parts[1].parse::<u32>()
            .map_err(|_| CdfaError::config_error("Invalid minor version"))?;
        let patch = parts[2].parse::<u32>()
            .map_err(|_| CdfaError::config_error("Invalid patch version"))?;
            
        Ok(Self::new(major, minor, patch))
    }
    
    /// Convert to string
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
    
    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &ConfigVersion) -> bool {
        // Major version must match
        if self.major != other.major {
            return false;
        }
        
        // This version should be greater than or equal to other
        if self.minor > other.minor {
            return true;
        }
        
        if self.minor == other.minor && self.patch >= other.patch {
            return true;
        }
        
        false
    }
    
    /// Check if migration is needed
    pub fn needs_migration(&self, target: &ConfigVersion) -> bool {
        !self.is_compatible_with(target)
    }
}

impl std::fmt::Display for ConfigVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for ConfigVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConfigVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.major.cmp(&other.major) {
            std::cmp::Ordering::Equal => {
                match self.minor.cmp(&other.minor) {
                    std::cmp::Ordering::Equal => self.patch.cmp(&other.patch),
                    other => other,
                }
            },
            other => other,
        }
    }
}

/// Migration operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MigrationOperation {
    /// Add a new field with default value
    AddField {
        path: String,
        default_value: serde_json::Value,
    },
    /// Remove a field
    RemoveField {
        path: String,
    },
    /// Rename a field
    RenameField {
        old_path: String,
        new_path: String,
    },
    /// Transform a field value
    TransformField {
        path: String,
        transformation: String, // Description of transformation
    },
    /// Move field to new location
    MoveField {
        old_path: String,
        new_path: String,
    },
    /// Split field into multiple fields
    SplitField {
        source_path: String,
        target_paths: Vec<String>,
    },
    /// Merge multiple fields into one
    MergeFields {
        source_paths: Vec<String>,
        target_path: String,
    },
}

/// Migration step from one version to another
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MigrationStep {
    pub from_version: ConfigVersion,
    pub to_version: ConfigVersion,
    pub description: String,
    pub operations: Vec<MigrationOperation>,
    pub breaking_changes: bool,
}

impl MigrationStep {
    /// Create a new migration step
    pub fn new(
        from: ConfigVersion,
        to: ConfigVersion,
        description: String,
        operations: Vec<MigrationOperation>,
    ) -> Self {
        let breaking_changes = from.major != to.major;
        Self {
            from_version: from,
            to_version: to,
            description,
            operations,
            breaking_changes,
        }
    }
}

/// Configuration migration manager
pub struct ConfigMigrator {
    migrations: Vec<MigrationStep>,
}

impl ConfigMigrator {
    /// Create a new migrator
    pub fn new() -> Self {
        let mut migrator = Self {
            migrations: Vec::new(),
        };
        
        // Register built-in migrations
        migrator.register_builtin_migrations();
        migrator
    }
    
    /// Register built-in migration steps
    fn register_builtin_migrations(&mut self) {
        // Migration from 0.9.x to 1.0.0
        self.add_migration(MigrationStep::new(
            ConfigVersion::new(0, 9, 0),
            ConfigVersion::new(1, 0, 0),
            "Major restructuring of configuration hierarchy".to_string(),
            vec![
                MigrationOperation::AddField {
                    path: "processing.enable_parallel_batches".to_string(),
                    default_value: serde_json::Value::Bool(true),
                },
                MigrationOperation::AddField {
                    path: "processing.memory_limit_mb".to_string(),
                    default_value: serde_json::Value::Number(serde_json::Number::from(1024)),
                },
                MigrationOperation::AddField {
                    path: "processing.enable_memory_mapping".to_string(),
                    default_value: serde_json::Value::Bool(false),
                },
                MigrationOperation::RenameField {
                    old_path: "cache_size".to_string(),
                    new_path: "performance.cache_size_mb".to_string(),
                },
                MigrationOperation::MoveField {
                    old_path: "enable_gpu".to_string(),
                    new_path: "processing.enable_gpu".to_string(),
                },
                MigrationOperation::AddField {
                    path: "algorithms.wavelet".to_string(),
                    default_value: serde_json::json!({
                        "wavelet_type": "haar",
                        "decomposition_levels": 4,
                        "boundary_condition": "symmetric",
                        "enable_packet_decomposition": false
                    }),
                },
                MigrationOperation::AddField {
                    path: "algorithms.entropy".to_string(),
                    default_value: serde_json::json!({
                        "sample_entropy_m": 2,
                        "sample_entropy_r": 0.2,
                        "approximate_entropy_m": 2,
                        "approximate_entropy_r": 0.2,
                        "permutation_entropy_order": 3,
                        "enable_multiscale": false
                    }),
                },
                MigrationOperation::AddField {
                    path: "ml".to_string(),
                    default_value: serde_json::json!({
                        "enable_ml_processing": false,
                        "model_type": "neural_network",
                        "training": {
                            "learning_rate": 0.001,
                            "epochs": 100,
                            "batch_size": 32,
                            "early_stopping_patience": 10,
                            "regularization_strength": 0.01
                        },
                        "neural_network": {
                            "hidden_layers": [64, 32],
                            "activation_function": "relu",
                            "dropout_rate": 0.2,
                            "optimizer": "adam",
                            "loss_function": "mse"
                        },
                        "reinforcement_learning": {
                            "exploration_rate": 0.1,
                            "discount_factor": 0.99,
                            "replay_buffer_size": 10000,
                            "target_update_frequency": 100
                        },
                        "enable_ensemble": false,
                        "validation_strategy": "cross_validation",
                        "feature_selection_method": "variance_threshold"
                    }),
                },
                MigrationOperation::AddField {
                    path: "redis".to_string(),
                    default_value: serde_json::json!({
                        "enabled": false,
                        "host": "localhost",
                        "port": 6379,
                        "database": 0,
                        "pool_size": 10,
                        "timeout_ms": 5000,
                        "enable_cluster": false,
                        "cluster_nodes": []
                    }),
                },
                MigrationOperation::AddField {
                    path: "advanced".to_string(),
                    default_value: serde_json::json!({
                        "enable_neuromorphic": false,
                        "stdp_optimization": {
                            "learning_rate": 0.01,
                            "tau_positive": 20.0,
                            "tau_negative": 20.0,
                            "max_weight_change": 0.1
                        },
                        "enable_torchscript": false,
                        "enable_cross_asset": false,
                        "experimental_features": [],
                        "feature_flags": {}
                    }),
                },
            ],
        ));
        
        // Future migration example (1.0.x to 1.1.0)
        self.add_migration(MigrationStep::new(
            ConfigVersion::new(1, 0, 0),
            ConfigVersion::new(1, 1, 0),
            "Add new pattern detection features".to_string(),
            vec![
                MigrationOperation::AddField {
                    path: "algorithms.pattern_detection.enable_fractal_patterns".to_string(),
                    default_value: serde_json::Value::Bool(false),
                },
                MigrationOperation::AddField {
                    path: "algorithms.pattern_detection.fractal_dimension_threshold".to_string(),
                    default_value: serde_json::Value::Number(serde_json::Number::from_f64(1.5).unwrap()),
                },
                MigrationOperation::AddField {
                    path: "performance.enable_adaptive_batching".to_string(),
                    default_value: serde_json::Value::Bool(true),
                },
            ],
        ));
    }
    
    /// Add a migration step
    pub fn add_migration(&mut self, migration: MigrationStep) {
        self.migrations.push(migration);
        // Keep migrations sorted by version
        self.migrations.sort_by(|a, b| a.from_version.cmp(&b.from_version));
    }
    
    /// Find migration path from one version to another
    pub fn find_migration_path(&self, from: &ConfigVersion, to: &ConfigVersion) -> Result<Vec<&MigrationStep>> {
        let mut path = Vec::new();
        let mut current_version = from.clone();
        
        while current_version < *to {
            // Find next migration step
            let next_step = self.migrations.iter()
                .find(|step| step.from_version <= current_version && step.to_version > current_version)
                .ok_or_else(|| CdfaError::config_error(format!(
                    "No migration path found from {} to {}",
                    current_version, to
                )))?;
            
            path.push(next_step);
            current_version = next_step.to_version.clone();
        }
        
        Ok(path)
    }
    
    /// Apply migrations to raw configuration data
    pub fn migrate_raw_config(
        &self,
        config_data: &mut serde_json::Value,
        from_version: &ConfigVersion,
        to_version: &ConfigVersion,
    ) -> Result<Vec<String>> {
        let migration_path = self.find_migration_path(from_version, to_version)?;
        let mut applied_migrations = Vec::new();
        
        for step in migration_path {
            self.apply_migration_step(config_data, step)?;
            applied_migrations.push(format!("{} -> {}: {}", 
                step.from_version, step.to_version, step.description));
        }
        
        Ok(applied_migrations)
    }
    
    /// Apply a single migration step
    fn apply_migration_step(&self, config_data: &mut serde_json::Value, step: &MigrationStep) -> Result<()> {
        for operation in &step.operations {
            self.apply_operation(config_data, operation)?;
        }
        Ok(())
    }
    
    /// Apply a single migration operation
    fn apply_operation(&self, config_data: &mut serde_json::Value, operation: &MigrationOperation) -> Result<()> {
        match operation {
            MigrationOperation::AddField { path, default_value } => {
                self.set_field_value(config_data, path, default_value.clone())?;
            },
            MigrationOperation::RemoveField { path } => {
                self.remove_field(config_data, path)?;
            },
            MigrationOperation::RenameField { old_path, new_path } => {
                if let Some(value) = self.get_field_value(config_data, old_path)? {
                    self.set_field_value(config_data, new_path, value)?;
                    self.remove_field(config_data, old_path)?;
                }
            },
            MigrationOperation::MoveField { old_path, new_path } => {
                if let Some(value) = self.get_field_value(config_data, old_path)? {
                    self.set_field_value(config_data, new_path, value)?;
                    self.remove_field(config_data, old_path)?;
                }
            },
            MigrationOperation::TransformField { path, transformation: _ } => {
                // Placeholder for field transformations
                // In practice, this would contain specific transformation logic
                tracing::info!("Applying field transformation at path: {}", path);
            },
            MigrationOperation::SplitField { source_path, target_paths: _ } => {
                // Placeholder for field splitting logic
                tracing::info!("Splitting field at path: {}", source_path);
            },
            MigrationOperation::MergeFields { source_paths: _, target_path } => {
                // Placeholder for field merging logic
                tracing::info!("Merging fields to path: {}", target_path);
            },
        }
        Ok(())
    }
    
    /// Get field value at path
    fn get_field_value(&self, config_data: &serde_json::Value, path: &str) -> Result<Option<serde_json::Value>> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = config_data;
        
        for part in parts {
            if let Some(obj) = current.as_object() {
                if let Some(value) = obj.get(part) {
                    current = value;
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }
        
        Ok(Some(current.clone()))
    }
    
    /// Set field value at path
    fn set_field_value(&self, config_data: &mut serde_json::Value, path: &str, value: serde_json::Value) -> Result<()> {
        let parts: Vec<&str> = path.split('.').collect();
        
        if parts.is_empty() {
            return Err(CdfaError::config_error("Empty path"));
        }
        
        let mut current = config_data;
        
        // Navigate to parent object, creating objects as needed
        for part in &parts[..parts.len() - 1] {
            if !current.is_object() {
                *current = serde_json::json!({});
            }
            
            let obj = current.as_object_mut().unwrap();
            if !obj.contains_key(*part) {
                obj.insert(part.to_string(), serde_json::json!({}));
            }
            current = obj.get_mut(*part).unwrap();
        }
        
        // Set the final value
        if !current.is_object() {
            *current = serde_json::json!({});
        }
        
        let obj = current.as_object_mut().unwrap();
        obj.insert(parts.last().unwrap().to_string(), value);
        
        Ok(())
    }
    
    /// Remove field at path
    fn remove_field(&self, config_data: &mut serde_json::Value, path: &str) -> Result<()> {
        let parts: Vec<&str> = path.split('.').collect();
        
        if parts.is_empty() {
            return Err(CdfaError::config_error("Empty path"));
        }
        
        let mut current = config_data;
        
        // Navigate to parent object
        for part in &parts[..parts.len() - 1] {
            if let Some(obj) = current.as_object_mut() {
                if let Some(value) = obj.get_mut(*part) {
                    current = value;
                } else {
                    return Ok(()); // Path doesn't exist, nothing to remove
                }
            } else {
                return Ok(()); // Not an object, nothing to remove
            }
        }
        
        // Remove the final key
        if let Some(obj) = current.as_object_mut() {
            obj.remove(parts.last().unwrap());
        }
        
        Ok(())
    }
    
    /// Get list of all available migrations
    pub fn get_available_migrations(&self) -> &[MigrationStep] {
        &self.migrations
    }
    
    /// Check if configuration needs migration
    pub fn needs_migration(&self, current_version: &ConfigVersion, target_version: &ConfigVersion) -> bool {
        current_version < target_version
    }
}

impl Default for ConfigMigrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply migrations to a configuration
pub fn apply_migrations(config: CdfaConfig) -> Result<CdfaConfig> {
    // For now, return the config as-is since we don't have version tracking in the config yet
    // In a real implementation, we would:
    // 1. Extract version from config
    // 2. Compare with current version
    // 3. Apply necessary migrations
    // 4. Update version in config
    Ok(config)
}

/// Detect configuration version from raw data
pub fn detect_config_version(config_data: &serde_json::Value) -> ConfigVersion {
    // Try to extract version from config
    if let Some(version_str) = config_data.get("version")
        .and_then(|v| v.as_str()) {
        if let Ok(version) = ConfigVersion::parse(version_str) {
            return version;
        }
    }
    
    // Heuristics to detect version based on structure
    if config_data.get("advanced").is_some() {
        ConfigVersion::new(1, 0, 0) // Has advanced config, probably 1.0+
    } else if config_data.get("ml").is_some() {
        ConfigVersion::new(0, 9, 0) // Has ML config, probably 0.9+
    } else {
        ConfigVersion::new(0, 8, 0) // Older version
    }
}

/// Create a migration report
pub fn create_migration_report(from: &ConfigVersion, to: &ConfigVersion, applied: &[String]) -> String {
    let mut report = String::new();
    report.push_str(&format!("Configuration Migration Report\n"));
    report.push_str(&format!("=============================\n\n"));
    report.push_str(&format!("From Version: {}\n", from));
    report.push_str(&format!("To Version: {}\n", to));
    report.push_str(&format!("Applied Migrations: {}\n\n", applied.len()));
    
    for (i, migration) in applied.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i + 1, migration));
    }
    
    report.push_str(&format!("\nMigration completed successfully.\n"));
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_version() {
        let v1 = ConfigVersion::new(1, 0, 0);
        let v2 = ConfigVersion::new(1, 1, 0);
        let v3 = ConfigVersion::new(2, 0, 0);
        
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }
    
    #[test]
    fn test_version_parsing() {
        let version = ConfigVersion::parse("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        
        assert!(ConfigVersion::parse("invalid").is_err());
        assert!(ConfigVersion::parse("1.2").is_err());
    }
    
    #[test]
    fn test_migration_operations() {
        let migrator = ConfigMigrator::new();
        let mut config = serde_json::json!({
            "old_field": "value"
        });
        
        let operation = MigrationOperation::RenameField {
            old_path: "old_field".to_string(),
            new_path: "new_field".to_string(),
        };
        
        migrator.apply_operation(&mut config, &operation).unwrap();
        
        assert!(config.get("old_field").is_none());
        assert_eq!(config.get("new_field").unwrap().as_str(), Some("value"));
    }
    
    #[test]
    fn test_nested_field_operations() {
        let migrator = ConfigMigrator::new();
        let mut config = serde_json::json!({});
        
        let operation = MigrationOperation::AddField {
            path: "processing.num_threads".to_string(),
            default_value: serde_json::Value::Number(serde_json::Number::from(4)),
        };
        
        migrator.apply_operation(&mut config, &operation).unwrap();
        
        assert_eq!(
            config.get("processing")
                .and_then(|p| p.get("num_threads"))
                .and_then(|v| v.as_u64()),
            Some(4)
        );
    }
    
    #[test]
    fn test_version_detection() {
        let config_with_version = serde_json::json!({
            "version": "1.0.0",
            "processing": {}
        });
        
        let detected = detect_config_version(&config_with_version);
        assert_eq!(detected, ConfigVersion::new(1, 0, 0));
        
        let config_without_version = serde_json::json!({
            "advanced": {},
            "processing": {}
        });
        
        let detected = detect_config_version(&config_without_version);
        assert_eq!(detected, ConfigVersion::new(1, 0, 0));
    }
    
    #[test]
    fn test_migration_path() {
        let migrator = ConfigMigrator::new();
        let from = ConfigVersion::new(0, 9, 0);
        let to = ConfigVersion::new(1, 0, 0);
        
        let path = migrator.find_migration_path(&from, &to).unwrap();
        assert!(!path.is_empty());
    }
}