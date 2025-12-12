//! Environment variable configuration override system
//!
//! This module provides functionality to override configuration parameters using
//! environment variables, following a hierarchical naming convention.

use crate::config::CdfaConfig;
use crate::error::{CdfaError, Result};
use std::collections::HashMap;
use std::env;
// FromStr available through std if needed

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Environment variable configuration override system
pub struct EnvironmentConfigLoader {
    prefix: String,
    case_sensitive: bool,
    type_conversions: HashMap<String, Box<dyn Fn(&str) -> Result<ConfigValue>>>,
}

/// Supported configuration value types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    UnsignedInteger(u64),
    Float(f64),
    Boolean(bool),
    StringList(Vec<String>),
    IntegerList(Vec<i64>),
    FloatList(Vec<f64>),
}

impl ConfigValue {
    /// Convert to string
    pub fn as_string(&self) -> Option<String> {
        match self {
            ConfigValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }
    
    /// Convert to integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            _ => None,
        }
    }
    
    /// Convert to unsigned integer
    pub fn as_unsigned_integer(&self) -> Option<u64> {
        match self {
            ConfigValue::UnsignedInteger(u) => Some(*u),
            _ => None,
        }
    }
    
    /// Convert to float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    
    /// Convert to boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Convert to string list
    pub fn as_string_list(&self) -> Option<Vec<String>> {
        match self {
            ConfigValue::StringList(list) => Some(list.clone()),
            _ => None,
        }
    }
    
    /// Convert to integer list
    pub fn as_integer_list(&self) -> Option<Vec<i64>> {
        match self {
            ConfigValue::IntegerList(list) => Some(list.clone()),
            _ => None,
        }
    }
    
    /// Convert to float list
    pub fn as_float_list(&self) -> Option<Vec<f64>> {
        match self {
            ConfigValue::FloatList(list) => Some(list.clone()),
            _ => None,
        }
    }
}

impl EnvironmentConfigLoader {
    /// Create a new environment configuration loader
    pub fn new() -> Self {
        let mut loader = Self {
            prefix: "CDFA_".to_string(),
            case_sensitive: false,
            type_conversions: HashMap::new(),
        };
        
        // Set up default type conversions
        loader.setup_default_conversions();
        loader
    }
    
    /// Create with custom prefix
    pub fn with_prefix<S: Into<String>>(prefix: S) -> Self {
        let mut loader = Self::new();
        loader.prefix = prefix.into();
        loader
    }
    
    /// Enable case-sensitive matching
    pub fn case_sensitive(mut self) -> Self {
        self.case_sensitive = true;
        self
    }
    
    /// Setup default type conversion functions
    fn setup_default_conversions(&mut self) {
        // String conversion
        self.type_conversions.insert(
            "string".to_string(),
            Box::new(|s| Ok(ConfigValue::String(s.to_string()))),
        );
        
        // Integer conversion
        self.type_conversions.insert(
            "integer".to_string(),
            Box::new(|s| {
                s.parse::<i64>()
                    .map(ConfigValue::Integer)
                    .map_err(|e| CdfaError::config_error(format!("Invalid integer: {}", e)))
            }),
        );
        
        // Unsigned integer conversion
        self.type_conversions.insert(
            "usize".to_string(),
            Box::new(|s| {
                s.parse::<u64>()
                    .map(ConfigValue::UnsignedInteger)
                    .map_err(|e| CdfaError::config_error(format!("Invalid unsigned integer: {}", e)))
            }),
        );
        
        // Float conversion
        self.type_conversions.insert(
            "float".to_string(),
            Box::new(|s| {
                s.parse::<f64>()
                    .map(ConfigValue::Float)
                    .map_err(|e| CdfaError::config_error(format!("Invalid float: {}", e)))
            }),
        );
        
        // Boolean conversion
        self.type_conversions.insert(
            "boolean".to_string(),
            Box::new(|s| {
                match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" | "enable" | "enabled" => Ok(ConfigValue::Boolean(true)),
                    "false" | "0" | "no" | "off" | "disable" | "disabled" => Ok(ConfigValue::Boolean(false)),
                    _ => Err(CdfaError::config_error(format!("Invalid boolean: {}", s))),
                }
            }),
        );
        
        // String list conversion (comma-separated)
        self.type_conversions.insert(
            "string_list".to_string(),
            Box::new(|s| {
                let list: Vec<String> = s.split(',')
                    .map(|item| item.trim().to_string())
                    .filter(|item| !item.is_empty())
                    .collect();
                Ok(ConfigValue::StringList(list))
            }),
        );
        
        // Integer list conversion (comma-separated)
        self.type_conversions.insert(
            "integer_list".to_string(),
            Box::new(|s| {
                let list: Result<Vec<i64>> = s.split(',')
                    .map(|item| item.trim())
                    .filter(|item| !item.is_empty())
                    .map(|item| item.parse::<i64>()
                        .map_err(|e| CdfaError::config_error(format!("Invalid integer in list: {}", e))))
                    .collect();
                list.map(ConfigValue::IntegerList)
            }),
        );
        
        // Float list conversion (comma-separated)
        self.type_conversions.insert(
            "float_list".to_string(),
            Box::new(|s| {
                let list: Result<Vec<f64>> = s.split(',')
                    .map(|item| item.trim())
                    .filter(|item| !item.is_empty())
                    .map(|item| item.parse::<f64>()
                        .map_err(|e| CdfaError::config_error(format!("Invalid float in list: {}", e))))
                    .collect();
                list.map(ConfigValue::FloatList)
            }),
        );
    }
    
    /// Load environment variable overrides
    pub fn load_overrides(&self) -> Result<HashMap<String, ConfigValue>> {
        let mut overrides = HashMap::new();
        
        // Get all environment variables with our prefix
        for (key, value) in env::vars() {
            if self.matches_prefix(&key) {
                let config_path = self.env_key_to_config_path(&key)?;
                let config_value = self.parse_env_value(&config_path, &value)?;
                overrides.insert(config_path, config_value);
            }
        }
        
        Ok(overrides)
    }
    
    /// Check if environment variable key matches our prefix
    fn matches_prefix(&self, key: &str) -> bool {
        if self.case_sensitive {
            key.starts_with(&self.prefix)
        } else {
            key.to_uppercase().starts_with(&self.prefix.to_uppercase())
        }
    }
    
    /// Convert environment variable key to configuration path
    fn env_key_to_config_path(&self, env_key: &str) -> Result<String> {
        let key_string = if self.case_sensitive {
            env_key.strip_prefix(&self.prefix)
                .ok_or_else(|| CdfaError::config_error("Invalid environment variable key"))?
                .to_string()
        } else {
            let upper_env_key = env_key.to_uppercase();
            let upper_prefix = self.prefix.to_uppercase();
            upper_env_key.strip_prefix(&upper_prefix)
                .ok_or_else(|| CdfaError::config_error("Invalid environment variable key"))?
                .to_string()
        };
        
        // Convert from UPPER_CASE_WITH_UNDERSCORES to nested.path.format
        let path = key_string.to_lowercase().replace('_', ".");
        Ok(path)
    }
    
    /// Parse environment variable value to appropriate type
    fn parse_env_value(&self, config_path: &str, value: &str) -> Result<ConfigValue> {
        // Determine type based on configuration path
        let value_type = self.infer_type_from_path(config_path);
        
        // Use appropriate conversion function
        if let Some(converter) = self.type_conversions.get(&value_type) {
            converter(value)
        } else {
            // Default to string if type cannot be determined
            Ok(ConfigValue::String(value.to_string()))
        }
    }
    
    /// Infer configuration value type from path
    fn infer_type_from_path(&self, path: &str) -> String {
        // Define type mappings based on configuration structure
        let type_mappings = [
            // Processing config
            ("processing.num.threads", "usize"),
            ("processing.enable.simd", "boolean"),
            ("processing.enable.gpu", "boolean"),
            ("processing.tolerance", "float"),
            ("processing.max.iterations", "usize"),
            ("processing.convergence.threshold", "float"),
            ("processing.batch.size", "usize"),
            ("processing.enable.parallel.batches", "boolean"),
            ("processing.memory.limit.mb", "usize"),
            ("processing.enable.memory.mapping", "boolean"),
            ("processing.enable.distributed", "boolean"),
            ("processing.process.priority", "integer"),
            
            // Algorithm config
            ("algorithms.diversity.methods", "string_list"),
            ("algorithms.fusion.methods", "string_list"),
            ("algorithms.default.strategy", "string"),
            ("algorithms.enable.auto.selection", "boolean"),
            
            // Wavelet config
            ("algorithms.wavelet.wavelet.type", "string"),
            ("algorithms.wavelet.decomposition.levels", "usize"),
            ("algorithms.wavelet.boundary.condition", "string"),
            ("algorithms.wavelet.enable.packet.decomposition", "boolean"),
            
            // Entropy config
            ("algorithms.entropy.sample.entropy.m", "usize"),
            ("algorithms.entropy.sample.entropy.r", "float"),
            ("algorithms.entropy.approximate.entropy.m", "usize"),
            ("algorithms.entropy.approximate.entropy.r", "float"),
            ("algorithms.entropy.permutation.entropy.order", "usize"),
            ("algorithms.entropy.enable.multiscale", "boolean"),
            
            // Performance config
            ("performance.cache.size.mb", "usize"),
            ("performance.enable.caching", "boolean"),
            ("performance.cache.eviction.policy", "string"),
            ("performance.enable.profiling", "boolean"),
            ("performance.optimization.level", "usize"),
            ("performance.enable.jit", "boolean"),
            ("performance.enable.prefetching", "boolean"),
            ("performance.memory.allocator", "string"),
            
            // ML config
            ("ml.enable.ml.processing", "boolean"),
            ("ml.model.type", "string"),
            ("ml.training.learning.rate", "float"),
            ("ml.training.epochs", "usize"),
            ("ml.training.batch.size", "usize"),
            ("ml.training.early.stopping.patience", "usize"),
            ("ml.training.regularization.strength", "float"),
            ("ml.neural.network.hidden.layers", "integer_list"),
            ("ml.neural.network.activation.function", "string"),
            ("ml.neural.network.dropout.rate", "float"),
            ("ml.neural.network.optimizer", "string"),
            ("ml.neural.network.loss.function", "string"),
            
            // Data config
            ("data.input.format", "string"),
            ("data.enable.preprocessing", "boolean"),
            ("data.normalization.method", "string"),
            ("data.missing.value.strategy", "string"),
            ("data.outlier.handling", "string"),
            ("data.enable.realtime", "boolean"),
            ("data.quality.thresholds.max.missing.percentage", "float"),
            ("data.quality.thresholds.min.snr.db", "float"),
            ("data.quality.thresholds.max.outlier.percentage", "float"),
            ("data.quality.thresholds.min.data.points", "usize"),
            
            // Redis config
            ("redis.enabled", "boolean"),
            ("redis.host", "string"),
            ("redis.port", "usize"),
            ("redis.database", "usize"),
            ("redis.pool.size", "usize"),
            ("redis.timeout.ms", "usize"),
            ("redis.enable.cluster", "boolean"),
            ("redis.cluster.nodes", "string_list"),
            
            // Logging config
            ("logging.level", "string"),
            ("logging.enable.file.logging", "boolean"),
            ("logging.log.file.path", "string"),
            ("logging.enable.metrics", "boolean"),
            ("logging.metrics.interval.seconds", "usize"),
        ];
        
        // Find matching type
        for (pattern, type_name) in &type_mappings {
            if path == *pattern {
                return type_name.to_string();
            }
        }
        
        // Default inference based on common patterns
        if path.contains(".enable.") || path.contains(".enabled") {
            "boolean".to_string()
        } else if path.contains(".size") || path.contains(".count") || path.contains(".port") || 
                  path.contains(".threads") || path.contains(".iterations") || path.contains(".levels") {
            "usize".to_string()
        } else if path.contains(".rate") || path.contains(".threshold") || path.contains(".tolerance") || 
                  path.contains(".percentage") {
            "float".to_string()
        } else if path.contains(".methods") || path.contains(".features") || path.contains(".nodes") {
            "string_list".to_string()
        } else {
            "string".to_string()
        }
    }
}

impl Default for EnvironmentConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply environment variable overrides to configuration
pub fn apply_environment_overrides(mut config: CdfaConfig) -> Result<CdfaConfig> {
    let loader = EnvironmentConfigLoader::new();
    let overrides = loader.load_overrides()?;
    
    // Apply overrides to configuration
    for (path, value) in overrides {
        apply_override_to_config(&mut config, &path, value)?;
    }
    
    Ok(config)
}

/// Apply a single override to the configuration
fn apply_override_to_config(config: &mut CdfaConfig, path: &str, value: ConfigValue) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    
    match parts.as_slice() {
        ["processing", "num", "threads"] => {
            if let Some(v) = value.as_unsigned_integer() {
                config.processing.num_threads = v as usize;
            }
        },
        ["processing", "enable", "simd"] => {
            if let Some(v) = value.as_boolean() {
                config.processing.enable_simd = v;
            }
        },
        ["processing", "enable", "gpu"] => {
            if let Some(v) = value.as_boolean() {
                config.processing.enable_gpu = v;
            }
        },
        ["processing", "tolerance"] => {
            if let Some(v) = value.as_float() {
                config.processing.tolerance = v;
            }
        },
        ["processing", "max", "iterations"] => {
            if let Some(v) = value.as_unsigned_integer() {
                config.processing.max_iterations = v as usize;
            }
        },
        ["processing", "convergence", "threshold"] => {
            if let Some(v) = value.as_float() {
                config.processing.convergence_threshold = v;
            }
        },
        ["processing", "batch", "size"] => {
            if let Some(v) = value.as_unsigned_integer() {
                config.processing.batch_size = v as usize;
            }
        },
        ["algorithms", "diversity", "methods"] => {
            if let Some(v) = value.as_string_list() {
                config.algorithms.diversity_methods = v;
            }
        },
        ["algorithms", "fusion", "methods"] => {
            if let Some(v) = value.as_string_list() {
                config.algorithms.fusion_methods = v;
            }
        },
        ["performance", "cache", "size", "mb"] => {
            if let Some(v) = value.as_unsigned_integer() {
                config.performance.cache_size_mb = v as usize;
            }
        },
        ["performance", "enable", "caching"] => {
            if let Some(v) = value.as_boolean() {
                config.performance.enable_caching = v;
            }
        },
        ["ml", "enable", "ml", "processing"] => {
            if let Some(v) = value.as_boolean() {
                config.ml.enable_ml_processing = v;
            }
        },
        ["ml", "training", "learning", "rate"] => {
            if let Some(v) = value.as_float() {
                config.ml.training.learning_rate = v;
            }
        },
        ["redis", "enabled"] => {
            if let Some(v) = value.as_boolean() {
                config.redis.enabled = v;
            }
        },
        ["redis", "host"] => {
            if let Some(v) = value.as_string() {
                config.redis.host = v;
            }
        },
        ["redis", "port"] => {
            if let Some(v) = value.as_unsigned_integer() {
                config.redis.port = v as u16;
            }
        },
        ["logging", "level"] => {
            if let Some(v) = value.as_string() {
                config.logging.level = v;
            }
        },
        // Add more overrides as needed...
        _ => {
            // Log unknown override path but don't fail
            tracing::warn!("Unknown configuration override path: {}", path);
        }
    }
    
    Ok(())
}

/// Get all available environment variable names for configuration
pub fn get_available_env_vars() -> Vec<String> {
    vec![
        // Processing
        "CDFA_PROCESSING_NUM_THREADS".to_string(),
        "CDFA_PROCESSING_ENABLE_SIMD".to_string(),
        "CDFA_PROCESSING_ENABLE_GPU".to_string(),
        "CDFA_PROCESSING_TOLERANCE".to_string(),
        "CDFA_PROCESSING_MAX_ITERATIONS".to_string(),
        "CDFA_PROCESSING_CONVERGENCE_THRESHOLD".to_string(),
        "CDFA_PROCESSING_BATCH_SIZE".to_string(),
        "CDFA_PROCESSING_ENABLE_PARALLEL_BATCHES".to_string(),
        "CDFA_PROCESSING_MEMORY_LIMIT_MB".to_string(),
        "CDFA_PROCESSING_ENABLE_MEMORY_MAPPING".to_string(),
        "CDFA_PROCESSING_ENABLE_DISTRIBUTED".to_string(),
        "CDFA_PROCESSING_PROCESS_PRIORITY".to_string(),
        
        // Algorithms
        "CDFA_ALGORITHMS_DIVERSITY_METHODS".to_string(),
        "CDFA_ALGORITHMS_FUSION_METHODS".to_string(),
        "CDFA_ALGORITHMS_DEFAULT_STRATEGY".to_string(),
        "CDFA_ALGORITHMS_ENABLE_AUTO_SELECTION".to_string(),
        
        // Performance
        "CDFA_PERFORMANCE_CACHE_SIZE_MB".to_string(),
        "CDFA_PERFORMANCE_ENABLE_CACHING".to_string(),
        "CDFA_PERFORMANCE_ENABLE_PROFILING".to_string(),
        "CDFA_PERFORMANCE_OPTIMIZATION_LEVEL".to_string(),
        
        // ML
        "CDFA_ML_ENABLE_ML_PROCESSING".to_string(),
        "CDFA_ML_MODEL_TYPE".to_string(),
        "CDFA_ML_TRAINING_LEARNING_RATE".to_string(),
        "CDFA_ML_TRAINING_EPOCHS".to_string(),
        "CDFA_ML_TRAINING_BATCH_SIZE".to_string(),
        
        // Redis
        "CDFA_REDIS_ENABLED".to_string(),
        "CDFA_REDIS_HOST".to_string(),
        "CDFA_REDIS_PORT".to_string(),
        "CDFA_REDIS_DATABASE".to_string(),
        "CDFA_REDIS_POOL_SIZE".to_string(),
        "CDFA_REDIS_TIMEOUT_MS".to_string(),
        
        // Logging
        "CDFA_LOGGING_LEVEL".to_string(),
        "CDFA_LOGGING_ENABLE_FILE_LOGGING".to_string(),
        "CDFA_LOGGING_LOG_FILE_PATH".to_string(),
        "CDFA_LOGGING_ENABLE_METRICS".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_environment_loader_creation() {
        let loader = EnvironmentConfigLoader::new();
        assert_eq!(loader.prefix, "CDFA_");
        assert!(!loader.case_sensitive);
    }
    
    #[test]
    fn test_custom_prefix() {
        let loader = EnvironmentConfigLoader::with_prefix("TEST_");
        assert_eq!(loader.prefix, "TEST_");
    }
    
    #[test]
    fn test_config_value_conversions() {
        let string_val = ConfigValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test".to_string()));
        assert_eq!(string_val.as_integer(), None);
        
        let int_val = ConfigValue::Integer(42);
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(int_val.as_string(), None);
        
        let bool_val = ConfigValue::Boolean(true);
        assert_eq!(bool_val.as_boolean(), Some(true));
    }
    
    #[test]
    fn test_env_key_to_config_path() {
        let loader = EnvironmentConfigLoader::new();
        let path = loader.env_key_to_config_path("CDFA_PROCESSING_NUM_THREADS").unwrap();
        assert_eq!(path, "processing.num.threads");
    }
    
    #[test]
    fn test_type_inference() {
        let loader = EnvironmentConfigLoader::new();
        
        assert_eq!(loader.infer_type_from_path("processing.enable.simd"), "boolean");
        assert_eq!(loader.infer_type_from_path("processing.num.threads"), "usize");
        assert_eq!(loader.infer_type_from_path("processing.tolerance"), "float");
        assert_eq!(loader.infer_type_from_path("algorithms.diversity.methods"), "string_list");
    }
    
    #[test]
    fn test_environment_override() {
        // Set test environment variable
        env::set_var("CDFA_PROCESSING_NUM_THREADS", "8");
        env::set_var("CDFA_PROCESSING_ENABLE_SIMD", "false");
        env::set_var("CDFA_PROCESSING_TOLERANCE", "1e-8");
        
        let config = CdfaConfig::default();
        let result = apply_environment_overrides(config);
        assert!(result.is_ok());
        
        let modified_config = result.unwrap();
        assert_eq!(modified_config.processing.num_threads, 8);
        assert!(!modified_config.processing.enable_simd);
        assert_eq!(modified_config.processing.tolerance, 1e-8);
        
        // Clean up
        env::remove_var("CDFA_PROCESSING_NUM_THREADS");
        env::remove_var("CDFA_PROCESSING_ENABLE_SIMD");
        env::remove_var("CDFA_PROCESSING_TOLERANCE");
    }
    
    #[test]
    fn test_available_env_vars() {
        let vars = get_available_env_vars();
        assert!(!vars.is_empty());
        assert!(vars.contains(&"CDFA_PROCESSING_NUM_THREADS".to_string()));
        assert!(vars.contains(&"CDFA_REDIS_ENABLED".to_string()));
    }
}