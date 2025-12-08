//! Unified configuration management system for CDFA
//!
//! This module provides comprehensive configuration management for all CDFA components,
//! including hierarchical configuration, runtime validation, environment variable overrides,
//! and hardware detection capabilities.

pub mod cdfa_config;
pub mod environment;
pub mod hardware_config;
pub mod migration;
pub mod validation;

// Re-export main types
pub use cdfa_config::*;
pub use environment::*;
pub use hardware_config::*;
pub use migration::*;
pub use validation::*;

use crate::error::{CdfaError, Result};
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration file formats supported by CDFA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConfigFormat {
    /// JSON format
    Json,
    /// TOML format  
    Toml,
    /// YAML format
    Yaml,
    /// RON (Rust Object Notation) format
    Ron,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext.to_lowercase().as_str() {
                "json" => Self::Json,
                "toml" => Self::Toml,
                "yaml" | "yml" => Self::Yaml,
                "ron" => Self::Ron,
                _ => Self::Json, // Default to JSON
            })
    }
    
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Toml => "toml", 
            Self::Yaml => "yaml",
            Self::Ron => "ron",
        }
    }
}

/// Configuration loading options
#[derive(Debug, Clone)]
pub struct ConfigLoadOptions {
    /// Allow environment variable overrides
    pub allow_env_overrides: bool,
    /// Validate configuration after loading
    pub validate_on_load: bool,
    /// Apply migrations if needed
    pub apply_migrations: bool,
    /// Merge with default configuration
    pub merge_with_defaults: bool,
    /// Strict mode - fail on unknown fields
    pub strict_mode: bool,
}

impl Default for ConfigLoadOptions {
    fn default() -> Self {
        Self {
            allow_env_overrides: true,
            validate_on_load: true,
            apply_migrations: true,
            merge_with_defaults: true,
            strict_mode: false,
        }
    }
}

// Integration tests module is defined at the end of this file

/// Main configuration loader
pub struct ConfigLoader {
    options: ConfigLoadOptions,
    hardware_config: Option<HardwareConfig>,
}

impl ConfigLoader {
    /// Create a new configuration loader
    pub fn new() -> Self {
        Self {
            options: ConfigLoadOptions::default(),
            hardware_config: None,
        }
    }
    
    /// Create with custom options
    pub fn with_options(options: ConfigLoadOptions) -> Self {
        Self {
            options,
            hardware_config: None,
        }
    }
    
    /// Enable hardware detection
    pub fn with_hardware_detection(mut self) -> Result<Self> {
        self.hardware_config = Some(HardwareConfig::detect()?);
        Ok(self)
    }
    
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<CdfaConfig> {
        let path = path.as_ref();
        let format = ConfigFormat::from_extension(path)
            .ok_or_else(|| CdfaError::config_error("Unable to detect config file format"))?;
            
        let content = std::fs::read_to_string(path)
            .map_err(|e| CdfaError::config_error(format!("Failed to read config file: {}", e)))?;
            
        self.load_from_string(&content, format)
    }
    
    /// Load configuration from string
    pub fn load_from_string(&self, content: &str, format: ConfigFormat) -> Result<CdfaConfig> {
        #[cfg(feature = "serde")]
        {
            let mut config: CdfaConfig = match format {
                ConfigFormat::Json => serde_json::from_str(content)
                    .map_err(|e| CdfaError::config_error(format!("JSON parse error: {}", e)))?,
                ConfigFormat::Toml => {
                    #[cfg(feature = "toml")]
                    {
                        toml::from_str(content)
                            .map_err(|e| CdfaError::config_error(format!("TOML parse error: {}", e)))?
                    }
                    #[cfg(not(feature = "toml"))]
                    {
                        return Err(CdfaError::config_error("TOML support not enabled"));
                    }
                },
                ConfigFormat::Yaml => {
                    #[cfg(feature = "yaml")]
                    {
                        serde_yaml::from_str(content)
                            .map_err(|e| CdfaError::config_error(format!("YAML parse error: {}", e)))?
                    }
                    #[cfg(not(feature = "yaml"))]
                    {
                        return Err(CdfaError::config_error("YAML support not enabled"));
                    }
                },
                ConfigFormat::Ron => {
                    #[cfg(feature = "ron")]
                    {
                        ron::from_str(content)
                            .map_err(|e| CdfaError::config_error(format!("RON parse error: {}", e)))?
                    }
                    #[cfg(not(feature = "ron"))]
                    {
                        return Err(CdfaError::config_error("RON support not enabled"));
                    }
                },
            };
            
            // Apply processing pipeline
            if self.options.merge_with_defaults {
                config = config.merge_with_defaults();
            }
            
            if self.options.apply_migrations {
                config = apply_migrations(config)?;
            }
            
            if self.options.allow_env_overrides {
                config = apply_environment_overrides(config)?;
            }
            
            if let Some(ref hw_config) = self.hardware_config {
                config = config.apply_hardware_config(hw_config.clone())?;
            }
            
            if self.options.validate_on_load {
                validate_config(&config)?;
            }
            
            Ok(config)
        }
        
        #[cfg(not(feature = "serde"))]
        Err(CdfaError::feature_not_enabled("serde"))
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, config: &CdfaConfig, path: P) -> Result<()> {
        let path = path.as_ref();
        let format = ConfigFormat::from_extension(path)
            .ok_or_else(|| CdfaError::config_error("Unable to detect config file format"))?;
            
        let content = self.save_to_string(config, format)?;
        
        std::fs::write(path, content)
            .map_err(|e| CdfaError::config_error(format!("Failed to write config file: {}", e)))?;
            
        Ok(())
    }
    
    /// Save configuration to string
    pub fn save_to_string(&self, config: &CdfaConfig, format: ConfigFormat) -> Result<String> {
        #[cfg(feature = "serde")]
        {
            match format {
                ConfigFormat::Json => serde_json::to_string_pretty(config)
                    .map_err(|e| CdfaError::config_error(format!("JSON serialize error: {}", e))),
                ConfigFormat::Toml => {
                    #[cfg(feature = "toml")]
                    {
                        toml::to_string_pretty(config)
                            .map_err(|e| CdfaError::config_error(format!("TOML serialize error: {}", e)))
                    }
                    #[cfg(not(feature = "toml"))]
                    {
                        Err(CdfaError::config_error("TOML support not enabled"))
                    }
                },
                ConfigFormat::Yaml => {
                    #[cfg(feature = "yaml")]
                    {
                        serde_yaml::to_string(config)
                            .map_err(|e| CdfaError::config_error(format!("YAML serialize error: {}", e)))
                    }
                    #[cfg(not(feature = "yaml"))]
                    {
                        Err(CdfaError::config_error("YAML support not enabled"))
                    }
                },
                ConfigFormat::Ron => {
                    #[cfg(feature = "ron")]
                    {
                        ron::ser::to_string_pretty(config, ron::ser::PrettyConfig::default())
                            .map_err(|e| CdfaError::config_error(format!("RON serialize error: {}", e)))
                    }
                    #[cfg(not(feature = "ron"))]
                    {
                        Err(CdfaError::config_error("RON support not enabled"))
                    }
                },
            }
        }
        
        #[cfg(not(feature = "serde"))]
        Err(CdfaError::feature_not_enabled("serde"))
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_format_detection() {
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.json")),
            Some(ConfigFormat::Json)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.toml")),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.yaml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.yml")),
            Some(ConfigFormat::Yaml)
        );
    }
    
    #[test]
    fn test_config_loader_creation() {
        let loader = ConfigLoader::new();
        assert!(loader.hardware_config.is_none());
        
        let options = ConfigLoadOptions {
            strict_mode: true,
            ..Default::default()
        };
        let loader = ConfigLoader::with_options(options);
        assert!(loader.options.strict_mode);
    }
    
    #[cfg(feature = "serde")]
    #[test]
    fn test_config_serialization() {
        let config = CdfaConfig::default();
        let loader = ConfigLoader::new();
        
        let json = loader.save_to_string(&config, ConfigFormat::Json).unwrap();
        assert!(json.contains("processing"));
        
        let loaded = loader.load_from_string(&json, ConfigFormat::Json).unwrap();
        // Basic comparison - detailed comparison in config tests
        assert_eq!(config.processing.num_threads, loaded.processing.num_threads);
    }
}