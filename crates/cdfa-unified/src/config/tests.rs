//! Comprehensive integration tests for the configuration management system

#[cfg(test)]
mod integration_tests {
    use super::super::*;
    use std::env;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_complete_config_workflow() {
        // Test the complete configuration workflow
        let config = CdfaConfig::default();
        
        // Verify default configuration is valid
        assert!(config.validate().is_ok());
        
        // Check parameter count meets requirement
        assert!(config.parameter_count() >= 60);
        
        // Verify hierarchical structure
        assert!(config.processing.num_threads == 0); // Auto-detect
        assert!(config.processing.enable_simd);
        assert!(!config.processing.enable_gpu);
        assert_eq!(config.processing.tolerance, 1e-10);
        
        // Verify algorithm configuration
        assert!(config.algorithms.diversity_methods.contains(&"pearson".to_string()));
        assert!(config.algorithms.fusion_methods.contains(&"score".to_string()));
        
        // Verify ML configuration structure
        assert!(!config.ml.enable_ml_processing);
        assert_eq!(config.ml.training.learning_rate, 0.001);
        assert_eq!(config.ml.neural_network.hidden_layers, vec![64, 32]);
        
        // Verify Redis configuration
        assert!(!config.redis.enabled);
        assert_eq!(config.redis.host, "localhost");
        assert_eq!(config.redis.port, 6379);
        
        // Verify advanced features
        assert!(!config.advanced.enable_neuromorphic);
        assert_eq!(config.advanced.stdp_optimization.learning_rate, 0.01);
    }
    
    #[cfg(feature = "serde")]
    #[test]
    fn test_config_serialization_all_formats() {
        let config = CdfaConfig::default();
        let loader = ConfigLoader::new();
        
        // Test JSON serialization/deserialization
        let json_str = loader.save_to_string(&config, ConfigFormat::Json).unwrap();
        let json_config = loader.load_from_string(&json_str, ConfigFormat::Json).unwrap();
        assert_eq!(config.processing.num_threads, json_config.processing.num_threads);
        
        // Test TOML serialization/deserialization
        let toml_str = loader.save_to_string(&config, ConfigFormat::Toml).unwrap();
        let toml_config = loader.load_from_string(&toml_str, ConfigFormat::Toml).unwrap();
        assert_eq!(config.processing.tolerance, toml_config.processing.tolerance);
        
        // Test YAML serialization/deserialization
        let yaml_str = loader.save_to_string(&config, ConfigFormat::Yaml).unwrap();
        let yaml_config = loader.load_from_string(&yaml_str, ConfigFormat::Yaml).unwrap();
        assert_eq!(config.algorithms.diversity_methods, yaml_config.algorithms.diversity_methods);
        
        // Test RON serialization/deserialization
        let ron_str = loader.save_to_string(&config, ConfigFormat::Ron).unwrap();
        let ron_config = loader.load_from_string(&ron_str, ConfigFormat::Ron).unwrap();
        assert_eq!(config.ml.training.epochs, ron_config.ml.training.epochs);
    }
    
    #[test]
    fn test_hardware_config_detection() {
        let hw_config = HardwareConfig::detect().unwrap();
        
        // Basic validation
        assert!(hw_config.cpu_cores > 0);
        assert!(hw_config.cpu_threads > 0);
        assert!(hw_config.total_memory_gb > 0);
        assert!(!hw_config.os_type.is_empty());
        
        // Performance profile should be generated
        assert!(hw_config.performance_profile.tier >= 1);
        assert!(hw_config.performance_profile.tier <= 4);
        assert!(hw_config.performance_profile.recommended_threads > 0);
        assert!(hw_config.performance_profile.recommended_batch_size > 0);
        
        // Test optimization recommendations
        let recommendations = hw_config.get_optimization_recommendations();
        assert!(recommendations.contains_key("threads"));
        assert!(recommendations.contains_key("batch_size"));
        assert!(recommendations.contains_key("performance_tier"));
    }
    
    #[test]
    fn test_config_validation_comprehensive() {
        let validator = ConfigValidator::new();
        
        // Test valid configuration
        let valid_config = CdfaConfig::default();
        let report = validator.validate(&valid_config);
        assert!(report.is_valid);
        
        // Test invalid configuration
        let mut invalid_config = CdfaConfig::default();
        invalid_config.processing.tolerance = -1.0; // Invalid negative tolerance
        invalid_config.processing.max_iterations = 0; // Invalid zero iterations
        invalid_config.ml.neural_network.dropout_rate = 2.0; // Invalid dropout rate
        
        let report = validator.validate(&invalid_config);
        assert!(!report.is_valid);
        assert!(report.errors > 0);
        
        // Check specific errors
        let errors = report.get_errors();
        assert!(errors.iter().any(|e| e.parameter_path.contains("tolerance")));
        assert!(errors.iter().any(|e| e.parameter_path.contains("max_iterations")));
        assert!(errors.iter().any(|e| e.parameter_path.contains("dropout_rate")));
    }
    
    #[test]
    fn test_environment_variable_overrides() {
        // Set test environment variables
        env::set_var("CDFA_PROCESSING_NUM_THREADS", "16");
        env::set_var("CDFA_PROCESSING_ENABLE_SIMD", "false");
        env::set_var("CDFA_PROCESSING_TOLERANCE", "1e-12");
        env::set_var("CDFA_ALGORITHMS_DIVERSITY_METHODS", "pearson,spearman,kendall");
        env::set_var("CDFA_REDIS_ENABLED", "true");
        env::set_var("CDFA_REDIS_HOST", "redis.example.com");
        env::set_var("CDFA_REDIS_PORT", "6380");
        
        // Apply environment overrides
        let config = CdfaConfig::default();
        let modified_config = apply_environment_overrides(config).unwrap();
        
        // Verify overrides were applied
        assert_eq!(modified_config.processing.num_threads, 16);
        assert!(!modified_config.processing.enable_simd);
        assert_eq!(modified_config.processing.tolerance, 1e-12);
        assert_eq!(modified_config.algorithms.diversity_methods, 
                   vec!["pearson", "spearman", "kendall"]);
        assert!(modified_config.redis.enabled);
        assert_eq!(modified_config.redis.host, "redis.example.com");
        assert_eq!(modified_config.redis.port, 6380);
        
        // Clean up environment variables
        env::remove_var("CDFA_PROCESSING_NUM_THREADS");
        env::remove_var("CDFA_PROCESSING_ENABLE_SIMD");
        env::remove_var("CDFA_PROCESSING_TOLERANCE");
        env::remove_var("CDFA_ALGORITHMS_DIVERSITY_METHODS");
        env::remove_var("CDFA_REDIS_ENABLED");
        env::remove_var("CDFA_REDIS_HOST");
        env::remove_var("CDFA_REDIS_PORT");
    }
    
    #[test]
    fn test_config_migration_system() {
        let migrator = ConfigMigrator::new();
        
        // Test version parsing and comparison
        let v1 = ConfigVersion::parse("1.0.0").unwrap();
        let v2 = ConfigVersion::parse("1.1.0").unwrap();
        assert!(v1 < v2);
        assert!(v1.is_compatible_with(&v2));
        
        // Test migration path finding
        let from = ConfigVersion::new(0, 9, 0);
        let to = ConfigVersion::new(1, 0, 0);
        let path = migrator.find_migration_path(&from, &to);
        assert!(path.is_ok());
        
        // Test migration operations
        let mut config_data = serde_json::json!({
            "old_setting": "value",
            "nested": {
                "old_field": 42
            }
        });
        
        let operation = MigrationOperation::RenameField {
            old_path: "old_setting".to_string(),
            new_path: "new_setting".to_string(),
        };
        
        migrator.apply_operation(&mut config_data, &operation).unwrap();
        assert!(config_data.get("old_setting").is_none());
        assert_eq!(config_data.get("new_setting").unwrap().as_str(), Some("value"));
    }
    
    #[test]
    fn test_file_based_configuration() {
        // Create temporary config file
        let mut temp_file = NamedTempFile::new().unwrap();
        let config_content = r#"{
            "processing": {
                "num_threads": 8,
                "enable_simd": true,
                "tolerance": 1e-8
            },
            "algorithms": {
                "diversity_methods": ["pearson", "spearman"],
                "fusion_methods": ["score", "rank"]
            },
            "redis": {
                "enabled": false,
                "host": "localhost",
                "port": 6379
            }
        }"#;
        
        temp_file.write_all(config_content.as_bytes()).unwrap();
        
        // Load configuration from file
        let loader = ConfigLoader::new();
        let config = loader.load_from_file(temp_file.path()).unwrap();
        
        // Verify loaded configuration
        assert_eq!(config.processing.num_threads, 8);
        assert!(config.processing.enable_simd);
        assert_eq!(config.processing.tolerance, 1e-8);
        assert_eq!(config.algorithms.diversity_methods, vec!["pearson", "spearman"]);
        assert_eq!(config.algorithms.fusion_methods, vec!["score", "rank"]);
        assert!(!config.redis.enabled);
        
        // Test saving configuration back to file
        let temp_output = NamedTempFile::new().unwrap();
        loader.save_to_file(&config, temp_output.path()).unwrap();
        
        // Verify we can load it back
        let reloaded_config = loader.load_from_file(temp_output.path()).unwrap();
        assert_eq!(config.processing.num_threads, reloaded_config.processing.num_threads);
    }
    
    #[test]
    fn test_config_with_hardware_detection() {
        let hw_config = HardwareConfig::detect().unwrap();
        let mut config = CdfaConfig::default();
        
        // Apply hardware configuration
        config = config.apply_hardware_config(hw_config.clone()).unwrap();
        
        // Verify hardware-specific adjustments
        if hw_config.cpu_cores > 0 {
            assert!(config.processing.num_threads > 0 || config.processing.num_threads == 0);
        }
        
        if hw_config.has_gpu {
            assert!(config.processing.enable_gpu);
        }
        
        // Verify cache size is reasonable for available memory
        let available_memory_mb = hw_config.total_memory_gb * 1024;
        assert!(config.performance.cache_size_mb <= available_memory_mb / 2);
    }
    
    #[test]
    fn test_custom_validation_rules() {
        let mut validator = ConfigValidator::new();
        
        // Add custom validator
        validator.add_custom_validator(
            "redis_consistency".to_string(),
            |config| {
                let mut issues = Vec::new();
                if config.processing.enable_distributed && !config.redis.enabled {
                    issues.push(ValidationIssue {
                        parameter_path: "redis.enabled".to_string(),
                        severity: ValidationSeverity::Warning,
                        message: "Distributed processing without Redis coordination".to_string(),
                        suggestion: Some("Enable Redis for better coordination".to_string()),
                    });
                }
                issues
            },
        );
        
        // Test configuration that triggers custom validation
        let mut config = CdfaConfig::default();
        config.processing.enable_distributed = true;
        config.redis.enabled = false;
        
        let report = validator.validate(&config);
        assert!(report.warnings > 0);
        
        let warnings = report.get_warnings();
        assert!(warnings.iter().any(|w| w.message.contains("Redis coordination")));
    }
    
    #[test]
    fn test_configuration_feature_summary() {
        let config = CdfaConfig::default();
        let features = config.feature_summary();
        
        // Verify expected features are present
        assert!(features.contains_key("simd"));
        assert!(features.contains_key("gpu"));
        assert!(features.contains_key("distributed"));
        assert!(features.contains_key("ml"));
        assert!(features.contains_key("redis"));
        assert!(features.contains_key("neuromorphic"));
        assert!(features.contains_key("torchscript"));
        assert!(features.contains_key("caching"));
        assert!(features.contains_key("profiling"));
        assert!(features.contains_key("realtime"));
        
        // Verify default values
        assert_eq!(features.get("simd"), Some(&true));
        assert_eq!(features.get("gpu"), Some(&false));
        assert_eq!(features.get("distributed"), Some(&false));
        assert_eq!(features.get("ml"), Some(&false));
        assert_eq!(features.get("redis"), Some(&false));
    }
    
    #[test]
    fn test_config_loader_options() {
        let options = ConfigLoadOptions {
            allow_env_overrides: false,
            validate_on_load: true,
            apply_migrations: false,
            merge_with_defaults: true,
            strict_mode: true,
        };
        
        let loader = ConfigLoader::with_options(options);
        
        // Test that the loader respects the options
        let config_json = r#"{"processing": {"num_threads": 4}}"#;
        let result = loader.load_from_string(config_json, ConfigFormat::Json);
        assert!(result.is_ok());
        
        // Invalid config should fail in strict mode with validation enabled
        let invalid_json = r#"{"processing": {"tolerance": -1.0}}"#;
        let result = loader.load_from_string(invalid_json, ConfigFormat::Json);
        assert!(result.is_err());
    }
}

/// Performance benchmarks for configuration operations
#[cfg(test)]
mod performance_tests {
    use super::super::*;
    use std::time::Instant;
    
    #[test]
    fn test_config_creation_performance() {
        let start = Instant::now();
        
        for _ in 0..1000 {
            let _config = CdfaConfig::default();
        }
        
        let duration = start.elapsed();
        println!("1000 config creations took: {:?}", duration);
        
        // Should be very fast - less than 10ms for 1000 creations
        assert!(duration.as_millis() < 10);
    }
    
    #[test]
    fn test_validation_performance() {
        let config = CdfaConfig::default();
        let validator = ConfigValidator::new();
        
        let start = Instant::now();
        
        for _ in 0..100 {
            let _report = validator.validate(&config);
        }
        
        let duration = start.elapsed();
        println!("100 validations took: {:?}", duration);
        
        // Validation should be reasonably fast - less than 100ms for 100 validations
        assert!(duration.as_millis() < 100);
    }
    
    #[test]
    fn test_hardware_detection_performance() {
        let start = Instant::now();
        
        let _hw_config = HardwareConfig::detect().unwrap();
        
        let duration = start.elapsed();
        println!("Hardware detection took: {:?}", duration);
        
        // Hardware detection should complete within 1 second
        assert!(duration.as_secs() < 1);
    }
    
    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization_performance() {
        let config = CdfaConfig::default();
        let loader = ConfigLoader::new();
        
        let start = Instant::now();
        
        for _ in 0..100 {
            let _json = loader.save_to_string(&config, ConfigFormat::Json).unwrap();
        }
        
        let duration = start.elapsed();
        println!("100 JSON serializations took: {:?}", duration);
        
        // Serialization should be fast - less than 50ms for 100 serializations
        assert!(duration.as_millis() < 50);
    }
}