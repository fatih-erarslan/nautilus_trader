//! Production Security Configuration Manager
//!
//! Secure configuration management system for production environments.
//! Eliminates hardcoded secrets and implements best practices for financial system security.

use crate::api::security::{SecurityConfig, JwtConfig, RateLimitingConfig, ValidationConfig, EncryptionConfig};
use crate::{AtsCoreError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Production Security Configuration Manager
#[derive(Debug, Clone)]
pub struct ProductionSecurityManager {
    config: SecurityConfig,
    secrets_backend: SecretsBackend,
    config_validation: ConfigValidation,
}

/// Secrets Backend Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretsBackend {
    /// Environment variables (for containers/simple deployments)
    Environment,
    /// HashiCorp Vault integration
    Vault {
        vault_url: String,
        vault_token_path: String,
        mount_path: String,
    },
    /// AWS Secrets Manager
    AwsSecretsManager {
        region: String,
        secret_prefix: String,
    },
    /// Azure Key Vault
    AzureKeyVault {
        vault_url: String,
        tenant_id: String,
    },
    /// Google Secret Manager
    GoogleSecretManager {
        project_id: String,
        secret_prefix: String,
    },
}

/// Configuration Validation Rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidation {
    /// Require strong JWT secrets (min length)
    pub min_jwt_secret_length: usize,
    /// Require specific algorithms
    pub allowed_jwt_algorithms: Vec<String>,
    /// Require minimum token expiry
    pub min_jwt_expiry_seconds: u64,
    /// Maximum token expiry
    pub max_jwt_expiry_seconds: u64,
    /// Required environment for production
    pub required_environment: String,
    /// Enforce encryption settings
    pub require_encryption_at_rest: bool,
    pub require_encryption_in_transit: bool,
}

/// Secure Configuration Template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureConfigTemplate {
    /// Template name
    pub name: String,
    /// Environment type (dev, staging, production)
    pub environment: String,
    /// Security level (standard, high, critical)
    pub security_level: SecurityLevel,
    /// Required secrets
    pub required_secrets: Vec<SecretDefinition>,
    /// Configuration overrides
    pub config_overrides: HashMap<String, serde_json::Value>,
}

/// Security Level Classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Standard security (development)
    Standard,
    /// High security (staging/testing)
    High,
    /// Critical security (production financial systems)
    Critical,
}

/// Secret Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretDefinition {
    /// Secret key name
    pub key: String,
    /// Environment variable name
    pub env_var: String,
    /// Whether secret is required
    pub required: bool,
    /// Minimum length requirement
    pub min_length: Option<usize>,
    /// Pattern validation (regex)
    pub pattern: Option<String>,
    /// Description for documentation
    pub description: String,
}

impl ProductionSecurityManager {
    /// Create new production security manager
    pub fn new() -> Result<Self> {
        let secrets_backend = Self::detect_secrets_backend()?;
        let config = Self::load_secure_config(&secrets_backend)?;
        let config_validation = Self::get_validation_rules();
        
        let manager = Self {
            config,
            secrets_backend,
            config_validation,
        };
        
        // Validate configuration before returning
        manager.validate_production_config()?;
        
        Ok(manager)
    }

    /// Create manager from template
    pub fn from_template(template: &SecureConfigTemplate) -> Result<Self> {
        let secrets_backend = Self::detect_secrets_backend()?;
        let config = Self::load_config_from_template(template, &secrets_backend)?;
        let config_validation = Self::get_validation_rules_for_level(&template.security_level);
        
        let manager = Self {
            config,
            secrets_backend,
            config_validation,
        };
        
        manager.validate_template_config(template)?;
        
        Ok(manager)
    }

    /// Get security configuration
    pub fn get_config(&self) -> &SecurityConfig {
        &self.config
    }

    /// Validate production configuration
    pub fn validate_production_config(&self) -> Result<()> {
        // Validate JWT configuration
        self.validate_jwt_config()?;
        
        // Validate encryption settings
        self.validate_encryption_config()?;
        
        // Validate rate limiting
        self.validate_rate_limiting_config()?;
        
        // Check for development artifacts
        self.check_development_artifacts()?;
        
        Ok(())
    }

    /// Load secure configuration
    fn load_secure_config(backend: &SecretsBackend) -> Result<SecurityConfig> {
        let jwt_config = Self::load_jwt_config(backend)?;
        let rate_limiting_config = Self::load_rate_limiting_config(backend)?;
        let validation_config = Self::load_validation_config(backend)?;
        let encryption_config = Self::load_encryption_config(backend)?;
        
        Ok(SecurityConfig {
            jwt: jwt_config,
            rate_limiting: rate_limiting_config,
            validation: validation_config,
            encryption: encryption_config,
        })
    }

    /// Load JWT configuration from secure backend
    fn load_jwt_config(backend: &SecretsBackend) -> Result<JwtConfig> {
        let secret = Self::get_secret(backend, "jwt.secret", "ATS_JWT_SECRET")?
            .ok_or_else(|| AtsCoreError::validation("jwt.secret", "JWT secret must be configured"))?;
        
        let expiry_duration = Duration::from_secs(
            Self::get_secret(backend, "jwt.expiry_seconds", "ATS_JWT_EXPIRY_SECONDS")?
                .unwrap_or_else(|| "3600".to_string()) // Default 1 hour for production
                .parse()
                .map_err(|_| AtsCoreError::validation("jwt.expiry_seconds", "Invalid expiry duration"))?
        );
        
        let issuer = Self::get_secret(backend, "jwt.issuer", "ATS_JWT_ISSUER")?
            .unwrap_or_else(|| "ats-core-production".to_string());
        
        let audiences = Self::get_secret(backend, "jwt.audiences", "ATS_JWT_AUDIENCES")?
            .unwrap_or_else(|| "ats-core-production-api".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        
        let algorithm = Self::get_secret(backend, "jwt.algorithm", "ATS_JWT_ALGORITHM")?
            .unwrap_or_else(|| "HS256".to_string());
        
        Ok(JwtConfig {
            secret,
            expiry_duration,
            issuer,
            audiences,
            algorithm,
        })
    }

    /// Load rate limiting configuration
    fn load_rate_limiting_config(backend: &SecretsBackend) -> Result<RateLimitingConfig> {
        Ok(RateLimitingConfig {
            requests_per_minute: Self::get_secret(backend, "rate_limit.requests_per_minute", "ATS_RATE_LIMIT_RPM")?
                .unwrap_or_else(|| "100".to_string())
                .parse()
                .unwrap_or(100),
            burst_size: Self::get_secret(backend, "rate_limit.burst_size", "ATS_RATE_LIMIT_BURST")?
                .unwrap_or_else(|| "20".to_string())
                .parse()
                .unwrap_or(20),
            window_duration: Duration::from_secs(60),
            by_ip: true,
            by_user: true,
            by_api_key: true,
        })
    }

    /// Load validation configuration
    fn load_validation_config(backend: &SecretsBackend) -> Result<ValidationConfig> {
        Ok(ValidationConfig {
            max_body_size: Self::get_secret(backend, "validation.max_body_size", "ATS_MAX_BODY_SIZE")?
                .unwrap_or_else(|| "1048576".to_string()) // 1MB default for production
                .parse()
                .unwrap_or(1048576),
            max_array_length: Self::get_secret(backend, "validation.max_array_length", "ATS_MAX_ARRAY_LENGTH")?
                .unwrap_or_else(|| "1000".to_string())
                .parse()
                .unwrap_or(1000),
            max_string_length: Self::get_secret(backend, "validation.max_string_length", "ATS_MAX_STRING_LENGTH")?
                .unwrap_or_else(|| "512".to_string())
                .parse()
                .unwrap_or(512),
            max_nesting_depth: Self::get_secret(backend, "validation.max_nesting_depth", "ATS_MAX_NESTING_DEPTH")?
                .unwrap_or_else(|| "5".to_string())
                .parse()
                .unwrap_or(5),
            confidence_level_range: (0.01, 0.99),
            max_features: Self::get_secret(backend, "validation.max_features", "ATS_MAX_FEATURES")?
                .unwrap_or_else(|| "1000".to_string())
                .parse()
                .unwrap_or(1000),
        })
    }

    /// Load encryption configuration
    fn load_encryption_config(backend: &SecretsBackend) -> Result<EncryptionConfig> {
        Ok(EncryptionConfig {
            encrypt_at_rest: Self::get_secret(backend, "encryption.at_rest", "ATS_ENCRYPT_AT_REST")?
                .unwrap_or_else(|| "true".to_string())
                .parse()
                .unwrap_or(true),
            encrypt_in_transit: Self::get_secret(backend, "encryption.in_transit", "ATS_ENCRYPT_IN_TRANSIT")?
                .unwrap_or_else(|| "true".to_string())
                .parse()
                .unwrap_or(true),
            algorithm: Self::get_secret(backend, "encryption.algorithm", "ATS_ENCRYPTION_ALGORITHM")?
                .unwrap_or_else(|| "AES-256-GCM".to_string()),
            key_rotation_interval: Duration::from_secs(
                Self::get_secret(backend, "encryption.key_rotation_seconds", "ATS_KEY_ROTATION_SECONDS")?
                    .unwrap_or_else(|| "604800".to_string()) // 7 days default
                    .parse()
                    .unwrap_or(604800)
            ),
        })
    }

    /// Get secret from backend
    fn get_secret(backend: &SecretsBackend, key: &str, env_var: &str) -> Result<Option<String>> {
        match backend {
            SecretsBackend::Environment => {
                Ok(std::env::var(env_var).ok())
            }
            SecretsBackend::Vault { .. } => {
                // Implement Vault integration
                Self::get_vault_secret(key)
            }
            SecretsBackend::AwsSecretsManager { .. } => {
                // Implement AWS Secrets Manager integration
                Self::get_aws_secret(key)
            }
            SecretsBackend::AzureKeyVault { .. } => {
                // Implement Azure Key Vault integration
                Self::get_azure_secret(key)
            }
            SecretsBackend::GoogleSecretManager { .. } => {
                // Implement Google Secret Manager integration
                Self::get_google_secret(key)
            }
        }
    }

    /// Detect secrets backend
    fn detect_secrets_backend() -> Result<SecretsBackend> {
        // Check for cloud provider secrets managers first
        if std::env::var("VAULT_ADDR").is_ok() {
            Ok(SecretsBackend::Vault {
                vault_url: std::env::var("VAULT_ADDR").unwrap(),
                vault_token_path: std::env::var("VAULT_TOKEN_PATH").unwrap_or_else(|_| "/var/secrets/vault-token".to_string()),
                mount_path: std::env::var("VAULT_MOUNT_PATH").unwrap_or_else(|_| "secret".to_string()),
            })
        } else if std::env::var("AWS_REGION").is_ok() {
            Ok(SecretsBackend::AwsSecretsManager {
                region: std::env::var("AWS_REGION").unwrap(),
                secret_prefix: std::env::var("AWS_SECRET_PREFIX").unwrap_or_else(|_| "ats-core".to_string()),
            })
        } else if std::env::var("AZURE_KEY_VAULT_URL").is_ok() {
            Ok(SecretsBackend::AzureKeyVault {
                vault_url: std::env::var("AZURE_KEY_VAULT_URL").unwrap(),
                tenant_id: std::env::var("AZURE_TENANT_ID").unwrap(),
            })
        } else if std::env::var("GOOGLE_CLOUD_PROJECT").is_ok() {
            Ok(SecretsBackend::GoogleSecretManager {
                project_id: std::env::var("GOOGLE_CLOUD_PROJECT").unwrap(),
                secret_prefix: std::env::var("GCP_SECRET_PREFIX").unwrap_or_else(|_| "ats-core".to_string()),
            })
        } else {
            // Default to environment variables
            Ok(SecretsBackend::Environment)
        }
    }

    /// Validate JWT configuration
    fn validate_jwt_config(&self) -> Result<()> {
        let jwt = &self.config.jwt;
        
        // Check secret length
        if jwt.secret.len() < self.config_validation.min_jwt_secret_length {
            return Err(AtsCoreError::validation(
                "jwt.secret",
                &format!("JWT secret must be at least {} characters", self.config_validation.min_jwt_secret_length)
            ));
        }
        
        // Check algorithm
        if !self.config_validation.allowed_jwt_algorithms.contains(&jwt.algorithm) {
            return Err(AtsCoreError::validation(
                "jwt.algorithm",
                &format!("JWT algorithm must be one of: {:?}", self.config_validation.allowed_jwt_algorithms)
            ));
        }
        
        // Check expiry duration
        let expiry_seconds = jwt.expiry_duration.as_secs();
        if expiry_seconds < self.config_validation.min_jwt_expiry_seconds {
            return Err(AtsCoreError::validation(
                "jwt.expiry_duration",
                &format!("JWT expiry must be at least {} seconds", self.config_validation.min_jwt_expiry_seconds)
            ));
        }
        if expiry_seconds > self.config_validation.max_jwt_expiry_seconds {
            return Err(AtsCoreError::validation(
                "jwt.expiry_duration",
                &format!("JWT expiry must not exceed {} seconds", self.config_validation.max_jwt_expiry_seconds)
            ));
        }
        
        // Check for default/weak secrets
        let weak_secrets = [
            "weak-secret-placeholder",
            "secret",
            "password",  
            "123456",
            "admin",
            "test",
            "changeme",
        ];
        if weak_secrets.contains(&jwt.secret.as_str()) {
            return Err(AtsCoreError::validation(
                "jwt.secret",
                "JWT secret contains a default or weak value that must be changed"
            ));
        }
        
        Ok(())
    }

    /// Validate encryption configuration
    fn validate_encryption_config(&self) -> Result<()> {
        let encryption = &self.config.encryption;
        
        if self.config_validation.require_encryption_at_rest && !encryption.encrypt_at_rest {
            return Err(AtsCoreError::validation(
                "encryption.encrypt_at_rest",
                "Encryption at rest is required for production"
            ));
        }
        
        if self.config_validation.require_encryption_in_transit && !encryption.encrypt_in_transit {
            return Err(AtsCoreError::validation(
                "encryption.encrypt_in_transit",
                "Encryption in transit is required for production"
            ));
        }
        
        Ok(())
    }

    /// Validate rate limiting configuration
    fn validate_rate_limiting_config(&self) -> Result<()> {
        let rate_limit = &self.config.rate_limiting;
        
        if rate_limit.requests_per_minute == 0 {
            return Err(AtsCoreError::validation(
                "rate_limiting.requests_per_minute",
                "Rate limiting must be enabled in production"
            ));
        }
        
        Ok(())
    }

    /// Check for development artifacts
    fn check_development_artifacts(&self) -> Result<()> {
        let environment = std::env::var("ATS_ENVIRONMENT").unwrap_or_else(|_| "development".to_string());
        
        if self.config_validation.required_environment == "production" && environment == "development" {
            return Err(AtsCoreError::validation(
                "environment",
                "Development environment detected in production configuration"
            ));
        }
        
        Ok(())
    }

    /// Get validation rules
    fn get_validation_rules() -> ConfigValidation {
        ConfigValidation {
            min_jwt_secret_length: 32,
            allowed_jwt_algorithms: vec!["HS256".to_string(), "RS256".to_string(), "ES256".to_string()],
            min_jwt_expiry_seconds: 300, // 5 minutes minimum
            max_jwt_expiry_seconds: 86400, // 24 hours maximum
            required_environment: std::env::var("ATS_REQUIRED_ENVIRONMENT").unwrap_or_else(|_| "production".to_string()),
            require_encryption_at_rest: true,
            require_encryption_in_transit: true,
        }
    }

    /// Get validation rules for security level
    fn get_validation_rules_for_level(level: &SecurityLevel) -> ConfigValidation {
        match level {
            SecurityLevel::Standard => ConfigValidation {
                min_jwt_secret_length: 16,
                allowed_jwt_algorithms: vec!["HS256".to_string(), "RS256".to_string()],
                min_jwt_expiry_seconds: 60,
                max_jwt_expiry_seconds: 604800, // 7 days
                required_environment: "development".to_string(),
                require_encryption_at_rest: false,
                require_encryption_in_transit: true,
            },
            SecurityLevel::High => ConfigValidation {
                min_jwt_secret_length: 24,
                allowed_jwt_algorithms: vec!["HS256".to_string(), "RS256".to_string(), "ES256".to_string()],
                min_jwt_expiry_seconds: 300,
                max_jwt_expiry_seconds: 172800, // 2 days
                required_environment: "staging".to_string(),
                require_encryption_at_rest: true,
                require_encryption_in_transit: true,
            },
            SecurityLevel::Critical => ConfigValidation {
                min_jwt_secret_length: 64,
                allowed_jwt_algorithms: vec!["RS256".to_string(), "ES256".to_string()],
                min_jwt_expiry_seconds: 300,
                max_jwt_expiry_seconds: 3600, // 1 hour
                required_environment: "production".to_string(),
                require_encryption_at_rest: true,
                require_encryption_in_transit: true,
            },
        }
    }

    /// Load configuration from template
    fn load_config_from_template(template: &SecureConfigTemplate, backend: &SecretsBackend) -> Result<SecurityConfig> {
        // Start with secure defaults
        let mut config = Self::load_secure_config(backend)?;
        
        // Apply template overrides
        for (key, value) in &template.config_overrides {
            Self::apply_config_override(&mut config, key, value)?;
        }
        
        Ok(config)
    }

    /// Apply configuration override
    fn apply_config_override(config: &mut SecurityConfig, key: &str, value: &serde_json::Value) -> Result<()> {
        match key {
            "jwt.expiry_duration" => {
                if let Some(seconds) = value.as_u64() {
                    config.jwt.expiry_duration = Duration::from_secs(seconds);
                }
            }
            "jwt.algorithm" => {
                if let Some(algorithm) = value.as_str() {
                    config.jwt.algorithm = algorithm.to_string();
                }
            }
            "rate_limiting.requests_per_minute" => {
                if let Some(rpm) = value.as_u64() {
                    config.rate_limiting.requests_per_minute = rpm as u32;
                }
            }
            _ => {
                // Log unknown override but don't fail
                eprintln!("Warning: Unknown configuration override: {}", key);
            }
        }
        
        Ok(())
    }

    /// Validate template configuration
    fn validate_template_config(&self, template: &SecureConfigTemplate) -> Result<()> {
        // Validate all required secrets are available
        for secret_def in &template.required_secrets {
            if secret_def.required {
                let value = Self::get_secret(&self.secrets_backend, &secret_def.key, &secret_def.env_var)?;
                if value.is_none() {
                    return Err(AtsCoreError::validation(
                        &secret_def.key,
                        &format!("Required secret '{}' is not configured", secret_def.key)
                    ));
                }
                
                if let Some(secret_value) = value {
                    // Validate length
                    if let Some(min_length) = secret_def.min_length {
                        if secret_value.len() < min_length {
                            return Err(AtsCoreError::validation(
                                &secret_def.key,
                                &format!("Secret '{}' must be at least {} characters", secret_def.key, min_length)
                            ));
                        }
                    }
                    
                    // Validate pattern
                    if let Some(pattern) = &secret_def.pattern {
                        let key_string = secret_def.key.to_string();
                        let regex = regex::Regex::new(pattern)
                            .map_err(|_| AtsCoreError::validation(&key_string, &"Invalid pattern in secret definition".to_string()))?;
                        if !regex.is_match(&secret_value) {
                            return Err(AtsCoreError::validation(
                                &secret_def.key,
                                &format!("Secret '{}' does not match required pattern", secret_def.key)
                            ));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    // Placeholder implementations for cloud secret managers
    fn get_vault_secret(_key: &str) -> Result<Option<String>> {
        // TODO: Implement HashiCorp Vault integration
        Ok(None)
    }

    fn get_aws_secret(_key: &str) -> Result<Option<String>> {
        // TODO: Implement AWS Secrets Manager integration
        Ok(None)
    }

    fn get_azure_secret(_key: &str) -> Result<Option<String>> {
        // TODO: Implement Azure Key Vault integration
        Ok(None)
    }

    fn get_google_secret(_key: &str) -> Result<Option<String>> {
        // TODO: Implement Google Secret Manager integration
        Ok(None)
    }

    /// Generate secure random secret
    pub fn generate_secure_secret(length: usize) -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                abcdefghijklmnopqrstuvwxyz\
                                0123456789\
                                !@#$%^&*()_+-=[]{}|;:,.<>?";
        let mut rng = rand::thread_rng();
        
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }
}

/// Security Configuration Templates
pub struct SecurityTemplates;

impl SecurityTemplates {
    /// Get production template for financial systems
    pub fn production_financial() -> SecureConfigTemplate {
        SecureConfigTemplate {
            name: "production-financial".to_string(),
            environment: "production".to_string(),
            security_level: SecurityLevel::Critical,
            required_secrets: vec![
                SecretDefinition {
                    key: "jwt.secret".to_string(),
                    env_var: "ATS_JWT_SECRET".to_string(),
                    required: true,
                    min_length: Some(64),
                    pattern: Some(r"^[A-Za-z0-9+/=]{64,}$".to_string()),
                    description: "JWT signing secret (base64 encoded, min 64 chars)".to_string(),
                },
                SecretDefinition {
                    key: "encryption.key".to_string(),
                    env_var: "ATS_ENCRYPTION_KEY".to_string(),
                    required: true,
                    min_length: Some(32),
                    pattern: Some(r"^[A-Fa-f0-9]{64}$".to_string()),
                    description: "Data encryption key (hex encoded, 256-bit)".to_string(),
                },
                SecretDefinition {
                    key: "database.password".to_string(),
                    env_var: "ATS_DB_PASSWORD".to_string(),
                    required: true,
                    min_length: Some(16),
                    pattern: None,
                    description: "Database connection password".to_string(),
                },
            ],
            config_overrides: [
                ("jwt.expiry_duration".to_string(), serde_json::Value::Number(3600.into())), // 1 hour
                ("jwt.algorithm".to_string(), serde_json::Value::String("RS256".to_string())),
                ("rate_limiting.requests_per_minute".to_string(), serde_json::Value::Number(60.into())),
            ].iter().cloned().collect(),
        }
    }

    /// Get staging template
    pub fn staging() -> SecureConfigTemplate {
        SecureConfigTemplate {
            name: "staging".to_string(),
            environment: "staging".to_string(),
            security_level: SecurityLevel::High,
            required_secrets: vec![
                SecretDefinition {
                    key: "jwt.secret".to_string(),
                    env_var: "ATS_JWT_SECRET".to_string(),
                    required: true,
                    min_length: Some(32),
                    pattern: None,
                    description: "JWT signing secret".to_string(),
                },
            ],
            config_overrides: [
                ("jwt.expiry_duration".to_string(), serde_json::Value::Number(7200.into())), // 2 hours
            ].iter().cloned().collect(),
        }
    }

    /// Get development template
    pub fn development() -> SecureConfigTemplate {
        SecureConfigTemplate {
            name: "development".to_string(),
            environment: "development".to_string(),
            security_level: SecurityLevel::Standard,
            required_secrets: vec![
                SecretDefinition {
                    key: "jwt.secret".to_string(),
                    env_var: "ATS_JWT_SECRET".to_string(),
                    required: false,
                    min_length: Some(16),
                    pattern: None,
                    description: "JWT signing secret (optional for dev)".to_string(),
                },
            ],
            config_overrides: [
                ("jwt.expiry_duration".to_string(), serde_json::Value::Number(86400.into())), // 24 hours
            ].iter().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_template_validation() {
        let template = SecurityTemplates::production_financial();
        assert_eq!(template.security_level, SecurityLevel::Critical);
        assert_eq!(template.required_secrets.len(), 3);
    }

    #[test]
    fn test_secret_generation() {
        let secret = ProductionSecurityManager::generate_secure_secret(64);
        assert_eq!(secret.len(), 64);
        assert!(secret.chars().all(|c| c.is_ascii()));
    }
}