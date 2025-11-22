//! Comprehensive security module for Autopoiesis trading system
//! 
//! This module implements enterprise-grade security measures including:
//! - Zero-trust authentication
//! - Cryptographically secure random number generation
//! - Advanced rate limiting with DDoS protection
//! - Intrusion detection system
//! - GDPR/SOX compliant audit logging
//! - Real-time threat monitoring

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

pub mod auth;
pub mod crypto;
pub mod audit;
pub mod intrusion;
pub mod compliance;
pub mod monitoring;

/// Security configuration for the entire system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub auth_config: AuthConfig,
    
    /// Cryptographic configuration
    pub crypto_config: CryptoConfig,
    
    /// Rate limiting configuration
    pub rate_limit_config: RateLimitConfig,
    
    /// Audit logging configuration
    pub audit_config: AuditConfig,
    
    /// Intrusion detection configuration
    pub intrusion_config: IntrusionConfig,
    
    /// Compliance configuration
    pub compliance_config: ComplianceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// JWT secret (from environment)
    pub jwt_secret_env_var: String,
    
    /// JWT expiration time in seconds
    pub jwt_expiration_seconds: u64,
    
    /// Require multi-factor authentication
    pub require_mfa: bool,
    
    /// Maximum failed login attempts
    pub max_failed_attempts: u32,
    
    /// Account lockout duration in seconds
    pub lockout_duration_seconds: u64,
    
    /// Device fingerprinting enabled
    pub device_fingerprinting: bool,
    
    /// Session timeout in seconds
    pub session_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Use cryptographically secure RNG
    pub secure_rng_only: bool,
    
    /// Minimum key length for encryption
    pub min_key_length_bits: u32,
    
    /// Encryption algorithm preference
    pub encryption_algorithm: String,
    
    /// Hash algorithm for passwords
    pub password_hash_algorithm: String,
    
    /// Key rotation interval in days
    pub key_rotation_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable DDoS protection
    pub ddos_protection: bool,
    
    /// Free tier limits (requests per hour)
    pub free_tier_rph: u32,
    
    /// Premium tier limits (requests per hour)
    pub premium_tier_rph: u32,
    
    /// Enterprise tier limits (requests per hour)
    pub enterprise_tier_rph: u32,
    
    /// Burst limit multiplier
    pub burst_multiplier: f64,
    
    /// IP reputation checking
    pub ip_reputation_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// SOX compliance mode
    pub sox_compliance: bool,
    
    /// GDPR compliance mode
    pub gdpr_compliance: bool,
    
    /// Log all security events
    pub log_all_events: bool,
    
    /// Audit log retention days
    pub retention_days: u32,
    
    /// Real-time alerting
    pub real_time_alerts: bool,
    
    /// Encrypted audit logs
    pub encrypted_logs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrusionConfig {
    /// Enable intrusion detection
    pub enabled: bool,
    
    /// Threat detection sensitivity (0.0-1.0)
    pub sensitivity: f64,
    
    /// Automatic response enabled
    pub auto_response: bool,
    
    /// Behavioral analysis enabled
    pub behavioral_analysis: bool,
    
    /// Machine learning threat detection
    pub ml_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Data classification levels
    pub data_classification: Vec<String>,
    
    /// Data retention policies
    pub retention_policies: HashMap<String, u32>,
    
    /// Encryption requirements
    pub encryption_required: bool,
    
    /// Right to erasure implementation
    pub right_to_erasure: bool,
    
    /// Consent management
    pub consent_management: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth_config: AuthConfig {
                jwt_secret_env_var: "JWT_SECRET".to_string(),
                jwt_expiration_seconds: 3600, // 1 hour
                require_mfa: true,
                max_failed_attempts: 5,
                lockout_duration_seconds: 300, // 5 minutes
                device_fingerprinting: true,
                session_timeout_seconds: 14400, // 4 hours
            },
            crypto_config: CryptoConfig {
                secure_rng_only: true,
                min_key_length_bits: 256,
                encryption_algorithm: "AES-256-GCM".to_string(),
                password_hash_algorithm: "Argon2id".to_string(),
                key_rotation_days: 90,
            },
            rate_limit_config: RateLimitConfig {
                ddos_protection: true,
                free_tier_rph: 100,
                premium_tier_rph: 10000,
                enterprise_tier_rph: 1000000,
                burst_multiplier: 2.0,
                ip_reputation_check: true,
            },
            audit_config: AuditConfig {
                sox_compliance: true,
                gdpr_compliance: true,
                log_all_events: true,
                retention_days: 2555, // 7 years for financial compliance
                real_time_alerts: true,
                encrypted_logs: true,
            },
            intrusion_config: IntrusionConfig {
                enabled: true,
                sensitivity: 0.8,
                auto_response: true,
                behavioral_analysis: true,
                ml_detection: true,
            },
            compliance_config: ComplianceConfig {
                data_classification: vec![
                    "public".to_string(),
                    "internal".to_string(),
                    "confidential".to_string(),
                    "restricted".to_string(),
                    "financial".to_string(),
                    "pii".to_string(),
                ],
                retention_policies: {
                    let mut policies = HashMap::new();
                    policies.insert("financial_data".to_string(), 2555); // 7 years
                    policies.insert("user_data".to_string(), 1095); // 3 years
                    policies.insert("log_data".to_string(), 365); // 1 year
                    policies.insert("backup_data".to_string(), 90); // 90 days
                    policies
                },
                encryption_required: true,
                right_to_erasure: true,
                consent_management: true,
            },
        }
    }
}

impl SecurityConfig {
    /// Load security configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Validate JWT secret
        let jwt_secret = std::env::var(&config.auth_config.jwt_secret_env_var)
            .map_err(|_| anyhow!("JWT_SECRET environment variable is required"))?;
            
        if jwt_secret.len() < 32 {
            return Err(anyhow!("JWT_SECRET must be at least 32 characters for security"));
        }
        
        // Check for insecure patterns in JWT secret
        let insecure_patterns = [
            "secret", "password", "key", "test", "dev", "demo", 
            "example", "default", "changeme", "admin", "root", 
            "user", "123456", "qwerty"
        ];
        
        let secret_lower = jwt_secret.to_lowercase();
        for pattern in &insecure_patterns {
            if secret_lower.contains(pattern) {
                return Err(anyhow!("JWT_SECRET contains insecure pattern: {}", pattern));
            }
        }
        
        // Override defaults with environment variables if present
        if let Ok(val) = std::env::var("AUTH_REQUIRE_MFA") {
            config.auth_config.require_mfa = val.parse().unwrap_or(true);
        }
        
        if let Ok(val) = std::env::var("SECURITY_SENSITIVITY") {
            config.intrusion_config.sensitivity = val.parse().unwrap_or(0.8);
        }
        
        if let Ok(val) = std::env::var("SOX_COMPLIANCE") {
            config.audit_config.sox_compliance = val.parse().unwrap_or(true);
        }
        
        if let Ok(val) = std::env::var("GDPR_COMPLIANCE") {
            config.audit_config.gdpr_compliance = val.parse().unwrap_or(true);
        }
        
        Ok(config)
    }
    
    /// Validate the security configuration
    pub fn validate(&self) -> Result<()> {
        // Validate JWT configuration
        if self.auth_config.jwt_expiration_seconds < 300 {
            return Err(anyhow!("JWT expiration must be at least 5 minutes"));
        }
        
        if self.auth_config.jwt_expiration_seconds > 86400 {
            return Err(anyhow!("JWT expiration should not exceed 24 hours for security"));
        }
        
        // Validate crypto configuration
        if self.crypto_config.min_key_length_bits < 256 {
            return Err(anyhow!("Minimum key length must be at least 256 bits"));
        }
        
        // Validate rate limiting
        if self.rate_limit_config.free_tier_rph > self.rate_limit_config.premium_tier_rph {
            return Err(anyhow!("Free tier limits cannot exceed premium tier limits"));
        }
        
        // Validate audit configuration
        if self.audit_config.retention_days < 365 && self.audit_config.sox_compliance {
            return Err(anyhow!("SOX compliance requires minimum 1 year audit retention"));
        }
        
        // Validate intrusion detection
        if self.intrusion_config.sensitivity < 0.0 || self.intrusion_config.sensitivity > 1.0 {
            return Err(anyhow!("Intrusion detection sensitivity must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
    
    /// Get security level based on configuration
    pub fn get_security_level(&self) -> SecurityLevel {
        let mut score = 0u8;
        
        // Authentication factors
        if self.auth_config.require_mfa { score += 20; }
        if self.auth_config.device_fingerprinting { score += 15; }
        if self.auth_config.jwt_expiration_seconds <= 3600 { score += 10; }
        
        // Cryptographic factors
        if self.crypto_config.secure_rng_only { score += 15; }
        if self.crypto_config.min_key_length_bits >= 256 { score += 10; }
        
        // Monitoring factors
        if self.intrusion_config.enabled { score += 15; }
        if self.audit_config.real_time_alerts { score += 10; }
        
        // Compliance factors
        if self.audit_config.sox_compliance { score += 10; }
        if self.audit_config.gdpr_compliance { score += 5; }
        
        match score {
            90..=100 => SecurityLevel::Maximum,
            80..=89 => SecurityLevel::High,
            70..=79 => SecurityLevel::Medium,
            60..=69 => SecurityLevel::Low,
            _ => SecurityLevel::Minimal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityLevel {
    Minimal,
    Low,
    Medium,
    High,
    Maximum,
}

impl SecurityLevel {
    pub fn as_score(&self) -> u8 {
        match self {
            SecurityLevel::Minimal => 1,
            SecurityLevel::Low => 2,
            SecurityLevel::Medium => 3,
            SecurityLevel::High => 4,
            SecurityLevel::Maximum => 5,
        }
    }
}

/// Security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: SecurityEventType,
    pub user_id: Option<String>,
    pub ip_address: Option<IpAddr>,
    pub resource: String,
    pub action: String,
    pub result: ActionResult,
    pub risk_score: u8,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemConfiguration,
    SecurityViolation,
    SuspiciousActivity,
    ThreatDetected,
    ComplianceViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionResult {
    Success,
    Failed,
    Blocked,
    Suspicious,
}

/// Central security manager
pub struct SecurityManager {
    config: SecurityConfig,
    rng: SystemRandom,
    events: Arc<RwLock<Vec<SecurityEvent>>>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        config.validate()?;
        
        Ok(Self {
            config,
            rng: SystemRandom::new(),
            events: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    pub async fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        // Log to structured logging
        match event.event_type {
            SecurityEventType::ThreatDetected | SecurityEventType::SecurityViolation => {
                error!(
                    target: "security",
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("unknown"),
                    resource = %event.resource,
                    risk_score = event.risk_score,
                    "High-risk security event detected"
                );
            },
            SecurityEventType::SuspiciousActivity => {
                warn!(
                    target: "security",
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("unknown"),
                    resource = %event.resource,
                    "Suspicious activity detected"
                );
            },
            _ => {
                info!(
                    target: "security",
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("unknown"),
                    resource = %event.resource,
                    result = ?event.result,
                    "Security event logged"
                );
            }
        }
        
        // Store in memory for analysis
        self.events.write().await.push(event);
        
        Ok(())
    }
    
    pub fn get_config(&self) -> &SecurityConfig {
        &self.config
    }
    
    pub fn get_rng(&self) -> &SystemRandom {
        &self.rng
    }
    
    pub async fn get_recent_events(&self, limit: usize) -> Vec<SecurityEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(limit).cloned().collect()
    }
    
    /// Generate cryptographically secure random bytes
    pub fn generate_secure_random(&self, length: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; length];
        self.rng.fill(&mut bytes)
            .map_err(|_| anyhow!("Failed to generate secure random bytes"))?;
        Ok(bytes)
    }
    
    /// Generate cryptographically secure random f64 in range [0.0, 1.0)
    pub fn generate_secure_random_f64(&self) -> Result<f64> {
        let mut bytes = [0u8; 8];
        self.rng.fill(&mut bytes)
            .map_err(|_| anyhow!("Failed to generate secure random number"))?;
        
        // Convert to f64 in range [0.0, 1.0)
        let value = u64::from_le_bytes(bytes);
        Ok((value as f64) / (u64::MAX as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_config_validation() {
        let config = SecurityConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_security_level_calculation() {
        let config = SecurityConfig::default();
        let level = config.get_security_level();
        assert!(matches!(level, SecurityLevel::High | SecurityLevel::Maximum));
    }
    
    #[tokio::test]
    async fn test_security_manager_creation() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config);
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_secure_random_generation() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config).unwrap();
        
        let random_bytes = manager.generate_secure_random(32);
        assert!(random_bytes.is_ok());
        assert_eq!(random_bytes.unwrap().len(), 32);
        
        let random_f64 = manager.generate_secure_random_f64();
        assert!(random_f64.is_ok());
        let value = random_f64.unwrap();
        assert!(value >= 0.0 && value < 1.0);
    }
}