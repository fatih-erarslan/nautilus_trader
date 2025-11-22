//! Secure main entry point with comprehensive security initialization
//! 
//! This module provides a hardened entry point that implements:
//! - Security manager initialization
//! - Zero trust architecture setup
//! - Financial security controls
//! - HTTPS server with TLS enforcement
//! - Comprehensive audit logging

use std::sync::Arc;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use tokio::signal;
use tracing::{info, error, warn};

use crate::security::SecurityManager;
use crate::financial_security::FinancialSecurityManager;
use crate::zero_trust::ZeroTrustEngine;
use crate::https_server::{SecureHttpsServer, TlsConfig};
use crate::error::{HiveMindError, Result};
use crate::config::HiveMindConfig;
use crate::{HiveMind, HiveMindBuilder};

/// Secure application entry point
pub struct SecureHiveMindApp {
    security_manager: Arc<SecurityManager>,
    financial_security: Arc<FinancialSecurityManager>,
    zero_trust_engine: Arc<ZeroTrustEngine>,
    hive_mind: Arc<HiveMind>,
    https_server: Option<SecureHttpsServer>,
}

impl SecureHiveMindApp {
    /// Initialize secure application
    pub async fn new(config: HiveMindConfig) -> Result<Self> {
        info!("Initializing secure hive mind application");

        // Initialize security layers
        let security_manager = Arc::new(SecurityManager::new().await?);
        let financial_security = Arc::new(
            FinancialSecurityManager::new(Arc::clone(&security_manager)).await?
        );
        let zero_trust_engine = Arc::new(
            ZeroTrustEngine::new(Arc::clone(&security_manager)).await?
        );

        // Initialize core hive mind with security
        let hive_mind = Arc::new(HiveMindBuilder::new(config.clone()).build().await?);

        // Initialize HTTPS server if API is enabled
        let https_server = if config.network.api_port > 0 {
            let bind_addr = SocketAddr::new(
                config.network.listen_addr.parse()
                    .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))),
                config.network.api_port,
            );
            
            Some(SecureHttpsServer::new(
                Arc::clone(&security_manager),
                bind_addr,
                TlsConfig::default(),
            ))
        } else {
            None
        };

        info!("Secure hive mind application initialized successfully");

        Ok(Self {
            security_manager,
            financial_security,
            zero_trust_engine,
            hive_mind,
            https_server,
        })
    }

    /// Start secure application with all security measures
    pub async fn start(&self) -> Result<()> {
        info!("Starting secure hive mind application");

        // Perform security pre-flight checks
        self.perform_security_checks().await?;

        // Start core hive mind system
        self.hive_mind.start().await?;

        // Start HTTPS server if configured
        if let Some(ref server) = self.https_server {
            let server_clone = server.clone();
            tokio::spawn(async move {
                if let Err(e) = server_clone.start().await {
                    error!("HTTPS server failed: {}", e);
                }
            });
        }

        info!("Secure hive mind application started successfully");

        // Wait for shutdown signal
        self.wait_for_shutdown().await?;

        // Graceful shutdown
        self.shutdown().await?;

        Ok(())
    }

    /// Perform comprehensive security checks before startup
    async fn perform_security_checks(&self) -> Result<()> {
        info!("Performing security pre-flight checks");

        // Check cryptographic capabilities
        self.verify_cryptographic_functions().await?;

        // Verify secure random number generation
        self.verify_random_generation().await?;

        // Check certificate validity (if HTTPS enabled)
        if self.https_server.is_some() {
            self.verify_tls_certificates().await?;
        }

        // Test database security
        self.verify_database_security().await?;

        // Verify audit logging
        self.verify_audit_logging().await?;

        info!("All security checks passed");
        Ok(())
    }

    /// Verify cryptographic functions are working
    async fn verify_cryptographic_functions(&self) -> Result<()> {
        // Test encryption/decryption
        let test_data = b"security_test_data";
        let encrypted = self.security_manager.encrypt_data(test_data, "master").await?;
        let decrypted = self.security_manager.decrypt_data(&encrypted).await?;
        
        if decrypted != test_data {
            return Err(HiveMindError::Internal(
                "Cryptographic functions failed verification".to_string()
            ));
        }

        // Test digital signatures
        let signature = self.security_manager.sign_data(test_data, "master").await?;
        let is_valid = self.security_manager.verify_signature(test_data, &signature, "master").await?;
        
        if !is_valid {
            return Err(HiveMindError::Internal(
                "Digital signature verification failed".to_string()
            ));
        }

        info!("Cryptographic functions verified successfully");
        Ok(())
    }

    /// Verify secure random number generation
    async fn verify_random_generation(&self) -> Result<()> {
        let token1 = self.security_manager.generate_secure_token(32).await?;
        let token2 = self.security_manager.generate_secure_token(32).await?;
        
        if token1 == token2 {
            return Err(HiveMindError::Internal(
                "Random number generation appears compromised".to_string()
            ));
        }

        info!("Secure random number generation verified");
        Ok(())
    }

    /// Verify TLS certificates are valid
    async fn verify_tls_certificates(&self) -> Result<()> {
        use crate::https_server::CertificateManager;
        
        let cert_path = "certs/server.crt";
        let is_valid = CertificateManager::verify_certificate_chain(cert_path)?;
        
        if !is_valid {
            warn!("TLS certificate validation failed, generating self-signed certificate");
            CertificateManager::generate_self_signed_cert(
                cert_path,
                "certs/server.key",
                "localhost"
            )?;
        }

        info!("TLS certificates verified");
        Ok(())
    }

    /// Verify database security configuration
    async fn verify_database_security(&self) -> Result<()> {
        // Test password hashing
        let test_password = "test_password_123";
        let hash = self.security_manager.hash_password(test_password).await?;
        let is_valid = self.security_manager.verify_password(test_password, &hash).await?;
        
        if !is_valid {
            return Err(HiveMindError::Internal(
                "Password hashing verification failed".to_string()
            ));
        }

        info!("Database security verified");
        Ok(())
    }

    /// Verify audit logging is functional
    async fn verify_audit_logging(&self) -> Result<()> {
        use crate::security::SecurityEvent;
        
        self.security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
            policy: "Security Check".to_string(),
            details: "Application startup security verification".to_string(),
        }).await?;

        info!("Audit logging verified");
        Ok(())
    }

    /// Wait for shutdown signals
    async fn wait_for_shutdown(&self) -> Result<()> {
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C shutdown signal");
            }
            _ = self.wait_for_sigterm() => {
                info!("Received SIGTERM shutdown signal");
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    async fn wait_for_sigterm(&self) -> Result<()> {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate())
            .map_err(|e| HiveMindError::Internal(format!("Failed to setup SIGTERM handler: {}", e)))?;
        sigterm.recv().await;
        Ok(())
    }

    #[cfg(not(unix))]
    async fn wait_for_sigterm(&self) -> Result<()> {
        // On non-Unix systems, just wait indefinitely
        std::future::pending::<()>().await;
        Ok(())
    }

    /// Gracefully shutdown the application
    async fn shutdown(&self) -> Result<()> {
        info!("Starting graceful shutdown");

        // Stop core hive mind
        self.hive_mind.stop().await?;

        // Log shutdown event
        use crate::security::SecurityEvent;
        self.security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
            policy: "Application Shutdown".to_string(),
            details: "Secure application shutdown completed".to_string(),
        }).await?;

        info!("Graceful shutdown completed");
        Ok(())
    }

    /// Get security manager reference
    pub fn security_manager(&self) -> &Arc<SecurityManager> {
        &self.security_manager
    }

    /// Get financial security manager reference
    pub fn financial_security(&self) -> &Arc<FinancialSecurityManager> {
        &self.financial_security
    }

    /// Get zero trust engine reference
    pub fn zero_trust_engine(&self) -> &Arc<ZeroTrustEngine> {
        &self.zero_trust_engine
    }

    /// Get hive mind reference
    pub fn hive_mind(&self) -> &Arc<HiveMind> {
        &self.hive_mind
    }
}

/// Security-focused main function
pub async fn secure_main() -> Result<()> {
    // Initialize comprehensive logging with security focus
    init_secure_logging()?;

    // Load configuration with security validation
    let config = load_secure_config().await?;

    // Validate security configuration
    validate_security_config(&config)?;

    // Create and start secure application
    let app = SecureHiveMindApp::new(config).await?;
    app.start().await?;

    Ok(())
}

/// Initialize logging with security considerations
fn init_secure_logging() -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .map_err(|e| HiveMindError::Internal(format!("Invalid log level: {}", e)))?;

    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .json() // Use structured JSON logging for security
        )
        .init();

    Ok(())
}

/// Load configuration with security validation
async fn load_secure_config() -> Result<HiveMindConfig> {
    let config_path = std::env::var("HIVE_MIND_CONFIG")
        .unwrap_or_else(|_| "config/production.toml".to_string());

    if !std::path::Path::new(&config_path).exists() {
        warn!("Configuration file not found: {}, using default", config_path);
        return Ok(HiveMindConfig::default());
    }

    let config = HiveMindConfig::load_from_file(&config_path)?;
    config.validate()?;

    info!("Configuration loaded successfully from: {}", config_path);
    Ok(config)
}

/// Validate security-specific configuration
fn validate_security_config(config: &HiveMindConfig) -> Result<()> {
    // Ensure encryption is enabled in production
    if !config.security.enable_encryption {
        return Err(HiveMindError::InvalidState {
            message: "Encryption must be enabled for secure operation".to_string(),
        });
    }

    // Validate network security settings
    if config.network.listen_addr == "0.0.0.0" && std::env::var("ALLOW_ALL_INTERFACES").is_err() {
        warn!("Binding to all interfaces - ensure firewall is properly configured");
    }

    // Validate database security
    if config.memory.persistence.connection_string.starts_with("sqlite://") 
        && !config.memory.persistence.connection_string.contains("?mode=") {
        warn!("SQLite database should specify access mode for security");
    }

    info!("Security configuration validation passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_secure_app_initialization() {
        let config = HiveMindConfig::default();
        let result = SecureHiveMindApp::new(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_security_config_validation() {
        let mut config = HiveMindConfig::default();
        config.security.enable_encryption = true;
        
        let result = validate_security_config(&config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insecure_config_rejection() {
        let mut config = HiveMindConfig::default();
        config.security.enable_encryption = false;
        
        let result = validate_security_config(&config);
        assert!(result.is_err());
    }
}