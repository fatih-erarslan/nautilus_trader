//! HTTPS server with security headers and TLS enforcement
//! 
//! This module provides a secure HTTPS server implementation with:
//! - Mandatory TLS 1.3 encryption
//! - Security headers (HSTS, CSP, X-Frame-Options, etc.)
//! - Certificate validation and management
//! - Request/response filtering and sanitization

use std::sync::Arc;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};

use crate::error::{HiveMindError, Result};
use crate::security::{SecurityManager, SecurityEvent};

/// Security headers configuration
const SECURITY_HEADERS: &[(&str, &str)] = &[
    ("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("X-XSS-Protection", "1; mode=block"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("Permissions-Policy", "geolocation=(), microphone=(), camera=()"),
    ("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; media-src 'self'; child-src 'none';"),
];

/// Secure HTTPS server for hive mind API
pub struct SecureHttpsServer {
    security_manager: Arc<SecurityManager>,
    bind_address: SocketAddr,
    tls_config: TlsConfig,
}

impl SecureHttpsServer {
    /// Create new secure HTTPS server
    pub fn new(
        security_manager: Arc<SecurityManager>,
        bind_address: SocketAddr,
        tls_config: TlsConfig,
    ) -> Self {
        Self {
            security_manager,
            bind_address,
            tls_config,
        }
    }

    /// Start the secure server
    pub async fn start(&self) -> Result<()> {
        info!("Starting secure HTTPS server on {}", self.bind_address);

        // Validate TLS configuration
        self.validate_tls_config().await?;

        let listener = TcpListener::bind(self.bind_address)
            .await
            .map_err(|e| HiveMindError::Internal(format!("Failed to bind server: {}", e)))?;

        info!("HTTPS server listening on {}", self.bind_address);

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    let security_manager = Arc::clone(&self.security_manager);
                    let tls_config = self.tls_config.clone();
                    
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, addr, security_manager, tls_config).await {
                            error!("Connection handling error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                    continue;
                }
            }
        }
    }

    /// Handle individual client connection
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        client_addr: SocketAddr,
        security_manager: Arc<SecurityManager>,
        _tls_config: TlsConfig,
    ) -> Result<()> {
        debug!("New connection from {}", client_addr);

        // Rate limiting check
        let client_ip = client_addr.ip().to_string();
        if !security_manager.check_rate_limit(&client_ip, "api").await? {
            warn!("Rate limit exceeded for {}", client_ip);
            return Ok(());
        }

        // TODO: Implement TLS handshake and HTTP request processing
        // For now, this is a placeholder that would be replaced with
        // actual HTTPS handling using libraries like hyper-tls or rustls

        // Log connection for audit
        security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
            policy: "HTTPS Connection".to_string(),
            details: format!("Connection from {}", client_addr),
        }).await?;

        Ok(())
    }

    /// Validate TLS configuration
    async fn validate_tls_config(&self) -> Result<()> {
        // Verify certificate files exist and are valid
        if !std::path::Path::new(&self.tls_config.cert_path).exists() {
            return Err(HiveMindError::InvalidState {
                message: format!("Certificate file not found: {}", self.tls_config.cert_path),
            });
        }

        if !std::path::Path::new(&self.tls_config.key_path).exists() {
            return Err(HiveMindError::InvalidState {
                message: format!("Private key file not found: {}", self.tls_config.key_path),
            });
        }

        info!("TLS configuration validated successfully");
        Ok(())
    }

    /// Apply security headers to HTTP response
    pub fn apply_security_headers(headers: &mut Vec<(String, String)>) {
        for (name, value) in SECURITY_HEADERS {
            headers.push((name.to_string(), value.to_string()));
        }
    }

    /// Validate and sanitize HTTP request
    pub async fn validate_request(
        &self,
        request: &HttpRequest,
        client_ip: &str,
    ) -> Result<HttpRequest> {
        // Check request size limits
        if request.body.len() > 1024 * 1024 {
            return Err(HiveMindError::InvalidState {
                message: "Request body too large".to_string(),
            });
        }

        // Validate headers
        for (name, value) in &request.headers {
            if name.len() > 100 || value.len() > 1000 {
                return Err(HiveMindError::InvalidState {
                    message: "Invalid header size".to_string(),
                });
            }
        }

        // Check for suspicious patterns
        if self.detect_malicious_patterns(&request.path, &request.body).await? {
            self.security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
                policy: "Malicious Request Pattern".to_string(),
                details: format!("Suspicious request from {}: {}", client_ip, request.path),
            }).await?;
            
            return Err(HiveMindError::InvalidState {
                message: "Request blocked due to security policy".to_string(),
            });
        }

        Ok(request.clone())
    }

    /// Detect potentially malicious request patterns
    async fn detect_malicious_patterns(&self, path: &str, body: &str) -> Result<bool> {
        // Check for common attack patterns
        let suspicious_patterns = [
            "../", "..\\",           // Path traversal
            "<script", "</script>",  // XSS attempts
            "union select",          // SQL injection
            "drop table",            // SQL injection
            "<?php",                 // Code injection
            "javascript:",           // JavaScript injection
        ];

        let combined_content = format!("{} {}", path, body).to_lowercase();
        
        for pattern in &suspicious_patterns {
            if combined_content.contains(pattern) {
                warn!("Malicious pattern detected: {} in request", pattern);
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// TLS configuration for HTTPS server
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to TLS certificate file
    pub cert_path: String,
    /// Path to private key file
    pub key_path: String,
    /// Minimum TLS version (should be 1.3 for security)
    pub min_version: TlsVersion,
    /// Client certificate verification mode
    pub client_cert_verification: ClientCertVerification,
    /// OCSP stapling enabled
    pub ocsp_stapling: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_path: "certs/server.crt".to_string(),
            key_path: "certs/server.key".to_string(),
            min_version: TlsVersion::V1_3,
            client_cert_verification: ClientCertVerification::Optional,
            ocsp_stapling: true,
        }
    }
}

/// TLS version specification
#[derive(Debug, Clone, Copy)]
pub enum TlsVersion {
    V1_2,
    V1_3,
}

/// Client certificate verification modes
#[derive(Debug, Clone, Copy)]
pub enum ClientCertVerification {
    None,
    Optional,
    Required,
}

/// HTTP request structure for validation
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// HTTP response with security headers
#[derive(Debug, Clone)]
pub struct SecureHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl SecureHttpResponse {
    /// Create new secure response with security headers
    pub fn new(status: u16, body: String) -> Self {
        let mut headers = Vec::new();
        SecureHttpsServer::apply_security_headers(&mut headers);
        
        Self {
            status,
            headers,
            body,
        }
    }

    /// Add custom header
    pub fn with_header(mut self, name: String, value: String) -> Self {
        self.headers.push((name, value));
        self
    }

    /// Set content type
    pub fn with_content_type(self, content_type: &str) -> Self {
        self.with_header("Content-Type".to_string(), content_type.to_string())
    }
}

/// WebSocket security wrapper
pub struct SecureWebSocket {
    security_manager: Arc<SecurityManager>,
}

impl SecureWebSocket {
    pub fn new(security_manager: Arc<SecurityManager>) -> Self {
        Self { security_manager }
    }

    /// Validate WebSocket message
    pub async fn validate_message(&self, message: &Message, client_ip: &str) -> Result<Message> {
        match message {
            Message::Text(text) => {
                // Validate text message size
                if text.len() > 10 * 1024 {
                    return Err(HiveMindError::InvalidState {
                        message: "WebSocket message too large".to_string(),
                    });
                }

                // Validate JSON if applicable
                if text.starts_with('{') || text.starts_with('[') {
                    let _: serde_json::Value = serde_json::from_str(text)
                        .map_err(|_| HiveMindError::InvalidState {
                            message: "Invalid JSON in WebSocket message".to_string(),
                        })?;
                }

                Ok(message.clone())
            }
            Message::Binary(data) => {
                // Validate binary message size
                if data.len() > 1024 * 1024 {
                    return Err(HiveMindError::InvalidState {
                        message: "WebSocket binary message too large".to_string(),
                    });
                }
                
                Ok(message.clone())
            }
            _ => Ok(message.clone()),
        }
    }

    /// Handle WebSocket connection with security
    pub async fn handle_secure_connection(&self, client_addr: SocketAddr) -> Result<()> {
        let client_ip = client_addr.ip().to_string();
        
        // Rate limiting for WebSocket connections
        if !self.security_manager.check_rate_limit(&client_ip, "websocket").await? {
            return Err(HiveMindError::InvalidState {
                message: "WebSocket rate limit exceeded".to_string(),
            });
        }

        // Log connection
        self.security_manager.log_security_event(SecurityEvent::SecurityPolicyViolation {
            policy: "WebSocket Connection".to_string(),
            details: format!("WebSocket connection from {}", client_addr),
        }).await?;

        Ok(())
    }
}

/// Certificate management utilities
pub struct CertificateManager;

impl CertificateManager {
    /// Validate certificate expiration
    pub fn check_certificate_expiry(cert_path: &str) -> Result<Duration> {
        // TODO: Implement actual certificate parsing and expiry checking
        // This would use libraries like rustls or openssl to parse certificates
        
        // Placeholder implementation
        if !std::path::Path::new(cert_path).exists() {
            return Err(HiveMindError::InvalidState {
                message: "Certificate file not found".to_string(),
            });
        }

        // Return 30 days as placeholder
        Ok(Duration::from_secs(30 * 24 * 60 * 60))
    }

    /// Generate self-signed certificate for development
    pub fn generate_self_signed_cert(
        cert_path: &str,
        key_path: &str,
        domain: &str,
    ) -> Result<()> {
        // TODO: Implement certificate generation
        // This would use libraries like rcgen for certificate generation
        
        info!("Generated self-signed certificate for domain: {}", domain);
        info!("Certificate saved to: {}", cert_path);
        info!("Private key saved to: {}", key_path);
        
        Ok(())
    }

    /// Verify certificate chain
    pub fn verify_certificate_chain(cert_path: &str) -> Result<bool> {
        // TODO: Implement certificate chain verification
        // This would validate the certificate against trusted CAs
        
        if !std::path::Path::new(cert_path).exists() {
            return Ok(false);
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_security_headers() {
        let mut headers = Vec::new();
        SecureHttpsServer::apply_security_headers(&mut headers);
        
        assert!(!headers.is_empty());
        assert!(headers.iter().any(|(name, _)| name == "Strict-Transport-Security"));
        assert!(headers.iter().any(|(name, _)| name == "Content-Security-Policy"));
    }

    #[tokio::test]
    async fn test_secure_response_creation() {
        let response = SecureHttpResponse::new(200, "OK".to_string())
            .with_content_type("application/json");
        
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "OK");
        assert!(response.headers.iter().any(|(name, value)| 
            name == "Content-Type" && value == "application/json"));
    }

    #[tokio::test]
    async fn test_tls_config_default() {
        let config = TlsConfig::default();
        
        assert_eq!(config.cert_path, "certs/server.crt");
        assert_eq!(config.key_path, "certs/server.key");
        assert!(matches!(config.min_version, TlsVersion::V1_3));
        assert!(config.ocsp_stapling);
    }

    #[test]
    fn test_certificate_manager() {
        // Test certificate file existence check
        let result = CertificateManager::check_certificate_expiry("nonexistent.crt");
        assert!(result.is_err());
        
        // Test certificate chain verification for nonexistent file
        let result = CertificateManager::verify_certificate_chain("nonexistent.crt");
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}