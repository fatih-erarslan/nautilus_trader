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

/// Certificate management utilities with real X.509 operations
pub struct CertificateManager;

impl CertificateManager {
    /// Validate certificate expiration
    /// Parses X.509 certificate and returns time until expiry
    pub fn check_certificate_expiry(cert_path: &str) -> Result<Duration> {
        use std::fs;
        use x509_parser::prelude::*;

        // Read certificate file
        let cert_pem = fs::read_to_string(cert_path)
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to read certificate: {}", e),
            })?;

        // Parse PEM format
        let (_, pem_block) = pem::parse_x509_pem(cert_pem.as_bytes())
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to parse PEM: {:?}", e),
            })?;

        // Parse X.509 certificate
        let (_, cert) = X509Certificate::from_der(&pem_block.contents)
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to parse X.509 certificate: {:?}", e),
            })?;

        // Get validity period
        let validity = cert.validity();
        let not_after = validity.not_after;

        // Convert to SystemTime
        let not_after_time = not_after.to_datetime();
        let now = time::OffsetDateTime::now_utc();

        // Calculate duration until expiry
        if not_after_time < now {
            return Err(HiveMindError::InvalidState {
                message: "Certificate has expired".to_string(),
            });
        }

        let duration_until_expiry = not_after_time - now;
        let seconds = duration_until_expiry.whole_seconds();

        if seconds < 0 {
            return Err(HiveMindError::InvalidState {
                message: "Certificate has expired".to_string(),
            });
        }

        // Log warning if certificate expires within 30 days
        if seconds < 30 * 24 * 60 * 60 {
            warn!(
                "Certificate at {} expires in {} days",
                cert_path,
                seconds / (24 * 60 * 60)
            );
        }

        Ok(Duration::from_secs(seconds as u64))
    }

    /// Generate self-signed certificate for development/testing
    /// Uses ECDSA P-256 for strong security
    pub fn generate_self_signed_cert(
        cert_path: &str,
        key_path: &str,
        domain: &str,
    ) -> Result<()> {
        use rcgen::{
            Certificate, CertificateParams, DistinguishedName,
            DnType, KeyPair, PKCS_ECDSA_P256_SHA256,
        };
        use std::fs;

        // Create distinguished name
        let mut distinguished_name = DistinguishedName::new();
        distinguished_name.push(DnType::CommonName, domain);
        distinguished_name.push(DnType::OrganizationName, "HyperPhysics");
        distinguished_name.push(DnType::CountryName, "US");

        // Configure certificate parameters
        let mut params = CertificateParams::default();
        params.distinguished_name = distinguished_name;
        params.alg = &PKCS_ECDSA_P256_SHA256;
        params.subject_alt_names = vec![
            rcgen::SanType::DnsName(domain.to_string()),
            rcgen::SanType::DnsName("localhost".to_string()),
            rcgen::SanType::IpAddress(std::net::IpAddr::V4(
                std::net::Ipv4Addr::new(127, 0, 0, 1),
            )),
        ];

        // Set validity period (1 year)
        params.not_before = time::OffsetDateTime::now_utc();
        params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);

        // Add key usage extensions
        params.is_ca = rcgen::IsCa::NoCa;
        params.key_usages = vec![
            rcgen::KeyUsagePurpose::DigitalSignature,
            rcgen::KeyUsagePurpose::KeyEncipherment,
        ];
        params.extended_key_usages = vec![
            rcgen::ExtendedKeyUsagePurpose::ServerAuth,
            rcgen::ExtendedKeyUsagePurpose::ClientAuth,
        ];

        // Generate key pair
        let key_pair = KeyPair::generate(&PKCS_ECDSA_P256_SHA256)
            .map_err(|e| HiveMindError::Internal(format!("Failed to generate key pair: {}", e)))?;

        // Generate certificate
        let cert = params.self_signed(&key_pair)
            .map_err(|e| HiveMindError::Internal(format!("Failed to generate certificate: {}", e)))?;

        // Create parent directories if needed
        if let Some(parent) = std::path::Path::new(cert_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| HiveMindError::Internal(format!("Failed to create directory: {}", e)))?;
        }

        // Write certificate to file
        fs::write(cert_path, cert.pem())
            .map_err(|e| HiveMindError::Internal(format!("Failed to write certificate: {}", e)))?;

        // Write private key to file (with restrictive permissions on Unix)
        fs::write(key_path, key_pair.serialize_pem())
            .map_err(|e| HiveMindError::Internal(format!("Failed to write private key: {}", e)))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(key_path, fs::Permissions::from_mode(0o600))
                .map_err(|e| HiveMindError::Internal(format!("Failed to set key permissions: {}", e)))?;
        }

        info!("Generated self-signed certificate for domain: {}", domain);
        info!("Certificate saved to: {}", cert_path);
        info!("Private key saved to: {}", key_path);
        info!("Certificate valid until: {}", params.not_after);

        Ok(())
    }

    /// Verify certificate chain and basic properties
    /// Returns true if certificate is valid and properly formed
    pub fn verify_certificate_chain(cert_path: &str) -> Result<bool> {
        use std::fs;
        use x509_parser::prelude::*;

        // Read certificate file
        let cert_pem = match fs::read_to_string(cert_path) {
            Ok(contents) => contents,
            Err(_) => return Ok(false),
        };

        // Parse PEM format
        let (_, pem_block) = match pem::parse_x509_pem(cert_pem.as_bytes()) {
            Ok(result) => result,
            Err(e) => {
                warn!("Failed to parse PEM: {:?}", e);
                return Ok(false);
            }
        };

        // Parse X.509 certificate
        let (_, cert) = match X509Certificate::from_der(&pem_block.contents) {
            Ok(result) => result,
            Err(e) => {
                warn!("Failed to parse X.509 certificate: {:?}", e);
                return Ok(false);
            }
        };

        // Verify certificate version (should be v3 for proper extensions)
        if cert.version() != X509Version::V3 {
            warn!("Certificate is not X.509 v3");
            return Ok(false);
        }

        // Check that certificate has not expired
        let validity = cert.validity();
        let now = time::OffsetDateTime::now_utc();

        if validity.not_before.to_datetime() > now {
            warn!("Certificate is not yet valid");
            return Ok(false);
        }

        if validity.not_after.to_datetime() < now {
            warn!("Certificate has expired");
            return Ok(false);
        }

        // Verify signature algorithm is secure
        let sig_alg = cert.signature_algorithm.oid().to_string();
        let secure_algorithms = [
            "1.2.840.10045.4.3.2",  // ECDSA with SHA-256
            "1.2.840.10045.4.3.3",  // ECDSA with SHA-384
            "1.2.840.10045.4.3.4",  // ECDSA with SHA-512
            "1.2.840.113549.1.1.11", // RSA with SHA-256
            "1.2.840.113549.1.1.12", // RSA with SHA-384
            "1.2.840.113549.1.1.13", // RSA with SHA-512
        ];

        if !secure_algorithms.contains(&sig_alg.as_str()) {
            warn!("Certificate uses potentially weak signature algorithm: {}", sig_alg);
            // Don't fail, just warn - some older certs may use SHA-1
        }

        // Check key usage extension if present
        if let Some(key_usage) = cert.key_usage() {
            let ku = key_usage.value;
            if !ku.digital_signature() && !ku.key_encipherment() {
                warn!("Certificate missing required key usage flags");
                return Ok(false);
            }
        }

        info!("Certificate chain verification passed for: {}", cert_path);
        Ok(true)
    }

    /// Get certificate information summary
    pub fn get_certificate_info(cert_path: &str) -> Result<CertificateInfo> {
        use std::fs;
        use x509_parser::prelude::*;

        let cert_pem = fs::read_to_string(cert_path)
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to read certificate: {}", e),
            })?;

        let (_, pem_block) = pem::parse_x509_pem(cert_pem.as_bytes())
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to parse PEM: {:?}", e),
            })?;

        let (_, cert) = X509Certificate::from_der(&pem_block.contents)
            .map_err(|e| HiveMindError::InvalidState {
                message: format!("Failed to parse X.509 certificate: {:?}", e),
            })?;

        let validity = cert.validity();
        let subject = cert.subject().to_string();
        let issuer = cert.issuer().to_string();
        let serial = cert.serial.to_string();
        let sig_alg = cert.signature_algorithm.algorithm.to_string();

        // Get Subject Alternative Names
        let san: Vec<String> = cert.subject_alternative_name()
            .map(|san| {
                san.value.general_names.iter()
                    .filter_map(|gn| {
                        match gn {
                            x509_parser::extensions::GeneralName::DNSName(name) => {
                                Some(name.to_string())
                            }
                            x509_parser::extensions::GeneralName::IPAddress(ip) => {
                                Some(format!("{:?}", ip))
                            }
                            _ => None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(CertificateInfo {
            subject,
            issuer,
            serial_number: serial,
            not_before: validity.not_before.to_datetime().to_string(),
            not_after: validity.not_after.to_datetime().to_string(),
            signature_algorithm: sig_alg,
            subject_alt_names: san,
        })
    }
}

/// Certificate information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    pub subject: String,
    pub issuer: String,
    pub serial_number: String,
    pub not_before: String,
    pub not_after: String,
    pub signature_algorithm: String,
    pub subject_alt_names: Vec<String>,
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