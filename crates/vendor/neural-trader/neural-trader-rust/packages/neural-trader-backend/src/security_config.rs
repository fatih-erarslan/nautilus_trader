//! Security Configuration Module
//!
//! Provides security configuration for:
//! - CORS (Cross-Origin Resource Sharing)
//! - Security headers
//! - Content Security Policy
//! - SSL/TLS settings
//! - Security policies

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct CorsConfig {
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Exposed headers
    pub exposed_headers: Vec<String>,
    /// Allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (seconds)
    pub max_age: u32,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["http://localhost:3000".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-API-Key".to_string(),
            ],
            exposed_headers: vec![
                "X-RateLimit-Limit".to_string(),
                "X-RateLimit-Remaining".to_string(),
                "X-RateLimit-Reset".to_string(),
            ],
            allow_credentials: true,
            max_age: 3600,
        }
    }
}

impl CorsConfig {
    /// Check if origin is allowed
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        if self.allowed_origins.contains(&"*".to_string()) {
            return true;
        }

        self.allowed_origins.contains(&origin.to_string())
    }

    /// Check if method is allowed
    pub fn is_method_allowed(&self, method: &str) -> bool {
        self.allowed_methods.contains(&method.to_uppercase())
    }

    /// Get CORS headers for response
    pub fn get_cors_headers(&self, origin: Option<&str>) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();

        // Set allowed origin
        if let Some(orig) = origin {
            if self.is_origin_allowed(orig) {
                headers.insert(
                    "Access-Control-Allow-Origin".to_string(),
                    orig.to_string(),
                );
            }
        }

        // Set allowed methods
        headers.insert(
            "Access-Control-Allow-Methods".to_string(),
            self.allowed_methods.join(", "),
        );

        // Set allowed headers
        headers.insert(
            "Access-Control-Allow-Headers".to_string(),
            self.allowed_headers.join(", "),
        );

        // Set exposed headers
        if !self.exposed_headers.is_empty() {
            headers.insert(
                "Access-Control-Expose-Headers".to_string(),
                self.exposed_headers.join(", "),
            );
        }

        // Set allow credentials
        if self.allow_credentials {
            headers.insert(
                "Access-Control-Allow-Credentials".to_string(),
                "true".to_string(),
            );
        }

        // Set max age
        headers.insert(
            "Access-Control-Max-Age".to_string(),
            self.max_age.to_string(),
        );

        headers
    }
}

/// Security headers configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct SecurityHeaders {
    /// Strict-Transport-Security header
    pub hsts_max_age: u32,
    /// X-Frame-Options
    pub frame_options: String,
    /// X-Content-Type-Options
    pub content_type_options: bool,
    /// X-XSS-Protection
    pub xss_protection: bool,
    /// Referrer-Policy
    pub referrer_policy: String,
    /// Content-Security-Policy
    pub csp: Option<String>,
}

impl Default for SecurityHeaders {
    fn default() -> Self {
        Self {
            hsts_max_age: 31536000, // 1 year
            frame_options: "DENY".to_string(),
            content_type_options: true,
            xss_protection: true,
            referrer_policy: "strict-origin-when-cross-origin".to_string(),
            csp: Some(
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
                    .to_string(),
            ),
        }
    }
}

impl SecurityHeaders {
    /// Get all security headers
    pub fn get_headers(&self) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();

        // HSTS
        headers.insert(
            "Strict-Transport-Security".to_string(),
            format!("max-age={}; includeSubDomains; preload", self.hsts_max_age),
        );

        // X-Frame-Options
        headers.insert("X-Frame-Options".to_string(), self.frame_options.clone());

        // X-Content-Type-Options
        if self.content_type_options {
            headers.insert("X-Content-Type-Options".to_string(), "nosniff".to_string());
        }

        // X-XSS-Protection
        if self.xss_protection {
            headers.insert("X-XSS-Protection".to_string(), "1; mode=block".to_string());
        }

        // Referrer-Policy
        headers.insert("Referrer-Policy".to_string(), self.referrer_policy.clone());

        // Content-Security-Policy
        if let Some(ref csp) = self.csp {
            headers.insert("Content-Security-Policy".to_string(), csp.clone());
        }

        // Additional security headers
        headers.insert(
            "X-Permitted-Cross-Domain-Policies".to_string(),
            "none".to_string(),
        );
        headers.insert("X-Download-Options".to_string(), "noopen".to_string());

        headers
    }
}

/// Complete security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub cors: CorsConfig,
    pub headers: SecurityHeaders,
    pub require_https: bool,
    pub trusted_proxies: HashSet<String>,
    pub ip_whitelist: Option<HashSet<String>>,
    pub ip_blacklist: HashSet<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            cors: CorsConfig::default(),
            headers: SecurityHeaders::default(),
            require_https: true,
            trusted_proxies: HashSet::new(),
            ip_whitelist: None,
            ip_blacklist: HashSet::new(),
        }
    }
}

impl SecurityConfig {
    /// Check if IP address is allowed
    pub fn is_ip_allowed(&self, ip: &str) -> bool {
        // Check blacklist first
        if self.ip_blacklist.contains(ip) {
            return false;
        }

        // If whitelist exists, IP must be in it
        if let Some(ref whitelist) = self.ip_whitelist {
            return whitelist.contains(ip);
        }

        // No whitelist means all non-blacklisted IPs are allowed
        true
    }

    /// Add IP to blacklist
    pub fn blacklist_ip(&mut self, ip: String) {
        self.ip_blacklist.insert(ip);
    }

    /// Remove IP from blacklist
    pub fn unblacklist_ip(&mut self, ip: &str) {
        self.ip_blacklist.remove(ip);
    }

    /// Add IP to whitelist
    pub fn whitelist_ip(&mut self, ip: String) {
        if self.ip_whitelist.is_none() {
            self.ip_whitelist = Some(HashSet::new());
        }

        if let Some(ref mut whitelist) = self.ip_whitelist {
            whitelist.insert(ip);
        }
    }

    /// Check if proxy is trusted
    pub fn is_trusted_proxy(&self, ip: &str) -> bool {
        self.trusted_proxies.contains(ip)
    }
}

/// Global security configuration
static SECURITY_CONFIG: once_cell::sync::Lazy<std::sync::RwLock<SecurityConfig>> =
    once_cell::sync::Lazy::new(|| std::sync::RwLock::new(SecurityConfig::default()));

/// Initialize security configuration
#[napi]
pub fn init_security_config(
    cors_config: Option<CorsConfig>,
    require_https: Option<bool>,
) -> Result<String> {
    let mut config = SecurityConfig::default();

    if let Some(cors) = cors_config {
        config.cors = cors;
    }

    if let Some(https) = require_https {
        config.require_https = https;
    }

    let mut sec_config = SECURITY_CONFIG.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    *sec_config = config;

    Ok("Security configuration initialized".to_string())
}

/// Get CORS headers for a request
#[napi]
pub fn get_cors_headers(origin: Option<String>) -> Result<String> {
    let config = SECURITY_CONFIG.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let headers = config.cors.get_cors_headers(origin.as_deref());

    serde_json::to_string(&headers)
        .map_err(|e| Error::from_reason(format!("JSON error: {}", e)))
}

/// Get security headers
#[napi]
pub fn get_security_headers() -> Result<String> {
    let config = SECURITY_CONFIG.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let headers = config.headers.get_headers();

    serde_json::to_string(&headers)
        .map_err(|e| Error::from_reason(format!("JSON error: {}", e)))
}

/// Check if IP is allowed
#[napi]
pub fn check_ip_allowed(ip: String) -> Result<bool> {
    let config = SECURITY_CONFIG.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    Ok(config.is_ip_allowed(&ip))
}

/// Check if origin is allowed for CORS
#[napi]
pub fn check_cors_origin(origin: String) -> Result<bool> {
    let config = SECURITY_CONFIG.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    Ok(config.cors.is_origin_allowed(&origin))
}

/// Add IP to blacklist (admin operation)
#[napi]
pub fn add_ip_to_blacklist(ip: String) -> Result<String> {
    let mut config = SECURITY_CONFIG.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    config.blacklist_ip(ip.clone());

    Ok(format!("IP {} added to blacklist", ip))
}

/// Remove IP from blacklist (admin operation)
#[napi]
pub fn remove_ip_from_blacklist(ip: String) -> Result<String> {
    let mut config = SECURITY_CONFIG.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    config.unblacklist_ip(&ip);

    Ok(format!("IP {} removed from blacklist", ip))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cors_origin_check() {
        let config = CorsConfig::default();
        assert!(config.is_origin_allowed("http://localhost:3000"));
        assert!(!config.is_origin_allowed("http://evil.com"));
    }

    #[test]
    fn test_security_headers() {
        let headers = SecurityHeaders::default();
        let header_map = headers.get_headers();

        assert!(header_map.contains_key("Strict-Transport-Security"));
        assert!(header_map.contains_key("X-Frame-Options"));
        assert!(header_map.contains_key("X-Content-Type-Options"));
    }

    #[test]
    fn test_ip_filtering() {
        let mut config = SecurityConfig::default();

        // All IPs allowed by default
        assert!(config.is_ip_allowed("192.168.1.1"));

        // Blacklist an IP
        config.blacklist_ip("192.168.1.100".to_string());
        assert!(!config.is_ip_allowed("192.168.1.100"));
        assert!(config.is_ip_allowed("192.168.1.1"));

        // Add whitelist
        config.whitelist_ip("192.168.1.1".to_string());
        assert!(config.is_ip_allowed("192.168.1.1"));
        assert!(!config.is_ip_allowed("192.168.1.2")); // Not in whitelist
    }
}
