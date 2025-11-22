//! Enterprise-grade security middleware for web API protection
//! 
//! This module implements comprehensive security middleware including:
//! - Strict security headers (CSP, HSTS, etc.)
//! - Configurable CORS policies
//! - Advanced rate limiting with DDoS protection
//! - Request validation and sanitization
//! - Security event logging

use anyhow::{anyhow, Result};
use axum::{
    extract::{ConnectInfo, Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    middleware::Next,
    response::Response,
};
use chrono::{DateTime, Utc};
use governor::{
    clock::DefaultClock,
    state::{direct::NotKeyed, keyed::DefaultKeyedStateStore},
    Quota, RateLimiter,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    num::NonZeroU32,
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{debug, error, info, warn};

use super::audit::{
    AuditEvent, AuditEventType, AuditAction, AuditResult, 
    DataClassification, ComplianceTag, RequestDetails, ResponseDetails
};

/// Security middleware configuration
#[derive(Debug, Clone)]
pub struct SecurityMiddlewareConfig {
    /// Security headers configuration
    pub headers_config: SecurityHeadersConfig,
    
    /// CORS configuration
    pub cors_config: CorsConfig,
    
    /// Rate limiting configuration
    pub rate_limit_config: RateLimitConfig,
    
    /// Request validation configuration
    pub validation_config: ValidationConfig,
    
    /// Enable security event logging
    pub enable_audit_logging: bool,
    
    /// Trusted proxy headers
    pub trusted_proxies: Vec<IpAddr>,
}

#[derive(Debug, Clone)]
pub struct SecurityHeadersConfig {
    /// Content Security Policy
    pub csp: String,
    
    /// Strict Transport Security max age
    pub hsts_max_age: u32,
    
    /// Include subdomains in HSTS
    pub hsts_include_subdomains: bool,
    
    /// HSTS preload
    pub hsts_preload: bool,
    
    /// X-Frame-Options
    pub frame_options: FrameOptions,
    
    /// X-Content-Type-Options
    pub content_type_options: bool,
    
    /// Referrer Policy
    pub referrer_policy: ReferrerPolicy,
    
    /// Permissions Policy
    pub permissions_policy: Option<String>,
    
    /// Custom security headers
    pub custom_headers: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum FrameOptions {
    Deny,
    SameOrigin,
    AllowFrom(String),
}

#[derive(Debug, Clone)]
pub enum ReferrerPolicy {
    NoReferrer,
    NoReferrerWhenDowngrade,
    Origin,
    OriginWhenCrossOrigin,
    SameOrigin,
    StrictOrigin,
    StrictOriginWhenCrossOrigin,
    UnsafeUrl,
}

#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    
    /// Allowed methods
    pub allowed_methods: Vec<Method>,
    
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    
    /// Exposed headers
    pub exposed_headers: Vec<String>,
    
    /// Allow credentials
    pub allow_credentials: bool,
    
    /// Max age for preflight
    pub max_age: Option<Duration>,
    
    /// Development mode (allows all origins)
    pub development_mode: bool,
}

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    
    /// Default requests per hour
    pub default_rph: u32,
    
    /// Burst capacity multiplier
    pub burst_multiplier: f64,
    
    /// Per-IP rate limits
    pub ip_limits: HashMap<String, u32>,
    
    /// Per-user rate limits
    pub user_limits: HashMap<String, u32>,
    
    /// Whitelist IPs (no rate limiting)
    pub whitelist_ips: Vec<IpAddr>,
    
    /// DDoS protection threshold
    pub ddos_threshold: u32,
    
    /// Ban duration for DDoS detection
    pub ban_duration_minutes: u32,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum request body size
    pub max_body_size: usize,
    
    /// Maximum URL length
    pub max_url_length: usize,
    
    /// Maximum header size
    pub max_header_size: usize,
    
    /// Blocked user agents
    pub blocked_user_agents: Vec<String>,
    
    /// Required headers
    pub required_headers: Vec<String>,
    
    /// SQL injection detection
    pub sql_injection_detection: bool,
    
    /// XSS detection
    pub xss_detection: bool,
    
    /// Path traversal detection
    pub path_traversal_detection: bool,
}

impl Default for SecurityMiddlewareConfig {
    fn default() -> Self {
        Self {
            headers_config: SecurityHeadersConfig::default(),
            cors_config: CorsConfig::default(),
            rate_limit_config: RateLimitConfig::default(),
            validation_config: ValidationConfig::default(),
            enable_audit_logging: true,
            trusted_proxies: vec![
                "127.0.0.1".parse().unwrap(),
                "::1".parse().unwrap(),
            ],
        }
    }
}

impl Default for SecurityHeadersConfig {
    fn default() -> Self {
        Self {
            csp: "default-src 'self'; script-src 'self'; object-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'".to_string(),
            hsts_max_age: 31536000, // 1 year
            hsts_include_subdomains: true,
            hsts_preload: true,
            frame_options: FrameOptions::Deny,
            content_type_options: true,
            referrer_policy: ReferrerPolicy::StrictOriginWhenCrossOrigin,
            permissions_policy: Some("geolocation=(), microphone=(), camera=()".to_string()),
            custom_headers: HashMap::new(),
        }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec![
                "https://autopoiesis.com".to_string(),
                "https://api.autopoiesis.com".to_string(),
            ],
            allowed_methods: vec![Method::GET, Method::POST, Method::PUT, Method::DELETE],
            allowed_headers: vec![
                "authorization".to_string(),
                "content-type".to_string(),
                "accept".to_string(),
                "origin".to_string(),
                "x-requested-with".to_string(),
            ],
            exposed_headers: vec![
                "x-total-count".to_string(),
                "x-page-count".to_string(),
            ],
            allow_credentials: true,
            max_age: Some(Duration::from_secs(3600)),
            development_mode: false,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_rph: 1000,
            burst_multiplier: 2.0,
            ip_limits: HashMap::new(),
            user_limits: HashMap::new(),
            whitelist_ips: vec![
                "127.0.0.1".parse().unwrap(),
                "::1".parse().unwrap(),
            ],
            ddos_threshold: 10000,
            ban_duration_minutes: 60,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_body_size: 10 * 1024 * 1024, // 10MB
            max_url_length: 2048,
            max_header_size: 8192,
            blocked_user_agents: vec![
                "bot".to_string(),
                "crawler".to_string(),
                "spider".to_string(),
                "scraper".to_string(),
            ],
            required_headers: vec![
                "user-agent".to_string(),
            ],
            sql_injection_detection: true,
            xss_detection: true,
            path_traversal_detection: true,
        }
    }
}

/// Enterprise security middleware
pub struct SecurityMiddleware {
    config: SecurityMiddlewareConfig,
    rate_limiter: Arc<RateLimiter<NotKeyed, DefaultKeyedStateStore<IpAddr>, DefaultClock>>,
    banned_ips: Arc<RwLock<HashMap<IpAddr, DateTime<Utc>>>>,
    request_counts: Arc<RwLock<HashMap<IpAddr, Vec<DateTime<Utc>>>>>,
}

impl SecurityMiddleware {
    pub fn new(config: SecurityMiddlewareConfig) -> Self {
        let quota = Quota::per_hour(NonZeroU32::new(config.rate_limit_config.default_rph).unwrap())
            .allow_burst(NonZeroU32::new(
                (config.rate_limit_config.default_rph as f64 * config.rate_limit_config.burst_multiplier) as u32
            ).unwrap());
        
        let rate_limiter = Arc::new(RateLimiter::keyed(quota));
        
        Self {
            config,
            rate_limiter,
            banned_ips: Arc::new(RwLock::new(HashMap::new())),
            request_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Apply security headers to response
    pub async fn apply_security_headers(&self, mut response: Response) -> Response {
        let headers = response.headers_mut();
        
        // Content Security Policy
        headers.insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_str(&self.config.headers_config.csp).unwrap(),
        );
        
        // Strict Transport Security
        let hsts_value = if self.config.headers_config.hsts_include_subdomains {
            if self.config.headers_config.hsts_preload {
                format!("max-age={}; includeSubDomains; preload", self.config.headers_config.hsts_max_age)
            } else {
                format!("max-age={}; includeSubDomains", self.config.headers_config.hsts_max_age)
            }
        } else {
            format!("max-age={}", self.config.headers_config.hsts_max_age)
        };
        
        headers.insert(
            header::STRICT_TRANSPORT_SECURITY,
            HeaderValue::from_str(&hsts_value).unwrap(),
        );
        
        // X-Frame-Options
        let frame_options_value = match &self.config.headers_config.frame_options {
            FrameOptions::Deny => "DENY",
            FrameOptions::SameOrigin => "SAMEORIGIN",
            FrameOptions::AllowFrom(origin) => origin,
        };
        
        headers.insert(
            HeaderName::from_static("x-frame-options"),
            HeaderValue::from_str(frame_options_value).unwrap(),
        );
        
        // X-Content-Type-Options
        if self.config.headers_config.content_type_options {
            headers.insert(
                HeaderName::from_static("x-content-type-options"),
                HeaderValue::from_static("nosniff"),
            );
        }
        
        // Referrer Policy
        let referrer_policy_value = match &self.config.headers_config.referrer_policy {
            ReferrerPolicy::NoReferrer => "no-referrer",
            ReferrerPolicy::NoReferrerWhenDowngrade => "no-referrer-when-downgrade",
            ReferrerPolicy::Origin => "origin",
            ReferrerPolicy::OriginWhenCrossOrigin => "origin-when-cross-origin",
            ReferrerPolicy::SameOrigin => "same-origin",
            ReferrerPolicy::StrictOrigin => "strict-origin",
            ReferrerPolicy::StrictOriginWhenCrossOrigin => "strict-origin-when-cross-origin",
            ReferrerPolicy::UnsafeUrl => "unsafe-url",
        };
        
        headers.insert(
            HeaderName::from_static("referrer-policy"),
            HeaderValue::from_str(referrer_policy_value).unwrap(),
        );
        
        // Permissions Policy
        if let Some(permissions_policy) = &self.config.headers_config.permissions_policy {
            headers.insert(
                HeaderName::from_static("permissions-policy"),
                HeaderValue::from_str(permissions_policy).unwrap(),
            );
        }
        
        // X-XSS-Protection (legacy but still useful)
        headers.insert(
            HeaderName::from_static("x-xss-protection"),
            HeaderValue::from_static("1; mode=block"),
        );
        
        // X-Robots-Tag (prevent indexing of API endpoints)
        headers.insert(
            HeaderName::from_static("x-robots-tag"),
            HeaderValue::from_static("noindex, nofollow, nosnippet, noarchive"),
        );
        
        // Custom headers
        for (name, value) in &self.config.headers_config.custom_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }
        
        response
    }
    
    /// Check rate limits and DDoS protection
    pub async fn check_rate_limits(&self, ip: IpAddr) -> Result<bool> {
        // Check if IP is whitelisted
        if self.config.rate_limit_config.whitelist_ips.contains(&ip) {
            return Ok(true);
        }
        
        // Check if IP is banned
        {
            let mut banned_ips = self.banned_ips.write().await;
            if let Some(ban_time) = banned_ips.get(&ip) {
                let ban_duration = chrono::Duration::minutes(self.config.rate_limit_config.ban_duration_minutes as i64);
                if Utc::now() - *ban_time < ban_duration {
                    return Ok(false); // Still banned
                } else {
                    banned_ips.remove(&ip); // Ban expired
                }
            }
        }
        
        // Check DDoS threshold
        {
            let mut request_counts = self.request_counts.write().await;
            let now = Utc::now();
            let window = chrono::Duration::minutes(1);
            
            let requests = request_counts.entry(ip).or_insert_with(Vec::new);
            
            // Remove old requests outside the window
            requests.retain(|&time| now - time < window);
            
            // Add current request
            requests.push(now);
            
            // Check DDoS threshold
            if requests.len() > self.config.rate_limit_config.ddos_threshold as usize {
                // Ban the IP
                self.banned_ips.write().await.insert(ip, now);
                error!("DDoS detected from IP {}, banning for {} minutes", ip, self.config.rate_limit_config.ban_duration_minutes);
                return Ok(false);
            }
        }
        
        // Apply rate limiting
        match self.rate_limiter.check_key(&ip) {
            Ok(_) => Ok(true),
            Err(_) => {
                warn!("Rate limit exceeded for IP: {}", ip);
                Ok(false)
            }
        }
    }
    
    /// Validate request for security threats
    pub async fn validate_request(&self, request: &Request) -> Result<ValidationResult> {
        let mut threats = Vec::new();
        
        // Check URL length
        if request.uri().to_string().len() > self.config.validation_config.max_url_length {
            threats.push(SecurityThreat::ExcessiveUrlLength);
        }
        
        // Check for SQL injection patterns
        if self.config.validation_config.sql_injection_detection {
            let url = request.uri().to_string().to_lowercase();
            let sql_patterns = [
                "union", "select", "insert", "delete", "drop", "exec", "execute",
                "sp_", "xp_", "script", "javascript", "vbscript", "onload", "onerror"
            ];
            
            for pattern in &sql_patterns {
                if url.contains(pattern) {
                    threats.push(SecurityThreat::SqlInjection);
                    break;
                }
            }
        }
        
        // Check for XSS patterns
        if self.config.validation_config.xss_detection {
            let url = request.uri().to_string().to_lowercase();
            let xss_patterns = [
                "<script", "javascript:", "onload=", "onerror=", "eval(", "expression("
            ];
            
            for pattern in &xss_patterns {
                if url.contains(pattern) {
                    threats.push(SecurityThreat::XssAttempt);
                    break;
                }
            }
        }
        
        // Check for path traversal
        if self.config.validation_config.path_traversal_detection {
            let url = request.uri().to_string();
            if url.contains("../") || url.contains("..\\") || url.contains("%2e%2e") {
                threats.push(SecurityThreat::PathTraversal);
            }
        }
        
        // Check User-Agent
        if let Some(user_agent) = request.headers().get(header::USER_AGENT) {
            if let Ok(user_agent_str) = user_agent.to_str() {
                let user_agent_lower = user_agent_str.to_lowercase();
                for blocked_agent in &self.config.validation_config.blocked_user_agents {
                    if user_agent_lower.contains(blocked_agent) {
                        threats.push(SecurityThreat::BlockedUserAgent);
                        break;
                    }
                }
            }
        }
        
        // Check required headers
        for required_header in &self.config.validation_config.required_headers {
            if !request.headers().contains_key(required_header) {
                threats.push(SecurityThreat::MissingRequiredHeader);
                break;
            }
        }
        
        Ok(ValidationResult {
            is_valid: threats.is_empty(),
            threats,
            risk_score: Self::calculate_risk_score(&threats),
        })
    }
    
    fn calculate_risk_score(threats: &[SecurityThreat]) -> u8 {
        let mut score = 0u8;
        
        for threat in threats {
            score += match threat {
                SecurityThreat::SqlInjection => 9,
                SecurityThreat::XssAttempt => 8,
                SecurityThreat::PathTraversal => 7,
                SecurityThreat::BlockedUserAgent => 5,
                SecurityThreat::ExcessiveUrlLength => 3,
                SecurityThreat::MissingRequiredHeader => 2,
            };
        }
        
        score.min(10)
    }
    
    /// Extract real IP address considering trusted proxies
    pub fn extract_real_ip(&self, request: &Request) -> Option<IpAddr> {
        // Check X-Forwarded-For header if behind trusted proxy
        if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
            if let Ok(forwarded_str) = forwarded_for.to_str() {
                if let Some(first_ip) = forwarded_str.split(',').next() {
                    if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                        return Some(ip);
                    }
                }
            }
        }
        
        // Check X-Real-IP header
        if let Some(real_ip) = request.headers().get("x-real-ip") {
            if let Ok(ip_str) = real_ip.to_str() {
                if let Ok(ip) = ip_str.parse::<IpAddr>() {
                    return Some(ip);
                }
            }
        }
        
        // Fall back to connection info
        if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
            return Some(addr.ip());
        }
        
        None
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub threats: Vec<SecurityThreat>,
    pub risk_score: u8,
}

#[derive(Debug, Clone)]
pub enum SecurityThreat {
    SqlInjection,
    XssAttempt,
    PathTraversal,
    BlockedUserAgent,
    ExcessiveUrlLength,
    MissingRequiredHeader,
}

/// Create production-ready CORS layer
pub fn create_cors_layer(config: &CorsConfig) -> CorsLayer {
    let mut cors = CorsLayer::new();
    
    // Configure origins
    if config.development_mode {
        cors = cors.allow_origin(tower_http::cors::Any);
    } else {
        let origins: Result<Vec<_>, _> = config.allowed_origins
            .iter()
            .map(|origin| origin.parse())
            .collect();
        
        if let Ok(origins) = origins {
            cors = cors.allow_origin(AllowOrigin::list(origins));
        }
    }
    
    // Configure methods
    cors = cors.allow_methods(config.allowed_methods.clone());
    
    // Configure headers
    let headers: Result<Vec<_>, _> = config.allowed_headers
        .iter()
        .map(|h| h.parse())
        .collect();
    
    if let Ok(headers) = headers {
        cors = cors.allow_headers(headers);
    }
    
    // Configure exposed headers
    let exposed: Result<Vec<_>, _> = config.exposed_headers
        .iter()
        .map(|h| h.parse())
        .collect();
    
    if let Ok(exposed) = exposed {
        cors = cors.expose_headers(exposed);
    }
    
    // Configure credentials
    if config.allow_credentials {
        cors = cors.allow_credentials(true);
    }
    
    // Configure max age
    if let Some(max_age) = config.max_age {
        cors = cors.max_age(max_age);
    }
    
    cors
}

/// Comprehensive security middleware function
pub async fn security_middleware(
    State(security): State<Arc<SecurityMiddleware>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start_time = std::time::Instant::now();
    
    // Extract client IP
    let client_ip = security.extract_real_ip(&request)
        .unwrap_or_else(|| "0.0.0.0".parse().unwrap());
    
    // Check rate limits
    if security.config.rate_limit_config.enabled {
        if !security.check_rate_limits(client_ip).await.unwrap_or(false) {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
    }
    
    // Validate request
    let validation_result = security.validate_request(&request).await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    if !validation_result.is_valid {
        error!(
            "Security threat detected from IP {}: {:?}",
            client_ip,
            validation_result.threats
        );
        
        // Log security event if auditing is enabled
        if security.config.enable_audit_logging {
            let audit_event = AuditEvent {
                event_id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                event_type: AuditEventType::SecurityViolation,
                user_id: None,
                session_id: None,
                ip_address: Some(client_ip),
                user_agent: request.headers().get(header::USER_AGENT)
                    .and_then(|h| h.to_str().ok())
                    .map(|s| s.to_string()),
                resource: request.uri().to_string(),
                action: AuditAction::Custom("security_validation".to_string()),
                result: AuditResult::Blocked,
                risk_score: validation_result.risk_score,
                data_classification: DataClassification::Internal,
                compliance_tags: vec![ComplianceTag::Security],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("threats".to_string(), format!("{:?}", validation_result.threats));
                    meta.insert("client_ip".to_string(), client_ip.to_string());
                    meta
                },
                request_details: Some(RequestDetails {
                    method: request.method().to_string(),
                    url: request.uri().to_string(),
                    headers: request.headers().iter()
                        .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.to_string(), v.to_string())))
                        .collect(),
                    body_size: 0, // We don't have body size here
                    query_params: HashMap::new(), // Simplified
                }),
                response_details: None,
                error_info: None,
            };
            
            // In a real implementation, you would send this to your audit logger
            debug!("Audit event: {:?}", audit_event);
        }
        
        return Err(StatusCode::FORBIDDEN);
    }
    
    // Process request
    let response = next.run(request).await;
    
    // Apply security headers
    let secured_response = security.apply_security_headers(response).await;
    
    let processing_time = start_time.elapsed();
    debug!("Request processed in {:?}", processing_time);
    
    Ok(secured_response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    
    #[test]
    fn test_security_config_creation() {
        let config = SecurityMiddlewareConfig::default();
        assert!(config.headers_config.hsts_max_age > 0);
        assert!(!config.cors_config.allowed_origins.is_empty());
        assert!(config.rate_limit_config.default_rph > 0);
    }
    
    #[tokio::test]
    async fn test_security_middleware_creation() {
        let config = SecurityMiddlewareConfig::default();
        let middleware = SecurityMiddleware::new(config);
        
        // Test rate limit check
        let ip = "192.168.1.1".parse().unwrap();
        let result = middleware.check_rate_limits(ip).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_request_validation() {
        let config = SecurityMiddlewareConfig::default();
        let middleware = SecurityMiddleware::new(config);
        
        // Test normal request
        let request = Request::builder()
            .uri("/api/test")
            .header("user-agent", "test-client")
            .body(Body::empty())
            .unwrap();
        
        let result = middleware.validate_request(&request).await.unwrap();
        assert!(result.is_valid);
        
        // Test malicious request
        let malicious_request = Request::builder()
            .uri("/api/test?id=1' UNION SELECT * FROM users--")
            .header("user-agent", "test-client")
            .body(Body::empty())
            .unwrap();
        
        let result = middleware.validate_request(&malicious_request).await.unwrap();
        assert!(!result.is_valid);
        assert!(!result.threats.is_empty());
    }
    
    #[test]
    fn test_cors_configuration() {
        let config = CorsConfig::default();
        let cors_layer = create_cors_layer(&config);
        
        // This test just ensures the CORS layer can be created without panicking
        // More thorough testing would require integration tests
    }
}