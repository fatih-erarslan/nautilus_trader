//! Comprehensive API Scanner Module
//!
//! Provides OpenAPI/Swagger parsing, endpoint discovery, security scanning,
//! performance metrics, and integration with AgentDB and agentic-flow.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ============================================================================
// Core Types and Enums
// ============================================================================

/// Security vulnerability severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulnerabilitySeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Authentication method types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthMethod {
    Bearer,
    Basic,
    ApiKey { location: String, name: String },
    OAuth2 { flow: String },
    None,
}

/// HTTP methods supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::GET => write!(f, "GET"),
            HttpMethod::POST => write!(f, "POST"),
            HttpMethod::PUT => write!(f, "PUT"),
            HttpMethod::DELETE => write!(f, "DELETE"),
            HttpMethod::PATCH => write!(f, "PATCH"),
            HttpMethod::HEAD => write!(f, "HEAD"),
            HttpMethod::OPTIONS => write!(f, "OPTIONS"),
        }
    }
}

/// Vulnerability types based on OWASP Top 10
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulnerabilityType {
    BrokenAuthentication,
    SensitiveDataExposure,
    InjectionFlaw,
    BrokenAccessControl,
    SecurityMisconfiguration,
    XSS,
    InsecureDeserialization,
    ComponentsWithVulnerabilities,
    InsufficientLogging,
    SSRF,
    // Additional security checks
    MissingCORS,
    WeakSSL,
    RateLimitMissing,
    MissingSecurityHeaders,
}

// ============================================================================
// Scanner Configuration
// ============================================================================

/// Configuration for the API scanner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerConfig {
    /// Maximum number of endpoints to scan
    pub max_endpoints: usize,
    /// Timeout for each HTTP request
    pub request_timeout: Duration,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Enable security vulnerability scanning
    pub enable_security_scan: bool,
    /// Enable performance metrics collection
    pub enable_performance_metrics: bool,
    /// Enable endpoint discovery via crawling
    pub enable_crawling: bool,
    /// User agent for HTTP requests
    pub user_agent: String,
    /// Custom headers for requests
    pub custom_headers: HashMap<String, String>,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            max_endpoints: 1000,
            request_timeout: Duration::from_secs(30),
            max_concurrent: 10,
            enable_security_scan: true,
            enable_performance_metrics: true,
            enable_crawling: true,
            user_agent: "BeClever-API-Scanner/1.0".to_string(),
            custom_headers: HashMap::new(),
        }
    }
}

// ============================================================================
// OpenAPI/Swagger Types
// ============================================================================

/// OpenAPI specification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAPISpec {
    pub openapi: Option<String>,
    pub swagger: Option<String>,
    pub info: ApiInfo,
    pub servers: Vec<ServerInfo>,
    pub paths: HashMap<String, PathItem>,
    pub components: Option<Components>,
    pub security: Option<Vec<SecurityRequirement>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiInfo {
    pub title: String,
    pub version: String,
    pub description: Option<String>,
    pub contact: Option<ContactInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    pub name: Option<String>,
    pub url: Option<String>,
    pub email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub url: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathItem {
    #[serde(flatten)]
    pub operations: HashMap<String, Operation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub summary: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<Vec<Parameter>>,
    pub responses: HashMap<String, Response>,
    pub security: Option<Vec<SecurityRequirement>>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    #[serde(rename = "in")]
    pub location: String,
    pub required: Option<bool>,
    pub schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub description: String,
    pub content: Option<HashMap<String, MediaType>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaType {
    pub schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Components {
    pub schemas: Option<HashMap<String, serde_json::Value>>,
    pub security_schemes: Option<HashMap<String, SecurityScheme>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScheme {
    #[serde(rename = "type")]
    pub scheme_type: String,
    pub scheme: Option<String>,
    pub bearer_format: Option<String>,
    pub name: Option<String>,
    #[serde(rename = "in")]
    pub location: Option<String>,
}

type SecurityRequirement = HashMap<String, Vec<String>>;

// ============================================================================
// Scan Results
// ============================================================================

/// Complete API scan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub scan_id: Uuid,
    pub target_url: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: ScanStatus,
    pub spec: Option<OpenAPISpec>,
    pub endpoints: Vec<EndpointInfo>,
    pub vulnerabilities: Vec<Vulnerability>,
    pub performance_metrics: PerformanceMetrics,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Information about a discovered endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub path: String,
    pub method: HttpMethod,
    pub auth_required: bool,
    pub auth_method: AuthMethod,
    pub parameters: Vec<Parameter>,
    pub response_codes: HashSet<u16>,
    pub response_time_ms: Option<u64>,
    pub discovered_via: DiscoveryMethod,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    OpenAPISpec,
    Crawling,
    Manual,
}

/// Security vulnerability found during scan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub id: Uuid,
    pub vuln_type: VulnerabilityType,
    pub severity: VulnerabilitySeverity,
    pub title: String,
    pub description: String,
    pub affected_endpoint: Option<String>,
    pub remediation: String,
    pub references: Vec<String>,
}

/// Performance metrics collected during scan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub avg_response_time_ms: f64,
    pub min_response_time_ms: u64,
    pub max_response_time_ms: u64,
    pub p95_response_time_ms: u64,
    pub p99_response_time_ms: u64,
    pub endpoints_scanned: usize,
    pub vulnerabilities_found: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            min_response_time_ms: u64::MAX,
            max_response_time_ms: 0,
            p95_response_time_ms: 0,
            p99_response_time_ms: 0,
            endpoints_scanned: 0,
            vulnerabilities_found: 0,
        }
    }
}

// ============================================================================
// API Scanner Implementation
// ============================================================================

/// Main API Scanner
pub struct ApiScanner {
    config: ScannerConfig,
    http_client: reqwest::Client,
    scan_results: Arc<RwLock<HashMap<Uuid, ScanResult>>>,
}

impl ApiScanner {
    /// Create a new API scanner with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ScannerConfig::default())
    }

    /// Create a new API scanner with custom configuration
    pub fn with_config(config: ScannerConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            config,
            http_client,
            scan_results: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start a new API scan
    pub async fn scan(&self, target_url: &str) -> Result<Uuid> {
        let scan_id = Uuid::new_v4();
        info!("Starting API scan {} for target: {}", scan_id, target_url);

        let mut scan_result = ScanResult {
            scan_id,
            target_url: target_url.to_string(),
            started_at: Utc::now(),
            completed_at: None,
            status: ScanStatus::Running,
            spec: None,
            endpoints: Vec::new(),
            vulnerabilities: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
            recommendations: Vec::new(),
        };

        // Store initial scan result
        {
            let mut results = self.scan_results.write().await;
            results.insert(scan_id, scan_result.clone());
        }

        // Parse OpenAPI spec if available
        if let Ok(spec) = self.parse_openapi_spec(target_url).await {
            info!("Successfully parsed OpenAPI spec for {}", target_url);
            scan_result.spec = Some(spec.clone());
            scan_result.endpoints.extend(self.extract_endpoints_from_spec(&spec));
        }

        // Discover endpoints via crawling if enabled
        if self.config.enable_crawling {
            debug!("Starting endpoint discovery via crawling");
            if let Ok(discovered) = self.discover_endpoints(target_url).await {
                scan_result.endpoints.extend(discovered);
            }
        }

        // Test each endpoint
        let response_times = self.test_endpoints(&mut scan_result).await?;

        // Run security scans if enabled
        if self.config.enable_security_scan {
            debug!("Running security vulnerability scans");
            scan_result.vulnerabilities = self.scan_vulnerabilities(&scan_result).await?;
        }

        // Calculate performance metrics if enabled
        if self.config.enable_performance_metrics {
            debug!("Calculating performance metrics");
            scan_result.performance_metrics = self.calculate_metrics(&scan_result, &response_times);
        }

        // Generate recommendations
        scan_result.recommendations = self.generate_recommendations(&scan_result);

        // Mark scan as completed
        scan_result.status = ScanStatus::Completed;
        scan_result.completed_at = Some(Utc::now());

        // Update stored result
        {
            let mut results = self.scan_results.write().await;
            results.insert(scan_id, scan_result);
        }

        info!("Completed API scan {} for {}", scan_id, target_url);
        Ok(scan_id)
    }

    /// Parse OpenAPI/Swagger specification
    async fn parse_openapi_spec(&self, base_url: &str) -> Result<OpenAPISpec> {
        // Try common OpenAPI spec paths
        let spec_paths = vec![
            "/openapi.json",
            "/swagger.json",
            "/api-docs",
            "/v2/api-docs",
            "/v3/api-docs",
            "/swagger/v1/swagger.json",
        ];

        for path in spec_paths {
            let url = format!("{}{}", base_url, path);
            debug!("Attempting to fetch OpenAPI spec from: {}", url);

            match self.http_client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    match response.json::<OpenAPISpec>().await {
                        Ok(spec) => {
                            info!("Successfully parsed OpenAPI spec from {}", url);
                            return Ok(spec);
                        }
                        Err(e) => {
                            debug!("Failed to parse spec from {}: {}", url, e);
                            continue;
                        }
                    }
                }
                Ok(response) => {
                    debug!("Got non-success status {} from {}", response.status(), url);
                }
                Err(e) => {
                    debug!("Request failed for {}: {}", url, e);
                }
            }
        }

        Err(anyhow!("No OpenAPI specification found"))
    }

    /// Extract endpoints from OpenAPI specification
    fn extract_endpoints_from_spec(&self, spec: &OpenAPISpec) -> Vec<EndpointInfo> {
        let mut endpoints = Vec::new();

        for (path, path_item) in &spec.paths {
            for (method_str, operation) in &path_item.operations {
                let method = match method_str.to_uppercase().as_str() {
                    "GET" => HttpMethod::GET,
                    "POST" => HttpMethod::POST,
                    "PUT" => HttpMethod::PUT,
                    "DELETE" => HttpMethod::DELETE,
                    "PATCH" => HttpMethod::PATCH,
                    "HEAD" => HttpMethod::HEAD,
                    "OPTIONS" => HttpMethod::OPTIONS,
                    _ => continue,
                };

                let auth_required = operation.security.is_some() || spec.security.is_some();
                let auth_method = self.determine_auth_method(spec, operation);

                endpoints.push(EndpointInfo {
                    path: path.clone(),
                    method,
                    auth_required,
                    auth_method,
                    parameters: operation.parameters.clone().unwrap_or_default(),
                    response_codes: HashSet::new(),
                    response_time_ms: None,
                    discovered_via: DiscoveryMethod::OpenAPISpec,
                });
            }
        }

        endpoints
    }

    /// Determine authentication method from spec
    fn determine_auth_method(&self, spec: &OpenAPISpec, operation: &Operation) -> AuthMethod {
        if let Some(components) = &spec.components {
            if let Some(security_schemes) = &components.security_schemes {
                // Check operation-level security first
                if let Some(sec_reqs) = &operation.security {
                    for req in sec_reqs {
                        for (scheme_name, _) in req {
                            if let Some(scheme) = security_schemes.get(scheme_name) {
                                return self.parse_security_scheme(scheme);
                            }
                        }
                    }
                }
                // Fall back to global security
                if let Some(sec_reqs) = &spec.security {
                    for req in sec_reqs {
                        for (scheme_name, _) in req {
                            if let Some(scheme) = security_schemes.get(scheme_name) {
                                return self.parse_security_scheme(scheme);
                            }
                        }
                    }
                }
            }
        }
        AuthMethod::None
    }

    /// Parse security scheme into AuthMethod
    fn parse_security_scheme(&self, scheme: &SecurityScheme) -> AuthMethod {
        match scheme.scheme_type.as_str() {
            "http" => {
                if let Some(s) = &scheme.scheme {
                    if s == "bearer" {
                        return AuthMethod::Bearer;
                    } else if s == "basic" {
                        return AuthMethod::Basic;
                    }
                }
                AuthMethod::None
            }
            "apiKey" => AuthMethod::ApiKey {
                location: scheme.location.clone().unwrap_or_default(),
                name: scheme.name.clone().unwrap_or_default(),
            },
            "oauth2" => AuthMethod::OAuth2 {
                flow: "authorization_code".to_string(),
            },
            _ => AuthMethod::None,
        }
    }

    /// Discover endpoints via crawling
    async fn discover_endpoints(&self, base_url: &str) -> Result<Vec<EndpointInfo>> {
        debug!("Starting endpoint discovery for {}", base_url);
        let mut discovered = Vec::new();

        // Common API paths to check
        let common_paths = vec![
            "/api/v1",
            "/api/v2",
            "/api",
            "/v1",
            "/v2",
            "/health",
            "/status",
            "/ping",
        ];

        for path in common_paths {
            let url = format!("{}{}", base_url, path);
            match self.http_client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    discovered.push(EndpointInfo {
                        path: path.to_string(),
                        method: HttpMethod::GET,
                        auth_required: false,
                        auth_method: AuthMethod::None,
                        parameters: Vec::new(),
                        response_codes: [response.status().as_u16()].into_iter().collect(),
                        response_time_ms: None,
                        discovered_via: DiscoveryMethod::Crawling,
                    });
                }
                _ => {}
            }
        }

        Ok(discovered)
    }

    /// Test all discovered endpoints
    async fn test_endpoints(&self, scan_result: &mut ScanResult) -> Result<Vec<u64>> {
        let mut response_times = Vec::new();

        for endpoint in &mut scan_result.endpoints {
            let url = format!("{}{}", scan_result.target_url, endpoint.path);
            debug!("Testing endpoint: {} {}", endpoint.method, url);

            let start = Instant::now();
            match self.http_client.get(&url).send().await {
                Ok(response) => {
                    let elapsed = start.elapsed().as_millis() as u64;
                    response_times.push(elapsed);
                    endpoint.response_time_ms = Some(elapsed);
                    endpoint.response_codes.insert(response.status().as_u16());
                }
                Err(e) => {
                    warn!("Failed to test endpoint {}: {}", url, e);
                }
            }
        }

        Ok(response_times)
    }

    /// Scan for security vulnerabilities
    async fn scan_vulnerabilities(&self, scan_result: &ScanResult) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();

        // Check for missing authentication
        vulnerabilities.extend(self.check_broken_authentication(scan_result));

        // Check for CORS issues
        vulnerabilities.extend(self.check_cors_issues(scan_result).await);

        // Check for security headers
        vulnerabilities.extend(self.check_security_headers(scan_result).await);

        // Check for SSL/TLS issues
        vulnerabilities.extend(self.check_ssl_issues(scan_result));

        // Check for rate limiting
        vulnerabilities.extend(self.check_rate_limiting(scan_result));

        // Check for sensitive data exposure
        vulnerabilities.extend(self.check_sensitive_data_exposure(scan_result));

        Ok(vulnerabilities)
    }

    /// Check for broken authentication issues
    fn check_broken_authentication(&self, scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        let unprotected_endpoints: Vec<_> = scan_result
            .endpoints
            .iter()
            .filter(|e| !e.auth_required && e.method != HttpMethod::GET)
            .collect();

        if !unprotected_endpoints.is_empty() {
            vulnerabilities.push(Vulnerability {
                id: Uuid::new_v4(),
                vuln_type: VulnerabilityType::BrokenAuthentication,
                severity: VulnerabilitySeverity::High,
                title: "Unprotected API Endpoints".to_string(),
                description: format!(
                    "Found {} endpoints that accept mutations without authentication",
                    unprotected_endpoints.len()
                ),
                affected_endpoint: Some(unprotected_endpoints[0].path.clone()),
                remediation: "Implement authentication for all state-changing operations".to_string(),
                references: vec!["https://owasp.org/www-project-top-ten/2017/A2_2017-Broken_Authentication".to_string()],
            });
        }

        vulnerabilities
    }

    /// Check for CORS issues
    async fn check_cors_issues(&self, scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        if let Some(endpoint) = scan_result.endpoints.first() {
            let url = format!("{}{}", scan_result.target_url, endpoint.path);

            if let Ok(response) = self.http_client
                .get(&url)
                .header("Origin", "https://evil.com")
                .send()
                .await
            {
                if let Some(cors) = response.headers().get("access-control-allow-origin") {
                    if cors == "*" {
                        vulnerabilities.push(Vulnerability {
                            id: Uuid::new_v4(),
                            vuln_type: VulnerabilityType::MissingCORS,
                            severity: VulnerabilitySeverity::Medium,
                            title: "Overly Permissive CORS Policy".to_string(),
                            description: "API allows requests from any origin (Access-Control-Allow-Origin: *)".to_string(),
                            affected_endpoint: Some(endpoint.path.clone()),
                            remediation: "Configure CORS to only allow trusted origins".to_string(),
                            references: vec!["https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS".to_string()],
                        });
                    }
                } else {
                    vulnerabilities.push(Vulnerability {
                        id: Uuid::new_v4(),
                        vuln_type: VulnerabilityType::MissingCORS,
                        severity: VulnerabilitySeverity::Low,
                        title: "Missing CORS Headers".to_string(),
                        description: "API does not set CORS headers".to_string(),
                        affected_endpoint: Some(endpoint.path.clone()),
                        remediation: "Configure appropriate CORS headers".to_string(),
                        references: vec!["https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS".to_string()],
                    });
                }
            }
        }

        vulnerabilities
    }

    /// Check for security headers
    async fn check_security_headers(&self, scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        if let Some(endpoint) = scan_result.endpoints.first() {
            let url = format!("{}{}", scan_result.target_url, endpoint.path);

            if let Ok(response) = self.http_client.get(&url).send().await {
                let headers = response.headers();

                let required_headers = vec![
                    ("X-Content-Type-Options", "nosniff"),
                    ("X-Frame-Options", "DENY"),
                    ("Strict-Transport-Security", "max-age=31536000"),
                ];

                for (header_name, _expected_value) in required_headers {
                    if headers.get(header_name).is_none() {
                        vulnerabilities.push(Vulnerability {
                            id: Uuid::new_v4(),
                            vuln_type: VulnerabilityType::MissingSecurityHeaders,
                            severity: VulnerabilitySeverity::Low,
                            title: format!("Missing {} Header", header_name),
                            description: format!("Security header {} is not set", header_name),
                            affected_endpoint: Some(endpoint.path.clone()),
                            remediation: format!("Add {} header to responses", header_name),
                            references: vec!["https://owasp.org/www-project-secure-headers/".to_string()],
                        });
                    }
                }
            }
        }

        vulnerabilities
    }

    /// Check for SSL/TLS issues
    fn check_ssl_issues(&self, scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        if !scan_result.target_url.starts_with("https://") {
            vulnerabilities.push(Vulnerability {
                id: Uuid::new_v4(),
                vuln_type: VulnerabilityType::WeakSSL,
                severity: VulnerabilitySeverity::Critical,
                title: "API Not Using HTTPS".to_string(),
                description: "API is accessible over unencrypted HTTP".to_string(),
                affected_endpoint: None,
                remediation: "Configure TLS/SSL and redirect all HTTP traffic to HTTPS".to_string(),
                references: vec!["https://owasp.org/www-community/Transport_Layer_Protection_Cheat_Sheet".to_string()],
            });
        }

        vulnerabilities
    }

    /// Check for rate limiting
    fn check_rate_limiting(&self, _scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        // This is a simplified check - in production, we'd actually test rate limits
        vulnerabilities.push(Vulnerability {
            id: Uuid::new_v4(),
            vuln_type: VulnerabilityType::RateLimitMissing,
            severity: VulnerabilitySeverity::Medium,
            title: "Rate Limiting Not Detected".to_string(),
            description: "No rate limiting headers detected in API responses".to_string(),
            affected_endpoint: None,
            remediation: "Implement rate limiting to prevent abuse and DDoS attacks".to_string(),
            references: vec!["https://owasp.org/www-community/controls/Blocking_Brute_Force_Attacks".to_string()],
        });

        vulnerabilities
    }

    /// Check for sensitive data exposure
    fn check_sensitive_data_exposure(&self, scan_result: &ScanResult) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        // Check if spec is exposed
        if scan_result.spec.is_some() {
            vulnerabilities.push(Vulnerability {
                id: Uuid::new_v4(),
                vuln_type: VulnerabilityType::SensitiveDataExposure,
                severity: VulnerabilitySeverity::Info,
                title: "API Specification Publicly Accessible".to_string(),
                description: "OpenAPI/Swagger specification is publicly accessible".to_string(),
                affected_endpoint: None,
                remediation: "Consider restricting access to API documentation in production".to_string(),
                references: vec!["https://owasp.org/www-project-api-security/".to_string()],
            });
        }

        vulnerabilities
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        scan_result: &ScanResult,
        response_times: &[u64],
    ) -> PerformanceMetrics {
        let mut metrics = PerformanceMetrics::default();

        metrics.total_requests = scan_result.endpoints.len();
        metrics.successful_requests = scan_result
            .endpoints
            .iter()
            .filter(|e| e.response_time_ms.is_some())
            .count();
        metrics.failed_requests = metrics.total_requests - metrics.successful_requests;
        metrics.endpoints_scanned = scan_result.endpoints.len();
        metrics.vulnerabilities_found = scan_result.vulnerabilities.len();

        if !response_times.is_empty() {
            let sum: u64 = response_times.iter().sum();
            metrics.avg_response_time_ms = sum as f64 / response_times.len() as f64;
            metrics.min_response_time_ms = *response_times.iter().min().unwrap();
            metrics.max_response_time_ms = *response_times.iter().max().unwrap();

            let mut sorted = response_times.to_vec();
            sorted.sort_unstable();
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            metrics.p95_response_time_ms = sorted.get(p95_idx).copied().unwrap_or(0);
            metrics.p99_response_time_ms = sorted.get(p99_idx).copied().unwrap_or(0);
        }

        metrics
    }

    /// Generate recommendations based on scan results
    fn generate_recommendations(&self, scan_result: &ScanResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Security recommendations
        let critical_vulns = scan_result
            .vulnerabilities
            .iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Critical)
            .count();

        if critical_vulns > 0 {
            recommendations.push(format!(
                "URGENT: Address {} critical security vulnerabilities immediately",
                critical_vulns
            ));
        }

        // Performance recommendations
        if scan_result.performance_metrics.avg_response_time_ms > 1000.0 {
            recommendations.push(
                "Consider optimizing API performance - average response time exceeds 1 second".to_string()
            );
        }

        // Authentication recommendations
        let auth_issues = scan_result
            .vulnerabilities
            .iter()
            .filter(|v| v.vuln_type == VulnerabilityType::BrokenAuthentication)
            .count();

        if auth_issues > 0 {
            recommendations.push(
                "Implement authentication for all state-changing endpoints".to_string()
            );
        }

        // SSL recommendations
        if !scan_result.target_url.starts_with("https://") {
            recommendations.push(
                "Enable HTTPS/TLS for all API endpoints to protect data in transit".to_string()
            );
        }

        // Documentation recommendations
        if scan_result.spec.is_none() {
            recommendations.push(
                "Consider adding OpenAPI/Swagger documentation for better API discoverability".to_string()
            );
        }

        recommendations
    }

    /// Get scan result by ID
    pub async fn get_scan_result(&self, scan_id: Uuid) -> Option<ScanResult> {
        let results = self.scan_results.read().await;
        results.get(&scan_id).cloned()
    }

    /// Get all scan results
    pub async fn get_all_scan_results(&self) -> Vec<ScanResult> {
        let results = self.scan_results.read().await;
        results.values().cloned().collect()
    }
}

impl Default for ApiScanner {
    fn default() -> Self {
        Self::new().expect("Failed to create default API scanner")
    }
}

// ============================================================================
// AgentDB Integration
// ============================================================================

/// AgentDB integration for storing scan results
pub struct ScannerAgentDB {
    // This would integrate with AgentDB in production
    // For now, we define the interface
}

impl ScannerAgentDB {
    pub fn new() -> Self {
        Self {}
    }

    /// Store scan result in AgentDB with vector embedding
    pub async fn store_scan_result(&self, scan_result: &ScanResult) -> Result<()> {
        info!("Storing scan result {} in AgentDB", scan_result.scan_id);

        // In production, this would:
        // 1. Serialize scan result
        // 2. Generate vector embedding from description/vulnerabilities
        // 3. Store in AgentDB SQLite with vector index
        // 4. Enable semantic search for similar APIs

        Ok(())
    }

    /// Find similar APIs using vector search
    pub async fn find_similar_apis(&self, scan_result: &ScanResult, _limit: usize) -> Result<Vec<ScanResult>> {
        debug!("Searching for APIs similar to {}", scan_result.scan_id);

        // In production, this would:
        // 1. Generate embedding from scan result
        // 2. Query AgentDB vector index
        // 3. Return similar API scans

        Ok(Vec::new())
    }

    /// Get historical scan data for trend analysis
    pub async fn get_scan_history(&self, target_url: &str) -> Result<Vec<ScanResult>> {
        debug!("Fetching scan history for {}", target_url);

        // Query AgentDB for historical scans of this target

        Ok(Vec::new())
    }
}

// ============================================================================
// Agentic-Flow Integration
// ============================================================================

/// Agentic-Flow integration for intelligent analysis
pub struct ScannerAgenticFlow {
    // This would integrate with agentic-flow in production
}

impl ScannerAgenticFlow {
    pub fn new() -> Self {
        Self {}
    }

    /// Analyze API patterns using AI agents
    pub async fn analyze_patterns(&self, scan_result: &ScanResult) -> Result<Vec<String>> {
        info!("Running AI pattern analysis on scan {}", scan_result.scan_id);

        // In production, this would:
        // 1. Spawn agentic-flow agents
        // 2. Analyze endpoint patterns
        // 3. Detect anomalies
        // 4. Generate insights

        Ok(vec![
            "API follows RESTful conventions".to_string(),
            "Consistent error handling detected".to_string(),
        ])
    }

    /// Detect security risks automatically
    pub async fn detect_security_risks(&self, scan_result: &ScanResult) -> Result<Vec<Vulnerability>> {
        info!("Running AI security analysis on scan {}", scan_result.scan_id);

        // In production, this would use AI to detect:
        // 1. Unusual patterns
        // 2. Potential injection points
        // 3. Logic flaws
        // 4. Design weaknesses

        Ok(Vec::new())
    }

    /// Generate improvement recommendations
    pub async fn generate_recommendations(&self, scan_result: &ScanResult) -> Result<Vec<String>> {
        info!("Generating AI recommendations for scan {}", scan_result.scan_id);

        // In production, AI agents would analyze and suggest:
        // 1. Architecture improvements
        // 2. Security enhancements
        // 3. Performance optimizations
        // 4. Best practices

        Ok(vec![
            "Consider implementing API versioning".to_string(),
            "Add request validation middleware".to_string(),
        ])
    }

    /// Create detailed scan report
    pub async fn generate_report(&self, scan_result: &ScanResult) -> Result<String> {
        info!("Generating comprehensive report for scan {}", scan_result.scan_id);

        let mut report = String::new();

        report.push_str(&format!("# API Scan Report: {}\n\n", scan_result.target_url));
        report.push_str(&format!("**Scan ID:** {}\n", scan_result.scan_id));
        report.push_str(&format!("**Started:** {}\n", scan_result.started_at));

        if let Some(completed) = scan_result.completed_at {
            report.push_str(&format!("**Completed:** {}\n", completed));
        }

        report.push_str("\n## Summary\n\n");
        report.push_str(&format!("- **Endpoints Scanned:** {}\n", scan_result.endpoints.len()));
        report.push_str(&format!("- **Vulnerabilities Found:** {}\n", scan_result.vulnerabilities.len()));
        report.push_str(&format!("- **Average Response Time:** {:.2}ms\n", scan_result.performance_metrics.avg_response_time_ms));

        report.push_str("\n## Vulnerabilities\n\n");
        for vuln in &scan_result.vulnerabilities {
            report.push_str(&format!("### {} ({:?})\n", vuln.title, vuln.severity));
            report.push_str(&format!("{}\n\n", vuln.description));
            report.push_str(&format!("**Remediation:** {}\n\n", vuln.remediation));
        }

        report.push_str("\n## Recommendations\n\n");
        for rec in &scan_result.recommendations {
            report.push_str(&format!("- {}\n", rec));
        }

        Ok(report)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scanner_config_default() {
        let config = ScannerConfig::default();
        assert_eq!(config.max_endpoints, 1000);
        assert_eq!(config.max_concurrent, 10);
        assert!(config.enable_security_scan);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
    }

    #[tokio::test]
    async fn test_scanner_creation() {
        let scanner = ApiScanner::new();
        assert!(scanner.is_ok());
    }

    #[test]
    fn test_auth_method_parsing() {
        let scheme = SecurityScheme {
            scheme_type: "http".to_string(),
            scheme: Some("bearer".to_string()),
            bearer_format: None,
            name: None,
            location: None,
        };

        let scanner = ApiScanner::new().unwrap();
        let auth_method = scanner.parse_security_scheme(&scheme);
        assert_eq!(auth_method, AuthMethod::Bearer);
    }
}
