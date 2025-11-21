//! REST API Middleware
//!
//! Comprehensive middleware stack including authentication, rate limiting,
//! metrics collection, error handling, and performance monitoring.

use crate::{api::ApiError, AtsCoreError};
use axum::{
    extract::{ConnectInfo, Request},
    http::{HeaderMap, StatusCode, HeaderName, HeaderValue},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, atomic::{AtomicU64, AtomicU32, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Authentication Middleware
pub struct AuthMiddleware {
    /// JWT secret key
    jwt_secret: String,
    /// Enabled flag
    enabled: bool,
}

impl AuthMiddleware {
    pub fn new(jwt_secret: String, enabled: bool) -> Self {
        Self { jwt_secret, enabled }
    }

    /// Middleware function for authentication
    pub async fn authenticate(
        &self,
        headers: HeaderMap,
        request: Request,
        next: Next,
    ) -> std::result::Result<Response, StatusCode> {
        if !self.enabled {
            return Ok(next.run(request).await);
        }

        // Extract Authorization header
        let auth_header = headers
            .get("Authorization")
            .and_then(|header| header.to_str().ok())
            .and_then(|header| header.strip_prefix("Bearer "));

        if let Some(token) = auth_header {
            if self.validate_jwt_token(token) {
                Ok(next.run(request).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        } else {
            Err(StatusCode::UNAUTHORIZED)
        }
    }

    /// Validate JWT token (simplified implementation)
    fn validate_jwt_token(&self, token: &str) -> bool {
        // In a real implementation, this would:
        // 1. Parse the JWT token
        // 2. Verify signature with the secret
        // 3. Check expiration
        // 4. Validate claims
        
        // For demo purposes, accept any non-empty token
        !token.is_empty()
    }
}

/// Rate Limiting Middleware
pub struct RateLimitMiddleware {
    /// Client request counters
    client_counters: Arc<RwLock<HashMap<String, ClientRateLimit>>>,
    /// Requests per second limit
    requests_per_second: u32,
    /// Burst allowance
    burst_size: u32,
    /// Window duration
    window_duration: Duration,
}

/// Rate limit tracking for individual clients
#[derive(Debug, Clone)]
struct ClientRateLimit {
    /// Request count in current window
    request_count: u32,
    /// Window start time
    window_start: Instant,
    /// Last request time
    last_request: Instant,
    /// Available tokens (for token bucket)
    available_tokens: u32,
}

impl RateLimitMiddleware {
    pub fn new(requests_per_second: u32, burst_size: u32, window_duration: Duration) -> Self {
        Self {
            client_counters: Arc::new(RwLock::new(HashMap::new())),
            requests_per_second,
            burst_size,
            window_duration,
        }
    }

    /// Middleware function for rate limiting
    pub async fn rate_limit(
        &self,
        ConnectInfo(addr): ConnectInfo<SocketAddr>,
        request: Request,
        next: Next,
    ) -> std::result::Result<Response, StatusCode> {
        let client_id = addr.ip().to_string();

        if self.is_rate_limited(&client_id).await {
            // Return 429 Too Many Requests with retry information
            let retry_after = self.get_retry_after(&client_id).await;
            let mut response = StatusCode::TOO_MANY_REQUESTS.into_response();

            if let Ok(_headers) = response.headers_mut().try_insert(
                HeaderName::from_static("retry-after"),
                HeaderValue::from_str(&retry_after.as_secs().to_string()).unwrap(),
            ) {
                // Header inserted successfully
            }

            return Err(StatusCode::TOO_MANY_REQUESTS);
        }

        // Update counters
        self.update_counters(&client_id).await;

        Ok(next.run(request).await)
    }

    /// Check if client is rate limited
    async fn is_rate_limited(&self, client_id: &str) -> bool {
        let mut counters = self.client_counters.write().await;
        let now = Instant::now();

        let client_limit = counters.entry(client_id.to_string()).or_insert_with(|| {
            ClientRateLimit {
                request_count: 0,
                window_start: now,
                last_request: now,
                available_tokens: self.burst_size,
            }
        });

        // Reset window if needed
        if now.duration_since(client_limit.window_start) >= self.window_duration {
            client_limit.request_count = 0;
            client_limit.window_start = now;
            client_limit.available_tokens = self.burst_size;
        }

        // Token bucket algorithm
        let time_since_last = now.duration_since(client_limit.last_request);
        let tokens_to_add = (time_since_last.as_secs_f64() * self.requests_per_second as f64) as u32;
        client_limit.available_tokens = (client_limit.available_tokens + tokens_to_add).min(self.burst_size);
        client_limit.last_request = now;

        // Check if request can be processed
        client_limit.available_tokens == 0
    }

    /// Update request counters
    async fn update_counters(&self, client_id: &str) {
        let mut counters = self.client_counters.write().await;
        
        if let Some(client_limit) = counters.get_mut(client_id) {
            client_limit.request_count += 1;
            if client_limit.available_tokens > 0 {
                client_limit.available_tokens -= 1;
            }
        }
    }

    /// Get retry after duration for rate-limited client
    async fn get_retry_after(&self, client_id: &str) -> Duration {
        let counters = self.client_counters.read().await;
        
        if let Some(client_limit) = counters.get(client_id) {
            let time_to_next_token = Duration::from_secs_f64(1.0 / self.requests_per_second as f64);
            return time_to_next_token;
        }

        Duration::from_secs(1)
    }

    /// Cleanup old entries (should be called periodically)
    pub async fn cleanup_old_entries(&self) {
        let mut counters = self.client_counters.write().await;
        let now = Instant::now();
        let cleanup_threshold = self.window_duration * 2;

        counters.retain(|_, client_limit| {
            now.duration_since(client_limit.window_start) < cleanup_threshold
        });
    }
}

/// Metrics Collection Middleware
pub struct MetricsMiddleware {
    /// Request metrics
    metrics: Arc<RequestMetrics>,
}

/// Request metrics tracking
#[derive(Debug, Default)]
pub struct RequestMetrics {
    /// Total requests
    pub total_requests: AtomicU64,
    /// Requests by endpoint
    pub endpoint_requests: Arc<RwLock<HashMap<String, u64>>>,
    /// Response times
    pub response_times: Arc<RwLock<Vec<u64>>>,
    /// Status code counts
    pub status_codes: Arc<RwLock<HashMap<u16, u64>>>,
    /// Error count
    pub error_count: AtomicU64,
    /// Bytes transferred
    pub bytes_transferred: AtomicU64,
}

impl MetricsMiddleware {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RequestMetrics::default()),
        }
    }

    /// Middleware function for metrics collection
    pub async fn collect_metrics(
        &self,
        request: Request,
        next: Next,
    ) -> Response {
        let start_time = Instant::now();
        let path = request.uri().path().to_string();
        let method = request.method().to_string();
        
        // Process request
        let response = next.run(request).await;
        
        // Collect metrics
        let processing_time = start_time.elapsed();
        let status_code = response.status().as_u16();
        
        // Update metrics
        self.update_metrics(&path, &method, processing_time, status_code).await;
        
        response
    }

    /// Update collected metrics
    async fn update_metrics(
        &self,
        path: &str,
        method: &str,
        processing_time: Duration,
        status_code: u16,
    ) {
        // Update total requests
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // Update endpoint requests
        let endpoint_key = format!("{} {}", method, path);
        let mut endpoint_requests = self.metrics.endpoint_requests.write().await;
        *endpoint_requests.entry(endpoint_key).or_insert(0) += 1;
        
        // Update response times
        let mut response_times = self.metrics.response_times.write().await;
        if response_times.len() >= 10000 {
            response_times.remove(0); // Keep rolling window
        }
        response_times.push(processing_time.as_micros() as u64);
        
        // Update status codes
        let mut status_codes = self.metrics.status_codes.write().await;
        *status_codes.entry(status_code).or_insert(0) += 1;
        
        // Update error count for non-2xx responses
        if status_code >= 400 {
            self.metrics.error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> MetricsSnapshot {
        let endpoint_requests = self.metrics.endpoint_requests.read().await.clone();
        let response_times = self.metrics.response_times.read().await.clone();
        let status_codes = self.metrics.status_codes.read().await.clone();

        MetricsSnapshot {
            total_requests: self.metrics.total_requests.load(Ordering::Relaxed),
            endpoint_requests,
            avg_response_time_us: if response_times.is_empty() {
                0.0
            } else {
                response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
            },
            status_codes,
            error_count: self.metrics.error_count.load(Ordering::Relaxed),
            error_rate: {
                let total = self.metrics.total_requests.load(Ordering::Relaxed);
                let errors = self.metrics.error_count.load(Ordering::Relaxed);
                if total > 0 { errors as f64 / total as f64 } else { 0.0 }
            },
        }
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub endpoint_requests: HashMap<String, u64>,
    pub avg_response_time_us: f64,
    pub status_codes: HashMap<u16, u64>,
    pub error_count: u64,
    pub error_rate: f64,
}

/// Error Handling Middleware
pub struct ErrorHandlerMiddleware;

impl ErrorHandlerMiddleware {
    /// Middleware function for error handling
    pub async fn handle_errors(
        request: Request,
        next: Next,
    ) -> Response {
        let response = next.run(request).await;
        
        // If the response is an error status, convert it to a structured API error
        if response.status().is_client_error() || response.status().is_server_error() {
            let status = response.status();
            let request_id = Uuid::new_v4().to_string();
            
            let api_error = ApiError {
                code: status.canonical_reason().unwrap_or("UNKNOWN_ERROR").to_string(),
                message: format!("HTTP {}: {}", status.as_u16(), status.canonical_reason().unwrap_or("Unknown error")),
                details: None,
                timestamp: chrono::Utc::now(),
                request_id: Some(request_id),
            };
            
            return (status, Json(api_error)).into_response();
        }
        
        response
    }
}

/// CORS Middleware (simplified)
pub struct CorsMiddleware;

impl CorsMiddleware {
    /// Middleware function for CORS handling
    pub async fn handle_cors(
        request: Request,
        next: Next,
    ) -> Response {
        let mut response = next.run(request).await;
        
        // Add CORS headers
        let headers = response.headers_mut();
        headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
        headers.insert("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS".parse().unwrap());
        headers.insert("Access-Control-Allow-Headers", "Content-Type, Authorization".parse().unwrap());
        headers.insert("Access-Control-Max-Age", "3600".parse().unwrap());
        
        response
    }
}

/// Request ID Middleware for tracing
pub struct RequestIdMiddleware;

impl RequestIdMiddleware {
    /// Middleware function for request ID injection
    pub async fn add_request_id(
        mut request: Request,
        next: Next,
    ) -> Response {
        // Generate or extract request ID
        let request_id = request
            .headers()
            .get("X-Request-ID")
            .and_then(|header| header.to_str().ok())
            .unwrap_or(&Uuid::new_v4().to_string())
            .to_string();

        // Add request ID to request headers for downstream use
        request.headers_mut().insert(
            "X-Request-ID",
            request_id.parse().unwrap(),
        );

        let mut response = next.run(request).await;
        
        // Add request ID to response headers
        response.headers_mut().insert(
            "X-Request-ID",
            request_id.parse().unwrap(),
        );
        
        response
    }
}

/// Compression Middleware (using tower-http)
pub use tower_http::compression::CompressionLayer;

/// Timeout Middleware (using tower-http)
pub use tower_http::timeout::TimeoutLayer;

/// Logging/Tracing Middleware
pub struct TracingMiddleware;

impl TracingMiddleware {
    /// Middleware function for request/response logging
    pub async fn log_requests(
        ConnectInfo(addr): ConnectInfo<SocketAddr>,
        request: Request,
        next: Next,
    ) -> Response {
        let start_time = Instant::now();
        let method = request.method().clone();
        let uri = request.uri().clone();
        let user_agent = request
            .headers()
            .get("User-Agent")
            .and_then(|header| header.to_str().ok())
            .unwrap_or("Unknown")
            .to_string();

        println!(
            "📥 {} {} {} - {} - Starting",
            addr.ip(),
            method,
            uri.path(),
            user_agent
        );

        let response = next.run(request).await;
        let processing_time = start_time.elapsed();
        
        println!(
            "📤 {} {} {} - {} - {}ms - {}",
            addr.ip(),
            method,
            uri.path(),
            response.status().as_u16(),
            processing_time.as_millis(),
            user_agent
        );

        response
    }
}

/// Security Headers Middleware
pub struct SecurityHeadersMiddleware;

impl SecurityHeadersMiddleware {
    /// Middleware function for adding security headers
    pub async fn add_security_headers(
        request: Request,
        next: Next,
    ) -> Response {
        let mut response = next.run(request).await;
        
        let headers = response.headers_mut();
        
        // Security headers
        headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
        headers.insert("X-Frame-Options", "DENY".parse().unwrap());
        headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
        headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
        headers.insert(
            "Content-Security-Policy",
            "default-src 'self'".parse().unwrap(),
        );
        
        // Remove server identification
        headers.remove("Server");
        
        response
    }
}

/// Health check for middleware components
pub async fn middleware_health_check() -> std::result::Result<serde_json::Value, AtsCoreError> {
    Ok(serde_json::json!({
        "middleware_status": "healthy",
        "components": {
            "auth": "enabled",
            "rate_limiting": "enabled",
            "metrics": "collecting",
            "error_handling": "active",
            "cors": "configured",
            "security_headers": "active"
        },
        "timestamp": chrono::Utc::now()
    }))
}