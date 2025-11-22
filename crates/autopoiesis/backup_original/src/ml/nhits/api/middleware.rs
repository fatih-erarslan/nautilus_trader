use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm, Header};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::ml::nhits::api::{
    models::ErrorResponse,
    server::AppState,
};

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,          // subject (user id)
    pub exp: usize,           // expiration time
    pub iat: usize,           // issued at
    pub roles: Vec<String>,   // user roles
    pub permissions: Vec<String>, // specific permissions
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window in seconds
    pub window_seconds: u64,
    /// Different limits for different user tiers
    pub tier_limits: HashMap<String, (u32, u64)>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        let mut tier_limits = HashMap::new();
        tier_limits.insert("free".to_string(), (100, 3600));      // 100 req/hour
        tier_limits.insert("premium".to_string(), (1000, 3600));  // 1000 req/hour
        tier_limits.insert("enterprise".to_string(), (10000, 3600)); // 10000 req/hour
        
        Self {
            max_requests: 1000,
            window_seconds: 3600,
            tier_limits,
        }
    }
}

/// Rate limiting tracker
#[derive(Debug)]
pub struct RateLimitTracker {
    /// Request counts per client
    client_requests: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    /// Configuration
    config: RateLimitConfig,
}

impl RateLimitTracker {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            client_requests: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if client is within rate limits
    pub async fn check_rate_limit(&self, client_id: &str, tier: Option<&str>) -> bool {
        let mut requests = self.client_requests.write().await;
        let now = Instant::now();
        
        // Get limits for this client tier
        let (max_requests, window_seconds) = tier
            .and_then(|t| self.config.tier_limits.get(t))
            .unwrap_or(&(self.config.max_requests, self.config.window_seconds));
        
        let window_duration = Duration::from_secs(*window_seconds);
        
        // Get or create request history for this client
        let client_requests = requests.entry(client_id.to_string()).or_insert_with(Vec::new);
        
        // Remove expired requests
        client_requests.retain(|&request_time| now.duration_since(request_time) < window_duration);
        
        // Check if within limits
        if client_requests.len() >= *max_requests as usize {
            false
        } else {
            // Add current request
            client_requests.push(now);
            true
        }
    }

    /// Get current usage for a client
    pub async fn get_usage(&self, client_id: &str) -> (usize, u32, u64) {
        let requests = self.client_requests.read().await;
        let now = Instant::now();
        
        if let Some(client_requests) = requests.get(client_id) {
            let active_requests = client_requests
                .iter()
                .filter(|&&request_time| now.duration_since(request_time) < Duration::from_secs(self.config.window_seconds))
                .count();
            
            (active_requests, self.config.max_requests, self.config.window_seconds)
        } else {
            (0, self.config.max_requests, self.config.window_seconds)
        }
    }

    /// Cleanup expired entries (should be called periodically)
    pub async fn cleanup(&self) {
        let mut requests = self.client_requests.write().await;
        let now = Instant::now();
        let window_duration = Duration::from_secs(self.config.window_seconds);
        
        requests.retain(|_, client_requests| {
            client_requests.retain(|&request_time| now.duration_since(request_time) < window_duration);
            !client_requests.is_empty()
        });
    }
}

// Global rate limiter instance
lazy_static::lazy_static! {
    static ref RATE_LIMITER: RateLimitTracker = RateLimitTracker::new(RateLimitConfig::default());
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for health check and metrics endpoints
    let path = request.uri().path();
    if path == "/api/v1/health" || path == "/api/v1/metrics" {
        return Ok(next.run(request).await);
    }

    let headers = request.headers();
    
    // Extract authorization header
    let auth_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing Authorization header");
            StatusCode::UNAUTHORIZED
        })?;

    // Parse Bearer token
    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| {
            warn!("Invalid Authorization header format");
            StatusCode::UNAUTHORIZED
        })?;

    // Validate JWT token
    let claims = validate_jwt_token(token).map_err(|e| {
        warn!("JWT validation failed: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    // Check if token is expired
    let now = chrono::Utc::now().timestamp() as usize;
    if claims.exp < now {
        warn!("JWT token expired");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Add user information to request extensions
    request.extensions_mut().insert(claims);

    debug!("Authentication successful for user: {}", 
           request.extensions().get::<Claims>().unwrap().sub);

    Ok(next.run(request).await)
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client identifier
    let client_id = extract_client_id(&request);
    
    // Get user tier from JWT claims if available
    let tier = request
        .extensions()
        .get::<Claims>()
        .and_then(|claims| {
            if claims.roles.contains(&"enterprise".to_string()) {
                Some("enterprise")
            } else if claims.roles.contains(&"premium".to_string()) {
                Some("premium")
            } else {
                Some("free")
            }
        });

    // Check rate limits
    if !RATE_LIMITER.check_rate_limit(&client_id, tier).await {
        warn!("Rate limit exceeded for client: {}", client_id);
        
        // Get current usage for headers
        let (current, limit, window) = RATE_LIMITER.get_usage(&client_id).await;
        
        let mut response = Response::new(
            serde_json::to_string(&ErrorResponse::new(
                "Rate limit exceeded".to_string(),
                "RATE_LIMIT_EXCEEDED".to_string(),
            )).unwrap().into()
        );
        
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        
        // Add rate limit headers
        response.headers_mut().insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
        response.headers_mut().insert("X-RateLimit-Remaining", (limit.saturating_sub(current as u32)).to_string().parse().unwrap());
        response.headers_mut().insert("X-RateLimit-Reset", (chrono::Utc::now().timestamp() + window as i64).to_string().parse().unwrap());
        
        return Ok(response);
    }

    debug!("Rate limit check passed for client: {}", client_id);

    let mut response = next.run(request).await;
    
    // Add rate limit headers to successful responses
    let (current, limit, window) = RATE_LIMITER.get_usage(&client_id).await;
    response.headers_mut().insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
    response.headers_mut().insert("X-RateLimit-Remaining", (limit.saturating_sub(current as u32)).to_string().parse().unwrap());
    response.headers_mut().insert("X-RateLimit-Reset", (chrono::Utc::now().timestamp() + window as i64).to_string().parse().unwrap());

    Ok(response)
}

/// CORS middleware (if needed beyond tower-http)
pub async fn cors_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    headers.insert("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS".parse().unwrap());
    headers.insert("Access-Control-Allow-Headers", "Content-Type, Authorization".parse().unwrap());
    headers.insert("Access-Control-Max-Age", "86400".parse().unwrap());
    
    response
}

/// Request logging middleware
pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start_time = Instant::now();
    
    debug!("Request started: {} {}", method, uri);
    
    let response = next.run(request).await;
    let elapsed = start_time.elapsed();
    
    info!("Request completed: {} {} - {} - {:?}", 
          method, uri, response.status(), elapsed);
    
    response
}

/// Extract client identifier from request
fn extract_client_id(request: &Request) -> String {
    // Try to get client ID from JWT claims first
    if let Some(claims) = request.extensions().get::<Claims>() {
        return claims.sub.clone();
    }
    
    // Fall back to IP address
    request
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .or_else(|| {
            request
                .headers()
                .get("x-real-ip")
                .and_then(|h| h.to_str().ok())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Validate JWT token
fn validate_jwt_token(token: &str) -> Result<Claims> {
    // Get JWT secret from environment - NEVER use hardcoded fallback
    let secret_key = std::env::var("JWT_SECRET")
        .map_err(|_| anyhow::anyhow!("JWT_SECRET environment variable is not set"))?;
    
    // Validate minimum security requirements for JWT secret
    if secret_key.len() < 32 {
        error!("JWT_SECRET must be at least 32 characters long for security");
        return Err(anyhow::anyhow!("JWT_SECRET does not meet minimum security requirements"));
    }
    
    // Check for obviously insecure secrets
    let insecure_patterns = [
        "secret", "password", "key", "test", "dev", "demo", "example", 
        "default", "changeme", "admin", "root", "user", "123456"
    ];
    
    let secret_lower = secret_key.to_lowercase();
    for pattern in &insecure_patterns {
        if secret_lower.contains(pattern) {
            error!("JWT_SECRET contains insecure pattern: {}", pattern);
            return Err(anyhow::anyhow!("JWT_SECRET contains insecure pattern"));
        }
    }
    
    let validation = Validation::new(Algorithm::HS256);
    let decoding_key = DecodingKey::from_secret(secret_key.as_ref());
    
    match decode::<Claims>(token, &decoding_key, &validation) {
        Ok(token_data) => Ok(token_data.claims),
        Err(e) => {
            error!("JWT validation error: {}", e);
            Err(anyhow::anyhow!("Invalid JWT token: {}", e))
        }
    }
}

/// Security headers middleware
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    
    // Security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Strict-Transport-Security", "max-age=31536000; includeSubDomains".parse().unwrap());
    headers.insert("Content-Security-Policy", "default-src 'self'".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    
    response
}

/// Background task to clean up rate limiter data
pub async fn start_rate_limiter_cleanup() {
    let mut interval = tokio::time::interval(Duration::from_secs(300)); // Clean up every 5 minutes
    
    loop {
        interval.tick().await;
        RATE_LIMITER.cleanup().await;
        debug!("Rate limiter cleanup completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimitConfig {
            max_requests: 2,
            window_seconds: 1,
            tier_limits: HashMap::new(),
        };
        
        let tracker = RateLimitTracker::new(config);
        let client_id = "test_client";
        
        // First two requests should pass
        assert!(tracker.check_rate_limit(client_id, None).await);
        assert!(tracker.check_rate_limit(client_id, None).await);
        
        // Third request should fail
        assert!(!tracker.check_rate_limit(client_id, None).await);
        
        // After window expires, should work again
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(tracker.check_rate_limit(client_id, None).await);
    }

    #[test]
    fn test_client_id_extraction() {
        let mut request = Request::builder()
            .header("x-forwarded-for", "192.168.1.1, 10.0.0.1")
            .body(())
            .unwrap();
        
        let client_id = extract_client_id(&request);
        assert_eq!(client_id, "192.168.1.1");
    }

    #[test]
    fn test_jwt_validation() {
        // This test would require a proper JWT token
        // For now, just test error handling
        let result = validate_jwt_token("invalid_token");
        assert!(result.is_err());
    }
}