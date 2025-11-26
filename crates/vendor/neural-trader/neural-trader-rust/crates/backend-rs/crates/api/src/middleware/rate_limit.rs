use axum::{
    extract::{ConnectInfo, Request},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct RateLimiter {
    // IP address -> (request count, window start time)
    requests: Arc<RwLock<HashMap<String, (u32, Instant)>>>,
    max_requests: u32,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_requests,
            window_duration: Duration::from_secs(window_secs),
        }
    }

    pub async fn check_rate_limit(&self, ip: &str) -> bool {
        let mut requests = self.requests.write().await;
        let now = Instant::now();

        if let Some((count, window_start)) = requests.get_mut(ip) {
            // Check if we're still in the same time window
            if now.duration_since(*window_start) < self.window_duration {
                if *count >= self.max_requests {
                    return false; // Rate limit exceeded
                }
                *count += 1;
            } else {
                // New time window
                *count = 1;
                *window_start = now;
            }
        } else {
            // First request from this IP
            requests.insert(ip.to_string(), (1, now));
        }

        true
    }

    pub async fn cleanup_old_entries(&self) {
        let mut requests = self.requests.write().await;
        let now = Instant::now();

        requests.retain(|_, (_, window_start)| {
            now.duration_since(*window_start) < self.window_duration * 2
        });
    }
}

// Middleware to apply rate limiting
pub async fn rate_limit_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    limiter: Arc<RateLimiter>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = addr.ip().to_string();

    if limiter.check_rate_limit(&ip).await {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

// Spawn a background task to periodically cleanup old rate limit entries
pub fn spawn_cleanup_task(limiter: Arc<RateLimiter>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
        loop {
            interval.tick().await;
            limiter.cleanup_old_entries().await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5, 60);

        for _ in 0..5 {
            assert!(limiter.check_rate_limit("127.0.0.1").await);
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_after_limit() {
        let limiter = RateLimiter::new(3, 60);

        // First 3 requests should succeed
        for _ in 0..3 {
            assert!(limiter.check_rate_limit("127.0.0.1").await);
        }

        // 4th request should fail
        assert!(!limiter.check_rate_limit("127.0.0.1").await);
    }

    #[tokio::test]
    async fn test_rate_limiter_different_ips() {
        let limiter = RateLimiter::new(2, 60);

        assert!(limiter.check_rate_limit("127.0.0.1").await);
        assert!(limiter.check_rate_limit("127.0.0.2").await);
        assert!(limiter.check_rate_limit("127.0.0.1").await);
        assert!(limiter.check_rate_limit("127.0.0.2").await);

        // Both IPs should now be at limit
        assert!(!limiter.check_rate_limit("127.0.0.1").await);
        assert!(!limiter.check_rate_limit("127.0.0.2").await);
    }
}
