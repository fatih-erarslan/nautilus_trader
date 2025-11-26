//! Rate Limiting Module
//!
//! Provides comprehensive rate limiting capabilities:
//! - Per-API key rate limiting
//! - Per-user rate limiting
//! - DDoS protection
//! - Burst handling
//! - Sliding window rate limiting

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Duration, Utc};

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub max_requests_per_minute: u32,
    /// Maximum burst size
    pub burst_size: u32,
    /// Window duration in seconds
    pub window_duration_secs: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_minute: 100,
            burst_size: 10,
            window_duration_secs: 60,
        }
    }
}

/// Rate limit bucket using token bucket algorithm
#[derive(Debug, Clone)]
struct RateLimitBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: DateTime<Utc>,
    total_requests: u64,
    blocked_requests: u64,
}

impl RateLimitBucket {
    fn new(max_requests_per_minute: u32, burst_size: u32) -> Self {
        let refill_rate = max_requests_per_minute as f64 / 60.0; // per second
        let max_tokens = burst_size as f64;

        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Utc::now(),
            total_requests: 0,
            blocked_requests: 0,
        }
    }

    fn refill(&mut self) {
        let now = Utc::now();
        let elapsed = (now - self.last_refill).num_milliseconds() as f64 / 1000.0;

        if elapsed > 0.0 {
            let new_tokens = elapsed * self.refill_rate;
            self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
            self.last_refill = now;
        }
    }

    fn try_consume(&mut self, tokens: f64) -> std::result::Result<(), RateLimitExceeded> {
        self.refill();
        self.total_requests += 1;

        if self.tokens >= tokens {
            self.tokens -= tokens;
            Ok(())
        } else {
            self.blocked_requests += 1;
            Err(RateLimitExceeded {
                retry_after_secs: ((tokens - self.tokens) / self.refill_rate).ceil() as u64,
                limit: self.max_tokens as u32,
                remaining: self.tokens as u32,
            })
        }
    }

    fn get_stats(&self) -> RateLimitStats {
        RateLimitStats {
            tokens_available: self.tokens as u32,
            max_tokens: self.max_tokens as u32,
            refill_rate: self.refill_rate,
            total_requests: self.total_requests.min(i32::MAX as u64) as i32,
            blocked_requests: self.blocked_requests.min(i32::MAX as u64) as i32,
            success_rate: if self.total_requests > 0 {
                ((self.total_requests - self.blocked_requests) as f64 / self.total_requests as f64) * 100.0
            } else {
                100.0
            },
        }
    }
}

/// Rate limit exceeded error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitExceeded {
    pub retry_after_secs: u64,
    pub limit: u32,
    pub remaining: u32,
}

impl std::fmt::Display for RateLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rate limit exceeded. Limit: {}, Remaining: {}, Retry after: {}s",
            self.limit, self.remaining, self.retry_after_secs
        )
    }
}

impl std::error::Error for RateLimitExceeded {}

impl AsRef<str> for RateLimitExceeded {
    fn as_ref(&self) -> &str {
        "Rate limit exceeded"
    }
}

/// Rate limiter manager
pub struct RateLimiter {
    buckets: Arc<RwLock<HashMap<String, RateLimitBucket>>>,
    default_config: RateLimitConfig,
    // Suspicious activity tracking for DDoS protection
    suspicious_ips: Arc<RwLock<HashMap<String, SuspiciousActivity>>>,
}

#[derive(Debug, Clone)]
struct SuspiciousActivity {
    rapid_requests: u32,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    is_blocked: bool,
}

impl RateLimiter {
    pub fn new(default_config: RateLimitConfig) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            default_config,
            suspicious_ips: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check and consume rate limit for an identifier (API key, IP, user ID)
    pub fn check_rate_limit(&self, identifier: &str, tokens: f64) -> std::result::Result<(), RateLimitExceeded> {
        let mut buckets = self.buckets.write()
            .map_err(|_| RateLimitExceeded {
                retry_after_secs: 1,
                limit: 0,
                remaining: 0,
            })?;

        let bucket = buckets.entry(identifier.to_string()).or_insert_with(|| {
            RateLimitBucket::new(
                self.default_config.max_requests_per_minute,
                self.default_config.burst_size,
            )
        });

        bucket.try_consume(tokens)
    }

    /// Get rate limit statistics for an identifier
    pub fn get_stats(&self, identifier: &str) -> Option<RateLimitStats> {
        let buckets = self.buckets.read().ok()?;
        buckets.get(identifier).map(|b| b.get_stats())
    }

    /// Reset rate limit for an identifier (admin operation)
    pub fn reset_rate_limit(&self, identifier: &str) -> std::result::Result<(), String> {
        let mut buckets = self.buckets.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        buckets.remove(identifier);
        Ok(())
    }

    /// Check for DDoS patterns and potentially block
    pub fn check_ddos_protection(&self, ip_address: &str, request_count: u32) -> std::result::Result<(), String> {
        let mut suspicious = self.suspicious_ips.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        let now = Utc::now();
        let activity = suspicious.entry(ip_address.to_string()).or_insert_with(|| {
            SuspiciousActivity {
                rapid_requests: 0,
                first_seen: now,
                last_seen: now,
                is_blocked: false,
            }
        });

        // If already blocked, check if we should unblock
        if activity.is_blocked {
            let blocked_duration = now - activity.last_seen;
            if blocked_duration > Duration::hours(1) {
                activity.is_blocked = false;
                activity.rapid_requests = 0;
            } else {
                return Err(format!("IP blocked due to suspicious activity. Retry after {} minutes",
                    (60 - blocked_duration.num_minutes()).max(0)));
            }
        }

        activity.rapid_requests += request_count;
        activity.last_seen = now;

        // Check for suspicious patterns
        let time_window = now - activity.first_seen;

        // More than 1000 requests in 1 minute = suspicious
        if time_window < Duration::minutes(1) && activity.rapid_requests > 1000 {
            activity.is_blocked = true;
            tracing::warn!("DDoS protection: Blocking IP {} due to rapid requests", ip_address);
            return Err("Too many rapid requests. DDoS protection activated.".to_string());
        }

        // Reset counter after time window
        if time_window > Duration::minutes(1) {
            activity.rapid_requests = request_count;
            activity.first_seen = now;
        }

        Ok(())
    }

    /// Get list of blocked IPs (admin operation)
    pub fn get_blocked_ips(&self) -> Vec<String> {
        if let Ok(suspicious) = self.suspicious_ips.read() {
            suspicious.iter()
                .filter(|(_, activity)| activity.is_blocked)
                .map(|(ip, _)| ip.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Manually block an IP (admin operation)
    pub fn block_ip(&self, ip_address: &str) -> std::result::Result<(), String> {
        let mut suspicious = self.suspicious_ips.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        let now = Utc::now();
        suspicious.insert(ip_address.to_string(), SuspiciousActivity {
            rapid_requests: 0,
            first_seen: now,
            last_seen: now,
            is_blocked: true,
        });

        Ok(())
    }

    /// Manually unblock an IP (admin operation)
    pub fn unblock_ip(&self, ip_address: &str) -> std::result::Result<(), String> {
        let mut suspicious = self.suspicious_ips.write()
            .map_err(|e| format!("Lock error: {}", e))?;

        suspicious.remove(ip_address);
        Ok(())
    }

    /// Clean up old entries to prevent memory bloat
    pub fn cleanup_old_entries(&self) {
        let now = Utc::now();
        let cutoff = now - Duration::hours(24);

        // Clean up rate limit buckets with no recent activity
        if let Ok(mut buckets) = self.buckets.write() {
            buckets.retain(|_, bucket| bucket.last_refill > cutoff);
        }

        // Clean up unblocked suspicious IPs older than 24 hours
        if let Ok(mut suspicious) = self.suspicious_ips.write() {
            suspicious.retain(|_, activity| {
                activity.is_blocked || activity.last_seen > cutoff
            });
        }
    }
}

/// Rate limit statistics
#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct RateLimitStats {
    pub tokens_available: u32,
    pub max_tokens: u32,
    pub refill_rate: f64,
    pub total_requests: i32,
    pub blocked_requests: i32,
    pub success_rate: f64,
}

/// Global rate limiter instance
static RATE_LIMITER: once_cell::sync::Lazy<Arc<RwLock<Option<RateLimiter>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize rate limiter
#[napi]
pub fn init_rate_limiter(config: Option<RateLimitConfig>) -> Result<String> {
    let rate_config = config.unwrap_or_default();
    let limiter = RateLimiter::new(rate_config);

    let mut rl = RATE_LIMITER.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    *rl = Some(limiter);

    Ok("Rate limiter initialized".to_string())
}

/// Check rate limit for an identifier
#[napi]
pub fn check_rate_limit(identifier: String, tokens: Option<f64>) -> Result<bool> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    match limiter.check_rate_limit(&identifier, tokens.unwrap_or(1.0)) {
        Ok(_) => Ok(true),
        Err(rate_limit_exceeded) => Err(Error::from_reason(format!(
            "Rate limit exceeded. Retry after {} seconds. Limit: {}, Remaining: {}",
            rate_limit_exceeded.retry_after_secs,
            rate_limit_exceeded.limit,
            rate_limit_exceeded.remaining
        ))),
    }
}

/// Get rate limit statistics
#[napi]
pub fn get_rate_limit_stats(identifier: String) -> Result<RateLimitStats> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    limiter.get_stats(&identifier)
        .ok_or_else(|| Error::from_reason("No rate limit data for identifier"))
}

/// Reset rate limit for an identifier (admin operation)
#[napi]
pub fn reset_rate_limit(identifier: String) -> Result<String> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    limiter.reset_rate_limit(&identifier)
        .map_err(|e| Error::from_reason(e))?;

    Ok("Rate limit reset successfully".to_string())
}

/// Check DDoS protection
#[napi]
pub fn check_ddos_protection(ip_address: String, request_count: Option<u32>) -> Result<bool> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    match limiter.check_ddos_protection(&ip_address, request_count.unwrap_or(1)) {
        Ok(_) => Ok(true),
        Err(e) => Err(Error::from_reason(e)),
    }
}

/// Get list of blocked IPs
#[napi]
pub fn get_blocked_ips() -> Result<Vec<String>> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    Ok(limiter.get_blocked_ips())
}

/// Block an IP address (admin operation)
#[napi]
pub fn block_ip(ip_address: String) -> Result<String> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    limiter.block_ip(&ip_address)
        .map_err(|e| Error::from_reason(e))?;

    Ok(format!("IP {} blocked successfully", ip_address))
}

/// Unblock an IP address (admin operation)
#[napi]
pub fn unblock_ip(ip_address: String) -> Result<String> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    limiter.unblock_ip(&ip_address)
        .map_err(|e| Error::from_reason(e))?;

    Ok(format!("IP {} unblocked successfully", ip_address))
}

/// Cleanup old rate limit entries
#[napi]
pub fn cleanup_rate_limiter() -> Result<String> {
    let rl = RATE_LIMITER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let limiter = rl.as_ref()
        .ok_or_else(|| Error::from_reason("Rate limiter not initialized"))?;

    limiter.cleanup_old_entries();

    Ok("Rate limiter cleaned up successfully".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_bucket() {
        let mut bucket = RateLimitBucket::new(60, 10);

        // Should allow up to burst size immediately
        for _ in 0..10 {
            assert!(bucket.try_consume(1.0).is_ok());
        }

        // Should block after burst
        assert!(bucket.try_consume(1.0).is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);

        // Should allow requests within limit
        assert!(limiter.check_rate_limit("test_user", 1.0).is_ok());

        // Should have stats after first request
        assert!(limiter.get_stats("test_user").is_some());
    }
}
