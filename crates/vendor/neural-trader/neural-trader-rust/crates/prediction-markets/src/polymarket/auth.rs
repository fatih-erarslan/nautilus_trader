//! Authentication utilities for Polymarket API

use crate::error::{PredictionMarketError, Result};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// API credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub api_key: String,
    pub api_secret: Option<String>,
}

impl Credentials {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_secret: None,
        }
    }

    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.api_secret = Some(secret.into());
        self
    }

    /// Validate credentials
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(PredictionMarketError::AuthError(
                "API key cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Generate authentication header value
    pub fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// Generate signature for request (if secret provided)
    pub fn sign_request(&self, method: &str, path: &str, body: &str) -> Result<String> {
        let secret = self
            .api_secret
            .as_ref()
            .ok_or_else(|| PredictionMarketError::AuthError("No API secret provided".to_string()))?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PredictionMarketError::InternalError(e.to_string()))?
            .as_secs();

        let message = format!("{}{}{}{}",timestamp, method.to_uppercase(), path, body);

        // In production, this would use HMAC-SHA256
        // For now, simplified signature
        let signature = format!("{}.{}", secret, message);

        Ok(signature)
    }
}

/// Rate limiter for API requests
pub struct RateLimiter {
    max_requests_per_second: u32,
    last_request_time: std::sync::Mutex<SystemTime>,
}

impl RateLimiter {
    pub fn new(max_requests_per_second: u32) -> Self {
        Self {
            max_requests_per_second,
            last_request_time: std::sync::Mutex::new(SystemTime::now()),
        }
    }

    /// Wait if necessary to comply with rate limits
    pub async fn wait_if_needed(&self) -> Result<()> {
        let min_interval = std::time::Duration::from_millis(1000 / self.max_requests_per_second as u64);

        let mut last_time = self
            .last_request_time
            .lock()
            .map_err(|e| PredictionMarketError::InternalError(e.to_string()))?;

        let now = SystemTime::now();
        let elapsed = now
            .duration_since(*last_time)
            .unwrap_or(std::time::Duration::ZERO);

        if elapsed < min_interval {
            let wait_time = min_interval - elapsed;
            tokio::time::sleep(wait_time).await;
        }

        *last_time = SystemTime::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_validation() {
        let valid_creds = Credentials::new("test_key");
        assert!(valid_creds.validate().is_ok());

        let invalid_creds = Credentials::new("");
        assert!(invalid_creds.validate().is_err());
    }

    #[test]
    fn test_auth_header() {
        let creds = Credentials::new("test_key_123");
        assert_eq!(creds.auth_header(), "Bearer test_key_123");
    }

    #[test]
    fn test_credentials_with_secret() {
        let creds = Credentials::new("key").with_secret("secret");
        assert_eq!(creds.api_secret, Some("secret".to_string()));
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(10); // 10 requests per second

        let start = SystemTime::now();
        for _ in 0..5 {
            limiter.wait_if_needed().await.unwrap();
        }
        let elapsed = start.elapsed().unwrap();

        // Should take at least 400ms for 5 requests at 10/sec
        assert!(elapsed.as_millis() >= 400);
    }
}
