pub mod alpaca;
pub mod newsapi;
pub mod polygon;
pub mod social;

use crate::error::Result;
use crate::models::NewsArticle;
use async_trait::async_trait;

#[async_trait]
pub trait NewsSource: Send + Sync {
    /// Fetch news articles from this source
    async fn fetch(&self, symbols: &[String]) -> Result<Vec<NewsArticle>>;

    /// Get the source name
    fn source_name(&self) -> &str;

    /// Check if the source is available
    async fn is_available(&self) -> bool {
        true
    }

    /// Get rate limit info
    fn rate_limit(&self) -> RateLimit {
        RateLimit::default()
    }
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            requests_per_second: 10,
            requests_per_minute: 100,
            requests_per_hour: 1000,
        }
    }
}

pub struct SourceConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}
