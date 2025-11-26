use crate::error::{NewsError, Result};
use crate::models::NewsArticle;
use crate::sources::{NewsSource, SourceConfig};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct AlpacaNews {
    client: Client,
    api_key: String,
    api_secret: String,
    base_url: String,
}

impl AlpacaNews {
    pub fn new(api_key: String, api_secret: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            api_key,
            api_secret,
            base_url: "https://data.alpaca.markets/v1beta1".to_string(),
        }
    }

    pub fn from_config(config: SourceConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| NewsError::Config("API key required".to_string()))?;

        let api_secret = std::env::var("ALPACA_SECRET")
            .map_err(|_| NewsError::Config("ALPACA_SECRET env var required".to_string()))?;

        Ok(Self::new(api_key, api_secret))
    }

    async fn fetch_news_raw(&self, symbols: &[String]) -> Result<Vec<AlpacaNewsItem>> {
        let symbols_str = symbols.join(",");
        let url = format!("{}/news?symbols={}&limit=50", self.base_url, symbols_str);

        let response = self
            .client
            .get(&url)
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .send()
            .await
            .map_err(|e| NewsError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(NewsError::Api(format!(
                "Alpaca API error: {}",
                response.status()
            )));
        }

        let data: AlpacaNewsResponse = response
            .json()
            .await
            .map_err(|e| NewsError::Parse(e.to_string()))?;

        Ok(data.news)
    }
}

#[async_trait]
impl NewsSource for AlpacaNews {
    async fn fetch(&self, symbols: &[String]) -> Result<Vec<NewsArticle>> {
        let items = self.fetch_news_raw(symbols).await?;

        let articles = items
            .into_iter()
            .map(|item| {
                NewsArticle::new(
                    item.id.to_string(),
                    item.headline,
                    item.summary,
                    "alpaca".to_string(),
                )
                .with_symbols(item.symbols)
                .with_relevance(0.8)
            })
            .collect();

        Ok(articles)
    }

    fn source_name(&self) -> &str {
        "alpaca"
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct AlpacaNewsResponse {
    news: Vec<AlpacaNewsItem>,
}

#[derive(Debug, Deserialize, Serialize)]
struct AlpacaNewsItem {
    id: u64,
    headline: String,
    summary: String,
    author: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    url: String,
    symbols: Vec<String>,
}
