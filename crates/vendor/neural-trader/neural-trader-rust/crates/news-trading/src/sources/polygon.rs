use crate::error::{NewsError, Result};
use crate::models::NewsArticle;
use crate::sources::{NewsSource, SourceConfig};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct PolygonNews {
    client: Client,
    api_key: String,
    base_url: String,
}

impl PolygonNews {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            api_key,
            base_url: "https://api.polygon.io/v2".to_string(),
        }
    }

    pub fn from_config(config: SourceConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| NewsError::Config("API key required".to_string()))?;

        Ok(Self::new(api_key))
    }

    async fn fetch_ticker_news(&self, symbol: &str) -> Result<Vec<PolygonNewsItem>> {
        let url = format!(
            "{}/reference/news?ticker={}&limit=50&apiKey={}",
            self.base_url, symbol, self.api_key
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NewsError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(NewsError::Api(format!(
                "Polygon API error: {}",
                response.status()
            )));
        }

        let data: PolygonNewsResponse = response
            .json()
            .await
            .map_err(|e| NewsError::Parse(e.to_string()))?;

        Ok(data.results)
    }
}

#[async_trait]
impl NewsSource for PolygonNews {
    async fn fetch(&self, symbols: &[String]) -> Result<Vec<NewsArticle>> {
        let mut all_articles = Vec::new();

        for symbol in symbols {
            match self.fetch_ticker_news(symbol).await {
                Ok(items) => {
                    for item in items {
                        let article = NewsArticle::new(
                            item.id.clone(),
                            item.title.clone(),
                            item.description.unwrap_or_default(),
                            "polygon".to_string(),
                        )
                        .with_symbols(vec![symbol.clone()])
                        .with_relevance(0.75);

                        all_articles.push(article);
                    }
                }
                Err(e) => {
                    eprintln!("Error fetching Polygon news for {}: {}", symbol, e);
                    continue;
                }
            }

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(all_articles)
    }

    fn source_name(&self) -> &str {
        "polygon"
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct PolygonNewsResponse {
    status: String,
    results: Vec<PolygonNewsItem>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PolygonNewsItem {
    id: String,
    publisher: PolygonPublisher,
    title: String,
    author: String,
    published_utc: DateTime<Utc>,
    article_url: String,
    tickers: Vec<String>,
    description: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PolygonPublisher {
    name: String,
    homepage_url: Option<String>,
    logo_url: Option<String>,
}
