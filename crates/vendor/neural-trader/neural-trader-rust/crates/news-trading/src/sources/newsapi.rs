use crate::error::{NewsError, Result};
use crate::models::NewsArticle;
use crate::sources::{NewsSource, SourceConfig};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct NewsAPISource {
    client: Client,
    api_key: String,
    base_url: String,
}

impl NewsAPISource {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            api_key,
            base_url: "https://newsapi.org/v2".to_string(),
        }
    }

    pub fn from_config(config: SourceConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| NewsError::Config("API key required".to_string()))?;

        Ok(Self::new(api_key))
    }

    async fn search_everything(&self, query: &str) -> Result<Vec<NewsAPIArticle>> {
        let url = format!(
            "{}/everything?q={}&sortBy=publishedAt&language=en&apiKey={}",
            self.base_url, query, self.api_key
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NewsError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(NewsError::Api(format!(
                "NewsAPI error: {}",
                response.status()
            )));
        }

        let data: NewsAPIResponse = response
            .json()
            .await
            .map_err(|e| NewsError::Parse(e.to_string()))?;

        Ok(data.articles)
    }
}

#[async_trait]
impl NewsSource for NewsAPISource {
    async fn fetch(&self, symbols: &[String]) -> Result<Vec<NewsArticle>> {
        let mut all_articles = Vec::new();

        for symbol in symbols {
            match self.search_everything(symbol).await {
                Ok(items) => {
                    for item in items {
                        let article = NewsArticle::new(
                            item.url.clone().unwrap_or_else(|| {
                                format!("newsapi_{}", chrono::Utc::now().timestamp())
                            }),
                            item.title.clone(),
                            item.description.unwrap_or_default(),
                            "newsapi".to_string(),
                        )
                        .with_symbols(vec![symbol.clone()])
                        .with_relevance(0.7);

                        all_articles.push(article);
                    }
                }
                Err(e) => {
                    eprintln!("Error fetching NewsAPI for {}: {}", symbol, e);
                    continue;
                }
            }

            // Rate limiting - NewsAPI free tier is limited
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Ok(all_articles)
    }

    fn source_name(&self) -> &str {
        "newsapi"
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct NewsAPIResponse {
    status: String,
    #[serde(rename = "totalResults")]
    total_results: u32,
    articles: Vec<NewsAPIArticle>,
}

#[derive(Debug, Deserialize, Serialize)]
struct NewsAPIArticle {
    source: NewsAPISource2,
    author: Option<String>,
    title: String,
    description: Option<String>,
    url: Option<String>,
    #[serde(rename = "urlToImage")]
    url_to_image: Option<String>,
    #[serde(rename = "publishedAt")]
    published_at: DateTime<Utc>,
    content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct NewsAPISource2 {
    id: Option<String>,
    name: String,
}
