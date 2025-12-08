use crate::*;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use futures::stream::{self, StreamExt};

#[derive(Debug, Clone)]
pub struct NewsAggregator {
    api_keys: Vec<String>,
    client: reqwest::Client,
    sources: Vec<NewsSource>,
}

#[derive(Debug, Clone)]
enum NewsSource {
    NewsAPI { api_key: String },
    CryptoCompare { api_key: String },
    CoinGecko { api_key: String },
    CryptoPanic { api_key: String },
}

impl NewsAggregator {
    pub fn new(api_keys: Vec<String>) -> Self {
        let mut sources = vec![];
        
        // Distribute API keys to different services
        for (idx, key) in api_keys.iter().enumerate() {
            match idx % 4 {
                0 => sources.push(NewsSource::NewsAPI { api_key: key.clone() }),
                1 => sources.push(NewsSource::CryptoCompare { api_key: key.clone() }),
                2 => sources.push(NewsSource::CoinGecko { api_key: key.clone() }),
                _ => sources.push(NewsSource::CryptoPanic { api_key: key.clone() }),
            }
        }
        
        Self {
            api_keys,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap(),
            sources,
        }
    }
    
    pub async fn fetch_and_analyze(&self, symbol: &str) -> Result<NewsData, SentimentError> {
        if self.sources.is_empty() {
            return Ok(self.generate_mock_data(symbol));
        }
        
        // Fetch news from multiple sources in parallel
        let mut all_articles = vec![];
        let mut all_headlines = vec![];
        
        let futures: Vec<_> = self.sources.iter()
            .map(|source| self.fetch_from_source(source, symbol))
            .collect();
        
        let results = stream::iter(futures)
            .buffer_unordered(4)
            .collect::<Vec<_>>()
            .await;
        
        for result in results {
            if let Ok(articles) = result {
                for article in articles {
                    all_headlines.push(article.title.clone());
                    if let Some(content) = article.content {
                        all_articles.push(content);
                    }
                }
            }
        }
        
        // Calculate metrics
        let total_articles = all_headlines.len() as u32;
        let articles_per_hour = self.calculate_velocity(&all_headlines);
        let major_outlets = self.count_major_outlets(&all_headlines);
        
        Ok(NewsData {
            headlines: all_headlines,
            articles: all_articles,
            total_articles,
            articles_per_hour,
            major_outlets,
        })
    }
    
    async fn fetch_from_source(&self, source: &NewsSource, symbol: &str) -> Result<Vec<Article>, SentimentError> {
        match source {
            NewsSource::NewsAPI { api_key } => self.fetch_newsapi(api_key, symbol).await,
            NewsSource::CryptoCompare { api_key } => self.fetch_cryptocompare(api_key, symbol).await,
            NewsSource::CoinGecko { api_key } => self.fetch_coingecko(api_key, symbol).await,
            NewsSource::CryptoPanic { api_key } => self.fetch_cryptopanic(api_key, symbol).await,
        }
    }
    
    async fn fetch_newsapi(&self, api_key: &str, symbol: &str) -> Result<Vec<Article>, SentimentError> {
        let url = format!(
            "https://newsapi.org/v2/everything?q={} cryptocurrency&apiKey={}&pageSize=20&sortBy=publishedAt",
            symbol, api_key
        );
        
        let response: NewsAPIResponse = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        
        Ok(response.articles.into_iter()
            .map(|a| Article {
                title: a.title,
                content: a.content,
                source: a.source.name,
                published_at: a.published_at,
            })
            .collect())
    }
    
    async fn fetch_cryptocompare(&self, api_key: &str, symbol: &str) -> Result<Vec<Article>, SentimentError> {
        let url = format!(
            "https://min-api.cryptocompare.com/data/v2/news/?categories={}&api_key={}",
            symbol, api_key
        );
        
        // Mock implementation
        Ok(vec![])
    }
    
    async fn fetch_coingecko(&self, api_key: &str, symbol: &str) -> Result<Vec<Article>, SentimentError> {
        // CoinGecko news endpoint
        // Mock implementation
        Ok(vec![])
    }
    
    async fn fetch_cryptopanic(&self, api_key: &str, symbol: &str) -> Result<Vec<Article>, SentimentError> {
        let url = format!(
            "https://cryptopanic.com/api/v1/posts/?auth_token={}&currencies={}&public=true",
            api_key, symbol
        );
        
        // Mock implementation
        Ok(vec![])
    }
    
    fn calculate_velocity(&self, headlines: &[String]) -> f64 {
        // In production, would calculate based on timestamps
        // For now, estimate based on volume
        let volume = headlines.len() as f64;
        (volume / 24.0).max(0.1) // Articles per hour
    }
    
    fn count_major_outlets(&self, headlines: &[String]) -> u32 {
        let major_outlets = vec![
            "Reuters", "Bloomberg", "CNBC", "CoinDesk", "CoinTelegraph",
            "The Block", "Decrypt", "Forbes", "Wall Street Journal", "Financial Times"
        ];
        
        headlines.iter()
            .filter(|h| major_outlets.iter().any(|&outlet| h.contains(outlet)))
            .count() as u32
    }
    
    fn generate_mock_data(&self, symbol: &str) -> NewsData {
        let base_volume = match symbol {
            "BTC" => 50,
            "ETH" => 40,
            "SOL" | "AVAX" => 25,
            _ => 15,
        };
        
        let volume_variation = (rand::random::<f64>() * 10.0) as u32;
        let total_articles = base_volume + volume_variation;
        
        let headlines = (0..total_articles)
            .map(|i| format!("{} shows {} momentum in latest trading", symbol, 
                if i % 2 == 0 { "bullish" } else { "strong" }))
            .collect();
        
        let articles = (0..total_articles / 2)
            .map(|_| format!("Market analysis indicates {} trend for {}...", 
                if rand::random::<bool>() { "positive" } else { "mixed" }, symbol))
            .collect();
        
        NewsData {
            headlines,
            articles,
            total_articles,
            articles_per_hour: (total_articles as f64 / 24.0).max(0.5),
            major_outlets: (rand::random::<f64>() * 5.0) as u32,
        }
    }
}

// API response structures
#[derive(Debug, Deserialize)]
struct NewsAPIResponse {
    articles: Vec<NewsAPIArticle>,
}

#[derive(Debug, Deserialize)]
struct NewsAPIArticle {
    title: String,
    #[serde(rename = "content")]
    content: Option<String>,
    source: NewsAPISource,
    #[serde(rename = "publishedAt")]
    published_at: String,
}

#[derive(Debug, Deserialize)]
struct NewsAPISource {
    name: String,
}

struct Article {
    title: String,
    content: Option<String>,
    source: String,
    published_at: String,
}

// News categorization and analysis
pub struct NewsAnalyzer {
    categorizer: NewsCategorizer,
    impact_scorer: ImpactScorer,
}

impl NewsAnalyzer {
    pub fn new() -> Self {
        Self {
            categorizer: NewsCategorizer::new(),
            impact_scorer: ImpactScorer::new(),
        }
    }
    
    pub fn categorize_news(&self, headlines: &[String]) -> HashMap<NewsCategory, Vec<String>> {
        self.categorizer.categorize_batch(headlines)
    }
    
    pub fn score_impact(&self, article: &str) -> f64 {
        self.impact_scorer.calculate_impact(article)
    }
}

struct NewsCategorizer {
    category_keywords: HashMap<NewsCategory, Vec<&'static str>>,
}

impl NewsCategorizer {
    fn new() -> Self {
        let mut keywords = HashMap::new();
        
        keywords.insert(NewsCategory::Regulatory, vec![
            "sec", "regulation", "legal", "lawsuit", "compliance", "government",
            "ban", "restriction", "policy", "congress", "senate"
        ]);
        
        keywords.insert(NewsCategory::Technical, vec![
            "upgrade", "fork", "development", "release", "update", "protocol",
            "mainnet", "testnet", "consensus", "algorithm"
        ]);
        
        keywords.insert(NewsCategory::Market, vec![
            "price", "market", "trading", "volume", "liquidity", "exchange",
            "bull", "bear", "trend", "analysis", "prediction"
        ]);
        
        keywords.insert(NewsCategory::Partnership, vec![
            "partnership", "collaboration", "integration", "adopt", "partner",
            "agreement", "deal", "venture", "alliance"
        ]);
        
        keywords.insert(NewsCategory::Security, vec![
            "hack", "breach", "exploit", "vulnerability", "attack", "security",
            "audit", "bug", "incident", "compromise"
        ]);
        
        Self { category_keywords: keywords }
    }
    
    fn categorize_batch(&self, headlines: &[String]) -> HashMap<NewsCategory, Vec<String>> {
        let mut categorized = HashMap::new();
        
        for headline in headlines {
            let category = self.categorize_single(headline);
            categorized.entry(category)
                .or_insert_with(Vec::new)
                .push(headline.clone());
        }
        
        categorized
    }
    
    fn categorize_single(&self, headline: &str) -> NewsCategory {
        let headline_lower = headline.to_lowercase();
        let mut best_match = NewsCategory::Other;
        let mut best_score = 0;
        
        for (category, keywords) in &self.category_keywords {
            let score = keywords.iter()
                .filter(|&&kw| headline_lower.contains(kw))
                .count();
            
            if score > best_score {
                best_score = score;
                best_match = category.clone();
            }
        }
        
        best_match
    }
}

struct ImpactScorer {
    high_impact_terms: Vec<&'static str>,
    medium_impact_terms: Vec<&'static str>,
}

impl ImpactScorer {
    fn new() -> Self {
        Self {
            high_impact_terms: vec![
                "breaking", "urgent", "major", "significant", "critical",
                "unprecedented", "historic", "massive", "emergency"
            ],
            medium_impact_terms: vec![
                "important", "notable", "update", "announcement", "development",
                "progress", "change", "shift", "move"
            ],
        }
    }
    
    fn calculate_impact(&self, content: &str) -> f64 {
        let content_lower = content.to_lowercase();
        
        let high_impact_count = self.high_impact_terms.iter()
            .filter(|&&term| content_lower.contains(term))
            .count();
        
        let medium_impact_count = self.medium_impact_terms.iter()
            .filter(|&&term| content_lower.contains(term))
            .count();
        
        let base_score = 0.5;
        let high_impact_boost = high_impact_count as f64 * 0.2;
        let medium_impact_boost = medium_impact_count as f64 * 0.1;
        
        (base_score + high_impact_boost + medium_impact_boost).min(1.0)
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum NewsCategory {
    Regulatory,
    Technical,
    Market,
    Partnership,
    Security,
    Other,
}

// Utility for random number generation
mod rand {
    pub fn random<T>() -> T 
    where 
        Standard: Distribution<T>,
    {
        use rand::Rng;
        rand::thread_rng().gen()
    }
    
    use rand::distributions::{Distribution, Standard};
}