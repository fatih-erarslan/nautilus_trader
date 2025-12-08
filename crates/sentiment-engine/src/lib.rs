use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

pub mod news_analyzer;
pub mod onchain_analyzer;
pub mod social_analyzer;
pub mod transformer_models;

#[derive(Error, Debug)]
pub enum SentimentError {
    #[error("API error: {0}")]
    ApiError(#[from] reqwest::Error),
    
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    
    #[error("Data processing error: {0}")]
    ProcessingError(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub overall_score: f64,
    pub social_sentiment: SocialSentiment,
    pub news_sentiment: NewsSentiment,
    pub whale_activity: WhaleActivity,
    pub defi_metrics: Option<DeFiMetrics>,
    pub confidence: f64,
    pub sources_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialSentiment {
    pub twitter_bullish_ratio: f64,
    pub reddit_sentiment: f64,
    pub engagement_velocity: f64,
    pub influencer_mentions: u32,
    pub viral_score: f64,
    pub community_size_growth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsSentiment {
    pub headline_sentiment: f64,
    pub article_sentiment: f64,
    pub news_volume: u32,
    pub news_velocity: f64,
    pub major_outlet_coverage: bool,
    pub fud_score: f64,
    pub hype_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivity {
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub exchange_netflow: f64,
    pub large_transactions: u32,
    pub whale_count_change: f64,
    pub smart_money_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeFiMetrics {
    pub tvl_change: f64,
    pub yield_attractiveness: f64,
    pub protocol_activity: f64,
    pub governance_participation: f64,
    pub liquidity_depth: f64,
}

#[derive(Clone)]
pub struct SentimentEngine {
    twitter_client: Arc<TwitterClient>,
    reddit_client: Arc<RedditClient>,
    news_aggregator: Arc<NewsAggregator>,
    onchain_analyzer: Arc<OnChainAnalyzer>,
    nlp_engine: Arc<NLPEngine>,
    cache: Arc<DashMap<String, (DateTime<Utc>, SentimentScore)>>,
    config: SentimentConfig,
}

#[derive(Clone)]
pub struct SentimentConfig {
    pub twitter_api_key: Option<String>,
    pub reddit_client_id: Option<String>,
    pub reddit_client_secret: Option<String>,
    pub news_api_keys: Vec<String>,
    pub web3_endpoints: Vec<String>,
    pub cache_duration_minutes: u64,
    pub max_concurrent_requests: usize,
}

impl SentimentEngine {
    pub fn new(config: SentimentConfig) -> Self {
        let twitter_client = Arc::new(TwitterClient::new(config.twitter_api_key.clone()));
        let reddit_client = Arc::new(RedditClient::new(
            config.reddit_client_id.clone(),
            config.reddit_client_secret.clone(),
        ));
        let news_aggregator = Arc::new(NewsAggregator::new(config.news_api_keys.clone()));
        let onchain_analyzer = Arc::new(OnChainAnalyzer::new(config.web3_endpoints.clone()));
        let nlp_engine = Arc::new(NLPEngine::new());
        
        Self {
            twitter_client,
            reddit_client,
            news_aggregator,
            onchain_analyzer,
            nlp_engine,
            cache: Arc::new(DashMap::new()),
            config,
        }
    }

    pub async fn analyze_pair_sentiment(&self, symbol: &str) -> Result<SentimentScore, SentimentError> {
        // Check cache first
        if let Some(cached) = self.get_cached_sentiment(symbol) {
            return Ok(cached);
        }
        
        // Parallel sentiment analysis
        let social_future = self.analyze_social_sentiment(symbol);
        let news_future = self.analyze_news_sentiment(symbol);
        let whale_future = self.analyze_whale_movements(symbol);
        let defi_future = self.analyze_defi_metrics(symbol);
        
        let (social, news, whale, defi) = tokio::join!(
            social_future,
            news_future,
            whale_future,
            defi_future
        );
        
        let social = social?;
        let news = news?;
        let whale = whale?;
        let defi = defi.ok(); // DeFi metrics are optional
        
        // Calculate overall score
        let overall_score = self.calculate_overall_score(&social, &news, &whale, &defi);
        let confidence = self.calculate_confidence(&social, &news, &whale);
        let sources_count = self.count_sources(&social, &news);
        
        let score = SentimentScore {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            overall_score,
            social_sentiment: social,
            news_sentiment: news,
            whale_activity: whale,
            defi_metrics: defi,
            confidence,
            sources_count,
        };
        
        // Cache the result
        self.cache_sentiment(symbol, &score);
        
        Ok(score)
    }

    async fn analyze_social_sentiment(&self, symbol: &str) -> Result<SocialSentiment, SentimentError> {
        let twitter_task = self.twitter_client.analyze_sentiment(symbol);
        let reddit_task = self.reddit_client.analyze_sentiment(symbol);
        
        let (twitter_result, reddit_result) = tokio::join!(twitter_task, reddit_task);
        
        let twitter_data = twitter_result?;
        let reddit_data = reddit_result?;
        
        Ok(SocialSentiment {
            twitter_bullish_ratio: twitter_data.bullish_ratio,
            reddit_sentiment: reddit_data.sentiment_score,
            engagement_velocity: twitter_data.engagement_growth,
            influencer_mentions: twitter_data.influencer_count,
            viral_score: (twitter_data.viral_score + reddit_data.viral_score) / 2.0,
            community_size_growth: reddit_data.subscriber_growth,
        })
    }

    async fn analyze_news_sentiment(&self, symbol: &str) -> Result<NewsSentiment, SentimentError> {
        let news_data = self.news_aggregator.fetch_and_analyze(symbol).await?;
        
        // Analyze sentiment using NLP
        let headlines: Vec<&str> = news_data.headlines.iter().map(|s| s.as_str()).collect();
        let articles: Vec<&str> = news_data.articles.iter().map(|s| s.as_str()).collect();
        
        let headline_sentiments = self.nlp_engine.batch_analyze(&headlines).await?;
        let article_sentiments = self.nlp_engine.batch_analyze(&articles).await?;
        
        let headline_sentiment = calculate_average_sentiment(&headline_sentiments);
        let article_sentiment = calculate_average_sentiment(&article_sentiments);
        
        // Detect FUD and hype
        let fud_score = self.detect_fud(&headlines, &articles).await?;
        let hype_score = self.detect_hype(&headlines, &articles).await?;
        
        Ok(NewsSentiment {
            headline_sentiment,
            article_sentiment,
            news_volume: news_data.total_articles,
            news_velocity: news_data.articles_per_hour,
            major_outlet_coverage: news_data.major_outlets > 0,
            fud_score,
            hype_score,
        })
    }

    async fn analyze_whale_movements(&self, symbol: &str) -> Result<WhaleActivity, SentimentError> {
        let whale_data = self.onchain_analyzer.analyze_whale_activity(symbol).await?;
        
        Ok(WhaleActivity {
            accumulation_score: whale_data.accumulation_ratio,
            distribution_score: whale_data.distribution_ratio,
            exchange_netflow: whale_data.exchange_netflow,
            large_transactions: whale_data.large_tx_count,
            whale_count_change: whale_data.whale_wallets_change,
            smart_money_confidence: whale_data.smart_money_score,
        })
    }

    async fn analyze_defi_metrics(&self, symbol: &str) -> Result<DeFiMetrics, SentimentError> {
        let defi_data = self.onchain_analyzer.analyze_defi_metrics(symbol).await?;
        
        Ok(DeFiMetrics {
            tvl_change: defi_data.tvl_24h_change,
            yield_attractiveness: defi_data.avg_yield_score,
            protocol_activity: defi_data.protocol_tx_growth,
            governance_participation: defi_data.governance_activity,
            liquidity_depth: defi_data.liquidity_score,
        })
    }

    fn calculate_overall_score(
        &self,
        social: &SocialSentiment,
        news: &NewsSentiment,
        whale: &WhaleActivity,
        defi: &Option<DeFiMetrics>,
    ) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;
        
        // Social sentiment (30% weight)
        let social_score = (social.twitter_bullish_ratio * 0.4 +
                           social.reddit_sentiment * 0.3 +
                           social.engagement_velocity * 0.2 +
                           social.viral_score * 0.1).max(0.0).min(1.0);
        score += social_score * 0.3;
        weight_sum += 0.3;
        
        // News sentiment (25% weight)
        let news_score = (news.headline_sentiment * 0.5 +
                         news.article_sentiment * 0.3 +
                         (1.0 - news.fud_score) * 0.1 +
                         news.hype_score * 0.1).max(0.0).min(1.0);
        score += news_score * 0.25;
        weight_sum += 0.25;
        
        // Whale activity (35% weight)
        let whale_score = (whale.accumulation_score * 0.4 +
                          (1.0 - whale.distribution_score) * 0.3 +
                          whale.smart_money_confidence * 0.3).max(0.0).min(1.0);
        score += whale_score * 0.35;
        weight_sum += 0.35;
        
        // DeFi metrics (10% weight if available)
        if let Some(defi) = defi {
            let defi_score = (defi.tvl_change.max(-1.0).min(1.0) * 0.3 +
                             defi.yield_attractiveness * 0.2 +
                             defi.protocol_activity * 0.2 +
                             defi.liquidity_depth * 0.3).max(0.0).min(1.0);
            score += defi_score * 0.1;
            weight_sum += 0.1;
        }
        
        score / weight_sum
    }

    fn calculate_confidence(
        &self,
        social: &SocialSentiment,
        news: &NewsSentiment,
        whale: &WhaleActivity,
    ) -> f64 {
        // Confidence based on data quality and agreement
        let social_confidence = if social.influencer_mentions > 10 { 0.9 } 
                               else if social.influencer_mentions > 5 { 0.7 }
                               else { 0.5 };
        
        let news_confidence = if news.news_volume > 20 { 0.9 }
                             else if news.news_volume > 10 { 0.7 }
                             else { 0.5 };
        
        let whale_confidence = if whale.large_transactions > 50 { 0.9 }
                              else if whale.large_transactions > 20 { 0.7 }
                              else { 0.5 };
        
        (social_confidence + news_confidence + whale_confidence) / 3.0
    }

    fn count_sources(&self, social: &SocialSentiment, news: &NewsSentiment) -> u32 {
        social.influencer_mentions + news.news_volume
    }

    async fn detect_fud(&self, headlines: &[&str], articles: &[&str]) -> Result<f64, SentimentError> {
        let fud_keywords = vec![
            "scam", "fraud", "hack", "exploit", "rug", "dump", "crash",
            "investigation", "lawsuit", "sec", "regulation", "ban", "illegal"
        ];
        
        let mut fud_score = 0.0;
        let total_content = headlines.len() + articles.len();
        
        for content in headlines.iter().chain(articles.iter()) {
            let content_lower = content.to_lowercase();
            let keyword_count = fud_keywords.iter()
                .filter(|&keyword| content_lower.contains(keyword))
                .count();
            
            fud_score += keyword_count as f64 / fud_keywords.len() as f64;
        }
        
        Ok((fud_score / total_content as f64).min(1.0))
    }

    async fn detect_hype(&self, headlines: &[&str], articles: &[&str]) -> Result<f64, SentimentError> {
        let hype_keywords = vec![
            "moon", "rocket", "bullish", "surge", "soar", "rally", "breakout",
            "ath", "profit", "gains", "pump", "explosive", "massive"
        ];
        
        let mut hype_score = 0.0;
        let total_content = headlines.len() + articles.len();
        
        for content in headlines.iter().chain(articles.iter()) {
            let content_lower = content.to_lowercase();
            let keyword_count = hype_keywords.iter()
                .filter(|&keyword| content_lower.contains(keyword))
                .count();
            
            hype_score += keyword_count as f64 / hype_keywords.len() as f64;
        }
        
        Ok((hype_score / total_content as f64).min(1.0))
    }

    fn get_cached_sentiment(&self, symbol: &str) -> Option<SentimentScore> {
        self.cache.get(symbol).and_then(|entry| {
            let (timestamp, score) = entry.value();
            if (Utc::now() - *timestamp).num_minutes() < self.config.cache_duration_minutes as i64 {
                Some(score.clone())
            } else {
                None
            }
        })
    }

    fn cache_sentiment(&self, symbol: &str, score: &SentimentScore) {
        self.cache.insert(symbol.to_string(), (Utc::now(), score.clone()));
    }
}

// Client implementations
pub struct TwitterClient {
    api_key: Option<String>,
    client: reqwest::Client,
}

impl TwitterClient {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str) -> Result<TwitterData, SentimentError> {
        // Placeholder implementation
        // In production, this would use Twitter API v2
        Ok(TwitterData {
            bullish_ratio: 0.65,
            engagement_growth: 1.2,
            influencer_count: 15,
            viral_score: 0.7,
        })
    }
}

pub struct RedditClient {
    client_id: Option<String>,
    client_secret: Option<String>,
    client: reqwest::Client,
}

impl RedditClient {
    pub fn new(client_id: Option<String>, client_secret: Option<String>) -> Self {
        Self {
            client_id,
            client_secret,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str) -> Result<RedditData, SentimentError> {
        // Placeholder implementation
        // In production, this would use Reddit API
        Ok(RedditData {
            sentiment_score: 0.6,
            viral_score: 0.5,
            subscriber_growth: 1.1,
        })
    }
}

pub struct NewsAggregator {
    api_keys: Vec<String>,
    client: reqwest::Client,
}

impl NewsAggregator {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_keys,
            client: reqwest::Client::new(),
        }
    }

    pub async fn fetch_and_analyze(&self, symbol: &str) -> Result<NewsData, SentimentError> {
        // Placeholder implementation
        // In production, this would aggregate from multiple news APIs
        Ok(NewsData {
            headlines: vec!["Bitcoin surges to new highs".to_string()],
            articles: vec!["Market analysis shows positive trends...".to_string()],
            total_articles: 25,
            articles_per_hour: 2.5,
            major_outlets: 3,
        })
    }
}

pub struct OnChainAnalyzer {
    web3_endpoints: Vec<String>,
    client: reqwest::Client,
}

impl OnChainAnalyzer {
    pub fn new(web3_endpoints: Vec<String>) -> Self {
        Self {
            web3_endpoints,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_whale_activity(&self, symbol: &str) -> Result<WhaleData, SentimentError> {
        // Placeholder implementation
        // In production, this would query blockchain data
        Ok(WhaleData {
            accumulation_ratio: 0.7,
            distribution_ratio: 0.3,
            exchange_netflow: -50000.0,
            large_tx_count: 45,
            whale_wallets_change: 0.05,
            smart_money_score: 0.8,
        })
    }

    pub async fn analyze_defi_metrics(&self, symbol: &str) -> Result<DeFiData, SentimentError> {
        // Placeholder implementation
        Ok(DeFiData {
            tvl_24h_change: 0.05,
            avg_yield_score: 0.7,
            protocol_tx_growth: 1.2,
            governance_activity: 0.6,
            liquidity_score: 0.8,
        })
    }
}

pub struct NLPEngine {
    // Transformer model would be loaded here
}

impl NLPEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn batch_analyze(&self, texts: &[&str]) -> Result<Vec<f64>, SentimentError> {
        // Placeholder implementation
        // In production, this would use transformer models
        Ok(texts.iter().map(|_| 0.7).collect())
    }
}

// Data structures
#[derive(Debug)]
struct TwitterData {
    bullish_ratio: f64,
    engagement_growth: f64,
    influencer_count: u32,
    viral_score: f64,
}

#[derive(Debug)]
struct RedditData {
    sentiment_score: f64,
    viral_score: f64,
    subscriber_growth: f64,
}

#[derive(Debug)]
struct NewsData {
    headlines: Vec<String>,
    articles: Vec<String>,
    total_articles: u32,
    articles_per_hour: f64,
    major_outlets: u32,
}

#[derive(Debug)]
struct WhaleData {
    accumulation_ratio: f64,
    distribution_ratio: f64,
    exchange_netflow: f64,
    large_tx_count: u32,
    whale_wallets_change: f64,
    smart_money_score: f64,
}

#[derive(Debug)]
struct DeFiData {
    tvl_24h_change: f64,
    avg_yield_score: f64,
    protocol_tx_growth: f64,
    governance_activity: f64,
    liquidity_score: f64,
}

fn calculate_average_sentiment(sentiments: &[f64]) -> f64 {
    if sentiments.is_empty() {
        return 0.5;
    }
    sentiments.iter().sum::<f64>() / sentiments.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let config = SentimentConfig {
            twitter_api_key: None,
            reddit_client_id: None,
            reddit_client_secret: None,
            news_api_keys: vec![],
            web3_endpoints: vec![],
            cache_duration_minutes: 5,
            max_concurrent_requests: 10,
        };
        
        let engine = SentimentEngine::new(config);
        let result = engine.analyze_pair_sentiment("BTC/USDT").await;
        assert!(result.is_ok());
    }
}