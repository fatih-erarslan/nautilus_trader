use crate::error::{NewsError, Result};
use crate::models::{NewsArticle, NewsQuery};
use crate::sentiment::SentimentAnalyzer;
use crate::sources::NewsSource;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct NewsAggregator {
    sources: Vec<Arc<dyn NewsSource>>,
    cache: Arc<RwLock<NewsCache>>,
    sentiment_analyzer: Arc<SentimentAnalyzer>,
}

impl NewsAggregator {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            cache: Arc::new(RwLock::new(NewsCache::new())),
            sentiment_analyzer: Arc::new(SentimentAnalyzer::default()),
        }
    }

    pub fn add_source(&mut self, source: Arc<dyn NewsSource>) {
        self.sources.push(source);
    }

    pub fn with_sources(mut self, sources: Vec<Arc<dyn NewsSource>>) -> Self {
        self.sources = sources;
        self
    }

    /// Fetch news from all sources
    pub async fn fetch_news(&self, symbols: &[String]) -> Result<Vec<NewsArticle>> {
        let mut all_articles = Vec::new();
        let mut errors = Vec::new();

        // Fetch from all sources concurrently
        let mut handles = Vec::new();

        for source in &self.sources {
            let source = Arc::clone(source);
            let symbols = symbols.to_vec();

            let handle = tokio::spawn(async move {
                match source.fetch(&symbols).await {
                    Ok(articles) => Ok((source.source_name().to_string(), articles)),
                    Err(e) => Err((source.source_name().to_string(), e)),
                }
            });

            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            match handle.await {
                Ok(Ok((source_name, articles))) => {
                    all_articles.extend(articles);
                }
                Ok(Err((source_name, error))) => {
                    errors.push((source_name, error));
                }
                Err(e) => {
                    errors.push(("unknown".to_string(), NewsError::Network(e.to_string())));
                }
            }
        }

        // Log errors but don't fail the entire operation
        for (source, error) in errors {
            eprintln!("Error fetching from {}: {}", source, error);
        }

        // Deduplicate articles
        all_articles = self.deduplicate(all_articles);

        // Analyze sentiment for all articles
        all_articles = self.analyze_sentiment(all_articles).await;

        // Update cache
        let mut cache = self.cache.write().await;
        for article in &all_articles {
            cache.insert(article.clone());
        }

        Ok(all_articles)
    }

    /// Fetch news with sentiment filtering
    pub async fn fetch_with_sentiment(
        &self,
        symbols: &[String],
        min_score: f64,
    ) -> Result<Vec<NewsArticle>> {
        let articles = self.fetch_news(symbols).await?;
        Ok(self.filter_by_sentiment(articles, min_score))
    }

    /// Query cached news
    pub async fn query(&self, query: NewsQuery) -> Vec<NewsArticle> {
        let cache = self.cache.read().await;
        cache.query(&query)
    }

    /// Get latest news from cache
    pub async fn get_latest(&self, limit: usize) -> Vec<NewsArticle> {
        let cache = self.cache.read().await;
        cache.get_latest(limit)
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    /// Filter articles by sentiment score
    pub fn filter_by_sentiment(&self, articles: Vec<NewsArticle>, threshold: f64) -> Vec<NewsArticle> {
        articles
            .into_iter()
            .filter(|article| {
                if let Some(sentiment) = &article.sentiment {
                    sentiment.score.abs() >= threshold
                } else {
                    false
                }
            })
            .collect()
    }

    async fn analyze_sentiment(&self, mut articles: Vec<NewsArticle>) -> Vec<NewsArticle> {
        for article in &mut articles {
            if article.sentiment.is_none() {
                let text = format!("{} {}", article.title, article.content);
                let sentiment = self.sentiment_analyzer.analyze(&text);
                article.sentiment = Some(sentiment);
            }
        }
        articles
    }

    fn deduplicate(&self, articles: Vec<NewsArticle>) -> Vec<NewsArticle> {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();

        for article in articles {
            if seen.insert(article.id.clone()) {
                unique.push(article);
            }
        }

        unique
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub async fn cache_size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.size()
    }
}

impl Default for NewsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct NewsCache {
    articles: HashMap<String, NewsArticle>,
    max_size: usize,
}

impl NewsCache {
    pub fn new() -> Self {
        Self {
            articles: HashMap::new(),
            max_size: 10000,
        }
    }

    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            articles: HashMap::new(),
            max_size,
        }
    }

    pub fn insert(&mut self, article: NewsArticle) {
        if self.articles.len() >= self.max_size {
            self.evict_oldest();
        }
        self.articles.insert(article.id.clone(), article);
    }

    pub fn get(&self, id: &str) -> Option<&NewsArticle> {
        self.articles.get(id)
    }

    pub fn query(&self, query: &NewsQuery) -> Vec<NewsArticle> {
        let mut results: Vec<NewsArticle> = self
            .articles
            .values()
            .filter(|article| query.matches(article))
            .cloned()
            .collect();

        // Sort by published date, most recent first
        results.sort_by(|a, b| b.published_at.cmp(&a.published_at));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        results
    }

    pub fn get_latest(&self, limit: usize) -> Vec<NewsArticle> {
        let mut articles: Vec<NewsArticle> = self.articles.values().cloned().collect();
        articles.sort_by(|a, b| b.published_at.cmp(&a.published_at));
        articles.truncate(limit);
        articles
    }

    pub fn clear(&mut self) {
        self.articles.clear();
    }

    pub fn size(&self) -> usize {
        self.articles.len()
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_id) = self
            .articles
            .iter()
            .min_by_key(|(_, article)| article.published_at)
            .map(|(id, _)| id.clone())
        {
            self.articles.remove(&oldest_id);
        }
    }
}

impl Default for NewsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aggregator_creation() {
        let aggregator = NewsAggregator::new();
        assert_eq!(aggregator.source_count(), 0);
        assert_eq!(aggregator.cache_size().await, 0);
    }

    #[test]
    fn test_cache_insertion() {
        let mut cache = NewsCache::new();
        let article = NewsArticle::new(
            "test1".to_string(),
            "Test Title".to_string(),
            "Test content".to_string(),
            "test".to_string(),
        );

        cache.insert(article.clone());
        assert_eq!(cache.size(), 1);
        assert!(cache.get("test1").is_some());
    }

    #[test]
    fn test_cache_query() {
        let mut cache = NewsCache::new();

        let article1 = NewsArticle::new(
            "1".to_string(),
            "AAPL news".to_string(),
            "Apple earnings".to_string(),
            "test".to_string(),
        )
        .with_symbols(vec!["AAPL".to_string()]);

        cache.insert(article1);

        let query = NewsQuery::new().with_symbols(vec!["AAPL".to_string()]);
        let results = cache.query(&query);

        assert_eq!(results.len(), 1);
    }
}
