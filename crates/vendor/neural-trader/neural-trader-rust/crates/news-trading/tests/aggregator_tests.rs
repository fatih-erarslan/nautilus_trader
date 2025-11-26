use nt_news_trading::{NewsAggregator, NewsArticle};

#[tokio::test]
async fn test_aggregator_creation() {
    let aggregator = NewsAggregator::new();
    assert_eq!(aggregator.source_count(), 0);
}

#[tokio::test]
async fn test_aggregator_cache() {
    let aggregator = NewsAggregator::new();
    assert_eq!(aggregator.cache_size().await, 0);
}

#[tokio::test]
async fn test_fetch_with_no_sources() {
    let aggregator = NewsAggregator::new();
    let symbols = vec!["AAPL".to_string()];

    let result = aggregator.fetch_news(&symbols).await;
    assert!(result.is_ok());

    let articles = result.unwrap();
    assert_eq!(articles.len(), 0);
}

#[tokio::test]
async fn test_sentiment_filtering() {
    let aggregator = NewsAggregator::new();

    let articles = vec![
        NewsArticle::new(
            "1".to_string(),
            "Positive news".to_string(),
            "Great earnings".to_string(),
            "test".to_string(),
        ),
    ];

    let filtered = aggregator.filter_by_sentiment(articles, 0.5);
    assert_eq!(filtered.len(), 0); // No sentiment analyzed yet
}

#[tokio::test]
async fn test_cache_operations() {
    let aggregator = NewsAggregator::new();

    // Initial state
    assert_eq!(aggregator.cache_size().await, 0);

    // Clear should work on empty cache
    aggregator.clear_cache().await;
    assert_eq!(aggregator.cache_size().await, 0);
}
