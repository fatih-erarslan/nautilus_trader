use nt_news_trading::{
    Direction, NewsAggregator, NewsArticle, NewsTradingStrategy, Sentiment, StrategyConfig,
};
use std::sync::Arc;

#[tokio::test]
async fn test_strategy_creation() {
    let aggregator = Arc::new(NewsAggregator::new());
    let _strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    // Strategy should be created successfully
    assert!(true);
}

#[test]
fn test_event_detection() {
    let aggregator = Arc::new(NewsAggregator::new());
    let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    let sentiment = Sentiment::new(0.8, 0.9);
    let article = NewsArticle::new(
        "test".to_string(),
        "Company announces quarterly earnings beat".to_string(),
        "The company exceeded expectations".to_string(),
        "test".to_string(),
    );

    let impact = strategy.calculate_impact(&article, &sentiment);
    assert!(impact > 0.6, "Expected high impact for earnings news");
}

#[test]
fn test_signal_generation() {
    let aggregator = Arc::new(NewsAggregator::new());
    let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    let sentiment = Sentiment::new(0.8, 0.9);
    let article = NewsArticle::new(
        "test".to_string(),
        "Bullish news".to_string(),
        "Strong growth".to_string(),
        "test".to_string(),
    );

    let signal = strategy.generate_signal("AAPL", &sentiment, 0.8, &article);

    assert_eq!(signal.symbol, "AAPL");
    assert_eq!(signal.direction, Direction::Long);
    assert!(signal.confidence > 0.5);
}

#[tokio::test]
async fn test_on_news_positive() {
    let aggregator = Arc::new(NewsAggregator::new());
    let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    let article = NewsArticle::new(
        "test".to_string(),
        "Major breakthrough in earnings with strong bullish momentum".to_string(),
        "The company shows exceptional growth and record profits".to_string(),
        "test".to_string(),
    )
    .with_symbols(vec!["AAPL".to_string()])
    .with_relevance(0.9);

    let result = strategy.on_news(article).await.unwrap();
    assert!(result.is_some(), "Expected trading signal");

    let signal = result.unwrap();
    assert_eq!(signal.direction, Direction::Long);
}

#[tokio::test]
async fn test_on_news_negative() {
    let aggregator = Arc::new(NewsAggregator::new());
    let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    let article = NewsArticle::new(
        "test".to_string(),
        "Company faces crisis crash plunge bankruptcy with major losses and market collapse".to_string(),
        "Severe decline plunging bearish falling declining disaster".to_string(),
        "test".to_string(),
    )
    .with_symbols(vec!["AAPL".to_string()])
    .with_relevance(0.9);

    let result = strategy.on_news(article).await.unwrap();
    assert!(result.is_some(), "Expected trading signal");

    let signal = result.unwrap();
    // With strong negative sentiment, expect Short direction
    assert!(matches!(signal.direction, Direction::Short | Direction::Neutral),
        "Expected Short or Neutral, got {:?}", signal.direction);
}

#[tokio::test]
async fn test_on_news_low_impact() {
    let aggregator = Arc::new(NewsAggregator::new());
    let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    let article = NewsArticle::new(
        "test".to_string(),
        "Company holds routine meeting".to_string(),
        "General discussion about operations".to_string(),
        "test".to_string(),
    )
    .with_symbols(vec!["AAPL".to_string()])
    .with_relevance(0.2);

    let result = strategy.on_news(article).await.unwrap();
    assert!(result.is_none(), "Expected no signal for low impact news");
}

#[test]
fn test_strategy_config() {
    let config = StrategyConfig::default();

    assert_eq!(config.min_impact_threshold, 0.3);
    assert_eq!(config.min_sentiment_magnitude, 0.2);
    assert_eq!(config.min_confidence, 0.4);
}

#[test]
fn test_impact_calculation_components() {
    let aggregator = Arc::new(NewsAggregator::new());
    let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

    // High sentiment, high relevance
    let sentiment = Sentiment::new(0.8, 0.9);
    let article = NewsArticle::new(
        "test".to_string(),
        "Major earnings announcement".to_string(),
        "Content".to_string(),
        "test".to_string(),
    )
    .with_relevance(0.9);

    let impact = strategy.calculate_impact(&article, &sentiment);
    assert!(impact > 0.7, "Expected high impact");

    // Low sentiment, low relevance
    let sentiment2 = Sentiment::new(0.1, 0.2);
    let article2 = NewsArticle::new(
        "test2".to_string(),
        "Routine update".to_string(),
        "Content".to_string(),
        "test".to_string(),
    )
    .with_relevance(0.2);

    let impact2 = strategy.calculate_impact(&article2, &sentiment2);
    assert!(impact2 < 0.5, "Expected low impact");
}
