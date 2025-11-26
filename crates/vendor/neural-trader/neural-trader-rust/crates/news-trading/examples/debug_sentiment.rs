use nt_news_trading::{NewsArticle, NewsTradingStrategy, NewsAggregator, SentimentAnalyzer};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let analyzer = SentimentAnalyzer::default();

    let test_texts = vec![
        ("Apple Reports Record Quarterly Earnings, Beats Expectations", "Apple Inc. announced exceptional quarterly results with strong growth across all product lines. Revenue surged past analyst expectations driven by robust iPhone sales."),
        ("Tesla Stock Plunges on Production Concerns", "Tesla shares experienced a significant decline following disappointing production numbers and missed delivery targets. Analysts express bearish outlook."),
    ];

    for (title, content) in test_texts {
        let text = format!("{} {}", title, content);
        let sentiment = analyzer.analyze(&text);

        println!("\nTitle: {}", title);
        println!("Score: {:.3}, Magnitude: {:.3}, Label: {:?}",
            sentiment.score, sentiment.magnitude, sentiment.label);
        println!("Is positive: {}, Is negative: {}, Is strong: {}",
            sentiment.is_positive(), sentiment.is_negative(), sentiment.is_strong());

        // Now test with strategy
        let article = NewsArticle::new(
            "test".to_string(),
            title.to_string(),
            content.to_string(),
            "test".to_string(),
        )
        .with_symbols(vec!["TEST".to_string()])
        .with_relevance(0.9);

        let aggregator = Arc::new(NewsAggregator::new());
        let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

        let impact = strategy.calculate_impact(&article, &sentiment);
        println!("Impact score: {:.3}", impact);

        match strategy.on_news(article).await {
            Ok(Some(signal)) => {
                println!("✓ Signal: {:?} with confidence {:.3}", signal.direction, signal.confidence);
            }
            Ok(None) => {
                println!("✗ No signal generated");
            }
            Err(e) => {
                println!("✗ Error: {}", e);
            }
        }
    }
}
