use nt_news_trading::{
    Direction, NewsAggregator, NewsArticle, NewsDB, NewsTradingStrategy,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== News Trading System Demo ===\n");

    // Initialize components
    let aggregator = Arc::new(NewsAggregator::new());
    let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator.clone());
    let db = NewsDB::in_memory()?;

    println!("✓ Initialized aggregator, strategy, and database\n");

    // Create sample news articles
    let articles = create_sample_articles();
    println!("Created {} sample news articles\n", articles.len());

    // Store articles in database
    db.store_batch(&articles)?;
    println!("✓ Stored articles in database\n");

    // Process each article and generate signals
    println!("=== Processing News Articles ===\n");

    let mut signals_generated = 0;

    for article in articles {
        println!("Article: {}", article.title);
        println!("  Source: {}", article.source);
        println!("  Symbols: {:?}", article.symbols);

        match strategy.on_news(article).await? {
            Some(signal) => {
                signals_generated += 1;
                println!("  ✓ Signal Generated:");
                println!("    Symbol: {}", signal.symbol);
                println!("    Direction: {:?}", signal.direction);
                println!("    Confidence: {:.2}", signal.confidence);
                println!("    Sentiment Score: {:.2}", signal.sentiment_score);
                println!("    Impact Score: {:.2}", signal.impact_score);
                println!("    Reason: {}", signal.reason);
            }
            None => {
                println!("  - No signal (insufficient impact or confidence)");
            }
        }
        println!();
    }

    // Print summary
    println!("=== Summary ===");
    println!("Total articles processed: {}", db.count());
    println!("Trading signals generated: {}", signals_generated);

    // Demonstrate backtesting
    println!("\n=== Backtesting ===");

    let symbols = vec!["AAPL".to_string(), "MSFT".to_string(), "GOOGL".to_string()];
    let backtest_results = strategy.backtest(&symbols, 7).await?;

    println!("{}", backtest_results.summary());
    println!("\nSignal breakdown:");

    for signal in backtest_results.signals() {
        match signal.direction {
            Direction::Long => println!("  LONG  {} (confidence: {:.2})", signal.symbol, signal.confidence),
            Direction::Short => println!("  SHORT {} (confidence: {:.2})", signal.symbol, signal.confidence),
            Direction::Neutral => println!("  HOLD  {}", signal.symbol),
        }
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn create_sample_articles() -> Vec<NewsArticle> {
    vec![
        NewsArticle::new(
            "1".to_string(),
            "Apple Reports Record Quarterly Earnings, Beats Expectations".to_string(),
            "Apple Inc. announced exceptional quarterly results with strong growth across all product lines. Revenue surged past analyst expectations driven by robust iPhone sales.".to_string(),
            "financial_times".to_string(),
        )
        .with_symbols(vec!["AAPL".to_string()])
        .with_relevance(0.95),

        NewsArticle::new(
            "2".to_string(),
            "Microsoft Announces Major Acquisition in AI Sector".to_string(),
            "Microsoft Corporation revealed plans to acquire a leading AI startup, marking a significant expansion in artificial intelligence capabilities.".to_string(),
            "reuters".to_string(),
        )
        .with_symbols(vec!["MSFT".to_string()])
        .with_relevance(0.90),

        NewsArticle::new(
            "3".to_string(),
            "Google Faces Regulatory Investigation Over Market Practices".to_string(),
            "Alphabet's Google is under investigation by regulators examining potential antitrust violations. The company faces uncertainty as the probe continues.".to_string(),
            "bloomberg".to_string(),
        )
        .with_symbols(vec!["GOOGL".to_string()])
        .with_relevance(0.85),

        NewsArticle::new(
            "4".to_string(),
            "Tesla Stock Plunges on Production Concerns and Delivery Miss".to_string(),
            "Tesla shares experienced a significant decline following disappointing production numbers and missed delivery targets. Analysts express bearish outlook.".to_string(),
            "cnbc".to_string(),
        )
        .with_symbols(vec!["TSLA".to_string()])
        .with_relevance(0.88),

        NewsArticle::new(
            "5".to_string(),
            "Amazon Launches Revolutionary Cloud Service Platform".to_string(),
            "Amazon Web Services unveiled a groundbreaking cloud platform featuring advanced AI integration. The launch demonstrates strong innovation momentum.".to_string(),
            "techcrunch".to_string(),
        )
        .with_symbols(vec!["AMZN".to_string()])
        .with_relevance(0.82),

        NewsArticle::new(
            "6".to_string(),
            "NVIDIA CEO Discusses Market Position in Routine Interview".to_string(),
            "NVIDIA's chief executive provided general commentary on the semiconductor market during a scheduled media appearance.".to_string(),
            "marketwatch".to_string(),
        )
        .with_symbols(vec!["NVDA".to_string()])
        .with_relevance(0.45),

        NewsArticle::new(
            "7".to_string(),
            "Meta Platforms Exceeds User Growth Expectations, Stock Rallies".to_string(),
            "Meta reported outstanding user engagement metrics with bullish momentum. The company's platform expansion shows exceptional growth trajectory.".to_string(),
            "wsj".to_string(),
        )
        .with_symbols(vec!["META".to_string()])
        .with_relevance(0.91),

        NewsArticle::new(
            "8".to_string(),
            "Banking Sector Remains Stable Amid Economic Uncertainty".to_string(),
            "Major financial institutions maintain steady performance despite ongoing economic concerns. Market observers note cautious optimism.".to_string(),
            "financial_times".to_string(),
        )
        .with_symbols(vec!["JPM".to_string(), "BAC".to_string()])
        .with_relevance(0.55),
    ]
}
