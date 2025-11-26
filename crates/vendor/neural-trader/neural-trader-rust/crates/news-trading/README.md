# News Trading System

A comprehensive Rust-based news aggregation and event-driven trading system with sentiment analysis.

## Features

- **Multi-Source News Aggregation**: Integrate news from Alpaca, Polygon, NewsAPI, Reddit, Twitter, and RSS feeds
- **Sentiment Analysis**: Advanced financial lexicon-based sentiment analysis with detailed scoring
- **Event Detection**: Automatic detection of market-moving events (earnings, M&A, regulatory)
- **Trading Signal Generation**: Event-driven trading signals with confidence scoring
- **News Database**: Persistent storage with sled database and efficient querying
- **Backtesting**: Historical analysis and signal validation
- **Real-time Processing**: Async architecture for high-performance news processing

## Architecture

```
news-trading/
├── src/
│   ├── aggregator.rs     # News aggregation and caching
│   ├── sources/          # News source integrations
│   │   ├── alpaca.rs     # Alpaca News API
│   │   ├── polygon.rs    # Polygon News API
│   │   ├── newsapi.rs    # NewsAPI integration
│   │   └── social.rs     # Reddit, Twitter, RSS
│   ├── sentiment/        # Sentiment analysis
│   │   ├── analyzer.rs   # Core analysis engine
│   │   └── models.rs     # Sentiment models
│   ├── strategy.rs       # Trading strategy logic
│   ├── database.rs       # News persistence
│   ├── models.rs         # Data structures
│   └── error.rs          # Error handling
├── tests/                # Comprehensive test suite
└── examples/             # Usage examples
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
nt-news-trading = { path = "../news-trading" }
```

## Quick Start

### Basic Usage

```rust
use nt_news_trading::{NewsAggregator, NewsTradingStrategy};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize aggregator
    let aggregator = Arc::new(NewsAggregator::new());

    // Create trading strategy
    let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator.clone());

    // Fetch news for symbols
    let symbols = vec!["AAPL".to_string(), "MSFT".to_string()];
    let articles = aggregator.fetch_news(&symbols).await?;

    // Generate trading signals
    for article in articles {
        if let Some(signal) = strategy.on_news(article).await? {
            println!("Signal: {} - {:?} (confidence: {:.2})",
                signal.symbol, signal.direction, signal.confidence);
        }
    }

    Ok(())
}
```

### With News Sources

```rust
use nt_news_trading::{NewsAggregator, sources::*};
use std::sync::Arc;

let mut aggregator = NewsAggregator::new();

// Add Alpaca News
let alpaca = AlpacaNews::new(
    "your-api-key".to_string(),
    "your-secret".to_string()
);
aggregator.add_source(Arc::new(alpaca));

// Add Polygon News
let polygon = PolygonNews::new("your-api-key".to_string());
aggregator.add_source(Arc::new(polygon));

// Add NewsAPI
let newsapi = NewsAPISource::new("your-api-key".to_string());
aggregator.add_source(Arc::new(newsapi));
```

### Sentiment Analysis

```rust
use nt_news_trading::SentimentAnalyzer;

let analyzer = SentimentAnalyzer::default();

let sentiment = analyzer.analyze("The stock shows strong growth with bullish momentum");

println!("Score: {:.2}", sentiment.score);           // 0.75
println!("Magnitude: {:.2}", sentiment.magnitude);   // 0.85
println!("Label: {:?}", sentiment.label);            // VeryPositive

if sentiment.is_positive() && sentiment.is_strong() {
    println!("Strong positive sentiment detected!");
}
```

### Database Operations

```rust
use nt_news_trading::{NewsDB, NewsQuery};

// Create database
let db = NewsDB::new("./news_data")?;

// Store articles
db.store(&article)?;

// Query by symbol
let query = NewsQuery::new()
    .with_symbols(vec!["AAPL".to_string()])
    .with_min_relevance(0.7);

let results = db.query(&query)?;

// Get history
let history = db.get_history("AAPL", 7)?;  // Last 7 days
```

### Event Detection

```rust
use nt_news_trading::{NewsTradingStrategy, EventCategory};

let strategy = NewsTradingStrategy::default_with_aggregator(aggregator);

let article = NewsArticle::new(
    "1".to_string(),
    "Company announces quarterly earnings beat".to_string(),
    "Exceeded expectations with strong growth".to_string(),
    "source".to_string(),
);

let sentiment = analyzer.analyze(&format!("{} {}", article.title, article.content));
let impact = strategy.calculate_impact(&article, &sentiment);

println!("Impact score: {:.2}", impact);  // High score for earnings news
```

## News Sources

### Alpaca News API

```rust
use nt_news_trading::sources::AlpacaNews;

let alpaca = AlpacaNews::new(
    api_key.to_string(),
    api_secret.to_string()
);

let articles = alpaca.fetch(&symbols).await?;
```

**Requirements:**
- Alpaca API key and secret
- Environment variable: `ALPACA_SECRET`

### Polygon News API

```rust
use nt_news_trading::sources::PolygonNews;

let polygon = PolygonNews::new(api_key.to_string());
let articles = polygon.fetch(&symbols).await?;
```

**Requirements:**
- Polygon.io API key

### NewsAPI

```rust
use nt_news_trading::sources::NewsAPISource;

let newsapi = NewsAPISource::new(api_key.to_string());
let articles = newsapi.fetch(&symbols).await?;
```

**Requirements:**
- NewsAPI.org API key
- Note: Free tier has rate limits

### Social Media (Reddit, Twitter)

```rust
use nt_news_trading::sources::{RedditScraper, TwitterStream};

// Reddit
let reddit = RedditScraper::new()
    .with_subreddits(vec![
        "wallstreetbets".to_string(),
        "stocks".to_string()
    ]);

// Twitter (requires bearer token)
let twitter = TwitterStream::new(Some(bearer_token));
```

## Sentiment Analysis

The sentiment analyzer uses a financial-specific lexicon with 50+ weighted terms:

**Positive Terms (Examples):**
- bullish (0.8), profit (0.7), growth (0.7), surge (0.8)
- rally (0.7), gain (0.6), beat (0.7), outperform (0.8)

**Negative Terms (Examples):**
- bearish (-0.8), loss (-0.7), crash (-0.9), plunge (-0.8)
- decline (-0.6), miss (-0.7), underperform (-0.8), crisis (-0.9)

**Sentiment Structure:**
```rust
pub struct Sentiment {
    pub score: f64,      // -1.0 to 1.0
    pub magnitude: f64,  // 0.0 to 1.0 (confidence)
    pub label: SentimentLabel,  // VeryNegative to VeryPositive
}
```

## Trading Strategy

### Signal Generation

```rust
pub struct TradingSignal {
    pub symbol: String,
    pub direction: Direction,     // Long, Short, or Neutral
    pub confidence: f64,          // 0.0 to 1.0
    pub reason: String,
    pub news_id: String,
    pub sentiment_score: f64,
    pub impact_score: f64,
}
```

### Configuration

```rust
let config = StrategyConfig {
    min_impact_threshold: 0.5,        // Minimum impact score
    min_sentiment_magnitude: 0.4,     // Minimum confidence
    min_confidence: 0.6,              // Overall confidence threshold
    max_signals_per_day: 10,          // Signal limit
};

let strategy = NewsTradingStrategy::new(aggregator, analyzer, config);
```

### Event Detection

The strategy automatically detects high-impact events:

- **Earnings** (0.8 weight): earnings, quarterly results, guidance
- **M&A** (0.9 weight): merger, acquisition, takeover
- **Regulatory** (0.7 weight): FDA approval, investigation
- **Product** (0.6 weight): product launch, new product
- **Leadership** (0.5 weight): CEO changes, resignations

## Backtesting

```rust
let symbols = vec!["AAPL".to_string(), "MSFT".to_string()];
let results = strategy.backtest(&symbols, 30).await?;

println!("{}", results.summary());
// Output:
// Backtest Results:
//   Total signals: 45
//   Long signals: 28
//   Short signals: 17

for signal in results.signals() {
    println!("{} - {:?} ({:.2})",
        signal.symbol, signal.direction, signal.confidence);
}
```

## API Reference

### Core Types

- `NewsAggregator`: Multi-source news aggregation with caching
- `NewsArticle`: Article data structure with metadata
- `SentimentAnalyzer`: Sentiment analysis engine
- `NewsTradingStrategy`: Trading signal generation
- `NewsDB`: Persistent news storage
- `TradingSignal`: Trading recommendation output

### Key Methods

**NewsAggregator:**
- `fetch_news(symbols)`: Fetch from all sources
- `fetch_with_sentiment(symbols, min_score)`: Filtered fetch
- `query(query)`: Query cached news
- `clear_cache()`: Clear article cache

**SentimentAnalyzer:**
- `analyze(text)`: Single text analysis
- `analyze_batch(texts)`: Batch analysis
- `analyze_detailed(text)`: Detailed breakdown

**NewsTradingStrategy:**
- `on_news(article)`: Process article, generate signal
- `calculate_impact(article, sentiment)`: Impact scoring
- `backtest(symbols, days)`: Historical analysis

## Testing

Run the comprehensive test suite:

```bash
# All tests
cargo test -p nt-news-trading

# Specific test file
cargo test -p nt-news-trading --test sentiment_tests

# With output
cargo test -p nt-news-trading -- --nocapture
```

**Test Coverage:**
- 20+ unit tests
- Sentiment analysis validation
- Database operations
- Strategy logic
- Signal generation

## Example Output

Running the demo example:

```bash
cargo run --example news_trading_demo
```

```
=== News Trading System Demo ===

✓ Initialized aggregator, strategy, and database

Created 8 sample news articles

✓ Stored articles in database

=== Processing News Articles ===

Article: Apple Reports Record Quarterly Earnings, Beats Expectations
  Source: financial_times
  Symbols: ["AAPL"]
  ✓ Signal Generated:
    Symbol: AAPL
    Direction: Long
    Confidence: 0.87
    Sentiment Score: 0.75
    Impact Score: 0.92
    Reason: News-based signal: very_positive sentiment (score: 0.75, magnitude: 0.85), impact: 0.92

[... more articles ...]

=== Summary ===
Total articles processed: 8
Trading signals generated: 6

=== Backtesting ===
Backtest Results:
  Total signals: 6
  Long signals: 4
  Short signals: 2

Signal breakdown:
  LONG  AAPL (confidence: 0.87)
  LONG  MSFT (confidence: 0.82)
  SHORT GOOGL (confidence: 0.74)
  SHORT TSLA (confidence: 0.79)
  LONG  AMZN (confidence: 0.76)
  LONG  META (confidence: 0.84)
```

## Performance Considerations

- **Async/Await**: All I/O operations are async for high throughput
- **Caching**: In-memory LRU cache for recent articles
- **Batching**: Batch processing for sentiment analysis
- **Rate Limiting**: Built-in delays for API compliance
- **Indexing**: Database indices on symbols and dates

## Error Handling

All operations return `Result<T, NewsError>`:

```rust
match aggregator.fetch_news(&symbols).await {
    Ok(articles) => process_articles(articles),
    Err(NewsError::RateLimit(msg)) => {
        // Handle rate limit
    },
    Err(NewsError::Network(msg)) => {
        // Handle network error
    },
    Err(e) => {
        // Handle other errors
    }
}
```

## Future Enhancements

- [ ] Machine learning-based sentiment models
- [ ] Real-time streaming via WebSocket
- [ ] Advanced event classification
- [ ] Multi-language support
- [ ] Custom lexicon loading
- [ ] Performance metrics tracking

## License

Part of the Neural Trader project.

## Contributing

See the main project README for contribution guidelines.
