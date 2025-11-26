//! News Trading System
//!
//! A comprehensive system for news aggregation, sentiment analysis, and event-driven trading.
//!
//! ## Features
//!
//! - Multi-source news aggregation (Alpaca, Polygon, NewsAPI, social media)
//! - Real-time sentiment analysis with financial lexicon
//! - Event detection (earnings, M&A, regulatory)
//! - Trading signal generation
//! - News database with persistence
//! - Backtesting support
//!
//! ## Example
//!
//! ```rust,no_run
//! use nt_news_trading::{NewsAggregator, NewsTradingStrategy, StrategyConfig};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let aggregator = Arc::new(NewsAggregator::new());
//!     let mut strategy = NewsTradingStrategy::default_with_aggregator(aggregator.clone());
//!
//!     // Fetch news
//!     let symbols = vec!["AAPL".to_string(), "MSFT".to_string()];
//!     let articles = aggregator.fetch_news(&symbols).await?;
//!
//!     // Generate signals
//!     for article in articles {
//!         if let Some(signal) = strategy.on_news(article).await? {
//!             println!("Signal: {:?}", signal);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod aggregator;
pub mod database;
pub mod error;
pub mod models;
pub mod sentiment;
pub mod sources;
pub mod strategy;

pub use aggregator::{NewsAggregator, NewsCache};
pub use database::NewsDB;
pub use error::{NewsError, Result};
pub use models::{
    Direction, EventCategory, NewsArticle, NewsQuery, Sentiment, SentimentLabel, TradingSignal,
};
pub use sentiment::SentimentAnalyzer;
pub use sources::{NewsSource, SourceConfig};
pub use strategy::{BacktestResults, NewsTradingStrategy, StrategyConfig};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
