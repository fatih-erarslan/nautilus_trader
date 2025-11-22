//! Market data ingestion and processing

pub mod feed;
pub mod aggregator;
pub mod normalizer;
pub mod storage;

pub use feed::MarketDataFeed;
pub use aggregator::DataAggregator;
pub use normalizer::DataNormalizer;
pub use storage::DataStorage;