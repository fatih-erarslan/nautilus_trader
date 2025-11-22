//! Volatility-based caching system
//!
//! Provides intelligent caching based on market volatility patterns
//! to optimize performance while maintaining data freshness.

pub mod volatility_cache;

pub use volatility_cache::{CacheError, CacheStatistics, MarketTick, VolatilityBasedCache};
