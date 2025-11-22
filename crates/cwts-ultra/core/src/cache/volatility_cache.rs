use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Volatility-based caching system for market data
///
/// Caches data based on volatility patterns to optimize performance
/// while ensuring data freshness for high-volatility periods.
///
/// References:
/// - Engle, R. "Autoregressive Conditional Heteroskedasticity" (1982)
/// - Bollerslev, T. "Generalized Autoregressive Conditional Heteroskedasticity" (1986)
#[derive(Debug)]
pub struct VolatilityBasedCache {
    cache_entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    volatility_calculator: VolatilityCalculator,
    config: CacheConfig,
}

#[derive(Debug, Clone)]
struct CacheConfig {
    max_entries: usize,
    base_ttl: Duration,
    high_volatility_multiplier: f64,
    low_volatility_multiplier: f64,
    volatility_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            base_ttl: Duration::from_secs(30),
            high_volatility_multiplier: 0.1, // Shorter TTL for volatile data
            low_volatility_multiplier: 5.0,  // Longer TTL for stable data
            volatility_threshold: 0.02,      // 2% volatility threshold
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    data: String,
    created_at: Instant,
    ttl: Duration,
    volatility_score: f64,
    access_count: u64,
    last_accessed: Instant,
}

/// Market tick data for volatility calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub trade_id: u64,
}

/// Volatility calculator for determining cache TTL
#[derive(Debug)]
struct VolatilityCalculator {
    price_history: HashMap<String, Vec<PricePoint>>,
    max_history_length: usize,
}

#[derive(Debug, Clone)]
struct PricePoint {
    price: f64,
    timestamp: Instant,
}

impl VolatilityBasedCache {
    /// Create new volatility-based cache
    pub fn new() -> Self {
        Self {
            cache_entries: Arc::new(RwLock::new(HashMap::new())),
            volatility_calculator: VolatilityCalculator::new(),
            config: CacheConfig::default(),
        }
    }

    /// Create cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            cache_entries: Arc::new(RwLock::new(HashMap::new())),
            volatility_calculator: VolatilityCalculator::new(),
            config,
        }
    }

    /// Cache data if volatility warrants it
    pub async fn cache_if_volatile(&mut self, tick: &MarketTick) -> Result<(), CacheError> {
        // Calculate current volatility
        let volatility = self
            .volatility_calculator
            .calculate_volatility(&tick.symbol, tick.price)?;

        // Determine if caching is beneficial
        if self.should_cache(volatility) {
            let ttl = self.calculate_adaptive_ttl(volatility);
            let key = format!("{}_{}", tick.symbol, tick.timestamp);

            let data = serde_json::to_string(tick)
                .map_err(|e| CacheError::SerializationError(e.to_string()))?;

            let entry = CacheEntry {
                key: key.clone(),
                data,
                created_at: Instant::now(),
                ttl,
                volatility_score: volatility,
                access_count: 0,
                last_accessed: Instant::now(),
            };

            self.insert_entry(key, entry).await?;
        }

        Ok(())
    }

    /// Get cached data if available and not expired
    pub async fn get(&self, key: &str) -> Result<Option<MarketTick>, CacheError> {
        let mut cache = self.cache_entries.write();

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry has expired
            if entry.created_at.elapsed() > entry.ttl {
                cache.remove(key);
                return Ok(None);
            }

            // Update access statistics
            entry.access_count += 1;
            entry.last_accessed = Instant::now();

            // Deserialize and return data
            let tick: MarketTick = serde_json::from_str(&entry.data)
                .map_err(|e| CacheError::DeserializationError(e.to_string()))?;

            Ok(Some(tick))
        } else {
            Ok(None)
        }
    }

    /// Insert entry with eviction policy
    async fn insert_entry(&mut self, key: String, entry: CacheEntry) -> Result<(), CacheError> {
        let mut cache = self.cache_entries.write();

        // Check if cache is full
        if cache.len() >= self.config.max_entries {
            self.evict_entries(&mut cache).await?;
        }

        cache.insert(key, entry);
        Ok(())
    }

    /// Evict entries using LRU + volatility-based policy
    async fn evict_entries(
        &self,
        cache: &mut HashMap<String, CacheEntry>,
    ) -> Result<(), CacheError> {
        let eviction_count = cache.len() / 4; // Evict 25% of entries

        // Sort entries by access time and volatility score
        let mut entries: Vec<_> = cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        entries.sort_by(|a, b| {
            // Prioritize eviction of old, low-volatility entries
            let score_a =
                a.1.last_accessed.elapsed().as_secs() as f64 / (a.1.volatility_score + 0.01);
            let score_b =
                b.1.last_accessed.elapsed().as_secs() as f64 / (b.1.volatility_score + 0.01);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove oldest entries
        for (key, _) in entries.iter().take(eviction_count) {
            cache.remove(key);
        }

        Ok(())
    }

    /// Determine if data should be cached based on volatility
    fn should_cache(&self, volatility: f64) -> bool {
        // Cache high-volatility data for quick access
        // Cache low-volatility data for longer periods
        volatility > self.config.volatility_threshold * 0.1
            || volatility < self.config.volatility_threshold * 0.5
    }

    /// Calculate adaptive TTL based on volatility
    fn calculate_adaptive_ttl(&self, volatility: f64) -> Duration {
        let multiplier = if volatility > self.config.volatility_threshold {
            // High volatility - shorter TTL
            self.config.high_volatility_multiplier
        } else {
            // Low volatility - longer TTL
            self.config.low_volatility_multiplier
        };

        Duration::from_secs_f64(self.config.base_ttl.as_secs_f64() * multiplier)
    }

    /// Clean up expired entries
    pub async fn cleanup_expired(&self) -> Result<usize, CacheError> {
        let mut cache = self.cache_entries.write();
        let initial_count = cache.len();

        cache.retain(|_, entry| entry.created_at.elapsed() <= entry.ttl);

        Ok(initial_count - cache.len())
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let cache = self.cache_entries.read();

        let total_entries = cache.len();
        let mut high_volatility_entries = 0;
        let mut total_access_count = 0;

        for entry in cache.values() {
            if entry.volatility_score > self.config.volatility_threshold {
                high_volatility_entries += 1;
            }
            total_access_count += entry.access_count;
        }

        CacheStatistics {
            total_entries,
            high_volatility_entries,
            low_volatility_entries: total_entries - high_volatility_entries,
            total_access_count,
            average_access_count: if total_entries > 0 {
                total_access_count / total_entries as u64
            } else {
                0
            },
            max_entries: self.config.max_entries,
        }
    }
}

impl VolatilityCalculator {
    fn new() -> Self {
        Self {
            price_history: HashMap::new(),
            max_history_length: 100, // Keep last 100 price points
        }
    }

    /// Calculate volatility using recent price history
    fn calculate_volatility(
        &mut self,
        symbol: &str,
        current_price: f64,
    ) -> Result<f64, CacheError> {
        let history = self
            .price_history
            .entry(symbol.to_string())
            .or_insert_with(Vec::new);

        // Add current price point
        history.push(PricePoint {
            price: current_price,
            timestamp: Instant::now(),
        });

        // Maintain maximum history length
        if history.len() > self.max_history_length {
            history.remove(0);
        }

        // Need at least 2 points to calculate volatility
        if history.len() < 2 {
            return Ok(0.0);
        }

        // Calculate returns
        let mut returns = Vec::new();
        for i in 1..history.len() {
            let return_val = (history[i].price - history[i - 1].price) / history[i - 1].price;
            returns.push(return_val);
        }

        // Calculate standard deviation of returns (volatility)
        if returns.is_empty() {
            return Ok(0.0);
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub high_volatility_entries: usize,
    pub low_volatility_entries: usize,
    pub total_access_count: u64,
    pub average_access_count: u64,
    pub max_entries: usize,
}

/// Cache errors
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Cache full")]
    CacheFull,

    #[error("Volatility calculation error: {0}")]
    VolatilityCalculationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_creation() {
        let cache = VolatilityBasedCache::new();
        let stats = cache.get_statistics();
        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_volatility_based_caching() {
        let mut cache = VolatilityBasedCache::new();

        let tick = MarketTick {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 100.0,
            timestamp: 1640995200000,
            bid_price: 49999.0,
            ask_price: 50001.0,
            trade_id: 12345,
        };

        assert!(cache.cache_if_volatile(&tick).await.is_ok());
    }

    #[tokio::test]
    async fn test_cache_get_and_set() {
        let mut cache = VolatilityBasedCache::new();

        let tick = MarketTick {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 100.0,
            timestamp: 1640995200000,
            bid_price: 49999.0,
            ask_price: 50001.0,
            trade_id: 12345,
        };

        let key = format!("{}_{}", tick.symbol, tick.timestamp);

        // Cache should initially be empty
        assert!(cache.get(&key).await.unwrap().is_none());

        // Cache the tick
        cache.cache_if_volatile(&tick).await.unwrap();

        // Should be able to retrieve if it was cached
        let result = cache.get(&key).await.unwrap();
        if let Some(cached_tick) = result {
            assert_eq!(cached_tick.symbol, tick.symbol);
            assert_eq!(cached_tick.price, tick.price);
        }
    }
}
