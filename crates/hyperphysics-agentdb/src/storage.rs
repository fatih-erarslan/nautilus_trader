//! Trading-specific storage layer built on ruvector-core's AgenticDB
//!
//! Provides persistent storage for trading episodes, strategies, and causal models
//! with optimizations for time-series retrieval and market-aware indexing.

use ruvector_core::agenticdb::AgenticDB;
use ruvector_core::types::{DbOptions, SearchQuery, SearchResult, VectorEntry};
use ruvector_core::error::RuvectorError;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

// Re-export core types
pub use ruvector_core::agenticdb::{
    ReflexionEpisode,
    Skill,
    CausalEdge,
    LearningSession,
    Experience,
    Prediction,
    UtilitySearchResult,
};

pub use ruvector_core::types::{
    DbOptions as CoreDbOptions,
    VectorEntry as CoreVectorEntry,
    SearchQuery as CoreSearchQuery,
    SearchResult as CoreSearchResult,
    DistanceMetric,
};

/// Trading-optimized storage with time-series indexing and market context
pub struct TradingStorage {
    /// Core AgenticDB instance
    inner: AgenticDB,
    /// Dimensions for embeddings
    dimensions: usize,
    /// Time-series index: timestamp -> episode IDs
    time_index: Arc<RwLock<TimeSeriesIndex>>,
    /// Symbol index: symbol -> episode IDs
    symbol_index: Arc<RwLock<SymbolIndex>>,
    /// Strategy index: strategy -> episode IDs
    strategy_index: Arc<RwLock<StrategyIndex>>,
}

/// Time-series index for temporal queries
#[derive(Debug, Default)]
pub struct TimeSeriesIndex {
    /// Buckets by day (Unix timestamp / 86400)
    daily_buckets: HashMap<i64, Vec<String>>,
    /// Buckets by hour for recent data
    hourly_buckets: HashMap<i64, Vec<String>>,
}

impl TimeSeriesIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add episode to time index
    pub fn insert(&mut self, timestamp: i64, episode_id: &str) {
        let day_bucket = timestamp / 86400;
        let hour_bucket = timestamp / 3600;
        
        self.daily_buckets
            .entry(day_bucket)
            .or_default()
            .push(episode_id.to_string());
        
        self.hourly_buckets
            .entry(hour_bucket)
            .or_default()
            .push(episode_id.to_string());
    }

    /// Query episodes in time range
    pub fn query_range(&self, start: i64, end: i64) -> Vec<String> {
        let start_day = start / 86400;
        let end_day = end / 86400;
        
        let mut results = Vec::new();
        for day in start_day..=end_day {
            if let Some(ids) = self.daily_buckets.get(&day) {
                results.extend(ids.clone());
            }
        }
        results
    }

    /// Get recent episodes (last N hours)
    pub fn recent(&self, hours: i64) -> Vec<String> {
        let now = chrono::Utc::now().timestamp();
        let cutoff = (now / 3600) - hours;
        
        let mut results = Vec::new();
        for (hour, ids) in &self.hourly_buckets {
            if *hour >= cutoff {
                results.extend(ids.clone());
            }
        }
        results
    }
}

/// Symbol-based index for instrument-specific queries
#[derive(Debug, Default)]
pub struct SymbolIndex {
    /// Symbol -> list of episode IDs
    symbols: HashMap<String, Vec<String>>,
}

impl SymbolIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, symbol: &str, episode_id: &str) {
        self.symbols
            .entry(symbol.to_uppercase())
            .or_default()
            .push(episode_id.to_string());
    }

    pub fn query(&self, symbol: &str) -> Vec<String> {
        self.symbols
            .get(&symbol.to_uppercase())
            .cloned()
            .unwrap_or_default()
    }

    pub fn symbols(&self) -> Vec<String> {
        self.symbols.keys().cloned().collect()
    }
}

/// Strategy-based index
#[derive(Debug, Default)]
pub struct StrategyIndex {
    /// Strategy name -> list of episode IDs
    strategies: HashMap<String, Vec<String>>,
    /// Strategy -> aggregated performance metrics
    performance: HashMap<String, StrategyPerformance>,
}

/// Aggregated strategy performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyPerformance {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub last_updated: i64,
}

impl StrategyIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, strategy: &str, episode_id: &str, pnl: f64) {
        let strategy_key = strategy.to_lowercase();
        
        self.strategies
            .entry(strategy_key.clone())
            .or_default()
            .push(episode_id.to_string());
        
        let perf = self.performance.entry(strategy_key).or_default();
        perf.total_trades += 1;
        if pnl > 0.0 {
            perf.winning_trades += 1;
        }
        perf.total_pnl += pnl;
        perf.last_updated = chrono::Utc::now().timestamp();
    }

    pub fn query(&self, strategy: &str) -> Vec<String> {
        self.strategies
            .get(&strategy.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    pub fn performance(&self, strategy: &str) -> Option<&StrategyPerformance> {
        self.performance.get(&strategy.to_lowercase())
    }

    pub fn all_strategies(&self) -> Vec<(String, StrategyPerformance)> {
        self.performance
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl TradingStorage {
    /// Create new trading storage at path
    pub fn new<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self, RuvectorError> {
        let mut options = DbOptions::default();
        options.storage_path = path.as_ref().to_string_lossy().to_string();
        options.dimensions = dimensions;

        let inner = AgenticDB::new(options)?;

        Ok(Self {
            inner,
            dimensions,
            time_index: Arc::new(RwLock::new(TimeSeriesIndex::new())),
            symbol_index: Arc::new(RwLock::new(SymbolIndex::new())),
            strategy_index: Arc::new(RwLock::new(StrategyIndex::new())),
        })
    }

    /// Store episode with indexing
    pub fn store_episode(
        &self,
        symbol: &str,
        strategy: &str,
        pnl: f64,
        timestamp: i64,
        task: String,
        actions: Vec<String>,
        observations: Vec<String>,
        critique: String,
    ) -> Result<String, RuvectorError> {
        let id = self.inner.store_episode(task, actions, observations, critique)?;
        
        // Update indices
        self.time_index.write().insert(timestamp, &id);
        self.symbol_index.write().insert(symbol, &id);
        self.strategy_index.write().insert(strategy, &id, pnl);
        
        Ok(id)
    }

    /// Query episodes by time range
    pub fn query_by_time(&self, start: i64, end: i64) -> Vec<String> {
        self.time_index.read().query_range(start, end)
    }

    /// Query episodes by symbol
    pub fn query_by_symbol(&self, symbol: &str) -> Vec<String> {
        self.symbol_index.read().query(symbol)
    }

    /// Query episodes by strategy
    pub fn query_by_strategy(&self, strategy: &str) -> Vec<String> {
        self.strategy_index.read().query(strategy)
    }

    /// Get recent episodes
    pub fn recent_episodes(&self, hours: i64) -> Vec<String> {
        self.time_index.read().recent(hours)
    }

    /// Get strategy performance
    pub fn strategy_performance(&self, strategy: &str) -> Option<StrategyPerformance> {
        self.strategy_index.read().performance(strategy).cloned()
    }

    /// Get all strategy performances
    pub fn all_strategy_performances(&self) -> Vec<(String, StrategyPerformance)> {
        self.strategy_index.read().all_strategies()
    }

    /// Get all traded symbols
    pub fn traded_symbols(&self) -> Vec<String> {
        self.symbol_index.read().symbols()
    }

    /// Semantic search across episodes
    pub fn semantic_search(&self, query: &str, k: usize) -> Result<Vec<ReflexionEpisode>, RuvectorError> {
        self.inner.retrieve_similar_episodes(query, k)
    }

    /// Search with utility scoring
    pub fn utility_search(
        &self,
        query: &str,
        k: usize,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Vec<UtilitySearchResult>, RuvectorError> {
        self.inner.query_with_utility(query, k, alpha, beta, gamma)
    }

    /// Create skill from patterns
    pub fn create_skill(
        &self,
        name: String,
        description: String,
        parameters: HashMap<String, String>,
        examples: Vec<String>,
    ) -> Result<String, RuvectorError> {
        self.inner.create_skill(name, description, parameters, examples)
    }

    /// Search skills
    pub fn search_skills(&self, query: &str, k: usize) -> Result<Vec<Skill>, RuvectorError> {
        self.inner.search_skills(query, k)
    }

    /// Add causal edge
    pub fn add_causal_edge(
        &self,
        causes: Vec<String>,
        effects: Vec<String>,
        confidence: f64,
        context: String,
    ) -> Result<String, RuvectorError> {
        self.inner.add_causal_edge(causes, effects, confidence, context)
    }

    /// Start learning session
    pub fn start_session(
        &self,
        algorithm: String,
        state_dim: usize,
        action_dim: usize,
    ) -> Result<String, RuvectorError> {
        self.inner.start_session(algorithm, state_dim, action_dim)
    }

    /// Add experience
    pub fn add_experience(
        &self,
        session_id: &str,
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f64,
        next_state: Vec<f32>,
        done: bool,
    ) -> Result<(), RuvectorError> {
        self.inner.add_experience(session_id, state, action, reward, next_state, done)
    }

    /// Predict with confidence
    pub fn predict(&self, session_id: &str, state: Vec<f32>) -> Result<Prediction, RuvectorError> {
        self.inner.predict_with_confidence(session_id, state)
    }

    /// Get session
    pub fn get_session(&self, session_id: &str) -> Result<Option<LearningSession>, RuvectorError> {
        self.inner.get_session(session_id)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_time_index() {
        let mut index = TimeSeriesIndex::new();
        let now = chrono::Utc::now().timestamp();
        
        index.insert(now, "ep1");
        index.insert(now - 3600, "ep2");  // 1 hour ago
        index.insert(now - 86400, "ep3"); // 1 day ago
        
        let recent = index.recent(2);
        assert!(recent.contains(&"ep1".to_string()));
        assert!(recent.contains(&"ep2".to_string()));
    }

    #[test]
    fn test_symbol_index() {
        let mut index = SymbolIndex::new();
        
        index.insert("BTC-USD", "ep1");
        index.insert("btc-usd", "ep2");  // Case insensitive
        index.insert("ETH-USD", "ep3");
        
        let btc = index.query("BTC-USD");
        assert_eq!(btc.len(), 2);
        
        let symbols = index.symbols();
        assert_eq!(symbols.len(), 2);
    }

    #[test]
    fn test_strategy_index() {
        let mut index = StrategyIndex::new();
        
        index.insert("momentum_v1", "ep1", 0.05);
        index.insert("momentum_v1", "ep2", -0.02);
        index.insert("mean_reversion", "ep3", 0.03);
        
        let perf = index.performance("momentum_v1").unwrap();
        assert_eq!(perf.total_trades, 2);
        assert_eq!(perf.winning_trades, 1);
        assert!((perf.total_pnl - 0.03).abs() < 0.001);
    }

    #[test]
    fn test_trading_storage() {
        let dir = tempdir().unwrap();
        let storage = TradingStorage::new(dir.path().join("test"), 128).unwrap();
        
        let id = storage.store_episode(
            "TEST-PAIR",
            "test_strategy",
            0.05,
            chrono::Utc::now().timestamp(),
            "Test trade".into(),
            vec!["action1".into()],
            vec!["obs1".into()],
            "Test critique".into(),
        ).unwrap();
        
        assert!(!id.is_empty());
        
        // Query by symbol
        let by_symbol = storage.query_by_symbol("TEST-PAIR");
        assert_eq!(by_symbol.len(), 1);
        
        // Query by strategy
        let by_strategy = storage.query_by_strategy("test_strategy");
        assert_eq!(by_strategy.len(), 1);
        
        // Check performance
        let perf = storage.strategy_performance("test_strategy").unwrap();
        assert_eq!(perf.total_trades, 1);
        assert_eq!(perf.winning_trades, 1);
    }
}
