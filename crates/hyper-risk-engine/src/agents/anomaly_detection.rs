//! Anomaly detection agent for identifying market irregularities.
//!
//! Operates in the medium path (<1ms) using statistical methods to detect
//! anomalous market behavior, price movements, and trading patterns.
//!
//! ## Detection Methods
//!
//! 1. **Z-Score**: Statistical deviation from rolling mean (always available)
//! 2. **HNSW Pattern Detection**: Sub-μs similarity search for pattern anomalies
//!    (requires `similarity` feature)
//! 3. **MinHash Whale Detection**: Jaccard similarity for transaction pattern
//!    matching to detect whale flows (requires `similarity` feature)
//!
//! ## Scientific References
//!
//! - Chandola et al. (2009): "Anomaly Detection: A Survey" ACM Computing Surveys
//! - Malkov & Yashunin (2020): "Efficient and robust approximate nearest neighbor search
//!   using Hierarchical Navigable Small World graphs" IEEE TPAMI
//! - Broder (1997): "On the resemblance and containment of documents" (MinHash)

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, Price, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

// Conditional imports for similarity-based detection
#[cfg(feature = "similarity")]
use hyperphysics_similarity::{HybridIndex, SearchConfig, SearchMode, SearchResult};
#[cfg(feature = "similarity")]
use hyperphysics_lsh::{MinHash, HashFamily, hash::MinHashSignature};

/// Configuration for the anomaly detection agent.
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Z-score threshold for anomaly detection.
    pub zscore_threshold: f64,
    /// Lookback window for statistics calculation.
    pub lookback_window: usize,
    /// Minimum observations before detecting anomalies.
    pub min_observations: usize,
    /// Enable volume anomaly detection.
    pub detect_volume_anomalies: bool,
    /// Enable price spike detection.
    pub detect_price_spikes: bool,
    /// Enable HNSW pattern anomaly detection (requires similarity feature).
    pub detect_pattern_anomalies: bool,
    /// Enable whale flow detection via MinHash (requires similarity feature).
    pub detect_whale_flows: bool,
    /// HNSW pattern vector dimensions (price, volume, spread, etc.).
    pub pattern_dimensions: usize,
    /// Number of nearest neighbors to check for pattern anomaly.
    pub pattern_k_neighbors: usize,
    /// Distance threshold for pattern anomaly (above = anomalous).
    pub pattern_distance_threshold: f32,
    /// MinHash signature size for whale detection.
    pub minhash_signature_size: usize,
    /// Jaccard similarity threshold for whale pattern match.
    pub whale_jaccard_threshold: f32,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "anomaly_detection_agent".to_string(),
                enabled: true,
                priority: 3,
                max_latency_us: 1000, // 1ms
                verbose: false,
            },
            zscore_threshold: 3.0,
            lookback_window: 100,
            min_observations: 20,
            detect_volume_anomalies: true,
            detect_price_spikes: true,
            // Similarity-based detection (requires feature)
            detect_pattern_anomalies: true,
            detect_whale_flows: true,
            pattern_dimensions: 8, // [price, volume, spread, volatility, momentum, rsi, macd, vwap]
            pattern_k_neighbors: 5,
            pattern_distance_threshold: 0.8, // Distance above this = anomalous
            minhash_signature_size: 128,
            whale_jaccard_threshold: 0.7, // Similarity above this = potential whale pattern
        }
    }
}

/// Type of detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Price spike anomaly (z-score based).
    PriceSpike,
    /// Volume surge anomaly (z-score based).
    VolumeSurge,
    /// Spread widening anomaly.
    SpreadWidening,
    /// Correlation breakdown.
    CorrelationBreakdown,
    /// Liquidity drought.
    LiquidityDrought,
    /// Pattern anomaly detected via HNSW (no similar historical patterns).
    PatternAnomaly,
    /// Whale flow detected via MinHash (matches known whale transaction patterns).
    WhaleFlow,
}

/// Detected market anomaly.
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Symbol affected by anomaly.
    pub symbol: Symbol,
    /// Type of anomaly detected.
    pub anomaly_type: AnomalyType,
    /// Z-score of the anomalous observation (for statistical anomalies).
    pub zscore: f64,
    /// Current value.
    pub current_value: f64,
    /// Expected value (mean for z-score, nearest neighbor distance for pattern).
    pub expected_value: f64,
    /// Standard deviation (or pattern distance threshold).
    pub std_dev: f64,
    /// Detection timestamp.
    pub detected_at: Timestamp,
    /// Severity score (0.0 to 1.0).
    pub severity: f64,
    /// Pattern similarity score (for pattern/whale anomalies).
    pub pattern_similarity: Option<f64>,
    /// Number of similar patterns found (for HNSW detection).
    pub similar_pattern_count: Option<usize>,
}

/// Market pattern vector for HNSW-based anomaly detection.
///
/// Each dimension captures a different market characteristic:
/// [price_return, volume_ratio, spread_bps, volatility, momentum, rsi, macd, vwap_deviation]
#[cfg(feature = "similarity")]
#[derive(Debug, Clone)]
pub struct MarketPattern {
    /// Symbol this pattern belongs to.
    pub symbol: Symbol,
    /// Feature vector for similarity search.
    pub features: Vec<f32>,
    /// Timestamp of this pattern.
    pub timestamp: Timestamp,
}

#[cfg(feature = "similarity")]
impl MarketPattern {
    /// Create a new market pattern from raw features.
    pub fn new(symbol: Symbol, features: Vec<f32>, timestamp: Timestamp) -> Self {
        Self { symbol, features, timestamp }
    }

    /// Create from price and volume data.
    ///
    /// Features computed:
    /// - price_return: (current - prev) / prev
    /// - volume_ratio: current_vol / avg_vol
    /// - spread_bps: spread in basis points (placeholder)
    /// - volatility: rolling volatility estimate
    /// - momentum: price momentum (placeholder)
    /// - rsi: relative strength index (placeholder)
    /// - macd: MACD signal (placeholder)
    /// - vwap_deviation: deviation from VWAP (placeholder)
    pub fn from_market_data(
        symbol: Symbol,
        current_price: f64,
        prev_price: f64,
        current_volume: f64,
        avg_volume: f64,
        volatility: f64,
        timestamp: Timestamp,
    ) -> Self {
        let price_return = if prev_price > 0.0 {
            ((current_price - prev_price) / prev_price) as f32
        } else {
            0.0
        };

        let volume_ratio = if avg_volume > 0.0 {
            (current_volume / avg_volume) as f32
        } else {
            1.0
        };

        // Normalize features to roughly [-1, 1] range for similarity search
        let features = vec![
            price_return * 100.0,       // Scale returns to ~1 range
            (volume_ratio - 1.0).clamp(-2.0, 2.0), // Center around 0
            0.0,                         // spread_bps (placeholder)
            (volatility as f32).clamp(0.0, 1.0), // Normalized volatility
            0.0,                         // momentum (placeholder)
            0.0,                         // RSI normalized to [-1, 1] (placeholder)
            0.0,                         // MACD (placeholder)
            0.0,                         // VWAP deviation (placeholder)
        ];

        Self { symbol, features, timestamp }
    }
}

/// Whale transaction pattern for MinHash-based detection.
///
/// Represents a set of transaction characteristics that may indicate whale activity.
#[cfg(feature = "similarity")]
#[derive(Debug, Clone)]
pub struct WhalePattern {
    /// Transaction feature hashes (variable-size set for Jaccard similarity).
    pub feature_hashes: Vec<u64>,
    /// Volume of the transaction.
    pub volume: f64,
    /// Whether this is a known whale pattern.
    pub is_known_whale: bool,
}

#[cfg(feature = "similarity")]
impl WhalePattern {
    /// Create feature hashes from transaction characteristics.
    pub fn from_transaction(
        volume: f64,
        price_impact_bps: f64,
        order_size_percentile: f64,
        time_of_day_bucket: u32,
        is_aggressive: bool,
    ) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut feature_hashes = Vec::new();

        // Volume bucket (logarithmic)
        let volume_bucket = (volume.log10().max(0.0) * 10.0) as u64;
        feature_hashes.push(hash_feature("vol", volume_bucket));

        // Price impact bucket
        let impact_bucket = (price_impact_bps * 10.0) as u64;
        feature_hashes.push(hash_feature("impact", impact_bucket));

        // Order size percentile bucket
        let size_bucket = (order_size_percentile * 10.0) as u64;
        feature_hashes.push(hash_feature("size", size_bucket));

        // Time of day
        feature_hashes.push(hash_feature("time", time_of_day_bucket as u64));

        // Aggression indicator
        if is_aggressive {
            feature_hashes.push(hash_feature("aggressive", 1));
        }

        Self {
            feature_hashes,
            volume,
            is_known_whale: false,
        }
    }
}

#[cfg(feature = "similarity")]
fn hash_feature(prefix: &str, value: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    prefix.hash(&mut hasher);
    value.hash(&mut hasher);
    hasher.finish()
}

/// Running statistics calculator.
#[derive(Debug)]
struct RunningStats {
    values: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
    window_size: usize,
}

impl RunningStats {
    fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
            window_size,
        }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum / self.values.len() as f64
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let variance = (self.sum_sq / n) - (self.sum / n).powi(2);
        variance.max(0.0).sqrt()
    }

    fn zscore(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean()) / std
    }

    fn count(&self) -> usize {
        self.values.len()
    }
}

/// Anomaly detection agent.
///
/// Combines statistical (z-score) and similarity-based (HNSW, MinHash) detection methods
/// for comprehensive market anomaly identification.
pub struct AnomalyDetectionAgent {
    config: AnomalyDetectionConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Price statistics by symbol.
    price_stats: RwLock<std::collections::HashMap<Symbol, RunningStats>>,
    /// Volume statistics by symbol.
    volume_stats: RwLock<std::collections::HashMap<Symbol, RunningStats>>,
    /// Detected anomalies.
    anomalies: RwLock<Vec<Anomaly>>,
    /// Previous prices for pattern computation.
    prev_prices: RwLock<std::collections::HashMap<Symbol, f64>>,

    // ============================================================================
    // Similarity-Based Detection Components (feature-gated)
    // ============================================================================

    /// HNSW hybrid index for pattern anomaly detection.
    #[cfg(feature = "similarity")]
    pattern_index: RwLock<Option<HybridIndex>>,

    /// MinHash for whale flow detection.
    #[cfg(feature = "similarity")]
    whale_minhash: RwLock<Option<MinHash>>,

    /// Known whale pattern signatures for comparison.
    #[cfg(feature = "similarity")]
    whale_signatures: RwLock<Vec<MinHashSignature>>,

    /// Pattern count for HNSW index management.
    #[cfg(feature = "similarity")]
    pattern_count: std::sync::atomic::AtomicU64,
}

// Implement Debug manually since HybridIndex may not implement Debug
impl std::fmt::Debug for AnomalyDetectionAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnomalyDetectionAgent")
            .field("config", &self.config)
            .field("status", &self.status)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl AnomalyDetectionAgent {
    /// Create a new anomaly detection agent.
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            price_stats: RwLock::new(std::collections::HashMap::new()),
            volume_stats: RwLock::new(std::collections::HashMap::new()),
            anomalies: RwLock::new(Vec::new()),
            prev_prices: RwLock::new(std::collections::HashMap::new()),
            #[cfg(feature = "similarity")]
            pattern_index: RwLock::new(None),
            #[cfg(feature = "similarity")]
            whale_minhash: RwLock::new(None),
            #[cfg(feature = "similarity")]
            whale_signatures: RwLock::new(Vec::new()),
            #[cfg(feature = "similarity")]
            pattern_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AnomalyDetectionConfig::default())
    }

    /// Initialize similarity-based detection components.
    ///
    /// Must be called before using pattern or whale detection.
    /// This is separate from `new()` to allow lazy initialization and
    /// to avoid allocation overhead when similarity features are not used.
    #[cfg(feature = "similarity")]
    pub fn init_similarity(&self) -> Result<()> {
        // Initialize HNSW pattern index
        if self.config.detect_pattern_anomalies {
            let search_config = SearchConfig::trading();
            let mut index = HybridIndex::new(search_config)
                .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                    format!("Failed to create HNSW index: {}", e)
                ))?;

            // Initialize with pattern dimensions
            index.init_hnsw(self.config.pattern_dimensions)
                .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                    format!("Failed to init HNSW: {}", e)
                ))?;

            index.init_lsh()
                .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                    format!("Failed to init LSH: {}", e)
                ))?;

            *self.pattern_index.write() = Some(index);
            tracing::info!(
                dimensions = self.config.pattern_dimensions,
                "HNSW pattern index initialized"
            );
        }

        // Initialize MinHash for whale detection
        if self.config.detect_whale_flows {
            let minhash = MinHash::new(self.config.minhash_signature_size, 42);
            *self.whale_minhash.write() = Some(minhash);
            tracing::info!(
                signature_size = self.config.minhash_signature_size,
                "MinHash whale detector initialized"
            );
        }

        Ok(())
    }

    /// Check if similarity detection is initialized.
    #[cfg(feature = "similarity")]
    pub fn is_similarity_initialized(&self) -> bool {
        let pattern_ready = !self.config.detect_pattern_anomalies
            || self.pattern_index.read().is_some();
        let whale_ready = !self.config.detect_whale_flows
            || self.whale_minhash.read().is_some();
        pattern_ready && whale_ready
    }

    /// Record a price observation and check for anomalies.
    pub fn record_price(&self, symbol: &Symbol, price: Price) -> Option<Anomaly> {
        let price_f64 = price.as_f64();
        let mut stats = self.price_stats.write();

        let running = stats
            .entry(symbol.clone())
            .or_insert_with(|| RunningStats::new(self.config.lookback_window));

        let anomaly = if running.count() >= self.config.min_observations {
            let zscore = running.zscore(price_f64);
            if zscore.abs() >= self.config.zscore_threshold && self.config.detect_price_spikes {
                Some(Anomaly {
                    symbol: symbol.clone(),
                    anomaly_type: AnomalyType::PriceSpike,
                    zscore,
                    current_value: price_f64,
                    expected_value: running.mean(),
                    std_dev: running.std_dev(),
                    detected_at: Timestamp::now(),
                    severity: (zscore.abs() / 5.0).min(1.0),
                    pattern_similarity: None,
                    similar_pattern_count: None,
                })
            } else {
                None
            }
        } else {
            None
        };

        running.push(price_f64);

        // Update previous price for pattern computation
        self.prev_prices.write().insert(symbol.clone(), price_f64);

        if let Some(ref a) = anomaly {
            self.anomalies.write().push(a.clone());
        }

        anomaly
    }

    /// Record a volume observation and check for anomalies.
    pub fn record_volume(&self, symbol: &Symbol, volume: f64) -> Option<Anomaly> {
        let mut stats = self.volume_stats.write();

        let running = stats
            .entry(symbol.clone())
            .or_insert_with(|| RunningStats::new(self.config.lookback_window));

        let anomaly = if running.count() >= self.config.min_observations {
            let zscore = running.zscore(volume);
            if zscore >= self.config.zscore_threshold && self.config.detect_volume_anomalies {
                Some(Anomaly {
                    symbol: symbol.clone(),
                    anomaly_type: AnomalyType::VolumeSurge,
                    zscore,
                    current_value: volume,
                    expected_value: running.mean(),
                    std_dev: running.std_dev(),
                    detected_at: Timestamp::now(),
                    severity: (zscore / 5.0).min(1.0),
                    pattern_similarity: None,
                    similar_pattern_count: None,
                })
            } else {
                None
            }
        } else {
            None
        };

        running.push(volume);

        if let Some(ref a) = anomaly {
            self.anomalies.write().push(a.clone());
        }

        anomaly
    }

    // ============================================================================
    // HNSW Pattern-Based Anomaly Detection (requires similarity feature)
    // ============================================================================

    /// Check for pattern-based anomalies using HNSW similarity search.
    ///
    /// This method searches for similar historical patterns. If no close matches
    /// are found (distance > threshold), the pattern is flagged as anomalous.
    ///
    /// Target latency: <100μs (leveraging HNSW sub-μs queries)
    #[cfg(feature = "similarity")]
    pub fn check_pattern_anomaly(&self, pattern: &MarketPattern) -> Option<Anomaly> {
        if !self.config.detect_pattern_anomalies {
            return None;
        }

        let index_guard = self.pattern_index.read();
        let index = index_guard.as_ref()?;

        // Search for similar patterns using hot path (sub-μs)
        let results = match index.search_hot(&pattern.features, self.config.pattern_k_neighbors) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Pattern search failed: {}", e);
                return None;
            }
        };

        // Check if this is an anomalous pattern (no close matches)
        let min_distance = results.iter()
            .map(|r| r.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f32::MAX);

        let similar_count = results.iter()
            .filter(|r| r.score < self.config.pattern_distance_threshold)
            .count();

        // Pattern anomaly: no similar historical patterns found
        if min_distance > self.config.pattern_distance_threshold || similar_count == 0 {
            let anomaly = Anomaly {
                symbol: pattern.symbol.clone(),
                anomaly_type: AnomalyType::PatternAnomaly,
                zscore: 0.0, // Not applicable for pattern detection
                current_value: min_distance as f64,
                expected_value: self.config.pattern_distance_threshold as f64,
                std_dev: 0.0,
                detected_at: Timestamp::now(),
                severity: ((min_distance - self.config.pattern_distance_threshold)
                    / self.config.pattern_distance_threshold)
                    .clamp(0.0, 1.0) as f64,
                pattern_similarity: Some(1.0 - min_distance as f64),
                similar_pattern_count: Some(similar_count),
            };

            self.anomalies.write().push(anomaly.clone());
            return Some(anomaly);
        }

        None
    }

    /// Ingest a market pattern into the HNSW index for future comparison.
    ///
    /// Call this after processing market data to build the pattern database.
    #[cfg(feature = "similarity")]
    pub fn ingest_pattern(&self, pattern: &MarketPattern) -> Result<()> {
        let index_guard = self.pattern_index.read();
        let index = index_guard.as_ref()
            .ok_or_else(|| crate::core::error::RiskError::ConfigurationError(
                "Pattern index not initialized".to_string()
            ))?;

        // Use streaming ingestion for O(1) insertion
        index.stream_ingest(pattern.features.clone())
            .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                format!("Pattern ingestion failed: {}", e)
            ))?;

        self.pattern_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get number of patterns in the index.
    #[cfg(feature = "similarity")]
    pub fn pattern_count(&self) -> u64 {
        self.pattern_count.load(Ordering::Relaxed)
    }

    // ============================================================================
    // MinHash Whale Flow Detection (requires similarity feature)
    // ============================================================================

    /// Check if a transaction matches known whale patterns using MinHash.
    ///
    /// Returns an anomaly if the transaction's Jaccard similarity to known
    /// whale patterns exceeds the threshold.
    #[cfg(feature = "similarity")]
    pub fn check_whale_flow(&self, symbol: &Symbol, whale_pattern: &WhalePattern) -> Option<Anomaly> {
        if !self.config.detect_whale_flows {
            return None;
        }

        let minhash_guard = self.whale_minhash.read();
        let minhash = minhash_guard.as_ref()?;

        // Compute signature for current transaction
        let current_sig = minhash.hash(&whale_pattern.feature_hashes);

        // Compare against known whale signatures
        let signatures = self.whale_signatures.read();
        let mut max_jaccard: f32 = 0.0;
        let mut matching_count = 0;

        for known_sig in signatures.iter() {
            let jaccard = current_sig.jaccard_estimate(known_sig);
            if jaccard > max_jaccard {
                max_jaccard = jaccard;
            }
            if jaccard >= self.config.whale_jaccard_threshold {
                matching_count += 1;
            }
        }

        // Whale flow detected if high similarity to known patterns
        if max_jaccard >= self.config.whale_jaccard_threshold {
            let anomaly = Anomaly {
                symbol: symbol.clone(),
                anomaly_type: AnomalyType::WhaleFlow,
                zscore: 0.0, // Not applicable for Jaccard similarity
                current_value: whale_pattern.volume,
                expected_value: 0.0, // Could track average volume
                std_dev: 0.0,
                detected_at: Timestamp::now(),
                severity: (max_jaccard as f64).clamp(0.0, 1.0),
                pattern_similarity: Some(max_jaccard as f64),
                similar_pattern_count: Some(matching_count),
            };

            self.anomalies.write().push(anomaly.clone());
            return Some(anomaly);
        }

        None
    }

    /// Register a known whale transaction pattern for future detection.
    #[cfg(feature = "similarity")]
    pub fn register_whale_pattern(&self, whale_pattern: &WhalePattern) {
        let minhash_guard = self.whale_minhash.read();
        if let Some(minhash) = minhash_guard.as_ref() {
            let sig = minhash.hash(&whale_pattern.feature_hashes);
            self.whale_signatures.write().push(sig);
        }
    }

    /// Get number of registered whale patterns.
    #[cfg(feature = "similarity")]
    pub fn whale_pattern_count(&self) -> usize {
        self.whale_signatures.read().len()
    }

    /// Get recent anomalies.
    pub fn get_anomalies(&self) -> Vec<Anomaly> {
        self.anomalies.read().clone()
    }

    /// Clear anomaly history.
    pub fn clear_anomalies(&self) {
        self.anomalies.write().clear();
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for AnomalyDetectionAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Process portfolio positions for anomaly detection
        for position in &portfolio.positions {
            self.record_price(&position.symbol, position.current_price);
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detection_agent_creation() {
        let agent = AnomalyDetectionAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_price_anomaly_detection() {
        let mut config = AnomalyDetectionConfig::default();
        config.min_observations = 5;
        config.zscore_threshold = 2.0;

        let agent = AnomalyDetectionAgent::new(config);
        let symbol = Symbol::new("TEST");

        // Build up history with varied prices (creates std_dev > 0)
        let base_prices = [99.0, 101.0, 100.5, 99.5, 100.0, 100.2, 99.8, 100.1, 99.9, 100.3];
        for price in base_prices {
            agent.record_price(&symbol, Price::from_f64(price));
        }

        // Introduce a spike (significantly outside normal range)
        let anomaly = agent.record_price(&symbol, Price::from_f64(150.0));
        assert!(anomaly.is_some());

        let a = anomaly.unwrap();
        assert_eq!(a.anomaly_type, AnomalyType::PriceSpike);
        assert!(a.zscore > 2.0);
    }

    #[test]
    fn test_volume_anomaly_detection() {
        let mut config = AnomalyDetectionConfig::default();
        config.min_observations = 5;
        config.zscore_threshold = 2.0;

        let agent = AnomalyDetectionAgent::new(config);
        let symbol = Symbol::new("TEST");

        // Build up history with varied volume (creates std_dev > 0)
        let base_volumes = [990.0, 1010.0, 1005.0, 995.0, 1000.0, 1002.0, 998.0, 1001.0, 999.0, 1003.0];
        for volume in base_volumes {
            agent.record_volume(&symbol, volume);
        }

        // Introduce a volume surge (significantly outside normal range)
        let anomaly = agent.record_volume(&symbol, 5000.0);
        assert!(anomaly.is_some());

        let a = anomaly.unwrap();
        assert_eq!(a.anomaly_type, AnomalyType::VolumeSurge);
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = AnomalyDetectionAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }

    #[test]
    fn test_anomaly_has_pattern_fields() {
        // Verify new fields are present
        let anomaly = Anomaly {
            symbol: Symbol::new("TEST"),
            anomaly_type: AnomalyType::PatternAnomaly,
            zscore: 0.0,
            current_value: 0.5,
            expected_value: 0.8,
            std_dev: 0.0,
            detected_at: Timestamp::now(),
            severity: 0.75,
            pattern_similarity: Some(0.3),
            similar_pattern_count: Some(2),
        };

        assert_eq!(anomaly.pattern_similarity, Some(0.3));
        assert_eq!(anomaly.similar_pattern_count, Some(2));
    }

    #[test]
    fn test_config_has_similarity_fields() {
        let config = AnomalyDetectionConfig::default();

        // Verify similarity config defaults
        assert!(config.detect_pattern_anomalies);
        assert!(config.detect_whale_flows);
        assert_eq!(config.pattern_dimensions, 8);
        assert_eq!(config.pattern_k_neighbors, 5);
        assert_eq!(config.pattern_distance_threshold, 0.8);
        assert_eq!(config.minhash_signature_size, 128);
        assert_eq!(config.whale_jaccard_threshold, 0.7);
    }
}

/// Tests for similarity-based detection (requires feature flag).
#[cfg(all(test, feature = "similarity"))]
mod similarity_tests {
    use super::*;

    #[test]
    fn test_market_pattern_creation() {
        let symbol = Symbol::new("AAPL");
        let pattern = MarketPattern::from_market_data(
            symbol.clone(),
            150.0, // current price
            148.0, // prev price
            1000.0, // current volume
            800.0,  // avg volume
            0.02,   // volatility
            Timestamp::now(),
        );

        assert_eq!(pattern.symbol, symbol);
        assert_eq!(pattern.features.len(), 8); // 8 dimensions
    }

    #[test]
    fn test_whale_pattern_creation() {
        let pattern = WhalePattern::from_transaction(
            1_000_000.0, // volume
            50.0,        // price impact bps
            0.95,        // 95th percentile
            14,          // 2pm bucket
            true,        // aggressive
        );

        assert!(pattern.feature_hashes.len() >= 4); // At least 4 features
        assert_eq!(pattern.volume, 1_000_000.0);
    }

    #[test]
    fn test_similarity_initialization() {
        let agent = AnomalyDetectionAgent::with_defaults();

        // Before init, should not be initialized
        assert!(!agent.is_similarity_initialized());

        // Initialize
        agent.init_similarity().expect("Init should succeed");

        // After init, should be initialized
        assert!(agent.is_similarity_initialized());
        assert_eq!(agent.pattern_count(), 0);
        assert_eq!(agent.whale_pattern_count(), 0);
    }

    #[test]
    fn test_whale_pattern_registration() {
        let agent = AnomalyDetectionAgent::with_defaults();
        agent.init_similarity().expect("Init should succeed");

        let whale = WhalePattern::from_transaction(
            5_000_000.0, 100.0, 0.99, 10, true,
        );

        agent.register_whale_pattern(&whale);
        assert_eq!(agent.whale_pattern_count(), 1);
    }
}
