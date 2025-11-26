//! Data adapter for converting HyperPhysics market data to neural input format.

use crate::config::NeuralBridgeConfig;
use crate::error::{NeuralBridgeError, Result};
use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::trace;

/// Market data feed abstraction
///
/// Defined locally to avoid cyclic dependency with hyperphysics-hft-ecosystem.
/// The hft-ecosystem crate has its own MarketFeed, and conversion happens at the
/// integration boundary.
#[derive(Debug, Clone)]
pub struct MarketFeed {
    /// Latest price
    pub price: f64,
    /// Recent returns (last N periods)
    pub returns: Vec<f64>,
    /// Current volatility estimate
    pub volatility: f64,
    /// Volume-weighted average price
    pub vwap: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Current timestamp (epoch seconds)
    pub timestamp: f64,
}

impl Default for MarketFeed {
    fn default() -> Self {
        Self {
            price: 100.0,
            returns: vec![0.0; 10],
            volatility: 0.02,
            vwap: 100.0,
            spread: 0.01,
            timestamp: 0.0,
        }
    }
}

/// Feature set extracted from market data for neural model input
#[derive(Debug, Clone)]
pub struct NeuralFeatures {
    /// Price sequence (normalized)
    pub prices: Vec<f64>,
    /// Return sequence
    pub returns: Vec<f64>,
    /// Volatility estimates
    pub volatility: Vec<f64>,
    /// Spread sequence
    pub spreads: Vec<f64>,
    /// VWAP sequence
    pub vwaps: Vec<f64>,
    /// Combined feature matrix (rows = timesteps, cols = features)
    pub feature_matrix: Array2<f64>,
    /// Target values for training (if available)
    pub targets: Option<Vec<f64>>,
    /// Timestamps for each observation
    pub timestamps: Vec<f64>,
}

/// Adapter for converting HyperPhysics data to neural input format
pub struct NeuralDataAdapter {
    /// Configuration
    config: NeuralBridgeConfig,
    /// Historical price buffer
    price_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Historical return buffer
    return_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Historical volatility buffer
    volatility_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Historical spread buffer
    spread_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Historical VWAP buffer
    vwap_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Timestamp buffer
    timestamp_buffer: Arc<RwLock<VecDeque<f64>>>,
    /// Normalization statistics
    norm_stats: Arc<RwLock<NormalizationStats>>,
}

/// Statistics for data normalization
#[derive(Debug, Clone, Default)]
pub struct NormalizationStats {
    /// Price mean
    pub price_mean: f64,
    /// Price standard deviation
    pub price_std: f64,
    /// Volatility mean
    pub vol_mean: f64,
    /// Volatility std
    pub vol_std: f64,
    /// Spread mean
    pub spread_mean: f64,
    /// Spread std
    pub spread_std: f64,
    /// Number of samples used for stats
    pub sample_count: usize,
}

impl NeuralDataAdapter {
    /// Create a new neural data adapter
    pub fn new(config: NeuralBridgeConfig) -> Self {
        let buffer_size = config.feature_window;
        Self {
            config,
            price_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            return_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            volatility_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            spread_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            vwap_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            timestamp_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            norm_stats: Arc::new(RwLock::new(NormalizationStats::default())),
        }
    }

    /// Process a market feed update and extract features
    pub async fn process_feed(&self, feed: &MarketFeed) -> Result<NeuralFeatures> {
        // Update buffers
        self.update_buffers(feed).await;

        // Extract features
        self.extract_features().await
    }

    /// Update internal buffers with new market data
    async fn update_buffers(&self, feed: &MarketFeed) {
        let max_size = self.config.feature_window;

        // Update price buffer
        {
            let mut prices = self.price_buffer.write().await;
            prices.push_back(feed.price);
            while prices.len() > max_size {
                prices.pop_front();
            }
        }

        // Update return buffer (use provided returns)
        {
            let mut returns = self.return_buffer.write().await;
            for r in &feed.returns {
                returns.push_back(*r);
            }
            while returns.len() > max_size {
                returns.pop_front();
            }
        }

        // Update volatility buffer
        {
            let mut vols = self.volatility_buffer.write().await;
            vols.push_back(feed.volatility);
            while vols.len() > max_size {
                vols.pop_front();
            }
        }

        // Update spread buffer
        {
            let mut spreads = self.spread_buffer.write().await;
            spreads.push_back(feed.spread);
            while spreads.len() > max_size {
                spreads.pop_front();
            }
        }

        // Update VWAP buffer
        {
            let mut vwaps = self.vwap_buffer.write().await;
            vwaps.push_back(feed.vwap);
            while vwaps.len() > max_size {
                vwaps.pop_front();
            }
        }

        // Update timestamp buffer
        {
            let mut timestamps = self.timestamp_buffer.write().await;
            timestamps.push_back(feed.timestamp);
            while timestamps.len() > max_size {
                timestamps.pop_front();
            }
        }

        // Update normalization statistics
        self.update_normalization_stats().await;

        trace!(
            price = feed.price,
            volatility = feed.volatility,
            "Updated neural buffers"
        );
    }

    /// Update running normalization statistics
    async fn update_normalization_stats(&self) {
        let prices = self.price_buffer.read().await;
        let vols = self.volatility_buffer.read().await;
        let spreads = self.spread_buffer.read().await;

        if prices.len() < 2 {
            return;
        }

        let mut stats = self.norm_stats.write().await;

        // Calculate price statistics
        let price_vec: Vec<f64> = prices.iter().copied().collect();
        stats.price_mean = price_vec.iter().sum::<f64>() / price_vec.len() as f64;
        stats.price_std = (price_vec
            .iter()
            .map(|p| (p - stats.price_mean).powi(2))
            .sum::<f64>()
            / (price_vec.len() - 1) as f64)
            .sqrt()
            .max(1e-8);

        // Calculate volatility statistics
        let vol_vec: Vec<f64> = vols.iter().copied().collect();
        if !vol_vec.is_empty() {
            stats.vol_mean = vol_vec.iter().sum::<f64>() / vol_vec.len() as f64;
            stats.vol_std = if vol_vec.len() > 1 {
                (vol_vec
                    .iter()
                    .map(|v| (v - stats.vol_mean).powi(2))
                    .sum::<f64>()
                    / (vol_vec.len() - 1) as f64)
                    .sqrt()
                    .max(1e-8)
            } else {
                1.0
            };
        }

        // Calculate spread statistics
        let spread_vec: Vec<f64> = spreads.iter().copied().collect();
        if !spread_vec.is_empty() {
            stats.spread_mean = spread_vec.iter().sum::<f64>() / spread_vec.len() as f64;
            stats.spread_std = if spread_vec.len() > 1 {
                (spread_vec
                    .iter()
                    .map(|s| (s - stats.spread_mean).powi(2))
                    .sum::<f64>()
                    / (spread_vec.len() - 1) as f64)
                    .sqrt()
                    .max(1e-8)
            } else {
                1.0
            };
        }

        stats.sample_count = prices.len();
    }

    /// Extract features from current buffers
    pub async fn extract_features(&self) -> Result<NeuralFeatures> {
        let prices = self.price_buffer.read().await;
        let returns = self.return_buffer.read().await;
        let vols = self.volatility_buffer.read().await;
        let spreads = self.spread_buffer.read().await;
        let vwaps = self.vwap_buffer.read().await;
        let timestamps = self.timestamp_buffer.read().await;
        let stats = self.norm_stats.read().await;

        let seq_len = prices.len();
        if seq_len < self.config.min_sequence_length {
            return Err(NeuralBridgeError::InsufficientData {
                required: self.config.min_sequence_length,
                actual: seq_len,
            });
        }

        // Normalize and collect features
        let normalized_prices: Vec<f64> = prices
            .iter()
            .map(|p| (p - stats.price_mean) / stats.price_std)
            .collect();

        let normalized_vols: Vec<f64> = vols
            .iter()
            .map(|v| (v - stats.vol_mean) / stats.vol_std)
            .collect();

        let normalized_spreads: Vec<f64> = spreads
            .iter()
            .map(|s| (s - stats.spread_mean) / stats.spread_std)
            .collect();

        let returns_vec: Vec<f64> = returns.iter().copied().collect();
        let vwaps_vec: Vec<f64> = vwaps.iter().copied().collect();
        let timestamps_vec: Vec<f64> = timestamps.iter().copied().collect();

        // Build feature matrix (timesteps x features)
        // Features: normalized_price, return, normalized_vol, normalized_spread, vwap_ratio
        let num_features = 5;
        let mut feature_matrix = Array2::zeros((seq_len, num_features));

        for i in 0..seq_len {
            feature_matrix[[i, 0]] = normalized_prices.get(i).copied().unwrap_or(0.0);
            feature_matrix[[i, 1]] = returns_vec.get(i).copied().unwrap_or(0.0);
            feature_matrix[[i, 2]] = normalized_vols.get(i).copied().unwrap_or(0.0);
            feature_matrix[[i, 3]] = normalized_spreads.get(i).copied().unwrap_or(0.0);
            // VWAP ratio to price
            let vwap_ratio = if let (Some(&p), Some(&v)) = (prices.get(i), vwaps_vec.get(i)) {
                if p > 0.0 {
                    (v - p) / p
                } else {
                    0.0
                }
            } else {
                0.0
            };
            feature_matrix[[i, 4]] = vwap_ratio;
        }

        Ok(NeuralFeatures {
            prices: prices.iter().copied().collect(),
            returns: returns_vec,
            volatility: vols.iter().copied().collect(),
            spreads: spreads.iter().copied().collect(),
            vwaps: vwaps_vec,
            feature_matrix,
            targets: None,
            timestamps: timestamps_vec,
        })
    }

    /// Get current buffer length
    pub async fn buffer_length(&self) -> usize {
        self.price_buffer.read().await.len()
    }

    /// Check if enough data is available for forecasting
    pub async fn is_ready(&self) -> bool {
        self.buffer_length().await >= self.config.min_sequence_length
    }

    /// Clear all buffers
    pub async fn clear(&self) {
        self.price_buffer.write().await.clear();
        self.return_buffer.write().await.clear();
        self.volatility_buffer.write().await.clear();
        self.spread_buffer.write().await.clear();
        self.vwap_buffer.write().await.clear();
        self.timestamp_buffer.write().await.clear();
        *self.norm_stats.write().await = NormalizationStats::default();
    }

    /// Get current normalization statistics
    pub async fn get_norm_stats(&self) -> NormalizationStats {
        self.norm_stats.read().await.clone()
    }
}

impl Clone for NeuralDataAdapter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            price_buffer: Arc::clone(&self.price_buffer),
            return_buffer: Arc::clone(&self.return_buffer),
            volatility_buffer: Arc::clone(&self.volatility_buffer),
            spread_buffer: Arc::clone(&self.spread_buffer),
            vwap_buffer: Arc::clone(&self.vwap_buffer),
            timestamp_buffer: Arc::clone(&self.timestamp_buffer),
            norm_stats: Arc::clone(&self.norm_stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_feed(price: f64, returns: Vec<f64>) -> MarketFeed {
        MarketFeed {
            price,
            returns,
            volatility: 0.02,
            vwap: price * 0.999,
            spread: 0.01,
            timestamp: 1000.0,
        }
    }

    #[tokio::test]
    async fn test_adapter_creation() {
        let config = NeuralBridgeConfig::default();
        let adapter = NeuralDataAdapter::new(config);
        assert!(!adapter.is_ready().await);
    }

    #[tokio::test]
    async fn test_buffer_update() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 5,
            feature_window: 10,
            ..Default::default()
        };
        let adapter = NeuralDataAdapter::new(config);

        for i in 0..10 {
            let feed = make_test_feed(100.0 + i as f64, vec![0.01, 0.02]);
            adapter.update_buffers(&feed).await;
        }

        assert!(adapter.is_ready().await);
        assert_eq!(adapter.buffer_length().await, 10);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 5,
            feature_window: 20,
            ..Default::default()
        };
        let adapter = NeuralDataAdapter::new(config);

        for i in 0..10 {
            let feed = make_test_feed(100.0 + i as f64, vec![0.01]);
            adapter.update_buffers(&feed).await;
        }

        let features = adapter.extract_features().await.unwrap();
        assert_eq!(features.prices.len(), 10);
        assert_eq!(features.feature_matrix.nrows(), 10);
        assert_eq!(features.feature_matrix.ncols(), 5);
    }

    #[tokio::test]
    async fn test_insufficient_data() {
        let config = NeuralBridgeConfig {
            min_sequence_length: 24,
            ..Default::default()
        };
        let adapter = NeuralDataAdapter::new(config);

        for i in 0..5 {
            let feed = make_test_feed(100.0 + i as f64, vec![0.01]);
            adapter.update_buffers(&feed).await;
        }

        let result = adapter.extract_features().await;
        assert!(matches!(
            result,
            Err(NeuralBridgeError::InsufficientData { .. })
        ));
    }
}
