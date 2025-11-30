//! Market state embeddings for vector similarity search

use crate::error::{Result, VectorError};
use crate::store::HyperVectorStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Market state embedding for similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStateEmbedding {
    /// Unique identifier
    pub id: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Asset symbol
    pub symbol: String,
    /// Timestamp (Unix epoch millis)
    pub timestamp: i64,
    /// Mid price at time of embedding
    pub mid_price: f64,
    /// Spread at time of embedding
    pub spread: f64,
    /// Volume at time of embedding
    pub volume: f64,
    /// Volatility estimate
    pub volatility: f64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Order book snapshot for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Asset symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: i64,
    /// Bid prices (best to worst)
    pub bid_prices: Vec<f64>,
    /// Bid quantities
    pub bid_quantities: Vec<f64>,
    /// Ask prices (best to worst)
    pub ask_prices: Vec<f64>,
    /// Ask quantities
    pub ask_quantities: Vec<f64>,
}

impl OrderBookSnapshot {
    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        let best_bid = self.bid_prices.first().copied().unwrap_or(0.0);
        let best_ask = self.ask_prices.first().copied().unwrap_or(0.0);
        (best_bid + best_ask) / 2.0
    }

    /// Calculate spread
    pub fn spread(&self) -> f64 {
        let best_bid = self.bid_prices.first().copied().unwrap_or(0.0);
        let best_ask = self.ask_prices.first().copied().unwrap_or(0.0);
        best_ask - best_bid
    }

    /// Calculate total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bid_quantities.iter().sum()
    }

    /// Calculate total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.ask_quantities.iter().sum()
    }

    /// Calculate order book imbalance [-1, 1]
    pub fn imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }

    /// Generate a basic embedding from order book features
    /// Returns a vector of normalized features
    pub fn to_embedding(&self, dimensions: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dimensions];

        // Basic features
        let mid = self.mid_price();
        let spread = self.spread();
        let imbalance = self.imbalance();
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();

        // Price-level features (normalized spreads from mid)
        let spread_pct = if mid > 0.0 { spread / mid } else { 0.0 };

        // Fill embedding with features
        if dimensions >= 8 {
            embedding[0] = (spread_pct * 1000.0) as f32; // Spread in basis points
            embedding[1] = imbalance as f32;
            embedding[2] = (bid_vol.ln().max(0.0) / 10.0) as f32; // Log bid volume
            embedding[3] = (ask_vol.ln().max(0.0) / 10.0) as f32; // Log ask volume

            // Price level distribution features
            for (i, (bid_p, bid_q)) in self.bid_prices.iter()
                .zip(self.bid_quantities.iter())
                .take((dimensions - 4) / 4)
                .enumerate()
            {
                let idx = 4 + i * 2;
                if idx + 1 < dimensions {
                    let dist = if mid > 0.0 { (mid - bid_p) / mid } else { 0.0 };
                    embedding[idx] = (dist * 100.0) as f32;
                    embedding[idx + 1] = (bid_q.ln().max(0.0) / 10.0) as f32;
                }
            }

            for (i, (ask_p, ask_q)) in self.ask_prices.iter()
                .zip(self.ask_quantities.iter())
                .take((dimensions - 4) / 4)
                .enumerate()
            {
                let base_idx = 4 + (dimensions - 4) / 2;
                let idx = base_idx + i * 2;
                if idx + 1 < dimensions {
                    let dist = if mid > 0.0 { (ask_p - mid) / mid } else { 0.0 };
                    embedding[idx] = (dist * 100.0) as f32;
                    embedding[idx + 1] = (ask_q.ln().max(0.0) / 10.0) as f32;
                }
            }
        }

        // Normalize the embedding
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

/// Market state vector operations
pub struct MarketVectorOps<'a> {
    store: &'a HyperVectorStore,
}

impl<'a> MarketVectorOps<'a> {
    /// Create new market vector operations
    pub fn new(store: &'a HyperVectorStore) -> Self {
        Self { store }
    }

    /// Store a market state embedding
    pub fn store_market_state(&self, state: &MarketStateEmbedding) -> Result<String> {
        let mut metadata = state.metadata.clone();
        metadata.insert("type".to_string(), serde_json::json!("market_state"));
        metadata.insert("symbol".to_string(), serde_json::json!(state.symbol));
        metadata.insert("timestamp".to_string(), serde_json::json!(state.timestamp));
        metadata.insert("mid_price".to_string(), serde_json::json!(state.mid_price));
        metadata.insert("spread".to_string(), serde_json::json!(state.spread));
        metadata.insert("volume".to_string(), serde_json::json!(state.volume));
        metadata.insert("volatility".to_string(), serde_json::json!(state.volatility));

        self.store.insert_with_id(&state.id, state.embedding.clone(), Some(metadata))
    }

    /// Store order book snapshot as embedding
    pub fn store_order_book(&self, snapshot: &OrderBookSnapshot) -> Result<String> {
        let embedding = snapshot.to_embedding(self.store.config().dimensions);

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), serde_json::json!("order_book"));
        metadata.insert("symbol".to_string(), serde_json::json!(snapshot.symbol));
        metadata.insert("timestamp".to_string(), serde_json::json!(snapshot.timestamp));
        metadata.insert("mid_price".to_string(), serde_json::json!(snapshot.mid_price()));
        metadata.insert("spread".to_string(), serde_json::json!(snapshot.spread()));
        metadata.insert("imbalance".to_string(), serde_json::json!(snapshot.imbalance()));

        self.store.insert(embedding, Some(metadata))
    }

    /// Find similar market states
    pub fn find_similar_states(
        &self,
        query_embedding: Vec<f32>,
        k: usize,
        symbol_filter: Option<&str>,
    ) -> Result<Vec<MarketStateEmbedding>> {
        let filter = symbol_filter.map(|symbol| {
            let mut f = HashMap::new();
            f.insert("symbol".to_string(), serde_json::json!(symbol));
            f.insert("type".to_string(), serde_json::json!("market_state"));
            f
        });

        let results = self.store.search(query_embedding, k, filter)?;

        let states = results
            .into_iter()
            .filter_map(|r| {
                let metadata = r.metadata.as_ref()?;
                Some(MarketStateEmbedding {
                    id: r.id.clone(),
                    embedding: r.vector,
                    symbol: metadata.get("symbol")?.as_str()?.to_string(),
                    timestamp: metadata.get("timestamp")?.as_i64()?,
                    mid_price: metadata.get("mid_price")?.as_f64()?,
                    spread: metadata.get("spread")?.as_f64()?,
                    volume: metadata.get("volume")?.as_f64().unwrap_or(0.0),
                    volatility: metadata.get("volatility")?.as_f64().unwrap_or(0.0),
                    metadata: metadata.clone(),
                })
            })
            .collect();

        Ok(states)
    }

    /// Find similar order book patterns
    pub fn find_similar_order_books(
        &self,
        snapshot: &OrderBookSnapshot,
        k: usize,
    ) -> Result<Vec<(String, f32, HashMap<String, serde_json::Value>)>> {
        let query_embedding = snapshot.to_embedding(self.store.config().dimensions);

        let mut filter = HashMap::new();
        filter.insert("type".to_string(), serde_json::json!("order_book"));

        let results = self.store.search(query_embedding, k, Some(filter))?;

        Ok(results
            .into_iter()
            .map(|r| (r.id, r.score, r.metadata.unwrap_or_default()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_embedding() {
        let snapshot = OrderBookSnapshot {
            symbol: "BTC-USD".to_string(),
            timestamp: 1000000,
            bid_prices: vec![100.0, 99.5, 99.0],
            bid_quantities: vec![10.0, 20.0, 30.0],
            ask_prices: vec![100.5, 101.0, 101.5],
            ask_quantities: vec![15.0, 25.0, 35.0],
        };

        let embedding = snapshot.to_embedding(32);
        assert_eq!(embedding.len(), 32);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001 || norm == 0.0);
    }

    #[test]
    fn test_order_book_metrics() {
        let snapshot = OrderBookSnapshot {
            symbol: "ETH-USD".to_string(),
            timestamp: 1000000,
            bid_prices: vec![3000.0, 2999.0],
            bid_quantities: vec![100.0, 200.0],
            ask_prices: vec![3001.0, 3002.0],
            ask_quantities: vec![50.0, 150.0],
        };

        assert!((snapshot.mid_price() - 3000.5).abs() < 0.01);
        assert!((snapshot.spread() - 1.0).abs() < 0.01);
        assert_eq!(snapshot.total_bid_volume(), 300.0);
        assert_eq!(snapshot.total_ask_volume(), 200.0);
        // Imbalance should be positive (more bids)
        assert!(snapshot.imbalance() > 0.0);
    }
}
