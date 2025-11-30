//! HyperPhysics Vector Database Integration
//!
//! This crate provides seamless integration between HyperPhysics and ruvector-core's AgenticDB
//! for high-performance vector similarity search with HNSW indexing.
//!
//! # Features
//!
//! - **HNSW Indexing**: O(log n) search with 95%+ recall
//! - **AgenticDB Integration**: Full 5-table agentic memory schema
//! - **Reflexion Memory**: Store and retrieve self-critique episodes
//! - **Skills Library**: Consolidated action patterns
//! - **Causal Memory**: Hypergraph-based cause-effect relationships
//! - **Learning Sessions**: RL experience replay storage
//!
//! # Example
//!
//! ```rust,ignore
//! use hyperphysics_vector::prelude::*;
//!
//! // Create vector store for market state embeddings
//! let store = HyperVectorStore::new(HyperVectorConfig {
//!     dimensions: 128,
//!     storage_path: "market_vectors.db".into(),
//!     ..Default::default()
//! })?;
//!
//! // Store market state embedding
//! let id = store.insert_market_state(&embedding, &metadata)?;
//!
//! // Search similar states
//! let results = store.search_similar(&query_embedding, 10)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod store;
pub mod market;
pub mod agent_memory;
pub mod embeddings;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::error::{VectorError, Result};
    pub use crate::store::{HyperVectorStore, HyperVectorConfig};
    pub use crate::market::{MarketStateEmbedding, OrderBookSnapshot};
    pub use crate::agent_memory::{AgentMemory, StrategyEpisode, TradingSkill};
    pub use crate::embeddings::EmbeddingGenerator;
}

// Re-exports from ruvector-core
pub use ruvector_core::{
    VectorDB, VectorEntry, VectorId, SearchQuery, SearchResult,
    DistanceMetric, ConformalPredictor, ConformalConfig, PredictionSet,
    HybridSearch, HybridConfig, FilteredSearch, FilterExpression,
    MMRSearch, MMRConfig, EnhancedPQ, PQConfig, BM25,
};

#[cfg(feature = "storage")]
pub use ruvector_core::AgenticDB;

pub use error::{VectorError, Result};
pub use store::{HyperVectorStore, HyperVectorConfig};
