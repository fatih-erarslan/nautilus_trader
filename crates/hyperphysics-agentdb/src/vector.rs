//! Vector index wrapper for trading-specific semantic search
//!
//! Wraps ruvector-core's vector database with trading-domain optimizations.

use ruvector_core::types::{SearchQuery, SearchResult, VectorEntry};
use ruvector_core::VectorDB;
use std::collections::HashMap;

/// Trading-optimized vector index
pub struct TradingVectorIndex {
    db: VectorDB,
    dimensions: usize,
}

impl TradingVectorIndex {
    /// Create new index with given dimensions
    pub fn new(dimensions: usize, storage_path: &str) -> Result<Self, ruvector_core::error::RuvectorError> {
        let mut options = ruvector_core::types::DbOptions::default();
        options.dimensions = dimensions;
        options.storage_path = storage_path.to_string();
        
        let db = VectorDB::new(options)?;
        Ok(Self { db, dimensions })
    }

    /// Insert vector with metadata
    pub fn insert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<String, ruvector_core::error::RuvectorError> {
        self.db.insert(VectorEntry {
            id: Some(id.to_string()),
            vector,
            metadata,
        })
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>, ruvector_core::error::RuvectorError> {
        self.db.search(SearchQuery {
            vector: query_vector,
            k,
            filter,
            ef_search: None,
        })
    }

    /// Delete vector by ID
    pub fn delete(&self, id: &str) -> Result<bool, ruvector_core::error::RuvectorError> {
        self.db.delete(id)
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>, ruvector_core::error::RuvectorError> {
        self.db.get(id)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}
