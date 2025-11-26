use beclever_common::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchRequest {
    pub query: Vec<f32>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

#[cfg_attr(test, mockall::automock)]
pub trait VectorSearchService: Send + Sync {
    fn search(&self, request: VectorSearchRequest) -> Result<Vec<VectorSearchResult>>;
}
