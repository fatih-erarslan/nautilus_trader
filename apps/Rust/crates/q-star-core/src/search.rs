//! Search tree for Q* algorithm

use async_trait::async_trait;
use crate::{MarketState, QStarAction, QStarError};

/// Search tree trait for Q* tree search
#[async_trait]
pub trait SearchTree: Send + Sync {
    /// Initialize search tree with root state
    async fn initialize(&self, state: &MarketState) -> Result<(), QStarError>;
    
    /// Expand tree with new state-action pair
    async fn expand(&self, state: &MarketState, action: &QStarAction) -> Result<MarketState, QStarError>;
    
    /// Get best path from tree search
    async fn get_best_path(&self) -> Result<Vec<QStarAction>, QStarError>;
}