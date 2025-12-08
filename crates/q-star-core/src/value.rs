//! Value function interface for Q* algorithm

use async_trait::async_trait;
use crate::{MarketState, Experience, QStarError};

/// Value function trait for state evaluation
#[async_trait]
pub trait ValueFunction: Send + Sync {
    /// Evaluate state value
    async fn evaluate(&self, state: &MarketState) -> Result<f64, QStarError>;
    
    /// Update value function with experiences
    async fn update(&self, experiences: &[Experience]) -> Result<(), QStarError>;
}