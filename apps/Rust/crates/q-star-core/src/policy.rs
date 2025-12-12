//! Policy interface for Q* algorithm

use async_trait::async_trait;
use crate::{MarketState, Experience, QStarError};

/// Policy trait for action selection
#[async_trait]
pub trait Policy: Send + Sync {
    /// Get action probabilities for given state
    async fn get_action_probabilities(&self, state: &MarketState) -> Result<Vec<f64>, QStarError>;
    
    /// Update policy with experiences
    async fn update(&self, experiences: &[Experience]) -> Result<(), QStarError>;
}