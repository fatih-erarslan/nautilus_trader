//! Experience memory for Q* algorithm

use async_trait::async_trait;
use crate::{Experience, QStarError};

/// Experience memory trait for replay buffer
#[async_trait]
pub trait ExperienceMemory: Send + Sync {
    /// Store experience in memory
    async fn store(&self, experience: Experience) -> Result<(), QStarError>;
    
    /// Sample batch of experiences
    async fn sample(&self, batch_size: usize) -> Result<Vec<Experience>, QStarError>;
    
    /// Get current memory size
    async fn size(&self) -> usize;
}