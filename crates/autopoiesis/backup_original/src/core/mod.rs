//! Core traits and types for the autopoiesis system

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::Result;

// Core system submodules
pub mod state;
pub mod boundary;
pub mod process;
pub mod environment;

// Core autopoiesis modules
pub mod autopoiesis;
pub mod dissipative;
pub mod mind;
pub mod sync;
pub mod web;
pub mod syntergy;

// Re-export core types
pub use state::SystemState;
pub use boundary::SystemBoundary;
pub use process::SystemProcess;
pub use environment::SystemEnvironment;

// Re-export core traits
pub use autopoiesis::{AutopoieticSystem, Boundary, State, ProcessTrait, Environment};
pub use dissipative::DissipativeStructure;
pub use mind::EcologyOfMind;
pub use sync::SynchronizationDynamics;
pub use web::WebOfLife;
pub use syntergy::Syntergic;

/// Observer trait - the fundamental building block of the autopoietic system
#[async_trait]
pub trait Observer: Send + Sync {
    /// The type of signal this observer produces
    type Signal: Send + Sync;
    
    /// The type of configuration this observer requires
    type Config: Send + Sync;
    
    /// Initialize the observer with the given configuration
    async fn initialize(&mut self, config: Self::Config) -> Result<()>;
    
    /// Observe the current state and produce a signal
    async fn observe(&self, context: &ObserverContext) -> Result<Self::Signal>;
    
    /// React to signals from other observers
    async fn react(&mut self, signal: &dyn std::any::Any) -> Result<()>;
    
    /// Get the observer's current state
    fn state(&self) -> ObserverState;
    
    /// Get the observer's unique identifier
    fn id(&self) -> &str;
}

/// Context provided to observers during observation
#[derive(Clone, Debug)]
pub struct ObserverContext {
    /// Current system state
    pub system_state: Arc<state::SystemState>,
    
    /// Timestamp of the observation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Additional context data
    pub data: std::collections::HashMap<String, serde_json::Value>,
}

/// State of an individual observer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObserverState {
    /// Observer is initializing
    Initializing,
    
    /// Observer is active and functioning normally
    Active,
    
    /// Observer is temporarily paused
    Paused,
    
    /// Observer has encountered an error
    Error(String),
    
    /// Observer is shutting down
    ShuttingDown,
}


/// Market conditions
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MarketConditions {
    /// Market volatility (0.0 - 1.0)
    pub volatility: f64,
    
    /// Market trend (-1.0 = strong down, 0.0 = neutral, 1.0 = strong up)
    pub trend: f64,
    
    /// Trading volume relative to average
    pub volume_ratio: f64,
    
    /// Market sentiment score
    pub sentiment: f64,
}

/// System performance metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Total profit/loss
    pub total_pnl: f64,
    
    /// Win rate
    pub win_rate: f64,
    
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Number of trades executed
    pub trade_count: u64,
}