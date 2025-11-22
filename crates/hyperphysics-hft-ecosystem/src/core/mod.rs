//! Core integration module
//!
//! Provides the main integration point for all ecosystem components.

pub mod biomimetic_coordinator;
pub mod ecosystem_builder;
pub mod physics_engine_router;

pub use biomimetic_coordinator::*;
pub use ecosystem_builder::*;
pub use physics_engine_router::*;

use crate::Result;
use std::sync::Arc;

/// Main HFT ecosystem coordinator
pub struct HFTEcosystem {
    /// Physics engine router
    pub physics_router: Arc<PhysicsEngineRouter>,

    /// Biomimetic algorithm coordinator
    pub biomimetic_coord: Arc<tokio::sync::RwLock<BiomimeticCoordinator>>,

    /// Configuration
    config: EcosystemConfig,
}

/// Ecosystem configuration
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Enable formal verification
    pub formal_verification: bool,

    /// Target latency in microseconds
    pub target_latency_us: u64,

    /// Enable SIMD optimizations
    pub simd_enabled: bool,

    /// Number of worker threads
    pub worker_threads: usize,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            formal_verification: true,
            target_latency_us: 1000, // 1ms
            simd_enabled: true,
            worker_threads: num_cpus::get(),
        }
    }
}

impl HFTEcosystem {
    /// Create a new ecosystem builder
    pub fn builder() -> EcosystemBuilder {
        EcosystemBuilder::new()
    }

    /// Execute a single HFT cycle
    pub async fn execute_cycle(&self, market_tick: &MarketTick) -> Result<TradingDecision> {
        // Step 1: Route to physics engine
        let physics_result = self.physics_router.route(market_tick).await?;

        // Step 2: Coordinate biomimetic algorithms
        let mut biomimetic_guard = self.biomimetic_coord.write().await;
        let biomimetic_decision = biomimetic_guard.coordinate_swarms(&physics_result).await?;
        drop(biomimetic_guard);

        // Step 3: Convert to trading decision
        Ok(biomimetic_decision.into())
    }
}

/// Market tick data
#[derive(Debug, Clone)]
pub struct MarketTick {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Order book snapshot
    pub orderbook: Vec<u8>, // Placeholder

    /// Recent trades
    pub trades: Vec<u8>, // Placeholder
}

impl Default for MarketTick {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            orderbook: Vec::new(),
            trades: Vec::new(),
        }
    }
}

/// Trading decision output
#[derive(Debug, Clone)]
pub struct TradingDecision {
    /// Action to take
    pub action: Action,

    /// Confidence score
    pub confidence: f64,

    /// Recommended size
    pub size: f64,
}

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold position
    Hold,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EcosystemConfig::default();
        assert!(config.formal_verification);
        assert_eq!(config.target_latency_us, 1000);
    }
}
