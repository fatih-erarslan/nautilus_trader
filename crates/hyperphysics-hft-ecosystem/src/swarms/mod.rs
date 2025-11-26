//! Swarms module - Biomimetic algorithm implementations
//!
//! Organized by latency tiers:
//! - Tier 1: <1ms (Whale, Cuckoo, PSO, Bat, Firefly)
//! - Tier 2: 1-10ms (GA, DE, GWO, ABC, ACO)
//! - Tier 3: 10ms+ (BFO, SSO, MFO, Salp)
//!
//! # Real vs Stub Implementations
//!
//! - `real_optimizer`: Uses hyperphysics-optimization (RECOMMENDED)
//! - `tier1_execution`: Legacy wrapper for bio-inspired-workspace stubs

pub mod tier1_execution;

/// Real optimizer using hyperphysics-optimization algorithms
#[cfg(feature = "optimization-real")]
pub mod real_optimizer;

pub use tier1_execution::*;

#[cfg(feature = "optimization-real")]
pub use real_optimizer::{RealOptimizer, OptimizationSignal, MarketObjective};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarms_module_loads() {
        // Module smoke test
        assert!(true);
    }
}
