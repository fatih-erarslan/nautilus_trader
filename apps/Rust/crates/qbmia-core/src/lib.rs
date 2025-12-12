//! QBMIA Core - Quantum-Biological Market Intuition Agent
//!
//! This crate provides high-performance Rust implementations of quantum-biological 
//! algorithms for market analysis and decision making.
//!
//! ## Key Components
//!
//! - **Quantum Nash Equilibrium Solver**: GPU-accelerated quantum game theory
//! - **Machiavellian Strategic Framework**: Market manipulation detection
//! - **Biological Memory System**: Adaptive memory patterns
//! - **State Management**: Efficient state serialization and recovery
//!
//! ## Performance Features
//!
//! - SIMD optimization for numerical operations
//! - Parallel processing with Rayon
//! - Zero-copy serialization
//! - Memory-mapped storage for large datasets
//!
//! ## Example Usage
//!
//! ```rust
//! use qbmia_core::{QBMIAAgent, Config};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = Config::default();
//!     let mut agent = QBMIAAgent::new(config).await?;
//!     
//!     // Analyze market data
//!     let analysis = agent.analyze_market(market_data).await?;
//!     
//!     println!("Decision: {:?}", analysis.decision);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod config;
pub mod error;
pub mod quantum;
pub mod strategy;
pub mod memory;
pub mod state;
pub mod utils;

// Re-export main types
pub use agent::QBMIAAgent;
pub use config::Config;
pub use error::{QBMIAError, Result};

// Feature-gated exports
#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_HASH: &str = match option_env!("VERGEN_GIT_SHA") {
    Some(hash) => hash,
    None => "unknown",
};

/// Initialize logging for the library
pub fn init_logging() {
    env_logger::init();
}

/// Performance profiling utilities
pub mod profiling {
    use std::time::{Duration, Instant};
    
    /// Simple profiler for measuring execution time
    pub struct Profiler {
        start: Instant,
        name: String,
    }
    
    impl Profiler {
        pub fn new(name: &str) -> Self {
            log::debug!("Starting profiler: {}", name);
            Self {
                start: Instant::now(),
                name: name.to_string(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
    }
    
    impl Drop for Profiler {
        fn drop(&mut self) {
            let elapsed = self.elapsed();
            log::debug!("Profiler '{}' took: {:?}", self.name, elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!GIT_HASH.is_empty());
    }
    
    #[test]
    fn test_profiler() {
        let profiler = profiling::Profiler::new("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(profiler.elapsed() >= std::time::Duration::from_millis(1));
    }
}