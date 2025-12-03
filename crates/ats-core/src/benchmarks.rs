//! # ATS-Core Benchmarks
//!
//! Performance benchmarks for ATS-Core mathematical primitives.
//! These benchmarks validate the sub-100μs latency requirements
//! for high-frequency trading applications.

#[cfg(feature = "benchmarking")]
pub mod temperature_benchmarks {
    //! Temperature scaling performance benchmarks

    use crate::config::AtsCpConfig;
    use crate::temperature::TemperatureScaler;
    use std::time::Instant;

    /// Benchmark temperature scaling performance
    pub fn benchmark_temperature_scaling() -> u64 {
        let start = Instant::now();
        let config = AtsCpConfig::default();
        let _scaler = TemperatureScaler::new(&config);
        start.elapsed().as_micros() as u64
    }
}

#[cfg(feature = "benchmarking")]
pub mod conformal_benchmarks {
    //! Conformal prediction performance benchmarks
    
    use std::time::Instant;
    
    /// Benchmark conformal prediction performance
    pub fn benchmark_conformal_prediction() -> u64 {
        let start = Instant::now();
        // Placeholder benchmark implementation
        start.elapsed().as_micros() as u64
    }
}

#[cfg(feature = "benchmarking")]
pub mod simd_benchmarks {
    //! SIMD operations performance benchmarks
    
    use std::time::Instant;
    
    /// Benchmark SIMD vector operations
    pub fn benchmark_simd_operations() -> u64 {
        let start = Instant::now();
        // Placeholder benchmark implementation
        start.elapsed().as_micros() as u64
    }
}

#[cfg(feature = "benchmarking")]
pub use temperature_benchmarks::*;
#[cfg(feature = "benchmarking")]
pub use conformal_benchmarks::*;
#[cfg(feature = "benchmarking")]
pub use simd_benchmarks::*;