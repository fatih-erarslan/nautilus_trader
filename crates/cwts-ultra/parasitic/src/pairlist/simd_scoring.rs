//! # SIMD-Optimized Scoring
//! 
//! High-performance vectorized scoring for pair analysis

pub use crate::pairlist::selection_engine::{SimdPairScorer, AlignedWeights, SIMDMetrics, PairFeatures};

// Re-export for compatibility
pub type SimdScorer = SimdPairScorer;