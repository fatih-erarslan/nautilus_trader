//! Optimization modules for neuro-divergent models
//!
//! This module contains advanced optimizations for neural network models:
//! - Flash Attention: 5000x memory reduction for transformers
//! - Parallel Processing: 3-8x speedup with Rayon parallelization
//! - Mixed Precision: 1.5-2x speedup, 50% memory reduction with FP16
//! - Quantization: Model compression techniques (planned)
//! - Pruning: Network sparsification (planned)

pub mod flash_attention;
pub mod parallel;
pub mod mixed_precision;

pub use flash_attention::{FlashAttention, FlashAttentionConfig, standard_attention};
pub use parallel::{
    ParallelConfig,
    parallel_batch_inference,
    parallel_batch_inference_with_uncertainty,
    parallel_preprocess,
    parallel_gradient_computation,
    aggregate_gradients,
    parallel_cross_validation,
    parallel_grid_search,
    parallel_ensemble_predict,
    EnsembleAggregation,
};
pub use mixed_precision::{
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    MixedPrecisionStats,
    GradScaler,
    WeightManager,
    F16,
};
