//! Prelude module for common imports
//!
//! This module re-exports commonly used traits and types for easier importing.

// Re-export common traits
pub use crate::traits::{
    SystemAnalyzer,
    DiversityMethod,
    FusionMethod,
    PatternDetector,
    SignalAlgorithm,
    FeatureExtractor,
    Analyzer,
};

// Re-export core types
pub use crate::types::{
    Float,
    FloatArray1,
    FloatArray2,
    FloatArrayView1,
    FloatArrayView2,
    CdfaConfig,
    PerformanceMetrics,
    Index,
    Timestamp,
};

// Re-export error types
pub use crate::error::{CdfaError, Result};

// Re-export unified API
#[cfg(feature = "core")]
pub use crate::unified::UnifiedCdfa;

#[cfg(feature = "core")]
pub use crate::builder::UnifiedCdfaBuilder;

// Re-export core algorithms and diversity methods
pub use crate::core::diversity::{
    pearson_correlation,
    spearman_correlation,
    kendall_tau,
    comprehensive_diversity_analysis,
    quick_correlation,
    CorrelationMethod,
    DiversityAnalysis,
    DiversityMatrix,
    PairwiseDiversity,
};

// Re-export diversity implementations
pub use crate::core::diversity::{
    PearsonDiversity,
    SpearmanDiversity,
    KendallTauDiversity,
};

// Re-export fusion methods (if they exist)
// TODO: Add fusion method structs when implemented