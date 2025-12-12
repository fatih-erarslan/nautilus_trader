//! CDFA (Combinatorial Diversity Fusion Analysis) integration
//!
//! This module provides advanced algorithm fusion capabilities that leverage
//! the existing CDFA infrastructure for ultra-high performance optimization.
//!
//! ## Core Components
//!
//! - **Fusion Analyzer**: Core combinatorial diversity fusion analysis
//! - **Algorithm Pool**: Dynamic algorithm selection and management
//! - **Diversity Metrics**: Comprehensive diversity measurement and analysis
//! - **Performance Tracker**: Real-time performance monitoring and benchmarking
//! - **Adaptive Tuning**: Intelligent parameter optimization using ML
//! - **Enhancement Framework**: Algorithm performance enhancement and hybridization
//!
//! ## Features
//!
//! - Ultra-high performance parallel processing
//! - SIMD-accelerated diversity calculations
//! - Real-time performance monitoring
//! - Machine learning-based parameter tuning
//! - Automatic algorithm enhancement
//! - Hybrid algorithm creation
//! - Comprehensive benchmarking suite

pub mod fusion_analyzer;
pub mod algorithm_pool;
pub mod diversity_metrics;
pub mod performance_tracker;
pub mod adaptive_tuning;
pub mod enhancement_framework;
pub mod ml_integration;
pub mod qstar_weight_optimizer;
pub mod neural_signal_processor;
pub mod ensemble_coordinator;
pub mod redis_integration;
pub mod visualization_tools;
pub mod multilayer_fusion;

// Re-exports for fusion analysis
pub use fusion_analyzer::{
    CombinatorialDiversityFusionAnalyzer, FusionStrategy, 
    FusionResult, CombinationMetrics, FusionTiming,
    ConvergenceMetrics, ResourceUsage, CdfaSettings
};

// Re-exports for algorithm management
pub use algorithm_pool::{
    AlgorithmPool, PooledAlgorithm, PoolStrategy,
    AlgorithmSelector, PerformanceHistory, UsageStats,
    PoolStatistics, ProblemCharacteristics, ProblemComplexity
};

// Re-exports for diversity analysis
pub use diversity_metrics::{
    DiversityMetrics, DiversityCalculator, DiversityMeasure,
    DiversityConfig, DiversityType, SamplingStrategy,
    AlgorithmContext
};

// Re-exports for performance tracking
pub use performance_tracker::{
    PerformanceTracker, PerformanceMetrics, Benchmark,
    AlgorithmPerformanceHistory, TimestampedMetrics,
    TimingMetrics, ResourceMetrics, QualityMetrics,
    EfficiencyMetrics, ScalabilityMetrics, PerformanceSummary,
    PerformanceTrends, BenchmarkResult, SystemResourceMonitor
};

// Re-exports for adaptive tuning
pub use adaptive_tuning::{
    AdaptiveParameterTuning, TuningStrategy, ParameterSpace,
    ParameterDefinition, ParameterType, ParameterRange,
    TuningResult, AdaptiveTuningConfig, GridSearchStrategy,
    RandomSearchStrategy, BayesianOptimizationStrategy,
    MLModel, TuningStatistics
};

// Re-exports for enhancement framework
pub use enhancement_framework::{
    EnhancementFramework, AlgorithmInfo, AlgorithmType,
    AlgorithmCharacteristics, EnhancementStrategy, EnhancementType,
    EnhancementResult, EnhancementMetrics, EnhancedAlgorithm,
    HybridAlgorithmFactory, HybridAlgorithmBlueprint,
    EnhancementRecommendation, EnhancementConfig
};