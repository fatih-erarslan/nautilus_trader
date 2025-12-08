//! Performance Module
//!
//! Comprehensive performance monitoring, metrics collection, and benchmarking for quantum trading operations.
//! 
//! This module provides:
//! - Real-time performance metrics collection and aggregation
//! - Comprehensive benchmarking suite for quantum circuits and Lightning devices
//! - Performance monitoring with threshold-based alerting
//! - Detailed performance analytics and reporting
//! - Lightning device hierarchy performance comparison

pub mod metrics;
pub mod benchmarks;

pub use metrics::{
    PerformanceMetricsManager, MetricType, AggregationType, TimeWindow,
    MetricDataPoint, AggregatedMetric, PerformanceThreshold, PerformanceAlert,
    PerformanceDashboard, MetricsConfig, MetricCollector, MetricExporter,
    AlertHandler, MockMetricCollector, MockMetricExporter, MockAlertHandler,
};

pub use benchmarks::{
    BenchmarkManager, BenchmarkType, BenchmarkConfig, BenchmarkResult,
    BenchmarkSuite, BenchmarkReport, LightningComparisonResult,
    CircuitInfo, PerformanceSummary,
};