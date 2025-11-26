//! Analytics Module Tests
//!
//! Comprehensive test suite for the analytics module following TDD methodology
//! with zero-mock enforcement and CQGS compliance.

pub mod cqgs_compliance_tests;
pub mod integration_tests;
pub mod organism_metrics_tests;
pub mod performance_analytics_tests;
pub mod performance_benchmarks;
pub mod system_health_tests;

use std::time::{Duration, Instant};

/// Test utilities for analytics module
pub struct TestUtils;

impl TestUtils {
    /// Generate test organism performance data
    pub fn generate_test_organism_data(
        count: usize,
    ) -> Vec<crate::analytics::OrganismPerformanceData> {
        (0..count)
            .map(|i| {
                crate::analytics::OrganismPerformanceData {
                    organism_id: uuid::Uuid::new_v4(),
                    organism_type: format!("test_organism_{}", i % 10),
                    timestamp: chrono::Utc::now(),
                    latency_ns: 50000 + (i as u64 * 1000), // Varying latency
                    throughput: 100.0 + (i as f64 * 10.0),
                    success_rate: 0.95 - (i as f64 * 0.001),
                    resource_usage: crate::organisms::ResourceMetrics {
                        cpu_usage: 10.0 + (i as f64),
                        memory_mb: 50.0 + (i as f64 * 5.0),
                        network_bandwidth_kbps: 100.0 + (i as f64 * 10.0),
                        api_calls_per_second: 10.0 + (i as f64),
                        latency_overhead_ns: 1000 + (i as u64 * 100),
                    },
                    profit: 100.0 + (i as f64 * 50.0),
                    trades_executed: 10 + i as u64,
                }
            })
            .collect()
    }

    /// Assert sub-millisecond performance
    pub fn assert_sub_millisecond<F>(operation: F)
    where
        F: FnOnce(),
    {
        let start = Instant::now();
        operation();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(1),
            "Operation took {:?}, exceeds sub-millisecond requirement",
            elapsed
        );
    }

    /// Assert CQGS compliance
    pub fn assert_cqgs_compliant<T, E: std::fmt::Debug>(result: &Result<T, E>) {
        assert!(
            result.is_ok(),
            "CQGS compliance failure: operation must succeed with real implementation. Error: {:?}",
            result.as_ref().err()
        );
    }
}
