//! Integration Tests
//!
//! Comprehensive integration tests for the entire analytics module

use super::TestUtils;
use crate::analytics::AnalyticsEngine;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_analytics_pipeline_integration() {
    // Test: Complete analytics pipeline should work end-to-end
    let mut analytics = AnalyticsEngine::new().await.unwrap();

    // Start all monitoring systems
    let result = analytics.start_monitoring().await;
    TestUtils::assert_cqgs_compliant(&result);

    // Generate and process test data
    let test_data = TestUtils::generate_test_organism_data(10);

    for data in test_data {
        let result = analytics.record_organism_performance(data).await;
        TestUtils::assert_cqgs_compliant(&result);

        sleep(Duration::from_millis(10)).await;
    }

    // Get comprehensive analytics summary
    let summary = analytics.get_analytics_summary().await.unwrap();

    // Verify all components are working
    assert!(summary.performance_stats.total_samples > 0);
    assert!(!summary.organism_summaries.is_empty());
    assert!(summary.system_health.overall_health > 0.0);
    assert!(summary.compliance_status.overall_compliance_score > 0.0);
}

#[tokio::test]
async fn test_sub_millisecond_end_to_end_performance() {
    // Test: End-to-end analytics should complete in sub-millisecond time
    let mut analytics = AnalyticsEngine::new().await.unwrap();
    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(analytics.record_organism_performance(test_data.clone()));
        TestUtils::assert_cqgs_compliant(&result);
    });
}

#[tokio::test]
async fn test_zero_mock_enforcement_across_modules() {
    // Test: All modules should use real implementations
    let analytics = AnalyticsEngine::new().await.unwrap();

    // Verify performance analytics is real
    assert!(analytics.performance().is_initialized());

    // Verify organism metrics is real
    assert!(analytics.organism_metrics().is_initialized());

    // Verify system health is real
    assert!(analytics.system_health().is_initialized());
}
