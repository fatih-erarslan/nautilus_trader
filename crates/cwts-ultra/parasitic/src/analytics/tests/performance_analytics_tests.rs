//! Performance Analytics Tests
//!
//! TDD tests for PerformanceAnalytics module with zero-mock enforcement

use super::TestUtils;
use crate::analytics::performance::PerformanceAnalytics;
use crate::analytics::{MetricAggregation, PerformanceMetric};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::test]
async fn test_performance_analytics_creation() {
    // Test: PerformanceAnalytics should be created with default configuration
    let analytics = PerformanceAnalytics::new();

    assert!(analytics.is_initialized());
    assert_eq!(analytics.get_buffer_size(), 10000); // Default buffer size
    assert!(analytics.get_active_metrics().is_empty());
}

#[tokio::test]
async fn test_sub_millisecond_metric_recording() {
    // Test: Recording performance metrics should complete in sub-millisecond time
    let mut analytics = PerformanceAnalytics::new();
    let test_data = TestUtils::generate_test_organism_data(1);

    TestUtils::assert_sub_millisecond(|| {
        let result = analytics.record_metric_sync(&test_data[0]);
        TestUtils::assert_cqgs_compliant(&result);
    });
}

#[tokio::test]
async fn test_async_metric_recording() {
    // Test: Async metric recording should work with real data
    let mut analytics = PerformanceAnalytics::new();
    let test_data = TestUtils::generate_test_organism_data(5);

    for data in test_data {
        let result = analytics.record_metric(data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    assert_eq!(analytics.get_total_recorded_metrics(), 5);
}

#[tokio::test]
async fn test_latency_tracking_precision() {
    // Test: Latency tracking should maintain nanosecond precision
    let mut analytics = PerformanceAnalytics::new();

    let precise_latencies = vec![1_234_567_u64, 2_345_678_u64, 3_456_789_u64];

    for (i, latency_ns) in precise_latencies.iter().enumerate() {
        let mut data = TestUtils::generate_test_organism_data(1)[0].clone();
        data.latency_ns = *latency_ns;

        let result = analytics.record_metric(data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let metrics = analytics.get_latency_metrics();
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].latency_ns, 1_234_567_u64);
}

#[tokio::test]
async fn test_throughput_calculation() {
    // Test: Real-time throughput calculation should be accurate
    let mut analytics = PerformanceAnalytics::new();
    let start_time = Instant::now();

    // Record metrics at different intervals
    for i in 0..10 {
        let mut data = TestUtils::generate_test_organism_data(1)[0].clone();
        data.trades_executed = (i + 1) as u64;
        data.throughput = (i + 1) as f64 * 10.0;

        let result = analytics.record_metric(data).await;
        TestUtils::assert_cqgs_compliant(&result);

        sleep(Duration::from_millis(10)).await;
    }

    let throughput_stats = analytics.calculate_throughput_stats().await;
    assert!(throughput_stats.current_tps > 0.0);
    assert!(throughput_stats.peak_tps >= throughput_stats.current_tps);
    assert!(throughput_stats.average_tps > 0.0);
}

#[tokio::test]
async fn test_performance_aggregation() {
    // Test: Performance metrics should be aggregated correctly
    let mut analytics = PerformanceAnalytics::new();
    let test_data = TestUtils::generate_test_organism_data(100);

    // Record all test data
    for data in test_data.iter() {
        let result = analytics.record_metric(data.clone()).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let aggregation = analytics
        .aggregate_metrics(MetricAggregation::Last1Minute)
        .await;
    assert!(aggregation.is_ok());

    let stats = aggregation.unwrap();
    assert_eq!(stats.total_samples, 100);
    assert!(stats.average_latency_ns > 0);
    assert!(stats.percentile_95_latency_ns >= stats.average_latency_ns);
    assert!(stats.peak_tps > 0.0);
}

#[tokio::test]
async fn test_real_time_monitoring() {
    // Test: Real-time monitoring should work with continuous data streams
    let mut analytics = PerformanceAnalytics::new();

    // Start monitoring
    let monitoring_handle = analytics.start_real_time_monitoring().await;
    assert!(monitoring_handle.is_ok());

    // Stream data for 100ms
    let stream_task = tokio::spawn(async move {
        for i in 0..10 {
            let data = TestUtils::generate_test_organism_data(1)[0].clone();
            let _ = analytics.record_metric(data).await;
            sleep(Duration::from_millis(10)).await;
        }
        analytics
    });

    let analytics = stream_task.await.unwrap();
    let real_time_stats = analytics.get_real_time_stats();

    assert!(real_time_stats.active_streams > 0);
    assert!(real_time_stats.total_processed > 0);
}

#[tokio::test]
async fn test_performance_alerts() {
    // Test: Performance alerts should trigger on threshold violations
    let mut analytics = PerformanceAnalytics::with_thresholds(
        Duration::from_millis(1), // Max latency: 1ms
        100.0,                    // Min throughput: 100 TPS
        0.95,                     // Min success rate: 95%
    );

    // Create data that violates thresholds
    let mut high_latency_data = TestUtils::generate_test_organism_data(1)[0].clone();
    high_latency_data.latency_ns = 2_000_000; // 2ms - exceeds threshold
    high_latency_data.success_rate = 0.90; // Below threshold

    let result = analytics.record_metric(high_latency_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let alerts = analytics.get_active_alerts();
    assert!(!alerts.is_empty());
    assert!(alerts
        .iter()
        .any(|alert| alert.alert_type.contains("Latency")));
    assert!(alerts
        .iter()
        .any(|alert| alert.alert_type.contains("SuccessRate")));
}

#[tokio::test]
async fn test_historical_data_retention() {
    // Test: Historical data should be retained according to policy
    let mut analytics = PerformanceAnalytics::with_retention_policy(
        1000,                    // Buffer size
        Duration::from_secs(60), // Retention time
    );

    // Fill buffer beyond capacity
    for i in 0..1500 {
        let data = TestUtils::generate_test_organism_data(1)[0].clone();
        let result = analytics.record_metric(data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    // Should retain only buffer_size metrics
    assert!(analytics.get_total_recorded_metrics() <= 1000);

    let historical_stats = analytics
        .get_historical_summary(Duration::from_secs(30))
        .await;
    assert!(historical_stats.is_ok());
}

#[tokio::test]
async fn test_concurrent_metric_recording() {
    // Test: Concurrent metric recording should be thread-safe
    let analytics = std::sync::Arc::new(tokio::sync::Mutex::new(PerformanceAnalytics::new()));
    let mut handles = vec![];

    // Spawn 10 concurrent tasks
    for task_id in 0..10 {
        let analytics_clone = analytics.clone();
        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let mut data = TestUtils::generate_test_organism_data(1)[0].clone();
                data.organism_type = format!("task_{}_{}", task_id, i);

                let mut analytics_guard = analytics_clone.lock().await;
                let result = analytics_guard.record_metric(data).await;
                drop(analytics_guard);

                TestUtils::assert_cqgs_compliant(&result);
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    let analytics_guard = analytics.lock().await;
    assert_eq!(analytics_guard.get_total_recorded_metrics(), 100); // 10 tasks * 10 metrics
}

#[tokio::test]
async fn test_zero_mock_enforcement() {
    // Test: All operations should use real implementations, no mocks
    let mut analytics = PerformanceAnalytics::new();

    // Use real system time
    let real_data = TestUtils::generate_test_organism_data(1)[0].clone();
    let before_record = chrono::Utc::now();

    let result = analytics.record_metric(real_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let after_record = chrono::Utc::now();
    let stored_metrics = analytics.get_recent_metrics(1);

    // Verify real timestamps
    assert!(!stored_metrics.is_empty());
    assert!(stored_metrics[0].timestamp >= before_record);
    assert!(stored_metrics[0].timestamp <= after_record);

    // Verify real system resources are tracked
    let system_stats = analytics.get_system_resource_usage();
    assert!(system_stats.memory_usage_bytes > 0);
    assert!(system_stats.cpu_usage_percent >= 0.0);
}
