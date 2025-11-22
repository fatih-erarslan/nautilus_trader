//! System Health Monitor Tests
//!
//! TDD tests for SystemHealthMonitor module with comprehensive system monitoring

use super::TestUtils;
use crate::analytics::health::SystemHealthMonitor;
use crate::analytics::{OrganismPerformanceData, SystemHealthStatus};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::test]
async fn test_system_health_monitor_creation() {
    // Test: SystemHealthMonitor should be created and initialized
    let monitor = SystemHealthMonitor::new().await.unwrap();

    assert!(monitor.is_initialized());
    assert!(monitor.get_monitored_components().len() > 0);
    assert_eq!(monitor.get_alert_threshold_count(), 5); // Default thresholds
}

#[tokio::test]
async fn test_system_health_scoring() {
    // Test: System health should be calculated accurately
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Record some performance data to establish baseline
    let test_data = TestUtils::generate_test_organism_data(5);
    for data in test_data {
        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let health_status = monitor.get_current_health().await.unwrap();
    assert!(health_status.overall_health >= 0.0 && health_status.overall_health <= 1.0);
    assert!(health_status.performance_score >= 0.0);
    assert!(health_status.resource_utilization >= 0.0);
    assert_eq!(health_status.component_health.len(), 6); // Expected number of components
}

#[tokio::test]
async fn test_component_health_tracking() {
    // Test: Individual component health should be tracked
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Simulate component-specific issues
    let degraded_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "test_organism".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 2_000_000, // 2ms - degraded performance
        throughput: 10.0,      // Low throughput
        success_rate: 0.70,    // Low success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 85.0,  // High CPU usage
            memory_mb: 500.0, // High memory usage
            network_bandwidth_kbps: 1000.0,
            api_calls_per_second: 2.0, // Low API calls
            latency_overhead_ns: 10000,
        },
        profit: 20.0, // Low profit
        trades_executed: 2,
    };

    let result = monitor.record_performance_data(&degraded_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let health_status = monitor.get_current_health().await.unwrap();

    // Check individual component health scores
    assert!(health_status.component_health.contains_key("latency"));
    assert!(health_status.component_health.contains_key("throughput"));
    assert!(health_status
        .component_health
        .contains_key("resource_usage"));
    assert!(health_status.component_health.contains_key("success_rate"));

    // Components should show degraded health
    assert!(health_status.component_health["latency"] < 0.8);
    assert!(health_status.component_health["throughput"] < 0.8);
    assert!(health_status.component_health["success_rate"] < 0.8);
}

#[tokio::test]
async fn test_health_trend_analysis() {
    // Test: Health trends should be tracked over time
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Simulate degrading performance over time
    let performance_levels = vec![0.95, 0.90, 0.85, 0.80, 0.75];

    for (i, performance) in performance_levels.iter().enumerate() {
        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: format!("organism_{}", i),
            timestamp: chrono::Utc::now() - chrono::Duration::seconds((5 - i) as i64),
            latency_ns: (50_000.0 / performance) as u64, // Inversely related to performance
            throughput: 100.0 * performance,
            success_rate: *performance,
            resource_usage: crate::organisms::ResourceMetrics {
                cpu_usage: 20.0 + (1.0 - performance) * 60.0, // Increases as performance drops
                memory_mb: 50.0 + (1.0 - performance) * 200.0,
                network_bandwidth_kbps: 100.0,
                api_calls_per_second: 10.0 * performance,
                latency_overhead_ns: ((1.0 - performance) * 5000.0) as u64,
            },
            profit: 200.0 * performance,
            trades_executed: (20.0 * performance) as u64,
        };

        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);

        sleep(Duration::from_millis(10)).await; // Ensure time progression
    }

    let health_trend = monitor
        .get_health_trend(chrono::Duration::minutes(1))
        .await
        .unwrap();
    assert!(health_trend.overall_trend < 0.0); // Should show negative trend
    assert!(health_trend.component_trends.len() > 0);
    assert!(health_trend.prediction_confidence > 0.0);
}

#[tokio::test]
async fn test_real_time_monitoring() {
    // Test: Real-time monitoring should provide live updates
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Start monitoring
    let result = monitor.start_monitoring().await;
    TestUtils::assert_cqgs_compliant(&result);

    // Record some data while monitoring is active
    for i in 0..5 {
        let data = TestUtils::generate_test_organism_data(1)[0].clone();
        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);

        sleep(Duration::from_millis(20)).await;
    }

    let monitoring_stats = monitor.get_monitoring_statistics().await.unwrap();
    assert!(monitoring_stats.is_active);
    assert!(monitoring_stats.samples_processed > 0);
    assert!(monitoring_stats.uptime_seconds > 0.0);
}

#[tokio::test]
async fn test_alert_generation() {
    // Test: Health alerts should be generated for critical conditions
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Set strict thresholds for testing
    monitor
        .set_health_thresholds(
            0.9,  // Min overall health
            0.95, // Min component health
            10,   // Max alerts before escalation
        )
        .await
        .unwrap();

    // Create data that should trigger alerts
    let critical_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "critical_organism".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 5_000_000, // 5ms - critical latency
        throughput: 5.0,       // Very low throughput
        success_rate: 0.50,    // Critical success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 95.0,  // Critical CPU usage
            memory_mb: 800.0, // Critical memory usage
            network_bandwidth_kbps: 2000.0,
            api_calls_per_second: 1.0,
            latency_overhead_ns: 50000,
        },
        profit: 5.0, // Very low profit
        trades_executed: 1,
    };

    let result = monitor.record_performance_data(&critical_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let alerts = monitor.get_active_alerts().await.unwrap();
    assert!(!alerts.is_empty());

    // Should have alerts for multiple components
    let alert_types: std::collections::HashSet<_> = alerts.iter().map(|a| &a.component).collect();
    assert!(alert_types.contains(&"latency".to_string()));
    assert!(alert_types.contains(&"success_rate".to_string()));
}

#[tokio::test]
async fn test_system_recovery_detection() {
    // Test: Should detect when system recovers from issues
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Start with poor performance
    let poor_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "recovery_test".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 3_000_000,
        throughput: 20.0,
        success_rate: 0.60,
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 80.0,
            memory_mb: 400.0,
            network_bandwidth_kbps: 800.0,
            api_calls_per_second: 5.0,
            latency_overhead_ns: 20000,
        },
        profit: 50.0,
        trades_executed: 3,
    };

    let result = monitor.record_performance_data(&poor_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let initial_health = monitor.get_current_health().await.unwrap().overall_health;

    // Improve performance
    let good_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "recovery_test".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 40_000,
        throughput: 120.0,
        success_rate: 0.95,
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 15.0,
            memory_mb: 60.0,
            network_bandwidth_kbps: 200.0,
            api_calls_per_second: 15.0,
            latency_overhead_ns: 2000,
        },
        profit: 300.0,
        trades_executed: 18,
    };

    let result = monitor.record_performance_data(&good_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let recovered_health = monitor.get_current_health().await.unwrap().overall_health;
    assert!(recovered_health > initial_health);

    let recovery_events = monitor.get_recovery_events().await.unwrap();
    assert!(!recovery_events.is_empty());
}

#[tokio::test]
async fn test_resource_utilization_tracking() {
    // Test: System resource utilization should be tracked accurately
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Record data with varying resource usage
    let resource_patterns = vec![
        (10.0, 50.0),  // Low CPU, Low Memory
        (30.0, 150.0), // Medium CPU, Medium Memory
        (60.0, 300.0), // High CPU, High Memory
        (20.0, 80.0),  // Back to lower usage
    ];

    for (cpu, memory) in resource_patterns {
        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: "resource_test".to_string(),
            timestamp: chrono::Utc::now(),
            latency_ns: 45_000,
            throughput: 100.0,
            success_rate: 0.90,
            resource_usage: crate::organisms::ResourceMetrics {
                cpu_usage: cpu,
                memory_mb: memory,
                network_bandwidth_kbps: 150.0,
                api_calls_per_second: 12.0,
                latency_overhead_ns: 3000,
            },
            profit: 200.0,
            trades_executed: 15,
        };

        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let resource_stats = monitor.get_resource_utilization_stats().await.unwrap();
    assert!(resource_stats.peak_cpu_usage >= 60.0);
    assert!(resource_stats.peak_memory_usage >= 300.0);
    assert!(resource_stats.average_cpu_usage > 0.0);
    assert!(resource_stats.average_memory_usage > 0.0);
    assert!(resource_stats.current_utilization_level > 0.0);
}

#[tokio::test]
async fn test_predictive_health_analysis() {
    // Test: Should provide predictive analysis of health trends
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Create a pattern that shows gradual degradation
    for i in 0..20 {
        let degradation_factor = i as f64 * 0.02; // Gradual decline

        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: format!("predictive_test_{}", i),
            timestamp: chrono::Utc::now() - chrono::Duration::seconds((20 - i) as i64),
            latency_ns: (45_000.0 * (1.0 + degradation_factor)) as u64,
            throughput: 100.0 * (1.0 - degradation_factor),
            success_rate: 0.95 - degradation_factor,
            resource_usage: crate::organisms::ResourceMetrics {
                cpu_usage: 20.0 + degradation_factor * 40.0,
                memory_mb: 60.0 + degradation_factor * 200.0,
                network_bandwidth_kbps: 150.0,
                api_calls_per_second: 12.0,
                latency_overhead_ns: (3000.0 * (1.0 + degradation_factor)) as u64,
            },
            profit: 200.0 * (1.0 - degradation_factor),
            trades_executed: (15.0 * (1.0 - degradation_factor)) as u64,
        };

        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let prediction = monitor.predict_health_in_minutes(30).await.unwrap();
    assert!(prediction.predicted_health < 0.8); // Should predict degradation
    assert!(prediction.confidence > 0.5); // Should have reasonable confidence
    assert!(!prediction.recommended_actions.is_empty()); // Should provide recommendations
}

#[tokio::test]
async fn test_sub_millisecond_performance() {
    // Test: Health monitoring operations should be sub-millisecond
    let mut monitor = SystemHealthMonitor::new().await.unwrap();
    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    // Test health update performance
    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(monitor.record_performance_data(&test_data));
        TestUtils::assert_cqgs_compliant(&result);
    });

    // Test health retrieval performance
    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(monitor.get_current_health());
        assert!(result.is_ok());
    });
}

#[tokio::test]
async fn test_zero_mock_enforcement() {
    // Test: All operations should use real implementations, no mocks
    let mut monitor = SystemHealthMonitor::new().await.unwrap();

    // Use real system data
    let real_data = TestUtils::generate_test_organism_data(1)[0].clone();
    let before_record = chrono::Utc::now();

    let result = monitor.record_performance_data(&real_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let after_record = chrono::Utc::now();
    let health_status = monitor.get_current_health().await.unwrap();

    // Verify real timestamps
    assert!(health_status.timestamp >= before_record);
    assert!(health_status.timestamp <= after_record);

    // Verify real system metrics are used
    assert!(health_status.resource_utilization >= 0.0);
    assert!(health_status.overall_health >= 0.0 && health_status.overall_health <= 1.0);

    // Verify component health uses real calculations
    for (component, health) in health_status.component_health {
        assert!(
            health >= 0.0 && health <= 1.0,
            "Component {} has invalid health: {}",
            component,
            health
        );
    }
}

#[tokio::test]
async fn test_cqgs_integration() {
    // Test: Should integrate properly with CQGS compliance system
    let mut monitor = SystemHealthMonitor::with_cqgs_integration().await.unwrap();

    // Record performance data that should trigger CQGS validation
    let test_data = TestUtils::generate_test_organism_data(3);
    for data in test_data {
        let result = monitor.record_performance_data(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    // Check CQGS compliance status
    let compliance_info = monitor.get_cqgs_compliance_info().await.unwrap();
    assert!(compliance_info.compliance_score >= 0.0 && compliance_info.compliance_score <= 1.0);
    assert!(!compliance_info.validation_results.is_empty());

    // Should have real validation results, not mocked ones
    for validation in compliance_info.validation_results {
        assert!(!validation.validator_name.is_empty());
        assert!(validation.timestamp > chrono::Utc::now() - chrono::Duration::minutes(1));
    }
}
