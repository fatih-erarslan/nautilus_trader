//! CQGS Compliance Tests
//!
//! TDD tests for CqgsComplianceTracker module with zero-mock enforcement

use super::TestUtils;
use crate::analytics::compliance::CqgsComplianceTracker;
use crate::analytics::OrganismPerformanceData;
use crate::cqgs::{QualityGateDecision, ViolationSeverity};

#[tokio::test]
async fn test_cqgs_compliance_tracker_creation() {
    // Test: CqgsComplianceTracker should be created and integrated with CQGS
    let tracker = CqgsComplianceTracker::new().await.unwrap();

    assert!(tracker.is_initialized());
    assert_eq!(tracker.get_sentinel_count(), 49); // Should integrate with 49 sentinels
    assert!(tracker.is_cqgs_connected().await);
}

#[tokio::test]
async fn test_zero_mock_enforcement() {
    // Test: All validations should use real CQGS sentinels, no mocks
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    let result = tracker.validate_performance(&test_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    // Verify real CQGS integration
    let compliance_status = tracker.get_compliance_status().await.unwrap();
    assert!(compliance_status.sentinel_validations.len() > 0);

    // Each validation should have real sentinel data
    for validation in compliance_status.sentinel_validations {
        assert!(!validation.sentinel_id.is_empty());
        assert!(validation.timestamp > chrono::Utc::now() - chrono::Duration::minutes(1));
        assert!(validation.validation_score >= 0.0 && validation.validation_score <= 1.0);
    }
}

#[tokio::test]
async fn test_performance_validation_with_sentinels() {
    // Test: Performance data should be validated by multiple CQGS sentinels
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // High-quality performance data
    let excellent_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "cuckoo".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 25_000, // Excellent latency (25μs)
        throughput: 200.0,  // High throughput
        success_rate: 0.98, // Excellent success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 5.0,
            memory_mb: 20.0,
            network_bandwidth_kbps: 50.0,
            api_calls_per_second: 25.0,
            latency_overhead_ns: 500,
        },
        profit: 500.0,
        trades_executed: 30,
    };

    let result = tracker.validate_performance(&excellent_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let status = tracker.get_compliance_status().await.unwrap();
    assert!(status.overall_compliance_score > 0.9); // Should be high compliance
    assert!(status.quality_gate_decision == QualityGateDecision::Pass);
}

#[tokio::test]
async fn test_quality_gate_decisions() {
    // Test: CQGS should make quality gate decisions based on performance
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // Poor performance data that should fail quality gates
    let poor_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "poor_performer".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 5_000_000, // 5ms - very poor latency
        throughput: 5.0,       // Very low throughput
        success_rate: 0.45,    // Poor success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 95.0,  // Critical CPU usage
            memory_mb: 800.0, // Critical memory usage
            network_bandwidth_kbps: 2000.0,
            api_calls_per_second: 1.0,
            latency_overhead_ns: 100_000,
        },
        profit: 10.0, // Very low profit
        trades_executed: 1,
    };

    let result = tracker.validate_performance(&poor_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let status = tracker.get_compliance_status().await.unwrap();
    assert!(status.overall_compliance_score < 0.5); // Should be low compliance
    assert!(
        status.quality_gate_decision == QualityGateDecision::Fail
            || status.quality_gate_decision == QualityGateDecision::RequireRemediation
    );

    // Should have violation reports
    assert!(!status.violation_reports.is_empty());

    // Violations should be properly categorized
    let critical_violations: Vec<_> = status
        .violation_reports
        .iter()
        .filter(|v| v.severity == ViolationSeverity::Critical)
        .collect();
    assert!(!critical_violations.is_empty());
}

#[tokio::test]
async fn test_sentinel_consensus() {
    // Test: Multiple sentinels should reach consensus on quality decisions
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    let result = tracker.validate_performance(&test_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    // Get consensus information
    let consensus_info = tracker.get_sentinel_consensus().await.unwrap();
    assert!(consensus_info.participating_sentinels >= 3); // Minimum for consensus
    assert!(consensus_info.consensus_reached);
    assert!(consensus_info.consensus_score >= 0.67); // 2/3 threshold

    // Verify voting results are from real sentinels
    assert!(!consensus_info.voting_results.is_empty());
    for vote in consensus_info.voting_results {
        assert!(!vote.sentinel_id.is_empty());
        assert!(vote.confidence >= 0.0 && vote.confidence <= 1.0);
    }
}

#[tokio::test]
async fn test_real_time_compliance_monitoring() {
    // Test: Should provide real-time compliance monitoring
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // Start real-time monitoring
    let result = tracker.start_compliance_monitoring().await;
    TestUtils::assert_cqgs_compliant(&result);

    // Feed multiple data points
    let test_data = TestUtils::generate_test_organism_data(5);
    for data in test_data {
        let result = tracker.validate_performance(&data).await;
        TestUtils::assert_cqgs_compliant(&result);

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Check monitoring statistics
    let monitoring_stats = tracker.get_monitoring_statistics().await.unwrap();
    assert!(monitoring_stats.is_monitoring_active);
    assert!(monitoring_stats.validations_performed > 0);
    assert!(monitoring_stats.sentinel_uptime_percentage > 90.0);
    assert!(monitoring_stats.average_validation_time_ms < 1.0); // Sub-millisecond
}

#[tokio::test]
async fn test_compliance_score_calculation() {
    // Test: Compliance scores should be calculated accurately
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // Test different quality levels
    let quality_levels = vec![
        (0.98, 200.0, 0.97), // Excellent
        (0.85, 100.0, 0.90), // Good
        (0.70, 50.0, 0.80),  // Average
        (0.50, 20.0, 0.60),  // Poor
    ];

    let mut compliance_scores = Vec::new();

    for (success_rate, throughput, expected_range) in quality_levels {
        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: "compliance_test".to_string(),
            timestamp: chrono::Utc::now(),
            latency_ns: 50_000,
            throughput,
            success_rate,
            resource_usage: crate::organisms::ResourceMetrics::default(),
            profit: throughput * 2.0,
            trades_executed: (throughput / 10.0) as u64,
        };

        let result = tracker.validate_performance(&data).await;
        TestUtils::assert_cqgs_compliant(&result);

        let status = tracker.get_compliance_status().await.unwrap();
        compliance_scores.push(status.overall_compliance_score);

        // Compliance score should correlate with performance quality
        assert!(status.overall_compliance_score <= expected_range + 0.1);
        assert!(status.overall_compliance_score >= expected_range - 0.2);
    }

    // Scores should generally decrease with lower quality
    assert!(compliance_scores[0] >= compliance_scores[1]);
    assert!(compliance_scores[1] >= compliance_scores[2]);
}

#[tokio::test]
async fn test_violation_categorization() {
    // Test: Violations should be properly categorized by severity
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // Create data with multiple violation types
    let violation_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "violation_test".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 10_000_000, // 10ms - critical violation
        throughput: 2.0,        // Critical throughput
        success_rate: 0.30,     // Critical success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 98.0,  // Critical CPU
            memory_mb: 900.0, // Critical memory
            network_bandwidth_kbps: 5000.0,
            api_calls_per_second: 0.5,
            latency_overhead_ns: 200_000,
        },
        profit: 1.0,        // Critical profit
        trades_executed: 0, // No trades - critical
    };

    let result = tracker.validate_performance(&violation_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let status = tracker.get_compliance_status().await.unwrap();
    let violations = status.violation_reports;

    // Should have violations across different severity levels
    let severity_counts: std::collections::HashMap<ViolationSeverity, usize> = violations
        .iter()
        .fold(std::collections::HashMap::new(), |mut acc, v| {
            *acc.entry(v.severity.clone()).or_insert(0) += 1;
            acc
        });

    assert!(severity_counts.contains_key(&ViolationSeverity::Critical));
    assert!(severity_counts.contains_key(&ViolationSeverity::Error));

    // Should have multiple violation categories
    let categories: std::collections::HashSet<_> = violations.iter().map(|v| &v.category).collect();

    assert!(categories.len() > 2); // Multiple categories
    assert!(categories.contains(&"performance".to_string()));
    assert!(categories.contains(&"resource_usage".to_string()));
}

#[tokio::test]
async fn test_remediation_suggestions() {
    // Test: Should provide actionable remediation suggestions
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    let degraded_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "remediation_test".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 2_000_000, // 2ms - needs improvement
        throughput: 30.0,      // Low throughput
        success_rate: 0.75,    // Below optimal
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 70.0,  // High but not critical
            memory_mb: 300.0, // High but not critical
            network_bandwidth_kbps: 800.0,
            api_calls_per_second: 8.0,
            latency_overhead_ns: 15_000,
        },
        profit: 120.0,
        trades_executed: 8,
    };

    let result = tracker.validate_performance(&degraded_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let remediation_plan = tracker.get_remediation_suggestions().await.unwrap();
    assert!(!remediation_plan.suggestions.is_empty());

    // Should have specific, actionable suggestions
    let suggestion_text: String = remediation_plan.suggestions.join(" ");
    assert!(
        suggestion_text.to_lowercase().contains("latency")
            || suggestion_text.to_lowercase().contains("throughput")
            || suggestion_text.to_lowercase().contains("resource")
    );

    // Should have priority ordering
    assert!(remediation_plan.priority_order.len() == remediation_plan.suggestions.len());

    // Should have estimated impact
    for impact in remediation_plan.estimated_impact {
        assert!(impact >= 0.0 && impact <= 1.0);
    }
}

#[tokio::test]
async fn test_sub_millisecond_validation_performance() {
    // Test: CQGS validation should complete in sub-millisecond time
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();
    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(tracker.validate_performance(&test_data));
        TestUtils::assert_cqgs_compliant(&result);
    });

    // Batch validation performance
    let batch_data = TestUtils::generate_test_organism_data(10);
    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        for data in &batch_data {
            let result = rt.block_on(tracker.validate_performance(data));
            TestUtils::assert_cqgs_compliant(&result);
        }
    });
}

#[tokio::test]
async fn test_hyperbolic_topology_integration() {
    // Test: Should integrate with CQGS hyperbolic topology
    let tracker = CqgsComplianceTracker::new().await.unwrap();

    let topology_info = tracker.get_hyperbolic_topology_info().await.unwrap();
    assert_eq!(topology_info.curvature, -1.5); // Default hyperbolic curvature
    assert_eq!(topology_info.sentinel_positions.len(), 49); // All sentinels positioned

    // Each sentinel should have valid hyperbolic coordinates
    for position in topology_info.sentinel_positions {
        assert!(position.radius >= 0.0 && position.radius < 1.0); // Valid Poincaré disk
        assert!(!position.sentinel_id.is_empty());
    }

    // Should have coordination efficiency metrics
    assert!(topology_info.coordination_efficiency > 0.9); // Hyperbolic advantage
    assert!(topology_info.path_optimization_factor > 1.0); // Better than Euclidean
}

#[tokio::test]
async fn test_neural_pattern_learning() {
    // Test: Should learn from validation patterns for improved accuracy
    let mut tracker = CqgsComplianceTracker::new().await.unwrap();

    // Feed consistent patterns
    let pattern_data = vec![
        (0.95, 150.0, 0.96), // High quality pattern
        (0.94, 148.0, 0.95),
        (0.96, 152.0, 0.97),
        (0.45, 25.0, 0.40), // Low quality pattern
        (0.44, 23.0, 0.41),
        (0.46, 27.0, 0.42),
    ];

    for (success_rate, throughput, expected_score) in pattern_data {
        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: "pattern_learning".to_string(),
            timestamp: chrono::Utc::now(),
            latency_ns: (100_000.0 / success_rate) as u64,
            throughput,
            success_rate,
            resource_usage: crate::organisms::ResourceMetrics::default(),
            profit: throughput * success_rate,
            trades_executed: (throughput / 10.0) as u64,
        };

        let result = tracker.validate_performance(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    // Check neural learning status
    let learning_status = tracker.get_neural_learning_status().await.unwrap();
    assert!(learning_status.patterns_learned > 0);
    assert!(learning_status.prediction_accuracy > 0.7);
    assert!(learning_status.confidence_improvement > 0.0);

    // Should have learned quality patterns
    let quality_patterns = learning_status.recognized_patterns;
    assert!(!quality_patterns.is_empty());

    for pattern in quality_patterns {
        assert!(!pattern.pattern_name.is_empty());
        assert!(pattern.recognition_confidence > 0.6);
    }
}

#[tokio::test]
async fn test_concurrent_validation_safety() {
    // Test: Concurrent validations should be thread-safe
    let tracker = std::sync::Arc::new(tokio::sync::Mutex::new(
        CqgsComplianceTracker::new().await.unwrap(),
    ));
    let mut handles = vec![];

    // Spawn 10 concurrent validation tasks
    for task_id in 0..10 {
        let tracker_clone = tracker.clone();
        let handle = tokio::spawn(async move {
            for validation_id in 0..5 {
                let data = OrganismPerformanceData {
                    organism_id: uuid::Uuid::new_v4(),
                    organism_type: format!("concurrent_{}_{}", task_id, validation_id),
                    timestamp: chrono::Utc::now(),
                    latency_ns: 45_000 + (task_id as u64 * 1000),
                    throughput: 100.0 + (validation_id as f64 * 10.0),
                    success_rate: 0.9,
                    resource_usage: crate::organisms::ResourceMetrics::default(),
                    profit: 200.0,
                    trades_executed: 15,
                };

                let mut tracker_guard = tracker_clone.lock().await;
                let result = tracker_guard.validate_performance(&data).await;
                drop(tracker_guard);

                TestUtils::assert_cqgs_compliant(&result);
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all validations were processed correctly
    let tracker_guard = tracker.lock().await;
    let final_status = tracker_guard.get_compliance_status().await.unwrap();

    // Should have processed all validations
    assert!(final_status.total_validations_performed >= 50); // 10 tasks * 5 validations
    assert!(final_status.validation_success_rate > 0.95); // High success rate
}
