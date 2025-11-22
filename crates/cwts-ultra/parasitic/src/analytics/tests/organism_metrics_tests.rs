//! Organism Metrics Tests
//!
//! TDD tests for OrganismMetrics module tracking all 10 organism types

use super::TestUtils;
use crate::analytics::metrics::OrganismMetrics;
use crate::analytics::{OrganismAnalyticsSummary, OrganismPerformanceData};
use crate::organisms::{OrganismFactory, ParasiticOrganism};
use std::collections::HashMap;

#[tokio::test]
async fn test_organism_metrics_creation() {
    // Test: OrganismMetrics should be created and track all 10 organism types
    let metrics = OrganismMetrics::new().await.unwrap();

    assert!(metrics.is_initialized());
    assert_eq!(metrics.get_tracked_organism_types().len(), 14); // All organism types
    assert!(metrics.get_active_organisms().is_empty());
}

#[tokio::test]
async fn test_all_organism_types_tracking() {
    // Test: Should track metrics for all available organism types
    let mut metrics = OrganismMetrics::new().await.unwrap();
    let organism_types = crate::organisms::OrganismFactory::available_types();

    for organism_type in &organism_types {
        let mut data = TestUtils::generate_test_organism_data(1)[0].clone();
        data.organism_type = organism_type.to_string();

        let result = metrics.update_metrics(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let active_organisms = metrics.get_active_organisms();
    assert_eq!(active_organisms.len(), organism_types.len());

    // Verify each organism type is tracked
    for organism_type in organism_types {
        assert!(active_organisms
            .iter()
            .any(|o| o.organism_type == organism_type));
    }
}

#[tokio::test]
async fn test_real_organism_integration() {
    // Test: Should integrate with real organism instances (zero-mock enforcement)
    let mut metrics = OrganismMetrics::new().await.unwrap();

    // Create real organism instances of each type
    let organism_types = ["cuckoo", "wasp", "virus", "bacteria", "cordyceps"];
    let mut real_organisms: Vec<Box<dyn ParasiticOrganism + Send + Sync>> = Vec::new();

    for organism_type in &organism_types {
        let organism = OrganismFactory::create_organism(organism_type).unwrap();
        real_organisms.push(organism);
    }

    // Record metrics for each real organism
    for organism in &real_organisms {
        let performance_data = OrganismPerformanceData {
            organism_id: organism.id(),
            organism_type: organism.organism_type().to_string(),
            timestamp: chrono::Utc::now(),
            latency_ns: 45_000, // Sub-millisecond
            throughput: 150.0,
            success_rate: 0.95,
            resource_usage: organism.resource_consumption(),
            profit: 250.0,
            trades_executed: 5,
        };

        let result = metrics.update_metrics(&performance_data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    // Verify all real organisms are tracked
    let summaries = metrics.get_all_organism_summaries().await.unwrap();
    assert_eq!(summaries.len(), organism_types.len());

    // Verify real resource consumption data
    for summary in &summaries {
        assert!(summary.resource_efficiency > 0.0);
        assert!(summary.total_trades > 0);
        assert!(summary.performance_score >= 0.0);
    }
}

#[tokio::test]
async fn test_performance_score_calculation() {
    // Test: Performance score should be calculated accurately for organisms
    let mut metrics = OrganismMetrics::new().await.unwrap();

    // Create organism with known performance characteristics
    let high_performance_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "cuckoo".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 25_000, // Excellent latency
        throughput: 200.0,  // High throughput
        success_rate: 0.98, // Excellent success rate
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 5.0,               // Low CPU usage
            memory_mb: 10.0,              // Low memory usage
            network_bandwidth_kbps: 50.0, // Moderate bandwidth
            api_calls_per_second: 20.0,   // High API calls
            latency_overhead_ns: 1000,    // Low overhead
        },
        profit: 500.0,       // High profit
        trades_executed: 25, // Many trades
    };

    let result = metrics.update_metrics(&high_performance_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let summary = metrics
        .get_organism_summary(high_performance_data.organism_id)
        .await
        .unwrap();
    assert!(summary.performance_score > 0.8); // Should be high performance
    assert_eq!(summary.total_trades, 25);
    assert_eq!(summary.total_profit, 500.0);
}

#[tokio::test]
async fn test_organism_ranking() {
    // Test: Organisms should be ranked by performance
    let mut metrics = OrganismMetrics::new().await.unwrap();

    // Create organisms with different performance levels
    let organism_performances = vec![
        (0.95, 300.0, 50), // High performer
        (0.85, 200.0, 30), // Medium performer
        (0.70, 100.0, 15), // Low performer
    ];

    let mut organism_ids = Vec::new();

    for (i, (success_rate, profit, trades)) in organism_performances.iter().enumerate() {
        let organism_id = uuid::Uuid::new_v4();
        organism_ids.push(organism_id);

        let data = OrganismPerformanceData {
            organism_id,
            organism_type: format!("test_organism_{}", i),
            timestamp: chrono::Utc::now(),
            latency_ns: 40_000,
            throughput: *profit / 10.0, // Derive throughput from profit
            success_rate: *success_rate,
            resource_usage: crate::organisms::ResourceMetrics::default(),
            profit: *profit,
            trades_executed: *trades,
        };

        let result = metrics.update_metrics(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let rankings = metrics.get_organism_rankings().await.unwrap();
    assert_eq!(rankings.len(), 3);

    // Should be sorted by performance score (descending)
    assert!(rankings[0].performance_score >= rankings[1].performance_score);
    assert!(rankings[1].performance_score >= rankings[2].performance_score);

    // High performer should be ranked first
    assert!(rankings[0].total_profit >= 300.0);
}

#[tokio::test]
async fn test_resource_efficiency_tracking() {
    // Test: Resource efficiency should be calculated accurately
    let mut metrics = OrganismMetrics::new().await.unwrap();

    // Create efficient organism (low resource usage, high profit)
    let efficient_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "efficient_organism".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 30_000,
        throughput: 100.0,
        success_rate: 0.92,
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 2.0, // Very low CPU
            memory_mb: 5.0, // Very low memory
            network_bandwidth_kbps: 20.0,
            api_calls_per_second: 10.0,
            latency_overhead_ns: 500,
        },
        profit: 400.0, // High profit
        trades_executed: 20,
    };

    let result = metrics.update_metrics(&efficient_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let summary = metrics
        .get_organism_summary(efficient_data.organism_id)
        .await
        .unwrap();
    assert!(summary.resource_efficiency > 0.9); // Should be very efficient

    // Create inefficient organism (high resource usage, low profit)
    let inefficient_data = OrganismPerformanceData {
        organism_id: uuid::Uuid::new_v4(),
        organism_type: "inefficient_organism".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 80_000,
        throughput: 50.0,
        success_rate: 0.75,
        resource_usage: crate::organisms::ResourceMetrics {
            cpu_usage: 50.0,  // High CPU usage
            memory_mb: 200.0, // High memory usage
            network_bandwidth_kbps: 500.0,
            api_calls_per_second: 5.0,
            latency_overhead_ns: 5000,
        },
        profit: 50.0, // Low profit
        trades_executed: 5,
    };

    let result = metrics.update_metrics(&inefficient_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    let inefficient_summary = metrics
        .get_organism_summary(inefficient_data.organism_id)
        .await
        .unwrap();
    assert!(inefficient_summary.resource_efficiency < 0.5); // Should be inefficient
    assert!(summary.resource_efficiency > inefficient_summary.resource_efficiency);
}

#[tokio::test]
async fn test_time_series_tracking() {
    // Test: Should track performance over time for trend analysis
    let mut metrics = OrganismMetrics::new().await.unwrap();
    let organism_id = uuid::Uuid::new_v4();

    // Simulate performance degradation over time
    let performance_values = vec![0.95, 0.90, 0.85, 0.80, 0.75];
    let mut timestamps = Vec::new();

    for (i, performance) in performance_values.iter().enumerate() {
        let timestamp = chrono::Utc::now() - chrono::Duration::seconds((5 - i) as i64);
        timestamps.push(timestamp);

        let data = OrganismPerformanceData {
            organism_id,
            organism_type: "degrading_organism".to_string(),
            timestamp,
            latency_ns: 50_000 + (i as u64 * 10_000), // Increasing latency
            throughput: 100.0 - (i as f64 * 10.0),    // Decreasing throughput
            success_rate: *performance,
            resource_usage: crate::organisms::ResourceMetrics::default(),
            profit: 100.0 - (i as f64 * 15.0), // Decreasing profit
            trades_executed: 10 - i as u64,    // Decreasing trades
        };

        let result = metrics.update_metrics(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let time_series = metrics
        .get_organism_time_series(organism_id, chrono::Duration::minutes(10))
        .await
        .unwrap();
    assert_eq!(time_series.len(), 5);

    // Verify timestamps are ordered
    for i in 1..time_series.len() {
        assert!(time_series[i].timestamp >= time_series[i - 1].timestamp);
    }

    // Verify performance trend (should show degradation)
    let trend = metrics
        .calculate_performance_trend(organism_id)
        .await
        .unwrap();
    assert!(trend < 0.0); // Negative trend indicates degradation
}

#[tokio::test]
async fn test_organism_lifecycle_tracking() {
    // Test: Should track complete organism lifecycle
    let mut metrics = OrganismMetrics::new().await.unwrap();
    let organism_id = uuid::Uuid::new_v4();

    // Initial performance data
    let initial_data = OrganismPerformanceData {
        organism_id,
        organism_type: "lifecycle_organism".to_string(),
        timestamp: chrono::Utc::now(),
        latency_ns: 60_000,
        throughput: 80.0,
        success_rate: 0.88,
        resource_usage: crate::organisms::ResourceMetrics::default(),
        profit: 150.0,
        trades_executed: 12,
    };

    let result = metrics.update_metrics(&initial_data).await;
    TestUtils::assert_cqgs_compliant(&result);

    // Verify organism is active
    assert!(metrics.is_organism_active(organism_id).await);

    let summary = metrics.get_organism_summary(organism_id).await.unwrap();
    assert_eq!(summary.organism_id, organism_id);
    assert!(summary.last_active > chrono::Utc::now() - chrono::Duration::minutes(1));

    // Simulate organism going inactive
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check lifecycle status
    let lifecycle_info = metrics
        .get_organism_lifecycle_info(organism_id)
        .await
        .unwrap();
    assert!(lifecycle_info.creation_time <= lifecycle_info.last_update);
    assert!(lifecycle_info.total_runtime.num_seconds() >= 0);
}

#[tokio::test]
async fn test_aggregate_statistics() {
    // Test: Should provide accurate aggregate statistics
    let mut metrics = OrganismMetrics::new().await.unwrap();

    // Create multiple organisms with different performance profiles
    let test_cases = vec![
        ("cuckoo", 0.95, 300.0, 25),
        ("wasp", 0.90, 250.0, 20),
        ("virus", 0.85, 200.0, 15),
        ("bacteria", 0.92, 275.0, 22),
    ];

    for (organism_type, success_rate, profit, trades) in test_cases {
        let data = OrganismPerformanceData {
            organism_id: uuid::Uuid::new_v4(),
            organism_type: organism_type.to_string(),
            timestamp: chrono::Utc::now(),
            latency_ns: 45_000,
            throughput: profit / 10.0,
            success_rate,
            resource_usage: crate::organisms::ResourceMetrics::default(),
            profit,
            trades_executed: trades,
        };

        let result = metrics.update_metrics(&data).await;
        TestUtils::assert_cqgs_compliant(&result);
    }

    let aggregate_stats = metrics.get_aggregate_statistics().await.unwrap();
    assert_eq!(aggregate_stats.total_organisms, 4);
    assert_eq!(aggregate_stats.total_trades, 82); // 25+20+15+22
    assert_eq!(aggregate_stats.total_profit, 1025.0); // 300+250+200+275

    let expected_avg_success_rate = (0.95 + 0.90 + 0.85 + 0.92) / 4.0;
    assert!((aggregate_stats.average_success_rate - expected_avg_success_rate).abs() < 0.01);
}

#[tokio::test]
async fn test_sub_millisecond_performance() {
    // Test: All organism metrics operations should complete in sub-millisecond time
    let mut metrics = OrganismMetrics::new().await.unwrap();
    let test_data = TestUtils::generate_test_organism_data(1)[0].clone();

    // Test metric update performance
    TestUtils::assert_sub_millisecond(|| {
        // Using async in sync context for performance test
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(metrics.update_metrics(&test_data));
        TestUtils::assert_cqgs_compliant(&result);
    });

    // Test summary retrieval performance
    TestUtils::assert_sub_millisecond(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(metrics.get_organism_summary(test_data.organism_id));
        assert!(result.is_ok());
    });
}

#[tokio::test]
async fn test_concurrent_organism_tracking() {
    // Test: Should handle concurrent organism updates safely
    let metrics = std::sync::Arc::new(tokio::sync::Mutex::new(
        OrganismMetrics::new().await.unwrap(),
    ));
    let mut handles = vec![];

    // Spawn 10 concurrent tasks, each tracking a different organism
    for task_id in 0..10 {
        let metrics_clone = metrics.clone();
        let handle = tokio::spawn(async move {
            let organism_id = uuid::Uuid::new_v4();

            for update_count in 0..5 {
                let data = OrganismPerformanceData {
                    organism_id,
                    organism_type: format!("concurrent_organism_{}", task_id),
                    timestamp: chrono::Utc::now(),
                    latency_ns: 40_000 + (update_count * 1000),
                    throughput: 100.0 + (update_count as f64 * 5.0),
                    success_rate: 0.9,
                    resource_usage: crate::organisms::ResourceMetrics::default(),
                    profit: 100.0 + (update_count as f64 * 20.0),
                    trades_executed: update_count + 1,
                };

                let mut metrics_guard = metrics_clone.lock().await;
                let result = metrics_guard.update_metrics(&data).await;
                drop(metrics_guard);

                TestUtils::assert_cqgs_compliant(&result);
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }

            organism_id
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete and collect organism IDs
    let mut organism_ids = Vec::new();
    for handle in handles {
        let organism_id = handle.await.unwrap();
        organism_ids.push(organism_id);
    }

    // Verify all organisms were tracked correctly
    let metrics_guard = metrics.lock().await;
    let active_organisms = metrics_guard.get_active_organisms();
    assert_eq!(active_organisms.len(), 10);

    // Each organism should have 5 updates recorded
    for organism_id in organism_ids {
        let summary = metrics_guard
            .get_organism_summary(organism_id)
            .await
            .unwrap();
        assert_eq!(summary.total_trades, 5);
    }
}
