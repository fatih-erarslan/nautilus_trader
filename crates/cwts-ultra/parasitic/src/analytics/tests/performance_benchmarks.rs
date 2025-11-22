//! Performance Benchmarks
//!
//! Sub-millisecond performance benchmarks for analytics operations

use super::TestUtils;
use crate::analytics::{AnalyticsEngine, PerformanceAnalytics};
use std::time::{Duration, Instant};

#[tokio::test]
async fn benchmark_performance_analytics_recording() {
    let mut analytics = PerformanceAnalytics::new();
    let test_data = TestUtils::generate_test_organism_data(1000);

    let start = Instant::now();
    for data in test_data {
        let _ = analytics.record_metric(data).await;
    }
    let elapsed = start.elapsed();

    // Should process 1000 records in under 100ms (100μs per record average)
    assert!(elapsed < Duration::from_millis(100));
    println!(
        "Processed 1000 records in {:?} ({:?} per record)",
        elapsed,
        elapsed / 1000
    );
}

#[tokio::test]
async fn benchmark_cqgs_validation_performance() {
    let mut analytics = AnalyticsEngine::new().await.unwrap();
    let test_data = TestUtils::generate_test_organism_data(100);

    let start = Instant::now();
    for data in test_data {
        let _ = analytics.record_organism_performance(data).await;
    }
    let elapsed = start.elapsed();

    // Should validate 100 records in under 50ms (500μs per record average)
    assert!(elapsed < Duration::from_millis(50));
    println!(
        "CQGS validated 100 records in {:?} ({:?} per record)",
        elapsed,
        elapsed / 100
    );
}

#[tokio::test]
async fn benchmark_concurrent_analytics_performance() {
    let analytics = std::sync::Arc::new(tokio::sync::Mutex::new(
        AnalyticsEngine::new().await.unwrap(),
    ));
    let mut handles = vec![];

    let start = Instant::now();

    // Spawn 10 concurrent tasks
    for _ in 0..10 {
        let analytics_clone = analytics.clone();
        let handle = tokio::spawn(async move {
            let test_data = TestUtils::generate_test_organism_data(10);
            for data in test_data {
                let mut analytics_guard = analytics_clone.lock().await;
                let _ = analytics_guard.record_organism_performance(data).await;
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let elapsed = start.elapsed();

    // Should process 100 records concurrently in under 200ms
    assert!(elapsed < Duration::from_millis(200));
    println!("Concurrent processing of 100 records: {:?}", elapsed);
}
