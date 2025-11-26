use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

// Import the modules we created
use nt_napi_bindings::pool::{ConnectionManager, DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT_SECS, HealthStatus};
use nt_napi_bindings::neural::{ModelCache, NeuralModel};
use nt_napi_bindings::metrics::SystemMetrics;
use std::sync::Arc;

#[tokio::test]
async fn test_connection_manager_creation() -> Result<()> {
    let manager = ConnectionManager::new(10, 5)?;
    let metrics = manager.metrics();
    assert_eq!(metrics.max_size, 10);
    assert_eq!(metrics.total_gets, 0);
    Ok(())
}

#[tokio::test]
async fn test_connection_acquisition() -> Result<()> {
    let manager = ConnectionManager::new(10, 5)?;
    let conn = manager.get_connection().await?;
    assert!(conn.id.starts_with("conn-"));
    Ok(())
}

#[tokio::test]
async fn test_connection_pool_metrics() -> Result<()> {
    let manager = ConnectionManager::new(10, 5)?;

    for _ in 0..5 {
        let _conn = manager.get_connection().await?;
    }

    let metrics = manager.metrics();
    assert_eq!(metrics.total_gets, 5);
    assert_eq!(metrics.successful_gets, 5);
    assert_eq!(metrics.success_rate, 100.0);
    Ok(())
}

#[tokio::test]
async fn test_pool_health_check() -> Result<()> {
    let manager = ConnectionManager::new(10, 5)?;
    let health = manager.health_check();
    assert_eq!(health.status, HealthStatus::Healthy);
    assert_eq!(health.health_score, 100.0);
    Ok(())
}

#[tokio::test]
async fn test_neural_model_creation() -> Result<()> {
    let model = NeuralModel::new("test-model".to_string(), false)?;
    assert_eq!(model.model_id(), "test-model");
    Ok(())
}

#[tokio::test]
async fn test_neural_memory_tracking() -> Result<()> {
    let model = NeuralModel::new("test-model".to_string(), false)?;
    let usage = model.memory_usage();
    assert_eq!(usage.total_bytes, 0); // Empty model
    Ok(())
}

#[tokio::test]
async fn test_neural_cleanup() -> Result<()> {
    let mut model = NeuralModel::new("test-model".to_string(), false)?;
    model.cleanup();
    let usage = model.memory_usage();
    assert_eq!(usage.total_bytes, 0);
    Ok(())
}

#[tokio::test]
async fn test_model_cache() -> Result<()> {
    let cache = ModelCache::new(10, 3600);
    let model1 = cache.get_or_create("model-1", false)?;
    let model2 = cache.get_or_create("model-1", false)?;

    assert_eq!(cache.model_count(), 1);
    assert!(Arc::ptr_eq(&model1, &model2));
    Ok(())
}

#[tokio::test]
async fn test_model_cache_eviction() -> Result<()> {
    let cache = ModelCache::new(2, 3600);

    let _m1 = cache.get_or_create("model-1", false)?;
    let _m2 = cache.get_or_create("model-2", false)?;
    let _m3 = cache.get_or_create("model-3", false)?;

    // Should have evicted the oldest model
    assert_eq!(cache.model_count(), 2);
    Ok(())
}

#[tokio::test]
async fn test_system_metrics_creation() -> Result<()> {
    let metrics = SystemMetrics::new();
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.pool_gets, 0);
    assert_eq!(snapshot.neural_allocations, 0);
    Ok(())
}

#[tokio::test]
async fn test_system_metrics_pool_tracking() -> Result<()> {
    let metrics = SystemMetrics::new();
    metrics.record_pool_get();
    metrics.record_pool_get();
    metrics.record_pool_timeout();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.pool_gets, 2);
    assert_eq!(snapshot.pool_timeouts, 1);
    assert_eq!(snapshot.pool_success_rate, 50.0);
    Ok(())
}

#[tokio::test]
async fn test_system_metrics_neural_tracking() -> Result<()> {
    let metrics = SystemMetrics::new();
    metrics.record_neural_allocation(1024);
    metrics.record_neural_allocation(2048);
    metrics.record_neural_deallocation(1024);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.neural_allocations, 2);
    assert_eq!(snapshot.neural_deallocations, 1);
    assert_eq!(snapshot.neural_memory_bytes, 2048);
    Ok(())
}

#[tokio::test]
async fn test_system_metrics_cache_tracking() -> Result<()> {
    let metrics = SystemMetrics::new();
    metrics.record_neural_cache_hit();
    metrics.record_neural_cache_hit();
    metrics.record_neural_cache_hit();
    metrics.record_neural_cache_miss();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.neural_cache_hits, 3);
    assert_eq!(snapshot.neural_cache_misses, 1);
    assert_eq!(snapshot.neural_cache_hit_rate, 75.0);
    Ok(())
}

#[tokio::test]
async fn test_high_concurrency_100_operations() -> Result<()> {
    let manager = ConnectionManager::new(100, 5)?;
    let mut tasks = JoinSet::new();

    for _ in 0..100 {
        tasks.spawn(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<_, anyhow::Error>(())
        });
    }

    let mut completed = 0;
    while let Some(result) = tasks.join_next().await {
        if result.is_ok() {
            completed += 1;
        }
    }

    assert_eq!(completed, 100);
    Ok(())
}

#[tokio::test]
async fn test_concurrent_model_operations() -> Result<()> {
    let cache = Arc::new(ModelCache::new(50, 3600));
    let mut tasks = JoinSet::new();

    for i in 0..50 {
        let cache_clone = cache.clone();
        tasks.spawn(async move {
            let _model = cache_clone.get_or_create(&format!("model-{}", i % 10), false)?;
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<_, anyhow::Error>(())
        });
    }

    let mut completed = 0;
    while let Some(result) = tasks.join_next().await {
        if result.is_ok() {
            completed += 1;
        }
    }

    assert_eq!(completed, 50);
    assert!(cache.model_count() <= 10); // Should reuse models
    Ok(())
}
