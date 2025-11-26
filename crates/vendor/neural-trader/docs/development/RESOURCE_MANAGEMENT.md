# Resource Management Guide

## Overview

This document describes the resource management patterns and best practices implemented in Neural Trader to prevent connection pool exhaustion and neural memory leaks.

## Connection Pool Management

### Architecture

The connection pool uses `deadpool` for efficient, thread-safe connection pooling with the following features:

- **Configurable pool size**: Default 2000 connections (up to 10,000 max)
- **Automatic recycling**: Connections older than 1 hour are recycled
- **Timeout handling**: 5-second default timeout with graceful degradation
- **Health monitoring**: Real-time health checks and metrics

### Configuration

```rust
use nt_napi_bindings::pool::{ConnectionManager, DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT_SECS};

// Create pool with defaults
let pool = ConnectionManager::new(DEFAULT_POOL_SIZE, DEFAULT_TIMEOUT_SECS)?;

// Or custom configuration
let pool = ConnectionManager::new(5000, 10)?; // 5000 connections, 10s timeout
```

### Usage Pattern

```rust
// Acquire connection
let conn = pool.get_connection().await?;

// Connection is automatically returned to pool when dropped
// Use connection...
drop(conn);
```

### Metrics

```rust
// Get pool metrics
let metrics = pool.metrics();
println!("Pool utilization: {:.1}%",
    (metrics.current_size - metrics.available) as f64 / metrics.max_size as f64 * 100.0);
println!("Success rate: {:.1}%", metrics.success_rate);

// Health check
let health = pool.health_check();
match health.status {
    HealthStatus::Healthy => println!("Pool is healthy"),
    HealthStatus::Degraded => println!("Pool performance degraded"),
    HealthStatus::Unhealthy => println!("Pool needs attention"),
}
```

## Neural Memory Management

### Architecture

Neural models implement comprehensive memory management:

- **Explicit Drop trait**: Guaranteed cleanup of GPU and CPU resources
- **Automatic tensor cache**: LRU eviction with configurable limits
- **Periodic cleanup**: Background task removes stale models
- **Memory tracking**: Real-time monitoring of memory usage

### Model Lifecycle

```rust
use nt_napi_bindings::neural::{NeuralModel, ModelCache};

// Create model cache
let cache = ModelCache::new(100, 3600); // 100 models max, 1 hour TTL

// Get or create model (automatically manages lifecycle)
let model = cache.get_or_create("my-model", use_gpu)?;

// Model is automatically cleaned up when:
// 1. Cache evicts it (LRU)
// 2. Age exceeds TTL
// 3. Process exits (Drop trait)
```

### Memory Tracking

```rust
// Check model memory usage
let usage = model.lock().memory_usage();
println!("Model memory:");
println!("  Data: {:.2} MB", usage.model_data_bytes as f64 / (1024.0 * 1024.0));
println!("  Cache: {:.2} MB", usage.cache_bytes as f64 / (1024.0 * 1024.0));
println!("  GPU: {:.2} MB", usage.gpu_bytes as f64 / (1024.0 * 1024.0));
println!("  Total: {:.2} MB", usage.total_mb());

// Cache-wide statistics
println!("Total models: {}", cache.model_count());
println!("Total memory: {:.2} MB",
    cache.total_memory_usage() as f64 / (1024.0 * 1024.0));
```

### Cleanup

```rust
// Explicit cleanup (optional, automatic on drop)
model.lock().cleanup();

// Clear entire cache
cache.clear_all();

// Start periodic cleanup task
let cleanup_task = tokio::spawn(cleanup_neural_resources(Arc::new(cache)));
```

## System Metrics

### Global Metrics Tracking

```rust
use nt_napi_bindings::metrics::SystemMetrics;

let metrics = SystemMetrics::new();

// Record events
metrics.record_pool_get();
metrics.record_neural_allocation(1024 * 1024); // 1MB

// Get snapshot
let snapshot = metrics.snapshot();
println!("Pool success rate: {:.1}%", snapshot.pool_success_rate);
println!("Neural memory: {:.2} MB", snapshot.neural_memory_mb);
println!("Cache hit rate: {:.1}%", snapshot.neural_cache_hit_rate);

// Get rates
let rates = metrics.rates();
println!("Pool requests/sec: {:.2}", rates.pool_gets_per_sec);
println!("Neural allocations/sec: {:.2}", rates.neural_allocations_per_sec);
```

### Periodic Reporting

```rust
use nt_napi_bindings::metrics::metrics_reporter;

// Start metrics reporter (reports every 60 seconds)
let reporter_task = tokio::spawn(metrics_reporter(
    Arc::new(metrics),
    60 // seconds
));
```

## Best Practices

### Connection Pool

1. **Size appropriately**: Start with `DEFAULT_POOL_SIZE` (2000) and adjust based on load
2. **Monitor metrics**: Check `pool.metrics()` regularly to detect issues
3. **Handle timeouts**: Always handle `ConnectionPoolExhausted` errors gracefully
4. **Use health checks**: Run `pool.health_check()` before critical operations

### Neural Memory

1. **Use model cache**: Always use `ModelCache` instead of creating models directly
2. **Set appropriate TTL**: Balance memory usage vs model recreation cost
3. **Monitor memory**: Check `model.memory_usage()` for large models
4. **Explicit cleanup**: Call `cleanup()` after heavy operations
5. **Enable periodic cleanup**: Run `cleanup_neural_resources()` in production

### General

1. **Enable jemalloc**: Add `jemalloc` feature for better memory management
2. **Monitor system metrics**: Use `SystemMetrics` for visibility
3. **Test under load**: Use provided benchmark suite to validate configuration
4. **Set resource limits**: Configure OS limits (ulimit) appropriately

## Performance Tuning

### Connection Pool Tuning

```rust
// High throughput (many short operations)
let pool = ConnectionManager::new(5000, 2)?; // Large pool, short timeout

// Low latency (fewer long operations)
let pool = ConnectionManager::new(500, 30)?; // Smaller pool, longer timeout

// Balanced
let pool = ConnectionManager::new(2000, 5)?; // Default configuration
```

### Neural Memory Tuning

```rust
// Memory-constrained environment
let cache = ModelCache::new(10, 600); // 10 models, 10 min TTL

// High-performance environment
let cache = ModelCache::new(1000, 7200); // 1000 models, 2 hour TTL

// Balanced
let cache = ModelCache::new(100, 3600); // Default configuration
```

## Testing

### Load Testing

Run the comprehensive test suite:

```bash
# Connection pool tests (5000+ concurrent ops)
cargo test --test connection_pool_load_test -- --nocapture

# Neural memory leak tests
cargo test --test neural_memory_leak_test -- --nocapture

# Performance benchmarks
cargo test --test benchmarks -- --nocapture
```

### Expected Results

- **Connection pool**: >95% success rate at 5000 concurrent operations
- **Neural memory**: No unbounded growth over 10,000+ allocations
- **Throughput**: >1000 ops/sec on modern hardware
- **Latency**: p99 < 100ms under normal load

## Monitoring in Production

### Key Metrics to Watch

1. **Pool Health**
   - Success rate should be >95%
   - Utilization should be <80% during normal operation
   - Timeout rate should be <1%

2. **Neural Memory**
   - Total memory growth should be bounded
   - Cache hit rate should be >70%
   - Active allocations should match model count

3. **System Performance**
   - CPU usage should be <70% under load
   - Memory usage should be stable over time
   - No OOM errors or crashes

### Alerts

Set up alerts for:
- Pool success rate < 90%
- Pool utilization > 90%
- Neural memory > configured limit
- Cache hit rate < 50%
- Any OOM conditions

## Troubleshooting

### Connection Pool Exhaustion

**Symptoms**: High timeout rate, `ConnectionPoolExhausted` errors

**Solutions**:
1. Increase pool size: `ConnectionManager::new(5000, 5)`
2. Reduce operation duration
3. Check for connection leaks (not returning connections)

### Memory Leaks

**Symptoms**: Unbounded memory growth, eventual OOM

**Solutions**:
1. Enable periodic cleanup: `cleanup_neural_resources()`
2. Reduce cache size: `ModelCache::new(50, 1800)`
3. Call explicit cleanup: `model.cleanup()`
4. Enable jemalloc: `cargo build --features jemalloc`

### Performance Degradation

**Symptoms**: Increasing latency, decreasing throughput

**Solutions**:
1. Check pool metrics: `pool.metrics()`
2. Monitor memory usage: `cache.total_memory_usage()`
3. Run benchmarks to identify bottlenecks
4. Consider scaling horizontally

## Advanced Configuration

### Custom Memory Allocator

Enable jemalloc for better memory management:

```toml
[dependencies]
nt-napi-bindings = { version = "2.0", features = ["jemalloc"] }
```

### Custom Cleanup Intervals

```rust
// More aggressive cleanup (every 1 minute)
let cleanup_task = tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        cache.cleanup_old_models();
    }
});
```

### Custom Pool Configuration

```rust
// Maximum pool size with aggressive recycling
let manager = ConnectionManagerInner::new();
let pool = Pool::builder(manager)
    .max_size(10000)
    .wait_timeout(Some(Duration::from_secs(10)))
    .recycle_timeout(Some(Duration::from_secs(10))) // More aggressive
    .build()?;
```

## References

- [deadpool documentation](https://docs.rs/deadpool)
- [parking_lot documentation](https://docs.rs/parking_lot)
- [jemalloc documentation](https://jemalloc.net/)

## Support

For issues or questions:
1. Check metrics and logs first
2. Run benchmark suite to validate
3. Review this guide for common solutions
4. File issue with metrics snapshot if problem persists
