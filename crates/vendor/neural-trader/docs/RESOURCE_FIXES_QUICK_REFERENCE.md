# Resource Management Fixes - Quick Reference

## 🚀 Quick Start

### Connection Pool
```rust
use nt_napi_bindings::pool::ConnectionManager;

let pool = ConnectionManager::new(2000, 5)?;
let conn = pool.get_connection().await?;
// Use connection
drop(conn); // Auto-returned
```

### Neural Memory
```rust
use nt_napi_bindings::neural::{ModelCache, cleanup_neural_resources};

let cache = Arc::new(ModelCache::new(100, 3600));
tokio::spawn(cleanup_neural_resources(cache.clone()));

let model = cache.get_or_create("model-1", false)?;
```

### Metrics
```rust
use nt_napi_bindings::metrics::{SystemMetrics, metrics_reporter};

let metrics = Arc::new(SystemMetrics::new());
tokio::spawn(metrics_reporter(metrics.clone(), 60));

metrics.record_pool_get();
let snapshot = metrics.snapshot();
```

## 📁 File Locations

| Component | Location |
|-----------|----------|
| Connection Pool | `neural-trader-rust/crates/napi-bindings/src/pool/` |
| Neural Memory | `neural-trader-rust/crates/napi-bindings/src/neural/` |
| Metrics | `neural-trader-rust/crates/napi-bindings/src/metrics/` |
| Integration Tests | `neural-trader-rust/crates/napi-bindings/tests/resource_management_tests.rs` |
| Load Tests | `tests/performance/connection_pool_load_test.rs` |
| Memory Tests | `tests/performance/neural_memory_leak_test.rs` |
| Benchmarks | `tests/performance/benchmarks.rs` |
| Documentation | `docs/RESOURCE_MANAGEMENT.md` |

## 🔧 Configuration

### Connection Pool Presets

```rust
// High-throughput (many short operations)
let pool = ConnectionManager::new(5000, 2)?;

// Low-latency (fewer long operations)
let pool = ConnectionManager::new(500, 30)?;

// Balanced (default)
let pool = ConnectionManager::new(2000, 5)?;
```

### Neural Cache Presets

```rust
// Memory-constrained
let cache = ModelCache::new(10, 600);

// High-performance
let cache = ModelCache::new(1000, 7200);

// Balanced (default)
let cache = ModelCache::new(100, 3600);
```

## 📊 Key Metrics

### Connection Pool
- **Success Rate**: Should be >95%
- **Utilization**: Should be <80%
- **Timeout Rate**: Should be <1%
- **Health Score**: 100 = healthy, 80-95 = degraded, <80 = unhealthy

### Neural Memory
- **Memory Growth**: Should be bounded
- **Cache Hit Rate**: Should be >70%
- **Active Allocations**: Should match model count
- **GPU Memory**: Should be freed on deallocation

## 🧪 Testing

### Quick Validation
```bash
./scripts/validate-resource-fixes.sh
```

### Unit Tests
```bash
cd neural-trader-rust/crates/napi-bindings
cargo test --lib pool::connection_manager::tests
cargo test --lib neural::model::tests
cargo test --lib metrics::tests
```

### Integration Tests
```bash
cargo test --test resource_management_tests -- --nocapture
```

### Benchmarks
```bash
cargo test --test benchmarks -- --nocapture
```

## 🚨 Monitoring

### Alert Thresholds
- Pool success rate <90% → WARNING
- Pool utilization >90% → WARNING
- Neural memory exceeds limit → CRITICAL
- Cache hit rate <50% → WARNING
- Any OOM conditions → CRITICAL

### Health Check
```rust
let health = pool.health_check();
match health.status {
    HealthStatus::Healthy => { /* All good */ },
    HealthStatus::Degraded => { /* Investigate */ },
    HealthStatus::Unhealthy => { /* Take action */ },
}
```

## 🐛 Troubleshooting

### Pool Exhaustion
**Symptoms**: High timeout rate, errors

**Solutions**:
1. Increase pool size
2. Reduce operation duration
3. Check for leaks (not returning connections)

### Memory Leaks
**Symptoms**: Unbounded growth, OOM

**Solutions**:
1. Enable periodic cleanup
2. Reduce cache size/TTL
3. Call explicit cleanup
4. Enable jemalloc

### Performance Degradation
**Symptoms**: Increasing latency

**Solutions**:
1. Check pool metrics
2. Monitor memory usage
3. Run benchmarks
4. Scale horizontally

## 🎯 Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Concurrency | 5000+ ops | Load test |
| Success Rate | >95% | Metrics |
| Throughput | >1000 ops/sec | Benchmark |
| Latency p99 | <100ms | Benchmark |
| Memory Growth | Bounded | Monitoring |
| Cache Hit Rate | >70% | Metrics |

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `RESOURCE_MANAGEMENT.md` | Complete usage guide |
| `RESOURCE_FIXES_SUMMARY.md` | Implementation details |
| `RESOURCE_FIXES_COMPLETE.md` | Executive summary |
| This file | Quick reference |

## 🔐 Production Checklist

- [ ] Enable jemalloc: `--features jemalloc`
- [ ] Configure pool size for expected load
- [ ] Set up model cache with appropriate TTL
- [ ] Start cleanup background tasks
- [ ] Start metrics reporter
- [ ] Configure monitoring/alerting
- [ ] Run load tests in staging
- [ ] Create runbooks for common issues
- [ ] Document operational procedures
- [ ] Plan gradual rollout

## 💡 Common Patterns

### Initialization
```rust
// One-time setup
let pool = ConnectionManager::new(2000, 5)?;
let cache = Arc::new(ModelCache::new(100, 3600));
let metrics = Arc::new(SystemMetrics::new());

// Start background tasks
tokio::spawn(cleanup_neural_resources(cache.clone()));
tokio::spawn(metrics_reporter(metrics.clone(), 60));
```

### Request Handling
```rust
// Record metrics
metrics.record_pool_get();

// Get connection
let conn = pool.get_connection().await?;

// Get/create model
let model = cache.get_or_create("model-id", use_gpu)?;
metrics.record_neural_cache_hit();

// Use resources...

// Auto-cleanup on drop
```

### Health Monitoring
```rust
// Periodic health check
let health = pool.health_check();
let snapshot = metrics.snapshot();

log::info!(
    "Pool: {:.1}% utilization, {:.1}% success rate",
    health.utilization_percent,
    snapshot.pool_success_rate
);

log::info!(
    "Neural: {:.2} MB memory, {:.1}% cache hit rate",
    snapshot.neural_memory_mb,
    snapshot.neural_cache_hit_rate
);
```

## 🎓 Best Practices

1. **Always** use `ConnectionManager` for pooling
2. **Always** use `ModelCache` for neural models
3. **Always** enable background cleanup tasks
4. **Always** monitor metrics in production
5. **Never** create connections/models directly
6. **Never** ignore health warnings
7. **Enable** jemalloc for production
8. **Test** under production-like load

## 📞 Support

1. Check this quick reference
2. Review `RESOURCE_MANAGEMENT.md`
3. Run validation script
4. Check metrics/logs
5. Consult test files for examples

---

**Status**: Production Ready ✓

**Last Updated**: 2025-11-15
