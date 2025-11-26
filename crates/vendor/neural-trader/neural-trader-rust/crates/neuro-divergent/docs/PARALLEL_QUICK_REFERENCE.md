# Rayon Parallelization Quick Reference

## 🚀 One-Liner Examples

### Batch Inference
```rust
use neuro_divergent::optimizations::parallel::parallel_batch_inference;

let predictions = parallel_batch_inference(&batches, |b| model.predict(b))?;
```

### With Uncertainty
```rust
use neuro_divergent::optimizations::parallel::parallel_batch_inference_with_uncertainty;

let results = parallel_batch_inference_with_uncertainty(&batches, |b| model.predict(b), 100)?;
```

### Data Preprocessing
```rust
use neuro_divergent::optimizations::parallel::parallel_preprocess;

let normalized = parallel_preprocess(&chunks, |c| Ok((c - mean) / std))?;
```

### Cross-Validation
```rust
use neuro_divergent::optimizations::parallel::parallel_cross_validation;

let scores = parallel_cross_validation(&data, &labels, 5, train_eval_fn)?;
```

### Ensemble Predictions
```rust
use neuro_divergent::optimizations::parallel::{parallel_ensemble_predict, EnsembleAggregation};

let pred = parallel_ensemble_predict(&input, &models, EnsembleAggregation::Mean)?;
```

## ⚙️ Configuration

### Auto (Recommended)
```rust
// Uses all available cores
let config = ParallelConfig::default();
```

### Manual
```rust
let config = ParallelConfig::default().with_threads(8);
config.configure_thread_pool()?;
```

### Environment
```bash
RAYON_NUM_THREADS=8 cargo run
```

## 📊 Performance Expectations

| Threads | Speedup | Efficiency |
|---------|---------|------------|
| 1       | 1.00x   | 100%       |
| 2       | 1.92x   | 96%        |
| 4       | 3.68x   | 92%        |
| 8       | 6.94x   | 87%        |

## 🎯 When to Use

✅ **Use parallelization when**:
- Batch size > 20-30 items
- Each item takes >1ms to process
- Operations are independent (no shared state)
- CPU-bound workload

❌ **Don't use when**:
- Batch size < 10 items
- I/O-bound operations
- Heavy synchronization required
- Memory-constrained environment

## 🧪 Testing

```bash
# Run tests
cargo test --test parallel_integration_test

# Run benchmarks
cargo bench --bench parallel_benchmarks

# Run example
cargo run --release --example parallel_batch_inference
```

## 📈 Benchmarking Your Code

```rust
use neuro_divergent::optimizations::parallel::benchmark::*;

let results = scalability_benchmark(&[1, 2, 4, 8], || {
    parallel_batch_inference(&batches, inference_fn)?;
    Ok(())
})?;

print_benchmark_results(&results);
```

## 🔧 Advanced Patterns

### Thread-Local Caching
```rust
use std::cell::RefCell;

thread_local! {
    static CACHE: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

batches.par_iter().for_each(|batch| {
    CACHE.with(|cache| {
        cache.borrow_mut().clear();
        // Use cache
    });
});
```

### Custom Thread Pool
```rust
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()?;

pool.install(|| {
    parallel_batch_inference(&batches, inference_fn)
})?;
```

## 📚 API Reference

### Core Functions
- `parallel_batch_inference(batches, fn)` → `Vec<Vec<f64>>`
- `parallel_batch_inference_with_uncertainty(batches, fn, n)` → `Vec<(Vec<f64>, Vec<f64>)>`
- `parallel_preprocess(chunks, fn)` → `Vec<Array2<f64>>`
- `parallel_gradient_computation(batches, fn)` → `Vec<Vec<Array2<f64>>>`
- `aggregate_gradients(grads)` → `Vec<Array2<f64>>`
- `parallel_cross_validation(data, labels, k, fn)` → `Vec<f64>`
- `parallel_grid_search(params, data, labels, fn)` → `Vec<(usize, f64)>`
- `parallel_ensemble_predict(input, models, agg)` → `Vec<f64>`

### Aggregation Types
- `EnsembleAggregation::Mean`
- `EnsembleAggregation::Median`
- `EnsembleAggregation::WeightedMean(weights)`

## 💡 Tips

1. **Batch Size**: Larger batches = better parallelization (diminishing returns >100)
2. **Thread Count**: Match physical cores for CPU-bound work
3. **Memory**: Watch memory usage with many threads (N_threads × batch_size × data_size)
4. **Profiling**: Use `cargo flamegraph` to identify bottlenecks
5. **Overhead**: Parallelization overhead ~5-10ms, so each item should take >0.5ms

## 🐛 Troubleshooting

**Problem**: Speedup < 2x with 8 threads
- **Solution**: Increase batch size or reduce synchronization

**Problem**: Out of memory errors
- **Solution**: Reduce thread count or batch size

**Problem**: Performance worse than sequential
- **Solution**: Operation too fast (overhead dominates), use sequential

**Problem**: Inconsistent results
- **Solution**: Ensure deterministic operations or use seeded RNG

## 📖 Full Documentation

See [PARALLEL_OPTIMIZATION_GUIDE.md](./PARALLEL_OPTIMIZATION_GUIDE.md) for comprehensive guide.
