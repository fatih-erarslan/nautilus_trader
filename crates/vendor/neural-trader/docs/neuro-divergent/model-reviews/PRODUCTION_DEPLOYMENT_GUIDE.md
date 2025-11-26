# Production Deployment Guide for Basic Models

## Executive Summary

This guide provides production-grade recommendations for deploying the neuro-divergent basic forecasting models in real-world trading systems.

**⚠️ CRITICAL STATUS WARNING**:
- **MLP**: 65% production-ready (missing backprop implementation)
- **DLinear**: ❌ NOT production-ready (naive baseline only)
- **NLinear**: ❌ NOT production-ready (naive baseline only)
- **MLPMultivariate**: ❌ NOT production-ready (naive baseline only)

**Recommendation**: Only MLP is close to production-ready, but requires critical fixes first.

---

## Model Selection Guide

### When to Use MLP

#### ✅ **IDEAL FOR**:

1. **Non-linear patterns**:
   - Stock prices with complex interactions
   - Cryptocurrency volatility
   - Commodity futures with multiple drivers

2. **Medium-sized datasets** (1,000+ samples):
   - Sufficient data to train neural network
   - Not so large that training becomes prohibitive

3. **Low-latency requirements** (~1-2ms inference):
   - High-frequency trading
   - Real-time portfolio rebalancing
   - Intraday strategy signals

4. **Feature-rich data**:
   - Multiple technical indicators
   - Market microstructure features
   - Alternative data sources

#### ❌ **NOT RECOMMENDED FOR**:

1. **Small datasets** (<500 samples):
   - High risk of overfitting
   - Use simpler models (linear regression, ARIMA)

2. **Ultra-low latency** (<100μs):
   - Neural network overhead too high
   - Use lookup tables or simple rules

3. **Explainability requirements**:
   - Black-box model
   - Regulatory compliance issues
   - Use linear models or decision trees instead

4. **Streaming/online learning**:
   - Current implementation requires full retraining
   - Consider incremental learning models

### When to Use DLinear (Once Properly Implemented)

**Note**: Current implementation is NOT usable. These recommendations are for the proper DLinear algorithm.

#### ✅ **IDEAL FOR**:

1. **Trended time series**:
   - Equity index forecasting
   - Commodity price trends
   - GDP/economic indicators

2. **Seasonal patterns**:
   - Retail sales forecasting
   - Energy demand prediction
   - Agricultural commodity prices

3. **Baseline comparisons**:
   - Benchmark for more complex models
   - Sanity check for forecasts

4. **Interpretability**:
   - Linear components are explainable
   - Easy to communicate to stakeholders

#### ❌ **NOT RECOMMENDED FOR**:

1. **High-frequency data**:
   - Too simple for tick-level patterns
   - Use more sophisticated models

2. **Regime changes**:
   - Assumes stable trend/seasonal decomposition
   - Use switching models instead

3. **Non-stationary data**:
   - Requires stationarity
   - Preprocess with differencing first

### When to Use NLinear (Once Properly Implemented)

Similar to DLinear, but better for:
- **Instance-normalized forecasts**
- **Data with varying scales**
- **Cross-sectional forecasting** (multiple series)

### When to Use MLPMultivariate (Once Properly Implemented)

#### ✅ **IDEAL FOR**:

1. **Multi-asset portfolios**:
   - Predict returns for 50+ stocks simultaneously
   - Cross-asset dependencies
   - Pairs trading signals

2. **Rich feature sets**:
   - OHLCV + indicators for each asset
   - Market regime indicators
   - Alternative data

3. **Multi-horizon forecasts**:
   - 1-day, 3-day, 7-day forecasts together
   - Coherent multi-step predictions

---

## Production Deployment Checklist

### Phase 1: Model Development (Weeks 1-4)

#### ✅ Implementation Completeness

- [ ] **MLP**:
  - [ ] Implement proper backpropagation
  - [ ] Add mini-batch training
  - [ ] Implement dropout
  - [ ] Add early stopping
  - [ ] Learning rate scheduling
  - [ ] Gradient clipping

- [ ] **DLinear**:
  - [ ] Implement trend-seasonal decomposition
  - [ ] Add linear layers
  - [ ] Proper training loop
  - [ ] Unit tests vs paper results

- [ ] **NLinear**:
  - [ ] Instance normalization
  - [ ] Linear projection
  - [ ] Denormalization
  - [ ] Validation tests

- [ ] **MLPMultivariate**:
  - [ ] Multi-output architecture
  - [ ] Multi-feature input handling
  - [ ] Cross-feature interactions

#### ✅ Testing

```rust
#[cfg(test)]
mod production_tests {
    use super::*;

    #[test]
    fn test_numerical_stability() {
        // Test with extreme values
        let data = vec![1e-10, 1e10, f64::MIN_POSITIVE, 1000.0];
        let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

        let mut model = MLP::new(ModelConfig::default());
        assert!(model.fit(&df).is_ok());

        let pred = model.predict(1).unwrap();
        assert!(pred[0].is_finite());
    }

    #[test]
    fn test_convergence_on_simple_pattern() {
        // Linear trend: should converge perfectly
        let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

        let mut model = MLP::new(
            ModelConfig::default()
                .with_input_size(10)
                .with_horizon(1)
                .with_hidden_size(64)
        );

        model.fit(&df).unwrap();

        // Test on held-out data
        let test_data: Vec<f64> = (200..210).map(|i| i as f64).collect();
        let test_df = TimeSeriesDataFrame::from_values(test_data.clone(), None).unwrap();

        let predictions = model.predict(10).unwrap();

        // Should predict linear trend accurately
        for (i, (&pred, &actual)) in predictions.iter().zip(&test_data).enumerate() {
            let error = (pred - actual).abs();
            assert!(error < 5.0, "Prediction {} off by {}", i, error);
        }
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let data = generate_random_data(500);
        let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

        let config = ModelConfig::default().with_seed(42);

        let mut model1 = MLP::new(config.clone());
        model1.fit(&df).unwrap();
        let pred1 = model1.predict(10).unwrap();

        let mut model2 = MLP::new(config.clone());
        model2.fit(&df).unwrap();
        let pred2 = model2.predict(10).unwrap();

        // Should be identical with same seed
        for (p1, p2) in pred1.iter().zip(&pred2) {
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn test_save_load_cycle() {
        let data = generate_random_data(500);
        let df = TimeSeriesDataFrame::from_values(data, None).unwrap();

        let mut model = MLP::new(ModelConfig::default());
        model.fit(&df).unwrap();

        let pred_before = model.predict(10).unwrap();

        // Save and load
        let path = std::path::Path::new("test_model.bin");
        model.save(path).unwrap();

        let loaded_model = MLP::load(path).unwrap();
        let pred_after = loaded_model.predict(10).unwrap();

        // Predictions should be identical
        assert_eq!(pred_before, pred_after);

        // Cleanup
        std::fs::remove_file(path).unwrap();
    }
}
```

### Phase 2: Hyperparameter Tuning (Weeks 5-6)

#### Recommended Ranges

**MLP**:
```rust
pub struct MLPHyperparams {
    pub input_size: RangeInclusive<usize>,    // 24..=168 (1 day to 1 week)
    pub horizon: RangeInclusive<usize>,        // 1..=48 (1hr to 2 days)
    pub hidden_size: Vec<usize>,               // [64, 128, 256, 512]
    pub num_layers: RangeInclusive<usize>,     // 2..=4
    pub dropout: Vec<f64>,                      // [0.0, 0.1, 0.2, 0.3]
    pub learning_rate: Vec<f64>,                // [1e-4, 1e-3, 1e-2]
    pub batch_size: Vec<usize>,                 // [16, 32, 64]
}

impl Default for MLPHyperparams {
    fn default() -> Self {
        Self {
            input_size: 24..=168,
            horizon: 1..=48,
            hidden_size: vec![64, 128, 256, 512],
            num_layers: 2..=4,
            dropout: vec![0.0, 0.1, 0.2, 0.3],
            learning_rate: vec![1e-4, 1e-3, 1e-2],
            batch_size: vec![16, 32, 64],
        }
    }
}
```

#### Grid Search Implementation

```rust
use itertools::iproduct;

pub struct GridSearch {
    param_grid: MLPHyperparams,
    cv_folds: usize,
    metric: &'static str,  // "mae", "mse", "mape"
}

impl GridSearch {
    pub fn fit(
        &self,
        data: &TimeSeriesDataFrame,
    ) -> Result<(ModelConfig, f64)> {
        let mut best_config = None;
        let mut best_score = f64::INFINITY;

        // Generate all combinations
        for (hidden, layers, dropout, lr, batch) in iproduct!(
            &self.param_grid.hidden_size,
            self.param_grid.num_layers.clone(),
            &self.param_grid.dropout,
            &self.param_grid.learning_rate,
            &self.param_grid.batch_size,
        ) {
            let config = ModelConfig::default()
                .with_hidden_size(*hidden)
                .with_num_layers(layers)
                .with_dropout(*dropout)
                .with_learning_rate(*lr)
                .with_batch_size(*batch);

            // Cross-validation
            let cv_score = self.cross_validate(&config, data)?;

            println!(
                "Config: h={}, l={}, d={:.2}, lr={:.4}, b={} => Score: {:.4}",
                hidden, layers, dropout, lr, batch, cv_score
            );

            if cv_score < best_score {
                best_score = cv_score;
                best_config = Some(config);
            }
        }

        Ok((best_config.unwrap(), best_score))
    }

    fn cross_validate(
        &self,
        config: &ModelConfig,
        data: &TimeSeriesDataFrame,
    ) -> Result<f64> {
        let fold_size = data.len() / self.cv_folds;
        let mut scores = Vec::new();

        for fold in 0..self.cv_folds {
            let val_start = fold * fold_size;
            let val_end = val_start + fold_size;

            // Train on all data except fold
            let train_data = data.exclude_range(val_start, val_end)?;
            let val_data = data.slice(val_start, val_end)?;

            let mut model = MLP::new(config.clone());
            model.fit(&train_data)?;

            let predictions = model.predict(fold_size)?;
            let actuals = val_data.get_feature(0)?;

            let score = match self.metric {
                "mae" => self.compute_mae(&predictions, &actuals),
                "mse" => self.compute_mse(&predictions, &actuals),
                "mape" => self.compute_mape(&predictions, &actuals),
                _ => panic!("Unknown metric: {}", self.metric),
            };

            scores.push(score);
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}
```

### Phase 3: Production Infrastructure (Weeks 7-8)

#### Model Serving Architecture

```rust
use tokio::sync::RwLock;
use std::sync::Arc;
use std::collections::HashMap;

/// Production model server
pub struct ModelServer {
    models: Arc<RwLock<HashMap<String, Box<dyn NeuralModel>>>>,
    metrics: Arc<RwLock<ServerMetrics>>,
}

#[derive(Default)]
pub struct ServerMetrics {
    pub total_predictions: u64,
    pub total_errors: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

impl ModelServer {
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ServerMetrics::default())),
        }
    }

    /// Load model into server
    pub async fn load_model(
        &self,
        name: String,
        path: &std::path::Path,
    ) -> Result<()> {
        let model = MLP::load(path)?;
        let mut models = self.models.write().await;
        models.insert(name, Box::new(model));
        Ok(())
    }

    /// Make prediction with error handling and metrics
    pub async fn predict(
        &self,
        model_name: &str,
        horizon: usize,
    ) -> Result<PredictionResponse> {
        let start = std::time::Instant::now();

        // Get model (read lock)
        let models = self.models.read().await;
        let model = models.get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;

        // Make prediction
        let prediction = model.predict(horizon)
            .map_err(|e| {
                // Log error
                tracing::error!("Prediction failed: {:?}", e);
                e
            })?;

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_predictions += 1;
        metrics.update_latency(latency);

        Ok(PredictionResponse {
            predictions: prediction,
            latency_ms: latency,
            model_name: model_name.to_string(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Health check endpoint
    pub async fn health(&self) -> HealthStatus {
        let models = self.models.read().await;
        let metrics = self.metrics.read().await;

        HealthStatus {
            healthy: true,
            num_models: models.len(),
            total_predictions: metrics.total_predictions,
            avg_latency_ms: metrics.avg_latency_ms,
        }
    }

    /// Hot-swap model (zero-downtime update)
    pub async fn update_model(
        &self,
        name: String,
        path: &std::path::Path,
    ) -> Result<()> {
        // Load new model
        let new_model = MLP::load(path)?;

        // Atomic swap
        let mut models = self.models.write().await;
        models.insert(name.clone(), Box::new(new_model));

        tracing::info!("Model {} updated successfully", name);
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PredictionResponse {
    pub predictions: Vec<f64>,
    pub latency_ms: f64,
    pub model_name: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub num_models: usize,
    pub total_predictions: u64,
    pub avg_latency_ms: f64,
}
```

#### Monitoring & Alerting

```rust
use prometheus::{Registry, Counter, Histogram, HistogramOpts};

pub struct ModelMonitoring {
    prediction_count: Counter,
    prediction_latency: Histogram,
    prediction_errors: Counter,
    model_drift: Histogram,
}

impl ModelMonitoring {
    pub fn new(registry: &Registry) -> Result<Self> {
        let prediction_count = Counter::new(
            "model_predictions_total",
            "Total number of predictions made"
        )?;
        registry.register(Box::new(prediction_count.clone()))?;

        let prediction_latency = Histogram::with_opts(
            HistogramOpts::new(
                "model_prediction_latency_ms",
                "Prediction latency in milliseconds"
            )
            .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        )?;
        registry.register(Box::new(prediction_latency.clone()))?;

        let prediction_errors = Counter::new(
            "model_prediction_errors_total",
            "Total number of prediction errors"
        )?;
        registry.register(Box::new(prediction_errors.clone()))?;

        let model_drift = Histogram::with_opts(
            HistogramOpts::new(
                "model_prediction_error",
                "Actual prediction error (MAE)"
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
        )?;
        registry.register(Box::new(model_drift.clone()))?;

        Ok(Self {
            prediction_count,
            prediction_latency,
            prediction_errors,
            model_drift,
        })
    }

    pub fn record_prediction(
        &self,
        latency_ms: f64,
        error: Option<f64>,
    ) {
        self.prediction_count.inc();
        self.prediction_latency.observe(latency_ms);

        if let Some(err) = error {
            self.model_drift.observe(err);
        }
    }

    pub fn record_error(&self) {
        self.prediction_errors.inc();
    }
}
```

### Phase 4: Production Validation (Week 9)

#### A/B Testing Framework

```rust
pub struct ABTest {
    model_a: Box<dyn NeuralModel>,
    model_b: Box<dyn NeuralModel>,
    traffic_split: f64,  // 0.0 = all A, 1.0 = all B
    metrics_a: Arc<Mutex<ModelMetrics>>,
    metrics_b: Arc<Mutex<ModelMetrics>>,
}

impl ABTest {
    pub fn predict(&mut self, horizon: usize) -> Result<Vec<f64>> {
        let use_b = rand::random::<f64>() < self.traffic_split;

        let start = std::time::Instant::now();
        let (prediction, metrics) = if use_b {
            (self.model_b.predict(horizon)?, &self.metrics_b)
        } else {
            (self.model_a.predict(horizon)?, &self.metrics_a)
        };
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // Update metrics
        metrics.lock().unwrap().record_prediction(latency);

        prediction
    }

    pub fn get_results(&self) -> ABTestResults {
        let metrics_a = self.metrics_a.lock().unwrap();
        let metrics_b = self.metrics_b.lock().unwrap();

        // Statistical significance test (t-test)
        let (t_stat, p_value) = self.t_test(&metrics_a, &metrics_b);

        ABTestResults {
            model_a_avg_latency: metrics_a.avg_latency_ms,
            model_b_avg_latency: metrics_b.avg_latency_ms,
            model_a_error_rate: metrics_a.error_rate(),
            model_b_error_rate: metrics_b.error_rate(),
            statistically_significant: p_value < 0.05,
            p_value,
            recommendation: self.make_recommendation(&metrics_a, &metrics_b),
        }
    }
}
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Overfitting on Small Datasets

**Symptom**: Great training accuracy, poor test accuracy

**Solution**:
```rust
// 1. Use stronger regularization
let config = ModelConfig::default()
    .with_dropout(0.3)  // Increase dropout
    .with_weight_decay(0.001);  // Add weight decay

// 2. Early stopping
let train_config = TrainingConfig::default()
    .with_patience(10)  // Stop after 10 epochs no improvement
    .with_validation_split(0.2);  // Use 20% for validation

// 3. Data augmentation
fn augment_timeseries(data: &[f64]) -> Vec<Vec<f64>> {
    vec![
        data.to_vec(),  // Original
        add_noise(data, 0.01),  // Add 1% noise
        scale(data, 0.95),  // Scale by 95%
        scale(data, 1.05),  // Scale by 105%
    ]
}
```

### Pitfall 2: Gradient Explosion

**Symptom**: Loss becomes NaN, weights blow up

**Solution**:
```rust
// 1. Gradient clipping
let train_config = TrainingConfig::default()
    .with_gradient_clip(Some(1.0));  // Clip gradients to max norm 1.0

// 2. Lower learning rate
let config = ModelConfig::default()
    .with_learning_rate(0.0001);  // Start with very low LR

// 3. Batch normalization (if implemented)
```

### Pitfall 3: Slow Convergence

**Symptom**: Loss decreases very slowly

**Solution**:
```rust
// 1. Increase learning rate
let config = ModelConfig::default()
    .with_learning_rate(0.01);  // Try higher LR

// 2. Use Adam optimizer (already default)
// 3. Better initialization (already using Xavier)

// 4. Check for dead neurons
fn detect_dead_neurons(&self) -> Vec<usize> {
    let mut dead = Vec::new();
    for (i, layer) in self.weights.iter().enumerate() {
        let num_dead = layer.iter()
            .filter(|&&w| w.abs() < 1e-8)
            .count();

        if num_dead > layer.len() / 2 {
            dead.push(i);
        }
    }
    dead
}
```

### Pitfall 4: Production Latency Spikes

**Symptom**: Occasional very slow predictions

**Solution**:
```rust
// 1. Warm up model cache
pub fn warmup(&mut self) {
    for _ in 0..100 {
        let _ = self.predict(1);
    }
}

// 2. Use connection pooling for concurrent requests
// 3. Monitor and alert on P99 latency

// 4. Implement circuit breaker
pub struct CircuitBreaker {
    failure_threshold: usize,
    timeout_ms: u64,
    current_failures: AtomicUsize,
    state: AtomicU8,  // 0=closed, 1=open, 2=half-open
}
```

---

## Production Deployment Patterns

### Pattern 1: Shadow Mode

Deploy new model alongside existing, compare predictions:

```rust
pub struct ShadowDeployment {
    production_model: Box<dyn NeuralModel>,
    shadow_model: Box<dyn NeuralModel>,
    comparison_log: Arc<Mutex<Vec<PredictionComparison>>>,
}

impl ShadowDeployment {
    pub fn predict(&mut self, horizon: usize) -> Result<Vec<f64>> {
        // Get both predictions
        let prod_pred = self.production_model.predict(horizon)?;
        let shadow_pred = self.shadow_model.predict(horizon)?;

        // Log comparison
        self.comparison_log.lock().unwrap().push(
            PredictionComparison {
                production: prod_pred.clone(),
                shadow: shadow_pred,
                timestamp: Utc::now(),
            }
        );

        // Return production prediction
        Ok(prod_pred)
    }

    pub fn analyze_divergence(&self) -> DivergenceReport {
        let log = self.comparison_log.lock().unwrap();

        let avg_diff: f64 = log.iter()
            .map(|c| {
                c.production.iter()
                    .zip(&c.shadow)
                    .map(|(p, s)| (p - s).abs())
                    .sum::<f64>() / c.production.len() as f64
            })
            .sum::<f64>() / log.len() as f64;

        DivergenceReport {
            avg_difference: avg_diff,
            max_difference: /* compute max */,
            recommendation: if avg_diff < 0.05 {
                "Safe to promote shadow to production"
            } else {
                "Investigate large divergence"
            },
        }
    }
}
```

### Pattern 2: Canary Deployment

Gradually shift traffic to new model:

```rust
pub struct CanaryDeployment {
    old_model: Box<dyn NeuralModel>,
    new_model: Box<dyn NeuralModel>,
    canary_percentage: Arc<AtomicU8>,  // 0-100
}

impl CanaryDeployment {
    pub fn predict(&mut self, horizon: usize) -> Result<Vec<f64>> {
        let use_new = rand::random::<u8>() < self.canary_percentage.load(Ordering::Relaxed);

        if use_new {
            self.new_model.predict(horizon)
        } else {
            self.old_model.predict(horizon)
        }
    }

    pub fn ramp_up(&self) {
        // Gradual ramp: 1% -> 5% -> 10% -> 50% -> 100%
        tokio::spawn(async move {
            for percentage in [1, 5, 10, 50, 100] {
                self.canary_percentage.store(percentage, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_hours(1)).await;

                // Check metrics, rollback if needed
                if self.should_rollback().await {
                    self.canary_percentage.store(0, Ordering::Relaxed);
                    break;
                }
            }
        });
    }
}
```

---

## Cost & Resource Planning

### Computational Costs

**Training** (per model, 1000 samples):
- CPU Time: ~35 ms (optimized) to 2,500 ms (current)
- Memory Peak: 7 MB (current) to 875 KB (optimized)
- GPU (optional): 2-5x faster training

**Inference** (per prediction):
- CPU Time: 0.2-1 ms
- Memory: 2 MB (loaded model)
- Throughput: 1,000-5,000 predictions/sec per core

### Infrastructure Recommendations

**Development**:
- 4-core CPU
- 8 GB RAM
- No GPU needed

**Production**:
- 8-16 core CPU
- 16-32 GB RAM
- Load balancer for HA
- Redis for model caching
- Prometheus + Grafana for monitoring

**Estimated Costs** (AWS, monthly):
- Dev: ~$50 (t3.large)
- Prod: ~$500 (c5.4xlarge + monitoring)

---

## Conclusion

### Current State Summary

| Model | Status | Prod Ready | Action Required |
|-------|--------|------------|-----------------|
| MLP | 65% | ⚠️ Soon | Fix backprop, add optimizations |
| DLinear | 30% | ❌ No | Complete reimplementation |
| NLinear | 30% | ❌ No | Complete reimplementation |
| MLPMultivariate | 25% | ❌ No | Complete reimplementation |

### Recommended Next Steps

1. **Week 1-2**: Fix MLP backpropagation
2. **Week 3-4**: Optimize MLP (SIMD, Rayon, f32)
3. **Week 5-6**: Production infrastructure (server, monitoring)
4. **Week 7-8**: Implement DLinear/NLinear properly
5. **Week 9**: A/B testing and validation
6. **Week 10**: Production deployment with canary rollout

### Success Criteria

**MLP Production-Ready When**:
- ✅ Backprop implemented and tested
- ✅ Achieves <1% MAE on benchmark datasets
- ✅ Inference latency <2ms P99
- ✅ Comprehensive error handling
- ✅ Monitoring and alerting in place
- ✅ Successful A/B test vs baseline

**DLinear/NLinear Production-Ready When**:
- ✅ Actual algorithms implemented
- ✅ Beats naive baseline by 50%+
- ✅ Inference latency <0.5ms
- ✅ Validated against paper results

---

**End of Production Deployment Guide**
