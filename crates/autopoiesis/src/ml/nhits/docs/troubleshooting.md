# Troubleshooting Guide

Comprehensive guide for diagnosing and resolving common issues with the NHITS system.

## Table of Contents

- [Common Issues](#common-issues)
- [Error Messages](#error-messages)
- [Performance Problems](#performance-problems)
- [Training Issues](#training-issues)
- [Inference Problems](#inference-problems)
- [Memory Issues](#memory-issues)
- [Configuration Problems](#configuration-problems)
- [Integration Issues](#integration-issues)
- [Diagnostic Tools](#diagnostic-tools)
- [Getting Help](#getting-help)

## Common Issues

### Model Not Converging

**Symptoms:**
- Training loss remains high or increases
- Validation loss doesn't improve
- Model predictions are poor or random

**Possible Causes:**
1. Learning rate too high or too low
2. Insufficient training data
3. Poor data quality or preprocessing
4. Model architecture mismatch with data complexity
5. Consciousness coherence issues

**Solutions:**

```rust
// 1. Adjust learning rate
let config = NHITSConfigBuilder::new()
    .with_lookback(168)
    .with_horizon(24)
    .build()?;

// Try different learning rates
config.learning_rate = 0.0001; // Lower if oscillating
config.learning_rate = 0.01;   // Higher if too slow

// 2. Increase training data
let train_data = load_more_data()?; // Get more samples

// 3. Improve data preprocessing
let normalized_data = normalize_data(&raw_data)?;
let cleaned_data = remove_outliers(&normalized_data)?;

// 4. Adjust model complexity
let simpler_config = NHITSConfig::minimal(); // Start simple
let complex_config = NHITSConfig::for_use_case(UseCase::LongTermForecasting);

// 5. Check consciousness coherence
let consciousness_state = model.consciousness.get_current_state();
if consciousness_state.coherence < 0.5 {
    // Reset consciousness field
    model.consciousness.reset()?;
}
```

### Poor Forecast Accuracy

**Symptoms:**
- High MAPE (>10%)
- Predictions don't follow data patterns
- Forecast intervals too wide or too narrow

**Solutions:**

```rust
// 1. Increase model complexity
let config = NHITSConfigBuilder::new()
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 512,  // Larger hidden size
            num_basis: 12,     // More basis functions
            ..Default::default()
        },
        // Add more blocks
        BlockConfig {
            hidden_size: 256,
            num_basis: 8,
            ..Default::default()
        }
    ])
    .build()?;

// 2. Use appropriate lookback window
config.lookback_window = detect_optimal_lookback(&data)?;

// 3. Enable consciousness for better pattern recognition
config.consciousness_enabled = true;
config.coherence_weight = 0.2;

// 4. Try ensemble methods
let pipeline = ForecastingPipeline::new(
    ForecastingConfig {
        ensemble_size: 10,
        ..Default::default()
    },
    consciousness,
    autopoietic
).await?;
```

### Memory Issues

**Symptoms:**
- Out of memory errors
- System becomes slow
- Process killed by OS

**Solutions:**

```rust
// 1. Reduce batch size
config.batch_size = 16; // Smaller batches

// 2. Use streaming processing
let mut model = NHITS::new(config, consciousness, autopoietic);
for chunk in data.chunks(1000) {
    let chunk_result = model.forward(chunk, lookback, horizon)?;
    // Process chunk_result immediately
}

// 3. Enable model compression
let compressed_model = compress_model(&model)?;

// 4. Use memory-efficient data types
use ndarray::Array3;
let data_f32 = Array3::<f32>::zeros((batch, seq, feat)); // Use f32 instead of f64
```

### Slow Training/Inference

**Symptoms:**
- Training takes too long
- Inference latency too high
- CPU/GPU utilization low

**Solutions:**

```rust
// 1. Optimize configuration for speed
let fast_config = NHITSConfigBuilder::new()
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 128,     // Smaller hidden size
            num_basis: 4,         // Fewer basis functions
            pooling_factor: 4,    // More aggressive pooling
            activation: ActivationType::ReLU, // Fastest activation
            dropout_rate: 0.0,    // No dropout for inference
            ..Default::default()
        }
    ])
    .with_consciousness(false, 0.0) // Disable for max speed
    .build()?;

// 2. Use parallel processing
use rayon::prelude::*;
let predictions: Vec<_> = inputs.par_iter()
    .map(|input| model.predict(input, horizon))
    .collect();

// 3. Enable caching
let cache = NHITSCache::new();
let cached_result = cache.get_or_compute_forecast(
    &input, &model, lookback, horizon
).await?;

// 4. Profile to find bottlenecks
let profiler = NHITSProfiler::new(&prometheus_registry);
profiler.start_timer("forward_pass");
let result = model.forward(&input, lookback, horizon)?;
profiler.end_timer("forward_pass");
```

## Error Messages

### `ShapeMismatch` Error

```
Error: Shape mismatch: expected [32, 168, 1], got [32, 100, 1]
```

**Cause:** Input data dimensions don't match configuration.

**Solution:**
```rust
// Check your data shape
println!("Input shape: {:?}", input.shape());
println!("Expected: batch_size={}, sequence_length={}, features={}", 
         config.batch_size, config.lookback_window, config.input_features);

// Reshape if necessary
let reshaped_input = input.slice(s![.., ..config.lookback_window, ..]).to_owned();

// Or adjust configuration
let config = NHITSConfigBuilder::new()
    .with_lookback(input.shape()[1]) // Match actual data
    .build()?;
```

### `InvalidConfig` Error

```
Error: Invalid configuration: lookback_window must be > 0
```

**Cause:** Configuration parameters are invalid.

**Solution:**
```rust
// Validate configuration before use
let config = NHITSConfigBuilder::new()
    .with_lookback(168)     // Must be > 0
    .with_horizon(24)       // Must be > 0
    .with_features(1, 1)    // Must be > 0
    .build()?;              // This will validate

// Check specific constraints
if config.lookback_window == 0 {
    return Err("Lookback window must be positive".into());
}
```

### `ComputationError` Error

```
Error: Computation error: NaN values detected in forward pass
```

**Cause:** Numerical instability or invalid input data.

**Solution:**
```rust
// Check for NaN/Inf in input data
fn validate_input(input: &Array3<f64>) -> Result<(), NHITSError> {
    for value in input.iter() {
        if !value.is_finite() {
            return Err(NHITSError::ComputationError(
                format!("Invalid value in input: {}", value)
            ));
        }
    }
    Ok(())
}

validate_input(&input)?;

// Handle NaN values
let cleaned_input = input.mapv(|x| if x.is_nan() { 0.0 } else { x });

// Reduce learning rate to improve stability
config.learning_rate = 0.0001;
config.gradient_clip = 0.5; // Clip gradients
```

### `AdaptationError` Error

```
Error: Adaptation error: Cannot adapt structure - minimum depth reached
```

**Cause:** Adaptive structure tried to make invalid changes.

**Solution:**
```rust
// Adjust adaptation constraints
let adaptation_config = AdaptationConfig {
    min_depth: 1,           // Allow more flexibility
    max_depth: 10,          // Set reasonable limits
    change_threshold: 0.01, // Less sensitive to changes
    adaptation_strategy: AdaptationStrategy::Conservative, // More stable
    ..Default::default()
};

// Or disable adaptation temporarily
let config = NHITSConfigBuilder::new()
    .with_adaptation(AdaptationStrategy::Conservative, 0.0) // No adaptation
    .build()?;
```

## Performance Problems

### High Memory Usage

**Diagnosis:**
```rust
use sysinfo::{System, SystemExt};

let mut system = System::new_all();
system.refresh_memory();
println!("Memory usage: {} MB", system.used_memory() / 1024 / 1024);

// Monitor during operation
let memory_before = system.used_memory();
let result = model.forward(&input, lookback, horizon)?;
system.refresh_memory();
let memory_after = system.used_memory();
println!("Memory increase: {} MB", (memory_after - memory_before) / 1024 / 1024);
```

**Solutions:**
```rust
// 1. Use smaller batch sizes
config.batch_size = 8;

// 2. Process in chunks
fn process_large_dataset(
    model: &mut NHITS,
    data: &Array3<f64>,
    chunk_size: usize,
) -> Result<Vec<Array3<f64>>, NHITSError> {
    let mut results = Vec::new();
    let n_samples = data.shape()[0];
    
    for start in (0..n_samples).step_by(chunk_size) {
        let end = (start + chunk_size).min(n_samples);
        let chunk = data.slice(s![start..end, .., ..]).to_owned();
        let result = model.forward(&chunk, lookback, horizon)?;
        results.push(result);
    }
    
    Ok(results)
}

// 3. Use memory pools
let tensor_pool = TensorPool::new(100);
let reused_tensor = tensor_pool.get_tensor((batch_size, seq_len, features));
```

### Slow Inference

**Diagnosis:**
```rust
use std::time::Instant;

let start = Instant::now();
let result = model.forward(&input, lookback, horizon)?;
let duration = start.elapsed();
println!("Inference time: {:?}", duration);

// Profile individual components
profiler.start_timer("decomposition");
let decomposed = model.decomposer.decompose(&input)?;
profiler.end_timer("decomposition");

profiler.start_timer("attention");
let attended = model.attention.apply(&decomposed, None)?;
profiler.end_timer("attention");
```

**Solutions:**
```rust
// 1. Optimize model architecture
let fast_config = NHITSConfig::for_use_case(UseCase::HighFrequencyTrading);

// 2. Use caching
let cache = NHITSCache::new();

// 3. Batch predictions
let batched_inputs = stack_inputs(&individual_inputs);
let batched_results = model.forward(&batched_inputs, lookback, horizon)?;
let individual_results = unstack_results(&batched_results);

// 4. Pre-compute features
let preprocessor = DataPreprocessor::new();
let preprocessed_input = preprocessor.preprocess(&raw_input)?;
```

### Training Instability

**Symptoms:**
- Loss jumps around or explodes
- Gradients become very large or very small
- Model weights become NaN

**Solutions:**
```rust
// 1. Gradient clipping
config.gradient_clip = 1.0; // Clip gradients to norm of 1.0

// 2. Learning rate scheduling
struct LearningRateScheduler {
    initial_lr: f64,
    decay_rate: f64,
    decay_steps: usize,
}

impl LearningRateScheduler {
    fn get_lr(&self, step: usize) -> f64 {
        self.initial_lr * (self.decay_rate.powf(step as f64 / self.decay_steps as f64))
    }
}

// 3. Batch normalization or layer normalization
config.use_layer_norm = true;

// 4. Weight initialization
fn initialize_weights_carefully(model: &mut NHITS) {
    // Use Xavier/He initialization
    for block in &mut model.blocks {
        block.initialize_weights_xavier()?;
    }
}

// 5. Monitor training metrics
struct TrainingMonitor {
    loss_history: Vec<f64>,
    gradient_norms: Vec<f64>,
}

impl TrainingMonitor {
    fn check_stability(&self) -> bool {
        if let Some(&last_loss) = self.loss_history.last() {
            return last_loss.is_finite() && last_loss < 1e6;
        }
        false
    }
}
```

## Training Issues

### Overfitting

**Symptoms:**
- Training loss decreases but validation loss increases
- Large gap between training and validation performance
- Model performs poorly on new data

**Solutions:**
```rust
// 1. Increase regularization
config.dropout_rate = 0.3;     // Higher dropout
config.weight_decay = 0.001;   // L2 regularization

// 2. Early stopping
struct EarlyStopping {
    patience: usize,
    best_loss: f64,
    wait: usize,
}

impl EarlyStopping {
    fn should_stop(&mut self, val_loss: f64) -> bool {
        if val_loss < self.best_loss {
            self.best_loss = val_loss;
            self.wait = 0;
            false
        } else {
            self.wait += 1;
            self.wait >= self.patience
        }
    }
}

// 3. Data augmentation
fn augment_time_series(data: &Array3<f64>) -> Array3<f64> {
    let mut augmented = data.clone();
    
    // Add noise
    let noise = Array3::random(data.shape(), Normal::new(0.0, 0.01).unwrap());
    augmented = augmented + noise;
    
    // Time warping
    // Scale shifting
    // etc.
    
    augmented
}

// 4. Cross-validation
fn cross_validate(
    data: &Array3<f64>,
    k_folds: usize,
) -> Result<Vec<f64>, NHITSError> {
    let fold_size = data.shape()[0] / k_folds;
    let mut scores = Vec::new();
    
    for fold in 0..k_folds {
        let val_start = fold * fold_size;
        let val_end = (fold + 1) * fold_size;
        
        let train_data = concatenate![
            Axis(0),
            data.slice(s![..val_start, .., ..]),
            data.slice(s![val_end.., .., ..])
        ];
        let val_data = data.slice(s![val_start..val_end, .., ..]).to_owned();
        
        let mut model = NHITS::new(config.clone(), consciousness.clone(), autopoietic.clone());
        let history = model.train(&train_data, Some(&val_data), 100)?;
        
        scores.push(history.best_val_loss);
    }
    
    Ok(scores)
}
```

### Underfitting

**Symptoms:**
- Both training and validation loss remain high
- Model predictions are too simple or constant
- Adding more data doesn't help

**Solutions:**
```rust
// 1. Increase model complexity
let complex_config = NHITSConfigBuilder::new()
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 1024,  // Larger hidden size
            num_basis: 15,      // More basis functions
            ..Default::default()
        },
        // Add more blocks
        BlockConfig { hidden_size: 512, num_basis: 10, ..Default::default() },
        BlockConfig { hidden_size: 256, num_basis: 8, ..Default::default() },
    ])
    .build()?;

// 2. Reduce regularization
config.dropout_rate = 0.0;    // No dropout
config.weight_decay = 0.0;    // No L2 regularization

// 3. Feature engineering
fn engineer_features(data: &Array3<f64>) -> Array3<f64> {
    let mut features = data.clone();
    
    // Add lagged features
    let lagged = create_lagged_features(&features, &[1, 7, 30]);
    
    // Add rolling statistics
    let rolling_mean = create_rolling_features(&features, 7, |x| x.mean());
    let rolling_std = create_rolling_features(&features, 7, |x| x.std(0.0));
    
    // Add trend and seasonality
    let trend = extract_trend(&features);
    let seasonal = extract_seasonal(&features, 24); // Daily seasonality
    
    concatenate![Axis(2), features, lagged, rolling_mean, rolling_std, trend, seasonal]
}

// 4. Longer training
let history = model.train(&train_data, Some(&val_data), 500)?; // More epochs
```

## Inference Problems

### Incorrect Predictions

**Diagnosis:**
```rust
// Check input data quality
fn diagnose_input(input: &Array3<f64>) {
    println!("Input statistics:");
    println!("  Shape: {:?}", input.shape());
    println!("  Mean: {:.4}", input.mean().unwrap());
    println!("  Std: {:.4}", input.std(0.0));
    println!("  Min: {:.4}", input.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max: {:.4}", input.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    
    // Check for patterns
    let first_series = input.slice(s![0, .., 0]);
    println!("  First series trend: {:.4}", calculate_trend(&first_series.to_vec()));
}

// Compare with training data distribution
fn compare_distributions(
    train_data: &Array3<f64>,
    inference_data: &Array3<f64>,
) {
    let train_mean = train_data.mean().unwrap();
    let train_std = train_data.std(0.0);
    let inference_mean = inference_data.mean().unwrap();
    let inference_std = inference_data.std(0.0);
    
    println!("Distribution comparison:");
    println!("  Train: mean={:.4}, std={:.4}", train_mean, train_std);
    println!("  Inference: mean={:.4}, std={:.4}", inference_mean, inference_std);
    
    let distribution_shift = ((inference_mean - train_mean) / train_std).abs();
    if distribution_shift > 2.0 {
        println!("  WARNING: Significant distribution shift detected!");
    }
}
```

**Solutions:**
```rust
// 1. Normalize inference data same as training data
struct DataNormalizer {
    mean: f64,
    std: f64,
}

impl DataNormalizer {
    fn fit(&mut self, data: &Array3<f64>) {
        self.mean = data.mean().unwrap();
        self.std = data.std(0.0);
    }
    
    fn transform(&self, data: &Array3<f64>) -> Array3<f64> {
        (data - self.mean) / self.std
    }
}

// 2. Handle distribution shift
fn adapt_to_distribution_shift(
    model: &mut NHITS,
    new_data: &Array3<f64>,
) -> Result<(), NHITSError> {
    // Online adaptation
    let adaptation_samples = new_data.slice(s![..10, .., ..]).to_owned();
    
    for i in 0..adaptation_samples.shape()[0] {
        let input = adaptation_samples.slice(s![i..i+1, .., ..]).to_owned();
        let target = input.clone(); // Self-supervised adaptation
        
        model.update_online(&[input.into_dimensionality()?], &[target.into_dimensionality()?])?;
    }
    
    Ok(())
}

// 3. Ensemble for robustness
fn ensemble_predict(
    models: &[NHITS],
    input: &Array3<f64>,
    lookback: usize,
    horizon: usize,
) -> Result<Array3<f64>, NHITSError> {
    let mut predictions = Vec::new();
    
    for model in models {
        let mut model_clone = model.clone();
        let pred = model_clone.forward(input, lookback, horizon)?;
        predictions.push(pred);
    }
    
    // Average predictions
    let shape = predictions[0].shape();
    let mut ensemble_pred = Array3::zeros(shape);
    
    for pred in &predictions {
        ensemble_pred = ensemble_pred + pred;
    }
    
    ensemble_pred = ensemble_pred / predictions.len() as f64;
    Ok(ensemble_pred)
}
```

### Prediction Intervals Too Wide/Narrow

**Solutions:**
```rust
// 1. Calibrate prediction intervals
struct IntervalCalibrator {
    quantiles: Vec<f64>,
    calibration_data: Vec<(f64, f64)>, // (prediction, actual)
}

impl IntervalCalibrator {
    fn fit(&mut self, predictions: &[f64], actuals: &[f64]) {
        self.calibration_data = predictions.iter()
            .zip(actuals.iter())
            .map(|(&p, &a)| (p, a))
            .collect();
    }
    
    fn predict_intervals(&self, point_prediction: f64, confidence: f64) -> (f64, f64) {
        // Use calibration data to adjust intervals
        let residuals: Vec<f64> = self.calibration_data.iter()
            .map(|(pred, actual)| (actual - pred).abs())
            .collect();
        
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quantile_idx = (confidence * residuals.len() as f64) as usize;
        let margin = residuals[quantile_idx.min(residuals.len() - 1)];
        
        (point_prediction - margin, point_prediction + margin)
    }
}

// 2. Use quantile regression
fn train_quantile_model(
    base_model: &NHITS,
    quantiles: &[f64],
) -> Result<HashMap<String, NHITS>, NHITSError> {
    let mut quantile_models = HashMap::new();
    
    for &quantile in quantiles {
        let mut model = base_model.clone();
        // Modify loss function for quantile regression
        // This would require extending the NHITS implementation
        quantile_models.insert(format!("q{:.2}", quantile), model);
    }
    
    Ok(quantile_models)
}
```

## Configuration Problems

### Invalid Parameter Combinations

**Common Issues:**
```rust
// Problem: Inconsistent dimensions
let bad_config = NHITSConfigBuilder::new()
    .with_features(5, 1)    // 5 input features
    .with_blocks(vec![
        BlockConfig {
            input_size: 3,  // ❌ Doesn't match input_features
            ..Default::default()
        }
    ])
    .build(); // Will fail validation

// Solution: Ensure consistency
let good_config = NHITSConfigBuilder::new()
    .with_features(5, 1)
    .with_blocks(vec![
        BlockConfig {
            input_size: 5,  // ✅ Matches input_features
            ..Default::default()
        }
    ])
    .build()?;

// Problem: Conflicting settings
let conflicting_config = NHITSConfigBuilder::new()
    .with_consciousness(true, 0.0)  // ❌ Enabled but zero weight
    .build()?;

// Solution: Logical settings
let logical_config = NHITSConfigBuilder::new()
    .with_consciousness(true, 0.1)  // ✅ Enabled with positive weight
    .build()?;
```

### Configuration Validation

```rust
fn validate_configuration(config: &NHITSConfig) -> Result<(), ConfigError> {
    // Check basic constraints
    if config.lookback_window == 0 {
        return Err(ConfigError::InvalidParameter(
            "lookback_window must be > 0".to_string()
        ));
    }
    
    if config.forecast_horizon == 0 {
        return Err(ConfigError::InvalidParameter(
            "forecast_horizon must be > 0".to_string()
        ));
    }
    
    // Check block consistency
    for (i, block) in config.block_configs.iter().enumerate() {
        if i == 0 && block.input_size != config.input_features {
            return Err(ConfigError::InvalidParameter(
                format!("First block input_size ({}) must match input_features ({})",
                       block.input_size, config.input_features)
            ));
        }
        
        if block.hidden_size == 0 {
            return Err(ConfigError::InvalidParameter(
                format!("Block {} hidden_size must be > 0", i)
            ));
        }
    }
    
    // Check consciousness settings
    if config.consciousness_enabled && config.coherence_weight <= 0.0 {
        return Err(ConfigError::InvalidParameter(
            "consciousness_enabled requires positive coherence_weight".to_string()
        ));
    }
    
    // Check adaptation settings
    if config.adaptation_config.max_depth <= config.adaptation_config.min_depth {
        return Err(ConfigError::InvalidParameter(
            "adaptation max_depth must be > min_depth".to_string()
        ));
    }
    
    Ok(())
}
```

## Integration Issues

### Consciousness Field Integration

**Problem:** Consciousness field not responding or providing useful signals.

**Solutions:**
```rust
// 1. Check consciousness field initialization
let consciousness = Arc::new(ConsciousnessField::new());
let state = consciousness.get_current_state();
println!("Consciousness coherence: {:.3}", state.coherence);

if state.coherence < 0.1 {
    println!("WARNING: Low consciousness coherence");
    consciousness.reset()?;
}

// 2. Monitor consciousness evolution
struct ConsciousnessMonitor {
    coherence_history: Vec<f64>,
}

impl ConsciousnessMonitor {
    fn update(&mut self, consciousness: &ConsciousnessField) {
        let state = consciousness.get_current_state();
        self.coherence_history.push(state.coherence);
        
        // Check for degradation
        if self.coherence_history.len() > 100 {
            let recent_avg = self.coherence_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older_avg = self.coherence_history.iter().rev().skip(50).take(10).sum::<f64>() / 10.0;
            
            if recent_avg < older_avg * 0.8 {
                println!("WARNING: Consciousness coherence degrading");
            }
        }
    }
}

// 3. Adjust consciousness integration
config.coherence_weight = 0.05; // Reduce if causing instability
config.min_coherence_threshold = 0.1; // Lower threshold
```

### Autopoietic System Integration

**Problem:** Autopoietic system not adapting or causing performance issues.

**Solutions:**
```rust
// 1. Check autopoietic system health
let autopoietic = Arc::new(AutopoieticSystem::new());
let system_state = autopoietic.get_system_state();

if !system_state.is_healthy() {
    println!("WARNING: Autopoietic system unhealthy");
    autopoietic.repair_system()?;
}

// 2. Adjust adaptation parameters
let adaptation_config = AdaptationConfig {
    adaptation_rate: 0.001,  // Slower adaptation
    performance_window: 50,  // Longer evaluation window
    change_threshold: 0.05,  // Less sensitive
    exploration_rate: 0.01,  // Less exploration
    adaptation_strategy: AdaptationStrategy::Conservative,
    ..Default::default()
};

// 3. Monitor adaptation effects
struct AdaptationMonitor {
    performance_before: Vec<f64>,
    performance_after: Vec<f64>,
    adaptations: Vec<StructuralChange>,
}

impl AdaptationMonitor {
    fn record_adaptation(&mut self, change: StructuralChange, performance: f64) {
        self.adaptations.push(change);
        self.performance_after.push(performance);
        
        // Analyze adaptation effectiveness
        if self.performance_before.len() == self.performance_after.len() {
            let improvement = performance - self.performance_before.last().unwrap();
            println!("Adaptation impact: {:.4}", improvement);
        }
    }
}
```

## Diagnostic Tools

### Built-in Diagnostics

```rust
use autopoiesis::ml::nhits::diagnostics::*;

// 1. Model health check
let health_report = ModelHealthChecker::new()
    .check_configuration(&config)?
    .check_model_state(&model)?
    .check_consciousness_integration(&consciousness)?
    .check_data_quality(&input_data)?
    .generate_report();

println!("{}", health_report);

// 2. Performance profiler
let mut profiler = NHITSProfiler::new();
profiler.start_profiling();

let result = model.forward(&input, lookback, horizon)?;

let profile_report = profiler.stop_profiling();
println!("Performance Profile:");
println!("  Forward pass: {:.2}ms", profile_report.forward_time_ms);
println!("  Memory usage: {:.1}MB", profile_report.peak_memory_mb);

// 3. Data analyzer
let data_report = DataAnalyzer::new()
    .analyze_distribution(&input_data)
    .analyze_stationarity(&input_data)
    .analyze_seasonality(&input_data)
    .check_missing_values(&input_data)
    .generate_report();

println!("{}", data_report);
```

### Custom Diagnostics

```rust
// Custom diagnostic function
fn diagnose_training_issues(
    model: &NHITS,
    train_data: &Array3<f64>,
    val_data: &Array3<f64>,
    history: &TrainingHistory,
) -> DiagnosticReport {
    let mut report = DiagnosticReport::new();
    
    // Check for overfitting
    if let (Some(&train_loss), Some(&val_loss)) = 
        (history.train_losses.last(), history.val_losses.last()) {
        let overfitting_ratio = val_loss / train_loss;
        if overfitting_ratio > 1.5 {
            report.add_warning("Potential overfitting detected".to_string());
            report.add_suggestion("Increase regularization or reduce model complexity".to_string());
        }
    }
    
    // Check for underfitting
    let final_train_loss = history.train_losses.last().unwrap_or(&f64::INFINITY);
    if *final_train_loss > 0.1 {  // Threshold depends on data scale
        report.add_warning("High training loss - potential underfitting".to_string());
        report.add_suggestion("Increase model complexity or reduce regularization".to_string());
    }
    
    // Check training stability
    let loss_variance = calculate_variance(&history.train_losses);
    if loss_variance > 0.01 {
        report.add_warning("Training instability detected".to_string());
        report.add_suggestion("Reduce learning rate or add gradient clipping".to_string());
    }
    
    // Check data quality
    let data_issues = check_data_quality(train_data);
    report.data_quality_issues = data_issues;
    
    report
}

#[derive(Debug)]
struct DiagnosticReport {
    warnings: Vec<String>,
    suggestions: Vec<String>,
    data_quality_issues: Vec<String>,
}

impl DiagnosticReport {
    fn new() -> Self {
        Self {
            warnings: Vec::new(),
            suggestions: Vec::new(),
            data_quality_issues: Vec::new(),
        }
    }
    
    fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    fn add_suggestion(&mut self, suggestion: String) {
        self.suggestions.push(suggestion);
    }
}

fn check_data_quality(data: &Array3<f64>) -> Vec<String> {
    let mut issues = Vec::new();
    
    // Check for NaN/Inf values
    for value in data.iter() {
        if !value.is_finite() {
            issues.push("Non-finite values detected in data".to_string());
            break;
        }
    }
    
    // Check for constant sequences
    let n_samples = data.shape()[0];
    for i in 0..n_samples {
        let series = data.slice(s![i, .., 0]);
        let first_value = series[0];
        if series.iter().all(|&x| (x - first_value).abs() < 1e-10) {
            issues.push(format!("Constant sequence detected in sample {}", i));
        }
    }
    
    // Check for outliers
    let mean = data.mean().unwrap();
    let std = data.std(0.0);
    let outlier_threshold = 3.0 * std;
    let outliers = data.iter().filter(|&&x| (x - mean).abs() > outlier_threshold).count();
    
    if outliers > data.len() / 100 {  // More than 1% outliers
        issues.push(format!("{} outliers detected ({}%)", outliers, 
                           100 * outliers / data.len()));
    }
    
    issues
}
```

## Getting Help

### Debug Information to Collect

When reporting issues, please include:

```rust
// 1. System information
fn collect_system_info() -> SystemInfo {
    SystemInfo {
        rust_version: env!("RUSTC_VERSION").to_string(),
        autopoiesis_version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

// 2. Configuration
let config_json = serde_json::to_string_pretty(&config)?;
println!("Configuration:\n{}", config_json);

// 3. Error details with stack trace
use std::backtrace::Backtrace;

fn handle_error(error: NHITSError) {
    println!("Error: {}", error);
    println!("Backtrace:\n{}", Backtrace::capture());
}

// 4. Data statistics
fn collect_data_stats(data: &Array3<f64>) -> DataStats {
    DataStats {
        shape: data.shape().to_vec(),
        mean: data.mean().unwrap(),
        std: data.std(0.0),
        min: data.iter().cloned().fold(f64::INFINITY, f64::min),
        max: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        has_nan: data.iter().any(|x| x.is_nan()),
        has_inf: data.iter().any(|x| x.is_infinite()),
    }
}

// 5. Model state
let model_state = model.save_state()?;
println!("Model state: {:?}", model_state);
```

### Support Channels

1. **GitHub Issues**: [https://github.com/autopoiesis/autopoiesis/issues](https://github.com/autopoiesis/autopoiesis/issues)
   - Bug reports
   - Feature requests
   - Documentation issues

2. **Discord Community**: [https://discord.gg/autopoiesis](https://discord.gg/autopoiesis)
   - Real-time help
   - Community discussions
   - Quick questions

3. **Documentation**: [https://docs.autopoiesis.ai](https://docs.autopoiesis.ai)
   - Comprehensive guides
   - API reference
   - Examples

4. **Email Support**: support@autopoiesis.ai
   - Enterprise support
   - Security issues
   - Private consultations

### Creating Minimal Reproducible Examples

```rust
// Template for bug reports
use std::sync::Arc;
use autopoiesis::ml::nhits::prelude::*;

fn minimal_reproduction() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Minimal configuration
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    let config = NHITSConfig::minimal(); // Use minimal config
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // 2. Minimal data
    let input = Array3::zeros((1, 24, 1)); // Smallest possible input
    
    // 3. Reproduce the issue
    let result = model.forward(&input, 24, 1);
    
    match result {
        Ok(_) => println!("No error reproduced"),
        Err(e) => {
            println!("Error reproduced: {}", e);
            // Include system info, configuration, etc.
        }
    }
    
    Ok(())
}
```

### Performance Issues Template

```rust
// Template for performance issues
fn performance_issue_reproduction() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    // Setup
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    let config = /* your configuration */;
    let mut model = NHITS::new(config, consciousness, autopoietic);
    let input = /* your input data */;
    
    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = model.forward(&input, lookback, horizon)?;
    }
    
    let duration = start.elapsed();
    let avg_time = duration / iterations;
    
    println!("Performance Results:");
    println!("  Average time per prediction: {:?}", avg_time);
    println!("  Predictions per second: {:.1}", 1.0 / avg_time.as_secs_f64());
    println!("  Configuration: {:?}", config);
    println!("  Input shape: {:?}", input.shape());
    
    // Include system specs
    println!("System Info:");
    println!("  {}", collect_system_info());
    
    Ok(())
}
```

Remember to always provide:
- Complete error messages
- Configuration used
- Input data characteristics
- Expected vs actual behavior
- System specifications
- Steps to reproduce