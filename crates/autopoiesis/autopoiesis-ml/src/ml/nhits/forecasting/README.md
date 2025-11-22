# NHITS Forecasting Pipeline

A production-ready time series forecasting system built on top of the NHITS (Neural Hierarchical Interpolation for Time Series) architecture. This pipeline provides comprehensive forecasting capabilities with consciousness-aware neural processing.

## Features

### 🎯 Core Capabilities

- **Multi-Horizon Forecasting**: Simultaneous predictions for multiple time horizons
- **Ensemble Methods**: Multiple models for robust predictions
- **Online Learning**: Continuous model updates with streaming data
- **Uncertainty Quantification**: Prediction intervals with configurable confidence levels
- **Anomaly Detection**: Integrated outlier detection with customizable thresholds
- **Adaptive Retraining**: Automatic model updates based on performance metrics
- **Concept Drift Detection**: Identifies distribution changes in data

### 🔧 Data Processing

- **Preprocessing Pipeline**:
  - Normalization and standardization
  - Detrending (linear, polynomial, moving average)
  - Seasonal decomposition
  - Outlier handling (clip, remove, impute)

- **Feature Engineering**:
  - Lag features
  - Rolling statistics (mean, std)
  - Fourier features for seasonality
  - Calendar features
  - External feature integration

### 💾 Model Management

- **Persistence**: Save/load models with versioning
- **Compression**: Optional model compression
- **Auto-save**: Periodic model checkpointing
- **Version Control**: Track model evolution

### 📊 Performance Monitoring

- **Real-time Metrics**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - MAPE (Mean Absolute Percentage Error)
  - Prediction interval coverage
  - Forecast bias

- **Event System**:
  - Forecast generation notifications
  - Model retraining alerts
  - Anomaly detection warnings
  - Performance degradation alerts
  - Concept drift notifications

## Usage

### Basic Setup

```rust
use std::sync::Arc;
use autopoiesis::ml::nhits::forecasting::{ForecastingPipeline, ForecastingConfig};
use autopoiesis::consciousness::ConsciousnessField;
use autopoiesis::core::autopoiesis::AutopoieticSystem;

// Initialize systems
let consciousness = Arc::new(ConsciousnessField::new());
let autopoietic = Arc::new(AutopoieticSystem::new());

// Configure pipeline
let config = ForecastingConfig {
    horizons: vec![1, 7, 30],  // 1-day, 1-week, 1-month
    ensemble_size: 5,
    confidence_levels: vec![0.90, 0.95],
    ..Default::default()
};

// Create pipeline
let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic).await?;
```

### Generating Forecasts

```rust
// Basic forecast
let forecast = pipeline.forecast(&input_data, None).await?;

// With external features
let forecast = pipeline.forecast(&input_data, Some(&external_features)).await?;

// Access results
for (horizon, prediction) in &forecast.forecasts {
    println!("{}-step forecast: {:?}", horizon, prediction);
}

// Check prediction intervals
if let Some((lower, upper)) = forecast.intervals.get(&(7, 0.95)) {
    println!("95% interval for 7-step: [{:?}, {:?}]", lower, upper);
}
```

### Online Learning

```rust
// Update with new data
pipeline.update(&new_input, &actual_output).await?;

// Model automatically retrains when needed based on:
// - Performance degradation
// - Time since last retraining
// - Concept drift detection
```

### Event Handling

```rust
// Subscribe to events
let mut events = pipeline.subscribe();

tokio::spawn(async move {
    while let Ok(event) = events.recv().await {
        match event {
            ForecastingEvent::AnomalyDetected(score) => {
                println!("Anomaly detected: {}", score);
            }
            ForecastingEvent::ModelRetrained(version) => {
                println!("Model retrained: {}", version);
            }
            // Handle other events...
        }
    }
});
```

## Configuration Options

### Forecasting Configuration

```rust
ForecastingConfig {
    // Prediction horizons
    horizons: vec![1, 7, 30],
    
    // Ensemble settings
    ensemble_size: 5,
    
    // Uncertainty quantification
    confidence_levels: vec![0.90, 0.95, 0.99],
    
    // Online learning
    online_window_size: 1000,
    update_frequency: 100,
    
    // Anomaly detection
    anomaly_threshold: 3.0,
    
    // Sub-configurations
    retraining_config: RetrainingConfig { ... },
    preprocessing_config: PreprocessingConfig { ... },
    persistence_config: PersistenceConfig { ... },
}
```

### Retraining Configuration

```rust
RetrainingConfig {
    // Trigger retraining if performance drops by 20%
    performance_threshold: 0.2,
    
    // Force retraining every 7 days
    max_time_between_retraining: Duration::days(7),
    
    // Minimum samples before retraining
    min_samples: 1000,
    
    // Enable concept drift detection
    detect_concept_drift: true,
}
```

### Preprocessing Configuration

```rust
PreprocessingConfig {
    // Enable normalization
    normalize: true,
    
    // Detrending method
    detrending: DetrendingMethod::Linear,
    
    // Seasonal decomposition
    seasonal_decomposition: true,
    
    // Feature engineering
    feature_engineering: FeatureEngineeringConfig {
        lag_features: vec![1, 7, 30],
        rolling_windows: vec![7, 30],
        fourier_features: Some(10),
        calendar_features: true,
    },
    
    // Outlier handling
    outlier_handling: OutlierHandling::Clip,
}
```

## Examples

### Financial Time Series

```rust
// Configure for financial data
let config = ForecastingConfig {
    horizons: vec![1, 5, 20],  // Daily, weekly, monthly
    anomaly_threshold: 3.5,     // Detect market anomalies
    preprocessing_config: PreprocessingConfig {
        detrending: DetrendingMethod::Linear,
        feature_engineering: FeatureEngineeringConfig {
            lag_features: vec![1, 5, 20, 60],
            rolling_windows: vec![5, 20, 60],
            ..Default::default()
        },
        ..Default::default()
    },
    ..Default::default()
};
```

### IoT Sensor Monitoring

```rust
// Configure for anomaly detection
let config = ForecastingConfig {
    horizons: vec![1, 3, 10],   // Short-term focus
    anomaly_threshold: 2.5,     // More sensitive
    update_frequency: 10,       // Frequent updates
    ..Default::default()
};
```

### Energy Demand Forecasting

```rust
// Configure with external features
let config = ForecastingConfig {
    horizons: vec![24, 168],    // Hourly and weekly
    preprocessing_config: PreprocessingConfig {
        feature_engineering: FeatureEngineeringConfig {
            calendar_features: true,  // Day/hour patterns
            fourier_features: Some(24), // Daily seasonality
            ..Default::default()
        },
        ..Default::default()
    },
    ..Default::default()
};

// Use with weather data
let weather_features = Array2::from_shape_vec(
    (n_samples, 3),
    vec![/* temperature, humidity, day_of_week */]
)?;

let forecast = pipeline.forecast(&demand, Some(&weather_features)).await?;
```

## Performance Characteristics

- **Scalability**: Handles time series of any length
- **Memory Efficiency**: Streaming updates with bounded buffers
- **Parallel Processing**: Ensemble models run concurrently
- **Adaptive Complexity**: Model structure evolves with data

## Integration with Consciousness System

The forecasting pipeline leverages the consciousness field for:

- **Attention Weighting**: Focus on important patterns
- **Coherence Monitoring**: Track model stability
- **Adaptive Learning**: Consciousness-guided parameter updates
- **Pattern Recognition**: Enhanced anomaly detection

## Best Practices

1. **Data Quality**: Ensure consistent sampling frequency
2. **Feature Selection**: Include relevant external variables
3. **Horizon Selection**: Choose horizons matching business needs
4. **Monitoring**: Set up event handlers for production
5. **Retraining**: Configure based on data volatility
6. **Persistence**: Regular model checkpointing

## Troubleshooting

### Common Issues

1. **Poor Forecast Accuracy**:
   - Check data preprocessing settings
   - Increase ensemble size
   - Add more relevant features
   - Adjust retraining frequency

2. **High Uncertainty**:
   - Increase training data
   - Reduce forecast horizon
   - Check for data quality issues

3. **Frequent Retraining**:
   - Adjust performance thresholds
   - Check for concept drift
   - Increase min_samples requirement

4. **Memory Usage**:
   - Reduce online_window_size
   - Enable model compression
   - Limit saved versions

## Future Enhancements

- [ ] Distributed ensemble training
- [ ] GPU acceleration support
- [ ] Advanced drift detection algorithms
- [ ] Automated hyperparameter tuning
- [ ] Multi-variate forecasting
- [ ] Hierarchical time series support