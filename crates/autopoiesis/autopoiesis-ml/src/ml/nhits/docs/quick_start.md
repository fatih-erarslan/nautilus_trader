# Quick Start Guide

Get up and running with NHITS in minutes. This guide covers installation, basic usage, and your first forecasting model.

## Prerequisites

- Rust 1.70+ with Cargo
- Basic understanding of time series data
- Familiarity with `ndarray` for numerical operations

## Installation

Add NHITS to your `Cargo.toml`:

```toml
[dependencies]
autopoiesis = { version = "0.1.0", features = ["ml"] }
tokio = { version = "1.35", features = ["full"] }
ndarray = "0.15"
```

## Your First Model

### 1. Basic Setup

```rust
use std::sync::Arc;
use autopoiesis::ml::nhits::prelude::*;
use autopoiesis::consciousness::ConsciousnessField;
use autopoiesis::core::autopoiesis::AutopoieticSystem;
use ndarray::Array3;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize consciousness and autopoietic systems
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // Create a simple configuration
    let config = NHITSConfig::minimal(); // Start with minimal complexity
    
    // Initialize the model
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    println!("✅ NHITS model initialized successfully!");
    Ok(())
}
```

### 2. Prepare Your Data

```rust
// Example: Hourly temperature data
let sequence_length = 168; // One week of hourly data
let batch_size = 1;
let features = 1;

// Create sample data (replace with your actual data)
let mut data = Array3::zeros((batch_size, sequence_length, features));
for i in 0..sequence_length {
    // Simulate temperature with daily cycle + noise
    let hour = i as f64;
    let daily_cycle = 20.0 + 5.0 * (hour * std::f64::consts::PI / 12.0).sin();
    let noise = rand::random::<f64>() * 2.0 - 1.0;
    data[[0, i, 0]] = daily_cycle + noise;
}

println!("📊 Data shape: {:?}", data.shape());
```

### 3. Generate Your First Forecast

```rust
// Generate 24-hour forecast
let lookback_window = 168; // Use full week for context
let forecast_horizon = 24; // Predict next 24 hours

let predictions = model.forward(&data, lookback_window, forecast_horizon)?;

println!("🔮 Forecast shape: {:?}", predictions.shape());
println!("📈 First 5 predictions: {:?}", 
    predictions.slice(s![0, ..5, 0]).to_vec());
```

### 4. Simple Training Loop

```rust
// Create training and validation data
let train_data = data.clone(); // Use your training data
let val_data = data.clone();   // Use your validation data

// Train the model
let history = model.train(&train_data, Some(&val_data), 10)?;

println!("📚 Training completed!");
println!("Final training loss: {:.4}", history.train_losses.last().unwrap());
println!("Final validation loss: {:.4}", history.val_losses.last().unwrap());
```

## Complete Example: Stock Price Forecasting

```rust
use std::sync::Arc;
use autopoiesis::ml::nhits::prelude::*;
use autopoiesis::consciousness::ConsciousnessField;
use autopoiesis::core::autopoiesis::AutopoieticSystem;
use ndarray::{Array1, Array3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize systems
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // 2. Configure for financial time series
    let config = NHITSConfigBuilder::new()
        .with_lookback(50)      // 50 days of history
        .with_horizon(5)        // 5 days forecast
        .with_features(1, 1)    // Single feature (price)
        .with_consciousness(true, 0.1) // Enable consciousness
        .build()?;
    
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // 3. Prepare sample stock data
    let prices = generate_sample_stock_data(200); // 200 days of data
    let data = prepare_time_series_data(&prices, 50, 1)?;
    
    // 4. Split data
    let split_point = data.shape()[1] - 60; // Keep last 60 days for validation
    let train_data = data.slice(s![.., ..split_point, ..]).to_owned();
    let val_data = data.slice(s![.., split_point.., ..]).to_owned();
    
    // 5. Train model
    println!("🎯 Training model...");
    let history = model.train(&train_data, Some(&val_data), 50)?;
    
    // 6. Generate forecasts
    let latest_data = data.slice(s![.., -50.., ..]).to_owned();
    let forecast = model.forward(&latest_data, 50, 5)?;
    
    println!("📊 Training Results:");
    println!("  - Final train loss: {:.6}", history.train_losses.last().unwrap());
    println!("  - Final val loss: {:.6}", history.val_losses.last().unwrap());
    println!("  - Best epoch: {}", history.best_epoch);
    
    println!("🔮 5-day Forecast:");
    for i in 0..5 {
        println!("  Day {}: ${:.2}", i + 1, forecast[[0, i, 0]]);
    }
    
    Ok(())
}

fn generate_sample_stock_data(days: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(days);
    let mut price = 100.0;
    
    for _ in 0..days {
        // Random walk with slight upward trend
        let change = rand::random::<f64>() * 4.0 - 2.0 + 0.01;
        price += change;
        price = price.max(1.0); // Prevent negative prices
        prices.push(price);
    }
    
    prices
}

fn prepare_time_series_data(
    prices: &[f64], 
    lookback: usize, 
    batch_size: usize
) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
    let n_samples = prices.len() - lookback;
    let mut data = Array3::zeros((batch_size, n_samples, 1));
    
    for i in 0..n_samples {
        data[[0, i, 0]] = prices[i + lookback];
    }
    
    Ok(data)
}
```

## Configuration Options

### Preset Configurations

```rust
// For different use cases
let short_term = NHITSConfig::for_use_case(UseCase::ShortTermForecasting);
let long_term = NHITSConfig::for_use_case(UseCase::LongTermForecasting);
let multivariate = NHITSConfig::for_use_case(UseCase::MultivariateSeries);
let hft = NHITSConfig::for_use_case(UseCase::HighFrequencyTrading);
let anomaly = NHITSConfig::for_use_case(UseCase::AnomalyDetection);
```

### Custom Configuration

```rust
let config = NHITSConfigBuilder::new()
    .with_lookback(100)           // 100 time steps history
    .with_horizon(20)             // 20 steps forecast
    .with_features(5, 1)          // 5 input features, 1 output
    .with_consciousness(true, 0.2) // Enable consciousness with weight 0.2
    .with_blocks(vec![
        BlockConfig {
            input_size: 256,
            hidden_size: 256,
            num_basis: 8,
            pooling_factor: 2,
            pooling_type: PoolingType::Average,
            interpolation_type: InterpolationType::Linear,
            dropout_rate: 0.1,
            activation: ActivationType::GELU,
        }
    ])
    .build()?;
```

## Online Learning

Update your model with new data as it arrives:

```rust
// New observations
let new_inputs = vec![Array1::from_vec(vec![45.2])];
let new_targets = vec![Array1::from_vec(vec![45.8])];

// Update model
model.update_online(&new_inputs, &new_targets)?;

// Generate updated forecast
let forecast = model.predict(&new_inputs[0], 10)?;
```

## Production Forecasting Pipeline

For production use, consider the forecasting pipeline:

```rust
use autopoiesis::ml::nhits::forecasting::{ForecastingPipeline, ForecastingConfig};

let config = ForecastingConfig {
    horizons: vec![1, 7, 30],        // Multi-horizon forecasts
    ensemble_size: 5,                // Use 5 models for ensemble
    confidence_levels: vec![0.90, 0.95], // Prediction intervals
    online_learning: true,           // Enable continuous learning
    ..Default::default()
};

let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic).await?;

// Generate comprehensive forecast
let forecast = pipeline.forecast(&input_data, None).await?;

// Access results
for (horizon, prediction) in &forecast.forecasts {
    println!("{}-step forecast: {:?}", horizon, prediction);
}
```

## Common Patterns

### 1. Data Preprocessing

```rust
// Normalize data
let mut normalized_data = data.clone();
let mean = data.mean().unwrap();
let std = data.std(0.0);
normalized_data = (normalized_data - mean) / std;
```

### 2. Model Persistence

```rust
// Save model state
let state = model.save_state()?;
// ... serialize state to file ...

// Later, restore state
model.load_state(state)?;
```

### 3. Error Handling

```rust
match model.forward(&data, 50, 10) {
    Ok(predictions) => {
        println!("Forecast generated successfully");
    },
    Err(NHITSError::ShapeMismatch { expected, got }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, got);
    },
    Err(e) => {
        eprintln!("Forecast error: {}", e);
    }
}
```

## Next Steps

1. **Explore Examples**: Check out the [examples directory](examples_guide.md) for domain-specific use cases
2. **Configuration**: Read the [configuration guide](configuration.md) for advanced options
3. **Performance**: Review [performance tuning](performance_tuning.md) for optimization tips
4. **API Reference**: Consult the [API documentation](api_reference.md) for detailed method descriptions
5. **Deployment**: See the [deployment guide](deployment_guide.md) for production considerations

## Common Issues

### Model Not Converging
- Reduce learning rate: `config.learning_rate = 0.0001`
- Increase batch size: `config.batch_size = 64`
- Enable consciousness: `config.consciousness_enabled = true`

### Poor Forecast Accuracy
- Increase lookback window
- Add more hierarchical blocks
- Enable adaptive structure: `adaptation_config.adaptation_strategy = AdaptationStrategy::ConsciousnessGuided`

### Memory Issues
- Use minimal configuration: `NHITSConfig::minimal()`
- Reduce batch size
- Consider using the streaming forecasting pipeline

## Support

- 📚 [Full Documentation](README.md)
- 🔧 [Troubleshooting Guide](troubleshooting.md)
- 💬 [Community Discord](https://discord.gg/autopoiesis)
- 🐛 [Report Issues](https://github.com/autopoiesis/autopoiesis/issues)