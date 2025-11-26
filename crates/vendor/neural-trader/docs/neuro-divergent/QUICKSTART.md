# Neuro-Divergent Quick Start

## 30-Second Example

```rust
use neuro_divergent::{NeuralModel, ModelConfig, TimeSeriesDataFrame, models::basic::MLP};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let data = TimeSeriesDataFrame::from_values(values, None)?;

    // Configure and train
    let config = ModelConfig::default().with_input_size(5).with_horizon(3);
    let mut model = MLP::new(config);
    model.fit(&data)?;

    // Predict
    let predictions = model.predict(3)?;
    println!("Predictions: {:?}", predictions);
    Ok(())
}
```

## All 27+ Models Available

Use any model via the registry:

```rust
use neuro_divergent::{ModelFactory, models::register_all_models};

register_all_models()?;

let config = ModelConfig::default();
let model = ModelFactory::create("nhits", &config)?;  // or "lstm", "gru", "nbeats", etc.
```

## Full Guide

See `/workspaces/neural-trader/docs/neuro-divergent/` for complete documentation.
