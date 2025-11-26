//! Example: Train NHITS model on synthetic stock data
//!
//! This example demonstrates:
//! - Generating synthetic stock price data
//! - Configuring and training an NHITS model
//! - Saving the trained model
//! - Validating model performance
//!
//! Run with:
//! ```bash
//! cargo run --example train_nhits_example --features candle
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use polars::prelude::*;
use nt_neural::{
    NHITSTrainer, NHITSTrainingConfig, OptimizerConfig, TimeSeriesDataset, TrainingConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 NHITS Training Example");
    println!("=" .repeat(60));

    // Step 1: Generate synthetic stock data
    println!("\n📊 Step 1: Generating synthetic stock data...");
    let df = generate_synthetic_stock_data(2000)?;
    println!("Generated {} samples", df.height());

    // Step 2: Save data to CSV for later use
    println!("\n💾 Step 2: Saving data to CSV...");
    let mut file = std::fs::File::create("synthetic_stock_data.csv")?;
    CsvWriter::new(&mut file).finish(&mut df.clone())?;
    println!("Saved to: synthetic_stock_data.csv");

    // Step 3: Configure training
    println!("\n⚙️  Step 3: Configuring training...");
    let config = NHITSTrainingConfig {
        base: TrainingConfig {
            batch_size: 32,
            num_epochs: 50,
            learning_rate: 0.001,
            weight_decay: 1e-5,
            gradient_clip: Some(1.0),
            early_stopping_patience: 10,
            validation_split: 0.2,
            mixed_precision: false,
        },
        model_config: nt_neural::NHITSConfig {
            base: nt_neural::ModelConfig {
                input_size: 168,  // 1 week of hourly data
                horizon: 24,      // 24 hour forecast
                hidden_size: 512,
                dropout: 0.1,
                num_features: 1,
                device: None,
            },
            n_stacks: 3,
            n_blocks: vec![1, 1, 1],
            n_freq_downsample: vec![4, 2, 1],
            mlp_units: vec![vec![512, 512], vec![512, 512], vec![512, 512]],
            interpolation_mode: nt_neural::nhits::InterpolationMode::Linear,
            pooling_mode: nt_neural::nhits::PoolingMode::MaxPool,
            quantiles: vec![0.1, 0.5, 0.9],
        },
        optimizer_config: OptimizerConfig::adamw(0.001, 1e-5),
        use_quantile_loss: false,
        target_quantiles: vec![0.1, 0.5, 0.9],
        checkpoint_dir: Some("checkpoints".into()),
        tensorboard_dir: None,
        save_every: 10,
        gpu_device: None,
    };

    println!("Configuration:");
    println!("  Input Size: {}", config.model_config.base.input_size);
    println!("  Horizon: {}", config.model_config.base.horizon);
    println!("  Hidden Size: {}", config.model_config.base.hidden_size);
    println!("  Batch Size: {}", config.base.batch_size);
    println!("  Epochs: {}", config.base.num_epochs);
    println!("  Learning Rate: {}", config.base.learning_rate);

    // Step 4: Create trainer
    println!("\n🔧 Step 4: Creating trainer...");
    let mut trainer = NHITSTrainer::new(config)?;

    // Step 5: Train model
    println!("\n🏋️  Step 5: Training model...");
    println!("This may take a few minutes...\n");

    let metrics = trainer.train_from_dataframe(df.clone(), "close").await?;

    println!("\n✅ Training completed!");
    println!("Final Metrics:");
    println!("  Train Loss: {:.6}", metrics.train_loss);
    if let Some(val_loss) = metrics.val_loss {
        println!("  Validation Loss: {:.6}", val_loss);
    }
    println!("  Learning Rate: {:.2e}", metrics.learning_rate);

    // Step 6: Print training history
    println!("\n📈 Training History:");
    let history = trainer.metrics_history();
    for (i, m) in history.iter().enumerate() {
        if i % 5 == 0 || i == history.len() - 1 {
            println!(
                "  Epoch {:3}: train={:.6}, val={:?}, lr={:.2e}",
                i + 1,
                m.train_loss,
                m.val_loss.map(|v| format!("{:.6}", v)),
                m.learning_rate
            );
        }
    }

    // Step 7: Validate on test set
    println!("\n🧪 Step 7: Validating on test set...");
    let test_df = generate_synthetic_stock_data(500)?;
    let test_dataset = TimeSeriesDataset::new(test_df, "close", 168, 24)?;

    let eval_metrics = trainer.validate(test_dataset).await?;

    println!("Test Set Metrics:");
    println!("  MAE: {:.6}", eval_metrics.mae);
    println!("  RMSE: {:.6}", eval_metrics.rmse);
    println!("  MAPE: {:.2}%", eval_metrics.mape);
    println!("  R²: {:.4}", eval_metrics.r2_score);

    // Step 8: Save model
    println!("\n💾 Step 8: Saving trained model...");
    trainer.save_model("trained_nhits_model.safetensors")?;
    println!("Saved to: trained_nhits_model.safetensors");

    // Summary
    println!("\n" .repeat(1));
    println!("=" .repeat(60));
    println!("🎉 Example completed successfully!");
    println!("=" .repeat(60));
    println!("\nNext steps:");
    println!("  1. Use the model: Load 'trained_nhits_model.safetensors'");
    println!("  2. View data: Open 'synthetic_stock_data.csv'");
    println!("  3. Check checkpoints: See 'checkpoints/' directory");

    Ok(())
}

/// Generate synthetic stock price data with realistic patterns
fn generate_synthetic_stock_data(n_samples: usize) -> Result<DataFrame> {
    let start_price = 100.0;
    let mut prices = Vec::with_capacity(n_samples);
    let mut volumes = Vec::with_capacity(n_samples);
    let mut timestamps = Vec::with_capacity(n_samples);

    let base_time = Utc::now() - Duration::hours(n_samples as i64);

    let mut price = start_price;

    for i in 0..n_samples {
        let t = i as f64 / 100.0;

        // Trend component (slight upward bias)
        let trend = t * 0.05;

        // Seasonal components (daily and weekly patterns)
        let hour_of_day = (i % 24) as f64;
        let day_pattern = ((hour_of_day / 24.0) * 2.0 * std::f64::consts::PI).sin() * 2.0;

        let day_of_week = ((i / 24) % 7) as f64;
        let week_pattern = ((day_of_week / 7.0) * 2.0 * std::f64::consts::PI).cos() * 3.0;

        // Random walk component
        let random_change = (rand::random::<f64>() - 0.5) * 1.5;

        // Update price with all components
        price += trend + day_pattern * 0.1 + week_pattern * 0.1 + random_change;

        // Ensure price doesn't go negative
        price = price.max(1.0);

        prices.push(price);

        // Volume with some correlation to price movement
        let volume = 1000000.0 + random_change.abs() * 100000.0;
        volumes.push(volume);

        // Timestamp
        let timestamp = (base_time + Duration::hours(i as i64))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        timestamps.push(timestamp);
    }

    // Create OHLC data
    let open = prices.iter().map(|&p| p * 0.999).collect::<Vec<_>>();
    let high = prices.iter().map(|&p| p * 1.005).collect::<Vec<_>>();
    let low = prices.iter().map(|&p| p * 0.995).collect::<Vec<_>>();

    let df = df!(
        "timestamp" => timestamps,
        "open" => open,
        "high" => high,
        "low" => low,
        "close" => prices,
        "volume" => volumes
    )?;

    Ok(df)
}
