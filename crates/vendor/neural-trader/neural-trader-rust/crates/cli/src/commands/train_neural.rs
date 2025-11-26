//! Train neural forecasting models

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[cfg(feature = "neural")]
use nt_neural::{NHITSTrainer, NHITSTrainingConfig, OptimizerConfig, TrainingConfig};

#[cfg(feature = "neural")]
use tracing::{info, warn};

#[derive(Parser, Debug)]
pub struct TrainNeuralArgs {
    /// Model type (nhits, lstm, transformer)
    #[arg(long, default_value = "nhits")]
    pub model: String,

    /// Path to training data (CSV or Parquet)
    #[arg(long, short = 'd')]
    pub data: PathBuf,

    /// Target column name
    #[arg(long, default_value = "close")]
    pub target: String,

    /// Input sequence length (lookback window)
    #[arg(long, default_value = "168")]
    pub input_size: usize,

    /// Forecast horizon
    #[arg(long, default_value = "24")]
    pub horizon: usize,

    /// Number of training epochs
    #[arg(long, default_value = "100")]
    pub epochs: usize,

    /// Batch size
    #[arg(long, default_value = "32")]
    pub batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    pub lr: f64,

    /// Weight decay (L2 regularization)
    #[arg(long, default_value = "1e-5")]
    pub weight_decay: f64,

    /// Hidden layer size
    #[arg(long, default_value = "512")]
    pub hidden_size: usize,

    /// Dropout rate
    #[arg(long, default_value = "0.1")]
    pub dropout: f64,

    /// Validation split ratio
    #[arg(long, default_value = "0.2")]
    pub val_split: f64,

    /// Early stopping patience
    #[arg(long, default_value = "10")]
    pub patience: usize,

    /// Output model path
    #[arg(long, short = 'o')]
    pub output: PathBuf,

    /// Checkpoint directory
    #[arg(long)]
    pub checkpoint_dir: Option<PathBuf>,

    /// GPU device ID (None for CPU)
    #[arg(long)]
    pub gpu: Option<usize>,

    /// Optimizer type (adam, adamw, sgd, rmsprop)
    #[arg(long, default_value = "adamw")]
    pub optimizer: String,

    /// NHITS: Number of stacks
    #[arg(long, default_value = "3")]
    pub n_stacks: usize,

    /// Enable mixed precision training (FP16)
    #[arg(long)]
    pub mixed_precision: bool,

    /// Resume from checkpoint
    #[arg(long)]
    pub resume: Option<PathBuf>,
}

#[cfg(feature = "neural")]
pub async fn execute(args: TrainNeuralArgs) -> Result<()> {
    info!("🚀 Neural Trader - Model Training");
    info!("Model: {}", args.model);
    info!("Data: {:?}", args.data);
    info!("Target: {}", args.target);
    info!("Epochs: {}, Batch Size: {}", args.epochs, args.batch_size);
    info!("Learning Rate: {}, Weight Decay: {}", args.lr, args.weight_decay);
    info!("Input Size: {}, Horizon: {}", args.input_size, args.horizon);

    // Validate data file exists
    if !args.data.exists() {
        anyhow::bail!("Data file not found: {:?}", args.data);
    }

    // Determine file format
    let is_parquet = args
        .data
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s == "parquet")
        .unwrap_or(false);

    if is_parquet {
        info!("Loading Parquet data (optimized for large datasets)");
    } else {
        info!("Loading CSV data");
    }

    match args.model.as_str() {
        "nhits" => train_nhits(args, is_parquet).await,
        "lstm" | "transformer" => {
            warn!("Model type '{}' not yet implemented, using NHITS", args.model);
            train_nhits(args, is_parquet).await
        }
        _ => anyhow::bail!("Unknown model type: {}", args.model),
    }
}

#[cfg(not(feature = "neural"))]
pub async fn execute(_args: TrainNeuralArgs) -> Result<()> {
    anyhow::bail!(
        "Neural training requires the 'neural' feature.\n\
         Rebuild with: cargo build --features neural\n\
         Or use the NPM package which includes neural support."
    )
}

#[cfg(feature = "neural")]
async fn train_nhits(args: TrainNeuralArgs, is_parquet: bool) -> Result<()> {
    info!("Training NHITS model...");

    // Create optimizer config
    let optimizer_config = match args.optimizer.as_str() {
        "adam" => OptimizerConfig::adam(args.lr),
        "adamw" => OptimizerConfig::adamw(args.lr, args.weight_decay),
        "sgd" => OptimizerConfig::sgd(args.lr, 0.9),
        "rmsprop" => OptimizerConfig::rmsprop(args.lr),
        _ => {
            warn!("Unknown optimizer '{}', using AdamW", args.optimizer);
            OptimizerConfig::adamw(args.lr, args.weight_decay)
        }
    };

    // Create training configuration
    let mut config = NHITSTrainingConfig {
        base: TrainingConfig {
            batch_size: args.batch_size,
            num_epochs: args.epochs,
            learning_rate: args.lr,
            weight_decay: args.weight_decay,
            gradient_clip: Some(1.0),
            early_stopping_patience: args.patience,
            validation_split: args.val_split,
            mixed_precision: args.mixed_precision,
        },
        model_config: nt_neural::NHITSConfig {
            base: nt_neural::ModelConfig {
                input_size: args.input_size,
                horizon: args.horizon,
                hidden_size: args.hidden_size,
                dropout: args.dropout,
                num_features: 1, // Will be set from data
                device: None,
            },
            n_stacks: args.n_stacks,
            n_blocks: vec![1; args.n_stacks],
            n_freq_downsample: if args.n_stacks == 3 {
                vec![4, 2, 1]
            } else {
                (0..args.n_stacks)
                    .rev()
                    .map(|i| 2_usize.pow(i as u32))
                    .collect()
            },
            mlp_units: vec![vec![args.hidden_size, args.hidden_size]; args.n_stacks],
            interpolation_mode: nt_neural::nhits::InterpolationMode::Linear,
            pooling_mode: nt_neural::nhits::PoolingMode::MaxPool,
            quantiles: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        optimizer_config,
        use_quantile_loss: false,
        target_quantiles: vec![0.1, 0.5, 0.9],
        checkpoint_dir: args.checkpoint_dir.clone(),
        tensorboard_dir: None,
        save_every: 10,
        gpu_device: args.gpu,
    };

    // Create trainer
    let mut trainer = NHITSTrainer::new(config.clone())
        .context("Failed to create NHITS trainer")?;

    // Resume from checkpoint if specified
    if let Some(ref resume_path) = args.resume {
        info!("Resuming from checkpoint: {:?}", resume_path);
        trainer
            .load_model(resume_path)
            .context("Failed to load checkpoint")?;
    }

    // Train model
    info!("Starting training...");
    info!("Press Ctrl+C to stop early (best model will be saved)");

    let metrics = if is_parquet {
        trainer
            .train_from_parquet(&args.data, &args.target)
            .await
            .context("Training failed")?
    } else {
        trainer
            .train_from_csv(&args.data, &args.target)
            .await
            .context("Training failed")?
    };

    info!("✅ Training completed successfully!");
    info!("Final metrics:");
    info!("  - Train Loss: {:.6}", metrics.train_loss);
    if let Some(val_loss) = metrics.val_loss {
        info!("  - Validation Loss: {:.6}", val_loss);
    }
    info!("  - Learning Rate: {:.2e}", metrics.learning_rate);
    info!("  - Epoch Time: {:.2}s", metrics.epoch_time_seconds);

    // Save final model
    info!("Saving model to: {:?}", args.output);
    trainer
        .save_model(&args.output)
        .context("Failed to save model")?;

    // Print metrics history summary
    let history = trainer.metrics_history();
    if !history.is_empty() {
        info!("\n📊 Training Summary:");
        info!("  Total Epochs: {}", history.len());

        let best_val_loss = history
            .iter()
            .filter_map(|m| m.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap());

        if let Some(best_loss) = best_val_loss {
            info!("  Best Validation Loss: {:.6}", best_loss);
        }

        let final_train_loss = history.last().map(|m| m.train_loss);
        if let Some(final_loss) = final_train_loss {
            info!("  Final Train Loss: {:.6}", final_loss);
        }
    }

    // Print next steps
    info!("\n📝 Next Steps:");
    info!("  1. Test model: neural-trader test-neural --model {:?}", args.output);
    info!("  2. Run inference: neural-trader predict --model {:?} --data <test_data.csv>", args.output);
    info!("  3. Backtest strategy: neural-trader backtest --neural-model {:?}", args.output);

    Ok(())
}

#[cfg(all(test, feature = "neural"))]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = TrainNeuralArgs::parse_from(&[
            "train-neural",
            "--data",
            "data.csv",
            "--target",
            "price",
            "--epochs",
            "50",
            "--lr",
            "0.01",
        ]);

        assert_eq!(args.target, "price");
        assert_eq!(args.epochs, 50);
        assert_eq!(args.lr, 0.01);
    }

    #[test]
    fn test_default_values() {
        let args = TrainNeuralArgs::parse_from(&[
            "train-neural",
            "--data",
            "data.csv",
            "--output",
            "model.safetensors",
        ]);

        assert_eq!(args.model, "nhits");
        assert_eq!(args.target, "close");
        assert_eq!(args.batch_size, 32);
        assert_eq!(args.epochs, 100);
    }
}
