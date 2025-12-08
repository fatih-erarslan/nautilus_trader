//! Example of using the ML training infrastructure

use ml_training_infrastructure::{
    initialize, MLInfrastructure, TrainingConfig, ModelType,
    data::DataLoader, TrainingOrchestrator, TrainingJob, JobStatus,
};
use std::path::PathBuf;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize infrastructure
    initialize().await?;
    
    // Create configuration
    let mut config = TrainingConfig::default();
    config.data.source_path = PathBuf::from("data/market_data.parquet");
    config.data.batch_size = 64;
    config.data.sequence_length = 100;
    config.data.horizon = 10;
    
    // Example 1: Train single model
    train_single_model(config.clone()).await?;
    
    // Example 2: Train with cross-validation
    train_with_cv(config.clone()).await?;
    
    // Example 3: Train multiple models with orchestrator
    train_with_orchestrator(config.clone()).await?;
    
    // Example 4: Hyperparameter optimization
    train_with_hpo(config.clone()).await?;
    
    Ok(())
}

/// Train a single model
async fn train_single_model(config: TrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training single XGBoost model...");
    
    // Create infrastructure
    let infra = MLInfrastructure::new(config).await?;
    
    // Load data
    let data_loader = DataLoader::new(config.data.clone().into());
    let data = data_loader.load(&config.data.source_path).await?;
    
    // Train model
    let model_id = infra.train_model(
        ModelType::XGBoost,
        data,
        "xgboost_experiment"
    ).await?;
    
    println!("Model trained successfully. ID: {}", model_id);
    
    // Load and use model for prediction
    let model = infra.deploy_model(&model_id).await?;
    let predictions = model.predict(&data.x_test).await?;
    
    println!("Predictions shape: {:?}", predictions.shape());
    
    Ok(())
}

/// Train with cross-validation
async fn train_with_cv(mut config: TrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training with cross-validation...");
    
    // Enable cross-validation
    config.validation.n_folds = 5;
    config.validation.cv_strategy = crate::config::CVStrategy::TimeSeriesSplit;
    
    // Create infrastructure
    let infra = MLInfrastructure::new(config).await?;
    
    // Load data
    let data_loader = DataLoader::new(config.data.clone().into());
    let data = data_loader.load(&config.data.source_path).await?;
    
    // Train multiple models
    let models = vec![
        ModelType::Transformer,
        ModelType::LSTM,
        ModelType::XGBoost,
        ModelType::LightGBM,
    ];
    
    for model_type in models {
        println!("Training {} with CV...", model_type.name());
        
        let model_id = infra.train_model(
            model_type,
            data.clone(),
            &format!("{}_cv_experiment", model_type.name())
        ).await?;
        
        println!("{} trained. ID: {}", model_type.name(), model_id);
    }
    
    // Get experiment history
    let experiments = infra.get_experiments().await?;
    println!("Total experiments: {}", experiments.len());
    
    Ok(())
}

/// Train with orchestrator
async fn train_with_orchestrator(config: TrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training with orchestrator...");
    
    // Create orchestrator
    let orchestrator = TrainingOrchestrator::new(4); // Max 4 concurrent jobs
    
    // Submit multiple training jobs
    let job_ids = vec![];
    
    for model_type in &[ModelType::Transformer, ModelType::LSTM, ModelType::XGBoost] {
        let job = TrainingJob {
            id: uuid::Uuid::new_v4().to_string(),
            model_type: *model_type,
            data_path: config.data.source_path.clone(),
            config: config.clone(),
            status: JobStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
        };
        
        let job_id = orchestrator.submit_job(job).await?;
        job_ids.push(job_id);
        println!("Submitted job for {}", model_type.name());
    }
    
    // Monitor job progress
    for job_id in &job_ids {
        loop {
            match orchestrator.get_job_status(job_id).await {
                Some(JobStatus::Completed) => {
                    println!("Job {} completed", job_id);
                    break;
                }
                Some(JobStatus::Failed) => {
                    println!("Job {} failed", job_id);
                    break;
                }
                Some(status) => {
                    println!("Job {} status: {:?}", job_id, status);
                }
                None => {
                    println!("Job {} not found", job_id);
                    break;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }
    
    Ok(())
}

/// Train with hyperparameter optimization
async fn train_with_hpo(mut config: TrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training with hyperparameter optimization...");
    
    // Enable HPO
    config.optimization.method = crate::config::OptimizationMethod::Bayesian;
    config.optimization.n_trials = 20;
    config.optimization.timeout = Some(3600); // 1 hour
    
    // Create infrastructure
    let infra = MLInfrastructure::new(config).await?;
    
    // Load data
    let data_loader = DataLoader::new(config.data.clone().into());
    let data = data_loader.load(&config.data.source_path).await?;
    
    // Train with HPO
    let model_id = infra.train_model(
        ModelType::XGBoost,
        data,
        "xgboost_hpo_experiment"
    ).await?;
    
    println!("HPO completed. Best model ID: {}", model_id);
    
    Ok(())
}

/// Model type extension
trait ModelTypeExt {
    fn name(&self) -> &'static str;
}

impl ModelTypeExt for ModelType {
    fn name(&self) -> &'static str {
        match self {
            ModelType::Transformer => "transformer",
            ModelType::LSTM => "lstm",
            ModelType::XGBoost => "xgboost",
            ModelType::LightGBM => "lightgbm",
            ModelType::NeuralNetwork => "neural_network",
            ModelType::Ensemble => "ensemble",
        }
    }
}