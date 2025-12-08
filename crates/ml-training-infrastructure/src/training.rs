//! Training pipeline module

use crate::{Result, TrainingError};
use crate::config::{TrainingConfig, TrainingParams};
use crate::data::{TrainingData, DataLoader};
use crate::models::{Model, ModelType, TrainingMetrics};
use crate::validation::CrossValidator;
use crate::optimization::HyperparameterOptimizer;
use std::sync::Arc;
use std::path::Path;
use tokio::sync::RwLock;
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Unified training pipeline for all models
pub struct TrainingPipeline {
    config: Arc<TrainingConfig>,
    data_loader: Arc<DataLoader>,
    cross_validator: Arc<CrossValidator>,
    optimizer: Arc<HyperparameterOptimizer>,
    checkpoints: Arc<DashMap<String, ModelCheckpoint>>,
    metrics_history: Arc<RwLock<Vec<PipelineMetrics>>>,
}

/// Model checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    /// Checkpoint ID
    pub id: String,
    /// Model type
    pub model_type: ModelType,
    /// Epoch number
    pub epoch: usize,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Checkpoint path
    pub path: std::path::PathBuf,
    /// Creation time
    pub created_at: DateTime<Utc>,
}

/// Pipeline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Pipeline run ID
    pub run_id: String,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Model type
    pub model_type: ModelType,
    /// Training configuration
    pub config: serde_json::Value,
    /// Final metrics
    pub final_metrics: Option<TrainingMetrics>,
    /// Cross-validation results
    pub cv_results: Option<Vec<CVFoldResult>>,
    /// Hyperparameter optimization results
    pub hpo_results: Option<HPOResults>,
}

/// Cross-validation fold result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVFoldResult {
    /// Fold number
    pub fold: usize,
    /// Training metrics
    pub train_metrics: crate::models::MetricSet,
    /// Validation metrics
    pub val_metrics: crate::models::MetricSet,
    /// Test metrics
    pub test_metrics: Option<crate::models::MetricSet>,
}

/// Hyperparameter optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPOResults {
    /// Best parameters found
    pub best_params: serde_json::Value,
    /// Best score achieved
    pub best_score: f32,
    /// Number of trials
    pub n_trials: usize,
    /// All trial results
    pub trials: Vec<HPOTrial>,
}

/// HPO trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPOTrial {
    /// Trial number
    pub trial_id: usize,
    /// Parameters tested
    pub params: serde_json::Value,
    /// Score achieved
    pub score: f32,
    /// Training time
    pub training_time: f64,
}

impl TrainingPipeline {
    /// Create new training pipeline
    pub async fn new(config: Arc<TrainingConfig>) -> Result<Self> {
        let data_loader = Arc::new(DataLoader::new(config.data.clone().into()));
        let cross_validator = Arc::new(CrossValidator::new(config.validation.clone()));
        let optimizer = Arc::new(HyperparameterOptimizer::new(config.optimization.clone()));
        
        Ok(Self {
            config,
            data_loader,
            cross_validator,
            optimizer,
            checkpoints: Arc::new(DashMap::new()),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Train a model with full pipeline support
    pub async fn train(
        &self,
        model: &mut dyn Model,
        data: TrainingData,
    ) -> Result<TrainingMetrics> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let start_time = Utc::now();
        
        tracing::info!("Starting training pipeline run: {}", run_id);
        
        // Initialize pipeline metrics
        let mut pipeline_metrics = PipelineMetrics {
            run_id: run_id.clone(),
            start_time,
            end_time: None,
            model_type: model.model_type(),
            config: serde_json::to_value(&self.config.training)?,
            final_metrics: None,
            cv_results: None,
            hpo_results: None,
        };
        
        // Perform hyperparameter optimization if enabled
        let final_params = if self.config.optimization.n_trials > 1 {
            tracing::info!("Starting hyperparameter optimization");
            let hpo_results = self.optimizer.optimize(model, &data).await?;
            pipeline_metrics.hpo_results = Some(hpo_results.clone());
            hpo_results.best_params
        } else {
            serde_json::to_value(&self.config.training)?
        };
        
        // Update model with best parameters
        let training_params: TrainingParams = serde_json::from_value(final_params)?;
        
        // Perform cross-validation if enabled
        if self.config.validation.n_folds > 1 {
            tracing::info!("Starting cross-validation with {} folds", self.config.validation.n_folds);
            let cv_results = self.cross_validator.validate(model, &data, &training_params).await?;
            pipeline_metrics.cv_results = Some(cv_results);
        }
        
        // Final training on full dataset
        tracing::info!("Starting final training on full dataset");
        let metrics = self.train_with_checkpointing(
            model,
            &data,
            &training_params,
            &run_id
        ).await?;
        
        // Update pipeline metrics
        pipeline_metrics.end_time = Some(Utc::now());
        pipeline_metrics.final_metrics = Some(metrics.clone());
        
        // Store metrics
        self.metrics_history.write().await.push(pipeline_metrics);
        
        tracing::info!("Training pipeline completed: {}", run_id);
        
        Ok(metrics)
    }
    
    /// Train with checkpointing
    async fn train_with_checkpointing(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        params: &TrainingParams,
        run_id: &str,
    ) -> Result<TrainingMetrics> {
        // Create checkpoint directory
        let checkpoint_dir = self.config.persistence.checkpoint_path.join(run_id);
        tokio::fs::create_dir_all(&checkpoint_dir).await?;
        
        // Custom training loop with checkpointing
        let mut best_metrics: Option<TrainingMetrics> = None;
        let mut best_val_loss = f32::INFINITY;
        
        // For now, delegate to model's train method
        // In a full implementation, we'd intercept at each epoch for checkpointing
        let metrics = model.train(data, params).await?;
        
        // Save final checkpoint
        let checkpoint_id = format!("{}_final", run_id);
        let checkpoint_path = checkpoint_dir.join(format!("{}.ckpt", checkpoint_id));
        
        model.save(&checkpoint_path).await?;
        
        let checkpoint = ModelCheckpoint {
            id: checkpoint_id.clone(),
            model_type: model.model_type(),
            epoch: metrics.best_epoch,
            metrics: metrics.clone(),
            path: checkpoint_path,
            created_at: Utc::now(),
        };
        
        self.checkpoints.insert(checkpoint_id, checkpoint);
        
        Ok(metrics)
    }
    
    /// Load checkpoint
    pub async fn load_checkpoint(
        &self,
        model: &mut dyn Model,
        checkpoint_id: &str,
    ) -> Result<()> {
        let checkpoint = self.checkpoints.get(checkpoint_id)
            .ok_or_else(|| TrainingError::Training(
                format!("Checkpoint {} not found", checkpoint_id)
            ))?;
        
        model.load(&checkpoint.path).await?;
        
        Ok(())
    }
    
    /// Get training history
    pub async fn get_history(&self) -> Vec<PipelineMetrics> {
        self.metrics_history.read().await.clone()
    }
    
    /// Clean up old checkpoints
    pub async fn cleanup_checkpoints(&self, keep_last: usize) -> Result<()> {
        let mut checkpoints: Vec<_> = self.checkpoints.iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        // Sort by creation time
        checkpoints.sort_by_key(|c| c.created_at);
        
        // Remove old checkpoints
        if checkpoints.len() > keep_last {
            let to_remove = checkpoints.len() - keep_last;
            for checkpoint in checkpoints.iter().take(to_remove) {
                // Remove from disk
                if checkpoint.path.exists() {
                    tokio::fs::remove_file(&checkpoint.path).await?;
                }
                // Remove from memory
                self.checkpoints.remove(&checkpoint.id);
            }
        }
        
        Ok(())
    }
}

/// Training orchestrator for managing multiple training jobs
pub struct TrainingOrchestrator {
    pipelines: DashMap<String, Arc<TrainingPipeline>>,
    active_jobs: Arc<RwLock<Vec<TrainingJob>>>,
    job_queue: Arc<RwLock<Vec<TrainingJob>>>,
    max_concurrent_jobs: usize,
}

/// Training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    /// Job ID
    pub id: String,
    /// Model type
    pub model_type: ModelType,
    /// Data path
    pub data_path: std::path::PathBuf,
    /// Configuration
    pub config: TrainingConfig,
    /// Status
    pub status: JobStatus,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Started at
    pub started_at: Option<DateTime<Utc>>,
    /// Completed at
    pub completed_at: Option<DateTime<Utc>>,
    /// Result
    pub result: Option<TrainingMetrics>,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Queued
    Queued,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

impl TrainingOrchestrator {
    /// Create new training orchestrator
    pub fn new(max_concurrent_jobs: usize) -> Self {
        Self {
            pipelines: DashMap::new(),
            active_jobs: Arc::new(RwLock::new(Vec::new())),
            job_queue: Arc::new(RwLock::new(Vec::new())),
            max_concurrent_jobs,
        }
    }
    
    /// Submit training job
    pub async fn submit_job(&self, job: TrainingJob) -> Result<String> {
        let job_id = job.id.clone();
        
        // Add to queue
        self.job_queue.write().await.push(job);
        
        // Try to start job if slots available
        self.process_queue().await?;
        
        Ok(job_id)
    }
    
    /// Process job queue
    async fn process_queue(&self) -> Result<()> {
        let mut active_jobs = self.active_jobs.write().await;
        let mut job_queue = self.job_queue.write().await;
        
        while active_jobs.len() < self.max_concurrent_jobs && !job_queue.is_empty() {
            if let Some(mut job) = job_queue.pop() {
                job.status = JobStatus::Running;
                job.started_at = Some(Utc::now());
                
                let job_id = job.id.clone();
                active_jobs.push(job.clone());
                
                // Spawn training task
                let orchestrator = self.clone();
                tokio::spawn(async move {
                    if let Err(e) = orchestrator.run_job(job).await {
                        tracing::error!("Job {} failed: {}", job_id, e);
                    }
                });
            }
        }
        
        Ok(())
    }
    
    /// Run training job
    async fn run_job(&self, mut job: TrainingJob) -> Result<()> {
        // Create pipeline for job
        let pipeline = Arc::new(TrainingPipeline::new(Arc::new(job.config.clone())).await?);
        self.pipelines.insert(job.id.clone(), pipeline.clone());
        
        // Load data
        let data = pipeline.data_loader.load(&job.data_path).await?;
        
        // Create model
        let mut model = crate::models::create_model(job.model_type, &job.config)?;
        
        // Train model
        match pipeline.train(model.as_mut(), data).await {
            Ok(metrics) => {
                job.status = JobStatus::Completed;
                job.result = Some(metrics);
            }
            Err(e) => {
                job.status = JobStatus::Failed;
                tracing::error!("Training failed for job {}: {}", job.id, e);
            }
        }
        
        job.completed_at = Some(Utc::now());
        
        // Update job status
        let mut active_jobs = self.active_jobs.write().await;
        active_jobs.retain(|j| j.id != job.id);
        
        // Process next job in queue
        drop(active_jobs);
        self.process_queue().await?;
        
        Ok(())
    }
    
    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        // Check active jobs
        let active_jobs = self.active_jobs.read().await;
        if let Some(job) = active_jobs.iter().find(|j| j.id == job_id) {
            return Some(job.status);
        }
        
        // Check queue
        let job_queue = self.job_queue.read().await;
        if let Some(job) = job_queue.iter().find(|j| j.id == job_id) {
            return Some(job.status);
        }
        
        None
    }
    
    /// Cancel job
    pub async fn cancel_job(&self, job_id: &str) -> Result<()> {
        // Remove from queue
        let mut job_queue = self.job_queue.write().await;
        job_queue.retain(|j| j.id != job_id);
        
        // TODO: Implement cancellation for running jobs
        
        Ok(())
    }
}

// Clone implementation for TrainingOrchestrator
impl Clone for TrainingOrchestrator {
    fn clone(&self) -> Self {
        Self {
            pipelines: self.pipelines.clone(),
            active_jobs: self.active_jobs.clone(),
            job_queue: self.job_queue.clone(),
            max_concurrent_jobs: self.max_concurrent_jobs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = Arc::new(TrainingConfig::default());
        let pipeline = TrainingPipeline::new(config).await;
        assert!(pipeline.is_ok());
    }
    
    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = TrainingOrchestrator::new(4);
        assert_eq!(orchestrator.max_concurrent_jobs, 4);
    }
}