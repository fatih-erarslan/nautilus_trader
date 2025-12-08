//! Experiment tracking module

use crate::{Result, TrainingError};
use crate::config::ExperimentConfig;
use crate::models::{ModelType, TrainingMetrics, MetricSet};
use sqlx::{SqlitePool, Row};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Initialize experiment tracking database
pub async fn initialize_db() -> Result<()> {
    // Database initialization happens when ExperimentTracker is created
    Ok(())
}

/// Experiment tracker for managing ML experiments
pub struct ExperimentTracker {
    config: Arc<ExperimentConfig>,
    db_pool: SqlitePool,
}

/// Experiment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Experiment ID
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Status
    pub status: ExperimentStatus,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Configuration
    pub config: Value,
    /// Metrics
    pub metrics: Option<ExperimentMetrics>,
    /// Tags
    pub tags: Vec<String>,
    /// Description
    pub description: Option<String>,
}

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Aborted
    Aborted,
}

/// Experiment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// Final training loss
    pub final_train_loss: f32,
    /// Final validation loss
    pub final_val_loss: f32,
    /// Best validation loss
    pub best_val_loss: f32,
    /// Best epoch
    pub best_epoch: usize,
    /// Total epochs
    pub total_epochs: usize,
    /// Training time (seconds)
    pub training_time: f64,
    /// Final metrics
    pub final_metrics: MetricSet,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f32>,
}

/// Run record for tracking individual training runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Run ID
    pub id: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Run name
    pub name: String,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Parameters
    pub parameters: Value,
    /// Metrics history
    pub metrics_history: Vec<RunMetric>,
    /// Artifacts
    pub artifacts: Vec<Artifact>,
}

/// Run metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetric {
    /// Step/epoch
    pub step: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f32,
}

/// Artifact record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact ID
    pub id: String,
    /// Run ID
    pub run_id: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Path
    pub path: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Metadata
    pub metadata: Value,
}

/// Artifact types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Model checkpoint
    Model,
    /// Training logs
    Logs,
    /// Plots/visualizations
    Plots,
    /// Dataset
    Dataset,
    /// Configuration
    Config,
    /// Other
    Other,
}

impl ExperimentTracker {
    /// Create new experiment tracker
    pub async fn new(config: Arc<ExperimentConfig>) -> Result<Self> {
        // Connect to database
        let db_pool = SqlitePool::connect(&config.database_url)
            .await
            .map_err(|e| TrainingError::Database(e))?;
        
        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&db_pool)
            .await
            .map_err(|e| TrainingError::Database(e))?;
        
        Ok(Self {
            config,
            db_pool,
        })
    }
    
    /// Start a new experiment
    pub async fn start_experiment(
        &self,
        name: &str,
        model_type: ModelType,
    ) -> Result<String> {
        let experiment_id = Uuid::new_v4().to_string();
        let full_name = format!("{}_{}", self.config.name_prefix, name);
        let now = Utc::now();
        
        // Insert into database
        sqlx::query(
            r#"
            INSERT INTO experiments (id, name, model_type, status, start_time, config)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#
        )
        .bind(&experiment_id)
        .bind(&full_name)
        .bind(serde_json::to_string(&model_type)?)
        .bind("running")
        .bind(now.timestamp())
        .bind("{}")
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        tracing::info!("Started experiment: {} ({})", full_name, experiment_id);
        
        Ok(experiment_id)
    }
    
    /// Log experiment configuration
    pub async fn log_config(
        &self,
        experiment_id: &str,
        config: Value,
    ) -> Result<()> {
        if !self.config.track_params {
            return Ok(());
        }
        
        sqlx::query(
            r#"
            UPDATE experiments SET config = ?1 WHERE id = ?2
            "#
        )
        .bind(serde_json::to_string(&config)?)
        .bind(experiment_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        Ok(())
    }
    
    /// Log metrics
    pub async fn log_metrics(
        &self,
        experiment_id: &str,
        metrics: &TrainingMetrics,
    ) -> Result<()> {
        if !self.config.track_metrics {
            return Ok(());
        }
        
        // Convert to experiment metrics
        let exp_metrics = ExperimentMetrics {
            final_train_loss: metrics.train_loss.last().copied().unwrap_or(0.0),
            final_val_loss: metrics.val_loss.last().copied().unwrap_or(0.0),
            best_val_loss: metrics.val_loss.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0),
            best_epoch: metrics.best_epoch,
            total_epochs: metrics.train_loss.len(),
            training_time: metrics.training_time_secs,
            final_metrics: metrics.val_metrics.last().cloned().unwrap_or_else(|| MetricSet {
                mse: 0.0,
                mae: 0.0,
                rmse: 0.0,
                mape: None,
                r2: None,
                custom: HashMap::new(),
            }),
            custom_metrics: HashMap::new(),
        };
        
        sqlx::query(
            r#"
            UPDATE experiments SET metrics = ?1 WHERE id = ?2
            "#
        )
        .bind(serde_json::to_string(&exp_metrics)?)
        .bind(experiment_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        Ok(())
    }
    
    /// Complete experiment
    pub async fn complete_experiment(&self, experiment_id: &str) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(
            r#"
            UPDATE experiments 
            SET status = 'completed', end_time = ?1 
            WHERE id = ?2
            "#
        )
        .bind(now.timestamp())
        .bind(experiment_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        tracing::info!("Completed experiment: {}", experiment_id);
        
        Ok(())
    }
    
    /// Fail experiment
    pub async fn fail_experiment(&self, experiment_id: &str, error: &str) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(
            r#"
            UPDATE experiments 
            SET status = 'failed', end_time = ?1, description = ?2 
            WHERE id = ?3
            "#
        )
        .bind(now.timestamp())
        .bind(error)
        .bind(experiment_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        tracing::error!("Failed experiment {}: {}", experiment_id, error);
        
        Ok(())
    }
    
    /// List experiments
    pub async fn list_experiments(&self) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, model_type, status, start_time, end_time, config, metrics, tags, description
            FROM experiments
            ORDER BY start_time DESC
            LIMIT 100
            "#
        )
        .fetch_all(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        let mut experiments = Vec::new();
        for row in rows {
            let experiment = Experiment {
                id: row.get("id"),
                name: row.get("name"),
                model_type: serde_json::from_str(row.get("model_type")).unwrap_or(ModelType::Transformer),
                status: match row.get::<String, _>("status").as_str() {
                    "running" => ExperimentStatus::Running,
                    "completed" => ExperimentStatus::Completed,
                    "failed" => ExperimentStatus::Failed,
                    _ => ExperimentStatus::Aborted,
                },
                start_time: DateTime::from_timestamp(row.get("start_time"), 0).unwrap_or_else(Utc::now),
                end_time: row.get::<Option<i64>, _>("end_time")
                    .and_then(|ts| DateTime::from_timestamp(ts, 0)),
                config: serde_json::from_str(row.get("config")).unwrap_or(Value::Null),
                metrics: row.get::<Option<String>, _>("metrics")
                    .and_then(|s| serde_json::from_str(&s).ok()),
                tags: row.get::<Option<String>, _>("tags")
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default(),
                description: row.get("description"),
            };
            experiments.push(experiment);
        }
        
        Ok(experiments)
    }
    
    /// Get experiment by ID
    pub async fn get_experiment(&self, experiment_id: &str) -> Result<Option<Experiment>> {
        let experiments = self.list_experiments().await?;
        Ok(experiments.into_iter().find(|e| e.id == experiment_id))
    }
    
    /// Add tags to experiment
    pub async fn add_tags(&self, experiment_id: &str, tags: Vec<String>) -> Result<()> {
        // Get current tags
        let current_tags = sqlx::query("SELECT tags FROM experiments WHERE id = ?1")
            .bind(experiment_id)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| TrainingError::Database(e))?
            .and_then(|row| row.get::<Option<String>, _>("tags"))
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        
        // Merge tags
        let mut all_tags = current_tags;
        all_tags.extend(tags);
        all_tags.sort();
        all_tags.dedup();
        
        // Update database
        sqlx::query("UPDATE experiments SET tags = ?1 WHERE id = ?2")
            .bind(serde_json::to_string(&all_tags)?)
            .bind(experiment_id)
            .execute(&self.db_pool)
            .await
            .map_err(|e| TrainingError::Database(e))?;
        
        Ok(())
    }
    
    /// Create a new run within an experiment
    pub async fn create_run(
        &self,
        experiment_id: &str,
        name: &str,
        parameters: Value,
    ) -> Result<String> {
        let run_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        sqlx::query(
            r#"
            INSERT INTO runs (id, experiment_id, name, start_time, parameters)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#
        )
        .bind(&run_id)
        .bind(experiment_id)
        .bind(name)
        .bind(now.timestamp())
        .bind(serde_json::to_string(&parameters)?)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        Ok(run_id)
    }
    
    /// Log run metric
    pub async fn log_run_metric(
        &self,
        run_id: &str,
        step: usize,
        name: &str,
        value: f32,
    ) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(
            r#"
            INSERT INTO run_metrics (run_id, step, timestamp, name, value)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#
        )
        .bind(run_id)
        .bind(step as i64)
        .bind(now.timestamp())
        .bind(name)
        .bind(value as f64)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        Ok(())
    }
    
    /// Log artifact
    pub async fn log_artifact(
        &self,
        run_id: &str,
        artifact_type: ArtifactType,
        path: &std::path::Path,
        metadata: Value,
    ) -> Result<String> {
        if !self.config.track_artifacts {
            return Ok(String::new());
        }
        
        let artifact_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        // Get file size
        let size_bytes = tokio::fs::metadata(path)
            .await?
            .len();
        
        sqlx::query(
            r#"
            INSERT INTO artifacts (id, run_id, artifact_type, path, size_bytes, created_at, metadata)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#
        )
        .bind(&artifact_id)
        .bind(run_id)
        .bind(serde_json::to_string(&artifact_type)?)
        .bind(path.to_string_lossy())
        .bind(size_bytes as i64)
        .bind(now.timestamp())
        .bind(serde_json::to_string(&metadata)?)
        .execute(&self.db_pool)
        .await
        .map_err(|e| TrainingError::Database(e))?;
        
        Ok(artifact_id)
    }
    
    /// Compare experiments
    pub async fn compare_experiments(
        &self,
        experiment_ids: Vec<String>,
    ) -> Result<ExperimentComparison> {
        let mut experiments = Vec::new();
        
        for id in experiment_ids {
            if let Some(exp) = self.get_experiment(&id).await? {
                experiments.push(exp);
            }
        }
        
        // Extract metrics for comparison
        let mut comparison = ExperimentComparison {
            experiment_ids: experiments.iter().map(|e| e.id.clone()).collect(),
            metrics_comparison: HashMap::new(),
            best_experiment: None,
        };
        
        // Compare key metrics
        for exp in &experiments {
            if let Some(metrics) = &exp.metrics {
                comparison.metrics_comparison.insert(
                    exp.id.clone(),
                    vec![
                        ("final_val_loss".to_string(), metrics.final_val_loss),
                        ("best_val_loss".to_string(), metrics.best_val_loss),
                        ("training_time".to_string(), metrics.training_time as f32),
                        ("mae".to_string(), metrics.final_metrics.mae),
                        ("rmse".to_string(), metrics.final_metrics.rmse),
                    ].into_iter().collect(),
                );
            }
        }
        
        // Find best experiment by validation loss
        comparison.best_experiment = experiments
            .iter()
            .filter_map(|e| e.metrics.as_ref().map(|m| (e.id.clone(), m.best_val_loss)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(id, _)| id);
        
        Ok(comparison)
    }
}

/// Experiment comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentComparison {
    /// Experiment IDs being compared
    pub experiment_ids: Vec<String>,
    /// Metrics comparison
    pub metrics_comparison: HashMap<String, HashMap<String, f32>>,
    /// Best experiment ID
    pub best_experiment: Option<String>,
}

// SQL migrations would be in migrations/ directory
// CREATE TABLE experiments (
//     id TEXT PRIMARY KEY,
//     name TEXT NOT NULL,
//     model_type TEXT NOT NULL,
//     status TEXT NOT NULL,
//     start_time INTEGER NOT NULL,
//     end_time INTEGER,
//     config TEXT,
//     metrics TEXT,
//     tags TEXT,
//     description TEXT
// );
//
// CREATE TABLE runs (
//     id TEXT PRIMARY KEY,
//     experiment_id TEXT NOT NULL,
//     name TEXT NOT NULL,
//     start_time INTEGER NOT NULL,
//     end_time INTEGER,
//     parameters TEXT,
//     FOREIGN KEY (experiment_id) REFERENCES experiments(id)
// );
//
// CREATE TABLE run_metrics (
//     run_id TEXT NOT NULL,
//     step INTEGER NOT NULL,
//     timestamp INTEGER NOT NULL,
//     name TEXT NOT NULL,
//     value REAL NOT NULL,
//     FOREIGN KEY (run_id) REFERENCES runs(id)
// );
//
// CREATE TABLE artifacts (
//     id TEXT PRIMARY KEY,
//     run_id TEXT NOT NULL,
//     artifact_type TEXT NOT NULL,
//     path TEXT NOT NULL,
//     size_bytes INTEGER NOT NULL,
//     created_at INTEGER NOT NULL,
//     metadata TEXT,
//     FOREIGN KEY (run_id) REFERENCES runs(id)
// );

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_experiment_tracker() {
        let config = Arc::new(ExperimentConfig {
            database_url: "sqlite::memory:".to_string(),
            name_prefix: "test".to_string(),
            track_params: true,
            track_metrics: true,
            track_artifacts: true,
            log_frequency: 10,
        });
        
        let tracker = ExperimentTracker::new(config).await;
        assert!(tracker.is_ok());
    }
}