//! Pipeline orchestration and coordination

use crate::{DataPipeline, types::DataItem, ComponentHealth};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error};

/// Pipeline orchestrator
pub struct PipelineOrchestrator {
    pipelines: Arc<RwLock<Vec<Arc<DataPipeline>>>>,
    metrics: Arc<RwLock<OrchestrationMetrics>>,
}

/// Orchestration metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OrchestrationMetrics {
    pub active_pipelines: u32,
    pub total_processed: u64,
    pub average_throughput: f64,
    pub error_rate: f64,
}

impl PipelineOrchestrator {
    pub fn new() -> Self {
        Self {
            pipelines: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(OrchestrationMetrics::default())),
        }
    }

    pub async fn add_pipeline(&self, pipeline: Arc<DataPipeline>) {
        let mut pipelines = self.pipelines.write().await;
        pipelines.push(pipeline);
        
        let mut metrics = self.metrics.write().await;
        metrics.active_pipelines = pipelines.len() as u32;
    }

    pub async fn process_data(&self, data: DataItem) -> anyhow::Result<Vec<crate::fusion::ProcessedData>> {
        let pipelines = self.pipelines.read().await;
        let mut results = Vec::new();
        
        for pipeline in pipelines.iter() {
            match pipeline.process_data(data.clone()).await {
                Ok(result) => results.push(result),
                Err(e) => error!("Pipeline processing error: {}", e),
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_processed += 1;
        }
        
        Ok(results)
    }

    pub async fn health_check(&self) -> anyhow::Result<ComponentHealth> {
        Ok(ComponentHealth::Healthy)
    }
}