//! Multi-asset batch processing

use crate::config::BatchConfig;
use crate::models::Model;
use crate::{Result, NeuralForecastError};
use ndarray::Array3;
use rayon::prelude::*;
use std::sync::Arc;

/// Batch processor for multiple assets
#[derive(Debug)]
pub struct BatchProcessor {
    config: BatchConfig,
}

impl BatchProcessor {
    /// Create a new batch processor with the given configuration
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }
    
    /// Process a batch of inputs using the specified model
    pub async fn process_batch<M>(
        &self,
        model: Arc<M>,
        inputs: Vec<Array3<f32>>,
    ) -> Result<Vec<Array3<f32>>>
    where
        M: Model + Send + Sync,
    {
        if self.config.parallel_processing {
            self.process_parallel(model, inputs).await
        } else {
            self.process_sequential(model, inputs).await
        }
    }
    
    async fn process_parallel<M>(
        &self,
        model: Arc<M>,
        inputs: Vec<Array3<f32>>,
    ) -> Result<Vec<Array3<f32>>>
    where
        M: Model + Send + Sync,
    {
        // Split inputs into chunks for parallel processing
        let chunk_size = self.config.max_batch_size;
        let mut results = Vec::new();
        
        for chunk in inputs.chunks(chunk_size) {
            let chunk_results: Result<Vec<_>> = chunk
                .par_iter()
                .map(|input| {
                    let model_clone = model.clone();
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            model_clone.predict(input).await
                        })
                    })
                })
                .collect();
            
            results.extend(chunk_results?);
        }
        
        Ok(results)
    }
    
    async fn process_sequential<M>(
        &self,
        model: Arc<M>,
        inputs: Vec<Array3<f32>>,
    ) -> Result<Vec<Array3<f32>>>
    where
        M: Model + Send + Sync,
    {
        let mut results = Vec::new();
        
        for input in inputs {
            let prediction = model.predict(&input).await?;
            results.push(prediction);
        }
        
        Ok(results)
    }
}

// Re-export is already handled in the use statement above