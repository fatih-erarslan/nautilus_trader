//! Pooling Layers for Hierarchical Time Series Processing

use ndarray::{Array3, s};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingType {
    Max,
    Average,
    Adaptive,
    Weighted { weights: Vec<f64> },
    Attention,
}

/// Pooling layer for downsampling time series
#[derive(Debug, Clone)]
pub struct PoolingLayer {
    factor: usize,
    pooling_type: PoolingType,
    learned_weights: Option<Vec<f64>>,
}

impl PoolingLayer {
    pub fn new(factor: usize, pooling_type: PoolingType) -> Self {
        let learned_weights = match &pooling_type {
            PoolingType::Adaptive | PoolingType::Attention => {
                Some(vec![1.0 / factor as f64; factor])
            }
            _ => None,
        };
        
        Self {
            factor,
            pooling_type,
            learned_weights,
        }
    }
    
    pub fn forward(&self, input: &Array3<f64>) -> Result<Array3<f64>, PoolingError> {
        let (batch_size, seq_len, features) = input.shape();
        
        if seq_len % self.factor != 0 {
            return Err(PoolingError::InvalidSequenceLength {
                seq_len: seq_len,
                factor: self.factor,
            });
        }
        
        let pooled_len = seq_len / self.factor;
        let mut output = Array3::zeros((batch_size, pooled_len, features));
        
        match &self.pooling_type {
            PoolingType::Max => self.max_pool(input, &mut output)?,
            PoolingType::Average => self.average_pool(input, &mut output)?,
            PoolingType::Adaptive => self.adaptive_pool(input, &mut output)?,
            PoolingType::Weighted { weights } => {
                self.weighted_pool(input, &mut output, weights)?
            }
            PoolingType::Attention => self.attention_pool(input, &mut output)?,
        }
        
        Ok(output)
    }
    
    fn max_pool(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> Result<(), PoolingError> {
        let (batch_size, _, features) = input.shape();
        let pooled_len = output.shape()[1];
        
        for b in 0..batch_size {
            for t in 0..pooled_len {
                for f in 0..features {
                    let start_idx = t * self.factor;
                    let end_idx = start_idx + self.factor;
                    
                    let max_val = input
                        .slice(s![b, start_idx..end_idx, f])
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    
                    output[[b, t, f]] = max_val;
                }
            }
        }
        
        Ok(())
    }
    
    fn average_pool(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> Result<(), PoolingError> {
        let (batch_size, _, features) = input.shape();
        let pooled_len = output.shape()[1];
        
        for b in 0..batch_size {
            for t in 0..pooled_len {
                for f in 0..features {
                    let start_idx = t * self.factor;
                    let end_idx = start_idx + self.factor;
                    
                    let sum: f64 = input
                        .slice(s![b, start_idx..end_idx, f])
                        .sum();
                    
                    output[[b, t, f]] = sum / self.factor as f64;
                }
            }
        }
        
        Ok(())
    }
    
    fn adaptive_pool(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> Result<(), PoolingError> {
        // Use learned weights for adaptive pooling
        let weights = self.learned_weights.as_ref()
            .ok_or(PoolingError::MissingWeights)?;
        
        self.weighted_pool(input, output, weights)
    }
    
    fn weighted_pool(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        weights: &[f64],
    ) -> Result<(), PoolingError> {
        if weights.len() != self.factor {
            return Err(PoolingError::WeightSizeMismatch {
                expected: self.factor,
                actual: weights.len(),
            });
        }
        
        let (batch_size, _, features) = input.shape();
        let pooled_len = output.shape()[1];
        
        for b in 0..batch_size {
            for t in 0..pooled_len {
                for f in 0..features {
                    let start_idx = t * self.factor;
                    let mut weighted_sum = 0.0;
                    
                    for i in 0..self.factor {
                        weighted_sum += input[[b, start_idx + i, f]] * weights[i];
                    }
                    
                    output[[b, t, f]] = weighted_sum;
                }
            }
        }
        
        Ok(())
    }
    
    fn attention_pool(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> Result<(), PoolingError> {
        let (batch_size, _, features) = input.shape();
        let pooled_len = output.shape()[1];
        
        for b in 0..batch_size {
            for t in 0..pooled_len {
                for f in 0..features {
                    let start_idx = t * self.factor;
                    let end_idx = start_idx + self.factor;
                    
                    // Compute attention scores based on values
                    let values: Vec<f64> = (start_idx..end_idx)
                        .map(|i| input[[b, i, f]])
                        .collect();
                    
                    let scores = self.compute_attention_scores(&values);
                    
                    // Apply attention-weighted pooling
                    let mut weighted_sum = 0.0;
                    for (i, &score) in scores.iter().enumerate() {
                        weighted_sum += values[i] * score;
                    }
                    
                    output[[b, t, f]] = weighted_sum;
                }
            }
        }
        
        Ok(())
    }
    
    fn compute_attention_scores(&self, values: &[f64]) -> Vec<f64> {
        // Simple attention based on value magnitude
        let sum_exp: f64 = values.iter()
            .map(|&v| v.exp())
            .sum();
        
        values.iter()
            .map(|&v| v.exp() / sum_exp)
            .collect()
    }
    
    pub fn update_weights(&mut self, new_weights: Vec<f64>) -> Result<(), PoolingError> {
        if new_weights.len() != self.factor {
            return Err(PoolingError::WeightSizeMismatch {
                expected: self.factor,
                actual: new_weights.len(),
            });
        }
        
        self.learned_weights = Some(new_weights);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PoolingError {
    #[error("Invalid sequence length {seq_len} for pooling factor {factor}")]
    InvalidSequenceLength { seq_len: usize, factor: usize },
    
    #[error("Missing weights for adaptive pooling")]
    MissingWeights,
    
    #[error("Weight size mismatch: expected {expected}, got {actual}")]
    WeightSizeMismatch { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pooling_operations() {
        // Test implementation
    }
}