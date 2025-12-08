//! Data preprocessing for neural forecasting

use crate::config::PreprocessingConfig;
use crate::{Result, NeuralForecastError};
use ndarray::Array3;

/// Data preprocessor for financial time series
#[derive(Debug)]
pub struct Preprocessor {
    config: PreprocessingConfig,
}

impl Preprocessor {
    /// Create a new preprocessor with the given configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }
    
    /// Preprocess input data according to the configuration
    pub fn preprocess(&self, data: &Array3<f32>) -> Result<Array3<f32>> {
        let mut processed = data.clone();
        
        // Apply normalization
        match self.config.normalization.as_str() {
            "zscore" => self.zscore_normalize(&mut processed)?,
            "minmax" => self.minmax_normalize(&mut processed)?,
            _ => {},
        }
        
        Ok(processed)
    }
    
    fn zscore_normalize(&self, data: &mut Array3<f32>) -> Result<()> {
        // Z-score normalization
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        
        if std > 1e-8 {
            data.mapv_inplace(|x| (x - mean) / std);
        }
        
        Ok(())
    }
    
    fn minmax_normalize(&self, data: &mut Array3<f32>) -> Result<()> {
        // Min-max normalization
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        if (max_val - min_val) > 1e-8 {
            data.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
        }
        
        Ok(())
    }
}

// Re-export is already handled in the use statement above