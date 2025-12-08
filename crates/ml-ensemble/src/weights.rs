use crate::types::{ModelType, ModelWeightsConfig};
use anyhow::Result;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info};

/// Performance record for weight calculation
#[derive(Debug, Clone)]
struct PerformanceRecord {
    timestamp: Instant,
    model_type: ModelType,
    prediction: f64,
    actual: f64,
    weight: f64,
}

/// Dynamic weight manager for ensemble models
pub struct WeightManager {
    config: ModelWeightsConfig,
    current_weights: HashMap<ModelType, f64>,
    performance_history: VecDeque<PerformanceRecord>,
    last_update: Instant,
}

impl WeightManager {
    /// Create new weight manager
    pub fn new(config: ModelWeightsConfig) -> Self {
        let current_weights: HashMap<ModelType, f64> = config.initial_weights.iter().cloned().collect();
        
        Self {
            config,
            current_weights,
            performance_history: VecDeque::new(),
            last_update: Instant::now(),
        }
    }
    
    /// Get weights for selected models
    pub fn get_weights(&self, models: &[Arc<dyn super::ensemble::ModelPredictor>]) -> HashMap<ModelType, f64> {
        let mut weights = HashMap::new();
        let model_types: Vec<ModelType> = models.iter().map(|m| m.model_type()).collect();
        
        // Get current weights for selected models
        let total_weight: f64 = model_types.iter()
            .filter_map(|&mt| self.current_weights.get(&mt))
            .sum();
        
        // Normalize weights
        if total_weight > 0.0 {
            for model_type in model_types {
                if let Some(&weight) = self.current_weights.get(&model_type) {
                    weights.insert(model_type, weight / total_weight);
                }
            }
        } else {
            // Equal weights as fallback
            let equal_weight = 1.0 / models.len() as f64;
            for model_type in model_types {
                weights.insert(model_type, equal_weight);
            }
        }
        
        weights
    }
    
    /// Get all current weights
    pub fn get_all_weights(&self) -> HashMap<ModelType, f64> {
        self.current_weights.clone()
    }
    
    /// Update performance history
    pub fn update_performance(&mut self, actual_return: f64) -> Result<()> {
        // Clean old records
        let cutoff = Instant::now() - Duration::from_secs(self.config.lookback_period);
        self.performance_history.retain(|record| record.timestamp > cutoff);
        
        // Check if it's time to update weights
        if self.config.dynamic_weights && 
           self.last_update.elapsed() > Duration::from_secs(self.config.update_frequency) {
            self.update_weights()?;
            self.last_update = Instant::now();
        }
        
        Ok(())
    }
    
    /// Add prediction record
    pub fn add_prediction_record(
        &mut self,
        model_type: ModelType,
        prediction: f64,
        actual: f64,
        weight: f64,
    ) {
        let record = PerformanceRecord {
            timestamp: Instant::now(),
            model_type,
            prediction,
            actual,
            weight,
        };
        
        self.performance_history.push_back(record);
        
        // Limit history size
        if self.performance_history.len() > 10000 {
            self.performance_history.pop_front();
        }
    }
    
    /// Update weights based on recent performance
    fn update_weights(&mut self) -> Result<()> {
        if self.performance_history.len() < 100 {
            debug!("Not enough history for weight update: {} records", self.performance_history.len());
            return Ok(());
        }
        
        // Calculate performance metrics for each model
        let mut model_performances: HashMap<ModelType, ModelPerformanceStats> = HashMap::new();
        
        for record in &self.performance_history {
            let stats = model_performances.entry(record.model_type).or_default();
            let error = (record.prediction - record.actual).abs();
            stats.add_sample(error, record.prediction, record.actual);
        }
        
        // Calculate new weights based on performance
        let mut new_weights: HashMap<ModelType, f64> = HashMap::new();
        let mut total_score = 0.0;
        
        for (model_type, stats) in &model_performances {
            if stats.count > 10 {
                // Score based on multiple factors
                let accuracy_score = 1.0 / (1.0 + stats.mean_error());
                let sharpe_score = stats.sharpe_ratio().max(0.0);
                let consistency_score = 1.0 / (1.0 + stats.std_error());
                
                // Combined score with different weightings
                let score = 0.5 * accuracy_score + 0.3 * sharpe_score + 0.2 * consistency_score;
                
                new_weights.insert(*model_type, score);
                total_score += score;
            }
        }
        
        // Normalize weights
        if total_score > 0.0 {
            for weight in new_weights.values_mut() {
                *weight /= total_score;
            }
            
            // Apply momentum to prevent rapid changes
            let momentum = 0.8; // Keep 80% of old weights
            for (model_type, new_weight) in new_weights {
                if let Some(old_weight) = self.current_weights.get_mut(&model_type) {
                    *old_weight = momentum * *old_weight + (1.0 - momentum) * new_weight;
                } else {
                    self.current_weights.insert(model_type, new_weight);
                }
            }
            
            // Ensure minimum weight for all models
            let min_weight = 0.05;
            let num_models = self.current_weights.len() as f64;
            
            for weight in self.current_weights.values_mut() {
                if *weight < min_weight {
                    *weight = min_weight;
                }
            }
            
            // Re-normalize
            let total: f64 = self.current_weights.values().sum();
            for weight in self.current_weights.values_mut() {
                *weight /= total;
            }
            
            info!("Updated model weights: {:?}", self.current_weights);
        }
        
        Ok(())
    }
}

/// Performance statistics for a model
#[derive(Debug, Default)]
struct ModelPerformanceStats {
    count: usize,
    sum_error: f64,
    sum_squared_error: f64,
    sum_returns: f64,
    sum_squared_returns: f64,
    predictions: Vec<f64>,
    actuals: Vec<f64>,
}

impl ModelPerformanceStats {
    fn add_sample(&mut self, error: f64, prediction: f64, actual: f64) {
        self.count += 1;
        self.sum_error += error;
        self.sum_squared_error += error * error;
        self.sum_returns += actual;
        self.sum_squared_returns += actual * actual;
        self.predictions.push(prediction);
        self.actuals.push(actual);
    }
    
    fn mean_error(&self) -> f64 {
        if self.count == 0 {
            return f64::MAX;
        }
        self.sum_error / self.count as f64
    }
    
    fn std_error(&self) -> f64 {
        if self.count < 2 {
            return f64::MAX;
        }
        let mean = self.mean_error();
        let variance = self.sum_squared_error / self.count as f64 - mean * mean;
        variance.sqrt()
    }
    
    fn sharpe_ratio(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        
        let mean_return = self.sum_returns / self.count as f64;
        let variance = self.sum_squared_returns / self.count as f64 - mean_return * mean_return;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            // Annualized Sharpe ratio assuming minute data
            (mean_return / std_dev) * (252.0 * 6.5 * 60.0).sqrt()
        } else {
            0.0
        }
    }
}