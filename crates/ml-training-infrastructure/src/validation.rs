//! Cross-validation strategies for time series

use crate::{Result, TrainingError};
use crate::config::{ValidationConfig, CVStrategy, MetricType};
use crate::data::TrainingData;
use crate::models::{Model, calculate_metrics, MetricSet};
use crate::training::CVFoldResult;
use ndarray::{Array3, s};
use std::sync::Arc;
use rayon::prelude::*;

/// Cross-validator for time series data
pub struct CrossValidator {
    config: ValidationConfig,
}

impl CrossValidator {
    /// Create new cross-validator
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }
    
    /// Perform cross-validation
    pub async fn validate(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        training_params: &crate::config::TrainingParams,
    ) -> Result<Vec<CVFoldResult>> {
        match self.config.cv_strategy {
            CVStrategy::TimeSeriesSplit => {
                self.time_series_split(model, data, training_params).await
            }
            CVStrategy::PurgedKFold => {
                self.purged_kfold(model, data, training_params).await
            }
            CVStrategy::WalkForward => {
                self.walk_forward_analysis(model, data, training_params).await
            }
            CVStrategy::CombinatorialPurged => {
                self.combinatorial_purged_cv(model, data, training_params).await
            }
        }
    }
    
    /// Time series split cross-validation
    async fn time_series_split(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        training_params: &crate::config::TrainingParams,
    ) -> Result<Vec<CVFoldResult>> {
        let mut results = Vec::new();
        let n_samples = data.x_train.shape()[0] + data.x_val.shape()[0] + data.x_test.shape()[0];
        
        // Combine all data
        let all_x = self.combine_arrays(&[&data.x_train, &data.x_val, &data.x_test])?;
        let all_y = self.combine_arrays(&[&data.y_train, &data.y_val, &data.y_test])?;
        
        // Calculate fold sizes
        let min_train_size = n_samples / (self.config.n_folds + 1);
        let fold_size = (n_samples - min_train_size) / self.config.n_folds;
        
        for fold in 0..self.config.n_folds {
            tracing::info!("Processing fold {}/{}", fold + 1, self.config.n_folds);
            
            // Calculate indices
            let train_end = min_train_size + fold * fold_size;
            let val_start = train_end + self.config.gap;
            let val_end = val_start + fold_size;
            
            if val_end > n_samples {
                break;
            }
            
            // Create fold data
            let fold_x_train = all_x.slice(s![..train_end, .., ..]).to_owned();
            let fold_y_train = all_y.slice(s![..train_end, .., ..]).to_owned();
            
            let fold_x_val = all_x.slice(s![val_start..val_end, .., ..]).to_owned();
            let fold_y_val = all_y.slice(s![val_start..val_end, .., ..]).to_owned();
            
            // Create fold training data
            let fold_data = TrainingData {
                x_train: fold_x_train,
                y_train: fold_y_train,
                x_val: fold_x_val.clone(),
                y_val: fold_y_val.clone(),
                x_test: fold_x_val.clone(), // Use validation as test for CV
                y_test: fold_y_val.clone(),
                feature_names: data.feature_names.clone(),
                timestamps: data.timestamps.clone(),
                assets: data.assets.clone(),
                normalization: data.normalization.clone(),
            };
            
            // Train model on fold
            let metrics = model.train(&fold_data, training_params).await?;
            
            // Evaluate on validation set
            let val_predictions = model.predict(&fold_x_val).await?;
            let val_metrics = calculate_metrics(&val_predictions, &fold_y_val)?;
            
            // Calculate additional financial metrics if requested
            let enhanced_val_metrics = self.calculate_financial_metrics(
                &val_predictions,
                &fold_y_val,
                val_metrics,
            )?;
            
            results.push(CVFoldResult {
                fold,
                train_metrics: metrics.train_metrics.last().cloned().unwrap_or_default(),
                val_metrics: enhanced_val_metrics,
                test_metrics: None,
            });
        }
        
        Ok(results)
    }
    
    /// Purged K-fold cross-validation
    async fn purged_kfold(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        training_params: &crate::config::TrainingParams,
    ) -> Result<Vec<CVFoldResult>> {
        let mut results = Vec::new();
        let n_samples = data.x_train.shape()[0] + data.x_val.shape()[0] + data.x_test.shape()[0];
        
        // Combine all data
        let all_x = self.combine_arrays(&[&data.x_train, &data.x_val, &data.x_test])?;
        let all_y = self.combine_arrays(&[&data.y_train, &data.y_val, &data.y_test])?;
        
        // Create folds with purging
        let fold_size = n_samples / self.config.n_folds;
        let purge_gap = self.config.gap;
        
        for fold in 0..self.config.n_folds {
            tracing::info!("Processing purged fold {}/{}", fold + 1, self.config.n_folds);
            
            // Calculate fold boundaries
            let test_start = fold * fold_size;
            let test_end = (fold + 1) * fold_size;
            
            // Create training indices with purging
            let mut train_indices = Vec::new();
            for i in 0..n_samples {
                if i < test_start - purge_gap || i >= test_end + purge_gap {
                    train_indices.push(i);
                }
            }
            
            // Create test indices
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            // Create fold data
            let fold_x_train = self.select_indices(&all_x, &train_indices)?;
            let fold_y_train = self.select_indices(&all_y, &train_indices)?;
            
            let fold_x_test = self.select_indices(&all_x, &test_indices)?;
            let fold_y_test = self.select_indices(&all_y, &test_indices)?;
            
            // Split training data for validation
            let val_split_idx = (fold_x_train.shape()[0] as f32 * 0.8) as usize;
            let fold_x_val = fold_x_train.slice(s![val_split_idx.., .., ..]).to_owned();
            let fold_y_val = fold_y_train.slice(s![val_split_idx.., .., ..]).to_owned();
            let fold_x_train = fold_x_train.slice(s![..val_split_idx, .., ..]).to_owned();
            let fold_y_train = fold_y_train.slice(s![..val_split_idx, .., ..]).to_owned();
            
            // Create fold training data
            let fold_data = TrainingData {
                x_train: fold_x_train,
                y_train: fold_y_train,
                x_val: fold_x_val,
                y_val: fold_y_val,
                x_test: fold_x_test.clone(),
                y_test: fold_y_test.clone(),
                feature_names: data.feature_names.clone(),
                timestamps: data.timestamps.clone(),
                assets: data.assets.clone(),
                normalization: data.normalization.clone(),
            };
            
            // Train model on fold
            let metrics = model.train(&fold_data, training_params).await?;
            
            // Evaluate on test set
            let test_predictions = model.predict(&fold_x_test).await?;
            let test_metrics = calculate_metrics(&test_predictions, &fold_y_test)?;
            
            results.push(CVFoldResult {
                fold,
                train_metrics: metrics.train_metrics.last().cloned().unwrap_or_default(),
                val_metrics: metrics.val_metrics.last().cloned().unwrap_or_default(),
                test_metrics: Some(test_metrics),
            });
        }
        
        Ok(results)
    }
    
    /// Walk-forward analysis
    async fn walk_forward_analysis(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        training_params: &crate::config::TrainingParams,
    ) -> Result<Vec<CVFoldResult>> {
        let mut results = Vec::new();
        let n_samples = data.x_train.shape()[0] + data.x_val.shape()[0] + data.x_test.shape()[0];
        
        // Combine all data
        let all_x = self.combine_arrays(&[&data.x_train, &data.x_val, &data.x_test])?;
        let all_y = self.combine_arrays(&[&data.y_train, &data.y_val, &data.y_test])?;
        
        // Walk-forward parameters
        let initial_train_size = n_samples / (self.config.n_folds + 1);
        let step_size = (n_samples - initial_train_size) / self.config.n_folds;
        
        for fold in 0..self.config.n_folds {
            tracing::info!("Walk-forward fold {}/{}", fold + 1, self.config.n_folds);
            
            // Calculate window boundaries
            let train_start = if fold == 0 { 0 } else { fold * step_size };
            let train_end = initial_train_size + fold * step_size;
            let test_start = train_end + self.config.gap;
            let test_end = test_start + step_size;
            
            if test_end > n_samples {
                break;
            }
            
            // Create fold data
            let fold_x_train = all_x.slice(s![train_start..train_end, .., ..]).to_owned();
            let fold_y_train = all_y.slice(s![train_start..train_end, .., ..]).to_owned();
            
            let fold_x_test = all_x.slice(s![test_start..test_end, .., ..]).to_owned();
            let fold_y_test = all_y.slice(s![test_start..test_end, .., ..]).to_owned();
            
            // Create validation set from end of training data
            let val_size = (fold_x_train.shape()[0] as f32 * 0.2) as usize;
            let val_start = fold_x_train.shape()[0] - val_size;
            
            let fold_x_val = fold_x_train.slice(s![val_start.., .., ..]).to_owned();
            let fold_y_val = fold_y_train.slice(s![val_start.., .., ..]).to_owned();
            let fold_x_train = fold_x_train.slice(s![..val_start, .., ..]).to_owned();
            let fold_y_train = fold_y_train.slice(s![..val_start, .., ..]).to_owned();
            
            // Create fold training data
            let fold_data = TrainingData {
                x_train: fold_x_train,
                y_train: fold_y_train,
                x_val: fold_x_val,
                y_val: fold_y_val,
                x_test: fold_x_test.clone(),
                y_test: fold_y_test.clone(),
                feature_names: data.feature_names.clone(),
                timestamps: data.timestamps.clone(),
                assets: data.assets.clone(),
                normalization: data.normalization.clone(),
            };
            
            // Train model on fold
            let metrics = model.train(&fold_data, training_params).await?;
            
            // Evaluate on test set
            let test_predictions = model.predict(&fold_x_test).await?;
            let test_metrics = calculate_metrics(&test_predictions, &fold_y_test)?;
            
            // Calculate financial metrics
            let enhanced_test_metrics = self.calculate_financial_metrics(
                &test_predictions,
                &fold_y_test,
                test_metrics,
            )?;
            
            results.push(CVFoldResult {
                fold,
                train_metrics: metrics.train_metrics.last().cloned().unwrap_or_default(),
                val_metrics: metrics.val_metrics.last().cloned().unwrap_or_default(),
                test_metrics: Some(enhanced_test_metrics),
            });
        }
        
        Ok(results)
    }
    
    /// Combinatorial purged cross-validation
    async fn combinatorial_purged_cv(
        &self,
        model: &mut dyn Model,
        data: &TrainingData,
        training_params: &crate::config::TrainingParams,
    ) -> Result<Vec<CVFoldResult>> {
        // This is a simplified version - full implementation would involve
        // more sophisticated group selection and purging
        self.purged_kfold(model, data, training_params).await
    }
    
    /// Combine multiple arrays along the first axis
    fn combine_arrays(&self, arrays: &[&Array3<f32>]) -> Result<Array3<f32>> {
        if arrays.is_empty() {
            return Err(TrainingError::Validation("No arrays to combine".to_string()));
        }
        
        let total_size = arrays.iter().map(|a| a.shape()[0]).sum();
        let shape1 = arrays[0].shape()[1];
        let shape2 = arrays[0].shape()[2];
        
        let mut combined = Array3::<f32>::zeros((total_size, shape1, shape2));
        let mut offset = 0;
        
        for array in arrays {
            let size = array.shape()[0];
            combined.slice_mut(s![offset..offset + size, .., ..]).assign(array);
            offset += size;
        }
        
        Ok(combined)
    }
    
    /// Select indices from array
    fn select_indices(&self, array: &Array3<f32>, indices: &[usize]) -> Result<Array3<f32>> {
        let shape1 = array.shape()[1];
        let shape2 = array.shape()[2];
        let mut selected = Array3::<f32>::zeros((indices.len(), shape1, shape2));
        
        for (i, &idx) in indices.iter().enumerate() {
            selected.slice_mut(s![i, .., ..]).assign(&array.slice(s![idx, .., ..]));
        }
        
        Ok(selected)
    }
    
    /// Calculate financial metrics
    fn calculate_financial_metrics(
        &self,
        predictions: &Array3<f32>,
        targets: &Array3<f32>,
        mut base_metrics: MetricSet,
    ) -> Result<MetricSet> {
        // Calculate requested metrics
        for metric_type in &self.config.metrics {
            match metric_type {
                MetricType::SharpeRatio => {
                    let sharpe = self.calculate_sharpe_ratio(predictions, targets)?;
                    base_metrics.custom.insert("sharpe_ratio".to_string(), sharpe);
                }
                MetricType::MaxDrawdown => {
                    let mdd = self.calculate_max_drawdown(predictions)?;
                    base_metrics.custom.insert("max_drawdown".to_string(), mdd);
                }
                MetricType::HitRate => {
                    let hit_rate = self.calculate_hit_rate(predictions, targets)?;
                    base_metrics.custom.insert("hit_rate".to_string(), hit_rate);
                }
                MetricType::ProfitFactor => {
                    let pf = self.calculate_profit_factor(predictions, targets)?;
                    base_metrics.custom.insert("profit_factor".to_string(), pf);
                }
                _ => {} // Base metrics already calculated
            }
        }
        
        Ok(base_metrics)
    }
    
    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(
        &self,
        predictions: &Array3<f32>,
        _targets: &Array3<f32>,
    ) -> Result<f32> {
        // Simplified Sharpe ratio calculation
        let returns = self.calculate_returns(predictions)?;
        let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
        let std_return = self.calculate_std(&returns, mean_return)?;
        
        // Annualized Sharpe ratio (assuming daily data)
        let sharpe = if std_return > 1e-8 {
            mean_return / std_return * (252.0_f32).sqrt()
        } else {
            0.0
        };
        
        Ok(sharpe)
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, predictions: &Array3<f32>) -> Result<f32> {
        let prices = self.extract_prices(predictions)?;
        let mut peak = prices[0];
        let mut max_dd = 0.0;
        
        for &price in &prices[1..] {
            if price > peak {
                peak = price;
            }
            let dd = (peak - price) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        
        Ok(max_dd)
    }
    
    /// Calculate hit rate (directional accuracy)
    fn calculate_hit_rate(
        &self,
        predictions: &Array3<f32>,
        targets: &Array3<f32>,
    ) -> Result<f32> {
        let pred_flat = predictions.as_slice()
            .ok_or_else(|| TrainingError::Validation("Failed to flatten predictions".to_string()))?;
        let target_flat = targets.as_slice()
            .ok_or_else(|| TrainingError::Validation("Failed to flatten targets".to_string()))?;
        
        let mut hits = 0;
        let mut total = 0;
        
        for i in 1..pred_flat.len() {
            let pred_direction = pred_flat[i] - pred_flat[i-1];
            let target_direction = target_flat[i] - target_flat[i-1];
            
            if pred_direction * target_direction > 0.0 {
                hits += 1;
            }
            total += 1;
        }
        
        Ok(hits as f32 / total as f32)
    }
    
    /// Calculate profit factor
    fn calculate_profit_factor(
        &self,
        predictions: &Array3<f32>,
        targets: &Array3<f32>,
    ) -> Result<f32> {
        let pred_returns = self.calculate_returns(predictions)?;
        let target_returns = self.calculate_returns(targets)?;
        
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;
        
        for (pred, target) in pred_returns.iter().zip(target_returns.iter()) {
            let pnl = pred * target.signum(); // Profit if prediction matches target direction
            if pnl > 0.0 {
                gross_profit += pnl;
            } else {
                gross_loss += pnl.abs();
            }
        }
        
        Ok(if gross_loss > 1e-8 {
            gross_profit / gross_loss
        } else {
            f32::INFINITY
        })
    }
    
    /// Calculate returns from price series
    fn calculate_returns(&self, prices: &Array3<f32>) -> Result<Vec<f32>> {
        let price_vec = self.extract_prices(prices)?;
        let mut returns = Vec::with_capacity(price_vec.len() - 1);
        
        for i in 1..price_vec.len() {
            let ret = (price_vec[i] - price_vec[i-1]) / price_vec[i-1];
            returns.push(ret);
        }
        
        Ok(returns)
    }
    
    /// Extract price series from 3D array (using first feature of last timestep)
    fn extract_prices(&self, array: &Array3<f32>) -> Result<Vec<f32>> {
        let n_samples = array.shape()[0];
        let horizon = array.shape()[1];
        let mut prices = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            prices.push(array[[i, horizon - 1, 0]]);
        }
        
        Ok(prices)
    }
    
    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f32], mean: f32) -> Result<f32> {
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        Ok(variance.sqrt())
    }
}

/// Summary statistics for cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVSummary {
    /// Mean metrics across folds
    pub mean_metrics: MetricSet,
    /// Standard deviation of metrics
    pub std_metrics: MetricSet,
    /// Best fold index
    pub best_fold: usize,
    /// Worst fold index
    pub worst_fold: usize,
    /// Individual fold results
    pub fold_results: Vec<CVFoldResult>,
}

impl CVSummary {
    /// Create summary from fold results
    pub fn from_results(results: Vec<CVFoldResult>) -> Self {
        // Calculate mean metrics
        let n_folds = results.len() as f32;
        let mut mean_mse = 0.0;
        let mut mean_mae = 0.0;
        let mut mean_rmse = 0.0;
        
        for result in &results {
            mean_mse += result.val_metrics.mse / n_folds;
            mean_mae += result.val_metrics.mae / n_folds;
            mean_rmse += result.val_metrics.rmse / n_folds;
        }
        
        let mean_metrics = MetricSet {
            mse: mean_mse,
            mae: mean_mae,
            rmse: mean_rmse,
            mape: None,
            r2: None,
            custom: std::collections::HashMap::new(),
        };
        
        // Find best and worst folds by validation MSE
        let (best_fold, _) = results.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.val_metrics.mse.partial_cmp(&b.val_metrics.mse).unwrap()
            })
            .unwrap();
        
        let (worst_fold, _) = results.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.val_metrics.mse.partial_cmp(&b.val_metrics.mse).unwrap()
            })
            .unwrap();
        
        // Calculate standard deviation (simplified)
        let std_metrics = MetricSet {
            mse: 0.0, // TODO: Calculate actual std
            mae: 0.0,
            rmse: 0.0,
            mape: None,
            r2: None,
            custom: std::collections::HashMap::new(),
        };
        
        Self {
            mean_metrics,
            std_metrics,
            best_fold,
            worst_fold,
            fold_results: results,
        }
    }
}

use serde::{Serialize, Deserialize};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cross_validator_creation() {
        let config = ValidationConfig {
            cv_strategy: CVStrategy::TimeSeriesSplit,
            n_folds: 5,
            gap: 10,
            metrics: vec![MetricType::MSE, MetricType::MAE],
            walk_forward: false,
            purged: true,
        };
        
        let validator = CrossValidator::new(config);
        assert_eq!(validator.config.n_folds, 5);
    }
}