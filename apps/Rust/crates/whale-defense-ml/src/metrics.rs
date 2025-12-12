//! Performance metrics and monitoring for whale detection
//! 
//! This module provides comprehensive metrics tracking including
//! inference timing, accuracy metrics, and performance monitoring.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Duration, Instant};
use crate::error::{Result, WhaleMLError};

/// Performance metrics for the whale detection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of predictions
    pub total_predictions: u64,
    /// True positives (correctly detected whales)
    pub true_positives: u64,
    /// False positives (false alarms)
    pub false_positives: u64,
    /// True negatives (correctly identified normal activity)
    pub true_negatives: u64,
    /// False negatives (missed whales)
    pub false_negatives: u64,
    /// Average inference time in microseconds
    pub avg_inference_time_us: f64,
    /// Minimum inference time in microseconds
    pub min_inference_time_us: u64,
    /// Maximum inference time in microseconds
    pub max_inference_time_us: u64,
    /// P95 inference time in microseconds
    pub p95_inference_time_us: u64,
    /// P99 inference time in microseconds
    pub p99_inference_time_us: u64,
    /// Number of times inference exceeded 500μs target
    pub inference_violations: u64,
}

impl PerformanceMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
            avg_inference_time_us: 0.0,
            min_inference_time_us: u64::MAX,
            max_inference_time_us: 0,
            p95_inference_time_us: 0,
            p99_inference_time_us: 0,
            inference_violations: 0,
        }
    }
    
    /// Calculate accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        
        let correct = self.true_positives + self.true_negatives;
        correct as f64 / self.total_predictions as f64
    }
    
    /// Calculate precision
    pub fn precision(&self) -> f64 {
        let predicted_positive = self.true_positives + self.false_positives;
        if predicted_positive == 0 {
            return 0.0;
        }
        
        self.true_positives as f64 / predicted_positive as f64
    }
    
    /// Calculate recall (sensitivity)
    pub fn recall(&self) -> f64 {
        let actual_positive = self.true_positives + self.false_negatives;
        if actual_positive == 0 {
            return 0.0;
        }
        
        self.true_positives as f64 / actual_positive as f64
    }
    
    /// Calculate F1 score
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        
        if precision + recall == 0.0 {
            return 0.0;
        }
        
        2.0 * (precision * recall) / (precision + recall)
    }
    
    /// Calculate specificity
    pub fn specificity(&self) -> f64 {
        let actual_negative = self.true_negatives + self.false_positives;
        if actual_negative == 0 {
            return 0.0;
        }
        
        self.true_negatives as f64 / actual_negative as f64
    }
    
    /// Get inference violation rate
    pub fn violation_rate(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        
        self.inference_violations as f64 / self.total_predictions as f64
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Inference timer for tracking individual inference times
pub struct InferenceTimer {
    start: Instant,
    target_us: u64,
}

impl InferenceTimer {
    /// Start a new inference timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            target_us: 500,  // 500μs target
        }
    }
    
    /// Start with custom target
    pub fn start_with_target(target_us: u64) -> Self {
        Self {
            start: Instant::now(),
            target_us,
        }
    }
    
    /// Stop the timer and return elapsed microseconds
    pub fn stop(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
    
    /// Check if target was met
    pub fn target_met(&self) -> bool {
        self.stop() <= self.target_us
    }
    
    /// Get remaining time before target violation (can be negative)
    pub fn remaining_us(&self) -> i64 {
        self.target_us as i64 - self.stop() as i64
    }
}

/// Metrics collector for concurrent metric updates
pub struct MetricsCollector {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    inference_times: Arc<RwLock<Vec<u64>>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
            inference_times: Arc::new(RwLock::new(Vec::with_capacity(10000))),
        }
    }
    
    /// Record a prediction result
    pub fn record_prediction(
        &self,
        predicted: bool,
        actual: bool,
        inference_time_us: u64,
    ) -> Result<()> {
        let mut metrics = self.metrics.write();
        
        metrics.total_predictions += 1;
        
        // Update confusion matrix
        match (predicted, actual) {
            (true, true) => metrics.true_positives += 1,
            (true, false) => metrics.false_positives += 1,
            (false, true) => metrics.false_negatives += 1,
            (false, false) => metrics.true_negatives += 1,
        }
        
        // Update inference time stats
        metrics.min_inference_time_us = metrics.min_inference_time_us.min(inference_time_us);
        metrics.max_inference_time_us = metrics.max_inference_time_us.max(inference_time_us);
        
        if inference_time_us > 500 {
            metrics.inference_violations += 1;
        }
        
        // Store inference time for percentile calculation
        self.inference_times.write().push(inference_time_us);
        
        Ok(())
    }
    
    /// Calculate and update percentile metrics
    pub fn update_percentiles(&self) -> Result<()> {
        let mut times = self.inference_times.write();
        if times.is_empty() {
            return Ok(());
        }
        
        // Sort for percentile calculation
        times.sort_unstable();
        
        let mut metrics = self.metrics.write();
        
        // Calculate average
        let sum: u64 = times.iter().sum();
        metrics.avg_inference_time_us = sum as f64 / times.len() as f64;
        
        // Calculate percentiles
        let p95_idx = (times.len() as f64 * 0.95) as usize;
        let p99_idx = (times.len() as f64 * 0.99) as usize;
        
        metrics.p95_inference_time_us = times[p95_idx.min(times.len() - 1)];
        metrics.p99_inference_time_us = times[p99_idx.min(times.len() - 1)];
        
        // Clear old times if too many (keep last 10k)
        if times.len() > 10000 {
            *times = times[times.len() - 10000..].to_vec();
        }
        
        Ok(())
    }
    
    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().clone()
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        *self.metrics.write() = PerformanceMetrics::new();
        self.inference_times.write().clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// ROC curve data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROCPoint {
    /// False positive rate
    pub fpr: f64,
    /// True positive rate (recall)
    pub tpr: f64,
    /// Threshold used
    pub threshold: f64,
}

/// Calculate ROC curve from predictions
pub fn calculate_roc_curve(
    predictions: &[(f64, bool)],  // (probability, actual_label)
    num_thresholds: usize,
) -> Vec<ROCPoint> {
    let mut roc_points = Vec::with_capacity(num_thresholds);
    
    // Sort by probability descending
    let mut sorted_preds = predictions.to_vec();
    sorted_preds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    // Calculate total positives and negatives
    let total_positives = sorted_preds.iter().filter(|(_, label)| *label).count() as f64;
    let total_negatives = sorted_preds.iter().filter(|(_, label)| !*label).count() as f64;
    
    // Generate thresholds
    for i in 0..num_thresholds {
        let threshold = i as f64 / (num_thresholds - 1) as f64;
        
        let mut tp = 0;
        let mut fp = 0;
        
        for (prob, actual) in &sorted_preds {
            if *prob >= threshold {
                if *actual {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        
        let tpr = if total_positives > 0.0 {
            tp as f64 / total_positives
        } else {
            0.0
        };
        
        let fpr = if total_negatives > 0.0 {
            fp as f64 / total_negatives
        } else {
            0.0
        };
        
        roc_points.push(ROCPoint { fpr, tpr, threshold });
    }
    
    roc_points
}

/// Calculate AUC (Area Under ROC Curve)
pub fn calculate_auc(roc_points: &[ROCPoint]) -> f64 {
    if roc_points.len() < 2 {
        return 0.0;
    }
    
    let mut auc = 0.0;
    
    for i in 1..roc_points.len() {
        let width = roc_points[i].fpr - roc_points[i - 1].fpr;
        let avg_height = (roc_points[i].tpr + roc_points[i - 1].tpr) / 2.0;
        auc += width * avg_height;
    }
    
    auc
}

/// Training metrics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: u32,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: f64,
    /// Validation accuracy
    pub val_accuracy: f64,
    /// Validation AUC
    pub val_auc: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Epoch duration in seconds
    pub epoch_duration_secs: f64,
}

/// Track training progress
pub struct TrainingTracker {
    metrics_history: Vec<TrainingMetrics>,
    best_val_loss: f64,
    best_epoch: u32,
    patience_counter: u32,
    patience: u32,
}

impl TrainingTracker {
    /// Create a new training tracker
    pub fn new(patience: u32) -> Self {
        Self {
            metrics_history: Vec::new(),
            best_val_loss: f64::INFINITY,
            best_epoch: 0,
            patience_counter: 0,
            patience,
        }
    }
    
    /// Record epoch metrics
    pub fn record_epoch(&mut self, metrics: TrainingMetrics) -> bool {
        self.metrics_history.push(metrics.clone());
        
        // Check if validation loss improved
        if metrics.val_loss < self.best_val_loss {
            self.best_val_loss = metrics.val_loss;
            self.best_epoch = metrics.epoch;
            self.patience_counter = 0;
            true  // Improved
        } else {
            self.patience_counter += 1;
            false  // No improvement
        }
    }
    
    /// Check if early stopping should trigger
    pub fn should_stop(&self) -> bool {
        self.patience_counter >= self.patience
    }
    
    /// Get training history
    pub fn get_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }
    
    /// Get best epoch info
    pub fn get_best_epoch(&self) -> (u32, f64) {
        (self.best_epoch, self.best_val_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.total_predictions = 100;
        metrics.true_positives = 30;
        metrics.false_positives = 10;
        metrics.true_negatives = 50;
        metrics.false_negatives = 10;
        
        assert!((metrics.accuracy() - 0.8).abs() < 1e-6);
        assert!((metrics.precision() - 0.75).abs() < 1e-6);
        assert!((metrics.recall() - 0.75).abs() < 1e-6);
        assert!((metrics.f1_score() - 0.75).abs() < 1e-6);
    }
    
    #[test]
    fn test_inference_timer() {
        let timer = InferenceTimer::start();
        std::thread::sleep(Duration::from_micros(100));
        
        let elapsed = timer.stop();
        assert!(elapsed >= 100);
        assert!(timer.target_met());
    }
    
    #[test]
    fn test_roc_calculation() {
        let predictions = vec![
            (0.9, true),
            (0.8, true),
            (0.7, false),
            (0.6, true),
            (0.4, false),
            (0.3, false),
            (0.2, false),
            (0.1, false),
        ];
        
        let roc_points = calculate_roc_curve(&predictions, 10);
        assert_eq!(roc_points.len(), 10);
        
        let auc = calculate_auc(&roc_points);
        assert!(auc >= 0.0 && auc <= 1.0);
    }
}