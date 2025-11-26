//! Learning rate schedulers for training optimization
//!
//! Provides various learning rate scheduling strategies:
//! - Cosine annealing
//! - Warmup with linear/exponential decay
//! - Step decay
//! - Reduce on plateau
//! - Custom schedules

use crate::{Result, NeuroDivergentError};
use serde::{Deserialize, Serialize};

/// Learning rate scheduler trait
pub trait LRScheduler: Send + Sync {
    /// Update learning rate based on epoch/step
    fn step(&mut self, epoch: usize, metrics: Option<&SchedulerMetrics>) -> f64;

    /// Get current learning rate
    fn get_lr(&self) -> f64;

    /// Reset scheduler state
    fn reset(&mut self);

    /// Scheduler name
    fn name(&self) -> &str;
}

/// Metrics passed to scheduler for adaptive scheduling
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub epoch: usize,
}

/// Cosine annealing learning rate scheduler
/// LR follows a cosine curve from base_lr to eta_min over T_max epochs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineAnnealingLR {
    base_lr: f64,
    eta_min: f64,
    t_max: usize,
    current_lr: f64,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            eta_min,
            t_max,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, epoch: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        let epoch = epoch.min(self.t_max);
        let cos_inner = std::f64::consts::PI * (epoch as f64) / (self.t_max as f64);
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cos_inner.cos()) / 2.0;
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = self.base_lr;
    }

    fn name(&self) -> &str {
        "CosineAnnealing"
    }
}

/// Warmup followed by linear decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupLinearLR {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_lr: f64,
}

impl WarmupLinearLR {
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Result<Self> {
        if warmup_steps > total_steps {
            return Err(NeuroDivergentError::TrainingError(
                "Warmup steps must be less than total steps".to_string()
            ));
        }

        Ok(Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_lr: 0.0,
        })
    }
}

impl LRScheduler for WarmupLinearLR {
    fn step(&mut self, step: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.current_lr = self.base_lr * (step as f64) / (self.warmup_steps as f64);
        } else {
            // Linear decay
            let decay_steps = step - self.warmup_steps;
            let total_decay_steps = self.total_steps - self.warmup_steps;
            let decay_ratio = 1.0 - (decay_steps as f64) / (total_decay_steps as f64);
            self.current_lr = self.base_lr * decay_ratio.max(0.0);
        }

        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = 0.0;
    }

    fn name(&self) -> &str {
        "WarmupLinear"
    }
}

/// Warmup followed by cosine annealing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupCosineLR {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: f64,
    current_lr: f64,
}

impl WarmupCosineLR {
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, eta_min: f64) -> Result<Self> {
        if warmup_steps > total_steps {
            return Err(NeuroDivergentError::TrainingError(
                "Warmup steps must be less than total steps".to_string()
            ));
        }

        Ok(Self {
            base_lr,
            warmup_steps,
            total_steps,
            eta_min,
            current_lr: 0.0,
        })
    }
}

impl LRScheduler for WarmupCosineLR {
    fn step(&mut self, step: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.current_lr = self.base_lr * (step as f64) / (self.warmup_steps as f64);
        } else {
            // Cosine annealing
            let decay_steps = step - self.warmup_steps;
            let total_decay_steps = self.total_steps - self.warmup_steps;
            let cos_inner = std::f64::consts::PI * (decay_steps as f64) / (total_decay_steps as f64);
            self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cos_inner.cos()) / 2.0;
        }

        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = 0.0;
    }

    fn name(&self) -> &str {
        "WarmupCosine"
    }
}

/// Step decay - reduce LR by factor every N epochs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    current_lr: f64,
}

impl StepLR {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, epoch: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        let num_steps = epoch / self.step_size;
        self.current_lr = self.base_lr * self.gamma.powi(num_steps as i32);
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = self.base_lr;
    }

    fn name(&self) -> &str {
        "Step"
    }
}

/// Exponential decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
    current_lr: f64,
}

impl ExponentialLR {
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, epoch: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        self.current_lr = self.base_lr * self.gamma.powi(epoch as i32);
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = self.base_lr;
    }

    fn name(&self) -> &str {
        "Exponential"
    }
}

/// Reduce learning rate when metric plateaus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceLROnPlateau {
    base_lr: f64,
    current_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    threshold: f64,
    best_metric: Option<f64>,
    num_bad_epochs: usize,
    mode: PlateauMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PlateauMode {
    Min,  // Reduce when metric stops decreasing
    Max,  // Reduce when metric stops increasing
}

impl ReduceLROnPlateau {
    pub fn new(
        base_lr: f64,
        factor: f64,
        patience: usize,
        min_lr: f64,
        threshold: f64,
        mode: PlateauMode,
    ) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            factor,
            patience,
            min_lr,
            threshold,
            best_metric: None,
            num_bad_epochs: 0,
            mode,
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self, _epoch: usize, metrics: Option<&SchedulerMetrics>) -> f64 {
        if let Some(metrics) = metrics {
            if let Some(val_loss) = metrics.val_loss {
                let is_better = match self.mode {
                    PlateauMode::Min => {
                        if let Some(best) = self.best_metric {
                            val_loss < best - self.threshold
                        } else {
                            true
                        }
                    },
                    PlateauMode::Max => {
                        if let Some(best) = self.best_metric {
                            val_loss > best + self.threshold
                        } else {
                            true
                        }
                    },
                };

                if is_better {
                    self.best_metric = Some(val_loss);
                    self.num_bad_epochs = 0;
                } else {
                    self.num_bad_epochs += 1;

                    if self.num_bad_epochs >= self.patience {
                        let new_lr = (self.current_lr * self.factor).max(self.min_lr);
                        if new_lr < self.current_lr {
                            tracing::info!(
                                "Reducing learning rate from {:.6} to {:.6}",
                                self.current_lr,
                                new_lr
                            );
                            self.current_lr = new_lr;
                        }
                        self.num_bad_epochs = 0;
                    }
                }
            }
        }

        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.best_metric = None;
        self.num_bad_epochs = 0;
    }

    fn name(&self) -> &str {
        "ReduceOnPlateau"
    }
}

/// Constant learning rate (no scheduling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn step(&mut self, _epoch: usize, _metrics: Option<&SchedulerMetrics>) -> f64 {
        self.lr
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn reset(&mut self) {
        // Nothing to reset
    }

    fn name(&self) -> &str {
        "Constant"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cosine_annealing() {
        let mut scheduler = CosineAnnealingLR::new(1.0, 100, 0.0);

        // At epoch 0, should be base_lr
        assert_relative_eq!(scheduler.step(0, None), 1.0, epsilon = 1e-10);

        // At epoch 50 (halfway), should be 0.5
        assert_relative_eq!(scheduler.step(50, None), 0.5, epsilon = 1e-2);

        // At epoch 100, should be eta_min (0.0)
        assert_relative_eq!(scheduler.step(100, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_warmup_linear() {
        let mut scheduler = WarmupLinearLR::new(1.0, 10, 100).unwrap();

        // At step 0, should be 0
        assert_relative_eq!(scheduler.step(0, None), 0.0, epsilon = 1e-10);

        // At step 5 (half warmup), should be 0.5
        assert_relative_eq!(scheduler.step(5, None), 0.5, epsilon = 1e-10);

        // At step 10 (end warmup), should be 1.0
        assert_relative_eq!(scheduler.step(10, None), 1.0, epsilon = 1e-10);

        // At step 55 (halfway through decay), should be 0.5
        assert_relative_eq!(scheduler.step(55, None), 0.5, epsilon = 1e-10);

        // At step 100, should be 0
        assert_relative_eq!(scheduler.step(100, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLR::new(1.0, 10, 0.5);

        assert_relative_eq!(scheduler.step(0, None), 1.0, epsilon = 1e-10);
        assert_relative_eq!(scheduler.step(9, None), 1.0, epsilon = 1e-10);
        assert_relative_eq!(scheduler.step(10, None), 0.5, epsilon = 1e-10);
        assert_relative_eq!(scheduler.step(20, None), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_lr() {
        let mut scheduler = ExponentialLR::new(1.0, 0.9);

        assert_relative_eq!(scheduler.step(0, None), 1.0, epsilon = 1e-10);
        assert_relative_eq!(scheduler.step(1, None), 0.9, epsilon = 1e-10);
        assert_relative_eq!(scheduler.step(2, None), 0.81, epsilon = 1e-10);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(
            1.0,    // base_lr
            0.5,    // factor
            2,      // patience
            0.001,  // min_lr
            0.01,   // threshold
            PlateauMode::Min,
        );

        let metrics1 = SchedulerMetrics {
            train_loss: 1.0,
            val_loss: Some(1.0),
            epoch: 0,
        };

        // First epoch - establish baseline
        assert_relative_eq!(scheduler.step(0, Some(&metrics1)), 1.0, epsilon = 1e-10);

        // Improvement - no reduction
        let metrics2 = SchedulerMetrics {
            train_loss: 0.8,
            val_loss: Some(0.8),
            epoch: 1,
        };
        assert_relative_eq!(scheduler.step(1, Some(&metrics2)), 1.0, epsilon = 1e-10);

        // Plateau - first bad epoch
        let metrics3 = SchedulerMetrics {
            train_loss: 0.81,
            val_loss: Some(0.81),
            epoch: 2,
        };
        assert_relative_eq!(scheduler.step(2, Some(&metrics3)), 1.0, epsilon = 1e-10);

        // Plateau - second bad epoch, should reduce
        let metrics4 = SchedulerMetrics {
            train_loss: 0.82,
            val_loss: Some(0.82),
            epoch: 3,
        };
        assert_relative_eq!(scheduler.step(3, Some(&metrics4)), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = StepLR::new(1.0, 10, 0.5);

        scheduler.step(20, None);
        assert_relative_eq!(scheduler.get_lr(), 0.25, epsilon = 1e-10);

        scheduler.reset();
        assert_relative_eq!(scheduler.get_lr(), 1.0, epsilon = 1e-10);
    }
}
