//! Utility functions for hyperphysics-ml

mod serialization;
mod metrics;

pub use serialization::*;
pub use metrics::*;

use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};
use crate::backends::Device;

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy)]
pub enum LrScheduler {
    /// Constant learning rate
    Constant,
    /// Step decay
    StepDecay { step_size: usize, gamma: f32 },
    /// Exponential decay
    Exponential { gamma: f32 },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f32 },
    /// Warmup with linear schedule
    WarmupLinear { warmup_steps: usize },
    /// One cycle policy
    OneCycle { max_lr: f32, total_steps: usize },
}

/// Compute learning rate for given step
pub fn get_lr(scheduler: LrScheduler, base_lr: f32, step: usize) -> f32 {
    match scheduler {
        LrScheduler::Constant => base_lr,
        LrScheduler::StepDecay { step_size, gamma } => {
            let decay_count = step / step_size;
            base_lr * gamma.powi(decay_count as i32)
        }
        LrScheduler::Exponential { gamma } => {
            base_lr * gamma.powi(step as i32)
        }
        LrScheduler::CosineAnnealing { t_max, eta_min } => {
            let progress = (step % t_max) as f32 / t_max as f32;
            eta_min + (base_lr - eta_min) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
        }
        LrScheduler::WarmupLinear { warmup_steps } => {
            if step < warmup_steps {
                base_lr * (step + 1) as f32 / warmup_steps as f32
            } else {
                base_lr
            }
        }
        LrScheduler::OneCycle { max_lr, total_steps } => {
            let half = total_steps / 2;
            if step < half {
                // Warmup phase
                base_lr + (max_lr - base_lr) * step as f32 / half as f32
            } else {
                // Decay phase
                let decay_steps = step - half;
                let remaining = total_steps - half;
                max_lr * (1.0 - decay_steps as f32 / remaining as f32).max(0.0)
            }
        }
    }
}

/// Gradient clipping modes
#[derive(Debug, Clone, Copy)]
pub enum GradientClip {
    /// Clip by value
    Value(f32),
    /// Clip by norm
    Norm(f32),
    /// Clip by global norm (across all parameters)
    GlobalNorm(f32),
}

/// Clip gradients in-place
pub fn clip_gradients(grads: &mut [Tensor], clip: GradientClip) -> MlResult<f32> {
    match clip {
        GradientClip::Value(max_val) => {
            for grad in grads {
                clip_tensor_by_value(grad, max_val)?;
            }
            Ok(max_val)
        }
        GradientClip::Norm(max_norm) => {
            for grad in grads {
                let norm = tensor_norm(grad)?;
                if norm > max_norm {
                    let scale = max_norm / norm;
                    scale_tensor_inplace(grad, scale)?;
                }
            }
            Ok(max_norm)
        }
        GradientClip::GlobalNorm(max_norm) => {
            // Compute global norm
            let mut total_norm_sq = 0.0_f32;
            for grad in grads.iter() {
                let norm = tensor_norm(grad)?;
                total_norm_sq += norm * norm;
            }
            let global_norm = total_norm_sq.sqrt();

            // Scale if needed
            if global_norm > max_norm {
                let scale = max_norm / global_norm;
                for grad in grads {
                    scale_tensor_inplace(grad, scale)?;
                }
            }

            Ok(global_norm)
        }
    }
}

/// Compute tensor L2 norm
fn tensor_norm(t: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = t.as_slice() {
            let norm_sq: f32 = data.iter().map(|x| x * x).sum();
            return Ok(norm_sq.sqrt());
        }
    }
    Ok(0.0)
}

/// Clip tensor values to [-max, max]
fn clip_tensor_by_value(t: &mut Tensor, max_val: f32) -> MlResult<()> {
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = t.as_slice_mut() {
            for x in data {
                *x = x.clamp(-max_val, max_val);
            }
        }
    }
    Ok(())
}

/// Scale tensor in-place
fn scale_tensor_inplace(t: &mut Tensor, scale: f32) -> MlResult<()> {
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = t.as_slice_mut() {
            for x in data {
                *x *= scale;
            }
        }
    }
    Ok(())
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Patience (epochs without improvement before stopping)
    pub patience: usize,
    /// Minimum delta to consider as improvement
    pub min_delta: f32,
    /// Mode (minimize or maximize)
    pub mode: EarlyStoppingMode,
    /// Current best value
    best_value: Option<f32>,
    /// Epochs without improvement
    counter: usize,
}

/// Early stopping mode
#[derive(Debug, Clone, Copy)]
pub enum EarlyStoppingMode {
    /// Minimize metric (e.g., loss)
    Min,
    /// Maximize metric (e.g., accuracy)
    Max,
}

impl EarlyStopping {
    /// Create new early stopping
    pub fn new(patience: usize, min_delta: f32, mode: EarlyStoppingMode) -> Self {
        Self {
            patience,
            min_delta,
            mode,
            best_value: None,
            counter: 0,
        }
    }

    /// Check if should stop
    pub fn check(&mut self, value: f32) -> bool {
        let improved = match self.best_value {
            None => true,
            Some(best) => match self.mode {
                EarlyStoppingMode::Min => value < best - self.min_delta,
                EarlyStoppingMode::Max => value > best + self.min_delta,
            },
        };

        if improved {
            self.best_value = Some(value);
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.best_value = None;
        self.counter = 0;
    }

    /// Get best value
    pub fn best(&self) -> Option<f32> {
        self.best_value
    }
}

/// Weight initialization strategies
#[derive(Debug, Clone, Copy)]
pub enum WeightInit {
    /// Xavier/Glorot uniform
    XavierUniform,
    /// Xavier/Glorot normal
    XavierNormal,
    /// Kaiming/He uniform (for ReLU)
    KaimingUniform,
    /// Kaiming/He normal (for ReLU)
    KaimingNormal,
    /// Orthogonal initialization
    Orthogonal,
    /// Uniform in range [-a, a]
    Uniform(f32),
    /// Normal with given std
    Normal(f32),
    /// Constant value
    Constant(f32),
}

/// Initialize tensor with specified strategy
pub fn init_weights(shape: &[usize], init: WeightInit, device: &Device) -> MlResult<Tensor> {
    let size: usize = shape.iter().product();
    let fan_in = if shape.len() >= 2 { shape[1] } else { shape[0] };
    let fan_out = shape[0];

    let data: Vec<f32> = match init {
        WeightInit::XavierUniform => {
            let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
            (0..size).map(|_| rand_uniform(-limit, limit)).collect()
        }
        WeightInit::XavierNormal => {
            let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
            (0..size).map(|_| rand_normal(0.0, std)).collect()
        }
        WeightInit::KaimingUniform => {
            let limit = (6.0 / fan_in as f32).sqrt();
            (0..size).map(|_| rand_uniform(-limit, limit)).collect()
        }
        WeightInit::KaimingNormal => {
            let std = (2.0 / fan_in as f32).sqrt();
            (0..size).map(|_| rand_normal(0.0, std)).collect()
        }
        WeightInit::Orthogonal => {
            // Simplified: use Xavier for now
            let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
            (0..size).map(|_| rand_normal(0.0, std)).collect()
        }
        WeightInit::Uniform(a) => {
            (0..size).map(|_| rand_uniform(-a, a)).collect()
        }
        WeightInit::Normal(std) => {
            (0..size).map(|_| rand_normal(0.0, std)).collect()
        }
        WeightInit::Constant(val) => {
            vec![val; size]
        }
    };

    Tensor::from_slice(&data, shape.to_vec(), device)
}

// Simple random number generation (for initialization only)
// In production, would use proper RNG from rand crate

fn rand_uniform(min: f32, max: f32) -> f32 {
    // Linear congruential generator (simple but fast)
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (SEED >> 33) as f32 / (1u64 << 31) as f32;
        min + u * (max - min)
    }
}

fn rand_normal(mean: f32, std: f32) -> f32 {
    // Box-Muller transform
    let u1 = rand_uniform(1e-10, 1.0);
    let u2 = rand_uniform(0.0, 1.0);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + z * std
}

/// Set random seed for reproducibility
pub fn set_seed(seed: u64) {
    unsafe {
        static mut SEED: u64 = 0;
        SEED = seed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_scheduler_constant() {
        let lr = get_lr(LrScheduler::Constant, 0.01, 100);
        assert!((lr - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_step_decay() {
        let scheduler = LrScheduler::StepDecay { step_size: 10, gamma: 0.1 };
        let lr_0 = get_lr(scheduler, 1.0, 0);
        let lr_10 = get_lr(scheduler, 1.0, 10);
        let lr_20 = get_lr(scheduler, 1.0, 20);

        assert!((lr_0 - 1.0).abs() < 1e-6);
        assert!((lr_10 - 0.1).abs() < 1e-6);
        assert!((lr_20 - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::new(3, 0.01, EarlyStoppingMode::Min);

        assert!(!es.check(1.0));
        assert!(!es.check(0.9));
        assert!(!es.check(0.95)); // No improvement
        assert!(!es.check(0.94)); // No improvement
        assert!(es.check(0.94));  // Third time, should stop
    }
}
