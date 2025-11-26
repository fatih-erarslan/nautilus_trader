//! Optimizers for neural network training

use ndarray::Array2;
use crate::Result;

/// Configuration for optimizers
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub epsilon: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0001,
            epsilon: 1e-8,
        }
    }
}

/// Base optimizer trait
pub trait Optimizer: Send + Sync {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) -> Result<()>;
    fn zero_grad(&mut self);
    fn get_lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
}

/// Adam optimizer
pub struct Adam {
    config: OptimizerConfig,
    m: Option<Array2<f64>>,
    v: Option<Array2<f64>>,
    t: usize,
    beta1: f64,
    beta2: f64,
}

impl Adam {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: None,
            v: None,
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) -> Result<()> {
        self.t += 1;

        // Initialize moment estimates
        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.dim()));
            self.v = Some(Array2::zeros(params.dim()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first and second moment estimates
        *m = m.mapv(|x| x * self.beta1) + gradients.mapv(|x| x * (1.0 - self.beta1));
        *v = v.mapv(|x| x * self.beta2) + gradients.mapv(|x| x.powi(2) * (1.0 - self.beta2));

        // Bias correction
        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.t as i32)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.t as i32)));

        // Update parameters
        for ((param, grad), (m_val, v_val)) in params.iter_mut()
            .zip(gradients.iter())
            .zip(m_hat.iter().zip(v_hat.iter()))
        {
            *param -= self.config.learning_rate * m_val / (v_val.sqrt() + self.config.epsilon);
            *param -= self.config.weight_decay * *param; // L2 regularization
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Nothing to do for Adam
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }
}

/// SGD optimizer with momentum
pub struct SGD {
    config: OptimizerConfig,
    velocity: Option<Array2<f64>>,
    momentum: f64,
}

impl SGD {
    pub fn new(config: OptimizerConfig, momentum: f64) -> Self {
        Self {
            config,
            velocity: None,
            momentum,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) -> Result<()> {
        if self.velocity.is_none() {
            self.velocity = Some(Array2::zeros(params.dim()));
        }

        let v = self.velocity.as_mut().unwrap();

        // Update velocity
        *v = v.mapv(|x| x * self.momentum) - gradients.mapv(|x| x * self.config.learning_rate);

        // Update parameters
        *params += v;

        // Apply weight decay
        if self.config.weight_decay > 0.0 {
            *params = params.mapv(|x| x * (1.0 - self.config.weight_decay));
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Nothing to do for SGD
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }
}
