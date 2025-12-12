//! Cognitive reappraisal and reinforcement learning

use std::collections::VecDeque;

/// Cognitive reappraisal engine for adaptive learning
pub struct CognitiveReappraisal {
    /// Base learning rate
    pub base_learning_rate: f64,
    /// Enable adaptive learning rate
    pub adaptive_learning: bool,
    /// Momentum factor
    pub momentum: f64,
    /// Learning progress tracker
    progress_tracker: LearningProgress,
    /// Configuration
    config: CognitiveConfig,
}

/// Cognitive configuration
#[derive(Debug, Clone)]
pub struct CognitiveConfig {
    /// Window size for performance tracking
    pub performance_window: usize,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Maximum learning rate
    pub max_learning_rate: f64,
    /// Decay factor for learning rate
    pub decay_factor: f64,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            performance_window: 50,
            min_learning_rate: 0.001,
            max_learning_rate: 0.1,
            decay_factor: 0.999,
        }
    }
}

/// Learning progress tracker
#[derive(Debug, Clone)]
pub struct LearningProgress {
    /// Error history
    error_history: VecDeque<f64>,
    /// Gradient history for momentum
    gradient_history: VecDeque<f64>,
    /// Total updates
    total_updates: u64,
    /// Best performance
    best_performance: f64,
    /// Current streak (consecutive improvements)
    improvement_streak: u32,
}

impl LearningProgress {
    fn new(window_size: usize) -> Self {
        Self {
            error_history: VecDeque::with_capacity(window_size),
            gradient_history: VecDeque::with_capacity(window_size),
            total_updates: 0,
            best_performance: f64::NEG_INFINITY,
            improvement_streak: 0,
        }
    }
    
    /// Update with new error
    fn update(&mut self, error: f64, window_size: usize) {
        self.error_history.push_back(error.abs());
        if self.error_history.len() > window_size {
            self.error_history.pop_front();
        }
        
        self.total_updates += 1;
        
        // Track improvement
        let current_performance = -error.abs(); // Negative error is better
        if current_performance > self.best_performance {
            self.best_performance = current_performance;
            self.improvement_streak += 1;
        } else {
            self.improvement_streak = 0;
        }
    }
    
    /// Get average error
    fn average_error(&self) -> f64 {
        if self.error_history.is_empty() {
            0.0
        } else {
            self.error_history.iter().sum::<f64>() / self.error_history.len() as f64
        }
    }
    
    /// Get error trend (positive means improving)
    fn error_trend(&self) -> f64 {
        if self.error_history.len() < 2 {
            return 0.0;
        }
        
        let n = self.error_history.len();
        let first_half: f64 = self.error_history.iter()
            .take(n / 2)
            .sum::<f64>() / (n / 2) as f64;
        
        let second_half: f64 = self.error_history.iter()
            .skip(n / 2)
            .sum::<f64>() / (n - n / 2) as f64;
        
        first_half - second_half // Positive if errors are decreasing
    }
}

impl CognitiveReappraisal {
    /// Create new cognitive reappraisal engine
    pub fn new(config: crate::LearningConfig) -> Self {
        Self {
            base_learning_rate: config.base_learning_rate,
            adaptive_learning: config.adaptive_learning,
            momentum: config.momentum,
            progress_tracker: LearningProgress::new(50),
            config: CognitiveConfig::default(),
        }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(crate::LearningConfig::default())
    }
    
    /// Calculate adaptive learning rate based on performance history
    pub fn adaptive_learning_rate(&self, performance_history: &[f64]) -> f64 {
        if !self.adaptive_learning {
            return self.base_learning_rate;
        }
        
        // Base rate with decay
        let decayed_rate = self.base_learning_rate * 
            self.config.decay_factor.powi(self.progress_tracker.total_updates as i32);
        
        // Adjust based on performance trend
        let trend = self.progress_tracker.error_trend();
        let trend_multiplier = if trend > 0.0 {
            1.0 + trend.min(0.5) // Increase rate if improving
        } else {
            1.0 + trend.max(-0.5) // Decrease rate if worsening
        };
        
        // Adjust based on consistency
        let consistency_multiplier = if self.progress_tracker.improvement_streak > 5 {
            1.2 // Bonus for consistent improvement
        } else if self.progress_tracker.improvement_streak == 0 {
            0.8 // Penalty for no improvement
        } else {
            1.0
        };
        
        // Combine factors
        let adaptive_rate = decayed_rate * trend_multiplier * consistency_multiplier;
        
        // Clamp to configured bounds
        adaptive_rate.clamp(self.config.min_learning_rate, self.config.max_learning_rate)
    }
    
    /// Update learning progress with new error
    pub fn update_progress(&mut self, predicted: f64, actual: f64) {
        let error = actual - predicted;
        self.progress_tracker.update(error, self.config.performance_window);
    }
    
    /// Get current learning progress
    pub fn learning_progress(&self) -> f64 {
        // Progress based on error reduction and consistency
        let error_score = 1.0 / (1.0 + self.progress_tracker.average_error());
        let consistency_score = (self.progress_tracker.improvement_streak as f64 / 10.0).min(1.0);
        
        (error_score * 0.7 + consistency_score * 0.3).clamp(0.0, 1.0)
    }
    
    /// Calculate weight update with momentum
    pub fn calculate_weight_update(
        &mut self,
        error: f64,
        gradient: f64,
        learning_rate: f64,
    ) -> f64 {
        // Store gradient for momentum
        self.progress_tracker.gradient_history.push_back(gradient);
        if self.progress_tracker.gradient_history.len() > 10 {
            self.progress_tracker.gradient_history.pop_front();
        }
        
        // Calculate momentum term
        let momentum_term = if self.momentum > 0.0 && !self.progress_tracker.gradient_history.is_empty() {
            let avg_gradient = self.progress_tracker.gradient_history.iter()
                .sum::<f64>() / self.progress_tracker.gradient_history.len() as f64;
            self.momentum * avg_gradient
        } else {
            0.0
        };
        
        // Standard gradient update with momentum
        learning_rate * error * gradient + momentum_term
    }
    
    /// Get learning statistics
    pub fn get_statistics(&self) -> LearningStatistics {
        LearningStatistics {
            total_updates: self.progress_tracker.total_updates,
            average_error: self.progress_tracker.average_error(),
            error_trend: self.progress_tracker.error_trend(),
            improvement_streak: self.progress_tracker.improvement_streak,
            current_learning_rate: self.adaptive_learning_rate(&[]),
            learning_progress: self.learning_progress(),
        }
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone, Copy)]
pub enum LearningRate {
    /// Fixed learning rate
    Fixed(f64),
    /// Exponential decay
    ExponentialDecay { initial: f64, decay: f64 },
    /// Step decay
    StepDecay { initial: f64, step_size: u64, gamma: f64 },
    /// Adaptive based on performance
    Adaptive { base: f64, min: f64, max: f64 },
}

impl LearningRate {
    /// Get learning rate for given iteration
    pub fn get_rate(&self, iteration: u64, performance: Option<f64>) -> f64 {
        match self {
            Self::Fixed(rate) => *rate,
            
            Self::ExponentialDecay { initial, decay } => {
                initial * decay.powi(iteration as i32)
            }
            
            Self::StepDecay { initial, step_size, gamma } => {
                let steps = iteration / step_size;
                initial * gamma.powi(steps as i32)
            }
            
            Self::Adaptive { base, min, max } => {
                if let Some(perf) = performance {
                    // Higher performance -> higher learning rate
                    let rate = base * (0.5 + perf);
                    rate.clamp(*min, *max)
                } else {
                    *base
                }
            }
        }
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStatistics {
    pub total_updates: u64,
    pub average_error: f64,
    pub error_trend: f64,
    pub improvement_streak: u32,
    pub current_learning_rate: f64,
    pub learning_progress: f64,
}

/// Reinforcement learning update strategies
#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    /// Simple gradient descent
    GradientDescent,
    /// Gradient descent with momentum
    Momentum { beta: f64 },
    /// Adam optimizer
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
    /// RMSprop
    RMSprop { beta: f64, epsilon: f64 },
}

impl UpdateStrategy {
    /// Calculate parameter update
    pub fn calculate_update(
        &self,
        gradient: f64,
        learning_rate: f64,
        history: &mut UpdateHistory,
    ) -> f64 {
        match self {
            Self::GradientDescent => learning_rate * gradient,
            
            Self::Momentum { beta } => {
                history.momentum = beta * history.momentum + gradient;
                learning_rate * history.momentum
            }
            
            Self::Adam { beta1, beta2, epsilon } => {
                history.iteration += 1;
                
                // Update biased first moment estimate
                history.m = beta1 * history.m + (1.0 - beta1) * gradient;
                
                // Update biased second moment estimate
                history.v = beta2 * history.v + (1.0 - beta2) * gradient * gradient;
                
                // Bias correction
                let m_hat = history.m / (1.0 - beta1.powi(history.iteration as i32));
                let v_hat = history.v / (1.0 - beta2.powi(history.iteration as i32));
                
                // Adam update
                learning_rate * m_hat / (v_hat.sqrt() + epsilon)
            }
            
            Self::RMSprop { beta, epsilon } => {
                history.v = beta * history.v + (1.0 - beta) * gradient * gradient;
                learning_rate * gradient / (history.v.sqrt() + epsilon)
            }
        }
    }
}

/// Update history for optimizers
#[derive(Debug, Clone, Default)]
pub struct UpdateHistory {
    pub momentum: f64,
    pub m: f64, // First moment (Adam)
    pub v: f64, // Second moment (Adam/RMSprop)
    pub iteration: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_learning_progress() {
        let mut progress = LearningProgress::new(10);
        
        // Simulate improving performance
        for i in 0..10 {
            let error = 0.1 * (10.0 - i as f64); // Decreasing error
            progress.update(error, 10);
        }
        
        assert!(progress.error_trend() > 0.0); // Positive trend (improving)
        assert!(progress.improvement_streak > 0);
    }
    
    #[test]
    fn test_adaptive_learning_rate() {
        let config = crate::LearningConfig {
            base_learning_rate: 0.01,
            adaptive_learning: true,
            momentum: 0.9,
        };
        
        let cognitive = CognitiveReappraisal::new(config);
        
        // With empty history, should return base rate
        let rate = cognitive.adaptive_learning_rate(&[]);
        assert!((rate - 0.01).abs() < 1e-6);
    }
    
    #[test]
    fn test_learning_rate_schedules() {
        // Test exponential decay
        let exp_decay = LearningRate::ExponentialDecay {
            initial: 0.1,
            decay: 0.9,
        };
        
        let rate_0 = exp_decay.get_rate(0, None);
        let rate_10 = exp_decay.get_rate(10, None);
        
        assert!(rate_0 > rate_10);
        assert!((rate_10 - 0.1 * 0.9_f64.powi(10)).abs() < 1e-6);
        
        // Test adaptive
        let adaptive = LearningRate::Adaptive {
            base: 0.01,
            min: 0.001,
            max: 0.1,
        };
        
        let rate_good = adaptive.get_rate(0, Some(0.9));
        let rate_bad = adaptive.get_rate(0, Some(0.1));
        
        assert!(rate_good > rate_bad);
    }
    
    #[test]
    fn test_update_strategies() {
        let mut history = UpdateHistory::default();
        
        // Test momentum
        let momentum = UpdateStrategy::Momentum { beta: 0.9 };
        let update1 = momentum.calculate_update(1.0, 0.01, &mut history);
        let update2 = momentum.calculate_update(1.0, 0.01, &mut history);
        
        // Second update should be larger due to momentum
        assert!(update2 > update1);
        
        // Test Adam
        let mut adam_history = UpdateHistory::default();
        let adam = UpdateStrategy::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        
        let adam_update = adam.calculate_update(1.0, 0.01, &mut adam_history);
        assert!(adam_update > 0.0);
        assert_eq!(adam_history.iteration, 1);
    }
}