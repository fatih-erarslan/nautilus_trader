// Auto-scaling and load balancing module

mod auto_scale;
mod load_balance;
mod health_check;

pub use auto_scale::{AutoScaler, ScalingPolicy, ScalingMetrics};
pub use load_balance::{LoadBalancer, LoadBalancingStrategy};
pub use health_check::{HealthChecker, HealthStatus, HealthReport};

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};

/// Scaling decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingDecision {
    /// Scale up (add resources)
    ScaleUp,

    /// Scale down (remove resources)
    ScaleDown,

    /// No scaling needed
    NoChange,
}

/// Resource metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,

    /// Memory usage (0.0-1.0)
    pub memory_usage: f64,

    /// Task queue size
    pub queue_size: usize,

    /// Active tasks
    pub active_tasks: usize,

    /// Total tasks completed
    pub completed_tasks: u64,

    /// Average task duration (ms)
    pub avg_task_duration_ms: u64,

    /// Error rate (0.0-1.0)
    pub error_rate: f64,
}

impl ResourceMetrics {
    /// Check if metrics indicate overload
    pub fn is_overloaded(&self, cpu_threshold: f64, memory_threshold: f64) -> bool {
        self.cpu_usage > cpu_threshold || self.memory_usage > memory_threshold
    }

    /// Check if metrics indicate underutilization
    pub fn is_underutilized(&self, cpu_threshold: f64, memory_threshold: f64) -> bool {
        self.cpu_usage < cpu_threshold && self.memory_usage < memory_threshold
    }

    /// Calculate overall load score (0.0-1.0)
    pub fn load_score(&self) -> f64 {
        // Weighted average of different metrics
        let cpu_weight = 0.4;
        let memory_weight = 0.3;
        let queue_weight = 0.2;
        let error_weight = 0.1;

        let queue_normalized = (self.queue_size as f64 / 1000.0).min(1.0);

        cpu_weight * self.cpu_usage
            + memory_weight * self.memory_usage
            + queue_weight * queue_normalized
            + error_weight * self.error_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_metrics_overload() {
        let mut metrics = ResourceMetrics::default();
        metrics.cpu_usage = 0.9;
        metrics.memory_usage = 0.5;

        assert!(metrics.is_overloaded(0.8, 0.8));
        assert!(!metrics.is_underutilized(0.2, 0.2));
    }

    #[test]
    fn test_load_score() {
        let mut metrics = ResourceMetrics::default();
        metrics.cpu_usage = 0.5;
        metrics.memory_usage = 0.5;
        metrics.queue_size = 100;
        metrics.error_rate = 0.1;

        let score = metrics.load_score();
        assert!(score > 0.0 && score < 1.0);
    }
}
