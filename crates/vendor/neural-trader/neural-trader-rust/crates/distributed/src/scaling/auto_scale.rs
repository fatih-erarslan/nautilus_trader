// Auto-scaling logic

use super::{ResourceMetrics, ScalingDecision};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Minimum instances
    pub min_instances: usize,

    /// Maximum instances
    pub max_instances: usize,

    /// CPU threshold for scaling up (0.0-1.0)
    pub scale_up_cpu_threshold: f64,

    /// CPU threshold for scaling down (0.0-1.0)
    pub scale_down_cpu_threshold: f64,

    /// Memory threshold for scaling up (0.0-1.0)
    pub scale_up_memory_threshold: f64,

    /// Memory threshold for scaling down (0.0-1.0)
    pub scale_down_memory_threshold: f64,

    /// Cooldown period between scaling actions (seconds)
    pub cooldown_seconds: u64,

    /// Number of consecutive violations before scaling
    pub violation_threshold: u32,
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            min_instances: 2,
            max_instances: 20,
            scale_up_cpu_threshold: 0.75,
            scale_down_cpu_threshold: 0.25,
            scale_up_memory_threshold: 0.75,
            scale_down_memory_threshold: 0.25,
            cooldown_seconds: 300,
            violation_threshold: 3,
        }
    }
}

/// Scaling metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalingMetrics {
    /// Current instance count
    pub current_instances: usize,

    /// Target instance count
    pub target_instances: usize,

    /// Total scale-up events
    pub scale_up_events: u64,

    /// Total scale-down events
    pub scale_down_events: u64,

    /// Last scaling timestamp
    pub last_scaling: Option<chrono::DateTime<chrono::Utc>>,

    /// Current resource metrics
    pub resource_metrics: ResourceMetrics,
}

/// Auto-scaler
pub struct AutoScaler {
    /// Scaling policy
    policy: Arc<RwLock<ScalingPolicy>>,

    /// Scaling metrics
    metrics: Arc<RwLock<ScalingMetrics>>,

    /// Violation counter
    violation_counter: Arc<RwLock<u32>>,
}

impl AutoScaler {
    /// Create new auto-scaler
    pub fn new(policy: ScalingPolicy) -> Self {
        Self {
            policy: Arc::new(RwLock::new(policy)),
            metrics: Arc::new(RwLock::new(ScalingMetrics::default())),
            violation_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Evaluate scaling decision based on resource metrics
    pub async fn evaluate(
        &self,
        current_instances: usize,
        resource_metrics: ResourceMetrics,
    ) -> Result<ScalingDecision> {
        let policy = self.policy.read().await;

        // Update current metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.current_instances = current_instances;
            metrics.resource_metrics = resource_metrics.clone();
        }

        // Check cooldown period
        if !self.is_cooldown_over().await {
            return Ok(ScalingDecision::NoChange);
        }

        // Check if overloaded (scale up)
        if resource_metrics.is_overloaded(
            policy.scale_up_cpu_threshold,
            policy.scale_up_memory_threshold,
        ) {
            *self.violation_counter.write().await += 1;

            if *self.violation_counter.read().await >= policy.violation_threshold {
                if current_instances < policy.max_instances {
                    return Ok(ScalingDecision::ScaleUp);
                }
            }

            return Ok(ScalingDecision::NoChange);
        }

        // Check if underutilized (scale down)
        if resource_metrics.is_underutilized(
            policy.scale_down_cpu_threshold,
            policy.scale_down_memory_threshold,
        ) {
            *self.violation_counter.write().await += 1;

            if *self.violation_counter.read().await >= policy.violation_threshold {
                if current_instances > policy.min_instances {
                    return Ok(ScalingDecision::ScaleDown);
                }
            }

            return Ok(ScalingDecision::NoChange);
        }

        // Reset violation counter if within thresholds
        *self.violation_counter.write().await = 0;
        Ok(ScalingDecision::NoChange)
    }

    /// Execute scaling decision
    pub async fn execute_scaling(&self, decision: ScalingDecision) -> Result<usize> {
        let mut metrics = self.metrics.write().await;

        let new_instances = match decision {
            ScalingDecision::ScaleUp => {
                metrics.scale_up_events += 1;
                metrics.current_instances + 1
            }
            ScalingDecision::ScaleDown => {
                metrics.scale_down_events += 1;
                metrics.current_instances.saturating_sub(1)
            }
            ScalingDecision::NoChange => metrics.current_instances,
        };

        metrics.target_instances = new_instances;
        metrics.last_scaling = Some(chrono::Utc::now());

        // Reset violation counter
        *self.violation_counter.write().await = 0;

        tracing::info!(
            "Scaling decision: {:?}, new instances: {}",
            decision,
            new_instances
        );

        Ok(new_instances)
    }

    /// Check if cooldown period is over
    async fn is_cooldown_over(&self) -> bool {
        let metrics = self.metrics.read().await;
        let policy = self.policy.read().await;

        if let Some(last_scaling) = metrics.last_scaling {
            let elapsed = chrono::Utc::now()
                .signed_duration_since(last_scaling)
                .num_seconds() as u64;

            elapsed >= policy.cooldown_seconds
        } else {
            true // No previous scaling, cooldown not applicable
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ScalingMetrics {
        self.metrics.read().await.clone()
    }

    /// Update policy
    pub async fn update_policy(&self, policy: ScalingPolicy) {
        *self.policy.write().await = policy;
    }

    /// Start auto-scaling loop
    pub async fn start_auto_scaling<F>(
        &self,
        interval_seconds: u64,
        metrics_provider: F,
    ) where
        F: Fn() -> ResourceMetrics + Send + Sync + 'static,
    {
        let scaler = Arc::new(self.clone_for_loop());
        let mut ticker = interval(Duration::from_secs(interval_seconds));

        tokio::spawn(async move {
            loop {
                ticker.tick().await;

                let current_instances = {
                    let metrics = scaler.metrics.read().await;
                    metrics.current_instances
                };

                let resource_metrics = metrics_provider();

                match scaler.evaluate(current_instances, resource_metrics).await {
                    Ok(decision) if decision != ScalingDecision::NoChange => {
                        let _ = scaler.execute_scaling(decision).await;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("Auto-scaling error: {}", e);
                    }
                }
            }
        });
    }

    /// Helper for cloning for background loop
    fn clone_for_loop(&self) -> Self {
        Self {
            policy: Arc::clone(&self.policy),
            metrics: Arc::clone(&self.metrics),
            violation_counter: Arc::clone(&self.violation_counter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scaling_evaluation() {
        let policy = ScalingPolicy {
            min_instances: 2,
            max_instances: 10,
            scale_up_cpu_threshold: 0.7,
            scale_down_cpu_threshold: 0.3,
            scale_up_memory_threshold: 0.7,
            scale_down_memory_threshold: 0.3,
            cooldown_seconds: 1,
            violation_threshold: 1,
        };

        let scaler = AutoScaler::new(policy);

        // Test scale up
        let mut metrics = ResourceMetrics::default();
        metrics.cpu_usage = 0.9;

        let decision = scaler.evaluate(5, metrics).await.unwrap();
        assert_eq!(decision, ScalingDecision::ScaleUp);

        // Test scale down
        let mut metrics = ResourceMetrics::default();
        metrics.cpu_usage = 0.1;
        metrics.memory_usage = 0.1;

        tokio::time::sleep(Duration::from_secs(2)).await; // Wait for cooldown

        let decision = scaler.evaluate(5, metrics).await.unwrap();
        assert_eq!(decision, ScalingDecision::ScaleDown);
    }

    #[tokio::test]
    async fn test_execute_scaling() {
        let scaler = AutoScaler::new(ScalingPolicy::default());

        let new_instances = scaler.execute_scaling(ScalingDecision::ScaleUp).await.unwrap();
        assert_eq!(new_instances, 1);

        let metrics = scaler.get_metrics().await;
        assert_eq!(metrics.scale_up_events, 1);
    }
}
