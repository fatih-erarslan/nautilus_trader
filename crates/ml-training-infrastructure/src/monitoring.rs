//! Performance monitoring module

use crate::Result;
use prometheus::{Registry, Counter, Histogram, Gauge};
use once_cell::sync::Lazy;

/// Global metrics registry
pub static METRICS_REGISTRY: Lazy<Registry> = Lazy::new(|| {
    Registry::new()
});

/// Training metrics
pub struct TrainingMonitor {
    pub epochs_counter: Counter,
    pub training_time_histogram: Histogram,
    pub current_loss_gauge: Gauge,
}

impl TrainingMonitor {
    /// Create new monitor
    pub fn new() -> Result<Self> {
        let epochs_counter = Counter::new("training_epochs_total", "Total training epochs")?;
        let training_time_histogram = Histogram::new("training_time_seconds", "Training time distribution")?;
        let current_loss_gauge = Gauge::new("current_loss", "Current training loss")?;
        
        METRICS_REGISTRY.register(Box::new(epochs_counter.clone()))?;
        METRICS_REGISTRY.register(Box::new(training_time_histogram.clone()))?;
        METRICS_REGISTRY.register(Box::new(current_loss_gauge.clone()))?;
        
        Ok(Self {
            epochs_counter,
            training_time_histogram,
            current_loss_gauge,
        })
    }
}