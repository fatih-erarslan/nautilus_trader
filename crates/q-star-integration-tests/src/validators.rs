//! Performance validators

use q_star_core::*;
use q_star_orchestrator::*;

/// Validate that decision meets latency requirements
pub fn validate_latency(decision_time_us: u64, max_allowed_us: u64) -> bool {
    decision_time_us <= max_allowed_us
}

/// Validate confidence threshold
pub fn validate_confidence(confidence: f64, min_confidence: f64) -> bool {
    confidence >= min_confidence
}

/// Validate consensus was achieved
pub fn validate_consensus(result: &QStarResult) -> bool {
    result.consensus_achieved && result.agents_participated.len() >= 3
}

/// Validate performance targets
pub fn validate_performance_targets(metrics: &SystemMetrics, targets: &PerformanceTargets) -> bool {
    metrics.avg_latency_us <= targets.max_latency_us as f64 &&
    metrics.throughput >= targets.min_throughput as f64 &&
    metrics.memory_usage_mb <= targets.max_memory_mb as f64 &&
    metrics.accuracy >= targets.target_accuracy
}

/// Performance report structure
#[derive(Debug)]
pub struct PerformanceReport {
    pub test_name: String,
    pub passed: bool,
    pub avg_latency_us: f64,
    pub p99_latency_us: f64,
    pub throughput: f64,
    pub accuracy: f64,
    pub memory_usage_mb: f64,
    pub errors: Vec<String>,
}

impl PerformanceReport {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            passed: true,
            avg_latency_us: 0.0,
            p99_latency_us: 0.0,
            throughput: 0.0,
            accuracy: 0.0,
            memory_usage_mb: 0.0,
            errors: Vec::new(),
        }
    }
    
    pub fn validate_target(&mut self, metric: &str, value: f64, target: f64, less_than: bool) {
        let passed = if less_than { value <= target } else { value >= target };
        
        if !passed {
            self.passed = false;
            self.errors.push(format!(
                "{}: {:.2} {} target {:.2}",
                metric,
                value,
                if less_than { "exceeds" } else { "below" },
                target
            ));
        }
    }
    
    pub fn print_summary(&self) {
        println!("\n{} Performance Report", self.test_name);
        println!("=====================================");
        println!("Status: {}", if self.passed { "PASSED ✓" } else { "FAILED ✗" });
        println!("Avg Latency: {:.2}μs", self.avg_latency_us);
        println!("P99 Latency: {:.2}μs", self.p99_latency_us);
        println!("Throughput: {:.0} ops/sec", self.throughput);
        println!("Accuracy: {:.2}%", self.accuracy * 100.0);
        println!("Memory Usage: {:.2} MB", self.memory_usage_mb);
        
        if !self.errors.is_empty() {
            println!("\nFailures:");
            for error in &self.errors {
                println!("  - {}", error);
            }
        }
        println!();
    }
}