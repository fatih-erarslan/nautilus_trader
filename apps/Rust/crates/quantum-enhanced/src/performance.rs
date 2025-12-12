//! Performance monitoring for quantum pattern detection

use crate::types::*;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};
use tracing::{debug, info, warn};

/// Performance monitor for quantum operations
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Current performance metrics
    current_metrics: QuantumPerformanceMetrics,
    /// Performance history for trend analysis
    performance_history: VecDeque<TimestampedMetrics>,
    /// Configuration target latency
    target_latency_us: u64,
    /// Detection start time for this session
    session_start: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimestampedMetrics {
    timestamp: Instant,
    metrics: QuantumPerformanceMetrics,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            current_metrics: QuantumPerformanceMetrics {
                avg_detection_latency_us: 0.0,
                detection_success_rate: 1.0,
                coherence_preservation_rate: 1.0,
                memory_efficiency: 1.0,
                cpu_utilization: 0.0,
                gpu_utilization: None,
                patterns_per_second: 0.0,
                total_calculations: 0,
                quantum_error_rate: 0.0,
            },
            performance_history: VecDeque::with_capacity(1000),
            target_latency_us: 100,
            session_start: Instant::now(),
        }
    }

    /// Record a quantum detection operation
    pub fn record_detection(&mut self, latency_us: u64, confidence: f64) {
        // Update total calculations
        self.current_metrics.total_calculations += 1;
        let total = self.current_metrics.total_calculations as f64;

        // Update average latency with exponential moving average
        let alpha = 0.1; // Smoothing factor
        self.current_metrics.avg_detection_latency_us = 
            alpha * latency_us as f64 + (1.0 - alpha) * self.current_metrics.avg_detection_latency_us;

        // Update success rate based on confidence
        let success = if confidence > 0.5 { 1.0 } else { 0.0 };
        self.current_metrics.detection_success_rate = 
            (self.current_metrics.detection_success_rate * (total - 1.0) + success) / total;

        // Update patterns per second
        let elapsed_seconds = self.session_start.elapsed().as_secs_f64();
        if elapsed_seconds > 0.0 {
            self.current_metrics.patterns_per_second = total / elapsed_seconds;
        }

        // Check performance targets
        self.check_performance_targets(latency_us);

        // Store metrics in history
        self.store_metrics_snapshot();
    }

    /// Record coherence preservation metrics
    pub fn record_coherence_preservation(&mut self, preservation_rate: f64) {
        let alpha = 0.1;
        self.current_metrics.coherence_preservation_rate = 
            alpha * preservation_rate + (1.0 - alpha) * self.current_metrics.coherence_preservation_rate;
    }

    /// Record memory efficiency metrics
    pub fn record_memory_efficiency(&mut self, efficiency: f64) {
        self.current_metrics.memory_efficiency = efficiency.min(1.0).max(0.0);
    }

    /// Record CPU utilization
    pub fn record_cpu_utilization(&mut self, utilization: f64) {
        self.current_metrics.cpu_utilization = utilization.min(100.0).max(0.0);
    }

    /// Record GPU utilization (if available)
    pub fn record_gpu_utilization(&mut self, utilization: Option<f64>) {
        self.current_metrics.gpu_utilization = utilization.map(|u| u.min(100.0).max(0.0));
    }

    /// Record quantum error
    pub fn record_quantum_error(&mut self) {
        let total = self.current_metrics.total_calculations as f64;
        if total > 0.0 {
            // Calculate error rate as exponential moving average
            let alpha = 0.05; // Lower alpha for error rate (more stable)
            let new_error_contribution = 1.0 / total;
            self.current_metrics.quantum_error_rate = 
                alpha * new_error_contribution + (1.0 - alpha) * self.current_metrics.quantum_error_rate;
        }
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> QuantumPerformanceMetrics {
        self.current_metrics.clone()
    }

    /// Get performance trend analysis
    pub fn get_performance_trends(&self) -> PerformanceTrends {
        if self.performance_history.len() < 2 {
            return PerformanceTrends::default();
        }

        let recent_metrics = &self.performance_history.back().unwrap().metrics;
        let past_metrics = &self.performance_history.front().unwrap().metrics;

        PerformanceTrends {
            latency_trend: self.calculate_trend(
                past_metrics.avg_detection_latency_us,
                recent_metrics.avg_detection_latency_us,
            ),
            success_rate_trend: self.calculate_trend(
                past_metrics.detection_success_rate,
                recent_metrics.detection_success_rate,
            ),
            throughput_trend: self.calculate_trend(
                past_metrics.patterns_per_second,
                recent_metrics.patterns_per_second,
            ),
            error_rate_trend: self.calculate_trend(
                past_metrics.quantum_error_rate,
                recent_metrics.quantum_error_rate,
            ),
        }
    }

    /// Generate performance recommendations
    pub fn generate_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();
        let metrics = &self.current_metrics;

        // Latency recommendations
        if metrics.avg_detection_latency_us > self.target_latency_us as f64 * 1.2 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Latency,
                priority: RecommendationPriority::High,
                description: format!(
                    "Detection latency ({:.1}μs) exceeds target ({:.1}μs) by 20%. Consider enabling SIMD or GPU acceleration.",
                    metrics.avg_detection_latency_us,
                    self.target_latency_us
                ),
                estimated_improvement: 30.0,
            });
        }

        // Success rate recommendations
        if metrics.detection_success_rate < 0.9 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Accuracy,
                priority: RecommendationPriority::High,
                description: format!(
                    "Detection success rate ({:.1}%) is below 90%. Consider adjusting quantum coherence threshold or entanglement sensitivity.",
                    metrics.detection_success_rate * 100.0
                ),
                estimated_improvement: 15.0,
            });
        }

        // Memory efficiency recommendations
        if metrics.memory_efficiency < 0.8 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "Memory efficiency ({:.1}%) could be improved. Consider reducing maximum superposition states or enabling memory compression.",
                    metrics.memory_efficiency * 100.0
                ),
                estimated_improvement: 20.0,
            });
        }

        // Error rate recommendations
        if metrics.quantum_error_rate > 0.05 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Reliability,
                priority: RecommendationPriority::High,
                description: format!(
                    "Quantum error rate ({:.2}%) exceeds acceptable threshold. Consider improving quantum state preparation or reducing noise.",
                    metrics.quantum_error_rate * 100.0
                ),
                estimated_improvement: 25.0,
            });
        }

        // Throughput recommendations
        if metrics.patterns_per_second < 10.0 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Throughput,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "Pattern detection throughput ({:.1} patterns/sec) is low. Consider parallel processing or batch optimization.",
                    metrics.patterns_per_second
                ),
                estimated_improvement: 50.0,
            });
        }

        recommendations
    }

    /// Reset performance metrics for new session
    pub fn reset_metrics(&mut self) {
        self.current_metrics = QuantumPerformanceMetrics {
            avg_detection_latency_us: 0.0,
            detection_success_rate: 1.0,
            coherence_preservation_rate: 1.0,
            memory_efficiency: 1.0,
            cpu_utilization: 0.0,
            gpu_utilization: None,
            patterns_per_second: 0.0,
            total_calculations: 0,
            quantum_error_rate: 0.0,
        };
        self.performance_history.clear();
        self.session_start = Instant::now();

        info!("Performance metrics reset for new session");
    }

    // Private helper methods

    fn check_performance_targets(&self, latency_us: u64) {
        if latency_us > self.target_latency_us * 2 {
            warn!(
                "Detection latency {}μs significantly exceeds target {}μs",
                latency_us, self.target_latency_us
            );
        } else if latency_us <= self.target_latency_us {
            debug!(
                "Detection latency {}μs meets target {}μs",
                latency_us, self.target_latency_us
            );
        }
    }

    fn store_metrics_snapshot(&mut self) {
        let snapshot = TimestampedMetrics {
            timestamp: Instant::now(),
            metrics: self.current_metrics.clone(),
        };

        self.performance_history.push_back(snapshot);

        // Keep only recent history to prevent memory bloat
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }
    }

    fn calculate_trend(&self, old_value: f64, new_value: f64) -> TrendDirection {
        let change_threshold = 0.05; // 5% change threshold
        let relative_change = (new_value - old_value) / old_value.max(1e-6);

        if relative_change > change_threshold {
            TrendDirection::Improving
        } else if relative_change < -change_threshold {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub latency_trend: TrendDirection,
    pub success_rate_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub error_rate_trend: TrendDirection,
}

/// Trend direction for performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Performance improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub estimated_improvement: f64, // Percentage improvement
}

/// Category of performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationCategory {
    Latency,
    Throughput,
    Memory,
    Accuracy,
    Reliability,
}

/// Priority of performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            latency_trend: TrendDirection::Stable,
            success_rate_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Stable,
            error_rate_trend: TrendDirection::Stable,
        }
    }
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::Improving => write!(f, "↗ Improving"),
            TrendDirection::Stable => write!(f, "→ Stable"),
            TrendDirection::Degrading => write!(f, "↘ Degrading"),
        }
    }
}

impl std::fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendationPriority::Low => write!(f, "🟢 Low"),
            RecommendationPriority::Medium => write!(f, "🟡 Medium"),
            RecommendationPriority::High => write!(f, "🔴 High"),
            RecommendationPriority::Critical => write!(f, "🚨 Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_monitoring() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record some operations
        monitor.record_detection(50, 0.9);
        monitor.record_detection(75, 0.8);
        monitor.record_detection(100, 0.95);
        
        let metrics = monitor.get_current_metrics();
        assert!(metrics.avg_detection_latency_us > 0.0);
        assert!(metrics.detection_success_rate > 0.0);
        assert_eq!(metrics.total_calculations, 3);
    }

    #[test]
    fn test_recommendations() {
        let mut monitor = PerformanceMonitor::new();
        
        // Simulate poor performance
        for _ in 0..10 {
            monitor.record_detection(200, 0.5); // High latency, low confidence
        }
        
        let recommendations = monitor.generate_recommendations();
        assert!(!recommendations.is_empty());
        
        // Should have latency and accuracy recommendations
        let has_latency_rec = recommendations.iter()
            .any(|r| r.category == RecommendationCategory::Latency);
        let has_accuracy_rec = recommendations.iter()
            .any(|r| r.category == RecommendationCategory::Accuracy);
            
        assert!(has_latency_rec);
        assert!(has_accuracy_rec);
    }

    #[test]
    fn test_trend_analysis() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record initial metrics
        for _ in 0..5 {
            monitor.record_detection(100, 0.8);
        }
        
        // Simulate some delay to separate measurements
        thread::sleep(Duration::from_millis(10));
        
        // Record improved metrics
        for _ in 0..5 {
            monitor.record_detection(50, 0.95);
        }
        
        let trends = monitor.get_performance_trends();
        // Note: In a real scenario with more data points, trends would be more meaningful
        assert!(matches!(trends.latency_trend, TrendDirection::Improving | TrendDirection::Stable));
    }

    #[test]
    fn test_error_tracking() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record successful operations
        for _ in 0..10 {
            monitor.record_detection(50, 0.9);
        }
        
        // Record some errors
        monitor.record_quantum_error();
        monitor.record_quantum_error();
        
        let metrics = monitor.get_current_metrics();
        assert!(metrics.quantum_error_rate > 0.0);
        assert!(metrics.quantum_error_rate < 1.0);
    }
}
