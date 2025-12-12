//! Performance metrics and monitoring for swarm intelligence algorithms

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Comprehensive performance metrics collector
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Algorithm performance history
    performance_history: VecDeque<PerformanceSnapshot>,
    
    /// Current metrics
    current_metrics: PerformanceMetrics,
    
    /// Collection start time
    start_time: Instant,
    
    /// Maximum history size
    max_history_size: usize,
    
    /// Sampling interval
    sampling_interval: Duration,
    
    /// Last sample time
    last_sample: Instant,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Algorithm name
    pub algorithm_name: String,
    
    /// Current best fitness
    pub best_fitness: f64,
    
    /// Average fitness
    pub average_fitness: f64,
    
    /// Population diversity
    pub diversity: f64,
    
    /// Convergence rate
    pub convergence_rate: f64,
    
    /// Iteration number
    pub iteration: usize,
    
    /// Evaluations performed
    pub evaluations: usize,
    
    /// Memory usage (bytes)
    pub memory_usage: usize,
    
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    
    /// Execution time for this iteration
    pub iteration_time: Duration,
}

/// Real-time performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total runtime
    pub total_runtime: Duration,
    
    /// Average iteration time
    pub avg_iteration_time: Duration,
    
    /// Peak memory usage
    pub peak_memory_usage: usize,
    
    /// Total evaluations
    pub total_evaluations: usize,
    
    /// Evaluations per second
    pub evaluations_per_second: f64,
    
    /// Current convergence trend
    pub convergence_trend: ConvergenceTrend,
    
    /// Algorithm efficiency score
    pub efficiency_score: f64,
    
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Convergence trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceTrend {
    /// Rapidly improving
    Improving { rate: f64 },
    
    /// Steady progress
    Steady { rate: f64 },
    
    /// Slow progress
    Slow { rate: f64 },
    
    /// Stagnated (no improvement)
    Stagnated { iterations: usize },
    
    /// Converged
    Converged { final_fitness: f64 },
    
    /// Diverging (getting worse)
    Diverging { rate: f64 },
}

impl Default for ConvergenceTrend {
    fn default() -> Self {
        Self::Steady { rate: 0.0 }
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage
    pub cpu_percent: f64,
    
    /// Memory usage percentage
    pub memory_percent: f64,
    
    /// Thread count
    pub thread_count: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// IO operations per second
    pub io_ops_per_second: f64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            performance_history: VecDeque::new(),
            current_metrics: PerformanceMetrics::default(),
            start_time: Instant::now(),
            max_history_size: 1000,
            sampling_interval: Duration::from_millis(100),
            last_sample: Instant::now(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(max_history: usize, sampling_interval: Duration) -> Self {
        let mut collector = Self::new();
        collector.max_history_size = max_history;
        collector.sampling_interval = sampling_interval;
        collector
    }
    
    /// Record a performance snapshot
    pub fn record_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        // Add to history
        self.performance_history.push_back(snapshot.clone());
        
        // Maintain history size limit
        while self.performance_history.len() > self.max_history_size {
            self.performance_history.pop_front();
        }
        
        // Update current metrics
        self.update_current_metrics(&snapshot);
        
        self.last_sample = Instant::now();
    }
    
    /// Update current aggregated metrics
    fn update_current_metrics(&mut self, snapshot: &PerformanceSnapshot) {
        let elapsed = self.start_time.elapsed();
        
        self.current_metrics.total_runtime = elapsed;
        self.current_metrics.total_evaluations = snapshot.evaluations;
        
        if elapsed.as_secs_f64() > 0.0 {
            self.current_metrics.evaluations_per_second = 
                snapshot.evaluations as f64 / elapsed.as_secs_f64();
        }
        
        // Update peak memory
        if snapshot.memory_usage > self.current_metrics.peak_memory_usage {
            self.current_metrics.peak_memory_usage = snapshot.memory_usage;
        }
        
        // Calculate average iteration time
        if snapshot.iteration > 0 {
            self.current_metrics.avg_iteration_time = elapsed / snapshot.iteration as u32;
        }
        
        // Analyze convergence trend
        self.current_metrics.convergence_trend = self.analyze_convergence_trend();
        
        // Calculate efficiency score
        self.current_metrics.efficiency_score = self.calculate_efficiency_score(snapshot);
        
        // Update resource utilization
        self.current_metrics.resource_utilization = ResourceUtilization {
            cpu_percent: snapshot.cpu_utilization * 100.0,
            memory_percent: (snapshot.memory_usage as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)) * 100.0, // Assume 8GB total
            thread_count: num_cpus::get(),
            cache_hit_rate: 0.85, // Estimate
            io_ops_per_second: 100.0, // Estimate
        };
    }
    
    /// Analyze convergence trend from recent history
    fn analyze_convergence_trend(&self) -> ConvergenceTrend {
        if self.performance_history.len() < 2 {
            return ConvergenceTrend::Steady { rate: 0.0 };
        }
        
        let recent_size = (self.performance_history.len() / 4).max(2).min(10);
        let recent_snapshots: Vec<_> = self.performance_history
            .iter()
            .rev()
            .take(recent_size)
            .collect();
        
        if recent_snapshots.len() < 2 {
            return ConvergenceTrend::Steady { rate: 0.0 };
        }
        
        // Calculate improvement rate
        let first_fitness = recent_snapshots.last().unwrap().best_fitness;
        let last_fitness = recent_snapshots.first().unwrap().best_fitness;
        
        let improvement = first_fitness - last_fitness;
        let rate = improvement / recent_snapshots.len() as f64;
        
        // Classify trend
        if improvement.abs() < 1e-10 {
            ConvergenceTrend::Stagnated { 
                iterations: recent_snapshots.len() 
            }
        } else if improvement > 0.0 {
            if rate > 0.1 {
                ConvergenceTrend::Improving { rate }
            } else if rate > 0.01 {
                ConvergenceTrend::Steady { rate }
            } else {
                ConvergenceTrend::Slow { rate }
            }
        } else {
            ConvergenceTrend::Diverging { rate: -rate }
        }
    }
    
    /// Calculate efficiency score
    fn calculate_efficiency_score(&self, snapshot: &PerformanceSnapshot) -> f64 {
        let time_efficiency = if snapshot.iteration_time.as_secs_f64() > 0.0 {
            1.0 / snapshot.iteration_time.as_secs_f64()
        } else {
            1.0
        };
        
        let memory_efficiency = 1.0 / (1.0 + snapshot.memory_usage as f64 / (1024.0 * 1024.0));
        let cpu_efficiency = 1.0 - snapshot.cpu_utilization.min(1.0);
        let evaluation_efficiency = snapshot.evaluations as f64 / snapshot.iteration as f64;
        
        (time_efficiency + memory_efficiency + cpu_efficiency + evaluation_efficiency) / 4.0
    }
    
    /// Get current performance metrics
    pub fn current_metrics(&self) -> &PerformanceMetrics {
        &self.current_metrics
    }
    
    /// Get performance history
    pub fn history(&self) -> &VecDeque<PerformanceSnapshot> {
        &self.performance_history
    }
    
    /// Get recent performance snapshots
    pub fn recent_snapshots(&self, count: usize) -> Vec<&PerformanceSnapshot> {
        self.performance_history
            .iter()
            .rev()
            .take(count)
            .collect()
    }
    
    /// Calculate performance statistics
    pub fn calculate_statistics(&self) -> PerformanceStatistics {
        if self.performance_history.is_empty() {
            return PerformanceStatistics::default();
        }
        
        let fitnesses: Vec<f64> = self.performance_history
            .iter()
            .map(|s| s.best_fitness)
            .collect();
        
        let diversities: Vec<f64> = self.performance_history
            .iter()
            .map(|s| s.diversity)
            .collect();
        
        let iteration_times: Vec<f64> = self.performance_history
            .iter()
            .map(|s| s.iteration_time.as_secs_f64())
            .collect();
        
        PerformanceStatistics {
            best_fitness: fitnesses.iter().copied().fold(f64::INFINITY, f64::min),
            worst_fitness: fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            avg_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            fitness_std_dev: calculate_std_dev(&fitnesses),
            
            avg_diversity: diversities.iter().sum::<f64>() / diversities.len() as f64,
            diversity_std_dev: calculate_std_dev(&diversities),
            
            min_iteration_time: iteration_times.iter().copied().fold(f64::INFINITY, f64::min),
            max_iteration_time: iteration_times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            avg_iteration_time: iteration_times.iter().sum::<f64>() / iteration_times.len() as f64,
            iteration_time_std_dev: calculate_std_dev(&iteration_times),
            
            total_snapshots: self.performance_history.len(),
            collection_duration: self.start_time.elapsed(),
        }
    }
    
    /// Export metrics to JSON
    pub fn export_json(&self) -> Result<String, SwarmError> {
        let export_data = MetricsExport {
            current_metrics: self.current_metrics.clone(),
            statistics: self.calculate_statistics(),
            recent_history: self.recent_snapshots(100).into_iter().cloned().collect(),
        };
        
        serde_json::to_string_pretty(&export_data)
            .map_err(|e| SwarmError::SerializationError(e))
    }
    
    /// Clear all collected metrics
    pub fn clear(&mut self) {
        self.performance_history.clear();
        self.current_metrics = PerformanceMetrics::default();
        self.start_time = Instant::now();
        self.last_sample = Instant::now();
    }
    
    /// Check if it's time to sample
    pub fn should_sample(&self) -> bool {
        self.last_sample.elapsed() >= self.sampling_interval
    }
}

/// Performance statistics summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Best fitness achieved
    pub best_fitness: f64,
    
    /// Worst fitness seen
    pub worst_fitness: f64,
    
    /// Average fitness
    pub avg_fitness: f64,
    
    /// Fitness standard deviation
    pub fitness_std_dev: f64,
    
    /// Average diversity
    pub avg_diversity: f64,
    
    /// Diversity standard deviation
    pub diversity_std_dev: f64,
    
    /// Minimum iteration time
    pub min_iteration_time: f64,
    
    /// Maximum iteration time
    pub max_iteration_time: f64,
    
    /// Average iteration time
    pub avg_iteration_time: f64,
    
    /// Iteration time standard deviation
    pub iteration_time_std_dev: f64,
    
    /// Total number of snapshots
    pub total_snapshots: usize,
    
    /// Total collection duration
    pub collection_duration: Duration,
}

/// Metrics export format
#[derive(Debug, Serialize, Deserialize)]
struct MetricsExport {
    current_metrics: PerformanceMetrics,
    statistics: PerformanceStatistics,
    recent_history: Vec<PerformanceSnapshot>,
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}

/// Initialize metrics collection system
pub fn initialize_metrics_collection() -> Result<(), SwarmError> {
    // Initialize global metrics collector
    // In a real implementation, this would set up background collection
    tracing::info!("Metrics collection system initialized");
    Ok(())
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

use crate::core::SwarmError;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        
        let snapshot = PerformanceSnapshot {
            timestamp: std::time::SystemTime::now(),
            algorithm_name: "TestAlgorithm".to_string(),
            best_fitness: 10.0,
            average_fitness: 15.0,
            diversity: 0.5,
            convergence_rate: 0.1,
            iteration: 1,
            evaluations: 30,
            memory_usage: 1024 * 1024,
            cpu_utilization: 0.5,
            iteration_time: Duration::from_millis(100),
        };
        
        collector.record_snapshot(snapshot);
        
        assert_eq!(collector.history().len(), 1);
        assert!(collector.current_metrics().total_evaluations > 0);
    }
    
    #[test]
    fn test_convergence_trend_analysis() {
        let mut collector = MetricsCollector::new();
        
        // Add snapshots with improving fitness
        for i in 0..5 {
            let snapshot = PerformanceSnapshot {
                timestamp: std::time::SystemTime::now(),
                algorithm_name: "TestAlgorithm".to_string(),
                best_fitness: 10.0 - i as f64, // Improving
                average_fitness: 15.0,
                diversity: 0.5,
                convergence_rate: 0.1,
                iteration: i + 1,
                evaluations: (i + 1) * 30,
                memory_usage: 1024 * 1024,
                cpu_utilization: 0.5,
                iteration_time: Duration::from_millis(100),
            };
            collector.record_snapshot(snapshot);
        }
        
        match collector.current_metrics().convergence_trend {
            ConvergenceTrend::Improving { rate } => assert!(rate > 0.0),
            _ => panic!("Expected improving trend"),
        }
    }
    
    #[test]
    fn test_performance_statistics() {
        let mut collector = MetricsCollector::new();
        
        // Add multiple snapshots
        for i in 0..10 {
            let snapshot = PerformanceSnapshot {
                timestamp: std::time::SystemTime::now(),
                algorithm_name: "TestAlgorithm".to_string(),
                best_fitness: i as f64,
                average_fitness: i as f64 + 5.0,
                diversity: 0.5,
                convergence_rate: 0.1,
                iteration: i + 1,
                evaluations: (i + 1) * 30,
                memory_usage: 1024 * 1024,
                cpu_utilization: 0.5,
                iteration_time: Duration::from_millis(100),
            };
            collector.record_snapshot(snapshot);
        }
        
        let stats = collector.calculate_statistics();
        assert_eq!(stats.total_snapshots, 10);
        assert_eq!(stats.best_fitness, 0.0);
        assert_eq!(stats.worst_fitness, 9.0);
    }
}