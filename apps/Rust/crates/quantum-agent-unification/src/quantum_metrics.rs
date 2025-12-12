//! Quantum performance metrics and monitoring system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::quantum_optimizer::QuantumAlgorithm;

/// Comprehensive quantum performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// System-wide metrics
    pub system_metrics: SystemQuantumMetrics,
    /// Per-algorithm metrics
    pub algorithm_metrics: HashMap<QuantumAlgorithm, AlgorithmMetrics>,
    /// Quantum computing specific metrics
    pub quantum_computing_metrics: QuantumComputingMetrics,
    /// Performance benchmarks
    pub performance_benchmarks: PerformanceBenchmarks,
    /// Real-time monitoring data
    pub real_time_metrics: RealTimeMetrics,
}

/// System-wide quantum metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemQuantumMetrics {
    /// Total quantum operations performed
    pub total_operations: u64,
    /// System uptime
    pub uptime: Duration,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Thread pool utilization
    pub thread_utilization: f64,
    /// Quantum error rate
    pub quantum_error_rate: f64,
    /// System health score (0.0 to 1.0)
    pub health_score: f64,
}

/// Per-algorithm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    /// Algorithm type
    pub algorithm: QuantumAlgorithm,
    /// Total iterations completed
    pub iterations_completed: u64,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Average convergence rate
    pub convergence_rate: f64,
    /// Quantum state coherence time
    pub coherence_time: f64,
    /// Number of quantum tunneling events
    pub tunneling_events: u64,
    /// Entanglement correlation strength
    pub entanglement_strength: f64,
    /// Processing time statistics
    pub processing_times: ProcessingTimeStats,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Quantum computing specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumComputingMetrics {
    /// Total qubits in use
    pub total_qubits: usize,
    /// Quantum gate operations per second
    pub gate_operations_per_sec: f64,
    /// Quantum measurement events
    pub measurement_events: u64,
    /// Decoherence events detected
    pub decoherence_events: u64,
    /// Quantum error correction operations
    pub error_correction_ops: u64,
    /// Superposition state count
    pub superposition_states: u64,
    /// Entangled qubit pairs
    pub entangled_pairs: u64,
    /// Quantum interference patterns generated
    pub interference_patterns: u64,
    /// Quantum phase evolution tracking
    pub phase_evolution: Vec<f64>,
    /// Quantum fidelity measurements
    pub quantum_fidelity: f64,
    /// Bell state violations detected
    pub bell_violations: u64,
}

/// Performance benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarks {
    /// Operations per second
    pub ops_per_second: f64,
    /// Memory throughput (MB/s)
    pub memory_throughput: f64,
    /// Latency percentiles
    pub latency_percentiles: LatencyPercentiles,
    /// Quantum speedup factor
    pub quantum_speedup: f64,
    /// SIMD acceleration factor
    pub simd_acceleration: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Cache hit rates
    pub cache_metrics: CacheMetrics,
}

/// Latency measurement percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p99_9: Duration,
    pub max: Duration,
    pub min: Duration,
    pub mean: Duration,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub memory_hit_rate: f64,
    pub quantum_state_cache_hits: u64,
    pub quantum_state_cache_misses: u64,
}

/// Real-time monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    /// Current timestamp
    pub timestamp: Instant,
    /// Instantaneous metrics
    pub instantaneous: InstantaneousMetrics,
    /// Rolling window metrics
    pub rolling_window: RollingWindowMetrics,
    /// Trend analysis
    pub trends: TrendMetrics,
}

/// Instantaneous performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantaneousMetrics {
    /// Current CPU usage
    pub cpu_usage: f64,
    /// Current memory usage
    pub memory_usage: f64,
    /// Current quantum operations per second
    pub quantum_ops_per_sec: f64,
    /// Current coherence level
    pub coherence_level: f64,
    /// Active quantum states
    pub active_quantum_states: usize,
    /// Current error rate
    pub error_rate: f64,
}

/// Rolling window aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingWindowMetrics {
    /// Window size in seconds
    pub window_size: u64,
    /// Average metrics over window
    pub averages: InstantaneousMetrics,
    /// Standard deviations
    pub std_devs: InstantaneousMetrics,
    /// Min/max values
    pub min_values: InstantaneousMetrics,
    pub max_values: InstantaneousMetrics,
}

/// Trend analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendMetrics {
    /// Performance trend (improving/degrading)
    pub performance_trend: f64, // -1.0 to 1.0
    /// Convergence trend
    pub convergence_trend: f64,
    /// Resource utilization trend
    pub resource_trend: f64,
    /// Quantum coherence trend
    pub coherence_trend: f64,
    /// Predicted next value
    pub performance_prediction: f64,
}

/// Processing time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeStats {
    /// Total processing time
    pub total_time: Duration,
    /// Average processing time
    pub average_time: Duration,
    /// Minimum processing time
    pub min_time: Duration,
    /// Maximum processing time
    pub max_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// 95th percentile
    pub p95_time: Duration,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU cores utilized
    pub cpu_cores_used: f64,
    /// Memory bandwidth utilized
    pub memory_bandwidth_used: f64,
    /// SIMD units utilized
    pub simd_units_used: f64,
    /// Thread pool utilization
    pub thread_pool_used: f64,
    /// GPU utilization (if available)
    pub gpu_utilization: Option<f64>,
}

impl Default for QuantumMetrics {
    fn default() -> Self {
        Self {
            system_metrics: SystemQuantumMetrics::default(),
            algorithm_metrics: HashMap::new(),
            quantum_computing_metrics: QuantumComputingMetrics::default(),
            performance_benchmarks: PerformanceBenchmarks::default(),
            real_time_metrics: RealTimeMetrics::default(),
        }
    }
}

impl Default for SystemQuantumMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            uptime: Duration::from_secs(0),
            memory_usage_bytes: 0,
            cpu_utilization: 0.0,
            simd_utilization: 0.0,
            thread_utilization: 0.0,
            quantum_error_rate: 0.0,
            health_score: 1.0,
        }
    }
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            algorithm: QuantumAlgorithm::QuantumParticleSwarm,
            iterations_completed: 0,
            best_fitness: f64::INFINITY,
            convergence_rate: 0.0,
            coherence_time: 1.0,
            tunneling_events: 0,
            entanglement_strength: 0.0,
            processing_times: ProcessingTimeStats::default(),
            success_rate: 0.0,
            resource_utilization: ResourceUtilization::default(),
        }
    }
}

impl Default for QuantumComputingMetrics {
    fn default() -> Self {
        Self {
            total_qubits: 0,
            gate_operations_per_sec: 0.0,
            measurement_events: 0,
            decoherence_events: 0,
            error_correction_ops: 0,
            superposition_states: 0,
            entangled_pairs: 0,
            interference_patterns: 0,
            phase_evolution: Vec::new(),
            quantum_fidelity: 1.0,
            bell_violations: 0,
        }
    }
}

impl Default for PerformanceBenchmarks {
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            memory_throughput: 0.0,
            latency_percentiles: LatencyPercentiles::default(),
            quantum_speedup: 1.0,
            simd_acceleration: 1.0,
            parallel_efficiency: 1.0,
            cache_metrics: CacheMetrics::default(),
        }
    }
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: Duration::from_nanos(0),
            p90: Duration::from_nanos(0),
            p95: Duration::from_nanos(0),
            p99: Duration::from_nanos(0),
            p99_9: Duration::from_nanos(0),
            max: Duration::from_nanos(0),
            min: Duration::from_nanos(0),
            mean: Duration::from_nanos(0),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            l1_hit_rate: 0.0,
            l2_hit_rate: 0.0,
            l3_hit_rate: 0.0,
            memory_hit_rate: 0.0,
            quantum_state_cache_hits: 0,
            quantum_state_cache_misses: 0,
        }
    }
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            instantaneous: InstantaneousMetrics::default(),
            rolling_window: RollingWindowMetrics::default(),
            trends: TrendMetrics::default(),
        }
    }
}

impl Default for InstantaneousMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            quantum_ops_per_sec: 0.0,
            coherence_level: 1.0,
            active_quantum_states: 0,
            error_rate: 0.0,
        }
    }
}

impl Default for RollingWindowMetrics {
    fn default() -> Self {
        Self {
            window_size: 60, // 60 seconds default
            averages: InstantaneousMetrics::default(),
            std_devs: InstantaneousMetrics::default(),
            min_values: InstantaneousMetrics::default(),
            max_values: InstantaneousMetrics::default(),
        }
    }
}

impl Default for TrendMetrics {
    fn default() -> Self {
        Self {
            performance_trend: 0.0,
            convergence_trend: 0.0,
            resource_trend: 0.0,
            coherence_trend: 0.0,
            performance_prediction: 0.0,
        }
    }
}

impl Default for ProcessingTimeStats {
    fn default() -> Self {
        Self {
            total_time: Duration::from_nanos(0),
            average_time: Duration::from_nanos(0),
            min_time: Duration::from_nanos(0),
            max_time: Duration::from_nanos(0),
            std_dev: Duration::from_nanos(0),
            p95_time: Duration::from_nanos(0),
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_cores_used: 0.0,
            memory_bandwidth_used: 0.0,
            simd_units_used: 0.0,
            thread_pool_used: 0.0,
            gpu_utilization: None,
        }
    }
}

/// Metrics collector for real-time monitoring
pub struct QuantumMetricsCollector {
    metrics: QuantumMetrics,
    start_time: Instant,
    history: Vec<(Instant, QuantumMetrics)>,
    max_history_size: usize,
}

impl QuantumMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: QuantumMetrics::default(),
            start_time: Instant::now(),
            history: Vec::new(),
            max_history_size: 1000,
        }
    }
    
    /// Update system metrics
    pub fn update_system_metrics(&mut self, cpu_usage: f64, memory_usage: usize, simd_util: f64) {
        self.metrics.system_metrics.cpu_utilization = cpu_usage;
        self.metrics.system_metrics.memory_usage_bytes = memory_usage;
        self.metrics.system_metrics.simd_utilization = simd_util;
        self.metrics.system_metrics.uptime = self.start_time.elapsed();
    }
    
    /// Update algorithm-specific metrics
    pub fn update_algorithm_metrics(&mut self, algorithm: QuantumAlgorithm, metrics: AlgorithmMetrics) {
        self.metrics.algorithm_metrics.insert(algorithm, metrics);
    }
    
    /// Update quantum computing metrics
    pub fn update_quantum_metrics(&mut self, qubits: usize, gate_ops: f64, coherence: f64) {
        self.metrics.quantum_computing_metrics.total_qubits = qubits;
        self.metrics.quantum_computing_metrics.gate_operations_per_sec = gate_ops;
        self.metrics.quantum_computing_metrics.quantum_fidelity = coherence;
    }
    
    /// Record quantum operation
    pub fn record_quantum_operation(&mut self, operation_type: &str, duration: Duration) {
        self.metrics.system_metrics.total_operations += 1;
        
        // Update performance benchmarks
        let ops_per_sec = 1.0 / duration.as_secs_f64();
        if ops_per_sec > self.metrics.performance_benchmarks.ops_per_second {
            self.metrics.performance_benchmarks.ops_per_second = ops_per_sec;
        }
    }
    
    /// Record quantum tunneling event
    pub fn record_tunneling_event(&mut self, algorithm: QuantumAlgorithm) {
        if let Some(metrics) = self.metrics.algorithm_metrics.get_mut(&algorithm) {
            metrics.tunneling_events += 1;
        }
    }
    
    /// Record entanglement creation
    pub fn record_entanglement(&mut self, strength: f64) {
        self.metrics.quantum_computing_metrics.entangled_pairs += 1;
        
        // Update average entanglement strength
        for (_, metrics) in &mut self.metrics.algorithm_metrics {
            metrics.entanglement_strength = (metrics.entanglement_strength + strength) / 2.0;
        }
    }
    
    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> &QuantumMetrics {
        &self.metrics
    }
    
    /// Get historical metrics
    pub fn get_history(&self) -> &[(Instant, QuantumMetrics)] {
        &self.history
    }
    
    /// Export metrics to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.metrics)
    }
    
    /// Calculate quantum efficiency score
    pub fn quantum_efficiency_score(&self) -> f64 {
        let coherence_score = self.metrics.quantum_computing_metrics.quantum_fidelity;
        let performance_score = self.metrics.performance_benchmarks.quantum_speedup / 10.0; // Normalize
        let resource_score = 1.0 - self.metrics.system_metrics.cpu_utilization / 100.0;
        
        (coherence_score + performance_score + resource_score) / 3.0
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "=== Quantum Agent Optimization Report ===\n\
            System Uptime: {:?}\n\
            Total Operations: {}\n\
            Quantum Efficiency: {:.2}%\n\
            Active Qubits: {}\n\
            Gate Operations/sec: {:.0}\n\
            Quantum Fidelity: {:.3}\n\
            Entangled Pairs: {}\n\
            CPU Utilization: {:.1}%\n\
            Memory Usage: {:.1} MB\n\
            SIMD Utilization: {:.1}%\n\
            Quantum Speedup: {:.2}x\n",
            self.metrics.system_metrics.uptime,
            self.metrics.system_metrics.total_operations,
            self.quantum_efficiency_score() * 100.0,
            self.metrics.quantum_computing_metrics.total_qubits,
            self.metrics.quantum_computing_metrics.gate_operations_per_sec,
            self.metrics.quantum_computing_metrics.quantum_fidelity,
            self.metrics.quantum_computing_metrics.entangled_pairs,
            self.metrics.system_metrics.cpu_utilization,
            self.metrics.system_metrics.memory_usage_bytes as f64 / 1_048_576.0,
            self.metrics.system_metrics.simd_utilization,
            self.metrics.performance_benchmarks.quantum_speedup,
        )
    }
}

impl Default for QuantumMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector() {
        let mut collector = QuantumMetricsCollector::new();
        collector.update_system_metrics(50.0, 1024 * 1024, 75.0);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.system_metrics.cpu_utilization, 50.0);
        assert_eq!(metrics.system_metrics.memory_usage_bytes, 1024 * 1024);
        assert_eq!(metrics.system_metrics.simd_utilization, 75.0);
    }
    
    #[test]
    fn test_quantum_efficiency_score() {
        let collector = QuantumMetricsCollector::new();
        let score = collector.quantum_efficiency_score();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_metrics_json_export() {
        let collector = QuantumMetricsCollector::new();
        let json = collector.export_json();
        assert!(json.is_ok());
        assert!(!json.unwrap().is_empty());
    }
}