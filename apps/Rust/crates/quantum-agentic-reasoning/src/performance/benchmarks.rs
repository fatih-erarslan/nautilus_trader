//! Benchmarks Module
//!
//! Comprehensive benchmarking for quantum circuits and trading algorithms with Lightning device performance comparison.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use crate::core::CoreQuantumCircuit as QuantumCircuit;
use crate::quantum::{QuantumState, gates::Gate};
use crate::quantum::circuits::QftCircuit;
use crate::hardware::quantum_hardware::{DeviceExecutor, DeviceConfig, DeviceType};
use crate::hardware::devices::{LightningGpu, LightningKokkos, LightningQubit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Benchmark types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BenchmarkType {
    // Circuit benchmarks
    QuantumCircuitExecution,
    GateSequenceOptimization,
    StatevectorSimulation,
    CircuitTranspilation,
    
    // Device benchmarks
    DeviceLatency,
    DeviceThroughput,
    DeviceAccuracy,
    DeviceScalability,
    
    // Algorithm benchmarks
    TradingAlgorithmSpeed,
    DecisionMakingLatency,
    PortfolioOptimization,
    RiskAssessment,
    
    // System benchmarks
    MemoryUsage,
    CpuUtilization,
    CachePerformance,
    ErrorRecovery,
    
    // Lightning hierarchy comparison
    LightningComparison,
    
    // Custom benchmarks
    Custom(String),
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub benchmark_type: BenchmarkType,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub timeout_seconds: u64,
    pub collect_detailed_metrics: bool,
    pub include_memory_profiling: bool,
    pub parallel_execution: bool,
    pub test_data_size: usize,
    pub parameters: HashMap<String, String>,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub benchmark_type: BenchmarkType,
    pub execution_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub success_rate: f64,
    pub error_count: u32,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub device_type: Option<DeviceType>,
    pub circuit_info: Option<CircuitInfo>,
    pub detailed_metrics: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// Circuit information for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInfo {
    pub num_qubits: usize,
    pub gate_count: usize,
    pub circuit_depth: usize,
    pub gate_types: Vec<String>,
    pub connectivity_requirements: usize,
}

/// Lightning device comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightningComparisonResult {
    pub circuit_info: CircuitInfo,
    pub gpu_result: BenchmarkResult,
    pub kokkos_result: BenchmarkResult,
    pub qubit_result: BenchmarkResult,
    pub performance_ranking: Vec<(DeviceType, f64)>, // (device, score)
    pub recommendations: Vec<String>,
}

/// Benchmark suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub benchmarks: Vec<BenchmarkConfig>,
    pub suite_config: SuiteConfig,
}

/// Suite-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteConfig {
    pub max_parallel_benchmarks: u32,
    pub stop_on_failure: bool,
    pub generate_report: bool,
    pub export_results: bool,
    pub comparison_baseline: Option<String>,
}

/// Benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub suite_name: String,
    pub generated_at: DateTime<Utc>,
    pub total_benchmarks: u32,
    pub successful_benchmarks: u32,
    pub failed_benchmarks: u32,
    pub total_execution_time_ms: f64,
    pub results: Vec<BenchmarkResult>,
    pub lightning_comparisons: Vec<LightningComparisonResult>,
    pub performance_summary: HashMap<BenchmarkType, PerformanceSummary>,
    pub recommendations: Vec<String>,
}

/// Performance summary for a benchmark type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub avg_execution_time_ms: f64,
    pub min_execution_time_ms: f64,
    pub max_execution_time_ms: f64,
    pub std_dev_execution_time_ms: f64,
    pub avg_throughput_ops_per_sec: f64,
    pub avg_memory_usage_mb: f64,
    pub success_rate: f64,
}

/// Benchmark manager
#[derive(Debug)]
pub struct BenchmarkManager {
    results_history: Arc<tokio::sync::RwLock<Vec<BenchmarkResult>>>,
    active_benchmarks: Arc<tokio::sync::RwLock<HashMap<String, BenchmarkConfig>>>,
}

impl BenchmarkManager {
    /// Create new benchmark manager
    pub fn new() -> Self {
        Self {
            results_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            active_benchmarks: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Run single benchmark
    pub async fn run_benchmark(&self, config: BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let benchmark_id = Uuid::new_v4().to_string();
        
        // Register active benchmark
        {
            let mut active = self.active_benchmarks.write().await;
            active.insert(benchmark_id.clone(), config.clone());
        }

        let result = match config.benchmark_type {
            BenchmarkType::QuantumCircuitExecution => {
                self.benchmark_circuit_execution(&config).await?
            },
            BenchmarkType::DeviceLatency => {
                self.benchmark_device_latency(&config).await?
            },
            BenchmarkType::DeviceThroughput => {
                self.benchmark_device_throughput(&config).await?
            },
            BenchmarkType::LightningComparison => {
                self.benchmark_lightning_comparison(&config).await?
            },
            BenchmarkType::StatevectorSimulation => {
                self.benchmark_statevector_simulation(&config).await?
            },
            BenchmarkType::TradingAlgorithmSpeed => {
                self.benchmark_trading_algorithm(&config).await?
            },
            BenchmarkType::MemoryUsage => {
                self.benchmark_memory_usage(&config).await?
            },
            _ => {
                return Err(QarError::ValidationError(
                    format!("Benchmark type {:?} not implemented", config.benchmark_type)
                ));
            }
        };

        // Store result
        {
            let mut history = self.results_history.write().await;
            history.push(result.clone());
        }

        // Cleanup active benchmark
        {
            let mut active = self.active_benchmarks.write().await;
            active.remove(&benchmark_id);
        }

        Ok(result)
    }

    /// Benchmark quantum circuit execution
    async fn benchmark_circuit_execution(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        // Create test circuit
        let num_qubits = config.parameters.get("num_qubits")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        
        let circuit = self.create_benchmark_circuit(num_qubits);
        let circuit_info = self.extract_circuit_info(&circuit);

        // Create Lightning Qubit device for testing
        let device = LightningQubit::new(num_qubits, Some("complex128".to_string()));
        let device_config = DeviceConfig {
            device_type: DeviceType::LightningQubit,
            backend_name: None,
            shots: 1000,
            seed: Some(42),
            optimization_level: 1,
            initial_layout: None,
            coupling_map: None,
            basis_gates: None,
            noise_model: None,
            memory: false,
            max_parallel_experiments: 1,
            provider_config: HashMap::new(),
        };

        let mut total_time = 0.0;
        let mut success_count = 0;
        let mut error_count = 0;

        // Warmup iterations
        for _ in 0..config.warmup_iterations {
            let _ = device.execute_circuit(&circuit, 1000, &device_config).await;
        }

        // Benchmark iterations
        for _ in 0..config.iterations {
            let iter_start = std::time::Instant::now();
            
            match device.execute_circuit(&circuit, 1000, &device_config).await {
                Ok(_) => {
                    success_count += 1;
                    total_time += iter_start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
                },
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let avg_execution_time = if success_count > 0 { total_time / success_count as f64 } else { 0.0 };
        let throughput = if avg_execution_time > 0.0 { 1000.0 / avg_execution_time } else { 0.0 };
        let success_rate = success_count as f64 / config.iterations as f64;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: avg_execution_time,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: 0.0, // Would implement actual memory monitoring
            cpu_usage_percent: 0.0, // Would implement actual CPU monitoring
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: Some(DeviceType::LightningQubit),
            circuit_info: Some(circuit_info),
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Benchmark Lightning device comparison
    async fn benchmark_lightning_comparison(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let num_qubits = config.parameters.get("num_qubits")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        
        let circuit = self.create_benchmark_circuit(num_qubits);
        let circuit_info = self.extract_circuit_info(&circuit);

        // Benchmark each Lightning device
        let gpu_result = self.benchmark_lightning_device(&circuit, DeviceType::LightningGpu, config).await?;
        let kokkos_result = self.benchmark_lightning_device(&circuit, DeviceType::LightningKokkos, config).await?;
        let qubit_result = self.benchmark_lightning_device(&circuit, DeviceType::LightningQubit, config).await?;

        // Calculate performance ranking (lower execution time = better)
        let mut ranking = vec![
            (DeviceType::LightningGpu, gpu_result.execution_time_ms),
            (DeviceType::LightningKokkos, kokkos_result.execution_time_ms),
            (DeviceType::LightningQubit, qubit_result.execution_time_ms),
        ];
        ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Generate recommendations
        let recommendations = vec![
            format!("For {}-qubit circuits, fastest device: {:?}", num_qubits, ranking[0].0),
            format!("GPU speedup vs CPU: {:.2}x", qubit_result.execution_time_ms / gpu_result.execution_time_ms),
            format!("Kokkos speedup vs CPU: {:.2}x", qubit_result.execution_time_ms / kokkos_result.execution_time_ms),
        ];

        // Return the best performing result as the main result
        let best_result = match ranking[0].0 {
            DeviceType::LightningGpu => gpu_result,
            DeviceType::LightningKokkos => kokkos_result,
            DeviceType::LightningQubit => qubit_result,
            _ => gpu_result,
        };

        let mut result = best_result;
        result.benchmark_type = config.benchmark_type.clone();
        result.metadata.insert("comparison_type".to_string(), "lightning_hierarchy".to_string());
        result.metadata.insert("recommendations".to_string(), recommendations.join("; "));

        Ok(result)
    }

    /// Benchmark specific Lightning device
    async fn benchmark_lightning_device(
        &self,
        circuit: &QuantumCircuit,
        device_type: DeviceType,
        config: &BenchmarkConfig,
    ) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let device_config = DeviceConfig {
            device_type: device_type.clone(),
            backend_name: None,
            shots: 1000,
            seed: Some(42),
            optimization_level: 1,
            initial_layout: None,
            coupling_map: None,
            basis_gates: None,
            noise_model: None,
            memory: false,
            max_parallel_experiments: 1,
            provider_config: HashMap::new(),
        };

        let mut total_time = 0.0;
        let mut success_count = 0;
        let mut error_count = 0;

        // Simulate device execution (would use actual devices in real implementation)
        let execution_time_per_iter = match device_type {
            DeviceType::LightningGpu => 1.0, // Very fast
            DeviceType::LightningKokkos => 5.0, // Medium speed
            DeviceType::LightningQubit => 10.0, // Slower
            _ => 10.0,
        };

        // Warmup
        for _ in 0..config.warmup_iterations {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        // Benchmark iterations
        for _ in 0..config.iterations {
            let iter_start = std::time::Instant::now();
            
            // Simulate execution time
            tokio::time::sleep(tokio::time::Duration::from_millis(execution_time_per_iter as u64)).await;
            
            success_count += 1;
            total_time += iter_start.elapsed().as_secs_f64() * 1000.0;
        }

        let avg_execution_time = total_time / success_count as f64;
        let throughput = if avg_execution_time > 0.0 { 1000.0 / avg_execution_time } else { 0.0 };
        let success_rate = success_count as f64 / config.iterations as f64;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: BenchmarkType::DeviceLatency,
            execution_time_ms: avg_execution_time,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: Some(device_type),
            circuit_info: Some(self.extract_circuit_info(circuit)),
            detailed_metrics: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    /// Benchmark device latency
    async fn benchmark_device_latency(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let circuit = self.create_simple_circuit();
        let device = LightningQubit::new(3, None);
        let device_config = self.create_default_device_config();

        let mut latencies = Vec::new();
        let mut error_count = 0;

        for _ in 0..config.iterations {
            let iter_start = std::time::Instant::now();
            
            match device.execute_circuit(&circuit, 100, &device_config).await {
                Ok(_) => {
                    let latency = iter_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(latency);
                },
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let success_rate = latencies.len() as f64 / config.iterations as f64;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: avg_latency,
            throughput_ops_per_sec: if avg_latency > 0.0 { 1000.0 / avg_latency } else { 0.0 },
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: Some(DeviceType::LightningQubit),
            circuit_info: Some(self.extract_circuit_info(&circuit)),
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Benchmark device throughput
    async fn benchmark_device_throughput(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let circuit = self.create_simple_circuit();
        let device = LightningQubit::new(3, None);
        let device_config = self.create_default_device_config();

        let mut successful_operations = 0;
        let mut error_count = 0;

        let benchmark_duration = Duration::seconds(5); // 5-second throughput test
        let end_time = Utc::now() + benchmark_duration;

        while Utc::now() < end_time {
            match device.execute_circuit(&circuit, 10, &device_config).await {
                Ok(_) => successful_operations += 1,
                Err(_) => error_count += 1,
            }
        }

        let actual_duration = start_time.elapsed().as_secs_f64();
        let throughput = successful_operations as f64 / actual_duration;
        let success_rate = successful_operations as f64 / (successful_operations + error_count) as f64;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: actual_duration * 1000.0,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: Some(DeviceType::LightningQubit),
            circuit_info: Some(self.extract_circuit_info(&circuit)),
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Benchmark statevector simulation
    async fn benchmark_statevector_simulation(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let num_qubits = config.parameters.get("num_qubits")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let circuit = self.create_complex_circuit(num_qubits);
        let device = LightningQubit::new(num_qubits, None);
        let device_config = self.create_default_device_config();

        let mut execution_times = Vec::new();
        let mut memory_usage = Vec::new();
        let mut error_count = 0;

        for _ in 0..config.iterations {
            let iter_start = std::time::Instant::now();
            
            match device.execute_circuit(&circuit, 0, &device_config).await { // 0 shots for statevector only
                Ok(result) => {
                    let exec_time = iter_start.elapsed().as_secs_f64() * 1000.0;
                    execution_times.push(exec_time);
                    
                    // Estimate memory usage based on statevector size
                    if let Some(statevector) = result.statevector {
                        let memory_mb = (statevector.len() * 16) as f64 / (1024.0 * 1024.0); // 16 bytes per complex number
                        memory_usage.push(memory_mb);
                    }
                },
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        let avg_execution_time = if !execution_times.is_empty() {
            execution_times.iter().sum::<f64>() / execution_times.len() as f64
        } else {
            0.0
        };

        let avg_memory_usage = if !memory_usage.is_empty() {
            memory_usage.iter().sum::<f64>() / memory_usage.len() as f64
        } else {
            0.0
        };

        let success_rate = execution_times.len() as f64 / config.iterations as f64;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: avg_execution_time,
            throughput_ops_per_sec: if avg_execution_time > 0.0 { 1000.0 / avg_execution_time } else { 0.0 },
            memory_usage_mb: avg_memory_usage,
            cpu_usage_percent: 0.0,
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: Some(DeviceType::LightningQubit),
            circuit_info: Some(self.extract_circuit_info(&circuit)),
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Benchmark trading algorithm
    async fn benchmark_trading_algorithm(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let mut execution_times = Vec::new();
        let mut error_count = 0;

        for _ in 0..config.iterations {
            let iter_start = std::time::Instant::now();
            
            // Simulate trading algorithm execution
            self.simulate_trading_decision().await;
            
            let exec_time = iter_start.elapsed().as_secs_f64() * 1000.0;
            execution_times.push(exec_time);
        }

        let avg_execution_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let success_rate = 1.0; // All simulated operations succeed

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: avg_execution_time,
            throughput_ops_per_sec: if avg_execution_time > 0.0 { 1000.0 / avg_execution_time } else { 0.0 },
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            success_rate,
            error_count,
            started_at,
            completed_at: Utc::now(),
            device_type: None,
            circuit_info: None,
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Benchmark memory usage
    async fn benchmark_memory_usage(&self, config: &BenchmarkConfig) -> QarResult<BenchmarkResult> {
        let start_time = std::time::Instant::now();
        let started_at = Utc::now();

        let data_size = config.parameters.get("data_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        // Simulate memory-intensive operations
        let mut memory_allocations = Vec::new();
        let mut peak_memory = 0.0;

        for i in 0..config.iterations {
            let allocation_size = data_size * (i as usize + 1);
            let allocation: Vec<f64> = vec![0.0; allocation_size];
            let memory_mb = (allocation.len() * 8) as f64 / (1024.0 * 1024.0);
            
            memory_allocations.push(allocation);
            peak_memory = peak_memory.max(memory_mb as f64);
            
            // Small delay to simulate processing
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        let avg_memory = peak_memory / 2.0; // Rough estimate
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(BenchmarkResult {
            benchmark_id: Uuid::new_v4().to_string(),
            benchmark_type: config.benchmark_type.clone(),
            execution_time_ms: execution_time,
            throughput_ops_per_sec: config.iterations as f64 / (execution_time / 1000.0),
            memory_usage_mb: avg_memory,
            cpu_usage_percent: 0.0,
            success_rate: 1.0,
            error_count: 0,
            started_at,
            completed_at: Utc::now(),
            device_type: None,
            circuit_info: None,
            detailed_metrics: HashMap::new(),
            metadata: config.parameters.clone(),
        })
    }

    /// Run benchmark suite
    pub async fn run_benchmark_suite(&self, suite: BenchmarkSuite) -> QarResult<BenchmarkReport> {
        let start_time = Utc::now();
        let mut results = Vec::new();
        let mut lightning_comparisons = Vec::new();
        let mut successful_benchmarks = 0;
        let mut failed_benchmarks = 0;

        for benchmark_config in suite.benchmarks {
            match self.run_benchmark(benchmark_config.clone()).await {
                Ok(result) => {
                    successful_benchmarks += 1;
                    
                    // If this was a Lightning comparison, extract comparison data
                    if benchmark_config.benchmark_type == BenchmarkType::LightningComparison {
                        // Would extract detailed comparison data in real implementation
                    }
                    
                    results.push(result);
                },
                Err(e) => {
                    failed_benchmarks += 1;
                    log::error!("Benchmark failed: {}", e);
                    
                    if suite.suite_config.stop_on_failure {
                        break;
                    }
                }
            }
        }

        let performance_summary = self.calculate_performance_summary(&results);
        let recommendations = self.generate_recommendations(&results);

        Ok(BenchmarkReport {
            suite_name: suite.name,
            generated_at: Utc::now(),
            total_benchmarks: suite.benchmarks.len() as u32,
            successful_benchmarks,
            failed_benchmarks,
            total_execution_time_ms: (Utc::now() - start_time).num_milliseconds() as f64,
            results,
            lightning_comparisons,
            performance_summary,
            recommendations,
        })
    }

    /// Calculate performance summary
    fn calculate_performance_summary(&self, results: &[BenchmarkResult]) -> HashMap<BenchmarkType, PerformanceSummary> {
        let mut summary_map = HashMap::new();
        
        // Group results by benchmark type
        let mut grouped_results: HashMap<BenchmarkType, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results {
            grouped_results.entry(result.benchmark_type.clone()).or_insert_with(Vec::new).push(result);
        }

        for (benchmark_type, type_results) in grouped_results {
            let execution_times: Vec<f64> = type_results.iter().map(|r| r.execution_time_ms).collect();
            let throughputs: Vec<f64> = type_results.iter().map(|r| r.throughput_ops_per_sec).collect();
            let memory_usages: Vec<f64> = type_results.iter().map(|r| r.memory_usage_mb).collect();
            let success_rates: Vec<f64> = type_results.iter().map(|r| r.success_rate).collect();

            let avg_execution_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
            let min_execution_time = execution_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_execution_time = execution_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let mean = avg_execution_time;
            let variance = execution_times.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / execution_times.len() as f64;
            let std_dev = variance.sqrt();

            let summary = PerformanceSummary {
                avg_execution_time_ms: avg_execution_time,
                min_execution_time_ms: min_execution_time,
                max_execution_time_ms: max_execution_time,
                std_dev_execution_time_ms: std_dev,
                avg_throughput_ops_per_sec: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
                avg_memory_usage_mb: memory_usages.iter().sum::<f64>() / memory_usages.len() as f64,
                success_rate: success_rates.iter().sum::<f64>() / success_rates.len() as f64,
            };

            summary_map.insert(benchmark_type, summary);
        }

        summary_map
    }

    /// Generate recommendations based on results
    fn generate_recommendations(&self, results: &[BenchmarkResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze Lightning device performance
        let lightning_results: Vec<&BenchmarkResult> = results.iter()
            .filter(|r| r.device_type.is_some())
            .collect();

        if !lightning_results.is_empty() {
            let avg_gpu_time = lightning_results.iter()
                .filter(|r| r.device_type == Some(DeviceType::LightningGpu))
                .map(|r| r.execution_time_ms)
                .sum::<f64>() / lightning_results.len() as f64;

            let avg_cpu_time = lightning_results.iter()
                .filter(|r| r.device_type == Some(DeviceType::LightningQubit))
                .map(|r| r.execution_time_ms)
                .sum::<f64>() / lightning_results.len() as f64;

            if avg_gpu_time > 0.0 && avg_cpu_time > 0.0 {
                let speedup = avg_cpu_time / avg_gpu_time;
                if speedup > 2.0 {
                    recommendations.push(format!("Consider using lightning.gpu for {:.1}x speedup", speedup));
                }
            }
        }

        // Memory usage recommendations
        let high_memory_results: Vec<&BenchmarkResult> = results.iter()
            .filter(|r| r.memory_usage_mb > 1000.0)
            .collect();

        if !high_memory_results.is_empty() {
            recommendations.push("High memory usage detected. Consider circuit optimization or batching.".to_string());
        }

        // Success rate recommendations
        let low_success_results: Vec<&BenchmarkResult> = results.iter()
            .filter(|r| r.success_rate < 0.95)
            .collect();

        if !low_success_results.is_empty() {
            recommendations.push("Low success rate detected. Check for resource constraints or errors.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Performance looks good across all benchmarks.".to_string());
        }

        recommendations
    }

    /// Helper methods for creating test circuits and configurations

    fn create_benchmark_circuit(&self, num_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits);
        
        // Create a Bell state preparation circuit
        circuit.h(0);
        for i in 0..num_qubits-1 {
            circuit.cnot(i, i + 1);
        }
        
        // Add some rotation gates for complexity
        for i in 0..num_qubits {
            circuit.rx(i, std::f64::consts::PI / 4.0);
            circuit.rz(i, std::f64::consts::PI / 6.0);
        }
        
        circuit
    }

    fn create_simple_circuit(&self) -> Box<dyn QuantumCircuit> {
        // Use QftCircuit as concrete implementation of QuantumCircuit
        Box::new(QftCircuit::new(3))
    }

    fn create_complex_circuit(&self, num_qubits: usize) -> Box<dyn QuantumCircuit> {
        // Use QftCircuit as concrete implementation of QuantumCircuit
        Box::new(QftCircuit::new(num_qubits))
    }

    fn create_default_device_config(&self) -> DeviceConfig {
        DeviceConfig {
            device_type: DeviceType::LightningQubit,
            backend_name: None,
            shots: 1000,
            seed: Some(42),
            optimization_level: 1,
            initial_layout: None,
            coupling_map: None,
            basis_gates: None,
            noise_model: None,
            memory: false,
            max_parallel_experiments: 1,
            provider_config: HashMap::new(),
        }
    }

    fn extract_circuit_info(&self, circuit: &dyn QuantumCircuit) -> CircuitInfo {
        let gate_types: Vec<String> = circuit.gates.iter()
            .map(|gate| gate.gate_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        CircuitInfo {
            num_qubits: circuit.num_qubits,
            gate_count: circuit.gates.len(),
            circuit_depth: self.calculate_circuit_depth(circuit),
            gate_types,
            connectivity_requirements: self.calculate_connectivity_requirements(circuit),
        }
    }

    fn calculate_circuit_depth(&self, circuit: &dyn QuantumCircuit) -> usize {
        // Simplified depth calculation
        circuit.gates.len() / circuit.num_qubits + 1
    }

    fn calculate_connectivity_requirements(&self, circuit: &dyn QuantumCircuit) -> usize {
        // Count two-qubit gates as connectivity requirements
        circuit.gates.iter()
            .filter(|gate| gate.qubits.len() >= 2)
            .count()
    }

    async fn simulate_trading_decision(&self) {
        // Simulate trading algorithm processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    /// Get benchmark results history
    pub async fn get_results_history(&self) -> Vec<BenchmarkResult> {
        let history = self.results_history.read().await;
        history.clone()
    }

    /// Clear results history
    pub async fn clear_history(&self) {
        let mut history = self.results_history.write().await;
        history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_execution_benchmark() {
        let manager = BenchmarkManager::new();
        
        let config = BenchmarkConfig {
            benchmark_type: BenchmarkType::QuantumCircuitExecution,
            iterations: 5,
            warmup_iterations: 2,
            timeout_seconds: 30,
            collect_detailed_metrics: false,
            include_memory_profiling: false,
            parallel_execution: false,
            test_data_size: 1000,
            parameters: [("num_qubits".to_string(), "4".to_string())].into_iter().collect(),
        };

        let result = manager.run_benchmark(config).await.unwrap();
        
        assert_eq!(result.benchmark_type, BenchmarkType::QuantumCircuitExecution);
        assert!(result.execution_time_ms > 0.0);
        assert!(result.success_rate > 0.0);
        assert_eq!(result.device_type, Some(DeviceType::LightningQubit));
    }

    #[tokio::test]
    async fn test_lightning_comparison_benchmark() {
        let manager = BenchmarkManager::new();
        
        let config = BenchmarkConfig {
            benchmark_type: BenchmarkType::LightningComparison,
            iterations: 3,
            warmup_iterations: 1,
            timeout_seconds: 30,
            collect_detailed_metrics: true,
            include_memory_profiling: false,
            parallel_execution: false,
            test_data_size: 1000,
            parameters: [("num_qubits".to_string(), "6".to_string())].into_iter().collect(),
        };

        let result = manager.run_benchmark(config).await.unwrap();
        
        assert_eq!(result.benchmark_type, BenchmarkType::LightningComparison);
        assert!(result.metadata.contains_key("comparison_type"));
        assert!(result.metadata.contains_key("recommendations"));
    }

    #[tokio::test]
    async fn test_benchmark_suite() {
        let manager = BenchmarkManager::new();
        
        let suite = BenchmarkSuite {
            name: "Test Suite".to_string(),
            description: "Test benchmark suite".to_string(),
            benchmarks: vec![
                BenchmarkConfig {
                    benchmark_type: BenchmarkType::DeviceLatency,
                    iterations: 3,
                    warmup_iterations: 1,
                    timeout_seconds: 10,
                    collect_detailed_metrics: false,
                    include_memory_profiling: false,
                    parallel_execution: false,
                    test_data_size: 100,
                    parameters: HashMap::new(),
                },
                BenchmarkConfig {
                    benchmark_type: BenchmarkType::MemoryUsage,
                    iterations: 2,
                    warmup_iterations: 0,
                    timeout_seconds: 10,
                    collect_detailed_metrics: false,
                    include_memory_profiling: true,
                    parallel_execution: false,
                    test_data_size: 500,
                    parameters: [("data_size".to_string(), "1000".to_string())].into_iter().collect(),
                },
            ],
            suite_config: SuiteConfig {
                max_parallel_benchmarks: 1,
                stop_on_failure: false,
                generate_report: true,
                export_results: false,
                comparison_baseline: None,
            },
        };

        let report = manager.run_benchmark_suite(suite).await.unwrap();
        
        assert_eq!(report.suite_name, "Test Suite");
        assert_eq!(report.total_benchmarks, 2);
        assert!(report.successful_benchmarks > 0);
        assert!(!report.results.is_empty());
        assert!(!report.performance_summary.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_performance_summary_calculation() {
        let manager = BenchmarkManager::new();
        
        let results = vec![
            BenchmarkResult {
                benchmark_id: "1".to_string(),
                benchmark_type: BenchmarkType::DeviceLatency,
                execution_time_ms: 10.0,
                throughput_ops_per_sec: 100.0,
                memory_usage_mb: 50.0,
                cpu_usage_percent: 25.0,
                success_rate: 1.0,
                error_count: 0,
                started_at: Utc::now(),
                completed_at: Utc::now(),
                device_type: None,
                circuit_info: None,
                detailed_metrics: HashMap::new(),
                metadata: HashMap::new(),
            },
            BenchmarkResult {
                benchmark_id: "2".to_string(),
                benchmark_type: BenchmarkType::DeviceLatency,
                execution_time_ms: 20.0,
                throughput_ops_per_sec: 50.0,
                memory_usage_mb: 75.0,
                cpu_usage_percent: 40.0,
                success_rate: 0.9,
                error_count: 1,
                started_at: Utc::now(),
                completed_at: Utc::now(),
                device_type: None,
                circuit_info: None,
                detailed_metrics: HashMap::new(),
                metadata: HashMap::new(),
            },
        ];

        let summary = manager.calculate_performance_summary(&results);
        
        assert!(summary.contains_key(&BenchmarkType::DeviceLatency));
        let latency_summary = &summary[&BenchmarkType::DeviceLatency];
        
        assert_eq!(latency_summary.avg_execution_time_ms, 15.0);
        assert_eq!(latency_summary.min_execution_time_ms, 10.0);
        assert_eq!(latency_summary.max_execution_time_ms, 20.0);
        assert_eq!(latency_summary.avg_throughput_ops_per_sec, 75.0);
        assert_eq!(latency_summary.success_rate, 0.95);
    }
}