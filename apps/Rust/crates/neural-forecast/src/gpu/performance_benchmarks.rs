//! Performance benchmarks and profiling tools for GPU acceleration
//!
//! This module provides comprehensive benchmarking and profiling capabilities
//! to measure and optimize GPU performance for neural network inference.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

use crate::{Result, NeuralForecastError};
use super::{GPUBackend, StreamMultiplexer, StreamConfig, StreamOperation, TaskPriority};

/// Comprehensive GPU benchmark suite
pub struct GPUBenchmarkSuite {
    #[cfg(feature = "cuda")]
    cuda_device: Option<std::sync::Arc<CudaDevice>>,
    webgpu_backend: Option<std::sync::Arc<GPUBackend>>,
    stream_multiplexer: Option<StreamMultiplexer>,
    config: BenchmarkConfig,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub timeout_seconds: u64,
    pub measure_memory_usage: bool,
    pub measure_power_consumption: bool,
    pub detailed_profiling: bool,
    pub export_results: bool,
    pub result_format: ResultFormat,
}

/// Result export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultFormat {
    JSON,
    CSV,
    HTML,
    Markdown,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub backend: String,
    pub configuration: BenchmarkConfiguration,
    pub metrics: PerformanceMetrics,
    pub detailed_timings: Vec<DetailedTiming>,
    pub memory_usage: Option<MemoryUsage>,
    pub power_consumption: Option<PowerConsumption>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Benchmark configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfiguration {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub data_type: String,
    pub use_tensor_cores: bool,
    pub use_flash_attention: bool,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_latency_us: f64,
    pub median_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub min_latency_us: f64,
    pub max_latency_us: f64,
    pub throughput_ops_per_sec: f64,
    pub throughput_tokens_per_sec: f64,
    pub flops_per_sec: f64,
    pub memory_bandwidth_gb_per_sec: f64,
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub efficiency_score: f64,
}

/// Detailed timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedTiming {
    pub operation: String,
    pub duration_us: f64,
    pub cpu_time_us: f64,
    pub gpu_time_us: f64,
    pub memory_transfer_us: f64,
    pub kernel_launch_overhead_us: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_allocated_mb: f64,
    pub peak_allocated_mb: f64,
    pub fragmentation_ratio: f64,
    pub allocation_efficiency: f64,
    pub cache_hit_ratio: f64,
}

/// Power consumption metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConsumption {
    pub average_power_w: f64,
    pub peak_power_w: f64,
    pub energy_efficiency_ops_per_j: f64,
    pub thermal_throttling_events: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            timeout_seconds: 300,
            measure_memory_usage: true,
            measure_power_consumption: false,
            detailed_profiling: true,
            export_results: true,
            result_format: ResultFormat::JSON,
        }
    }
}

impl GPUBenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "cuda")]
            cuda_device: None,
            webgpu_backend: None,
            stream_multiplexer: None,
            config,
        })
    }

    /// Initialize CUDA backend
    #[cfg(feature = "cuda")]
    pub fn with_cuda(&mut self) -> Result<&mut Self> {
        use cudarc::driver::CudaDevice;
        
        let device = CudaDevice::new(0)
            .map_err(|e| NeuralForecastError::GpuError(format!("Failed to initialize CUDA device: {}", e)))?;
        
        self.cuda_device = Some(std::sync::Arc::new(device));
        Ok(self)
    }

    /// Initialize WebGPU backend
    pub async fn with_webgpu(&mut self) -> Result<&mut Self> {
        use crate::config::GPUConfig;
        
        let gpu_config = GPUConfig::default();
        let backend = GPUBackend::new(gpu_config).await?;
        
        self.webgpu_backend = Some(std::sync::Arc::new(backend));
        Ok(self)
    }

    /// Initialize stream multiplexer
    pub fn with_stream_multiplexer(&mut self) -> Result<&mut Self> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.cuda_device {
            let stream_config = StreamConfig::default();
            let multiplexer = StreamMultiplexer::new(device.clone(), stream_config)?;
            self.stream_multiplexer = Some(multiplexer);
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            let stream_config = StreamConfig::default();
            let multiplexer = StreamMultiplexer::new(stream_config)?;
            self.stream_multiplexer = Some(multiplexer);
        }
        
        Ok(self)
    }

    /// Run comprehensive benchmark suite
    pub async fn run_full_benchmark(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        // Matrix multiplication benchmarks
        results.extend(self.benchmark_matrix_multiplication().await?);
        
        // Attention mechanism benchmarks
        results.extend(self.benchmark_attention_mechanisms().await?);
        
        // Activation function benchmarks
        results.extend(self.benchmark_activation_functions().await?);
        
        // Memory transfer benchmarks
        results.extend(self.benchmark_memory_transfers().await?);
        
        // Stream multiplexing benchmarks
        results.extend(self.benchmark_stream_multiplexing().await?);
        
        // Export results if configured
        if self.config.export_results {
            self.export_results(&results)?;
        }
        
        Ok(results)
    }

    /// Benchmark matrix multiplication operations
    async fn benchmark_matrix_multiplication(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        let matrix_sizes = vec![
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];
        
        for (m, n, k) in matrix_sizes {
            // FP32 benchmark
            let fp32_result = self.benchmark_gemm_fp32(m, n, k, false).await?;
            results.push(fp32_result);
            
            // FP16 with tensor cores benchmark
            let fp16_result = self.benchmark_gemm_fp16(m, n, k, true).await?;
            results.push(fp16_result);
            
            // INT8 benchmark if available
            if self.supports_int8() {
                let int8_result = self.benchmark_gemm_int8(m, n, k).await?;
                results.push(int8_result);
            }
        }
        
        Ok(results)
    }

    /// Benchmark attention mechanisms
    async fn benchmark_attention_mechanisms(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        let attention_configs = vec![
            (8, 12, 512, 64),   // BERT-base
            (16, 12, 512, 64),  // Larger batch
            (8, 16, 1024, 64),  // GPT-2 medium
            (4, 24, 1024, 64),  // GPT-2 large
            (1, 32, 2048, 64),  // GPT-3 style
        ];
        
        for (batch_size, num_heads, seq_length, head_dim) in attention_configs {
            // Standard attention
            let std_result = self.benchmark_standard_attention(batch_size, num_heads, seq_length, head_dim).await?;
            results.push(std_result);
            
            // Flash attention
            let flash_result = self.benchmark_flash_attention(batch_size, num_heads, seq_length, head_dim).await?;
            results.push(flash_result);
        }
        
        Ok(results)
    }

    /// Benchmark activation functions
    async fn benchmark_activation_functions(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        let tensor_sizes = vec![1024, 4096, 16384, 65536, 262144];
        let activations = vec!["relu", "gelu", "silu", "tanh", "sigmoid"];
        
        for size in tensor_sizes {
            for activation in &activations {
                let result = self.benchmark_activation_function(activation, size).await?;
                results.push(result);
            }
        }
        
        Ok(results)
    }

    /// Benchmark memory transfers
    async fn benchmark_memory_transfers(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        let transfer_sizes = vec![
            1024,      // 1KB
            1024 * 1024,      // 1MB
            16 * 1024 * 1024, // 16MB
            64 * 1024 * 1024, // 64MB
            256 * 1024 * 1024, // 256MB
        ];
        
        for size in transfer_sizes {
            // Host to device
            let h2d_result = self.benchmark_host_to_device_transfer(size).await?;
            results.push(h2d_result);
            
            // Device to host
            let d2h_result = self.benchmark_device_to_host_transfer(size).await?;
            results.push(d2h_result);
            
            // Device to device
            let d2d_result = self.benchmark_device_to_device_transfer(size).await?;
            results.push(d2d_result);
        }
        
        Ok(results)
    }

    /// Benchmark stream multiplexing
    async fn benchmark_stream_multiplexing(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        if let Some(multiplexer) = &self.stream_multiplexer {
            // Concurrent operations benchmark
            let concurrent_result = self.benchmark_concurrent_operations(multiplexer).await?;
            results.push(concurrent_result);
            
            // Pipeline efficiency benchmark
            let pipeline_result = self.benchmark_pipeline_efficiency(multiplexer).await?;
            results.push(pipeline_result);
        }
        
        Ok(results)
    }

    /// Benchmark FP32 GEMM
    async fn benchmark_gemm_fp32(&self, m: usize, n: usize, k: usize, use_tensor_cores: bool) -> Result<BenchmarkResult> {
        let mut timings = Vec::new();
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let start = Instant::now();
            self.execute_gemm_fp32(m, n, k, use_tensor_cores).await?;
            timings.push(start.elapsed());
        }
        
        // Clear warmup timings
        timings.clear();
        
        // Benchmark
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_gemm_fp32(m, n, k, use_tensor_cores).await?;
            timings.push(start.elapsed());
        }
        
        let metrics = self.calculate_metrics(&timings, m * n * k * 2); // 2 ops per multiply-add
        
        Ok(BenchmarkResult {
            name: format!("GEMM_FP32_{}x{}x{}", m, n, k),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size: 1,
                sequence_length: m,
                hidden_size: n,
                num_heads: 1,
                num_layers: 1,
                data_type: "FP32".to_string(),
                use_tensor_cores,
                use_flash_attention: false,
            },
            metrics,
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Benchmark FP16 GEMM with tensor cores
    async fn benchmark_gemm_fp16(&self, m: usize, n: usize, k: usize, use_tensor_cores: bool) -> Result<BenchmarkResult> {
        let mut timings = Vec::new();
        
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let start = Instant::now();
            self.execute_gemm_fp16(m, n, k, use_tensor_cores).await?;
            timings.push(start.elapsed());
        }
        
        timings.clear();
        
        // Benchmark
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_gemm_fp16(m, n, k, use_tensor_cores).await?;
            timings.push(start.elapsed());
        }
        
        let metrics = self.calculate_metrics(&timings, m * n * k * 2);
        
        Ok(BenchmarkResult {
            name: format!("GEMM_FP16_{}x{}x{}", m, n, k),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size: 1,
                sequence_length: m,
                hidden_size: n,
                num_heads: 1,
                num_layers: 1,
                data_type: "FP16".to_string(),
                use_tensor_cores,
                use_flash_attention: false,
            },
            metrics,
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Benchmark INT8 GEMM
    async fn benchmark_gemm_int8(&self, m: usize, n: usize, k: usize) -> Result<BenchmarkResult> {
        let mut timings = Vec::new();
        
        // Warmup and benchmark similar to FP32/FP16
        for _ in 0..self.config.warmup_iterations {
            let start = Instant::now();
            self.execute_gemm_int8(m, n, k).await?;
            timings.push(start.elapsed());
        }
        
        timings.clear();
        
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_gemm_int8(m, n, k).await?;
            timings.push(start.elapsed());
        }
        
        let metrics = self.calculate_metrics(&timings, m * n * k * 2);
        
        Ok(BenchmarkResult {
            name: format!("GEMM_INT8_{}x{}x{}", m, n, k),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size: 1,
                sequence_length: m,
                hidden_size: n,
                num_heads: 1,
                num_layers: 1,
                data_type: "INT8".to_string(),
                use_tensor_cores: true,
                use_flash_attention: false,
            },
            metrics,
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Benchmark standard attention
    async fn benchmark_standard_attention(&self, batch_size: usize, num_heads: usize, seq_length: usize, head_dim: usize) -> Result<BenchmarkResult> {
        let mut timings = Vec::new();
        
        for _ in 0..self.config.warmup_iterations {
            let start = Instant::now();
            self.execute_standard_attention(batch_size, num_heads, seq_length, head_dim).await?;
            timings.push(start.elapsed());
        }
        
        timings.clear();
        
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_standard_attention(batch_size, num_heads, seq_length, head_dim).await?;
            timings.push(start.elapsed());
        }
        
        let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
        let metrics = self.calculate_metrics(&timings, ops);
        
        Ok(BenchmarkResult {
            name: format!("Attention_Standard_{}x{}x{}x{}", batch_size, num_heads, seq_length, head_dim),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size,
                sequence_length: seq_length,
                hidden_size: head_dim,
                num_heads,
                num_layers: 1,
                data_type: "FP32".to_string(),
                use_tensor_cores: false,
                use_flash_attention: false,
            },
            metrics,
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Benchmark Flash Attention
    async fn benchmark_flash_attention(&self, batch_size: usize, num_heads: usize, seq_length: usize, head_dim: usize) -> Result<BenchmarkResult> {
        let mut timings = Vec::new();
        
        for _ in 0..self.config.warmup_iterations {
            let start = Instant::now();
            self.execute_flash_attention(batch_size, num_heads, seq_length, head_dim).await?;
            timings.push(start.elapsed());
        }
        
        timings.clear();
        
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_flash_attention(batch_size, num_heads, seq_length, head_dim).await?;
            timings.push(start.elapsed());
        }
        
        let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
        let metrics = self.calculate_metrics(&timings, ops);
        
        Ok(BenchmarkResult {
            name: format!("Attention_Flash_{}x{}x{}x{}", batch_size, num_heads, seq_length, head_dim),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size,
                sequence_length: seq_length,
                hidden_size: head_dim,
                num_heads,
                num_layers: 1,
                data_type: "FP32".to_string(),
                use_tensor_cores: false,
                use_flash_attention: true,
            },
            metrics,
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Execute GEMM operations (placeholder implementations)
    async fn execute_gemm_fp32(&self, m: usize, n: usize, k: usize, _use_tensor_cores: bool) -> Result<()> {
        // Simulate FP32 GEMM execution
        let ops = m * n * k * 2;
        let throughput = 19.5e12; // Peak FP32 throughput
        let duration = Duration::from_nanos((ops as f64 / throughput * 1e9) as u64);
        tokio::time::sleep(duration).await;
        Ok(())
    }

    async fn execute_gemm_fp16(&self, m: usize, n: usize, k: usize, use_tensor_cores: bool) -> Result<()> {
        let ops = m * n * k * 2;
        let throughput = if use_tensor_cores { 312e12 } else { 39e12 };
        let duration = Duration::from_nanos((ops as f64 / throughput * 1e9) as u64);
        tokio::time::sleep(duration).await;
        Ok(())
    }

    async fn execute_gemm_int8(&self, m: usize, n: usize, k: usize) -> Result<()> {
        let ops = m * n * k * 2;
        let throughput = 1248e12; // INT8 tensor core throughput
        let duration = Duration::from_nanos((ops as f64 / throughput * 1e9) as u64);
        tokio::time::sleep(duration).await;
        Ok(())
    }

    // Additional execution methods would be implemented here...
    async fn execute_standard_attention(&self, batch_size: usize, num_heads: usize, seq_length: usize, head_dim: usize) -> Result<()> {
        let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
        let throughput = 50e12;
        let duration = Duration::from_nanos((ops as f64 / throughput * 1e9) as u64);
        tokio::time::sleep(duration).await;
        Ok(())
    }

    async fn execute_flash_attention(&self, batch_size: usize, num_heads: usize, seq_length: usize, head_dim: usize) -> Result<()> {
        let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
        let throughput = 100e12; // Flash attention is more efficient
        let duration = Duration::from_nanos((ops as f64 / throughput * 1e9) as u64);
        tokio::time::sleep(duration).await;
        Ok(())
    }

    // More placeholder methods would be implemented...
    async fn benchmark_activation_function(&self, _activation: &str, _size: usize) -> Result<BenchmarkResult> {
        // Placeholder
        Ok(BenchmarkResult {
            name: "activation_placeholder".to_string(),
            backend: self.get_backend_name(),
            configuration: BenchmarkConfiguration {
                batch_size: 1,
                sequence_length: 1,
                hidden_size: 1,
                num_heads: 1,
                num_layers: 1,
                data_type: "FP32".to_string(),
                use_tensor_cores: false,
                use_flash_attention: false,
            },
            metrics: PerformanceMetrics {
                average_latency_us: 100.0,
                median_latency_us: 100.0,
                p95_latency_us: 120.0,
                p99_latency_us: 150.0,
                min_latency_us: 90.0,
                max_latency_us: 200.0,
                throughput_ops_per_sec: 10000.0,
                throughput_tokens_per_sec: 1000.0,
                flops_per_sec: 1e12,
                memory_bandwidth_gb_per_sec: 500.0,
                compute_utilization: 0.8,
                memory_utilization: 0.6,
                efficiency_score: 0.75,
            },
            detailed_timings: vec![],
            memory_usage: None,
            power_consumption: None,
            timestamp: chrono::Utc::now(),
        })
    }

    // More placeholder methods...
    async fn benchmark_host_to_device_transfer(&self, _size: usize) -> Result<BenchmarkResult> { todo!() }
    async fn benchmark_device_to_host_transfer(&self, _size: usize) -> Result<BenchmarkResult> { todo!() }
    async fn benchmark_device_to_device_transfer(&self, _size: usize) -> Result<BenchmarkResult> { todo!() }
    async fn benchmark_concurrent_operations(&self, _multiplexer: &StreamMultiplexer) -> Result<BenchmarkResult> { todo!() }
    async fn benchmark_pipeline_efficiency(&self, _multiplexer: &StreamMultiplexer) -> Result<BenchmarkResult> { todo!() }

    /// Calculate performance metrics from timings
    fn calculate_metrics(&self, timings: &[Duration], total_ops: usize) -> PerformanceMetrics {
        let timings_us: Vec<f64> = timings.iter().map(|d| d.as_nanos() as f64 / 1000.0).collect();
        
        let mut sorted_timings = timings_us.clone();
        sorted_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let average_latency_us = timings_us.iter().sum::<f64>() / timings_us.len() as f64;
        let median_latency_us = sorted_timings[sorted_timings.len() / 2];
        let p95_latency_us = sorted_timings[(sorted_timings.len() as f64 * 0.95) as usize];
        let p99_latency_us = sorted_timings[(sorted_timings.len() as f64 * 0.99) as usize];
        let min_latency_us = sorted_timings[0];
        let max_latency_us = sorted_timings[sorted_timings.len() - 1];
        
        let throughput_ops_per_sec = total_ops as f64 / (average_latency_us / 1e6);
        let flops_per_sec = total_ops as f64 / (average_latency_us / 1e6);
        
        PerformanceMetrics {
            average_latency_us,
            median_latency_us,
            p95_latency_us,
            p99_latency_us,
            min_latency_us,
            max_latency_us,
            throughput_ops_per_sec,
            throughput_tokens_per_sec: throughput_ops_per_sec / 1000.0, // Approximate
            flops_per_sec,
            memory_bandwidth_gb_per_sec: 500.0, // Placeholder
            compute_utilization: 0.8, // Placeholder
            memory_utilization: 0.6, // Placeholder
            efficiency_score: 0.75, // Placeholder
        }
    }

    /// Get backend name
    fn get_backend_name(&self) -> String {
        #[cfg(feature = "cuda")]
        if self.cuda_device.is_some() {
            return "CUDA".to_string();
        }
        
        if self.webgpu_backend.is_some() {
            return "WebGPU".to_string();
        }
        
        "CPU".to_string()
    }

    /// Check if INT8 is supported
    fn supports_int8(&self) -> bool {
        #[cfg(feature = "cuda")]
        if self.cuda_device.is_some() {
            return true; // Assume modern GPUs support INT8
        }
        
        false
    }

    /// Export benchmark results
    fn export_results(&self, results: &[BenchmarkResult]) -> Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        
        match self.config.result_format {
            ResultFormat::JSON => {
                let filename = format!("benchmark_results_{}.json", timestamp);
                let json = serde_json::to_string_pretty(results)?;
                std::fs::write(&filename, json)?;
                println!("Results exported to {}", filename);
            }
            ResultFormat::CSV => {
                let filename = format!("benchmark_results_{}.csv", timestamp);
                self.export_csv(results, &filename)?;
                println!("Results exported to {}", filename);
            }
            ResultFormat::HTML => {
                let filename = format!("benchmark_results_{}.html", timestamp);
                self.export_html(results, &filename)?;
                println!("Results exported to {}", filename);
            }
            ResultFormat::Markdown => {
                let filename = format!("benchmark_results_{}.md", timestamp);
                self.export_markdown(results, &filename)?;
                println!("Results exported to {}", filename);
            }
        }
        
        Ok(())
    }

    /// Export results as CSV
    fn export_csv(&self, results: &[BenchmarkResult], filename: &str) -> Result<()> {
        let mut csv_content = String::new();
        csv_content.push_str("Name,Backend,Avg Latency (μs),Throughput (ops/sec),FLOPS/sec,Efficiency\n");
        
        for result in results {
            csv_content.push_str(&format!(
                "{},{},{:.2},{:.2},{:.2e},{:.2}\n",
                result.name,
                result.backend,
                result.metrics.average_latency_us,
                result.metrics.throughput_ops_per_sec,
                result.metrics.flops_per_sec,
                result.metrics.efficiency_score
            ));
        }
        
        std::fs::write(filename, csv_content)?;
        Ok(())
    }

    /// Export results as HTML
    fn export_html(&self, results: &[BenchmarkResult], filename: &str) -> Result<()> {
        let mut html_content = String::new();
        html_content.push_str("<!DOCTYPE html><html><head><title>GPU Benchmark Results</title></head><body>");
        html_content.push_str("<h1>GPU Benchmark Results</h1>");
        html_content.push_str("<table border='1'><tr><th>Name</th><th>Backend</th><th>Avg Latency (μs)</th><th>Throughput (ops/sec)</th><th>FLOPS/sec</th><th>Efficiency</th></tr>");
        
        for result in results {
            html_content.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2e}</td><td>{:.2}</td></tr>",
                result.name,
                result.backend,
                result.metrics.average_latency_us,
                result.metrics.throughput_ops_per_sec,
                result.metrics.flops_per_sec,
                result.metrics.efficiency_score
            ));
        }
        
        html_content.push_str("</table></body></html>");
        std::fs::write(filename, html_content)?;
        Ok(())
    }

    /// Export results as Markdown
    fn export_markdown(&self, results: &[BenchmarkResult], filename: &str) -> Result<()> {
        let mut md_content = String::new();
        md_content.push_str("# GPU Benchmark Results\n\n");
        md_content.push_str("| Name | Backend | Avg Latency (μs) | Throughput (ops/sec) | FLOPS/sec | Efficiency |\n");
        md_content.push_str("|------|---------|------------------|----------------------|-----------|------------|\n");
        
        for result in results {
            md_content.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2e} | {:.2} |\n",
                result.name,
                result.backend,
                result.metrics.average_latency_us,
                result.metrics.throughput_ops_per_sec,
                result.metrics.flops_per_sec,
                result.metrics.efficiency_score
            ));
        }
        
        std::fs::write(filename, md_content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert!(config.measure_memory_usage);
    }
    
    #[tokio::test]
    async fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = GPUBenchmarkSuite::new(config);
        assert!(suite.is_ok());
    }
    
    #[test]
    fn test_metrics_calculation() {
        let suite = GPUBenchmarkSuite::new(BenchmarkConfig::default()).unwrap();
        let timings = vec![
            Duration::from_micros(100),
            Duration::from_micros(110),
            Duration::from_micros(90),
            Duration::from_micros(120),
            Duration::from_micros(105),
        ];
        
        let metrics = suite.calculate_metrics(&timings, 1000);
        
        assert!(metrics.average_latency_us > 0.0);
        assert!(metrics.median_latency_us > 0.0);
        assert!(metrics.throughput_ops_per_sec > 0.0);
        assert!(metrics.min_latency_us <= metrics.max_latency_us);
    }
}