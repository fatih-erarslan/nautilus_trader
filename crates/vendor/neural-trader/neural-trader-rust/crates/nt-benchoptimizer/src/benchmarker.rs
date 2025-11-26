//! Core benchmarking engine with multi-threaded execution

use crate::{BenchmarkOptions, BenchmarkResult, Statistics};
use napi::Result;
use quanta::Clock;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Duration;
use sysinfo::{System, Pid};

pub struct Benchmarker {
    package_path: PathBuf,
    clock: Clock,
}

impl Benchmarker {
    pub fn new(package_path: String) -> Result<Self> {
        Ok(Self {
            package_path: PathBuf::from(package_path),
            clock: Clock::new(),
        })
    }

    pub async fn run_benchmark(&self, options: BenchmarkOptions) -> Result<BenchmarkResult> {
        let iterations = options.iterations.unwrap_or(100) as u32;
        let warmup = options.warmup_iterations.unwrap_or(10) as u32;
        let include_memory = options.include_memory_profiling.unwrap_or(true);
        let include_bundle = options.include_bundle_analysis.unwrap_or(true);

        // Warmup phase
        for _ in 0..warmup {
            self.execute_benchmark_iteration().await?;
        }

        // Actual benchmark phase
        let timings: Vec<Duration> = if options.parallel.unwrap_or(false) {
            (0..iterations)
                .into_par_iter()
                .map(|_| {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(self.measure_execution_time())
                })
                .collect()
        } else {
            let mut results = Vec::with_capacity(iterations as usize);
            for _ in 0..iterations {
                results.push(self.measure_execution_time().await);
            }
            results
        };

        // Calculate statistics
        let stats = self.calculate_statistics(&timings);

        // Memory profiling
        let memory_usage = if include_memory {
            self.measure_memory_usage().await?
        } else {
            0
        };

        // Bundle size analysis
        let bundle_size = if include_bundle {
            self.analyze_bundle_size().await?
        } else {
            0
        };

        let package_name = self.package_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(BenchmarkResult {
            package_name,
            package_path: self.package_path.to_string_lossy().to_string(),
            execution_time_ms: stats.mean,
            memory_usage_mb: memory_usage as i64,
            bundle_size_kb: bundle_size as i64,
            statistics: stats,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    async fn execute_benchmark_iteration(&self) -> Result<()> {
        // Simulate package execution/import
        // In production, this would actually load and execute the package
        tokio::time::sleep(Duration::from_micros(100)).await;
        Ok(())
    }

    async fn measure_execution_time(&self) -> Duration {
        let start = self.clock.raw();
        let _ = self.execute_benchmark_iteration().await;
        let end = self.clock.raw();
        Duration::from_nanos(end - start)
    }

    async fn measure_memory_usage(&self) -> Result<u64> {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Get current process memory usage
        let pid = Pid::from_u32(std::process::id());

        if let Some(process) = sys.process(pid) {
            // Memory in MB
            Ok(process.memory() / (1024 * 1024))
        } else {
            Ok(0)
        }
    }

    async fn analyze_bundle_size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        // Walk the package directory and sum file sizes
        for entry in walkdir::WalkDir::new(&self.package_path)
            .max_depth(5)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            }
        }

        // Convert to KB
        Ok(total_size / 1024)
    }

    fn calculate_statistics(&self, timings: &[Duration]) -> Statistics {
        let mut values: Vec<f64> = timings
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0) // Convert to milliseconds
            .collect();

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        let variance = values.iter()
            .map(|v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>() / values.len() as f64;

        let std_dev = variance.sqrt();

        let min = values.first().copied().unwrap_or(0.0);
        let max = values.last().copied().unwrap_or(0.0);

        let p95_idx = (values.len() as f64 * 0.95) as usize;
        let p99_idx = (values.len() as f64 * 0.99) as usize;

        let p95 = values.get(p95_idx).copied().unwrap_or(max);
        let p99 = values.get(p99_idx).copied().unwrap_or(max);

        Statistics {
            mean,
            median,
            std_dev,
            min,
            max,
            p95,
            p99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_calculation() {
        let benchmarker = Benchmarker {
            package_path: PathBuf::from("/tmp"),
            clock: Clock::new(),
        };

        let timings = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let stats = benchmarker.calculate_statistics(&timings);

        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.median, 30.0);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
    }
}
