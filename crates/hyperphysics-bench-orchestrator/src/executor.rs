//! Benchmark Executor - Runs targets with timeout and resource management
//!
//! Executes benchmarks/examples via `cargo bench` and `cargo run --example`.

use crate::{ExecutionResult, OrchestratorConfig, OrchestratorError, PerformanceMetrics, Result, Target, TargetKind};
use regex::Regex;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use tokio::process::Command;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::timeout;

/// Executor for running benchmark targets
pub struct Executor {
    config: Arc<OrchestratorConfig>,
}

impl Executor {
    /// Create new executor with config
    pub fn new(config: Arc<OrchestratorConfig>) -> Self {
        Self { config }
    }

    /// Execute a single target
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn execute(&self, target: &Target) -> Result<ExecutionResult> {
        let start = Instant::now();
        let timestamp = chrono::Utc::now();

        // Build command based on target kind
        let mut cmd = self.build_command(target);

        // Set working directory
        cmd.current_dir(&target.crate_path);

        // Capture output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Execute with timeout
        let timeout_duration = Duration::from_secs(self.config.timeout_secs);

        let output = match timeout(timeout_duration, cmd.output()).await {
            Ok(result) => result.map_err(OrchestratorError::Io)?,
            Err(_) => {
                return Ok(ExecutionResult {
                    target: target.clone(),
                    success: false,
                    duration: start.elapsed(),
                    exit_code: None,
                    stdout: String::new(),
                    stderr: format!("Timeout after {} seconds", self.config.timeout_secs),
                    metrics: None,
                    timestamp,
                });
            }
        };

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let success = output.status.success();
        let exit_code = output.status.code();

        // Parse metrics from Criterion output if benchmark
        let metrics = if target.kind == TargetKind::Benchmark && success {
            self.parse_criterion_output(&stdout, &stderr)
        } else {
            None
        };

        Ok(ExecutionResult {
            target: target.clone(),
            success,
            duration,
            exit_code,
            stdout,
            stderr,
            metrics,
            timestamp,
        })
    }

    /// Build command for target
    #[cfg(not(target_arch = "wasm32"))]
    fn build_command(&self, target: &Target) -> Command {
        let mut cmd = Command::new("cargo");

        match target.kind {
            TargetKind::Benchmark => {
                cmd.arg("bench");
                cmd.arg("--bench").arg(&target.name);
                cmd.arg("-p").arg(&target.crate_name);

                // Add -- to pass args to criterion
                cmd.arg("--");
                cmd.arg("--noplot"); // Disable plot generation for CI
            }
            TargetKind::Example => {
                cmd.arg("run");
                cmd.arg("--example").arg(&target.name);
                cmd.arg("-p").arg(&target.crate_name);

                if self.config.release {
                    cmd.arg("--release");
                }
            }
            TargetKind::Test => {
                cmd.arg("test");
                cmd.arg("--test").arg(&target.name);
                cmd.arg("-p").arg(&target.crate_name);

                if self.config.release {
                    cmd.arg("--release");
                }
            }
        }

        cmd
    }

    /// Parse Criterion benchmark output for metrics
    fn parse_criterion_output(&self, stdout: &str, stderr: &str) -> Option<PerformanceMetrics> {
        let combined = format!("{}\n{}", stdout, stderr);

        // Criterion outputs lines like:
        // "benchmark_name    time:   [1.234 µs 1.256 µs 1.278 µs]"
        // We want to extract the median (middle value)

        let time_regex = Regex::new(
            r"time:\s*\[(\d+\.?\d*)\s*(ns|µs|μs|us|ms|s)\s+(\d+\.?\d*)\s*(ns|µs|μs|us|ms|s)\s+(\d+\.?\d*)\s*(ns|µs|μs|us|ms|s)\]"
        ).ok()?;

        let throughput_regex = Regex::new(
            r"thrpt:\s*\[.*?(\d+\.?\d*)\s*(K|M|G)?elem/s.*?\]"
        ).ok()?;

        let mut metrics = PerformanceMetrics::default();
        let mut found_time = false;

        for line in combined.lines() {
            // Parse time measurements
            if let Some(caps) = time_regex.captures(line) {
                let low: f64 = caps.get(1)?.as_str().parse().ok()?;
                let low_unit = caps.get(2)?.as_str();
                let mid: f64 = caps.get(3)?.as_str().parse().ok()?;
                let mid_unit = caps.get(4)?.as_str();
                let high: f64 = caps.get(5)?.as_str().parse().ok()?;
                let high_unit = caps.get(6)?.as_str();

                metrics.mean_ns = self.to_nanoseconds(mid, mid_unit);
                metrics.median_ns = self.to_nanoseconds(mid, mid_unit);
                metrics.std_dev_ns = (self.to_nanoseconds(high, high_unit) - self.to_nanoseconds(low, low_unit)) / 4.0;
                found_time = true;
            }

            // Parse throughput
            if let Some(caps) = throughput_regex.captures(line) {
                let value: f64 = caps.get(1)?.as_str().parse().ok()?;
                let multiplier = match caps.get(2).map(|m| m.as_str()) {
                    Some("K") => 1_000.0,
                    Some("M") => 1_000_000.0,
                    Some("G") => 1_000_000_000.0,
                    _ => 1.0,
                };
                metrics.throughput = Some(value * multiplier);
            }
        }

        if found_time {
            metrics.samples = 100; // Criterion default
            Some(metrics)
        } else {
            None
        }
    }

    /// Convert time value to nanoseconds
    fn to_nanoseconds(&self, value: f64, unit: &str) -> f64 {
        match unit {
            "ns" => value,
            "µs" | "μs" | "us" => value * 1_000.0,
            "ms" => value * 1_000_000.0,
            "s" => value * 1_000_000_000.0,
            _ => value,
        }
    }

    /// Execute multiple targets in parallel
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn execute_parallel(&self, targets: &[&Target]) -> Vec<ExecutionResult> {
        use futures::stream::{self, StreamExt};

        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.parallelism));

        let futures = targets.iter().map(|target| {
            let sem = semaphore.clone();
            let executor = Self::new(self.config.clone());
            let target = (*target).clone();

            async move {
                let _permit = sem.acquire().await.unwrap();
                executor.execute(&target).await.unwrap_or_else(|e| {
                    ExecutionResult {
                        target,
                        success: false,
                        duration: Duration::ZERO,
                        exit_code: None,
                        stdout: String::new(),
                        stderr: e.to_string(),
                        metrics: None,
                        timestamp: chrono::Utc::now(),
                    }
                })
            }
        });

        stream::iter(futures)
            .buffer_unordered(self.config.parallelism)
            .collect()
            .await
    }

    /// Execute targets sequentially
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn execute_sequential(&self, targets: &[&Target]) -> Vec<ExecutionResult> {
        let mut results = Vec::with_capacity(targets.len());

        for target in targets {
            let result = self.execute(target).await.unwrap_or_else(|e| {
                ExecutionResult {
                    target: (*target).clone(),
                    success: false,
                    duration: Duration::ZERO,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: e.to_string(),
                    metrics: None,
                    timestamp: chrono::Utc::now(),
                }
            });
            results.push(result);
        }

        results
    }
}

/// Progress callback for execution updates
pub trait ExecutionProgress: Send + Sync {
    fn on_start(&self, target: &Target);
    fn on_complete(&self, result: &ExecutionResult);
    fn on_progress(&self, completed: usize, total: usize);
}

/// Default no-op progress handler
pub struct NoProgress;

impl ExecutionProgress for NoProgress {
    fn on_start(&self, _target: &Target) {}
    fn on_complete(&self, _result: &ExecutionResult) {}
    fn on_progress(&self, _completed: usize, _total: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_nanoseconds() {
        let config = Arc::new(OrchestratorConfig::default());
        let executor = Executor::new(config);

        assert_eq!(executor.to_nanoseconds(100.0, "ns"), 100.0);
        assert_eq!(executor.to_nanoseconds(1.0, "µs"), 1000.0);
        assert_eq!(executor.to_nanoseconds(1.0, "ms"), 1_000_000.0);
        assert_eq!(executor.to_nanoseconds(1.0, "s"), 1_000_000_000.0);
    }
}
