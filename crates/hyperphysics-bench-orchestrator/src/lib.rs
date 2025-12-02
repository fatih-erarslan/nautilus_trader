//! # HyperPhysics Benchmark Orchestrator
//!
//! Unified orchestration system for discovering, executing, and reporting on
//! benchmarks, examples, and tests across the HyperPhysics workspace.
//!
//! ## Features
//!
//! - **Auto-Discovery**: Scans workspace for `[[bench]]` and `[[example]]` targets
//! - **Parallel Execution**: Configurable concurrency with resource management
//! - **Performance Tracking**: Regression detection against baseline measurements
//! - **Report Generation**: JSON/HTML output with trends and comparisons
//! - **WASM Support**: Browser-based dashboard for interactive exploration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    HyperPhysics Bench Orchestrator                      │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
//! │  │  Registry   │──▶│  Executor   │──▶│  Collector  │──▶│  Reporter   │ │
//! │  │  (discover) │   │  (run)      │   │  (aggregate)│   │  (output)   │ │
//! │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
//! │         │                 │                 │                 │        │
//! │         ▼                 ▼                 ▼                 ▼        │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    Benchmark Results Store                      │   │
//! │  │  (JSON/SQLite persistence for trend analysis)                   │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_bench_orchestrator::{Orchestrator, OrchestratorConfig};
//!
//! let config = OrchestratorConfig::default()
//!     .with_workspace_root(".")
//!     .with_parallelism(4)
//!     .with_timeout_secs(300);
//!
//! let orchestrator = Orchestrator::new(config)?;
//! let results = orchestrator.run_all().await?;
//! orchestrator.generate_report(&results, "benchmark_report.html")?;
//! ```

pub mod collector;
pub mod executor;
pub mod registry;
pub mod reporter;

#[cfg(feature = "wasm")]
pub mod wasm;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Orchestrator errors
#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Workspace not found at: {0}")]
    WorkspaceNotFound(PathBuf),

    #[error("Benchmark execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Timeout exceeded for: {0}")]
    Timeout(String),

    #[error("No benchmarks found matching filter: {0}")]
    NoBenchmarksFound(String),
}

pub type Result<T> = std::result::Result<T, OrchestratorError>;

/// Target type for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetKind {
    /// Criterion benchmark (`cargo bench`)
    Benchmark,
    /// Example (`cargo run --example`)
    Example,
    /// Test (`cargo test`)
    Test,
}

impl std::fmt::Display for TargetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benchmark => write!(f, "bench"),
            Self::Example => write!(f, "example"),
            Self::Test => write!(f, "test"),
        }
    }
}

/// A discovered benchmark/example/test target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Target name (e.g., "query_latency")
    pub name: String,
    /// Crate name (e.g., "hyperphysics-hnsw")
    pub crate_name: String,
    /// Path to the crate root
    pub crate_path: PathBuf,
    /// Target kind
    pub kind: TargetKind,
    /// Optional description from doc comments
    pub description: Option<String>,
    /// Performance category tags
    pub tags: Vec<String>,
}

impl Target {
    /// Create a new target
    pub fn new(name: impl Into<String>, crate_name: impl Into<String>, crate_path: PathBuf, kind: TargetKind) -> Self {
        Self {
            name: name.into(),
            crate_name: crate_name.into(),
            crate_path,
            kind,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Unique identifier for this target
    pub fn id(&self) -> String {
        format!("{}::{}", self.crate_name, self.name)
    }
}

/// Execution result for a single target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Target that was executed
    pub target: Target,
    /// Whether execution succeeded
    pub success: bool,
    /// Execution duration
    pub duration: Duration,
    /// Exit code (if process-based)
    pub exit_code: Option<i32>,
    /// Captured stdout
    pub stdout: String,
    /// Captured stderr
    pub stderr: String,
    /// Parsed performance metrics (if Criterion output)
    pub metrics: Option<PerformanceMetrics>,
    /// Timestamp of execution
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Parsed performance metrics from Criterion or custom output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Mean execution time in nanoseconds
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,
    /// Median execution time in nanoseconds
    pub median_ns: f64,
    /// Throughput (ops/sec) if available
    pub throughput: Option<f64>,
    /// Sample count
    pub samples: usize,
    /// Custom metrics from benchmark output
    pub custom: std::collections::HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            mean_ns: 0.0,
            std_dev_ns: 0.0,
            median_ns: 0.0,
            throughput: None,
            samples: 0,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Workspace root path
    pub workspace_root: PathBuf,
    /// Maximum parallel executions
    pub parallelism: usize,
    /// Timeout per target in seconds
    pub timeout_secs: u64,
    /// Filter pattern for target names
    pub filter: Option<String>,
    /// Target kinds to include
    pub kinds: Vec<TargetKind>,
    /// Release mode for benchmarks
    pub release: bool,
    /// Verbose output
    pub verbose: bool,
    /// Output directory for reports
    pub output_dir: PathBuf,
    /// Baseline file for regression detection
    pub baseline: Option<PathBuf>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("."),
            parallelism: num_cpus(),
            timeout_secs: 300,
            filter: None,
            kinds: vec![TargetKind::Benchmark, TargetKind::Example],
            release: true,
            verbose: false,
            output_dir: PathBuf::from("target/bench-results"),
            baseline: None,
        }
    }
}

impl OrchestratorConfig {
    /// Set workspace root
    pub fn with_workspace_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace_root = path.into();
        self
    }

    /// Set parallelism
    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.parallelism = n;
        self
    }

    /// Set timeout
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set filter pattern
    pub fn with_filter(mut self, pattern: impl Into<String>) -> Self {
        self.filter = Some(pattern.into());
        self
    }

    /// Set target kinds
    pub fn with_kinds(mut self, kinds: Vec<TargetKind>) -> Self {
        self.kinds = kinds;
        self
    }

    /// Set release mode
    pub fn with_release(mut self, release: bool) -> Self {
        self.release = release;
        self
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set output directory
    pub fn with_output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Set baseline file
    pub fn with_baseline(mut self, path: impl Into<PathBuf>) -> Self {
        self.baseline = Some(path.into());
        self
    }
}

/// Main orchestrator
pub struct Orchestrator {
    config: OrchestratorConfig,
    registry: registry::Registry,
}

impl Orchestrator {
    /// Create new orchestrator with config
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let workspace_root = config.workspace_root.canonicalize().map_err(|_| {
            OrchestratorError::WorkspaceNotFound(config.workspace_root.clone())
        })?;

        let mut registry = registry::Registry::new(workspace_root);
        registry.discover()?;

        Ok(Self { config, registry })
    }

    /// Get discovered targets
    pub fn targets(&self) -> &[Target] {
        self.registry.targets()
    }

    /// Get filtered targets based on config
    pub fn filtered_targets(&self) -> Vec<&Target> {
        let mut targets: Vec<&Target> = self
            .registry
            .targets()
            .iter()
            .filter(|t| self.config.kinds.contains(&t.kind))
            .collect();

        if let Some(ref filter) = self.config.filter {
            let pattern = filter.to_lowercase();
            targets.retain(|t| {
                t.name.to_lowercase().contains(&pattern)
                    || t.crate_name.to_lowercase().contains(&pattern)
            });
        }

        targets
    }

    /// Get orchestrator configuration
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }
}

/// Get number of CPUs (cross-platform)
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_id() {
        let target = Target::new(
            "query_latency",
            "hyperphysics-hnsw",
            PathBuf::from("/tmp"),
            TargetKind::Benchmark,
        );
        assert_eq!(target.id(), "hyperphysics-hnsw::query_latency");
    }

    #[test]
    fn test_config_builder() {
        let config = OrchestratorConfig::default()
            .with_parallelism(8)
            .with_timeout_secs(600)
            .with_filter("hnsw");

        assert_eq!(config.parallelism, 8);
        assert_eq!(config.timeout_secs, 600);
        assert_eq!(config.filter, Some("hnsw".to_string()));
    }
}
