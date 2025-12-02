//! Result Collector - Aggregates and stores benchmark results
//!
//! Provides storage, comparison, and trend analysis for benchmark results.

use crate::{ExecutionResult, PerformanceMetrics, Result, TargetKind};
#[cfg(test)]
use crate::Target;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Collected benchmark results with aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Total execution duration
    pub total_duration_secs: f64,
    /// Individual results
    pub results: Vec<ExecutionResult>,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Results grouped by crate
    pub by_crate: HashMap<String, CrateResults>,
    /// Results grouped by tag
    pub by_tag: HashMap<String, Vec<String>>,
}

/// Summary statistics for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_targets: usize,
    pub passed: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub benchmarks: usize,
    pub examples: usize,
    pub tests: usize,
    pub fastest_ns: Option<f64>,
    pub slowest_ns: Option<f64>,
    pub average_ns: Option<f64>,
}

/// Results for a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateResults {
    pub crate_name: String,
    pub targets: Vec<String>,
    pub passed: usize,
    pub failed: usize,
    pub metrics: Vec<TargetMetrics>,
}

/// Metrics for a single target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMetrics {
    pub target_id: String,
    pub target_name: String,
    pub kind: TargetKind,
    pub success: bool,
    pub duration_secs: f64,
    pub performance: Option<PerformanceMetrics>,
}

/// Collector for aggregating results
pub struct Collector {
    results: Vec<ExecutionResult>,
    start_time: chrono::DateTime<chrono::Utc>,
}

impl Collector {
    /// Create new collector
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            start_time: chrono::Utc::now(),
        }
    }

    /// Add a result
    pub fn add(&mut self, result: ExecutionResult) {
        self.results.push(result);
    }

    /// Add multiple results
    pub fn add_all(&mut self, results: Vec<ExecutionResult>) {
        self.results.extend(results);
    }

    /// Get all results
    pub fn results(&self) -> &[ExecutionResult] {
        &self.results
    }

    /// Generate aggregated report
    pub fn generate_report(&self) -> BenchmarkReport {
        let timestamp = chrono::Utc::now();
        let total_duration = timestamp
            .signed_duration_since(self.start_time)
            .num_milliseconds() as f64
            / 1000.0;

        // Calculate summary
        let summary = self.calculate_summary();

        // Group by crate
        let by_crate = self.group_by_crate();

        // Group by tag
        let by_tag = self.group_by_tag();

        BenchmarkReport {
            timestamp,
            total_duration_secs: total_duration,
            results: self.results.clone(),
            summary,
            by_crate,
            by_tag,
        }
    }

    /// Calculate summary statistics
    fn calculate_summary(&self) -> ReportSummary {
        let total_targets = self.results.len();
        let passed = self.results.iter().filter(|r| r.success).count();
        let failed = self.results.iter().filter(|r| !r.success && r.exit_code.is_some()).count();
        let timed_out = self.results.iter().filter(|r| !r.success && r.exit_code.is_none()).count();

        let benchmarks = self.results.iter().filter(|r| r.target.kind == TargetKind::Benchmark).count();
        let examples = self.results.iter().filter(|r| r.target.kind == TargetKind::Example).count();
        let tests = self.results.iter().filter(|r| r.target.kind == TargetKind::Test).count();

        // Extract performance metrics
        let metrics: Vec<f64> = self.results
            .iter()
            .filter_map(|r| r.metrics.as_ref())
            .map(|m| m.mean_ns)
            .collect();

        let (fastest_ns, slowest_ns, average_ns) = if metrics.is_empty() {
            (None, None, None)
        } else {
            let fastest = metrics.iter().copied().fold(f64::INFINITY, f64::min);
            let slowest = metrics.iter().copied().fold(0.0f64, f64::max);
            let average = metrics.iter().sum::<f64>() / metrics.len() as f64;
            (Some(fastest), Some(slowest), Some(average))
        };

        ReportSummary {
            total_targets,
            passed,
            failed,
            timed_out,
            benchmarks,
            examples,
            tests,
            fastest_ns,
            slowest_ns,
            average_ns,
        }
    }

    /// Group results by crate
    fn group_by_crate(&self) -> HashMap<String, CrateResults> {
        let mut by_crate: HashMap<String, CrateResults> = HashMap::new();

        for result in &self.results {
            let crate_name = &result.target.crate_name;

            let entry = by_crate.entry(crate_name.clone()).or_insert_with(|| CrateResults {
                crate_name: crate_name.clone(),
                targets: Vec::new(),
                passed: 0,
                failed: 0,
                metrics: Vec::new(),
            });

            entry.targets.push(result.target.name.clone());

            if result.success {
                entry.passed += 1;
            } else {
                entry.failed += 1;
            }

            entry.metrics.push(TargetMetrics {
                target_id: result.target.id(),
                target_name: result.target.name.clone(),
                kind: result.target.kind,
                success: result.success,
                duration_secs: result.duration.as_secs_f64(),
                performance: result.metrics.clone(),
            });
        }

        by_crate
    }

    /// Group results by tag
    fn group_by_tag(&self) -> HashMap<String, Vec<String>> {
        let mut by_tag: HashMap<String, Vec<String>> = HashMap::new();

        for result in &self.results {
            for tag in &result.target.tags {
                by_tag
                    .entry(tag.clone())
                    .or_default()
                    .push(result.target.id());
            }
        }

        by_tag
    }

    /// Compare with baseline report
    pub fn compare_with_baseline(&self, baseline: &BenchmarkReport) -> ComparisonReport {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut unchanged = Vec::new();
        let mut new_targets = Vec::new();

        // Build baseline lookup
        let baseline_metrics: HashMap<String, &PerformanceMetrics> = baseline
            .results
            .iter()
            .filter_map(|r| r.metrics.as_ref().map(|m| (r.target.id(), m)))
            .collect();

        for result in &self.results {
            let target_id = result.target.id();

            if let Some(current_metrics) = &result.metrics {
                if let Some(baseline_metrics) = baseline_metrics.get(&target_id) {
                    let change_pct = (current_metrics.mean_ns - baseline_metrics.mean_ns)
                        / baseline_metrics.mean_ns * 100.0;

                    let comparison = TargetComparison {
                        target_id: target_id.clone(),
                        baseline_ns: baseline_metrics.mean_ns,
                        current_ns: current_metrics.mean_ns,
                        change_pct,
                    };

                    // 5% threshold for significance
                    if change_pct > 5.0 {
                        regressions.push(comparison);
                    } else if change_pct < -5.0 {
                        improvements.push(comparison);
                    } else {
                        unchanged.push(comparison);
                    }
                } else {
                    new_targets.push(target_id);
                }
            }
        }

        // Sort by change magnitude
        regressions.sort_by(|a, b| b.change_pct.partial_cmp(&a.change_pct).unwrap());
        improvements.sort_by(|a, b| a.change_pct.partial_cmp(&b.change_pct).unwrap());

        ComparisonReport {
            regressions,
            improvements,
            unchanged,
            new_targets,
        }
    }

    /// Save report to JSON file
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let report = self.generate_report();
        let json = serde_json::to_string_pretty(&report)?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, json)?;
        Ok(())
    }

    /// Load baseline from JSON file
    pub fn load_baseline(path: &Path) -> Result<BenchmarkReport> {
        let json = fs::read_to_string(path)?;
        let report: BenchmarkReport = serde_json::from_str(&json)?;
        Ok(report)
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison between current and baseline results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Targets that got slower (>5% regression)
    pub regressions: Vec<TargetComparison>,
    /// Targets that got faster (>5% improvement)
    pub improvements: Vec<TargetComparison>,
    /// Targets with minimal change
    pub unchanged: Vec<TargetComparison>,
    /// New targets not in baseline
    pub new_targets: Vec<String>,
}

/// Single target comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetComparison {
    pub target_id: String,
    pub baseline_ns: f64,
    pub current_ns: f64,
    pub change_pct: f64,
}

impl ComparisonReport {
    /// Check if there are any significant regressions
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// Get worst regression
    pub fn worst_regression(&self) -> Option<&TargetComparison> {
        self.regressions.first()
    }

    /// Get best improvement
    pub fn best_improvement(&self) -> Option<&TargetComparison> {
        self.improvements.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use std::path::PathBuf;

    fn make_result(name: &str, crate_name: &str, success: bool, mean_ns: f64) -> ExecutionResult {
        ExecutionResult {
            target: Target::new(name, crate_name, PathBuf::from("/tmp"), TargetKind::Benchmark),
            success,
            duration: Duration::from_secs(1),
            exit_code: Some(if success { 0 } else { 1 }),
            stdout: String::new(),
            stderr: String::new(),
            metrics: Some(PerformanceMetrics {
                mean_ns,
                std_dev_ns: mean_ns * 0.1,
                median_ns: mean_ns,
                throughput: None,
                samples: 100,
                custom: HashMap::new(),
            }),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_collector_summary() {
        let mut collector = Collector::new();
        collector.add(make_result("bench1", "crate1", true, 1000.0));
        collector.add(make_result("bench2", "crate1", false, 2000.0));
        collector.add(make_result("bench3", "crate2", true, 500.0));

        let report = collector.generate_report();

        assert_eq!(report.summary.total_targets, 3);
        assert_eq!(report.summary.passed, 2);
        assert_eq!(report.summary.failed, 1);
    }

    #[test]
    fn test_comparison() {
        let mut baseline_collector = Collector::new();
        baseline_collector.add(make_result("bench1", "crate1", true, 1000.0));
        let baseline = baseline_collector.generate_report();

        let mut current_collector = Collector::new();
        current_collector.add(make_result("bench1", "crate1", true, 1200.0)); // 20% slower

        let comparison = current_collector.compare_with_baseline(&baseline);

        assert!(comparison.has_regressions());
        assert_eq!(comparison.regressions.len(), 1);
        assert!(comparison.regressions[0].change_pct > 15.0);
    }
}
