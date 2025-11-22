//! Regression Detection Framework
//! 
//! This module detects performance and accuracy regressions by comparing
//! current results against historical baselines.

use crate::Result;
use crate::validation::{ValidationConfig, RegressionAnalysisResults, RegressionResult, PerformanceRegression, AccuracyRegression, RegressionSeverity};
use crate::utils::MathUtils;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

pub struct RegressionDetector {
    config: ValidationConfig,
    baseline_path: String,
}

/// Historical baseline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineData {
    pub timestamp: DateTime<Utc>,
    pub performance_metrics: HashMap<String, f64>,
    pub accuracy_metrics: HashMap<String, f64>,
    pub system_info: SystemInfo,
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub architecture: String,
    pub cpu_model: String,
    pub memory_gb: f64,
    pub rust_version: String,
}

/// Regression detection configuration
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    pub performance_threshold: f64,  // % degradation to consider regression
    pub accuracy_threshold: f64,     // Absolute accuracy loss to consider regression
    pub statistical_significance: f64, // p-value threshold for significance
    pub min_samples: usize,          // Minimum samples for statistical testing
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            performance_threshold: 10.0, // 10% slowdown
            accuracy_threshold: 0.01,    // 1% accuracy loss
            statistical_significance: 0.05, // 5% significance level
            min_samples: 10,
        }
    }
}

impl RegressionDetector {
    pub fn new(config: &ValidationConfig) -> Result<Self> {
        let baseline_path = "data/validation_baselines.json".to_string();
        
        Ok(Self {
            config: config.clone(),
            baseline_path,
        })
    }

    /// Detect regressions by comparing current metrics against baselines
    pub async fn detect_regressions(&self) -> Result<RegressionAnalysisResults> {
        let baselines = self.load_baselines()?;
        let current_metrics = self.collect_current_metrics().await?;
        
        let mut baseline_comparisons = HashMap::new();
        let mut performance_regressions = Vec::new();
        let mut accuracy_regressions = Vec::new();
        
        // Compare performance metrics
        for (metric_name, &current_value) in &current_metrics.performance_metrics {
            if let Some(baseline) = baselines.performance_metrics.get(metric_name) {
                let comparison = self.compare_performance_metric(
                    metric_name,
                    current_value,
                    *baseline,
                )?;
                
                baseline_comparisons.insert(metric_name.clone(), comparison.clone());
                
                if comparison.is_regression {
                    let severity = self.assess_performance_severity(&comparison);
                    performance_regressions.push(PerformanceRegression {
                        operation: metric_name.clone(),
                        slowdown_factor: comparison.current_value / comparison.baseline_value,
                        impact_severity: severity,
                    });
                }
            }
        }
        
        // Compare accuracy metrics
        for (metric_name, &current_value) in &current_metrics.accuracy_metrics {
            if let Some(baseline) = baselines.accuracy_metrics.get(metric_name) {
                let comparison = self.compare_accuracy_metric(
                    metric_name,
                    current_value,
                    *baseline,
                )?;
                
                baseline_comparisons.insert(
                    format!("accuracy_{}", metric_name),
                    comparison.clone()
                );
                
                if comparison.is_regression {
                    let severity = self.assess_accuracy_severity(&comparison);
                    accuracy_regressions.push(AccuracyRegression {
                        algorithm: metric_name.clone(),
                        accuracy_loss: comparison.baseline_value - comparison.current_value,
                        impact_severity: severity,
                    });
                }
            }
        }
        
        Ok(RegressionAnalysisResults {
            baseline_comparisons,
            performance_regressions,
            accuracy_regressions,
        })
    }

    /// Store current metrics as new baseline
    pub async fn store_baseline(&self) -> Result<()> {
        let current_metrics = self.collect_current_metrics().await?;
        let system_info = self.collect_system_info();
        let git_commit = self.get_git_commit_hash();
        
        let baseline = BaselineData {
            timestamp: Utc::now(),
            performance_metrics: current_metrics.performance_metrics,
            accuracy_metrics: current_metrics.accuracy_metrics,
            system_info,
            git_commit,
        };
        
        self.save_baseline(&baseline)?;
        Ok(())
    }

    /// Load historical baseline data
    fn load_baselines(&self) -> Result<BaselineData> {
        if !Path::new(&self.baseline_path).exists() {
            // Return empty baseline if no file exists
            return Ok(BaselineData {
                timestamp: Utc::now(),
                performance_metrics: HashMap::new(),
                accuracy_metrics: HashMap::new(),
                system_info: self.collect_system_info(),
                git_commit: None,
            });
        }
        
        let contents = fs::read_to_string(&self.baseline_path)
            .map_err(|e| crate::error::Error::IoError(format!("Failed to read baseline file: {}", e)))?;
        
        let baseline: BaselineData = serde_json::from_str(&contents)
            .map_err(|e| crate::error::Error::ParseError(format!("Failed to parse baseline data: {}", e)))?;
        
        Ok(baseline)
    }

    /// Save baseline data to file
    fn save_baseline(&self, baseline: &BaselineData) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(&self.baseline_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| crate::error::Error::IoError(format!("Failed to create directory: {}", e)))?;
        }
        
        let json = serde_json::to_string_pretty(baseline)
            .map_err(|e| crate::error::Error::ParseError(format!("Failed to serialize baseline: {}", e)))?;
        
        fs::write(&self.baseline_path, json)
            .map_err(|e| crate::error::Error::IoError(format!("Failed to write baseline file: {}", e)))?;
        
        Ok(())
    }

    /// Collect current performance and accuracy metrics
    async fn collect_current_metrics(&self) -> Result<BaselineData> {
        let mut performance_metrics = HashMap::new();
        let mut accuracy_metrics = HashMap::new();
        
        // Run performance benchmarks
        performance_metrics.extend(self.benchmark_core_operations().await?);
        
        // Run accuracy tests
        accuracy_metrics.extend(self.test_algorithm_accuracy().await?);
        
        Ok(BaselineData {
            timestamp: Utc::now(),
            performance_metrics,
            accuracy_metrics,
            system_info: self.collect_system_info(),
            git_commit: self.get_git_commit_hash(),
        })
    }

    /// Benchmark core mathematical operations
    async fn benchmark_core_operations(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        let test_data = self.generate_test_data(1000);
        let iterations = 1000;
        
        // Benchmark EMA
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = MathUtils::ema(&test_data, 0.3);
        }
        let ema_time = start.elapsed().as_nanos() as f64 / iterations as f64;
        metrics.insert("ema_time_ns".to_string(), ema_time);
        
        // Benchmark standard deviation
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = MathUtils::std_dev(&test_data);
        }
        let std_time = start.elapsed().as_nanos() as f64 / iterations as f64;
        metrics.insert("std_dev_time_ns".to_string(), std_time);
        
        // Benchmark correlation
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = MathUtils::correlation(&test_data, &test_data);
        }
        let corr_time = start.elapsed().as_nanos() as f64 / iterations as f64;
        metrics.insert("correlation_time_ns".to_string(), corr_time);
        
        // Benchmark percentile
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = MathUtils::percentile(&test_data, 0.5);
        }
        let percentile_time = start.elapsed().as_nanos() as f64 / iterations as f64;
        metrics.insert("percentile_time_ns".to_string(), percentile_time);
        
        Ok(metrics)
    }

    /// Test algorithm accuracy against known results
    async fn test_algorithm_accuracy(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Test EMA accuracy
        let test_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema_result = MathUtils::ema(&test_values, 0.3);
        let expected_ema = vec![1.0, 1.3, 1.71, 2.197, 2.7379];
        let ema_error = self.calculate_rmse(&ema_result, &expected_ema);
        metrics.insert("ema_rmse".to_string(), ema_error);
        
        // Test standard deviation accuracy
        let std_result = MathUtils::std_dev(&test_values);
        let expected_std = 1.5811388300841898; // Known result
        let std_error = (std_result - expected_std).abs();
        metrics.insert("std_dev_error".to_string(), std_error);
        
        // Test correlation accuracy
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr_result = MathUtils::correlation(&x, &y);
        let expected_corr = 1.0; // Perfect correlation
        let corr_error = (corr_result - expected_corr).abs();
        metrics.insert("correlation_error".to_string(), corr_error);
        
        // Test percentile accuracy
        let percentile_result = MathUtils::percentile(&test_values, 0.5);
        let expected_percentile = 3.0; // Median of [1,2,3,4,5]
        let percentile_error = (percentile_result - expected_percentile).abs();
        metrics.insert("percentile_error".to_string(), percentile_error);
        
        Ok(metrics)
    }

    /// Compare performance metric against baseline
    fn compare_performance_metric(
        &self,
        metric_name: &str,
        current_value: f64,
        baseline_value: f64,
    ) -> Result<RegressionResult> {
        let percentage_change = ((current_value - baseline_value) / baseline_value) * 100.0;
        let is_regression = percentage_change > self.get_regression_config().performance_threshold;
        
        // Calculate statistical significance (simplified)
        let significance_level = if percentage_change.abs() > 5.0 { 0.01 } else { 0.05 };
        
        Ok(RegressionResult {
            current_value,
            baseline_value,
            percentage_change,
            is_regression,
            significance_level,
        })
    }

    /// Compare accuracy metric against baseline
    fn compare_accuracy_metric(
        &self,
        metric_name: &str,
        current_value: f64,
        baseline_value: f64,
    ) -> Result<RegressionResult> {
        let absolute_change = baseline_value - current_value; // Positive if accuracy decreased
        let percentage_change = if baseline_value != 0.0 {
            (absolute_change / baseline_value) * 100.0
        } else {
            0.0
        };
        
        let is_regression = absolute_change > self.get_regression_config().accuracy_threshold;
        let significance_level = if absolute_change.abs() > 0.05 { 0.01 } else { 0.05 };
        
        Ok(RegressionResult {
            current_value,
            baseline_value,
            percentage_change,
            is_regression,
            significance_level,
        })
    }

    /// Assess severity of performance regression
    fn assess_performance_severity(&self, comparison: &RegressionResult) -> RegressionSeverity {
        let slowdown = comparison.current_value / comparison.baseline_value;
        
        if slowdown > 5.0 {
            RegressionSeverity::Critical
        } else if slowdown > 2.0 {
            RegressionSeverity::High
        } else if slowdown > 1.5 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }

    /// Assess severity of accuracy regression
    fn assess_accuracy_severity(&self, comparison: &RegressionResult) -> RegressionSeverity {
        let accuracy_loss = comparison.baseline_value - comparison.current_value;
        
        if accuracy_loss > 0.1 {
            RegressionSeverity::Critical
        } else if accuracy_loss > 0.05 {
            RegressionSeverity::High
        } else if accuracy_loss > 0.02 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }

    /// Collect system information for baseline context
    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_model: self.get_cpu_model(),
            memory_gb: self.get_memory_gb(),
            rust_version: self.get_rust_version(),
        }
    }

    fn get_cpu_model(&self) -> String {
        // Simplified CPU detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(model) = line.split(':').nth(1) {
                            return model.trim().to_string();
                        }
                    }
                }
            }
        }
        
        "Unknown CPU".to_string()
    }

    fn get_memory_gb(&self) -> f64 {
        // Simplified memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(mem_str) = line.split_whitespace().nth(1) {
                            if let Ok(mem_kb) = mem_str.parse::<f64>() {
                                return mem_kb / 1024.0 / 1024.0; // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }
        
        8.0 // Default assumption
    }

    fn get_rust_version(&self) -> String {
        // This would need to be detected at runtime or compile time
        "1.70.0".to_string() // Placeholder
    }

    fn get_git_commit_hash(&self) -> Option<String> {
        // Try to get current git commit hash
        if let Ok(output) = std::process::Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
        {
            if output.status.success() {
                if let Ok(hash) = String::from_utf8(output.stdout) {
                    return Some(hash.trim().to_string());
                }
            }
        }
        None
    }

    fn get_regression_config(&self) -> RegressionConfig {
        RegressionConfig::default()
    }

    fn generate_test_data(&self, size: usize) -> Vec<f64> {
        (0..size).map(|i| (i as f64).sin()).collect()
    }

    fn calculate_rmse(&self, actual: &[f64], expected: &[f64]) -> f64 {
        if actual.len() != expected.len() {
            return f64::INFINITY;
        }
        
        let mse = actual.iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).powi(2))
            .sum::<f64>() / actual.len() as f64;
        
        mse.sqrt()
    }

    /// Analyze trend in historical baselines
    pub async fn analyze_historical_trends(&self) -> Result<HashMap<String, TrendAnalysis>> {
        let mut trends = HashMap::new();
        
        // This would require storing multiple historical baselines
        // For now, return empty analysis
        Ok(trends)
    }

    /// Detect performance anomalies using statistical methods
    pub async fn detect_anomalies(&self) -> Result<Vec<AnomalyReport>> {
        let mut anomalies = Vec::new();
        let current_metrics = self.collect_current_metrics().await?;
        
        // Simple anomaly detection based on Z-score
        for (metric_name, &value) in &current_metrics.performance_metrics {
            // This would use historical data to calculate Z-score
            // For now, detect obvious anomalies
            if value > 1e9 || value < 0.0 {
                anomalies.push(AnomalyReport {
                    metric: metric_name.clone(),
                    value,
                    anomaly_type: AnomalyType::Statistical,
                    severity: if value > 1e12 { RegressionSeverity::Critical } else { RegressionSeverity::Medium },
                    description: format!("Unusual value detected: {}", value),
                });
            }
        }
        
        Ok(anomalies)
    }
}

/// Trend analysis result
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub prediction: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    pub metric: String,
    pub value: f64,
    pub anomaly_type: AnomalyType,
    pub severity: RegressionSeverity,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum AnomalyType {
    Statistical,
    Threshold,
    Pattern,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_baseline_serialization() {
        let baseline = BaselineData {
            timestamp: Utc::now(),
            performance_metrics: {
                let mut map = HashMap::new();
                map.insert("test_metric".to_string(), 123.45);
                map
            },
            accuracy_metrics: HashMap::new(),
            system_info: SystemInfo {
                os: "linux".to_string(),
                architecture: "x86_64".to_string(),
                cpu_model: "Test CPU".to_string(),
                memory_gb: 8.0,
                rust_version: "1.70.0".to_string(),
            },
            git_commit: Some("abc123".to_string()),
        };
        
        let json = serde_json::to_string(&baseline).unwrap();
        let deserialized: BaselineData = serde_json::from_str(&json).unwrap();
        
        assert_eq!(baseline.performance_metrics, deserialized.performance_metrics);
        assert_eq!(baseline.git_commit, deserialized.git_commit);
    }

    #[tokio::test]
    async fn test_regression_detection() {
        let dir = tempdir().unwrap();
        let baseline_path = dir.path().join("test_baseline.json");
        
        let config = ValidationConfig::default();
        let mut detector = RegressionDetector::new(&config).unwrap();
        detector.baseline_path = baseline_path.to_string_lossy().to_string();
        
        // Store initial baseline
        detector.store_baseline().await.unwrap();
        
        // Detect regressions (should be none since we just stored the baseline)
        let results = detector.detect_regressions().await.unwrap();
        
        // Should have some metrics but no regressions
        assert!(results.performance_regressions.is_empty());
        assert!(results.accuracy_regressions.is_empty());
    }

    #[test]
    fn test_performance_comparison() {
        let config = ValidationConfig::default();
        let detector = RegressionDetector::new(&config).unwrap();
        
        // Test no regression
        let result = detector.compare_performance_metric("test", 100.0, 95.0).unwrap();
        assert!(!result.is_regression);
        
        // Test regression (20% slowdown)
        let result = detector.compare_performance_metric("test", 120.0, 100.0).unwrap();
        assert!(result.is_regression);
        assert_eq!(result.percentage_change, 20.0);
    }

    #[test]
    fn test_accuracy_comparison() {
        let config = ValidationConfig::default();
        let detector = RegressionDetector::new(&config).unwrap();
        
        // Test no regression
        let result = detector.compare_accuracy_metric("test", 0.95, 0.94).unwrap();
        assert!(!result.is_regression);
        
        // Test regression (significant accuracy loss)
        let result = detector.compare_accuracy_metric("test", 0.90, 0.95).unwrap();
        assert!(result.is_regression);
    }

    #[test]
    fn test_severity_assessment() {
        let config = ValidationConfig::default();
        let detector = RegressionDetector::new(&config).unwrap();
        
        // Critical performance regression
        let comparison = RegressionResult {
            current_value: 1000.0,
            baseline_value: 100.0,
            percentage_change: 900.0,
            is_regression: true,
            significance_level: 0.01,
        };
        
        let severity = detector.assess_performance_severity(&comparison);
        assert!(matches!(severity, RegressionSeverity::Critical));
        
        // Low accuracy regression
        let comparison = RegressionResult {
            current_value: 0.99,
            baseline_value: 0.995,
            percentage_change: 0.5,
            is_regression: true,
            significance_level: 0.05,
        };
        
        let severity = detector.assess_accuracy_severity(&comparison);
        assert!(matches!(severity, RegressionSeverity::Low));
    }
}