//! Report generation and formatting

use crate::{BenchmarkResult, ValidationReport, OptimizationReport};
use napi::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ComprehensiveReport {
    pub metadata: ReportMetadata,
    pub summary: ReportSummary,
    pub benchmarks: Vec<BenchmarkResult>,
    pub validations: Vec<ValidationReport>,
    pub optimizations: Vec<OptimizationReport>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generated_at: String,
    pub version: String,
    pub total_packages: usize,
    pub system_info: SystemInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_gb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_execution_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub total_memory_usage_mb: i64,
    pub total_bundle_size_kb: i64,
    pub validation_success_rate: f64,
    pub total_optimization_savings_kb: i64,
    pub estimated_performance_gain: f64,
}

pub fn generate_comprehensive_report(
    benchmarks: Vec<BenchmarkResult>,
    validations: Vec<ValidationReport>,
    optimizations: Vec<OptimizationReport>,
    format: &str,
) -> Result<String> {
    let report = build_report(benchmarks, validations, optimizations)?;

    match format {
        "json" => serialize_json(&report),
        "markdown" => serialize_markdown(&report),
        "html" => serialize_html(&report),
        _ => serialize_json(&report),
    }
}

fn build_report(
    benchmarks: Vec<BenchmarkResult>,
    validations: Vec<ValidationReport>,
    optimizations: Vec<OptimizationReport>,
) -> Result<ComprehensiveReport> {
    let summary = calculate_summary(&benchmarks, &validations, &optimizations);
    let recommendations = generate_recommendations(&benchmarks, &validations, &optimizations);

    Ok(ComprehensiveReport {
        metadata: ReportMetadata {
            generated_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            total_packages: benchmarks.len(),
            system_info: SystemInfo {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                cpu_count: num_cpus::get(),
                total_memory_gb: 0.0, // Will be filled by caller
            },
        },
        summary,
        benchmarks,
        validations,
        optimizations,
        recommendations,
    })
}

fn calculate_summary(
    benchmarks: &[BenchmarkResult],
    validations: &[ValidationReport],
    optimizations: &[OptimizationReport],
) -> ReportSummary {
    let total_execution_time: f64 = benchmarks.iter()
        .map(|b| b.execution_time_ms)
        .sum();

    let average_execution_time = if !benchmarks.is_empty() {
        total_execution_time / benchmarks.len() as f64
    } else {
        0.0
    };

    let total_memory_usage: i64 = benchmarks.iter()
        .map(|b| b.memory_usage_mb)
        .sum();

    let total_bundle_size: i64 = benchmarks.iter()
        .map(|b| b.bundle_size_kb)
        .sum();

    let valid_count = validations.iter()
        .filter(|v| v.is_valid)
        .count();

    let validation_success_rate = if !validations.is_empty() {
        (valid_count as f64 / validations.len() as f64) * 100.0
    } else {
        0.0
    };

    let total_savings: i64 = optimizations.iter()
        .map(|o| o.potential_savings_kb)
        .sum();

    let total_gain: f64 = optimizations.iter()
        .map(|o| o.estimated_performance_gain)
        .sum();

    ReportSummary {
        total_execution_time_ms: total_execution_time,
        average_execution_time_ms: average_execution_time,
        total_memory_usage_mb: total_memory_usage,
        total_bundle_size_kb: total_bundle_size,
        validation_success_rate,
        total_optimization_savings_kb: total_savings,
        estimated_performance_gain: total_gain,
    }
}

fn generate_recommendations(
    benchmarks: &[BenchmarkResult],
    validations: &[ValidationReport],
    optimizations: &[OptimizationReport],
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Performance recommendations
    let slow_packages: Vec<_> = benchmarks.iter()
        .filter(|b| b.execution_time_ms > 100.0)
        .collect();

    if !slow_packages.is_empty() {
        recommendations.push(format!(
            "Consider optimizing {} slow packages with execution time > 100ms",
            slow_packages.len()
        ));
    }

    // Memory recommendations
    let high_memory: Vec<_> = benchmarks.iter()
        .filter(|b| b.memory_usage_mb > 100)
        .collect();

    if !high_memory.is_empty() {
        recommendations.push(format!(
            "Review {} packages with high memory usage (> 100MB)",
            high_memory.len()
        ));
    }

    // Validation recommendations
    let invalid: Vec<_> = validations.iter()
        .filter(|v| !v.is_valid)
        .collect();

    if !invalid.is_empty() {
        recommendations.push(format!(
            "Fix validation issues in {} packages",
            invalid.len()
        ));
    }

    // Optimization recommendations
    let high_impact: Vec<_> = optimizations.iter()
        .filter(|o| o.potential_savings_kb > 500)
        .collect();

    if !high_impact.is_empty() {
        recommendations.push(format!(
            "Prioritize {} high-impact optimizations with > 500KB savings",
            high_impact.len()
        ));
    }

    recommendations
}

fn serialize_json(report: &ComprehensiveReport) -> Result<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| napi::Error::from_reason(format!("JSON serialization failed: {}", e)))
}

fn serialize_markdown(report: &ComprehensiveReport) -> Result<String> {
    let mut md = String::new();

    md.push_str("# Neural Trader Benchmark Report\n\n");
    md.push_str(&format!("Generated: {}\n\n", report.metadata.generated_at));

    md.push_str("## Summary\n\n");
    md.push_str(&format!("- Total Packages: {}\n", report.metadata.total_packages));
    md.push_str(&format!("- Average Execution Time: {:.2}ms\n", report.summary.average_execution_time_ms));
    md.push_str(&format!("- Total Memory Usage: {}MB\n", report.summary.total_memory_usage_mb));
    md.push_str(&format!("- Validation Success Rate: {:.1}%\n", report.summary.validation_success_rate));
    md.push_str(&format!("- Potential Savings: {}KB\n\n", report.summary.total_optimization_savings_kb));

    md.push_str("## Recommendations\n\n");
    for rec in &report.recommendations {
        md.push_str(&format!("- {}\n", rec));
    }

    Ok(md)
}

fn serialize_html(report: &ComprehensiveReport) -> Result<String> {
    let mut html = String::new();

    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Neural Trader Benchmark Report</title>\n");
    html.push_str("<style>body{font-family:Arial,sans-serif;margin:20px;}</style>\n");
    html.push_str("</head>\n<body>\n");

    html.push_str("<h1>Neural Trader Benchmark Report</h1>\n");
    html.push_str(&format!("<p>Generated: {}</p>\n", report.metadata.generated_at));

    html.push_str("<h2>Summary</h2>\n<ul>\n");
    html.push_str(&format!("<li>Total Packages: {}</li>\n", report.metadata.total_packages));
    html.push_str(&format!("<li>Average Execution Time: {:.2}ms</li>\n", report.summary.average_execution_time_ms));
    html.push_str(&format!("<li>Validation Success Rate: {:.1}%</li>\n", report.summary.validation_success_rate));
    html.push_str("</ul>\n");

    html.push_str("</body>\n</html>");

    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Statistics;

    #[test]
    fn test_summary_calculation() {
        let benchmarks = vec![
            BenchmarkResult {
                package_name: "test1".to_string(),
                package_path: "/test1".to_string(),
                execution_time_ms: 100.0,
                memory_usage_mb: 50,
                bundle_size_kb: 200,
                statistics: Statistics {
                    mean: 100.0,
                    median: 100.0,
                    std_dev: 10.0,
                    min: 90.0,
                    max: 110.0,
                    p95: 108.0,
                    p99: 109.0,
                },
                timestamp: "2024-01-01T00:00:00Z".to_string(),
            },
            BenchmarkResult {
                package_name: "test2".to_string(),
                package_path: "/test2".to_string(),
                execution_time_ms: 200.0,
                memory_usage_mb: 100,
                bundle_size_kb: 300,
                statistics: Statistics {
                    mean: 200.0,
                    median: 200.0,
                    std_dev: 20.0,
                    min: 180.0,
                    max: 220.0,
                    p95: 216.0,
                    p99: 218.0,
                },
                timestamp: "2024-01-01T00:00:00Z".to_string(),
            },
        ];

        let summary = calculate_summary(&benchmarks, &[], &[]);

        assert_eq!(summary.total_execution_time_ms, 300.0);
        assert_eq!(summary.average_execution_time_ms, 150.0);
        assert_eq!(summary.total_memory_usage_mb, 150);
        assert_eq!(summary.total_bundle_size_kb, 500);
    }
}
