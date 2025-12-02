//! Report Generator - Produces JSON/HTML benchmark reports
//!
//! Generates human-readable and machine-parseable benchmark reports.

use crate::collector::{BenchmarkReport, ComparisonReport};
use crate::{Result, TargetKind};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Reporter for generating output
pub struct Reporter;

impl Reporter {
    /// Generate JSON report
    pub fn to_json(report: &BenchmarkReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// Generate HTML report
    pub fn to_html(report: &BenchmarkReport, comparison: Option<&ComparisonReport>) -> String {
        let mut html = String::new();

        // HTML header
        html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HyperPhysics Benchmark Report</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
            --border: #30363d;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }

        .container { max-width: 1400px; margin: 0 auto; }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.5rem; margin: 1.5rem 0 1rem; color: var(--accent-blue); }
        h3 { font-size: 1.2rem; margin: 1rem 0 0.5rem; }

        .timestamp { color: var(--text-secondary); font-size: 0.9rem; }

        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
        }

        .success { color: var(--accent-green); }
        .failure { color: var(--accent-red); }
        .warning { color: var(--accent-yellow); }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }

        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.8rem;
        }

        tr:hover { background: var(--bg-tertiary); }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .badge-bench { background: #1f6feb33; color: var(--accent-blue); }
        .badge-example { background: #238636; color: #3fb950; }
        .badge-test { background: #6e768166; color: var(--text-secondary); }

        .tag {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            margin: 0.1rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }

        .metric { font-family: 'SF Mono', Monaco, monospace; font-size: 0.9rem; }

        .comparison-section { margin-top: 2rem; }

        .regression { background: #f8514922; }
        .improvement { background: #3fb95022; }

        footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
<div class="container">
"#);

        // Header
        html.push_str(&format!(r#"
    <header>
        <h1>🚀 HyperPhysics Benchmark Report</h1>
        <p class="timestamp">Generated: {}</p>
        <p class="timestamp">Total Duration: {:.2}s</p>
    </header>
"#,
            report.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            report.total_duration_secs
        ));

        // Summary cards
        html.push_str(r#"
    <section>
        <h2>📊 Summary</h2>
        <div class="summary">
"#);

        html.push_str(&format!(r#"
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Total Targets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failure">{}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Benchmarks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Examples</div>
            </div>
"#,
            report.summary.total_targets,
            report.summary.passed,
            report.summary.failed,
            report.summary.benchmarks,
            report.summary.examples
        ));

        // Performance summary
        if let Some(avg) = report.summary.average_ns {
            html.push_str(&format!(r#"
            <div class="stat-card">
                <div class="stat-value metric">{}</div>
                <div class="stat-label">Avg Time</div>
            </div>
"#, Self::format_time(avg)));
        }

        html.push_str("        </div>\n    </section>\n");

        // Comparison section (if available)
        if let Some(comp) = comparison {
            html.push_str(r#"
    <section class="comparison-section">
        <h2>📈 Comparison with Baseline</h2>
"#);

            if !comp.regressions.is_empty() {
                html.push_str(&format!(r#"
        <h3 class="failure">⚠️ Regressions ({})</h3>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"#, comp.regressions.len()));

                for r in &comp.regressions {
                    html.push_str(&format!(r#"
                <tr class="regression">
                    <td>{}</td>
                    <td class="metric">{}</td>
                    <td class="metric">{}</td>
                    <td class="failure">+{:.1}%</td>
                </tr>
"#,
                        r.target_id,
                        Self::format_time(r.baseline_ns),
                        Self::format_time(r.current_ns),
                        r.change_pct
                    ));
                }

                html.push_str("            </tbody>\n        </table>\n");
            }

            if !comp.improvements.is_empty() {
                html.push_str(&format!(r#"
        <h3 class="success">✅ Improvements ({})</h3>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"#, comp.improvements.len()));

                for i in &comp.improvements {
                    html.push_str(&format!(r#"
                <tr class="improvement">
                    <td>{}</td>
                    <td class="metric">{}</td>
                    <td class="metric">{}</td>
                    <td class="success">{:.1}%</td>
                </tr>
"#,
                        i.target_id,
                        Self::format_time(i.baseline_ns),
                        Self::format_time(i.current_ns),
                        i.change_pct
                    ));
                }

                html.push_str("            </tbody>\n        </table>\n");
            }

            html.push_str("    </section>\n");
        }

        // Results by crate
        html.push_str(r#"
    <section>
        <h2>📦 Results by Crate</h2>
"#);

        for (crate_name, crate_results) in &report.by_crate {
            let status_class = if crate_results.failed == 0 { "success" } else { "failure" };

            html.push_str(&format!(r#"
        <h3>{} <span class="{}">[{}/{}]</span></h3>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Kind</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Performance</th>
                </tr>
            </thead>
            <tbody>
"#,
                crate_name,
                status_class,
                crate_results.passed,
                crate_results.passed + crate_results.failed
            ));

            for metric in &crate_results.metrics {
                let status = if metric.success { "✅ Pass" } else { "❌ Fail" };
                let status_class = if metric.success { "success" } else { "failure" };
                let badge_class = match metric.kind {
                    TargetKind::Benchmark => "badge-bench",
                    TargetKind::Example => "badge-example",
                    TargetKind::Test => "badge-test",
                };

                let perf = metric.performance.as_ref()
                    .map(|p| Self::format_time(p.mean_ns))
                    .unwrap_or_else(|| "-".to_string());

                html.push_str(&format!(r#"
                <tr>
                    <td>{}</td>
                    <td><span class="badge {}">{}</span></td>
                    <td class="{}">{}</td>
                    <td class="metric">{:.2}s</td>
                    <td class="metric">{}</td>
                </tr>
"#,
                    metric.target_name,
                    badge_class,
                    metric.kind,
                    status_class,
                    status,
                    metric.duration_secs,
                    perf
                ));
            }

            html.push_str("            </tbody>\n        </table>\n");
        }

        html.push_str("    </section>\n");

        // Tags section
        if !report.by_tag.is_empty() {
            html.push_str(r#"
    <section>
        <h2>🏷️ Tags</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
"#);

            for (tag, targets) in &report.by_tag {
                html.push_str(&format!(
                    r#"            <span class="tag">{} ({})</span>
"#,
                    tag, targets.len()
                ));
            }

            html.push_str("        </div>\n    </section>\n");
        }

        // Footer
        html.push_str(r#"
    <footer>
        <p>Generated by HyperPhysics Benchmark Orchestrator</p>
    </footer>
</div>
</body>
</html>
"#);

        html
    }

    /// Format time value with appropriate units
    fn format_time(ns: f64) -> String {
        if ns >= 1_000_000_000.0 {
            format!("{:.2}s", ns / 1_000_000_000.0)
        } else if ns >= 1_000_000.0 {
            format!("{:.2}ms", ns / 1_000_000.0)
        } else if ns >= 1_000.0 {
            format!("{:.2}µs", ns / 1_000.0)
        } else {
            format!("{:.0}ns", ns)
        }
    }

    /// Save report to file (auto-detect format)
    pub fn save(report: &BenchmarkReport, path: &Path, comparison: Option<&ComparisonReport>) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = if path.extension().map_or(false, |e| e == "html") {
            Self::to_html(report, comparison)
        } else {
            Self::to_json(report)?
        };

        let mut file = fs::File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
    }

    /// Generate console summary
    pub fn console_summary(report: &BenchmarkReport) -> String {
        let mut output = String::new();

        output.push_str("\n");
        output.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║                    HyperPhysics Benchmark Summary                            ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        output.push_str("\n");

        output.push_str(&format!(
            "  Total:      {} targets across {} crates\n",
            report.summary.total_targets,
            report.by_crate.len()
        ));
        output.push_str(&format!(
            "  Passed:     {} ✅\n",
            report.summary.passed
        ));
        output.push_str(&format!(
            "  Failed:     {} ❌\n",
            report.summary.failed
        ));
        output.push_str(&format!(
            "  Duration:   {:.2}s\n",
            report.total_duration_secs
        ));

        if let Some(avg) = report.summary.average_ns {
            output.push_str(&format!(
                "  Avg Time:   {}\n",
                Self::format_time(avg)
            ));
        }

        output.push_str("\n");

        // Show failures
        let failures: Vec<_> = report.results.iter().filter(|r| !r.success).collect();
        if !failures.is_empty() {
            output.push_str("  Failed Targets:\n");
            for f in failures {
                output.push_str(&format!("    • {}::{}\n", f.target.crate_name, f.target.name));
            }
            output.push_str("\n");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_time() {
        assert_eq!(Reporter::format_time(500.0), "500ns");
        assert_eq!(Reporter::format_time(1500.0), "1.50µs");
        assert_eq!(Reporter::format_time(1_500_000.0), "1.50ms");
        assert_eq!(Reporter::format_time(1_500_000_000.0), "1.50s");
    }
}
