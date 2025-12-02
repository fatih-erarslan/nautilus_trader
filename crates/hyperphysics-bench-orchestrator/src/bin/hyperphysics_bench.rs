//! HyperPhysics Benchmark Orchestrator CLI
//!
//! Command-line interface for discovering, running, and reporting benchmarks.
//!
//! ## Usage
//!
//! ```bash
//! # List all discovered targets
//! hyperphysics-bench list
//!
//! # Run all benchmarks
//! hyperphysics-bench run
//!
//! # Run only HNSW benchmarks
//! hyperphysics-bench run --filter hnsw
//!
//! # Run with HTML report generation
//! hyperphysics-bench run --output report.html
//!
//! # Compare with baseline
//! hyperphysics-bench run --baseline baseline.json --output report.html
//! ```

use clap::{Parser, Subcommand};
use console::{style, Emoji};
use hyperphysics_bench_orchestrator::{
    collector::Collector,
    executor::Executor,
    registry::Registry,
    reporter::Reporter,
    OrchestratorConfig, TargetKind,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;

static SPARKLE: Emoji = Emoji("✨ ", "");
static ROCKET: Emoji = Emoji("🚀 ", "");
static CHECK: Emoji = Emoji("✅ ", "[OK] ");
static CROSS: Emoji = Emoji("❌ ", "[FAIL] ");
static CLOCK: Emoji = Emoji("⏱️  ", "");
static CHART: Emoji = Emoji("📊 ", "");

#[derive(Parser)]
#[command(
    name = "hyperphysics-bench",
    about = "HyperPhysics Benchmark Orchestrator",
    version,
    author
)]
struct Cli {
    /// Workspace root path
    #[arg(short, long, default_value = ".")]
    workspace: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List discovered benchmark targets
    List {
        /// Filter by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Filter by kind (benchmark, example, test)
        #[arg(short, long)]
        kind: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Run benchmarks and generate report
    Run {
        /// Filter by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Include only benchmarks
        #[arg(long)]
        benchmarks_only: bool,

        /// Include only examples
        #[arg(long)]
        examples_only: bool,

        /// Maximum parallel executions
        #[arg(short, long, default_value = "4")]
        parallelism: usize,

        /// Timeout per target in seconds
        #[arg(short, long, default_value = "300")]
        timeout: u64,

        /// Output file (json or html)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Baseline file for regression detection
        #[arg(short, long)]
        baseline: Option<PathBuf>,

        /// Run sequentially (disable parallelism)
        #[arg(long)]
        sequential: bool,
    },

    /// Generate report from existing results
    Report {
        /// Input JSON results file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (html)
        #[arg(short, long)]
        output: PathBuf,

        /// Baseline file for comparison
        #[arg(short, long)]
        baseline: Option<PathBuf>,
    },

    /// Show statistics about discovered targets
    Stats,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize registry
    let workspace_root = cli.workspace.canonicalize()?;
    let mut registry = Registry::new(workspace_root.clone());

    println!(
        "\n{}{} HyperPhysics Benchmark Orchestrator",
        SPARKLE,
        style("").bold()
    );
    println!("   Workspace: {}\n", style(workspace_root.display()).cyan());

    // Discover targets
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message("Discovering targets...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    registry.discover()?;

    spinner.finish_with_message(format!(
        "{}Discovered {}",
        CHECK,
        style(format!("{} targets", registry.targets().len())).green()
    ));

    match cli.command {
        Commands::List { filter, kind, detailed } => {
            cmd_list(&registry, filter.as_deref(), kind.as_deref(), detailed)?;
        }

        Commands::Run {
            filter,
            benchmarks_only,
            examples_only,
            parallelism,
            timeout,
            output,
            baseline,
            sequential,
        } => {
            let kinds = if benchmarks_only {
                vec![TargetKind::Benchmark]
            } else if examples_only {
                vec![TargetKind::Example]
            } else {
                vec![TargetKind::Benchmark, TargetKind::Example]
            };

            cmd_run(
                &registry,
                filter.as_deref(),
                kinds,
                parallelism,
                timeout,
                output.as_ref(),
                baseline.as_ref(),
                sequential,
            )
            .await?;
        }

        Commands::Report { input, output, baseline } => {
            cmd_report(&input, &output, baseline.as_ref())?;
        }

        Commands::Stats => {
            cmd_stats(&registry)?;
        }
    }

    Ok(())
}

fn cmd_list(
    registry: &Registry,
    filter: Option<&str>,
    kind: Option<&str>,
    detailed: bool,
) -> anyhow::Result<()> {
    let kind_filter = kind.map(|k| match k {
        "benchmark" | "bench" => TargetKind::Benchmark,
        "example" => TargetKind::Example,
        "test" => TargetKind::Test,
        _ => TargetKind::Benchmark,
    });

    let targets: Vec<_> = registry
        .targets()
        .iter()
        .filter(|t| {
            let name_match = filter
                .map(|f| {
                    t.name.to_lowercase().contains(&f.to_lowercase())
                        || t.crate_name.to_lowercase().contains(&f.to_lowercase())
                })
                .unwrap_or(true);

            let kind_match = kind_filter.map(|k| t.kind == k).unwrap_or(true);

            name_match && kind_match
        })
        .collect();

    println!("\n{} Found {} targets:\n", CHART, style(targets.len()).bold());

    // Group by crate
    let mut by_crate: std::collections::HashMap<&str, Vec<_>> = std::collections::HashMap::new();
    for target in &targets {
        by_crate
            .entry(target.crate_name.as_str())
            .or_default()
            .push(target);
    }

    for (crate_name, crate_targets) in by_crate {
        println!("  {} {}", style("📦").dim(), style(crate_name).cyan().bold());

        for target in crate_targets {
            let kind_badge = match target.kind {
                TargetKind::Benchmark => style("[bench]").blue(),
                TargetKind::Example => style("[example]").green(),
                TargetKind::Test => style("[test]").yellow(),
            };

            println!("     {} {} {}", kind_badge, style("→").dim(), target.name);

            if detailed {
                if let Some(ref desc) = target.description {
                    println!("       {}", style(desc).dim());
                }
                if !target.tags.is_empty() {
                    println!(
                        "       Tags: {}",
                        style(target.tags.join(", ")).dim()
                    );
                }
            }
        }
        println!();
    }

    Ok(())
}

async fn cmd_run(
    registry: &Registry,
    filter: Option<&str>,
    kinds: Vec<TargetKind>,
    parallelism: usize,
    timeout: u64,
    output: Option<&PathBuf>,
    baseline: Option<&PathBuf>,
    sequential: bool,
) -> anyhow::Result<()> {
    let targets: Vec<_> = registry
        .targets()
        .iter()
        .filter(|t| {
            let name_match = filter
                .map(|f| {
                    t.name.to_lowercase().contains(&f.to_lowercase())
                        || t.crate_name.to_lowercase().contains(&f.to_lowercase())
                })
                .unwrap_or(true);

            let kind_match = kinds.contains(&t.kind);

            name_match && kind_match
        })
        .collect();

    if targets.is_empty() {
        println!("\n{} No targets found matching criteria", CROSS);
        return Ok(());
    }

    println!(
        "\n{} Running {} targets (parallelism: {}, {}timeout: {}s)\n",
        ROCKET,
        style(targets.len()).bold(),
        parallelism,
        CLOCK,
        timeout
    );

    // Create executor config
    let config = Arc::new(
        OrchestratorConfig::default()
            .with_parallelism(if sequential { 1 } else { parallelism })
            .with_timeout_secs(timeout)
            .with_release(true),
    );

    let executor = Executor::new(config);

    // Progress bar
    let progress = ProgressBar::new(targets.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );

    // Execute
    let mut collector = Collector::new();

    if sequential {
        for target in &targets {
            progress.set_message(format!("{}::{}", target.crate_name, target.name));

            let result = executor.execute(target).await?;
            let success = result.success;

            collector.add(result);
            progress.inc(1);

            if success {
                progress.println(format!(
                    "  {} {}::{}",
                    CHECK,
                    style(&target.crate_name).cyan(),
                    target.name
                ));
            } else {
                progress.println(format!(
                    "  {} {}::{}",
                    CROSS,
                    style(&target.crate_name).red(),
                    target.name
                ));
            }
        }
    } else {
        let results = executor.execute_parallel(&targets).await;

        for result in results {
            if result.success {
                progress.println(format!(
                    "  {} {}::{}",
                    CHECK,
                    style(&result.target.crate_name).cyan(),
                    result.target.name
                ));
            } else {
                progress.println(format!(
                    "  {} {}::{}",
                    CROSS,
                    style(&result.target.crate_name).red(),
                    result.target.name
                ));
            }
            progress.inc(1);
            collector.add(result);
        }
    }

    progress.finish_with_message("Done");

    // Generate report
    let report = collector.generate_report();

    // Load baseline and compare
    let comparison = if let Some(baseline_path) = baseline {
        if baseline_path.exists() {
            match Collector::load_baseline(baseline_path) {
                Ok(baseline_report) => {
                    let comp = collector.compare_with_baseline(&baseline_report);
                    println!("\n{} Comparison with baseline:", CHART);

                    if comp.has_regressions() {
                        println!(
                            "  {} {} regressions detected!",
                            CROSS,
                            style(comp.regressions.len()).red().bold()
                        );
                        for r in &comp.regressions {
                            println!(
                                "     {} {} (+{:.1}%)",
                                style("↑").red(),
                                r.target_id,
                                r.change_pct
                            );
                        }
                    }

                    if !comp.improvements.is_empty() {
                        println!(
                            "  {} {} improvements",
                            CHECK,
                            style(comp.improvements.len()).green().bold()
                        );
                    }

                    Some(comp)
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load baseline: {}", e);
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Print summary
    println!("{}", Reporter::console_summary(&report));

    // Save output
    if let Some(output_path) = output {
        Reporter::save(&report, output_path, comparison.as_ref())?;
        println!(
            "{} Report saved to: {}",
            CHECK,
            style(output_path.display()).cyan()
        );
    }

    // Exit with error if there were failures
    if report.summary.failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn cmd_report(input: &PathBuf, output: &PathBuf, baseline: Option<&PathBuf>) -> anyhow::Result<()> {
    let report = Collector::load_baseline(input)?;

    let comparison = if let Some(baseline_path) = baseline {
        if baseline_path.exists() {
            let baseline_report = Collector::load_baseline(baseline_path)?;
            let mut collector = Collector::new();
            for result in &report.results {
                collector.add(result.clone());
            }
            Some(collector.compare_with_baseline(&baseline_report))
        } else {
            None
        }
    } else {
        None
    };

    Reporter::save(&report, output, comparison.as_ref())?;
    println!(
        "{} Report generated: {}",
        CHECK,
        style(output.display()).cyan()
    );

    Ok(())
}

fn cmd_stats(registry: &Registry) -> anyhow::Result<()> {
    let stats = registry.stats();

    println!("\n{} Registry Statistics:\n", CHART);
    println!("  Total targets:  {}", style(stats.total_targets).bold());
    println!("  Benchmarks:     {}", style(stats.benchmarks).blue());
    println!("  Examples:       {}", style(stats.examples).green());
    println!("  Tests:          {}", style(stats.tests).yellow());
    println!("  Crates:         {}", style(stats.crates).cyan());
    println!();

    // Show crates
    println!("  Crates with targets:");
    for crate_name in registry.crates() {
        let count = registry.targets_by_crate(crate_name).len();
        println!("    {} {} ({})", style("→").dim(), crate_name, count);
    }
    println!();

    Ok(())
}
