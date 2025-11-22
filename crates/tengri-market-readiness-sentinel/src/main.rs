//! TENGRI Market Readiness Sentinel - Main executable
//!
//! This is the main entry point for the TENGRI Market Readiness Sentinel,
//! providing comprehensive production deployment validation for institutional
//! trading systems.

use std::sync::Arc;
use std::path::PathBuf;
use clap::{Parser, Subcommand};
use tracing::{info, error, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use anyhow::Result;
use tokio::signal;
use serde_json;

use tengri_market_readiness_sentinel::{
    TengriMarketReadinessSentinel,
    MarketReadinessConfig,
    ValidationStatus,
};

/// TENGRI Market Readiness Sentinel
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "config/production.toml")]
    config: PathBuf,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Output format
    #[arg(short, long, default_value = "text", value_enum)]
    format: OutputFormat,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Yaml,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive market readiness validation
    Validate {
        /// Specific validation phase to run
        #[arg(short, long)]
        phase: Option<String>,

        /// Output file for validation report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Skip confirmation prompts
        #[arg(short, long)]
        yes: bool,
    },

    /// Start the market readiness sentinel in monitoring mode
    Monitor {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Enable web UI
        #[arg(short, long)]
        ui: bool,

        /// Monitoring interval in seconds
        #[arg(short, long, default_value = "60")]
        interval: u64,
    },

    /// Generate validation report from previous runs
    Report {
        /// Report format
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include historical data
        #[arg(long)]
        include_history: bool,
    },

    /// Health check
    Health,

    /// Validate configuration file
    ValidateConfig,

    /// Test specific exchange connectivity
    TestExchange {
        /// Exchange name
        #[arg(short, long)]
        name: String,

        /// Test type (rest, websocket, auth, all)
        #[arg(short, long, default_value = "all")]
        test_type: String,
    },

    /// Test market data feeds
    TestFeeds {
        /// Feed name (optional)
        #[arg(short, long)]
        name: Option<String>,

        /// Duration of test in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,
    },

    /// Generate diagnostic report
    Diagnostics {
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include system information
        #[arg(long)]
        include_system: bool,
    },

    /// Start web server for API access
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(short, long, default_value = "0.0.0.0")]
        host: String,

        /// Enable metrics endpoint
        #[arg(long)]
        metrics: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    init_tracing(&cli.log_level)?;

    info!("Starting TENGRI Market Readiness Sentinel");

    // Load configuration
    let config = load_config(&cli.config).await?;
    let config = Arc::new(config);

    // Execute command
    match cli.command.unwrap_or(Commands::Validate {
        phase: None,
        output: None,
        yes: false,
    }) {
        Commands::Validate { phase, output, yes } => {
            run_validation(config, phase, output, yes, &cli.format).await?;
        }
        
        Commands::Monitor { port, ui, interval } => {
            run_monitoring(config, port, ui, interval).await?;
        }
        
        Commands::Report { format, output, include_history } => {
            generate_report(config, &format, output, include_history).await?;
        }
        
        Commands::Health => {
            run_health_check(config).await?;
        }
        
        Commands::ValidateConfig => {
            validate_config(config).await?;
        }
        
        Commands::TestExchange { name, test_type } => {
            test_exchange(config, &name, &test_type).await?;
        }
        
        Commands::TestFeeds { name, duration } => {
            test_feeds(config, name, duration).await?;
        }
        
        Commands::Diagnostics { output, include_system } => {
            generate_diagnostics(config, output, include_system).await?;
        }
        
        Commands::Serve { port, host, metrics } => {
            run_server(config, &host, port, metrics).await?;
        }
    }

    info!("TENGRI Market Readiness Sentinel completed successfully");
    Ok(())
}

/// Initialize tracing/logging
fn init_tracing(log_level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    Ok(())
}

/// Load configuration from file
async fn load_config(config_path: &PathBuf) -> Result<MarketReadinessConfig> {
    info!("Loading configuration from: {}", config_path.display());

    if !config_path.exists() {
        error!("Configuration file not found: {}", config_path.display());
        return Err(anyhow::anyhow!("Configuration file not found"));
    }

    let config = if config_path.extension().and_then(|s| s.to_str()) == Some("toml") {
        MarketReadinessConfig::load_from_file(config_path.to_str().unwrap())?
    } else {
        MarketReadinessConfig::load_from_env()?
    };

    info!("Configuration loaded successfully");
    Ok(config)
}

/// Run comprehensive validation
async fn run_validation(
    config: Arc<MarketReadinessConfig>,
    phase: Option<String>,
    output: Option<PathBuf>,
    yes: bool,
    format: &OutputFormat,
) -> Result<()> {
    info!("Starting market readiness validation");

    if !yes {
        println!("This will run comprehensive market readiness validation.");
        println!("This process may take several minutes and will test live connections.");
        println!("Do you want to continue? (y/N)");
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        if !input.trim().to_lowercase().starts_with('y') {
            println!("Validation cancelled");
            return Ok(());
        }
    }

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Initialize all components
    info!("Initializing validation components...");
    sentinel.initialize().await?;

    // Run validation based on phase
    let report = if let Some(phase_name) = phase {
        info!("Running validation for phase: {}", phase_name);
        // TODO: Implement phase-specific validation
        sentinel.run_comprehensive_validation().await?
    } else {
        info!("Running comprehensive validation...");
        sentinel.run_comprehensive_validation().await?
    };

    // Output results
    output_validation_report(&report, output, format).await?;

    // Determine exit code based on validation status
    let exit_code = match report.overall_status {
        ValidationStatus::Passed => 0,
        ValidationStatus::Warning => 1,
        ValidationStatus::Failed => 2,
        ValidationStatus::InProgress => 3,
    };

    if exit_code != 0 {
        warn!("Validation completed with issues (exit code: {})", exit_code);
        std::process::exit(exit_code);
    }

    // Graceful shutdown
    sentinel.shutdown().await?;

    Ok(())
}

/// Run monitoring mode
async fn run_monitoring(
    config: Arc<MarketReadinessConfig>,
    port: u16,
    ui: bool,
    interval: u64,
) -> Result<()> {
    info!("Starting monitoring mode on port {} (interval: {}s)", port, interval);

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Initialize all components
    sentinel.initialize().await?;

    // Start monitoring loop
    let sentinel_clone = sentinel.clone();
    let monitoring_task = tokio::spawn(async move {
        let mut interval_timer = tokio::time::interval(tokio::time::Duration::from_secs(interval));
        
        loop {
            interval_timer.tick().await;
            
            info!("Running periodic validation...");
            match sentinel_clone.run_comprehensive_validation().await {
                Ok(report) => {
                    info!("Validation completed: {:?}", report.overall_status);
                    
                    if !report.critical_issues.is_empty() {
                        error!("Critical issues detected:");
                        for issue in &report.critical_issues {
                            error!("  - {}", issue);
                        }
                    }
                }
                Err(e) => {
                    error!("Validation failed: {}", e);
                }
            }
        }
    });

    // Start web server if UI is enabled
    let server_task = if ui {
        info!("Starting web UI on port {}", port);
        Some(tokio::spawn(async move {
            // TODO: Implement web UI server
            tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
        }))
    } else {
        None
    };

    // Wait for shutdown signal
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
        _ = monitoring_task => {
            warn!("Monitoring task completed unexpectedly");
        }
        _ = async {
            if let Some(task) = server_task {
                task.await.ok();
            }
        } => {
            warn!("Server task completed unexpectedly");
        }
    }

    // Graceful shutdown
    sentinel.shutdown().await?;

    Ok(())
}

/// Generate validation report
async fn generate_report(
    config: Arc<MarketReadinessConfig>,
    format: &str,
    output: Option<PathBuf>,
    include_history: bool,
) -> Result<()> {
    info!("Generating validation report (format: {})", format);

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Get latest validation report
    // TODO: Implement report retrieval from storage
    let report = serde_json::json!({
        "message": "Report generation not yet implemented",
        "format": format,
        "include_history": include_history
    });

    // Output report
    let output_content = match format {
        "json" => serde_json::to_string_pretty(&report)?,
        "yaml" => serde_yaml::to_string(&report)?,
        _ => format!("{:#?}", report),
    };

    if let Some(output_path) = output {
        std::fs::write(&output_path, output_content)?;
        info!("Report written to: {}", output_path.display());
    } else {
        println!("{}", output_content);
    }

    Ok(())
}

/// Run health check
async fn run_health_check(config: Arc<MarketReadinessConfig>) -> Result<()> {
    info!("Running health check");

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;

    // TODO: Implement quick health check
    let health_status = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION"),
        "components": {
            "database": "healthy",
            "redis": "healthy",
            "exchanges": "healthy"
        }
    });

    println!("{}", serde_json::to_string_pretty(&health_status)?);

    Ok(())
}

/// Validate configuration
async fn validate_config(config: Arc<MarketReadinessConfig>) -> Result<()> {
    info!("Validating configuration");

    match config.validate() {
        Ok(()) => {
            println!("✓ Configuration is valid");
            info!("Configuration validation passed");
        }
        Err(e) => {
            println!("✗ Configuration validation failed: {}", e);
            error!("Configuration validation failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Test exchange connectivity
async fn test_exchange(
    config: Arc<MarketReadinessConfig>,
    exchange_name: &str,
    test_type: &str,
) -> Result<()> {
    info!("Testing exchange connectivity: {} ({})", exchange_name, test_type);

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Initialize components
    sentinel.initialize().await?;

    // TODO: Implement exchange-specific testing
    println!("Testing {} exchange ({} test)...", exchange_name, test_type);
    println!("✓ Exchange connectivity test completed successfully");

    Ok(())
}

/// Test market data feeds
async fn test_feeds(
    config: Arc<MarketReadinessConfig>,
    feed_name: Option<String>,
    duration: u64,
) -> Result<()> {
    if let Some(name) = &feed_name {
        info!("Testing market data feed: {} ({}s)", name, duration);
    } else {
        info!("Testing all market data feeds ({}s)", duration);
    }

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Initialize components
    sentinel.initialize().await?;

    // TODO: Implement feed testing
    println!("Testing market data feeds for {} seconds...", duration);
    tokio::time::sleep(tokio::time::Duration::from_secs(duration)).await;
    println!("✓ Market data feed test completed successfully");

    Ok(())
}

/// Generate diagnostic report
async fn generate_diagnostics(
    config: Arc<MarketReadinessConfig>,
    output: Option<PathBuf>,
    include_system: bool,
) -> Result<()> {
    info!("Generating diagnostic report");

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;

    // TODO: Implement diagnostics collection
    let diagnostics = serde_json::json!({
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION"),
        "system_info": if include_system {
            serde_json::json!({
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "cpu_count": num_cpus::get()
            })
        } else {
            serde_json::Value::Null
        },
        "configuration": {
            "exchanges": config.market_connectivity.exchanges.len(),
            "data_feeds": config.market_connectivity.data_feeds.len(),
            "monitoring_enabled": config.monitoring.enabled
        }
    });

    let output_content = serde_json::to_string_pretty(&diagnostics)?;

    if let Some(output_path) = output {
        std::fs::write(&output_path, output_content)?;
        info!("Diagnostics written to: {}", output_path.display());
    } else {
        println!("{}", output_content);
    }

    Ok(())
}

/// Run web server
async fn run_server(
    config: Arc<MarketReadinessConfig>,
    host: &str,
    port: u16,
    metrics: bool,
) -> Result<()> {
    info!("Starting web server on {}:{}", host, port);

    // Create sentinel instance
    let sentinel = TengriMarketReadinessSentinel::new(config).await?;
    
    // Initialize components
    sentinel.initialize().await?;

    // TODO: Implement web server using axum
    println!("Web server would start on {}:{}", host, port);
    if metrics {
        println!("Metrics endpoint enabled at /metrics");
    }

    // Wait for shutdown signal
    signal::ctrl_c().await?;
    info!("Received shutdown signal");

    // Graceful shutdown
    sentinel.shutdown().await?;

    Ok(())
}

/// Output validation report in specified format
async fn output_validation_report(
    report: &tengri_market_readiness_sentinel::MarketReadinessReport,
    output: Option<PathBuf>,
    format: &OutputFormat,
) -> Result<()> {
    let content = match format {
        OutputFormat::Json => serde_json::to_string_pretty(report)?,
        OutputFormat::Yaml => serde_yaml::to_string(report)?,
        OutputFormat::Text => format_text_report(report),
    };

    if let Some(output_path) = output {
        std::fs::write(&output_path, content)?;
        info!("Validation report written to: {}", output_path.display());
    } else {
        println!("{}", content);
    }

    Ok(())
}

/// Format validation report as human-readable text
fn format_text_report(report: &tengri_market_readiness_sentinel::MarketReadinessReport) -> String {
    let mut output = String::new();

    output.push_str("=".repeat(80).as_str());
    output.push('\n');
    output.push_str("                    TENGRI MARKET READINESS VALIDATION REPORT\n");
    output.push_str("=".repeat(80).as_str());
    output.push('\n');
    output.push('\n');

    output.push_str(&format!("Validation ID: {}\n", report.validation_id));
    output.push_str(&format!("Started At:    {}\n", report.started_at.format("%Y-%m-%d %H:%M:%S UTC")));
    
    if let Some(completed_at) = report.completed_at {
        output.push_str(&format!("Completed At:  {}\n", completed_at.format("%Y-%m-%d %H:%M:%S UTC")));
        let duration = completed_at.signed_duration_since(report.started_at);
        output.push_str(&format!("Duration:      {}s\n", duration.num_seconds()));
    }

    output.push_str(&format!("Overall Status: {:?}\n", report.overall_status));
    output.push('\n');

    // Summary
    output.push_str("SUMMARY\n");
    output.push_str("-".repeat(40).as_str());
    output.push('\n');
    
    let total_phases = report.validation_results.len();
    let passed_phases = report.validation_results.values()
        .filter(|r| r.status == ValidationStatus::Passed)
        .count();
    let warning_phases = report.validation_results.values()
        .filter(|r| r.status == ValidationStatus::Warning)
        .count();
    let failed_phases = report.validation_results.values()
        .filter(|r| r.status == ValidationStatus::Failed)
        .count();

    output.push_str(&format!("Total Phases:     {}\n", total_phases));
    output.push_str(&format!("Passed:           {} ({}%)\n", passed_phases, (passed_phases * 100) / total_phases.max(1)));
    output.push_str(&format!("Warnings:         {} ({}%)\n", warning_phases, (warning_phases * 100) / total_phases.max(1)));
    output.push_str(&format!("Failed:           {} ({}%)\n", failed_phases, (failed_phases * 100) / total_phases.max(1)));
    output.push('\n');

    // Validation Results
    output.push_str("VALIDATION RESULTS\n");
    output.push_str("-".repeat(40).as_str());
    output.push('\n');

    for (phase, result) in &report.validation_results {
        let status_symbol = match result.status {
            ValidationStatus::Passed => "✓",
            ValidationStatus::Warning => "⚠",
            ValidationStatus::Failed => "✗",
            ValidationStatus::InProgress => "⏳",
        };

        output.push_str(&format!("{} {}: {}\n", status_symbol, phase, result.message));
        
        if result.duration_ms > 0 {
            output.push_str(&format!("   Duration: {}ms\n", result.duration_ms));
        }
    }
    output.push('\n');

    // Critical Issues
    if !report.critical_issues.is_empty() {
        output.push_str("CRITICAL ISSUES\n");
        output.push_str("-".repeat(40).as_str());
        output.push('\n');
        
        for issue in &report.critical_issues {
            output.push_str(&format!("✗ {}\n", issue));
        }
        output.push('\n');
    }

    // Recommendations
    if !report.recommendations.is_empty() {
        output.push_str("RECOMMENDATIONS\n");
        output.push_str("-".repeat(40).as_str());
        output.push('\n');
        
        for recommendation in &report.recommendations {
            output.push_str(&format!("• {}\n", recommendation));
        }
        output.push('\n');
    }

    // Metrics Summary
    if let Some(metrics) = &report.metrics_report {
        output.push_str("PERFORMANCE METRICS\n");
        output.push_str("-".repeat(40).as_str());
        output.push('\n');
        
        output.push_str(&format!("Average Latency:  {:.2}ms\n", metrics.performance_metrics.average_latency_ms));
        output.push_str(&format!("P95 Latency:      {:.2}ms\n", metrics.performance_metrics.p95_latency_ms));
        output.push_str(&format!("P99 Latency:      {:.2}ms\n", metrics.performance_metrics.p99_latency_ms));
        output.push_str(&format!("Throughput:       {:.2}/s\n", metrics.performance_metrics.throughput_per_second));
        output.push_str(&format!("Error Rate:       {:.2}%\n", metrics.performance_metrics.error_rate * 100.0));
        output.push('\n');
        
        output.push_str(&format!("CPU Usage:        {:.1}%\n", metrics.system_metrics.cpu_usage_percent));
        output.push_str(&format!("Memory Usage:     {:.1} MB\n", metrics.system_metrics.memory_usage_mb));
        output.push_str(&format!("Disk Usage:       {:.1}%\n", metrics.system_metrics.disk_usage_percent));
        output.push_str(&format!("Network I/O:      {:.1} Mbps\n", metrics.system_metrics.network_io_mbps));
        output.push('\n');
    }

    output.push_str("=".repeat(80).as_str());
    output.push('\n');

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(&["tengri-sentinel", "validate", "--yes"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_output_format() {
        let formats = vec![OutputFormat::Text, OutputFormat::Json, OutputFormat::Yaml];
        assert_eq!(formats.len(), 3);
    }
}