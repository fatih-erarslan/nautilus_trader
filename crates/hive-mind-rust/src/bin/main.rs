//! Main entry point for the hive mind system

use std::path::PathBuf;
use std::process;
use clap::{Parser, Subcommand};
use tracing::{info, error, debug};
use tracing_subscriber::{EnvFilter, fmt};

use hive_mind_rust::{
    HiveMind, HiveMindBuilder, HiveMindConfig,
    config::{NetworkConfig, ConsensusConfig, MemoryConfig, NeuralConfig, AgentConfig},
    error::{HiveMindError, Result},
};

/// Command line interface for the hive mind system
#[derive(Parser, Debug)]
#[command(name = "hive-mind")]
#[command(about = "A distributed collective intelligence system for Ximera trading platform")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    
    /// Log level (error, warn, info, debug, trace)
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,
    
    /// Working directory
    #[arg(short, long)]
    work_dir: Option<PathBuf>,
    
    #[command(subcommand)]
    command: Commands,
}

/// Available commands
#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the hive mind system
    Start {
        /// Daemon mode
        #[arg(short, long)]
        daemon: bool,
        
        /// PID file path
        #[arg(short, long)]
        pid_file: Option<PathBuf>,
    },
    
    /// Stop the hive mind system
    Stop {
        /// PID file path
        #[arg(short, long)]
        pid_file: Option<PathBuf>,
    },
    
    /// Show system status
    Status {
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    
    /// Generate default configuration
    Config {
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Configuration template type
        #[arg(short, long, default_value = "default")]
        template: String,
    },
    
    /// Validate configuration
    Validate {
        /// Configuration file path
        #[arg(short, long)]
        config: PathBuf,
    },
    
    /// Show version information
    Version,
    
    /// Benchmark the system
    Benchmark {
        /// Benchmark type
        #[arg(short, long, default_value = "all")]
        bench_type: String,
        
        /// Duration in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,
        
        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Interactive shell mode
    Shell,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    // Initialize logging
    if let Err(e) = init_logging(&cli.log_level, cli.json_logs) {
        eprintln!("Failed to initialize logging: {}", e);
        process::exit(1);
    }
    
    // Change working directory if specified
    if let Some(work_dir) = &cli.work_dir {
        if let Err(e) = std::env::set_current_dir(work_dir) {
            error!("Failed to change working directory: {}", e);
            process::exit(1);
        }
    }
    
    // Execute command
    let result = match cli.command {
        Commands::Start { daemon, pid_file } => {
            start_command(cli.config, daemon, pid_file).await
        }
        Commands::Stop { pid_file } => {
            stop_command(pid_file).await
        }
        Commands::Status { format } => {
            status_command(format).await
        }
        Commands::Config { output, template } => {
            config_command(output, template).await
        }
        Commands::Validate { config } => {
            validate_command(config).await
        }
        Commands::Version => {
            version_command().await
        }
        Commands::Benchmark { bench_type, duration, output } => {
            benchmark_command(bench_type, duration, output).await
        }
        Commands::Shell => {
            shell_command().await
        }
    };
    
    if let Err(e) = result {
        error!("Command failed: {}", e);
        process::exit(1);
    }
}

/// Initialize logging system
fn init_logging(log_level: &str, json_logs: bool) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .map_err(|e| HiveMindError::Internal(format!("Invalid log level: {}", e)))?;
    
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);
    
    if json_logs {
        subscriber.json().init();
    } else {
        subscriber.init();
    }
    
    Ok(())
}

/// Start the hive mind system
async fn start_command(
    config_path: Option<PathBuf>,
    daemon: bool,
    pid_file: Option<PathBuf>,
) -> Result<()> {
    info!("Starting hive mind system");
    
    // Load configuration
    let config = load_configuration(config_path).await?;
    
    // Validate configuration
    config.validate()?;
    
    // Write PID file if specified
    if let Some(pid_path) = &pid_file {
        write_pid_file(pid_path)?;
    }
    
    // Create and start hive mind
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    // Start the system
    hive_mind.start().await?;
    
    info!("Hive mind system started successfully");
    
    if daemon {
        // In daemon mode, run indefinitely
        let mut shutdown_signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .map_err(|e| HiveMindError::Internal(format!("Failed to setup signal handler: {}", e)))?;
        
        tokio::select! {
            _ = shutdown_signal.recv() => {
                info!("Received shutdown signal");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C");
            }
        }
        
        // Graceful shutdown
        info!("Shutting down hive mind system");
        hive_mind.stop().await?;
        
        // Clean up PID file
        if let Some(pid_path) = &pid_file {
            let _ = std::fs::remove_file(pid_path);
        }
        
        info!("Hive mind system stopped");
    } else {
        // In non-daemon mode, wait for Ctrl+C
        tokio::signal::ctrl_c().await
            .map_err(|e| HiveMindError::Internal(format!("Failed to wait for Ctrl+C: {}", e)))?;
        
        info!("Shutting down hive mind system");
        hive_mind.stop().await?;
        info!("Hive mind system stopped");
    }
    
    Ok(())
}

/// Stop the hive mind system
async fn stop_command(pid_file: Option<PathBuf>) -> Result<()> {
    info!("Stopping hive mind system");
    
    if let Some(pid_path) = pid_file {
        // Read PID from file and send termination signal
        let pid_str = std::fs::read_to_string(&pid_path)
            .map_err(|e| HiveMindError::Internal(format!("Failed to read PID file: {}", e)))?;
        
        let pid: u32 = pid_str.trim().parse()
            .map_err(|e| HiveMindError::Internal(format!("Invalid PID in file: {}", e)))?;
        
        // Send SIGTERM signal (Unix-specific)
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            
            let pid = Pid::from_raw(pid as i32);
            kill(pid, Signal::SIGTERM)
                .map_err(|e| HiveMindError::Internal(format!("Failed to send signal: {}", e)))?;
        }
        
        // Remove PID file
        std::fs::remove_file(&pid_path)
            .map_err(|e| HiveMindError::Internal(format!("Failed to remove PID file: {}", e)))?;
        
        info!("Stop signal sent to process {}", pid);
    } else {
        return Err(HiveMindError::InvalidState {
            message: "No PID file specified".to_string(),
        });
    }
    
    Ok(())
}

/// Show system status
async fn status_command(format: String) -> Result<()> {
    info!("Checking hive mind system status");
    
    // This would typically connect to a running instance to get status
    // For now, we'll show a placeholder status
    
    match format.as_str() {
        "json" => {
            let status = serde_json::json!({
                "status": "unknown",
                "version": env!("CARGO_PKG_VERSION"),
                "timestamp": chrono::Utc::now(),
                "message": "Status check not implemented yet"
            });
            match serde_json::to_string_pretty(&status) {
                Ok(json) => println!("{}", json),
                Err(e) => {
                    error!("Failed to serialize status to JSON: {}", e);
                    return Err(HiveMindError::Serialization(e));
                }
            }
        }
        "text" | _ => {
            println!("Hive Mind System Status");
            println!("=======================");
            println!("Version: {}", env!("CARGO_PKG_VERSION"));
            println!("Status: Unknown (status check not implemented yet)");
            println!("Timestamp: {}", chrono::Utc::now());
        }
    }
    
    Ok(())
}

/// Generate configuration file
async fn config_command(output: Option<PathBuf>, template: String) -> Result<()> {
    info!("Generating configuration");
    
    let config = match template.as_str() {
        "minimal" => create_minimal_config(),
        "production" => create_production_config(),
        "development" => create_development_config(),
        "default" | _ => HiveMindConfig::default(),
    };
    
    let config_toml = toml::to_string_pretty(&config)
        .map_err(|e| HiveMindError::Internal(format!("Failed to serialize config: {}", e)))?;
    
    if let Some(output_path) = output {
        std::fs::write(&output_path, config_toml)
            .map_err(|e| HiveMindError::Internal(format!("Failed to write config file: {}", e)))?;
        info!("Configuration written to: {}", output_path.display());
    } else {
        println!("{}", config_toml);
    }
    
    Ok(())
}

/// Validate configuration file
async fn validate_command(config_path: PathBuf) -> Result<()> {
    info!("Validating configuration file: {}", config_path.display());
    
    let config = HiveMindConfig::load_from_file(&config_path)?;
    config.validate()?;
    
    println!("Configuration is valid");
    Ok(())
}

/// Show version information
async fn version_command() -> Result<()> {
    println!("Hive Mind System");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Built: {}", env!("VERGEN_BUILD_TIMESTAMP"));
    println!("Commit: {}", env!("VERGEN_GIT_SHA"));
    println!("Rust: {}", env!("VERGEN_RUSTC_SEMVER"));
    Ok(())
}

/// Run benchmark tests
async fn benchmark_command(
    bench_type: String,
    duration: u64,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Running benchmark: {} for {}s", bench_type, duration);
    
    // Create minimal config for benchmarking
    let config = create_minimal_config();
    
    // Create hive mind instance
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    hive_mind.start().await?;
    
    // Run benchmark based on type
    let results = match bench_type.as_str() {
        "consensus" => run_consensus_benchmark(&hive_mind, duration).await?,
        "memory" => run_memory_benchmark(&hive_mind, duration).await?,
        "neural" => run_neural_benchmark(&hive_mind, duration).await?,
        "network" => run_network_benchmark(&hive_mind, duration).await?,
        "all" => run_all_benchmarks(&hive_mind, duration).await?,
        _ => {
            return Err(HiveMindError::InvalidState {
                message: format!("Unknown benchmark type: {}", bench_type),
            });
        }
    };
    
    // Stop the system
    hive_mind.stop().await?;
    
    // Output results
    if let Some(output_path) = output {
        let results_json = serde_json::to_string_pretty(&results)
            .map_err(|e| HiveMindError::Internal(format!("Failed to serialize results: {}", e)))?;
        std::fs::write(&output_path, results_json)
            .map_err(|e| HiveMindError::Internal(format!("Failed to write results: {}", e)))?;
        info!("Benchmark results written to: {}", output_path.display());
    } else {
        match serde_json::to_string_pretty(&results) {
            Ok(json) => println!("{}", json),
            Err(e) => {
                error!("Failed to serialize benchmark results to JSON: {}", e);
                return Err(HiveMindError::Serialization(e));
            }
        }
    }
    
    Ok(())
}

/// Interactive shell mode
async fn shell_command() -> Result<()> {
    info!("Starting interactive shell");
    
    println!("Hive Mind Interactive Shell");
    println!("Type 'help' for available commands, 'exit' to quit");
    
    // This would implement an interactive shell
    // For now, just show a placeholder
    loop {
        print!("> ");
        use std::io::{self, Write};
        if let Err(e) = io::stdout().flush() {
            warn!("Failed to flush stdout: {}", e);
        }
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)
            .map_err(|e| HiveMindError::Io(e))?
            .saturating_sub(1) // Remove newline character count;
        let input = input.trim();
        
        match input {
            "exit" | "quit" => break,
            "help" => {
                println!("Available commands:");
                println!("  help     - Show this help");
                println!("  status   - Show system status");
                println!("  agents   - List active agents");
                println!("  memory   - Show memory usage");
                println!("  metrics  - Show performance metrics");
                println!("  exit     - Exit shell");
            }
            "status" => {
                println!("System status: Not implemented yet");
            }
            "agents" => {
                println!("Active agents: Not implemented yet");
            }
            "memory" => {
                println!("Memory usage: Not implemented yet");
            }
            "metrics" => {
                println!("Performance metrics: Not implemented yet");
            }
            "" => continue,
            _ => {
                println!("Unknown command: {}. Type 'help' for available commands.", input);
            }
        }
    }
    
    println!("Goodbye!");
    Ok(())
}

/// Load configuration from file or create default
async fn load_configuration(config_path: Option<PathBuf>) -> Result<HiveMindConfig> {
    if let Some(path) = config_path {
        debug!("Loading configuration from: {}", path.display());
        HiveMindConfig::load_from_file(path)
    } else {
        debug!("Using default configuration");
        Ok(HiveMindConfig::default())
    }
}

/// Write process ID to file
fn write_pid_file(pid_path: &PathBuf) -> Result<()> {
    let pid = process::id();
    std::fs::write(pid_path, pid.to_string())
        .map_err(|e| HiveMindError::Internal(format!("Failed to write PID file: {}", e)))?;
    debug!("PID {} written to: {}", pid, pid_path.display());
    Ok(())
}

/// Create minimal configuration for testing/benchmarking
fn create_minimal_config() -> HiveMindConfig {
    let mut config = HiveMindConfig::default();
    
    // Reduce resource usage for testing
    config.agents.max_agents = 5;
    config.network.max_peers = 10;
    config.memory.max_pool_size = 50 * 1024 * 1024; // 50MB
    config.consensus.min_nodes = 1;
    config.metrics.enabled = false;
    
    config
}

/// Create production configuration
fn create_production_config() -> HiveMindConfig {
    let mut config = HiveMindConfig::default();
    
    // Production optimizations
    config.agents.max_agents = 100;
    config.network.max_peers = 50;
    config.memory.max_pool_size = 2 * 1024 * 1024 * 1024; // 2GB
    config.consensus.min_nodes = 5;
    config.metrics.enabled = true;
    config.security.enable_encryption = true;
    
    config
}

/// Create development configuration
fn create_development_config() -> HiveMindConfig {
    let mut config = HiveMindConfig::default();
    
    // Development settings
    config.agents.max_agents = 20;
    config.network.max_peers = 20;
    config.memory.max_pool_size = 200 * 1024 * 1024; // 200MB
    config.consensus.min_nodes = 3;
    config.metrics.enabled = true;
    config.security.enable_encryption = false; // Faster for development
    
    config
}

/// Benchmark consensus operations
async fn run_consensus_benchmark(hive_mind: &HiveMind, duration: u64) -> Result<serde_json::Value> {
    let start_time = std::time::Instant::now();
    let mut operations = 0;
    
    while start_time.elapsed().as_secs() < duration {
        // Submit test proposals
        let proposal = serde_json::json!({
            "type": "test_proposal",
            "value": operations,
            "timestamp": chrono::Utc::now()
        });
        
        if hive_mind.submit_proposal(proposal).await.is_ok() {
            operations += 1;
        }
        
        // Small delay to prevent overwhelming the system
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    
    let elapsed = start_time.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();
    
    Ok(serde_json::json!({
        "benchmark": "consensus",
        "duration_seconds": elapsed.as_secs_f64(),
        "total_operations": operations,
        "operations_per_second": ops_per_sec
    }))
}

/// Benchmark memory operations
async fn run_memory_benchmark(hive_mind: &HiveMind, duration: u64) -> Result<serde_json::Value> {
    let start_time = std::time::Instant::now();
    let mut operations = 0;
    
    while start_time.elapsed().as_secs() < duration {
        let key = format!("benchmark_key_{}", operations);
        let value = serde_json::json!({
            "data": format!("test_data_{}", operations),
            "timestamp": chrono::Utc::now()
        });
        
        if hive_mind.store_knowledge(&key, value).await.is_ok() {
            operations += 1;
        }
        
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    
    let elapsed = start_time.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();
    
    Ok(serde_json::json!({
        "benchmark": "memory",
        "duration_seconds": elapsed.as_secs_f64(),
        "total_operations": operations,
        "operations_per_second": ops_per_sec
    }))
}

/// Benchmark neural operations
async fn run_neural_benchmark(hive_mind: &HiveMind, duration: u64) -> Result<serde_json::Value> {
    let start_time = std::time::Instant::now();
    let mut operations = 0;
    
    while start_time.elapsed().as_secs() < duration {
        // Generate test data for neural processing
        let test_data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        
        if hive_mind.get_neural_insights(&test_data).await.is_ok() {
            operations += 1;
        }
        
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    
    let elapsed = start_time.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();
    
    Ok(serde_json::json!({
        "benchmark": "neural",
        "duration_seconds": elapsed.as_secs_f64(),
        "total_operations": operations,
        "operations_per_second": ops_per_sec
    }))
}

/// Benchmark network operations
async fn run_network_benchmark(_hive_mind: &HiveMind, duration: u64) -> Result<serde_json::Value> {
    // Network benchmarking would require multiple nodes
    // For now, just simulate the benchmark
    
    let start_time = std::time::Instant::now();
    let operations = duration * 100; // Simulated operations
    
    tokio::time::sleep(std::time::Duration::from_secs(duration)).await;
    
    let elapsed = start_time.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();
    
    Ok(serde_json::json!({
        "benchmark": "network",
        "duration_seconds": elapsed.as_secs_f64(),
        "total_operations": operations,
        "operations_per_second": ops_per_sec,
        "note": "Simulated benchmark - requires multiple nodes for real testing"
    }))
}

/// Run all benchmarks
async fn run_all_benchmarks(hive_mind: &HiveMind, duration: u64) -> Result<serde_json::Value> {
    let per_benchmark_duration = duration / 4; // Split time among benchmarks
    
    let consensus_results = run_consensus_benchmark(hive_mind, per_benchmark_duration).await?;
    let memory_results = run_memory_benchmark(hive_mind, per_benchmark_duration).await?;
    let neural_results = run_neural_benchmark(hive_mind, per_benchmark_duration).await?;
    let network_results = run_network_benchmark(hive_mind, per_benchmark_duration).await?;
    
    Ok(serde_json::json!({
        "benchmark": "all",
        "total_duration_seconds": duration,
        "results": {
            "consensus": consensus_results,
            "memory": memory_results,
            "neural": neural_results,
            "network": network_results
        }
    }))
}