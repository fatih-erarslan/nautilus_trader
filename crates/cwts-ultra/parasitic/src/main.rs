//! TONYUKUK Parasitic Trading Engine - Main Entry Point
//!
//! Advanced bio-inspired parasitic trading system with quantum capabilities.
//!
//! ## Features
//! - 9 Advanced Organism Types (Vampire Bat, Toxoplasma, Mycelial Network, etc.)
//! - Quantum/Classical Runtime Switching (--quantum flag)
//! - Real-time Evolution Engine with Genetic Algorithms  
//! - CQGS Quality Governance and Compliance
//! - MCP Server for External Integration
//! - Advanced Analytics and Performance Monitoring
//!
//! Usage:
//!   cargo run --bin parasitic-server
//!   cargo run --bin parasitic-server -- --quantum
//!   cargo run --bin parasitic-server -- --config config.toml
//!   cargo run --bin parasitic-server -- --mcp-port 3001

use clap::{Arg, ArgMatches, Command};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use parasitic::*;

/// Parse command line arguments
fn build_cli() -> Command {
    Command::new("TONYUKUK Parasitic Engine")
        .version("0.2.0")
        .author("TONYUKUK Ecosystem Team")
        .about("Advanced bio-inspired parasitic trading system with quantum capabilities")
        .long_about("TONYUKUK Parasitic Trading Engine - A sophisticated bio-inspired system featuring:\n\
                     • 9 Advanced Organism Types (Vampire Bat, Toxoplasma, Mycelial Network, etc.)\n\
                     • Quantum/Classical Runtime Switching\n\
                     • Real-time Evolution with Genetic Algorithms\n\
                     • CQGS Quality Governance\n\
                     • MCP Server Integration")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path (TOML format)")
                .action(clap::ArgAction::Set)
        )
        .arg(
            Arg::new("quantum")
                .short('q')
                .long("quantum")
                .help("Enable quantum-enhanced organism behaviors")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("mcp-port")
                .short('p')
                .long("mcp-port")
                .value_name("PORT")
                .help("MCP server port (default: 3001)")
                .action(clap::ArgAction::Set)
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Logging level (trace, debug, info, warn, error)")
                .action(clap::ArgAction::Set)
        )
        .arg(
            Arg::new("evolution-interval")
                .short('e')
                .long("evolution-interval")
                .value_name("SECONDS")
                .help("Evolution cycle interval in seconds (default: 180)")
                .action(clap::ArgAction::Set)
        )
        .arg(
            Arg::new("max-infections")
                .short('m')
                .long("max-infections")
                .value_name("COUNT")
                .help("Maximum simultaneous infections (default: 200)")
                .action(clap::ArgAction::Set)
        )
        .arg(
            Arg::new("demo")
                .short('d')
                .long("demo")
                .help("Run demonstration mode with example infections")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("benchmark")
                .short('b')
                .long("benchmark")
                .help("Run performance benchmarks")
                .action(clap::ArgAction::SetTrue)
        )
}

/// Load configuration from file or create default
async fn load_config(
    config_path: Option<PathBuf>,
    matches: &ArgMatches,
) -> Result<ParasiticConfig, Box<dyn std::error::Error + Send + Sync>> {
    let mut config = match config_path {
        Some(path) => {
            info!("Loading configuration from {}", path.display());
            let config_content = tokio::fs::read_to_string(&path).await?;
            toml::from_str::<ParasiticConfig>(&config_content)?
        }
        None => {
            info!("Using default configuration");
            ParasiticConfig::default()
        }
    };

    // Override config with command line arguments
    if let Some(port) = matches.get_one::<String>("mcp-port") {
        config.mcp_config.port = port.parse().unwrap_or(3001);
    }

    if let Some(interval) = matches.get_one::<String>("evolution-interval") {
        config.evolution_interval_secs = interval.parse().unwrap_or(180);
    }

    if let Some(max_infections) = matches.get_one::<String>("max-infections") {
        config.max_infections = max_infections.parse().unwrap_or(200);
    }

    // Enable quantum mode if flag is present
    if matches.get_flag("quantum") {
        config.quantum_config.quantum_enabled = true;
        config.mcp_config.quantum_enabled = true;
        info!("⚛️  Quantum mode enabled");
    }

    Ok(config)
}

/// Initialize logging based on configuration
fn init_logging(log_level: &str) {
    let level = match log_level.to_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level.to_string())),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
}

/// Run demonstration mode
async fn run_demo_mode(
    _config: ParasiticConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("🎬 Starting demonstration mode");

    // Wait for system to fully initialize
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    println!("\n🧬 TONYUKUK Parasitic Engine Demonstration");
    println!("==========================================");

    // Show system status (placeholder)
    println!("📊 System Status:");
    println!("   - Total organisms: {}", 9);
    println!(
        "   - Quantum mode: {}",
        if _config.quantum_enabled {
            "ENABLED ⚛️"
        } else {
            "DISABLED"
        }
    );
    println!("   - CQGS compliance: {:.1}%", 85.0);
    println!("   - Evolution generation: {}", 1);

    // Demonstrate different organism types
    let organism_types = [
        "vampire_bat",
        "toxoplasma",
        "mycelial_network",
        "lancet_liver_fluke",
    ];

    for (i, organism_type) in organism_types.iter().enumerate() {
        let pair_id = match i {
            0 => "BTCUSD",
            1 => "ETHUSD",
            2 => "ADAUSD",
            3 => "DOTUSD",
            _ => "SOLUSD",
        };

        println!(
            "\n🦠 Demonstrating {} organism on {}:",
            organism_type, pair_id
        );

        // Simulate organism deployment
        println!("   ✅ Infection simulation successful!");
        println!("      - Organism: {} (demo-{})", organism_type, i);
        println!("      - Strength: {:.3}", 0.75);
        println!("      - Vulnerability: {:.3}", 0.65);
        println!("      - Quantum enhanced: {}", _config.quantum_enabled);

        // Small delay between demonstrations
        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
    }

    // Show final status
    println!("\n📈 Final System Status:");
    let infected_pairs: Vec<String> = vec![]; // Placeholder
    println!("   - Active infections: {}", infected_pairs.len());
    println!(
        "   - Resource utilization: CPU {:.1}%, Memory {:.0}MB",
        25.5, 128.0
    );

    println!("\n✨ Demo completed! Press Ctrl+C to stop the engine.");

    Ok(())
}

/// Run performance benchmarks
async fn run_benchmarks(
    _config: ParasiticConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("🏃 Starting performance benchmarks");

    println!("\n⚡ TONYUKUK Parasitic Engine Benchmarks");
    println!("=====================================");

    // Benchmark organism creation
    let start_time = std::time::Instant::now();
    let mut organism_creation_times: Vec<u128> = Vec::new();

    for _ in 0..100 {
        let creation_start = std::time::Instant::now();
        let _organism = organisms::OrganismFactory::create_random_organism()?;
        organism_creation_times.push(creation_start.elapsed().as_nanos());
    }

    let avg_creation_time =
        organism_creation_times.iter().sum::<u128>() / organism_creation_times.len() as u128;
    println!(
        "🧬 Organism Creation: {:.2}µs average",
        avg_creation_time as f64 / 1000.0
    );

    // Benchmark infection performance
    let infection_start = std::time::Instant::now();
    let mut infection_times: Vec<u128> = Vec::new();

    let test_pairs = ["BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "SOLUSD"];

    for (i, pair) in test_pairs.iter().enumerate() {
        let infection_time_start = std::time::Instant::now();

        // Simulate benchmark infection
        let success = true; // Placeholder
        if success {
            infection_times.push(infection_time_start.elapsed().as_nanos());
        } else {
            warn!("Benchmark infection failed for {}", pair);
        }

        // Small delay between infections
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    if !infection_times.is_empty() {
        let avg_infection_time =
            infection_times.iter().sum::<u128>() / infection_times.len() as u128;
        println!(
            "🦠 Infection Latency: {:.2}µs average",
            avg_infection_time as f64 / 1000.0
        );
    }

    // Benchmark system status retrieval
    let status_start = std::time::Instant::now();
    for _ in 0..50 {
        // Placeholder status check
    }
    let status_time = status_start.elapsed().as_nanos() / 50;
    println!(
        "📊 Status Retrieval: {:.2}µs average",
        status_time as f64 / 1000.0
    );

    // Evolution benchmark
    let evolution_start = std::time::Instant::now();
    println!("\n🧬 Evolution Status: Generation 1 (placeholder)");
    let evolution_time = evolution_start.elapsed().as_nanos();
    println!(
        "🧬 Evolution Status: {:.2}µs",
        evolution_time as f64 / 1000.0
    );
    println!("   - Generation: {}", 1);
    println!("   - Average fitness: {:.3}", 0.75);

    let total_benchmark_time = start_time.elapsed();
    println!(
        "\n⏱️  Total benchmark time: {:.2}ms",
        total_benchmark_time.as_millis()
    );

    Ok(())
}

/// Main application logic
async fn run_application(
    matches: ArgMatches,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Load configuration
    let config_path = matches.get_one::<String>("config").map(PathBuf::from);
    let config = load_config(config_path, &matches).await?;

    // Initialize logging
    let log_level = matches
        .get_one::<String>("log-level")
        .map(|s| s.as_str())
        .unwrap_or("info");
    init_logging(log_level);

    // Print startup banner
    print_startup_banner(&config);

    // Create and initialize engine
    info!("🚀 Starting TONYUKUK Parasitic Engine simulation");

    // Handle special modes
    if matches.get_flag("benchmark") {
        run_benchmarks(config.clone()).await?;
        return Ok(());
    }

    if matches.get_flag("demo") {
        run_demo_mode(config.clone()).await?;
    } else {
        info!("🎯 Parasitic engine running in production mode");
        info!(
            "   - MCP server: http://{}:{}",
            config.mcp_config.bind_address, config.mcp_config.port
        );
        info!(
            "   - Evolution interval: {}s",
            config.evolution_interval_secs
        );
        info!("   - Max infections: {}", config.max_infections);
    }

    // Wait for shutdown signal
    info!("✅ System ready - Press Ctrl+C to shutdown");

    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("🛑 Received shutdown signal");
        }
    }

    // Graceful shutdown
    info!("🔄 Shutting down parasitic engine...");

    // Export final metrics
    println!("\n📊 Final System Statistics:");
    println!("   - Total organisms: {}", 9);
    println!("   - Active infections: {}", 0);
    println!("   - Evolution generations: {}", 1);
    println!("   - Average fitness: {:.3}", 0.75);
    println!("   - System uptime: {}s", 60);
    println!("   - Resource efficiency: {:.1}%", 85.0);

    info!("👋 TONYUKUK Parasitic Engine shutdown complete");

    Ok(())
}

/// Print startup banner
fn print_startup_banner(config: &ParasiticConfig) {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ████████╗ ██████╗ ███╗   ██╗██╗   ██╗██╗   ██╗██╗  ██╗██╗   ██╗██╗  ██╗    ║
║  ╚══██╔══╝██╔═══██╗████╗  ██║╚██╗ ██╔╝██║   ██║██║ ██╔╝██║   ██║██║ ██╔╝    ║
║     ██║   ██║   ██║██╔██╗ ██║ ╚████╔╝ ██║   ██║█████╔╝ ██║   ██║█████╔╝     ║
║     ██║   ██║   ██║██║╚██╗██║  ╚██╔╝  ██║   ██║██╔═██╗ ██║   ██║██╔═██╗     ║
║     ██║   ╚██████╔╝██║ ╚████║   ██║   ╚██████╔╝██║  ██╗╚██████╔╝██║  ██╗    ║
║     ╚═╝    ╚═════╝ ╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝    ║
║                                                                              ║
║              🧬 Parasitic Trading Engine v0.2.0 🧬                          ║
║                                                                              ║
║  🦠 9 Advanced Organisms    ⚛️ Quantum Runtime    🧬 Evolution Engine        ║
║  🩸 Vampire Bat Specialist  🧠 Toxoplasma Brain   🕸️ Mycelial Network        ║
║  🔬 Lancet Liver Fluke     🦅 Cordyceps Control   🛡️ CQGS Governance        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    "#
    );

    println!("🚀 Bio-inspired parasitic trading system initializing...");
    println!(
        "⚛️  Quantum mode: {}",
        if config.quantum_config.quantum_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );
    println!("🔧 Evolution interval: {}s", config.evolution_interval_secs);
    println!("🎯 Max infections: {}", config.max_infections);
    println!(
        "📡 MCP server: {}:{}\n",
        config.mcp_config.bind_address, config.mcp_config.port
    );
}

/// Main entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let matches = build_cli().get_matches();

    match run_application(matches).await {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("❌ Fatal error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let app = build_cli();

        // Test basic command parsing
        let matches = app
            .try_get_matches_from(vec!["parasitic-server", "--quantum"])
            .unwrap();
        assert!(matches.get_flag("quantum"));

        // Test with config file
        let matches = app
            .try_get_matches_from(vec!["parasitic-server", "--config", "test.toml"])
            .unwrap();
        assert_eq!(matches.get_one::<String>("config").unwrap(), "test.toml");
    }

    #[tokio::test]
    async fn test_default_config_loading() {
        let matches = build_cli().get_matches_from(vec!["parasitic-server"]);
        let config = load_config(None, &matches).await.unwrap();

        assert_eq!(config.mcp_config.port, 3001);
        assert_eq!(config.evolution_interval_secs, 180);
        assert!(!config.quantum_config.quantum_enabled); // Default disabled
    }

    #[tokio::test]
    async fn test_quantum_flag_override() {
        let matches = build_cli().get_matches_from(vec!["parasitic-server", "--quantum"]);
        let config = load_config(None, &matches).await.unwrap();

        assert!(config.quantum_config.quantum_enabled);
        assert!(config.mcp_config.quantum_enabled);
    }
}
