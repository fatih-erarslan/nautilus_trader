/// Standalone MCP server for parasitic trading system
/// This can be run independently of CWTS Ultra core

use anyhow::Result;
use tokio;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Import from parasitic library
use parasitic::mcp_server::ParasiticMCPServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("🐛 CWTS Ultra Parasitic MCP Server v0.1.0");
    info!("🧬 Initializing biomimetic trading organisms...");

    // Get port from environment or use default
    let port = std::env::var("PARASITIC_PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse::<u16>()
        .unwrap_or(3001);

    // Create and start MCP server
    let server = ParasiticMCPServer::new(port);
    
    info!("🚀 Starting MCP server on port {}", port);
    info!("🔌 WebSocket endpoint: ws://localhost:{}", port);
    info!("");
    info!("📊 Available MCP Resources:");
    info!("  - parasitic://organisms       - List parasitic organisms");
    info!("  - parasitic://pairs/infected  - Currently infected pairs");
    info!("  - parasitic://evolution/status - Evolution metrics");
    info!("");
    info!("🔧 Available MCP Tools:");
    info!("  - parasitic_select  - Select organism strategy");
    info!("  - parasitic_infect  - Infect trading pair");
    info!("  - parasitic_evolve  - Trigger evolution");
    info!("  - parasitic_analyze - Analyze vulnerability");
    info!("");
    info!("🦟 The parasites are ready to hunt...");

    // Run server
    match server.run().await {
        Ok(_) => {
            info!("✅ MCP server shutdown gracefully");
        }
        Err(e) => {
            error!("❌ MCP server error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}