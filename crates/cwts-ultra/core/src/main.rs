use cwts_ultra::mcp::{ServerConfig, TradingMCPServer};
use cwts_ultra::CWTSUltra;
use std::env;
use std::fs;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CWTS Ultra Trading System v2.0.0 with MCP Server");
    println!("Initializing high-performance trading system with Model Context Protocol...");

    // Initialize core trading system
    let _system = CWTSUltra::new();
    println!("✅ Core trading system initialized");

    // Load configuration from environment or config file
    let mcp_port = env::var("MCP_SERVER_PORT")
        .unwrap_or_else(|_| load_config_value("mcp_port").unwrap_or_else(|| "4000".to_string()));
    let bind_address = format!("127.0.0.1:{}", mcp_port);

    // Initialize MCP server with custom configuration
    let config = ServerConfig {
        bind_address: bind_address.parse().unwrap(),
        max_clients: 100,
        heartbeat_interval_ms: 30000,
        enable_compression: true,
        max_message_size: 1024 * 1024, // 1MB
    };

    let mcp_server = TradingMCPServer::new(Some(config)).await?;
    println!("✅ MCP Server initialized on {}", bind_address);

    // Display server capabilities
    println!("\n🚀 Server Features:");
    println!("  • WebSocket-based MCP protocol");
    println!("  • Real-time order book with lock-free operations");
    println!("  • Atomic order matching engine");
    println!("  • Live market data subscriptions");
    println!("  • Risk analysis and portfolio management");
    println!("  • 8+ trading tools available");
    println!("  • 7+ resource endpoints");
    println!("  • High-frequency order processing");

    println!("\n📊 Available Resources:");
    println!("  • trading://order_book/BTCUSD - Live order book");
    println!("  • trading://positions - Current positions");
    println!("  • trading://market_data/BTCUSD - Market data");
    println!("  • trading://trades/history - Trade history");
    println!("  • trading://account/summary - Account info");
    println!("  • trading://engine/stats - Engine metrics");
    println!("  • trading://risk/metrics - Risk analysis");

    println!("\n🛠 Available Tools:");
    println!("  • place_order - Place buy/sell orders");
    println!("  • cancel_order - Cancel existing orders");
    println!("  • modify_order - Modify order parameters");
    println!("  • get_positions - View current positions");
    println!("  • get_market_data - Real-time market data");
    println!("  • analyze_risk - Portfolio risk analysis");
    println!("  • get_order_status - Order status tracking");
    println!("  • calculate_profit_loss - P&L calculations");

    println!("\n🔄 Starting MCP server...");
    println!("Connect using WebSocket client at: ws://{}", bind_address);
    println!("Protocol: Model Context Protocol (MCP) 2024-11-05");
    println!("Press Ctrl+C to stop the server\n");

    // Start the MCP server (this will run indefinitely)
    mcp_server.start().await?;

    Ok(())
}

fn load_config_value(key: &str) -> Option<String> {
    // Try to load from config file
    let home = env::var("HOME").ok()?;
    let config_path = PathBuf::from(home).join(".local/cwts-ultra/config/production.toml");

    if config_path.exists() {
        let contents = fs::read_to_string(&config_path).ok()?;

        // Simple parsing for the mcp_port value
        for line in contents.lines() {
            if line.starts_with(&format!("{} =", key)) {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    return Some(parts[1].trim().trim_matches('"').to_string());
                }
            }
        }
    }

    None
}
