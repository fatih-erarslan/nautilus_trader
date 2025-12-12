//! Neural Trader MCP Server Binary
//!
//! Model Context Protocol server for Neural Trader algorithmic trading platform.
//! Supports STDIO transport for integration with LLM tools.

use neural_trader_mcp::transport::stdio::StdioTransport;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (to stderr to keep stdout clean for MCP)
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive("neural_trader_mcp=info".parse()?))
        .init();

    tracing::info!("Neural Trader MCP Server v{}", neural_trader_mcp::VERSION);
    tracing::info!("Starting STDIO transport...");

    // Run the STDIO transport
    let transport = StdioTransport::new();
    transport.run().await?;

    Ok(())
}
