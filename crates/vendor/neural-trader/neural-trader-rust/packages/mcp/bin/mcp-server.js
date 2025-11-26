#!/usr/bin/env node
/**
 * Neural Trader MCP Server CLI
 * MCP 2025-11 Compliant Implementation
 *
 * Provides 99+ trading tools accessible to AI assistants via JSON-RPC
 */

const { McpServer } = require('../src/server.js');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const config = {
  transport: 'stdio',
  port: 3000,
  host: 'localhost',
  toolsDir: path.join(__dirname, '../tools'),
  enableRustBridge: true,
  stubMode: false,
  enableAuditLog: true,
};

for (let i = 0; i < args.length; i++) {
  const arg = args[i];

  if (arg === '--transport' || arg === '-t') {
    config.transport = args[++i];
  } else if (arg === '--port' || arg === '-p') {
    config.port = parseInt(args[++i], 10);
  } else if (arg === '--host' || arg === '-h') {
    config.host = args[++i];
  } else if (arg === '--stub') {
    config.stubMode = true;
  } else if (arg === '--no-rust') {
    config.enableRustBridge = false;
  } else if (arg === '--no-audit') {
    config.enableAuditLog = false;
  } else if (arg === '--help') {
    console.log(`
╭──────────────────────────────────────────────────────────────────────────────╮
│                     Neural Trader MCP Server v2.0.0                         │
│                        MCP 2025-11 Compliant                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Usage:
  npx neural-trader mcp [options]

Options:
  -t, --transport <type>    Transport type: stdio (default)
  -p, --port <number>       Port number for future HTTP transport (default: 3000)
  -h, --host <address>      Host address (default: localhost)
  --stub                   Run in stub mode (for testing without Rust binary)
  --no-rust                Disable Rust NAPI bridge completely
  --no-audit               Disable audit logging
  --help                   Show this help message

Environment Variables:
  NEURAL_TRADER_API_KEY     API key for broker authentication
  ALPACA_API_KEY           Alpaca Markets API key
  ALPACA_SECRET_KEY        Alpaca Markets secret key

Examples:
  # Start MCP server (stdio transport, loads Rust NAPI)
  npx neural-trader mcp

  # Start in stub mode (for testing)
  npx neural-trader mcp --stub

  # Start without Rust bridge
  npx neural-trader mcp --no-rust

  # Test with Claude Desktop
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "neural-trader": {
        "command": "npx",
        "args": ["neural-trader", "mcp"]
      }
    }
  }

Tools Available: 99+
  - Trading (23): list_strategies, execute_trade, backtest_strategy, etc.
  - Neural (7): neural_train, neural_forecast, neural_optimize, etc.
  - News (8): analyze_news, get_news_sentiment, control_news_collection, etc.
  - Sports Betting (13): get_sports_odds, find_arbitrage, kelly_criterion, etc.
  - Prediction Markets (5): get_markets, place_order, analyze_sentiment, etc.
  - Syndicates (15): create_syndicate, allocate_funds, distribute_profits, etc.
  - E2B Cloud (9): create_sandbox, run_agent, deploy_template, etc.
  - Fantasy (5): create_league, make_prediction, calculate_scores, etc.

Documentation: https://github.com/ruvnet/neural-trader
Specification: MCP 2025-11 (https://gist.github.com/ruvnet/284f199d0e0836c1b5185e30f819e052)
`);
    process.exit(0);
  }
}

// Start the server
async function main() {
  try {
    const server = new McpServer(config);
    await server.start();

    // Handle graceful shutdown
    const shutdown = async (signal) => {
      console.error('');
      console.error(`Received ${signal}, shutting down gracefully...`);
      await server.stop();
      process.exit(0);
    };

    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));

    // Keep process alive
    process.stdin.resume();

  } catch (error) {
    console.error('❌ Failed to start MCP server:', error.message);
    console.error('');
    console.error('Troubleshooting:');
    console.error('  1. Ensure Node.js 18+ is installed');
    console.error('  2. Check Python 3.9+ is available (for Python bridge)');
    console.error('  3. Verify tool schemas exist in /tools directory');
    console.error('');
    process.exit(1);
  }
}

main();
