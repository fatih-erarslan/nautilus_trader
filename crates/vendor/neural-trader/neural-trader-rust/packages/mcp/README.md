# @neural-trader/mcp

Model Context Protocol (MCP) server for Neural Trader providing 102+ AI-accessible trading tools including advanced syndicate management, neural forecasting, and real-time market analysis for AI assistants like Claude.

## Features

- 🤖 **102+ AI-Accessible Tools** - Comprehensive trading toolkit for AI assistants
- 🔌 **Multiple Transport Layers** - stdio, HTTP, and WebSocket support
- 👥 **15 Syndicate Management Tools** - Collaborative trading with Kelly Criterion optimization
- 🧠 **Neural Network Integration** - AI-powered forecasting and pattern recognition
- ⚡ **High-Performance Rust Core** - Optimized execution with SIMD acceleration
- 🔒 **Type-Safe JSON-RPC 2.0** - Built on [@neural-trader/mcp-protocol](https://www.npmjs.com/package/@neural-trader/mcp-protocol)
- 📊 **Real-Time Market Data** - Live quotes, news, and sentiment analysis
- 💰 **Risk Management** - VaR, CVaR, Kelly Criterion, and portfolio optimization
- 🎯 **Strategy Backtesting** - GPU-accelerated historical testing and optimization
- 🤝 **Claude Desktop Ready** - Zero-config integration with Claude Desktop

## Installation

```bash
npm install @neural-trader/mcp
```

Or install globally for CLI access:

```bash
npm install -g @neural-trader/mcp
```

## Quick Start

### Start MCP Server (stdio)

```bash
# Using npx (recommended for Claude Desktop)
npx @neural-trader/mcp

# Or if installed globally
neural-trader-mcp
```

### Programmatic Usage

```javascript
const { McpServer } = require('@neural-trader/mcp');

async function main() {
  // Create MCP server with stdio transport (default)
  const server = new McpServer({
    transport: 'stdio',
    enableCors: true,
    maxConnections: 100
  });

  // Start the server
  await server.start();
  console.log('MCP server running!');

  // List all available tools
  const tools = await server.listTools();
  console.log(`Available tools: ${tools.length}`);

  // Get syndicate-specific tools
  const syndicateTools = server.getSyndicateTools();
  console.log(`Syndicate tools: ${syndicateTools.length}`);
}

main();
```

### HTTP Server

```javascript
const { McpServer } = require('@neural-trader/mcp');

async function startHttpServer() {
  const server = new McpServer({
    transport: 'http',
    port: 8080,
    host: 'localhost',
    enableCors: true
  });

  await server.start();
  console.log('HTTP MCP server listening on http://localhost:8080');
}

startHttpServer();
```

### WebSocket Server

```javascript
const { McpServer } = require('@neural-trader/mcp');

async function startWebSocketServer() {
  const server = new McpServer({
    transport: 'websocket',
    port: 3000,
    host: '0.0.0.0',
    maxConnections: 50
  });

  await server.start();
  console.log('WebSocket MCP server listening on ws://0.0.0.0:3000');
}

startWebSocketServer();
```

## Core Concepts

### Model Context Protocol (MCP)

MCP by Anthropic enables AI assistants to interact with external tools and data sources through a standardized JSON-RPC 2.0 protocol. The Neural Trader MCP server exposes 102+ trading tools to AI assistants, enabling natural language trading operations.

**Key components:**
- **Tools** - Callable functions exposed to AI assistants
- **Transport** - Communication layer (stdio, HTTP, WebSocket)
- **Protocol** - JSON-RPC 2.0 request/response format
- **Handlers** - Tool implementation logic

### Tool Categories

#### Strategy Analysis Tools
- `list_strategies` - List all available trading strategies
- `get_strategy_info` - Get detailed strategy information
- `quick_analysis` - Quick market analysis for symbols
- `simulate_trade` - Simulate trading operations

#### Portfolio Management Tools
- `get_portfolio_status` - Current portfolio status and analytics
- `risk_analysis` - Comprehensive risk analysis with VaR/CVaR
- `correlation_analysis` - Multi-asset correlation analysis
- `portfolio_rebalance` - Calculate optimal rebalancing

#### Backtesting Tools
- `run_backtest` - Historical backtest with GPU acceleration
- `optimize_strategy` - Parameter optimization with grid search
- `performance_report` - Detailed performance analytics
- `run_benchmark` - Strategy performance benchmarks

#### Neural Network Tools
- `neural_forecast` - Generate AI predictions
- `neural_train` - Train forecasting models
- `neural_evaluate` - Evaluate model performance
- `neural_backtest` - Backtest neural strategies
- `neural_optimize` - Hyperparameter optimization

#### News & Sentiment Tools
- `analyze_news` - AI sentiment analysis
- `get_news_sentiment` - Real-time sentiment
- `fetch_filtered_news` - Advanced news filtering
- `get_news_trends` - Multi-timeframe trend analysis

#### Execution Tools
- `execute_trade` - Execute live trades
- `execute_multi_asset_trade` - Multi-asset execution
- `place_prediction_order` - Prediction market orders
- `calculate_expected_value` - EV calculations

#### Sports Betting Tools
- `get_sports_events` - Upcoming events with analysis
- `get_sports_odds` - Real-time odds from multiple bookmakers
- `find_sports_arbitrage` - Arbitrage opportunities
- `calculate_kelly_criterion` - Optimal bet sizing
- `execute_sports_bet` - Place sports bets

#### Syndicate Management Tools (15)
- `create_syndicate` - Create investment syndicate
- `add_member` - Add members with role-based permissions
- `get_syndicate_status` - Syndicate metrics and health
- `allocate_funds` - Kelly Criterion optimal allocation
- `distribute_profits` - Multi-model profit distribution
- `process_withdrawal` - Member withdrawal processing
- `get_member_performance` - Detailed member metrics
- `create_vote` - Create governance proposals
- `cast_vote` - Cast weighted votes
- `simulate_allocation` - Monte Carlo portfolio simulation
- `compare_strategies` - Backtest allocation strategies
- `get_allocation_limits` - View limits and available capital
- `get_profit_history` - Historical distributions
- `calculate_tax_liability` - Jurisdiction-specific taxes
- `update_allocation_strategy` - Modify bankroll rules

### Transport Modes

#### stdio (Default)
Best for Claude Desktop and local AI assistant integration:
```bash
neural-trader-mcp
```

#### HTTP
For web-based integrations and REST APIs:
```bash
neural-trader-mcp --transport http --port 8080
```

#### WebSocket
For real-time bidirectional communication:
```bash
neural-trader-mcp --transport websocket --port 3000
```

### Syndicate Architecture

Syndicates enable collaborative trading with:
- **Kelly Criterion** - Mathematically optimal bet sizing
- **Multi-Model Distribution** - Proportional, performance, hybrid, equal
- **Governance System** - Weighted voting with quorum requirements
- **Risk Controls** - Automatic caps and stop-loss protection
- **Performance Tracking** - ROI, alpha, Sharpe ratio, win rate

## API Reference

### McpServer

Main server class for hosting MCP tools.

```typescript
class McpServer {
  constructor(config?: McpServerConfig);
  start(): Promise<void>;
  stop(): Promise<void>;
  registerTool(name: string, handler: (params: any) => Promise<any>): void;
  listTools(): Promise<string[]>;
  getSyndicateTools(): SyndicateTool[];
  executeSyndicateTool(toolName: string, params: any): Promise<any>;
}
```

**Constructor Parameters:**
```typescript
interface McpServerConfig {
  transport?: 'stdio' | 'http' | 'websocket';  // Default: 'stdio'
  port?: number;                                // Default: 3000
  host?: string;                                // Default: 'localhost'
  enableCors?: boolean;                         // Default: true
  maxConnections?: number;                      // Default: 100
}
```

**Example:**
```javascript
const server = new McpServer({
  transport: 'stdio',
  enableCors: true,
  maxConnections: 100
});
```

### start()

Start the MCP server and begin accepting connections.

```typescript
start(): Promise<void>
```

**Example:**
```javascript
await server.start();
console.log('MCP server is running');
```

### stop()

Stop the MCP server and close all connections.

```typescript
stop(): Promise<void>
```

**Example:**
```javascript
await server.stop();
console.log('MCP server stopped');
```

### registerTool()

Register a custom tool with the MCP server.

```typescript
registerTool(name: string, handler: (params: any) => Promise<any>): void
```

**Parameters:**
- `name` - Unique tool identifier
- `handler` - Async function that implements tool logic

**Example:**
```javascript
server.registerTool('custom_indicator', async (params) => {
  const { symbol, period } = params;
  // Calculate custom indicator
  return {
    symbol,
    value: 42.5,
    timestamp: new Date().toISOString()
  };
});
```

### listTools()

List all available tools (core + syndicate + custom).

```typescript
listTools(): Promise<string[]>
```

**Example:**
```javascript
const tools = await server.listTools();
console.log(`Total tools: ${tools.length}`);
// Output: Total tools: 102
```

### getSyndicateTools()

Get all syndicate management tool definitions.

```typescript
getSyndicateTools(): SyndicateTool[]
```

**Returns:**
```typescript
interface SyndicateTool {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
  handler: (params: any) => Promise<any>;
}
```

**Example:**
```javascript
const syndicateTools = server.getSyndicateTools();
syndicateTools.forEach(tool => {
  console.log(`${tool.name}: ${tool.description}`);
});
```

### executeSyndicateTool()

Execute a syndicate management tool directly.

```typescript
executeSyndicateTool(toolName: string, params: any): Promise<any>
```

**Example:**
```javascript
const result = await server.executeSyndicateTool('create_syndicate', {
  syndicate_id: 'alpha-001',
  name: 'Alpha Trading Syndicate',
  total_bankroll: 100000
});
console.log(result);
```

### startServer()

Helper function to create and start an MCP server in one call.

```typescript
function startServer(config?: McpServerConfig): Promise<McpServer>
```

**Example:**
```javascript
const { startServer } = require('@neural-trader/mcp');

const server = await startServer({
  transport: 'http',
  port: 8080
});
```

## Tutorials

### Tutorial 1: Building an AI Trading Assistant

Create an AI-powered trading assistant using Claude Desktop.

**Step 1: Configure Claude Desktop**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Step 2: Restart Claude Desktop**

Restart Claude Desktop to load the MCP server.

**Step 3: Use Natural Language**

Now you can ask Claude things like:

> "What trading strategies are available?"

> "Run a backtest of the momentum strategy on AAPL from 2023-01-01 to 2023-12-31"

> "Analyze the current risk of my portfolio"

> "Create a syndicate called 'Alpha Fund' with $100k bankroll"

**Step 4: Verify Tools Loaded**

Ask Claude:
> "What Neural Trader tools do you have access to?"

Claude will list all 102+ tools available.

### Tutorial 2: HTTP API Integration

Build a web service that exposes MCP tools via HTTP.

**Step 1: Create HTTP Server**

```javascript
// server.js
const { McpServer } = require('@neural-trader/mcp');
const express = require('express');

async function main() {
  // Create MCP server
  const mcpServer = new McpServer({
    transport: 'http',
    port: 8080,
    enableCors: true
  });

  await mcpServer.start();

  // Create Express wrapper for custom endpoints
  const app = express();
  app.use(express.json());

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', tools: 102 });
  });

  // List all tools endpoint
  app.get('/tools', async (req, res) => {
    const tools = await mcpServer.listTools();
    res.json({ tools, count: tools.length });
  });

  // Execute syndicate tool endpoint
  app.post('/syndicate/:toolName', async (req, res) => {
    try {
      const { toolName } = req.params;
      const result = await mcpServer.executeSyndicateTool(
        toolName,
        req.body
      );
      res.json(result);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  app.listen(3000, () => {
    console.log('API server listening on http://localhost:3000');
  });
}

main();
```

**Step 2: Test Endpoints**

```bash
# Health check
curl http://localhost:3000/health

# List tools
curl http://localhost:3000/tools

# Create syndicate
curl -X POST http://localhost:3000/syndicate/create_syndicate \
  -H "Content-Type: application/json" \
  -d '{
    "syndicate_id": "web-fund-001",
    "name": "Web Trading Fund",
    "total_bankroll": 50000
  }'
```

### Tutorial 3: Syndicate Management Workflow

Complete workflow for creating and managing a trading syndicate.

**Step 1: Create Syndicate**

```javascript
const { McpServer } = require('@neural-trader/mcp');

async function createTradingSyndicate() {
  const server = new McpServer({ transport: 'stdio' });
  await server.start();

  // Create syndicate
  const syndicate = await server.executeSyndicateTool('create_syndicate', {
    syndicate_id: 'pro-traders-001',
    name: 'Pro Traders Syndicate',
    description: 'Professional sports betting syndicate',
    total_bankroll: 100000,
    max_members: 25,
    distribution_model: 'hybrid'
  });

  console.log('Syndicate created:', syndicate);
  return server;
}
```

**Step 2: Add Members**

```javascript
async function addMembers(server) {
  // Add senior analyst
  const member1 = await server.executeSyndicateTool('add_member', {
    syndicate_id: 'pro-traders-001',
    member_id: 'member_001',
    name: 'Alice Johnson',
    email: 'alice@example.com',
    role: 'senior_analyst',
    initial_contribution: 25000
  });

  // Add junior analyst
  const member2 = await server.executeSyndicateTool('add_member', {
    syndicate_id: 'pro-traders-001',
    member_id: 'member_002',
    name: 'Bob Smith',
    email: 'bob@example.com',
    role: 'junior_analyst',
    initial_contribution: 15000
  });

  console.log('Members added:', member1, member2);
}
```

**Step 3: Allocate Funds (Kelly Criterion)**

```javascript
async function allocateFunds(server) {
  // Find betting opportunity
  const opportunity = {
    sport: 'NFL',
    event: 'Chiefs vs Eagles',
    bet_type: 'moneyline',
    selection: 'Chiefs',
    odds: 2.15,
    probability: 0.52,    // 52% estimated win probability
    edge: 0.045,          // 4.5% edge
    confidence: 0.85      // 85% confidence in analysis
  };

  // Calculate optimal bet size using Kelly Criterion
  const allocation = await server.executeSyndicateTool('allocate_funds', {
    syndicate_id: 'pro-traders-001',
    opportunity,
    strategy: 'kelly_criterion',
    kelly_fraction: 0.25  // Use fractional Kelly for safety
  });

  console.log('Optimal allocation:', allocation);
  // Expected output:
  // {
  //   recommended_stake: 1125,  // $1,125
  //   kelly_percentage: 0.0225, // 2.25% of bankroll
  //   adjusted_stake: 1125,     // After 0.25 fractional adjustment
  //   expected_value: 50.63,
  //   max_loss: -1125,
  //   within_limits: true
  // }
}
```

**Step 4: Simulate Portfolio**

```javascript
async function simulatePortfolio(server) {
  const opportunities = [
    {
      id: 'bet1',
      sport: 'NFL',
      odds: 2.1,
      probability: 0.53,
      edge: 0.05
    },
    {
      id: 'bet2',
      sport: 'NBA',
      odds: 1.9,
      probability: 0.58,
      edge: 0.06
    },
    {
      id: 'bet3',
      sport: 'MLB',
      odds: 2.3,
      probability: 0.48,
      edge: 0.04
    }
  ];

  const simulation = await server.executeSyndicateTool('simulate_allocation', {
    syndicate_id: 'pro-traders-001',
    opportunities,
    strategies: ['kelly_criterion', 'fractional_kelly'],
    monte_carlo_simulations: 10000
  });

  console.log('Portfolio simulation:', simulation);
}
```

**Step 5: Distribute Profits**

```javascript
async function distributeProfits(server) {
  const distribution = await server.executeSyndicateTool('distribute_profits', {
    syndicate_id: 'pro-traders-001',
    total_profit: 50000,
    distribution_model: 'hybrid',        // 70% capital, 30% performance
    operational_reserve_pct: 0.05,      // 5% to reserve
    authorized_by: 'member_001'
  });

  console.log('Profit distribution:', distribution);
  // Expected output:
  // {
  //   total_profit: 50000,
  //   operational_reserve: 2500,
  //   distributable_profit: 47500,
  //   distributions: [
  //     { member_id: 'member_001', amount: 28500, ... },
  //     { member_id: 'member_002', amount: 19000, ... }
  //   ]
  // }
}
```

**Step 6: Create Governance Vote**

```javascript
async function createGovernanceVote(server) {
  const vote = await server.executeSyndicateTool('create_vote', {
    syndicate_id: 'pro-traders-001',
    vote_id: 'vote_001',
    proposal_type: 'strategy_change',
    proposal_details: {
      title: 'Increase max bet size to 7%',
      description: 'Increase maximum single bet from 5% to 7% of bankroll',
      changes: {
        max_single_bet: 0.07
      }
    },
    proposed_by: 'member_001',
    voting_period_hours: 48
  });

  console.log('Vote created:', vote);
}
```

**Step 7: Complete Workflow**

```javascript
async function completeWorkflow() {
  const server = await createTradingSyndicate();
  await addMembers(server);
  await allocateFunds(server);
  await simulatePortfolio(server);
  await distributeProfits(server);
  await createGovernanceVote(server);
  await server.stop();
}

completeWorkflow();
```

### Tutorial 4: Custom Tool Registration

Extend the MCP server with custom trading tools.

**Step 1: Create Custom Indicator Tool**

```javascript
const { McpServer } = require('@neural-trader/mcp');

async function setupCustomTools() {
  const server = new McpServer({ transport: 'stdio' });

  // Register custom RSI calculator
  server.registerTool('calculate_rsi', async (params) => {
    const { symbol, period = 14, data } = params;

    // Simple RSI calculation
    let gains = 0;
    let losses = 0;

    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));

    return {
      symbol,
      rsi,
      period,
      signal: rsi > 70 ? 'OVERBOUGHT' : rsi < 30 ? 'OVERSOLD' : 'NEUTRAL',
      timestamp: new Date().toISOString()
    };
  });

  // Register custom divergence detector
  server.registerTool('detect_divergence', async (params) => {
    const { symbol, priceData, indicatorData } = params;

    const priceTrend = priceData[priceData.length - 1] > priceData[0]
      ? 'UP' : 'DOWN';
    const indicatorTrend = indicatorData[indicatorData.length - 1] >
      indicatorData[0] ? 'UP' : 'DOWN';

    const divergence = priceTrend !== indicatorTrend;

    return {
      symbol,
      divergence,
      type: divergence
        ? (priceTrend === 'UP' ? 'BEARISH' : 'BULLISH')
        : 'NONE',
      confidence: divergence ? 0.75 : 0,
      timestamp: new Date().toISOString()
    };
  });

  await server.start();
  return server;
}
```

**Step 2: Use Custom Tools**

```javascript
async function useCustomTools() {
  const server = await setupCustomTools();

  // Calculate RSI
  const rsiResult = await server.executeSyndicateTool('calculate_rsi', {
    symbol: 'AAPL',
    period: 14,
    data: [150, 152, 151, 153, 155, 154, 156, 158, 157, 159, 161, 160, 162, 164]
  });
  console.log('RSI:', rsiResult);

  // Detect divergence
  const divergenceResult = await server.executeSyndicateTool('detect_divergence', {
    symbol: 'AAPL',
    priceData: [150, 152, 154, 156, 158],
    indicatorData: [65, 63, 61, 59, 57]
  });
  console.log('Divergence:', divergenceResult);

  await server.stop();
}

useCustomTools();
```

### Tutorial 5: Real-Time WebSocket Streaming

Build a real-time trading dashboard with WebSocket.

**Step 1: Create WebSocket Server**

```javascript
// ws-server.js
const { McpServer } = require('@neural-trader/mcp');
const WebSocket = require('ws');

async function startRealtimeServer() {
  // Create MCP server with WebSocket transport
  const mcpServer = new McpServer({
    transport: 'websocket',
    port: 3000,
    host: '0.0.0.0'
  });

  await mcpServer.start();

  // Create WebSocket server for real-time updates
  const wss = new WebSocket.Server({ port: 3001 });

  wss.on('connection', (ws) => {
    console.log('Client connected');

    // Send portfolio status every 5 seconds
    const interval = setInterval(async () => {
      try {
        const status = await mcpServer.executeSyndicateTool(
          'get_portfolio_status',
          { include_analytics: true }
        );
        ws.send(JSON.stringify({
          type: 'portfolio_update',
          data: status
        }));
      } catch (error) {
        console.error('Error:', error);
      }
    }, 5000);

    ws.on('message', async (message) => {
      const request = JSON.parse(message);

      if (request.tool === 'get_syndicate_status') {
        const result = await mcpServer.executeSyndicateTool(
          'get_syndicate_status',
          request.params
        );
        ws.send(JSON.stringify({
          type: 'syndicate_status',
          data: result
        }));
      }
    });

    ws.on('close', () => {
      clearInterval(interval);
      console.log('Client disconnected');
    });
  });

  console.log('WebSocket server listening on ws://0.0.0.0:3001');
}

startRealtimeServer();
```

**Step 2: Create Client**

```javascript
// ws-client.js
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3001');

ws.on('open', () => {
  console.log('Connected to trading server');

  // Request syndicate status
  ws.send(JSON.stringify({
    tool: 'get_syndicate_status',
    params: {
      syndicate_id: 'pro-traders-001'
    }
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);

  if (message.type === 'portfolio_update') {
    console.log('Portfolio Update:', message.data);
  } else if (message.type === 'syndicate_status') {
    console.log('Syndicate Status:', message.data);
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## Integration Examples

### With @neural-trader/core

Use core types for type-safe MCP tool parameters.

```javascript
const { McpServer } = require('@neural-trader/mcp');
const core = require('@neural-trader/core');

async function typeSafeBacktest() {
  const server = new McpServer({ transport: 'stdio' });
  await server.start();

  // Create type-safe backtest config
  const config = {
    strategy: 'momentum',
    symbol: 'AAPL',
    start_date: '2023-01-01',
    end_date: '2023-12-31',
    initial_capital: 100000,
    commission: 0.001
  };

  // Validate using core types
  if (!config.strategy || !config.symbol) {
    throw new Error('Invalid backtest config');
  }

  // Execute backtest via MCP
  const result = await server.executeSyndicateTool('run_backtest', config);

  console.log('Backtest Result:', result);
  await server.stop();
}

typeSafeBacktest();
```

### With @neural-trader/mcp-protocol

Use protocol types for low-level MCP communication.

```javascript
const { McpServer } = require('@neural-trader/mcp');
const { createRequest, createSuccessResponse } = require('@neural-trader/mcp-protocol');

async function customProtocolHandler() {
  const server = new McpServer({ transport: 'http', port: 8080 });

  // Register custom handler using protocol types
  server.registerTool('custom_analysis', async (params) => {
    // Create internal request
    const request = createRequest('analyze_market', params, 'internal-001');

    // Process request
    const analysis = {
      symbol: params.symbol,
      trend: 'BULLISH',
      confidence: 0.85
    };

    // Create protocol-compliant response
    const response = createSuccessResponse(analysis, 'internal-001');

    return response.result;
  });

  await server.start();
  console.log('Server with custom protocol handler running');
}

customProtocolHandler();
```

### Multi-Package Integration

Combine all packages for a complete trading system.

```javascript
const { McpServer } = require('@neural-trader/mcp');
const { createRequest } = require('@neural-trader/mcp-protocol');
const core = require('@neural-trader/core');

async function completeTradingSystem() {
  // Initialize MCP server
  const server = new McpServer({
    transport: 'stdio',
    enableCors: true
  });

  await server.start();

  // 1. Analyze market using core types
  const analysisParams = {
    symbol: 'AAPL',
    use_gpu: true
  };

  const analysis = await server.executeSyndicateTool(
    'quick_analysis',
    analysisParams
  );

  // 2. Generate neural forecast
  const forecastParams = {
    symbol: 'AAPL',
    horizon: 5,
    confidence_level: 0.95,
    use_gpu: true
  };

  const forecast = await server.executeSyndicateTool(
    'neural_forecast',
    forecastParams
  );

  // 3. Calculate risk metrics
  const riskParams = {
    portfolio: [
      { symbol: 'AAPL', weight: 0.4 },
      { symbol: 'GOOGL', weight: 0.3 },
      { symbol: 'MSFT', weight: 0.3 }
    ],
    use_gpu: true
  };

  const risk = await server.executeSyndicateTool(
    'risk_analysis',
    riskParams
  );

  // 4. Execute trade if conditions met
  if (forecast.prediction > analysis.current_price &&
      risk.portfolio_var < 0.05) {
    const tradeParams = {
      strategy: 'neural_momentum',
      symbol: 'AAPL',
      action: 'buy',
      quantity: 100,
      order_type: 'market'
    };

    const trade = await server.executeSyndicateTool(
      'execute_trade',
      tradeParams
    );

    console.log('Trade executed:', trade);
  }

  await server.stop();
}

completeTradingSystem();
```

## Configuration

### Environment Variables

```bash
# API keys
NEURAL_TRADER_API_KEY=your_api_key_here

# Server configuration
MCP_TRANSPORT=stdio
MCP_PORT=3000
MCP_HOST=localhost

# Feature flags
ENABLE_GPU=true
ENABLE_SYNDICATE_TOOLS=true
MAX_CONNECTIONS=100

# Logging
LOG_LEVEL=info
LOG_FILE=/var/log/neural-trader-mcp.log
```

### CLI Options

```bash
neural-trader-mcp [options]

Options:
  -t, --transport <type>    Transport: stdio, http, websocket (default: stdio)
  -p, --port <number>       Port for HTTP/WebSocket (default: 3000)
  -h, --host <address>      Host address (default: localhost)
  --cors                    Enable CORS for HTTP transport (default: true)
  --max-connections <num>   Maximum concurrent connections (default: 100)
  --help                    Show help message
  --version                 Show version number
```

**Examples:**

```bash
# Start with stdio (default)
neural-trader-mcp

# Start HTTP server on port 8080
neural-trader-mcp --transport http --port 8080

# Start WebSocket server with custom host
neural-trader-mcp --transport websocket --host 0.0.0.0 --port 3000

# Limit connections
neural-trader-mcp --max-connections 50
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

Or with custom options:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": [
        "@neural-trader/mcp",
        "--transport", "stdio",
        "--max-connections", "50"
      ],
      "env": {
        "NEURAL_TRADER_API_KEY": "your_key_here",
        "ENABLE_GPU": "true"
      }
    }
  }
}
```

## Performance Tips

### 1. Use Rust Implementation for Production

For production deployments, use the optimized Rust implementation:

```bash
cd neural-trader-rust
cargo run --release --bin mcp-server
```

Benefits:
- 10-100x better performance
- Lower memory usage
- Built-in SIMD acceleration
- Advanced async I/O

### 2. Enable GPU Acceleration

Enable GPU acceleration for neural network and risk calculations:

```javascript
const server = new McpServer({
  transport: 'http',
  port: 8080
});

// Tools automatically use GPU when available
await server.executeSyndicateTool('neural_forecast', {
  symbol: 'AAPL',
  horizon: 5,
  use_gpu: true  // Enable GPU acceleration
});
```

### 3. Connection Pooling

Limit concurrent connections to prevent resource exhaustion:

```javascript
const server = new McpServer({
  transport: 'http',
  maxConnections: 50  // Limit to 50 concurrent connections
});
```

### 4. Batch Operations

Use batch operations to reduce overhead:

```javascript
// Instead of multiple single calls
const symbols = ['AAPL', 'GOOGL', 'MSFT'];
const results = await Promise.all(
  symbols.map(symbol =>
    server.executeSyndicateTool('quick_analysis', { symbol })
  )
);
```

### 5. WebSocket for Real-Time

Use WebSocket transport for real-time updates:

```javascript
const server = new McpServer({
  transport: 'websocket',
  port: 3000
});
// Lower latency for frequent updates
```

## Troubleshooting

### Server Won't Start

**Problem:** Server fails to start with port error.

**Solution:** Check if port is already in use:

```bash
# Check port usage
lsof -i :3000

# Kill existing process
kill -9 <PID>

# Or use a different port
neural-trader-mcp --port 8080
```

### Claude Desktop Can't Find Server

**Problem:** Claude Desktop doesn't show Neural Trader tools.

**Solution:** Verify configuration:

```bash
# Check config file exists
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Test server manually
npx @neural-trader/mcp

# Restart Claude Desktop
```

### Tools Return Errors

**Problem:** MCP tools return "Method not found" errors.

**Solution:** Verify tool name and check available tools:

```javascript
const server = new McpServer({ transport: 'stdio' });
await server.start();

const tools = await server.listTools();
console.log('Available tools:', tools);

// Use exact tool name from list
```

### WebSocket Connection Timeout

**Problem:** WebSocket connections timeout or disconnect.

**Solution:** Increase timeout and add keep-alive:

```javascript
const server = new McpServer({
  transport: 'websocket',
  port: 3000,
  maxConnections: 100
});

// Client-side keep-alive
const ws = new WebSocket('ws://localhost:3000');
setInterval(() => {
  ws.ping();
}, 30000);
```

### Syndicate Tool Errors

**Problem:** Syndicate tools fail with validation errors.

**Solution:** Check parameter schema:

```javascript
// Get tool schema
const syndicateTools = server.getSyndicateTools();
const createSyndicate = syndicateTools.find(
  t => t.name === 'create_syndicate'
);
console.log('Input schema:', createSyndicate.inputSchema);

// Use correct parameters
await server.executeSyndicateTool('create_syndicate', {
  syndicate_id: 'fund-001',      // Required
  name: 'Trading Fund',          // Required
  total_bankroll: 100000         // Required
});
```

## Related Packages

### Core Dependencies

- [@neural-trader/core](https://www.npmjs.com/package/@neural-trader/core) - Zero-dependency TypeScript types
- [@neural-trader/mcp-protocol](https://www.npmjs.com/package/@neural-trader/mcp-protocol) - JSON-RPC 2.0 protocol types

### Optional Integrations

- [@neural-trader/backtesting](https://www.npmjs.com/package/@neural-trader/backtesting) - Strategy backtesting engine
- [@neural-trader/neural](https://www.npmjs.com/package/@neural-trader/neural) - Neural network models
- [@neural-trader/risk](https://www.npmjs.com/package/@neural-trader/risk) - Risk management tools
- [@neural-trader/data](https://www.npmjs.com/package/@neural-trader/data) - Market data providers

### Recommended Combinations

**For AI Trading:**
```bash
npm install @neural-trader/mcp @neural-trader/core @neural-trader/neural
```

**For Syndicate Management:**
```bash
npm install @neural-trader/mcp @neural-trader/core @neural-trader/risk
```

**For Complete Platform:**
```bash
npm install neural-trader
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/ruvnet/neural-trader/blob/main/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust/packages/mcp

# Install dependencies
npm install

# Run tests
npm test

# Start development server
npm start
```

### Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- server.test.js

# Run with coverage
npm test -- --coverage
```

## License

MIT OR Apache-2.0

## Support

- **Documentation:** [https://github.com/ruvnet/neural-trader](https://github.com/ruvnet/neural-trader)
- **Issues:** [https://github.com/ruvnet/neural-trader/issues](https://github.com/ruvnet/neural-trader/issues)
- **Discord:** [https://discord.gg/neural-trader](https://discord.gg/neural-trader)
- **Twitter:** [@neural_trader](https://twitter.com/neural_trader)

---

Built with ❤️ by the Neural Trader Team
