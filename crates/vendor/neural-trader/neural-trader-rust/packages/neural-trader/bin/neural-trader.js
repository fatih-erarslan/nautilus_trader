#!/usr/bin/env node

/**
 * Neural Trader CLI
 * Complete command-line interface matching all README examples
 * Supports both subcommand and global option patterns
 */

const { Command } = require('commander');
const path = require('path');
const fs = require('fs');

const program = new Command();

// Get package version
const packageJson = require('../package.json');

program
  .name('neural-trader')
  .version(packageJson.version)
  .description('Complete AI-powered algorithmic trading platform');

// Global options that apply to all commands AND can trigger auto-routing
program
  .option('--broker <type>', 'Broker to use (alpaca, ib, binance, coinbase)')
  .option('--symbol <ticker>', 'Stock/crypto symbol to trade')
  .option('--symbols <list>', 'Comma-separated list of symbols')
  .option('--backtest', 'Run in backtest mode')
  .option('--live', 'Run live trading')
  .option('--paper', 'Use paper trading mode')
  .option('--verbose', 'Verbose output')
  .option('--config <path>', 'Path to config file')

  // Auto-routing options (trigger specific handlers when no subcommand given)
  .option('--strategy <type>', 'Strategy type (momentum, mean-reversion, pairs, arbitrage, etc)')
  .option('--model <type>', 'Neural model (lstm, gru, transformer, nbeats, deepar, tcn)')
  .option('--swarm <topology>', 'Swarm topology (hierarchical, mesh, ring, star, byzantine, adaptive, enabled)')

  // Strategy-specific options
  .option('--position-sizing <method>', 'Position sizing method (kelly, fixed, percent)')
  .option('--optimize', 'Optimize strategy parameters')
  .option('--start <date>', 'Start date for backtest (YYYY-MM-DD)')
  .option('--end <date>', 'End date for backtest (YYYY-MM-DD)')
  .option('--capital <amount>', 'Starting capital')
  .option('--max-drawdown <percent>', 'Maximum drawdown limit (0-1)')
  .option('--risk-tolerance <level>', 'Risk tolerance (0-1)')
  .option('--rebalance <frequency>', 'Rebalancing frequency (daily, weekly, monthly, quarterly)')
  .option('--allocation <ratio>', 'Asset allocation (e.g., 60/40)')
  .option('--tax-loss-harvest', 'Enable tax loss harvesting')

  // Neural-specific options
  .option('--models <list>', 'Comma-separated model list for comparison')
  .option('--train', 'Train a new model')
  .option('--predict', 'Generate predictions')
  .option('--confidence <level>', 'Confidence level (0-1)', '0.95')
  .option('--horizon <hours>', 'Prediction horizon in hours', '24')
  .option('--epochs <count>', 'Training epochs', '100')
  .option('--batch-size <size>', 'Batch size', '32')

  // Swarm-specific options
  .option('--agents <count>', 'Number of agents', '5')
  .option('--e2b', 'Deploy to E2B sandboxes')
  .option('--consensus-threshold <count>', 'Byzantine consensus threshold')
  .option('--sandboxes <count>', 'Number of E2B sandboxes')
  .option('--federated-learning', 'Enable federated learning')
  .option('--wasm', 'Enable WASM acceleration')
  .option('--topology-switching', 'Enable adaptive topology switching')
  .option('--knowledge-sharing', 'Enable knowledge sharing')
  .option('--persistent-memory', 'Enable persistent memory')
  .option('--self-healing', 'Enable self-healing')
  .option('--health-check-interval <time>', 'Health check interval (e.g., 10s)')
  .option('--swarm-optimize', 'Optimize swarm topology')
  .option('--topologies <list>', 'Topologies to compare')

  // Risk-specific options
  .option('--var', 'Calculate Value at Risk')
  .option('--cvar', 'Calculate Conditional VaR')
  .option('--monte-carlo', 'Run Monte Carlo simulation')
  .option('--scenarios <count>', 'Number of Monte Carlo scenarios', '10000')
  .option('--var-confidence <level>', 'VaR confidence level', '0.95')
  .option('--stress-test', 'Run stress testing')
  .option('--portfolio <path>', 'Portfolio JSON file')

  // Monitoring options
  .option('--metrics <list>', 'Metrics to monitor (sharpe,drawdown,winrate)')
  .option('--report', 'Generate performance report')
  .option('--period <timeframe>', 'Report period (7d, 30d, 90d)', '30d')
  .option('--charts', 'Include charts in report')
  .option('--alerts', 'Set up alerts')
  .option('--max-position <percent>', 'Alert on max position size')
  .option('--system-health', 'Check system health')
  .option('--restart-failed', 'Restart failed components')
  .option('--system-status', 'Show system status')
  .option('--memory', 'Show memory usage')
  .option('--neural-patterns', 'Show neural pattern status')
  .option('--export-audit', 'Export audit trail')

  // AgentDB options
  .option('--embeddings', 'Store/query embeddings')
  .option('--learning', 'Enable learning from embeddings')
  .option('--store <data>', 'Store data as embeddings')
  .option('--query <text>', 'Query similar patterns')
  .option('--limit <count>', 'Result limit', '20')

  // ReasoningBank options
  .option('--meta-learning', 'Enable meta-learning')
  .option('--track-trajectory', 'Track decision trajectories')
  .option('--verdict-system', 'Enable verdict judgment')
  .option('--memory-distillation', 'Distill memory patterns')

  // Sublinear options
  .option('--temporal-advantage', 'Use temporal advantage solving')
  .option('--predictive', 'Predictive pre-solving')
  .option('--matrix-size <n>', 'Matrix size', '1000')

  // Lean-agentic options
  .option('--micro-agents', 'Deploy micro-agents')
  .option('--edge-deployment', 'Deploy to edge/browser')
  .option('--agent-count <n>', 'Number of agents', '100')

  // Sports betting options
  .option('--sport <type>', 'Sport (nfl, nba, mlb, etc)')
  .option('--bookmakers <count>', 'Number of bookmakers', '5')
  .option('--kelly <fraction>', 'Kelly Criterion fraction', '0.25')
  .option('--syndicate', 'Create betting syndicate')
  .option('--members <count>', 'Syndicate members')
  .option('--arbitrage', 'Find arbitrage opportunities')

  // Prediction markets options
  .option('--platform <name>', 'Platform (polymarket, predictit)')
  .option('--market-depth-analysis', 'Analyze market depth')

  // Analysis options
  .option('--indicators <list>', 'Technical indicators to calculate')
  .option('--timeframe <tf>', 'Timeframe (1m, 5m, 1h, 1d)', '1d');

// Strategy commands
program
  .command('strategy')
  .description('Run trading strategies')
  .action(async (options) => {
    const { runStrategy } = require('../lib/cli-handlers');
    await runStrategy({ ...program.opts(), ...options });
  });

// Neural network commands
program
  .command('neural')
  .description('Neural network operations')
  .action(async (options) => {
    const { runNeural } = require('../lib/cli-handlers');
    await runNeural({ ...program.opts(), ...options });
  });

// E2B Swarm orchestration
program
  .command('swarm')
  .description('Multi-agent swarm orchestration')
  .action(async (options) => {
    const { runSwarm } = require('../lib/cli-handlers');
    await runSwarm({ ...program.opts(), ...options });
  });

// Risk management
program
  .command('risk')
  .description('Risk management and analysis')
  .action(async (options) => {
    const { runRisk } = require('../lib/cli-handlers');
    await runRisk({ ...program.opts(), ...options });
  });

// Monitoring and alerts
program
  .command('monitor')
  .description('Monitor system and strategies')
  .action(async (options) => {
    const { runMonitor } = require('../lib/cli-handlers');
    await runMonitor({ ...program.opts(), ...options });
  });

// AgentDB - Vector database for AI memory
program
  .command('agentdb')
  .description('AgentDB vector database operations')
  .action(async (options) => {
    const { runAgentDB } = require('../lib/cli-handlers');
    await runAgentDB({ ...program.opts(), ...options });
  });

// ReasoningBank - Adaptive learning system
program
  .command('reasoningbank')
  .description('ReasoningBank self-learning system')
  .action(async (options) => {
    const { runReasoningBank } = require('../lib/cli-handlers');
    await runReasoningBank({ ...program.opts(), ...options });
  });

// Sublinear-time solver
program
  .command('sublinear')
  .description('Sublinear-time matrix solver')
  .action(async (options) => {
    const { runSublinear } = require('../lib/cli-handlers');
    await runSublinear({ ...program.opts(), ...options });
  });

// Lean-agentic micro-agents
program
  .command('lean')
  .description('Lightweight micro-agent system')
  .action(async (options) => {
    const { runLean } = require('../lib/cli-handlers');
    await runLean({ ...program.opts(), ...options });
  });

// Sports betting
program
  .command('sports')
  .description('Sports betting operations')
  .action(async (options) => {
    const { runSports } = require('../lib/cli-handlers');
    await runSports({ ...program.opts(), ...options });
  });

// Prediction markets
program
  .command('prediction')
  .description('Prediction market operations')
  .action(async (options) => {
    const { runPrediction } = require('../lib/cli-handlers');
    await runPrediction({ ...program.opts(), ...options });
  });

// Market analysis
program
  .command('analyze <symbol>')
  .description('Quick market analysis')
  .action(async (symbol, options) => {
    const { runAnalyze } = require('../lib/cli-handlers');
    await runAnalyze(symbol, { ...program.opts(), ...options });
  });

// Forecast
program
  .command('forecast <symbol>')
  .description('Generate neural forecasts')
  .action(async (symbol, options) => {
    const { runForecast } = require('../lib/cli-handlers');
    await runForecast(symbol, { ...program.opts(), ...options });
  });

// MCP Server
program
  .command('mcp')
  .description('Start MCP server for AI assistants')
  .option('--transport <type>', 'Transport type (stdio, http, websocket)', 'stdio')
  .option('--port <number>', 'Port for HTTP/WebSocket', '3000')
  .option('--host <address>', 'Host address', 'localhost')
  .action((options) => {
    const mcpServerPath = require.resolve('@neural-trader/mcp/bin/mcp-server.js');
    const args = [];

    if (options.transport !== 'stdio') {
      args.push('--transport', options.transport);
    }
    if (options.port !== '3000') {
      args.push('--port', options.port);
    }
    if (options.host !== 'localhost') {
      args.push('--host', options.host);
    }

    require('child_process').spawn(
      process.execPath,
      [mcpServerPath, ...args],
      { stdio: 'inherit' }
    );
  });

// Examples command
program
  .command('examples')
  .description('Show usage examples')
  .action(() => {
    console.log(`
🚀 Neural Trader Examples

Basic Analysis:
  npx neural-trader analyze AAPL
  npx neural-trader forecast BTC --horizon 24

Strategy Backtesting:
  npx neural-trader --strategy momentum --symbol SPY --backtest --start 2020-01-01
  npx neural-trader strategy --strategy pairs --symbols AAPL,MSFT

Neural Networks:
  npx neural-trader --model lstm --train --symbol TSLA
  npx neural-trader neural --models lstm,gru,transformer --predict

E2B Swarm:
  npx neural-trader --swarm hierarchical --agents 12 --e2b
  npx neural-trader swarm --swarm byzantine --consensus-threshold 5

Risk Management:
  npx neural-trader --var --monte-carlo --scenarios 10000
  npx neural-trader monitor --alerts --max-drawdown 0.15

AgentDB & ReasoningBank:
  npx neural-trader agentdb --embeddings --query "profitable SPY patterns"
  npx neural-trader reasoningbank --meta-learning

Sports Betting:
  npx neural-trader sports --sport nfl --arbitrage --bookmakers 5

MCP Server:
  npx neural-trader mcp
  npx neural-trader mcp --transport http --port 8080

For more information:
  Documentation: https://neural-trader.ruv.io
  Issues: https://github.com/ruvnet/neural-trader/issues
`);
  });

// Auto-routing: When global options are used without a subcommand, route to appropriate handler
program.action(async (options) => {
  const opts = program.opts();

  // If no options provided, show help
  if (Object.keys(opts).length === 0) {
    program.outputHelp();
    return;
  }

  const handlers = require('../lib/cli-handlers');

  // Route based on which global options are present
  if (opts.strategy) {
    // Handle combined strategy + swarm
    if (opts.swarm) {
      console.log('Running strategy with swarm coordination...');
      await handlers.runStrategy(opts);
      await handlers.runSwarm(opts);
    } else {
      await handlers.runStrategy(opts);
    }
  } else if (opts.model || opts.models || opts.train || opts.predict) {
    await handlers.runNeural(opts);
  } else if (opts.swarm) {
    await handlers.runSwarm(opts);
  } else if (opts.var || opts.cvar || opts.monteCarlo || opts.stressTest) {
    await handlers.runRisk(opts);
  } else if (opts.report || opts.systemHealth || opts.alerts) {
    await handlers.runMonitor(opts);
  } else if (opts.embeddings || opts.learning || opts.store || opts.query) {
    await handlers.runAgentDB(opts);
  } else if (opts.metaLearning || opts.trackTrajectory || opts.verdictSystem || opts.memoryDistillation) {
    await handlers.runReasoningBank(opts);
  } else if (opts.temporalAdvantage || opts.predictive) {
    await handlers.runSublinear(opts);
  } else if (opts.microAgents || opts.edgeDeployment) {
    await handlers.runLean(opts);
  } else if (opts.sport) {
    await handlers.runSports(opts);
  } else if (opts.platform || opts.marketDepthAnalysis) {
    await handlers.runPrediction(opts);
  } else {
    // No recognized handler options, show help
    program.outputHelp();
  }
});

// Parse arguments
program.parse(process.argv);

// Show help if no command or options provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
