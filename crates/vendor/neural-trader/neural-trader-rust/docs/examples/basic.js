#!/usr/bin/env node

/**
 * Basic Neural Trader MCP Usage Examples
 *
 * This file demonstrates basic usage of Neural Trader MCP tools.
 */

const { McpServer } = require('@neural-trader/mcp');

async function main() {
  console.log('🚀 Neural Trader MCP - Basic Examples\n');

  // Create MCP server
  const server = new McpServer({ transport: 'stdio' });
  await server.start();

  console.log('✅ Server started\n');

  // Example 1: Ping
  console.log('📍 Example 1: Ping');
  const ping = await server.callTool('ping', {});
  console.log('Result:', ping);
  console.log('');

  // Example 2: List Strategies
  console.log('📋 Example 2: List Trading Strategies');
  const strategies = await server.callTool('list_strategies', {});
  console.log(`Found ${strategies.strategies.length} strategies:`);
  strategies.strategies.forEach((s, i) => {
    console.log(`  ${i + 1}. ${s.name} - GPU: ${s.gpu_capable}`);
  });
  console.log('');

  // Example 3: Quick Analysis
  console.log('📊 Example 3: Quick Market Analysis');
  try {
    const analysis = await server.callTool('quick_analysis', {
      symbol: 'AAPL',
      useGpu: false
    });
    console.log('Symbol:', analysis.symbol);
    console.log('Price:', analysis.current_price);
    console.log('Trend:', analysis.trend);
    console.log('Recommendation:', analysis.recommendation);
    console.log('Confidence:', (analysis.confidence * 100).toFixed(1) + '%');
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 4: Get Strategy Info
  console.log('ℹ️  Example 4: Strategy Information');
  try {
    const info = await server.callTool('get_strategy_info', {
      strategy: 'momentum'
    });
    console.log('Strategy:', info.name);
    console.log('Description:', info.description);
    console.log('GPU Capable:', info.gpu_capable);
    console.log('Parameters:', JSON.stringify(info.parameters, null, 2));
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 5: Simulate Trade
  console.log('💹 Example 5: Simulate Trade');
  try {
    const simulation = await server.callTool('simulate_trade', {
      strategy: 'momentum',
      symbol: 'AAPL',
      action: 'buy',
      useGpu: false
    });
    console.log('Symbol:', simulation.symbol);
    console.log('Action:', simulation.action);
    console.log('Entry Price:', simulation.entry_price);
    console.log('Exit Price:', simulation.exit_price);
    console.log('P&L:', simulation.profit_loss);
    console.log('P&L %:', (simulation.profit_loss_pct * 100).toFixed(2) + '%');
    console.log('Execution Time:', simulation.execution_time_ms + 'ms');
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 6: Portfolio Status
  console.log('💼 Example 6: Portfolio Status');
  try {
    const portfolio = await server.callTool('get_portfolio_status', {
      includeAnalytics: true
    });
    console.log('Total Value:', portfolio.total_value);
    console.log('Cash:', portfolio.cash);
    console.log('Positions:', portfolio.positions.length);
    console.log('Daily P&L:', portfolio.daily_pnl);
    if (portfolio.analytics) {
      console.log('Sharpe Ratio:', portfolio.analytics.sharpe_ratio);
      console.log('Max Drawdown:', (portfolio.analytics.max_drawdown * 100).toFixed(2) + '%');
    }
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 7: List Broker Types
  console.log('🏦 Example 7: Available Brokers');
  const brokers = await server.callTool('list_broker_types', {});
  console.log('Available brokers:', brokers.brokers.join(', '));
  console.log('');

  // Clean up
  await server.stop();
  console.log('✅ Examples complete');
}

// Error handling
main().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
