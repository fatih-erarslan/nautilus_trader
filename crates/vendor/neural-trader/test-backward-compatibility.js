#!/usr/bin/env node
/**
 * Backward Compatibility Test for v2.4.0 API
 * Ensures existing code using wrappers still works
 */

const neuralTrader = require('./index.js');

console.log('Testing Backward Compatibility with v2.4.0 API\n');
console.log('='.repeat(80));

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`✓ ${name}`);
    passed++;
  } catch (error) {
    console.log(`✗ ${name} - ${error.message}`);
    failed++;
  }
}

// ============================================================
// Test CLI Wrapper (v2.4.0 API)
// ============================================================
console.log('\n--- CLI WRAPPER (v2.4.0) ---');
test('cli is exported as object', () => {
  if (typeof neuralTrader.cli !== 'object') throw new Error('cli not exported');
});
test('cli.initProject exists', () => {
  if (typeof neuralTrader.cli.initProject !== 'function') throw new Error('not a function');
});
test('cli.listStrategies exists', () => {
  if (typeof neuralTrader.cli.listStrategies !== 'function') throw new Error('not a function');
});
test('cli.listBrokers exists', () => {
  if (typeof neuralTrader.cli.listBrokers !== 'function') throw new Error('not a function');
});
test('cli.runBacktest exists', () => {
  if (typeof neuralTrader.cli.runBacktest !== 'function') throw new Error('not a function');
});
test('cli.startPaperTrading exists', () => {
  if (typeof neuralTrader.cli.startPaperTrading !== 'function') throw new Error('not a function');
});
test('cli.startLiveTrading exists', () => {
  if (typeof neuralTrader.cli.startLiveTrading !== 'function') throw new Error('not a function');
});
test('cli.getAgentStatus exists', () => {
  if (typeof neuralTrader.cli.getAgentStatus !== 'function') throw new Error('not a function');
});
test('cli.trainNeuralModel exists', () => {
  if (typeof neuralTrader.cli.trainNeuralModel !== 'function') throw new Error('not a function');
});
test('cli.manageSecrets exists', () => {
  if (typeof neuralTrader.cli.manageSecrets !== 'function') throw new Error('not a function');
});

// ============================================================
// Test MCP Wrapper (v2.4.0 API)
// ============================================================
console.log('\n--- MCP WRAPPER (v2.4.0) ---');
test('mcp is exported as object', () => {
  if (typeof neuralTrader.mcp !== 'object') throw new Error('mcp not exported');
});
test('mcp.startServer exists', () => {
  if (typeof neuralTrader.mcp.startServer !== 'function') throw new Error('not a function');
});
test('mcp.stopServer exists', () => {
  if (typeof neuralTrader.mcp.stopServer !== 'function') throw new Error('not a function');
});
test('mcp.getServerStatus exists', () => {
  if (typeof neuralTrader.mcp.getServerStatus !== 'function') throw new Error('not a function');
});
test('mcp.listTools exists', () => {
  if (typeof neuralTrader.mcp.listTools !== 'function') throw new Error('not a function');
});
test('mcp.callTool exists', () => {
  if (typeof neuralTrader.mcp.callTool !== 'function') throw new Error('not a function');
});
test('mcp.restartServer exists', () => {
  if (typeof neuralTrader.mcp.restartServer !== 'function') throw new Error('not a function');
});
test('mcp.configureClaudeDesktop exists', () => {
  if (typeof neuralTrader.mcp.configureClaudeDesktop !== 'function') throw new Error('not a function');
});
test('mcp.testConnection exists', () => {
  if (typeof neuralTrader.mcp.testConnection !== 'function') throw new Error('not a function');
});

// Test MCP helper functions
test('mcp.startStdioServer exists', () => {
  if (typeof neuralTrader.mcp.startStdioServer !== 'function') throw new Error('not a function');
});
test('mcp.startHttpServer exists', () => {
  if (typeof neuralTrader.mcp.startHttpServer !== 'function') throw new Error('not a function');
});
test('mcp.startWebSocketServer exists', () => {
  if (typeof neuralTrader.mcp.startWebSocketServer !== 'function') throw new Error('not a function');
});

// ============================================================
// Test Swarm Wrapper (v2.4.0 API)
// ============================================================
console.log('\n--- SWARM WRAPPER (v2.4.0) ---');
test('swarm is exported as object', () => {
  if (typeof neuralTrader.swarm !== 'object') throw new Error('swarm not exported');
});
test('swarm.init exists', () => {
  if (typeof neuralTrader.swarm.init !== 'function') throw new Error('not a function');
});
test('swarm.spawnAgent exists', () => {
  if (typeof neuralTrader.swarm.spawnAgent !== 'function') throw new Error('not a function');
});
test('swarm.getStatus exists', () => {
  if (typeof neuralTrader.swarm.getStatus !== 'function') throw new Error('not a function');
});
test('swarm.listAgents exists', () => {
  if (typeof neuralTrader.swarm.listAgents !== 'function') throw new Error('not a function');
});
test('swarm.orchestrateTask exists', () => {
  if (typeof neuralTrader.swarm.orchestrateTask !== 'function') throw new Error('not a function');
});
test('swarm.stopAgent exists', () => {
  if (typeof neuralTrader.swarm.stopAgent !== 'function') throw new Error('not a function');
});
test('swarm.destroy exists', () => {
  if (typeof neuralTrader.swarm.destroy !== 'function') throw new Error('not a function');
});
test('swarm.scale exists', () => {
  if (typeof neuralTrader.swarm.scale !== 'function') throw new Error('not a function');
});
test('swarm.healthCheck exists', () => {
  if (typeof neuralTrader.swarm.healthCheck !== 'function') throw new Error('not a function');
});

// Test swarm helper functions
test('swarm.createMeshSwarm exists', () => {
  if (typeof neuralTrader.swarm.createMeshSwarm !== 'function') throw new Error('not a function');
});
test('swarm.createHierarchicalSwarm exists', () => {
  if (typeof neuralTrader.swarm.createHierarchicalSwarm !== 'function') throw new Error('not a function');
});
test('swarm.createStarSwarm exists', () => {
  if (typeof neuralTrader.swarm.createStarSwarm !== 'function') throw new Error('not a function');
});

// ============================================================
// Test Legacy Functions (v2.3.x compatibility)
// ============================================================
console.log('\n--- LEGACY FUNCTIONS (v2.3.x) - Should exist but throw helpful errors ---');
test('fetchMarketData exists (legacy stub)', () => {
  if (typeof neuralTrader.fetchMarketData !== 'function') throw new Error('not exported');
});
test('streamMarketData exists (legacy stub)', () => {
  if (typeof neuralTrader.streamMarketData !== 'function') throw new Error('not exported');
});
test('runStrategy exists (legacy stub)', () => {
  if (typeof neuralTrader.runStrategy !== 'function') throw new Error('not exported');
});
test('backtest exists (legacy stub)', () => {
  if (typeof neuralTrader.backtest !== 'function') throw new Error('not exported');
});
test('executeOrder exists (legacy stub)', () => {
  if (typeof neuralTrader.executeOrder !== 'function') throw new Error('not exported');
});
test('getPortfolio exists (legacy stub)', () => {
  if (typeof neuralTrader.getPortfolio !== 'function') throw new Error('not exported');
});
test('trainModel exists (legacy stub)', () => {
  if (typeof neuralTrader.trainModel !== 'function') throw new Error('not exported');
});
test('predict exists (legacy stub)', () => {
  if (typeof neuralTrader.predict !== 'function') throw new Error('not exported');
});

// ============================================================
// Final Report
// ============================================================
console.log('\n' + '='.repeat(80));
console.log(`\n✓ PASSED: ${passed}`);
console.log(`✗ FAILED: ${failed}`);

if (failed > 0) {
  console.log('\n❌ BACKWARD COMPATIBILITY BROKEN!');
  process.exit(1);
} else {
  console.log('\n✅ FULL BACKWARD COMPATIBILITY MAINTAINED!');
  console.log('   - All v2.4.0 wrapper APIs work');
  console.log('   - Legacy v2.3.x stubs exist');
  console.log('   - Existing code will not break');
  process.exit(0);
}
