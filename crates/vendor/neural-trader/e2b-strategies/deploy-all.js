#!/usr/bin/env node

/**
 * E2B Multi-Strategy Deployment Script
 * Deploys all 5 trading strategies to E2B sandboxes
 *
 * Usage:
 *   node deploy-all.js [--platform neural-trader|flow-nexus|e2b]
 */

const fs = require('fs');
const path = require('path');

// Configuration
const STRATEGIES = [
  { name: 'momentum', port: 3000, memory: 512, packages: ['@alpacahq/alpaca-trade-api', 'express'] },
  { name: 'neural-forecast', port: 3001, memory: 2048, packages: ['@alpacahq/alpaca-trade-api', '@tensorflow/tfjs-node', 'express'] },
  { name: 'mean-reversion', port: 3002, memory: 512, packages: ['@alpacahq/alpaca-trade-api', 'express'] },
  { name: 'risk-manager', port: 3003, memory: 512, packages: ['@alpacahq/alpaca-trade-api', 'express'] },
  { name: 'portfolio-optimizer', port: 3004, memory: 512, packages: ['@alpacahq/alpaca-trade-api', 'express'] }
];

const ENV_VARS = {
  ALPACA_API_KEY: process.env.ALPACA_API_KEY,
  ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
  ALPACA_BASE_URL: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets'
};

// Validate environment
function validateEnvironment() {
  const missing = [];

  if (!ENV_VARS.ALPACA_API_KEY) missing.push('ALPACA_API_KEY');
  if (!ENV_VARS.ALPACA_SECRET_KEY) missing.push('ALPACA_SECRET_KEY');

  if (missing.length > 0) {
    console.error('❌ Missing required environment variables:', missing.join(', '));
    console.error('\nSet them with:');
    console.error('  export ALPACA_API_KEY="your_key"');
    console.error('  export ALPACA_SECRET_KEY="your_secret"');
    process.exit(1);
  }

  console.log('✅ Environment validated');
}

// Deployment platform detection
const platform = process.argv.includes('--platform')
  ? process.argv[process.argv.indexOf('--platform') + 1]
  : 'neural-trader';

console.log(`\n🚀 E2B Multi-Strategy Deployment\n`);
console.log(`Platform: ${platform}`);
console.log(`Strategies: ${STRATEGIES.length}`);
console.log(`\n${'='.repeat(50)}\n`);

validateEnvironment();

// Generate deployment instructions based on platform
function generateDeploymentCommands() {
  console.log('📋 Deployment Commands:\n');

  STRATEGIES.forEach((strategy, index) => {
    console.log(`\n${index + 1}. ${strategy.name.toUpperCase()} Strategy`);
    console.log(`${'─'.repeat(50)}`);

    switch (platform) {
      case 'neural-trader':
        console.log(`
// Using Neural Trader MCP
mcp__neural-trader__create_e2b_sandbox({
  name: "${strategy.name}-strategy",
  template: "node",
  memory_mb: ${strategy.memory},
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "${strategy.port}"
  },
  install_packages: ${JSON.stringify(strategy.packages)}
})

// Upload files
mcp__neural-trader__sandbox_upload({
  sandbox_id: "${strategy.name}-strategy",
  file_path: "/app/index.js",
  content: fs.readFileSync("./e2b-strategies/${strategy.name}/index.js", "utf-8")
})

mcp__neural-trader__sandbox_upload({
  sandbox_id: "${strategy.name}-strategy",
  file_path: "/app/package.json",
  content: fs.readFileSync("./e2b-strategies/${strategy.name}/package.json", "utf-8")
})

// Start strategy
mcp__neural-trader__sandbox_execute({
  sandbox_id: "${strategy.name}-strategy",
  code: "cd /app && npm install && npm start",
  capture_output: true
})
        `);
        break;

      case 'flow-nexus':
        console.log(`
// Using Flow-Nexus Platform
mcp__flow-nexus__sandbox_create({
  template: "node",
  name: "${strategy.name}-strategy",
  memory_mb: ${strategy.memory},
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "${strategy.port}"
  },
  install_packages: ${JSON.stringify(strategy.packages)}
})

// Configure and start
mcp__flow-nexus__sandbox_configure({
  sandbox_id: "${strategy.name}-strategy",
  run_commands: [
    "cd /app",
    "npm install",
    "npm start"
  ]
})

// Optional: Subscribe to real-time monitoring
mcp__flow-nexus__execution_stream_subscribe({
  sandbox_id: "${strategy.name}-strategy",
  stream_type: "claude-flow-swarm"
})
        `);
        break;

      case 'e2b':
        console.log(`
# Using E2B CLI
e2b sandbox create --template node --name ${strategy.name}-strategy --memory ${strategy.memory}
e2b sandbox upload ${strategy.name}-strategy ./e2b-strategies/${strategy.name}/
e2b sandbox exec ${strategy.name}-strategy "cd /app && npm install && npm start"
        `);
        break;

      default:
        console.error(`Unknown platform: ${platform}`);
        process.exit(1);
    }
  });
}

// Generate monitoring commands
function generateMonitoringCommands() {
  console.log(`\n\n📊 Monitoring Commands:\n`);
  console.log(`${'='.repeat(50)}\n`);

  console.log('// Health check all strategies');
  console.log('const healthChecks = await Promise.all([');
  STRATEGIES.forEach(strategy => {
    console.log(`  fetch('http://${strategy.name}:${strategy.port}/health'),`);
  });
  console.log(']);\n');

  console.log('// Get status from all strategies');
  console.log('const statuses = await Promise.all([');
  STRATEGIES.forEach(strategy => {
    const endpoint = strategy.name === 'risk-manager' ? 'metrics' : 'status';
    console.log(`  fetch('http://${strategy.name}:${strategy.port}/${endpoint}'),`);
  });
  console.log(']);\n');

  console.log('// Execute all strategies');
  console.log('const executions = await Promise.all([');
  STRATEGIES.forEach(strategy => {
    const endpoint = strategy.name === 'risk-manager' ? 'monitor' : (strategy.name === 'portfolio-optimizer' ? 'optimize' : 'execute');
    console.log(`  fetch('http://${strategy.name}:${strategy.port}/${endpoint}', { method: 'POST' }),`);
  });
  console.log(']);\n');
}

// Generate Docker Compose for local testing
function generateDockerCompose() {
  console.log(`\n\n🐳 Docker Compose (Local Testing):\n`);
  console.log(`${'='.repeat(50)}\n`);

  const dockerCompose = {
    version: '3.8',
    services: {}
  };

  STRATEGIES.forEach(strategy => {
    dockerCompose.services[strategy.name] = {
      build: {
        context: `./e2b-strategies/${strategy.name}`,
        dockerfile: 'Dockerfile'
      },
      ports: [`${strategy.port}:${strategy.port}`],
      environment: {
        ALPACA_API_KEY: '${ALPACA_API_KEY}',
        ALPACA_SECRET_KEY: '${ALPACA_SECRET_KEY}',
        ALPACA_BASE_URL: '${ALPACA_BASE_URL:-https://paper-api.alpaca.markets}',
        PORT: strategy.port.toString()
      },
      restart: 'unless-stopped',
      mem_limit: `${strategy.memory}m`
    };
  });

  console.log('```yaml');
  console.log('# docker-compose.yml');
  console.log(require('util').inspect(dockerCompose, { depth: null, colors: false }).replace(/'/g, ''));
  console.log('```\n');

  console.log('# Start all strategies locally:');
  console.log('docker-compose up -d\n');
}

// Cost estimates
function generateCostEstimates() {
  console.log(`\n\n💰 Cost Estimates:\n`);
  console.log(`${'='.repeat(50)}\n`);

  const totalMemory = STRATEGIES.reduce((sum, s) => sum + s.memory, 0);
  const hourlyRate = 0.10; // Approximate E2B cost per GB-hour
  const hourlyCost = (totalMemory / 1024) * hourlyRate;
  const dailyCost = hourlyCost * 24;
  const monthlyCost = dailyCost * 30;

  console.log(`Total Memory: ${totalMemory}MB (${(totalMemory / 1024).toFixed(2)}GB)`);
  console.log(`Hourly Cost: $${hourlyCost.toFixed(2)}`);
  console.log(`Daily Cost (24/7): $${dailyCost.toFixed(2)}`);
  console.log(`Monthly Cost (24/7): $${monthlyCost.toFixed(2)}`);
  console.log(`\nOptimized (market hours only):`);
  console.log(`  6.5 hours/day × 5 days/week = 32.5 hours/week`);
  console.log(`  Monthly: $${(hourlyCost * 32.5 * 4).toFixed(2)}`);
  console.log(`\n✅ Savings with market-hours only: ${((1 - (32.5 * 4) / (24 * 30)) * 100).toFixed(0)}%\n`);
}

// Main execution
generateDeploymentCommands();
generateMonitoringCommands();
generateDockerCompose();
generateCostEstimates();

console.log(`\n${'='.repeat(50)}\n`);
console.log('📚 Next Steps:\n');
console.log('1. Copy deployment commands above');
console.log('2. Execute in your Claude Code environment');
console.log('3. Monitor with health check commands');
console.log('4. Review logs for any errors');
console.log('5. Scale up/down as needed\n');

console.log('📖 Documentation:');
console.log('   /workspaces/neural-trader/docs/e2b-deployment/\n');

console.log('✨ Happy Trading!\n');
