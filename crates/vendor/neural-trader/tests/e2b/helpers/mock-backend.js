/**
 * Mock Backend for E2B Tests
 *
 * Provides mock implementations when NAPI backend is not available
 */

// Mock delay utility
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Mock E2B sandbox creation
 */
async function createE2bSandbox(name, template = 'base') {
  await delay(100); // Simulate API delay

  return {
    sandboxId: `mock-sbx-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    name,
    template: template || 'base',
    status: 'running',
    createdAt: new Date().toISOString(),
  };
}

/**
 * Mock process execution
 */
async function executeE2bProcess(sandboxId, command) {
  await delay(50); // Simulate execution delay

  // Simulate command outputs
  let stdout = '';
  let exitCode = 0;

  if (command.includes('echo')) {
    stdout = command.replace(/^.*echo\s+"?([^"]+)"?.*$/, '$1');
  } else if (command.includes('node -e')) {
    // Simulate Node.js execution
    if (command.includes('Math.PI')) {
      stdout = '6.283185307179586';
    } else if (command.includes('process.memoryUsage')) {
      stdout = JSON.stringify({
        rss: 50000000,
        heapTotal: 20000000,
        heapUsed: 15000000,
        external: 1000000,
      });
    } else if (command.includes('Array')) {
      stdout = '1000000';
    } else if (command.includes('lodash')) {
      stdout = '6';
    } else if (command.includes('JSON.stringify')) {
      stdout = JSON.stringify({
        sma: '102.20',
        signal: 'BUY',
      });
    } else {
      stdout = 'Mock output';
    }
  } else if (command.includes('python')) {
    if (command.includes('numpy')) {
      stdout = '3.0';
    } else if (command.includes('json.dumps')) {
      stdout = JSON.stringify({
        current_price: 100.5,
        predicted_price: 102.3,
        confidence: 0.85,
      });
    } else {
      stdout = 'Mock Python output';
    }
  } else if (command.includes('npm install')) {
    stdout = 'added 1 package';
  } else if (command.includes('pip install')) {
    stdout = 'Successfully installed package';
  } else {
    stdout = 'Command executed successfully';
  }

  return {
    sandboxId,
    command,
    exitCode,
    stdout,
    stderr: '',
  };
}

/**
 * Mock neural trader initialization
 */
async function initNeuralTrader(config) {
  await delay(50);
  return 'Neural Trader initialized (mock mode)';
}

/**
 * Mock system info
 */
function getSystemInfo() {
  return {
    version: '2.1.1-mock',
    rustVersion: '1.70.0',
    buildTimestamp: new Date().toISOString(),
    features: ['mock', 'testing'],
    totalTools: 50,
  };
}

/**
 * Mock health check
 */
async function healthCheck() {
  await delay(10);
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptimeSeconds: 0,
  };
}

module.exports = {
  createE2bSandbox,
  executeE2bProcess,
  initNeuralTrader,
  getSystemInfo,
  healthCheck,
};
