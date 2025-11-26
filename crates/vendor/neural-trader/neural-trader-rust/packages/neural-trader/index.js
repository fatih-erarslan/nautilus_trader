// neural-trader - Complete platform meta package
// Enhanced with error handling, platform detection, and helpful messages

const os = require('os');
const path = require('path');

/**
 * Platform detection for NAPI bindings
 */
function getPlatform() {
  const platform = os.platform();
  const arch = os.arch();
  return { platform, arch };
}

/**
 * Try to load a package with helpful error handling
 */
function tryRequire(packageName, description) {
  try {
    return require(packageName);
  } catch (error) {
    if (error.code === 'MODULE_NOT_FOUND') {
      console.error(`\n❌ Missing dependency: ${packageName}`);
      console.error(`   ${description}`);
      console.error(`\n   Install with: npm install ${packageName}\n`);
      return null;
    }

    // Check for NAPI binding issues
    if (error.message.includes('.node')) {
      const { platform, arch } = getPlatform();
      console.error(`\n❌ Native binding error in ${packageName}`);
      console.error(`   Platform: ${platform} ${arch}`);
      console.error(`   ${error.message}`);
      console.error(`\n   This package may not be compiled for your platform.`);
      console.error(`   Try rebuilding: npm rebuild ${packageName}\n`);
      return null;
    }

    throw error;
  }
}

/**
 * Lazy-load packages to avoid startup overhead
 */
const packages = {
  get core() {
    return tryRequire('@neural-trader/core', 'Core types and interfaces');
  },
  get backtesting() {
    return tryRequire('@neural-trader/backtesting', 'Backtesting engine (requires Rust NAPI bindings)');
  },
  get neural() {
    return tryRequire('@neural-trader/neural', 'Neural network models (requires Rust NAPI bindings)');
  },
  get risk() {
    return tryRequire('@neural-trader/risk', 'Risk management (requires Rust NAPI bindings)');
  },
  get strategies() {
    return tryRequire('@neural-trader/strategies', 'Trading strategies');
  },
  get portfolio() {
    return tryRequire('@neural-trader/portfolio', 'Portfolio management');
  },
  get execution() {
    return tryRequire('@neural-trader/execution', 'Order execution');
  },
  get brokers() {
    return tryRequire('@neural-trader/brokers', 'Broker integrations');
  },
  get marketData() {
    return tryRequire('@neural-trader/market-data', 'Market data feeds');
  },
  get features() {
    return tryRequire('@neural-trader/features', 'Technical indicators');
  },
  get sportsBetting() {
    return tryRequire('@neural-trader/sports-betting', 'Sports betting strategies');
  },
  get predictionMarkets() {
    return tryRequire('@neural-trader/prediction-markets', 'Prediction market trading');
  },
  get newsTrading() {
    return tryRequire('@neural-trader/news-trading', 'News-based trading');
  }
};

/**
 * Main exports with safe loading
 */
module.exports = {
  // Platform info
  platform: getPlatform(),

  // Lazy-loaded packages (access via properties)
  packages,

  // Named exports with error handling
  ...(packages.core || {}),
  ...(packages.backtesting || {}),
  ...(packages.neural || {}),
  ...(packages.risk || {}),
  ...(packages.strategies || {}),
  ...(packages.portfolio || {}),
  ...(packages.execution || {}),
  ...(packages.brokers || {}),
  ...(packages.marketData || {}),
  ...(packages.features || {}),
  ...(packages.sportsBetting || {}),
  ...(packages.predictionMarkets || {}),
  ...(packages.newsTrading || {}),

  /**
   * Helper function to check if all dependencies are available
   */
  checkDependencies() {
    const results = {};
    const packageNames = Object.keys(packages);

    console.log('🔍 Checking Neural Trader dependencies...\n');

    for (const name of packageNames) {
      const pkg = packages[name];
      results[name] = pkg !== null;
      const status = pkg ? '✅' : '❌';
      console.log(`${status} @neural-trader/${name.replace(/([A-Z])/g, '-$1').toLowerCase()}`);
    }

    console.log();
    return results;
  },

  /**
   * Get version information
   */
  getVersionInfo() {
    const packageJson = require('./package.json');
    return {
      version: packageJson.version,
      dependencies: packageJson.dependencies,
      platform: getPlatform()
    };
  },

  /**
   * Quick start helper
   */
  quickStart() {
    console.log('🚀 Neural Trader Quick Start\n');
    console.log('1. Create a new project:');
    console.log('   npx neural-trader init my-project\n');
    console.log('2. Run backtesting:');
    console.log('   npx neural-trader backtest strategy.js\n');
    console.log('3. Start trading:');
    console.log('   npx neural-trader trade strategy.js --mode paper\n');
    console.log('Documentation: https://github.com/ruvnet/neural-trader');
  }
};

// Show helpful message if run directly
if (require.main === module) {
  console.log('📦 Neural Trader v' + require('./package.json').version);
  console.log('Complete AI-powered algorithmic trading platform\n');
  module.exports.quickStart();
}
