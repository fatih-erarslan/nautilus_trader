const { platform, arch } = require('os');
const path = require('path');

let nativeBinding;

// Platform-specific binding mapping
const PLATFORM_MAP = {
  'darwin': 'darwin',
  'linux': 'linux',
  'win32': 'win32',
  'freebsd': 'freebsd',
  'android': 'android'
};

const ARCH_MAP = {
  'x64': 'x86_64',
  'arm64': 'aarch64',
  'arm': 'armv7',
  'ia32': 'i686'
};

function loadNativeBinding() {
  const platformName = PLATFORM_MAP[platform()] || platform();
  const archName = ARCH_MAP[arch()] || arch();

  // Try loading platform-specific prebuilt binary
  const bindingName = `benchoptimizer.${platformName}-${archName}.node`;
  const bindingPath = path.join(__dirname, bindingName);

  try {
    return require(bindingPath);
  } catch (err) {
    // Fallback to default build location
    const fallbackPath = path.join(
      __dirname,
      '../../crates/nt-benchoptimizer/target/release/libnt_benchoptimizer.node'
    );

    try {
      return require(fallbackPath);
    } catch (fallbackErr) {
      // Try alternative Windows naming
      if (platformName === 'win32') {
        try {
          return require(fallbackPath.replace('.node', '.dll'));
        } catch (winErr) {
          throw new Error(
            `Failed to load native binding for ${platformName}-${archName}.\n` +
            `Attempted paths:\n` +
            `  - ${bindingPath}\n` +
            `  - ${fallbackPath}\n` +
            `Original error: ${err.message}\n` +
            `Fallback error: ${fallbackErr.message}\n` +
            `Please run 'npm run build' to compile the native module.`
          );
        }
      }

      throw new Error(
        `Failed to load native binding for ${platformName}-${archName}.\n` +
        `Attempted paths:\n` +
        `  - ${bindingPath}\n` +
        `  - ${fallbackPath}\n` +
        `Original error: ${err.message}\n` +
        `Fallback error: ${fallbackErr.message}\n` +
        `Please run 'npm run build' to compile the native module.`
      );
    }
  }
}

try {
  nativeBinding = loadNativeBinding();
  // Export all native functions and classes
  module.exports = {
    BenchOptimizer: nativeBinding.BenchOptimizer,
    benchmarkPackage: nativeBinding.benchmarkPackage,
    validatePackage: nativeBinding.validatePackage,
    optimizePackage: nativeBinding.optimizePackage,
    benchmarkAll: nativeBinding.benchmarkAll,
    generateReport: nativeBinding.generateReport,
    compareResults: nativeBinding.compareResults,
    getAllPackages: nativeBinding.getAllPackages
  };
  // Add version info
  module.exports.version = require('./package.json').version;
  module.exports.nativeVersion = nativeBinding.version || 'unknown';
} catch (err) {
  // Fallback to JavaScript implementation
  console.warn('Native binding not available, using JavaScript fallback');
  console.warn('For better performance, run: npm run build');

  const jsImpl = require('./lib/javascript-impl');

  module.exports = {
    validatePackage: jsImpl.validatePackage,
    validateAll: jsImpl.validateAll,
    benchmarkPackage: jsImpl.benchmarkPackage,
    benchmarkAll: jsImpl.benchmarkAll,
    optimizePackage: jsImpl.optimizePackage,
    generateReport: jsImpl.generateReport,
    compareResults: jsImpl.compareResults,
    getAllPackages: jsImpl.getAllPackages
  };

  module.exports.version = require('./package.json').version;
  module.exports.nativeVersion = 'javascript-fallback';
}
