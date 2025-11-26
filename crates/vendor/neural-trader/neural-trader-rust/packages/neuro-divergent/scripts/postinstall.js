#!/usr/bin/env node

/**
 * Post-install script for @neural-trader/neuro-divergent
 * Validates that the native module loaded correctly
 */

const { platform, arch } = process;

console.log('🔧 Post-install check for @neural-trader/neuro-divergent');
console.log(`   Platform: ${platform}`);
console.log(`   Architecture: ${arch}`);

try {
  const { version, isGpuAvailable } = require('..');
  console.log(`   Version: ${version()}`);
  console.log(`   GPU Available: ${isGpuAvailable()}`);
  console.log('✅ Native module loaded successfully\n');
} catch (error) {
  console.error('❌ Failed to load native module:');
  console.error(`   ${error.message}`);
  console.error('\n⚠️  The package may not support your platform/architecture combination.');
  console.error('   Supported platforms: Linux (x64, ARM64), macOS (x64, ARM64), Windows (x64, ARM64)');
  console.error('   Please open an issue at: https://github.com/ruvnet/neural-trader/issues\n');
  process.exit(1);
}
