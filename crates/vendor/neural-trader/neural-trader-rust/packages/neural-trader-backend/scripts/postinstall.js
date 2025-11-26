#!/usr/bin/env node

/**
 * Post-install script for @neural-trader/backend
 * Attempts to load the correct platform-specific native binary
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const PLATFORM_MAP = {
  'linux': {
    'x64': '@neural-trader/backend-linux-x64-gnu',
    'arm64': '@neural-trader/backend-linux-arm64-gnu'
  },
  'darwin': {
    'x64': '@neural-trader/backend-darwin-x64',
    'arm64': '@neural-trader/backend-darwin-arm64'
  },
  'win32': {
    'x64': '@neural-trader/backend-win32-x64-msvc',
    'arm64': '@neural-trader/backend-win32-arm64-msvc'
  }
};

function getPlatformPackage() {
  const platform = os.platform();
  const arch = os.arch();

  if (!PLATFORM_MAP[platform]) {
    throw new Error(`Unsupported platform: ${platform}`);
  }

  const platformPackage = PLATFORM_MAP[platform][arch];
  if (!platformPackage) {
    throw new Error(`Unsupported architecture: ${arch} on ${platform}`);
  }

  return platformPackage;
}

function findNativeBinary() {
  try {
    const platformPackage = getPlatformPackage();
    const binaryPath = require.resolve(platformPackage);

    if (fs.existsSync(binaryPath)) {
      console.log(`✓ Found native binary: ${platformPackage}`);
      return binaryPath;
    }
  } catch (err) {
    // Optional dependency not installed
  }

  // Try to find in node_modules
  const nodeModulesPath = path.join(__dirname, '..', 'node_modules');
  if (fs.existsSync(nodeModulesPath)) {
    const platformPackage = getPlatformPackage();
    const potentialPath = path.join(nodeModulesPath, platformPackage);
    if (fs.existsSync(potentialPath)) {
      console.log(`✓ Found native binary in node_modules: ${platformPackage}`);
      return potentialPath;
    }
  }

  console.warn('⚠ Native binary not found for your platform.');
  console.warn(`Platform: ${os.platform()}, Architecture: ${os.arch()}`);
  console.warn('You may need to build from source or wait for pre-built binaries.');

  return null;
}

// Run post-install check
try {
  findNativeBinary();
} catch (err) {
  console.error('Error during post-install:', err.message);
  process.exit(0); // Don't fail installation
}
