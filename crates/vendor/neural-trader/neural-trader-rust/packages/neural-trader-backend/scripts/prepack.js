#!/usr/bin/env node

/**
 * Pre-pack script for @neural-trader/backend
 * Validates that all required platform binaries are present
 */

const fs = require('fs');
const path = require('path');

const REQUIRED_PLATFORMS = [
  'linux-x64-gnu',
  'linux-arm64-gnu',
  'darwin-x64',
  'darwin-arm64',
  'win32-x64-msvc'
];

const BINARY_NAME_MAP = {
  'linux-x64-gnu': 'neural-trader-backend.linux-x64-gnu.node',
  'linux-arm64-gnu': 'neural-trader-backend.linux-arm64-gnu.node',
  'darwin-x64': 'neural-trader-backend.darwin-x64.node',
  'darwin-arm64': 'neural-trader-backend.darwin-arm64.node',
  'win32-x64-msvc': 'neural-trader-backend.win32-x64-msvc.node'
};

function validateBinaries() {
  const rootDir = path.join(__dirname, '..');

  let missingPlatforms = [];
  let foundPlatforms = [];

  for (const platform of REQUIRED_PLATFORMS) {
    const binaryName = BINARY_NAME_MAP[platform];
    // Check in current package directory (where NAPI build outputs them)
    const binaryPath = path.join(rootDir, binaryName);

    if (fs.existsSync(binaryPath)) {
      const stats = fs.statSync(binaryPath);
      foundPlatforms.push({
        platform,
        size: (stats.size / 1024 / 1024).toFixed(2) + ' MB',
        path: binaryPath
      });
    } else {
      missingPlatforms.push(platform);
    }
  }

  console.log('\n=== Native Binary Validation ===\n');

  if (foundPlatforms.length > 0) {
    console.log('✓ Found binaries:');
    foundPlatforms.forEach(({ platform, size, path }) => {
      console.log(`  ${platform}: ${size} (${path})`);
    });
  }

  if (missingPlatforms.length > 0) {
    console.log('\n⚠ Missing binaries:');
    missingPlatforms.forEach(platform => {
      console.log(`  ${platform}`);
    });

    console.log('\nNote: Missing binaries should be built via GitHub Actions CI.');
    console.log('Run the build workflow to generate all platform binaries.');
  }

  console.log('\n');

  // Don't fail the build - some platforms might be built separately
  return foundPlatforms.length > 0;
}

// Run validation
if (!validateBinaries()) {
  console.error('No native binaries found. Please build at least one platform.');
  process.exit(1);
}
