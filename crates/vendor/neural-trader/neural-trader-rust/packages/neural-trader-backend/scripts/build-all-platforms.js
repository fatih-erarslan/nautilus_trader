#!/usr/bin/env node

/**
 * Build script for all platforms
 * This coordinates building across different platforms via GitHub Actions
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PLATFORMS = {
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'linux-arm64': 'aarch64-unknown-linux-gnu',
  'darwin-x64': 'x86_64-apple-darwin',
  'darwin-arm64': 'aarch64-apple-darwin',
  'win32-x64': 'x86_64-pc-windows-msvc',
  'win32-arm64': 'aarch64-pc-windows-msvc'
};

function detectCurrentPlatform() {
  const platform = os.platform();
  const arch = os.arch();

  if (platform === 'linux' && arch === 'x64') return 'linux-x64';
  if (platform === 'linux' && arch === 'arm64') return 'linux-arm64';
  if (platform === 'darwin' && arch === 'x64') return 'darwin-x64';
  if (platform === 'darwin' && arch === 'arm64') return 'darwin-arm64';
  if (platform === 'win32' && arch === 'x64') return 'win32-x64';
  if (platform === 'win32' && arch === 'arm64') return 'win32-arm64';

  throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

function buildForTarget(target, triple) {
  console.log(`\n📦 Building for ${target} (${triple})...`);

  try {
    // Install target
    execSync(`rustup target add ${triple}`, { stdio: 'inherit' });

    // Build
    execSync(
      `cargo build --release --manifest-path=../../Cargo.toml --package nt-napi-bindings --target ${triple}`,
      { stdio: 'inherit' }
    );

    console.log(`✓ Build complete for ${target}`);
    return true;
  } catch (err) {
    console.error(`✗ Build failed for ${target}:`, err.message);
    return false;
  }
}

function main() {
  console.log('=== Neural Trader - Multi-Platform Build ===\n');

  const currentPlatform = detectCurrentPlatform();
  console.log(`Current platform: ${currentPlatform}`);

  const targetPlatform = process.argv[2];

  if (targetPlatform) {
    // Build specific platform
    const triple = PLATFORMS[targetPlatform];
    if (!triple) {
      console.error(`Unknown platform: ${targetPlatform}`);
      console.error(`Available platforms: ${Object.keys(PLATFORMS).join(', ')}`);
      process.exit(1);
    }

    const success = buildForTarget(targetPlatform, triple);
    process.exit(success ? 0 : 1);
  } else {
    // Build current platform
    const triple = PLATFORMS[currentPlatform];
    const success = buildForTarget(currentPlatform, triple);

    console.log('\n📝 Note: For multi-platform builds, use GitHub Actions workflow.');
    console.log('   Push to main branch or create a release tag to trigger CI builds.');

    process.exit(success ? 0 : 1);
  }
}

main();
