#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Neural Trader CLI Wrapper
 * Provides fallback strategy for running Rust binary across platforms
 */

function getBinaryPath() {
  const platform = process.platform;
  const arch = process.arch;

  // Map Node.js platform/arch to Rust target triples
  const targetMap = {
    'linux-x64': 'x86_64-unknown-linux-gnu',
    'linux-arm64': 'aarch64-unknown-linux-gnu',
    'darwin-x64': 'x86_64-apple-darwin',
    'darwin-arm64': 'aarch64-apple-darwin',
    'win32-x64': 'x86_64-pc-windows-msvc'
  };

  const target = targetMap[`${platform}-${arch}`];
  if (!target) {
    console.error(`Unsupported platform: ${platform}-${arch}`);
    process.exit(1);
  }

  // Try multiple locations
  const possiblePaths = [
    // Installed via NPM
    path.join(__dirname, '..', 'bin', platform === 'win32' ? 'neural-trader.exe' : 'neural-trader'),
    // Local development
    path.join(__dirname, '..', 'neural-trader-rust', 'target', 'release', platform === 'win32' ? 'neural-trader.exe' : 'neural-trader'),
    // Cargo install
    path.join(process.env.HOME || process.env.USERPROFILE, '.cargo', 'bin', platform === 'win32' ? 'neural-trader.exe' : 'neural-trader')
  ];

  for (const binaryPath of possiblePaths) {
    if (fs.existsSync(binaryPath)) {
      return binaryPath;
    }
  }

  return null;
}

function runWithNativeBinary(binaryPath, args) {
  const child = spawn(binaryPath, args, {
    stdio: 'inherit',
    env: process.env
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code || 0);
    }
  });

  process.on('SIGINT', () => child.kill('SIGINT'));
  process.on('SIGTERM', () => child.kill('SIGTERM'));
}

function runWithNapi() {
  try {
    // Try to load NAPI bindings
    const { runCli } = require('../index.js');
    const args = process.argv.slice(2);
    runCli(args);
  } catch (error) {
    console.error('NAPI bindings not available:', error.message);
    return false;
  }
  return true;
}

function runWithPythonFallback(args) {
  console.log('⚠️  Falling back to Python implementation...');
  const pythonPath = path.join(__dirname, '..', 'venv', 'bin', 'python');
  const scriptPath = path.join(__dirname, '..', 'src', 'mcp_server.py');

  const child = spawn(pythonPath, [scriptPath, ...args], {
    stdio: 'inherit',
    env: process.env
  });

  child.on('exit', (code) => process.exit(code || 0));
}

function main() {
  const args = process.argv.slice(2);

  // Strategy 1: Try native Rust binary
  const binaryPath = getBinaryPath();
  if (binaryPath) {
    console.log('✓ Using native Rust binary');
    runWithNativeBinary(binaryPath, args);
    return;
  }

  // Strategy 2: Try NAPI bindings
  console.log('⚠️  Native binary not found, trying NAPI bindings...');
  if (runWithNapi()) {
    return;
  }

  // Strategy 3: Fall back to Python
  console.log('⚠️  NAPI bindings not available, falling back to Python...');
  runWithPythonFallback(args);
}

main();
