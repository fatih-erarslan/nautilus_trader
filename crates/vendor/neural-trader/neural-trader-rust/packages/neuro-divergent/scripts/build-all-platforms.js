#!/usr/bin/env node

/**
 * Build script for all platforms
 * Uses napi-rs to build native bindings for multiple platforms
 */

const { execSync } = require('child_process');
const { platform } = process;

console.log('🔨 Building neuro-divergent for all platforms...\n');

const platforms = [
  'x86_64-unknown-linux-gnu',
  'x86_64-unknown-linux-musl',
  'aarch64-unknown-linux-gnu',
  'aarch64-unknown-linux-musl',
  'x86_64-apple-darwin',
  'aarch64-apple-darwin',
  'x86_64-pc-windows-msvc',
];

const currentPlatform = platforms.find(p => {
  if (platform === 'linux' && p.includes('linux') && p.includes('gnu')) return true;
  if (platform === 'darwin' && p.includes('darwin')) return true;
  if (platform === 'win32' && p.includes('windows')) return true;
  return false;
});

console.log(`Building for current platform: ${currentPlatform || 'default'}\n`);

try {
  if (currentPlatform) {
    console.log(`🔨 Building for ${currentPlatform}...`);
    execSync(`napi build --platform --release --target ${currentPlatform} --crate neuro-divergent-napi`, {
      stdio: 'inherit',
      cwd: __dirname + '/..'
    });
    console.log(`✅ Build complete for ${currentPlatform}\n`);
  } else {
    console.log('🔨 Building with default target...');
    execSync('napi build --platform --release --crate neuro-divergent-napi', {
      stdio: 'inherit',
      cwd: __dirname + '/..'
    });
    console.log('✅ Build complete\n');
  }

  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('✅ All builds completed successfully!');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

} catch (error) {
  console.error('❌ Build failed:', error.message);
  process.exit(1);
}
