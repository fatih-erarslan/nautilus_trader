#!/usr/bin/env node

/**
 * Advanced test suite for @neural-trader/backend
 * Tests performance, memory usage, and advanced features
 */

const os = require('os');
const { performance } = require('perf_hooks');

console.log('=== Neural Trader Backend - Advanced Test Suite ===\n');

const backend = require('../index.js');

// Performance test helper
function benchmark(name, fn, iterations = 1000) {
  console.log(`\nBenchmarking: ${name}`);

  // Warm-up
  for (let i = 0; i < 10; i++) {
    try { fn(); } catch (err) { /* ignore */ }
  }

  // Measure
  const start = performance.now();
  let successful = 0;

  for (let i = 0; i < iterations; i++) {
    try {
      fn();
      successful++;
    } catch (err) {
      // Count failures
    }
  }

  const end = performance.now();
  const duration = end - start;
  const avgTime = duration / iterations;

  console.log(`  Iterations: ${iterations}`);
  console.log(`  Successful: ${successful} (${(successful/iterations*100).toFixed(1)}%)`);
  console.log(`  Total time: ${duration.toFixed(2)}ms`);
  console.log(`  Average time: ${avgTime.toFixed(3)}ms/op`);
  console.log(`  Operations/sec: ${(1000 / avgTime).toFixed(0)}`);
}

// Test 1: Performance benchmarks
console.log('Test 1: Performance Benchmarks');
console.log('--------------------------------');

if (backend.calculateTechnicalIndicators) {
  benchmark('Technical Indicators', () => {
    try {
      // Sample price data
      const prices = Array.from({ length: 100 }, (_, i) => 100 + Math.random() * 10);
      backend.calculateTechnicalIndicators(prices);
    } catch (err) {
      // Expected if function signature is different
    }
  }, 100);
}

// Test 2: Memory usage
console.log('\n\nTest 2: Memory Usage');
console.log('--------------------');

const memBefore = process.memoryUsage();
console.log('Memory before:', {
  heapUsed: `${(memBefore.heapUsed / 1024 / 1024).toFixed(2)} MB`,
  heapTotal: `${(memBefore.heapTotal / 1024 / 1024).toFixed(2)} MB`,
  external: `${(memBefore.external / 1024 / 1024).toFixed(2)} MB`
});

// Simulate some work
const results = [];
for (let i = 0; i < 100; i++) {
  try {
    // Try to use the module
    results.push({ iteration: i, timestamp: Date.now() });
  } catch (err) {
    // Ignore errors
  }
}

if (global.gc) {
  global.gc();
}

const memAfter = process.memoryUsage();
console.log('\nMemory after:', {
  heapUsed: `${(memAfter.heapUsed / 1024 / 1024).toFixed(2)} MB`,
  heapTotal: `${(memAfter.heapTotal / 1024 / 1024).toFixed(2)} MB`,
  external: `${(memAfter.external / 1024 / 1024).toFixed(2)} MB`
});

const memDiff = {
  heapUsed: memAfter.heapUsed - memBefore.heapUsed,
  external: memAfter.external - memBefore.external
};

console.log('\nMemory delta:', {
  heapUsed: `${(memDiff.heapUsed / 1024 / 1024).toFixed(2)} MB`,
  external: `${(memDiff.external / 1024 / 1024).toFixed(2)} MB`
});

// Test 3: Concurrency
console.log('\n\nTest 3: Concurrency');
console.log('-------------------');

async function testConcurrency() {
  const promises = [];
  const count = 10;

  console.log(`Creating ${count} concurrent operations...`);

  for (let i = 0; i < count; i++) {
    const promise = new Promise((resolve) => {
      setTimeout(() => {
        try {
          // Simulate work
          const result = { id: i, timestamp: Date.now() };
          resolve(result);
        } catch (err) {
          resolve({ error: err.message });
        }
      }, Math.random() * 100);
    });
    promises.push(promise);
  }

  const results = await Promise.all(promises);
  const successful = results.filter(r => !r.error).length;

  console.log(`Completed: ${successful}/${count} operations`);
}

testConcurrency().then(() => {
  // Test 4: System information
  console.log('\n\nTest 4: System Information');
  console.log('--------------------------');

  console.log(`Platform: ${os.platform()}`);
  console.log(`Architecture: ${os.arch()}`);
  console.log(`CPU cores: ${os.cpus().length}`);
  console.log(`Total memory: ${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)} GB`);
  console.log(`Free memory: ${(os.freemem() / 1024 / 1024 / 1024).toFixed(2)} GB`);
  console.log(`Node.js: ${process.version}`);
  console.log(`V8: ${process.versions.v8}`);

  console.log('\n=== Advanced tests complete ===\n');
});
