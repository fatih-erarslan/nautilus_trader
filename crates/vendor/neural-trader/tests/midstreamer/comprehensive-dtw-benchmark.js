#!/usr/bin/env node
/**
 * COMPREHENSIVE DTW BENCHMARK
 * Tests multiple pattern sizes to find WASM crossover point
 */

const { TemporalCompare } = require('midstreamer');

// Pure JS DTW
function pureJsDTW(a, b) {
  const n = a.length;
  const m = b.length;
  const dtw = Array(n + 1).fill(null).map(() => Array(m + 1).fill(Infinity));
  dtw[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = Math.abs(a[i - 1] - b[j - 1]);
      dtw[i][j] = cost + Math.min(
        dtw[i - 1][j],
        dtw[i][j - 1],
        dtw[i - 1][j - 1]
      );
    }
  }
  return dtw[n][m];
}

function generatePattern(length) {
  const pattern = [100];
  for (let i = 1; i < length; i++) {
    const change = (Math.random() - 0.5) * 2;
    pattern.push(pattern[i - 1] + change);
  }
  return pattern;
}

async function benchmarkSize(size, iterations) {
  const pattern1 = generatePattern(size);
  const pattern2 = generatePattern(size);

  // Pure JS
  const jsStart = Date.now();
  let jsResult;
  for (let i = 0; i < iterations; i++) {
    jsResult = pureJsDTW(pattern1, pattern2);
  }
  const jsTime = Date.now() - jsStart;

  // WASM
  const temporal = new TemporalCompare();
  const wasmStart = Date.now();
  let wasmResult;
  for (let i = 0; i < iterations; i++) {
    wasmResult = temporal.dtw(
      new Float64Array(pattern1),
      new Float64Array(pattern2)
    );
  }
  const wasmTime = Date.now() - wasmStart;

  const speedup = jsTime / wasmTime;
  const match = Math.abs(jsResult - wasmResult) / jsResult < 0.01;

  return {
    size,
    iterations,
    jsTime,
    wasmTime,
    speedup,
    match,
    jsAvg: jsTime / iterations,
    wasmAvg: wasmTime / iterations
  };
}

async function main() {
  console.log('🔬 COMPREHENSIVE DTW BENCHMARK');
  console.log('='.repeat(80));
  console.log('Testing multiple pattern sizes to find WASM crossover point\n');

  const tests = [
    { size: 50, iterations: 200 },
    { size: 100, iterations: 100 },
    { size: 200, iterations: 50 },
    { size: 500, iterations: 20 },
    { size: 1000, iterations: 10 },
    { size: 2000, iterations: 5 }
  ];

  const results = [];

  for (const test of tests) {
    console.log(`Testing size ${test.size} (${test.iterations} iterations)...`);
    const result = await benchmarkSize(test.size, test.iterations);
    results.push(result);

    const verdict = result.speedup >= 1 ? '✅ WASM faster' : '❌ JS faster';
    console.log(`  Speedup: ${result.speedup.toFixed(2)}x ${verdict}`);
    console.log(`  JS avg: ${result.jsAvg.toFixed(2)}ms, WASM avg: ${result.wasmAvg.toFixed(2)}ms\n`);
  }

  console.log('\n' + '='.repeat(80));
  console.log('📊 SUMMARY');
  console.log('='.repeat(80));
  console.log('Pattern Size | Iterations | JS Time | WASM Time | Speedup | Verdict');
  console.log('-'.repeat(80));

  for (const r of results) {
    const verdict = r.speedup >= 1 ? '✅ WASM' : '❌ JS';
    console.log(
      `${r.size.toString().padStart(12)} | ` +
      `${r.iterations.toString().padStart(10)} | ` +
      `${r.jsTime.toString().padStart(7)}ms | ` +
      `${r.wasmTime.toString().padStart(9)}ms | ` +
      `${r.speedup.toFixed(2).padStart(7)}x | ` +
      verdict
    );
  }

  // Find crossover point
  const wasmFaster = results.filter(r => r.speedup > 1);
  if (wasmFaster.length > 0) {
    const best = wasmFaster.reduce((a, b) => a.speedup > b.speedup ? a : b);
    console.log(`\n✅ WASM becomes faster at ${wasmFaster[0].size} bars`);
    console.log(`   Best speedup: ${best.speedup.toFixed(2)}x at ${best.size} bars`);
  } else {
    console.log('\n❌ WASM NEVER faster than pure JS');
  }

  // Decision
  const avgSpeedup = results.reduce((sum, r) => sum + r.speedup, 0) / results.length;
  console.log(`\nAverage speedup: ${avgSpeedup.toFixed(2)}x`);

  if (avgSpeedup >= 50) {
    console.log('✅ GO: Meets 50x target');
    process.exit(0);
  } else if (avgSpeedup >= 10) {
    console.log('⚠️  CONDITIONAL: 10-50x speedup');
    process.exit(0);
  } else {
    console.log('❌ NO-GO: Use pure Rust DTW instead');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(2);
});
