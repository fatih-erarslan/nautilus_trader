#!/usr/bin/env node
/**
 * CRITICAL VALIDATION BENCHMARK
 *
 * Tests midstreamer WASM DTW vs pure JS implementation
 * Target: ≥50x speedup (GO criterion)
 *
 * Phase 1 validation - Day 1
 */

const { TemporalCompare, benchmark_dtw } = require('midstreamer');

// Pure JS DTW implementation (baseline)
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

// Generate test patterns
function generatePattern(length, volatility = 0.02) {
  const pattern = [100]; // Starting price
  for (let i = 1; i < length; i++) {
    const change = (Math.random() - 0.5) * volatility * pattern[i - 1];
    pattern.push(pattern[i - 1] + change);
  }
  return pattern;
}

// Benchmark results
const results = {
  pattern_size: 0,
  iterations: 0,
  js_time_ms: 0,
  wasm_time_ms: 0,
  speedup: 0,
  verdict: 'UNKNOWN',
  js_result: 0,
  wasm_result: 0,
  results_match: false
};

async function runBenchmark() {
  console.log('🔬 MIDSTREAMER VALIDATION BENCHMARK');
  console.log('=' .repeat(60));
  console.log('Target: ≥50x speedup for GO decision\n');

  // Test with realistic trading pattern (100 bars = ~6 hours of 5min candles)
  const patternSize = 100;
  const iterations = 100;

  results.pattern_size = patternSize;
  results.iterations = iterations;

  const pattern1 = generatePattern(patternSize);
  const pattern2 = generatePattern(patternSize);

  console.log(`Pattern size: ${patternSize} bars`);
  console.log(`Iterations: ${iterations}\n`);

  // Warm-up runs
  console.log('Warming up...');
  pureJsDTW(pattern1.slice(0, 10), pattern2.slice(0, 10));

  // Benchmark Pure JS
  console.log('\n📊 Benchmarking Pure JS DTW...');
  const jsStart = Date.now();
  for (let i = 0; i < iterations; i++) {
    results.js_result = pureJsDTW(pattern1, pattern2);
  }
  const jsTime = Date.now() - jsStart;
  results.js_time_ms = jsTime;

  const jsAvg = jsTime / iterations;
  console.log(`  Total time: ${jsTime}ms`);
  console.log(`  Average: ${jsAvg.toFixed(2)}ms per comparison`);
  console.log(`  Result: ${results.js_result.toFixed(2)}`);

  // Benchmark WASM (correct API: TemporalCompare.dtw method)
  console.log('\n⚡ Benchmarking WASM DTW...');
  const temporal = new TemporalCompare();
  const wasmStart = Date.now();
  for (let i = 0; i < iterations; i++) {
    // Use correct midstreamer API
    results.wasm_result = temporal.dtw(
      new Float64Array(pattern1),
      new Float64Array(pattern2)
    );
  }
  const wasmTime = Date.now() - wasmStart;
  results.wasm_time_ms = wasmTime;

  const wasmAvg = wasmTime / iterations;
  console.log(`  Total time: ${wasmTime}ms`);
  console.log(`  Average: ${wasmAvg.toFixed(2)}ms per comparison`);
  console.log(`  Result: ${results.wasm_result.toFixed(2)}`);

  // Calculate speedup
  const speedup = jsTime / wasmTime;
  results.speedup = speedup;
  results.results_match = Math.abs(results.js_result - results.wasm_result) / results.js_result < 0.01;

  // Verdict
  console.log('\n' + '='.repeat(60));
  console.log('📈 RESULTS ANALYSIS');
  console.log('='.repeat(60));
  console.log(`Speedup: ${speedup.toFixed(1)}x`);
  console.log(`Results match: ${results.results_match ? '✅' : '❌'} (${Math.abs(results.js_result - results.wasm_result).toFixed(2)} difference)`);

  if (speedup >= 50 && results.results_match) {
    results.verdict = 'GO';
    console.log(`\n✅ GO DECISION: ${speedup.toFixed(1)}x speedup ≥ 50x target`);
    console.log('   Phase 1 can proceed with confidence');
  } else if (speedup >= 25 && results.results_match) {
    results.verdict = 'CONDITIONAL GO';
    console.log(`\n⚠️  CONDITIONAL GO: ${speedup.toFixed(1)}x speedup`);
    console.log('   Below 50x target but still significant');
    console.log('   Recommend proceeding with adjusted expectations');
  } else if (speedup >= 10 && results.results_match) {
    results.verdict = 'USE RUST FALLBACK';
    console.log(`\n⚠️  USE RUST FALLBACK: ${speedup.toFixed(1)}x speedup`);
    console.log('   Below target, recommend pure Rust DTW instead');
  } else {
    results.verdict = 'NO-GO';
    console.log(`\n❌ NO-GO: ${speedup.toFixed(1)}x speedup < 10x minimum`);
    console.log('   Insufficient performance gain');
    console.log('   Use pure Rust DTW implementation');
  }

  // Performance projection
  console.log('\n📊 PERFORMANCE PROJECTION');
  console.log('='.repeat(60));
  const comparisonsPerSecond = 1000 / wasmAvg;
  console.log(`Throughput: ${comparisonsPerSecond.toFixed(0)} pattern comparisons/second`);
  console.log(`Time for 1000 patterns: ${(1000 * wasmAvg / 1000).toFixed(2)}s`);
  console.log(`Time for 10000 patterns: ${(10000 * wasmAvg / 1000).toFixed(2)}s`);

  // Export results
  const fs = require('fs');
  const resultsPath = '/workspaces/neural-trader/tests/midstreamer/benchmark-results.json';
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`\n📄 Results saved to: ${resultsPath}`);

  return results;
}

// Run benchmark
runBenchmark()
  .then(results => {
    console.log('\n✅ Benchmark completed successfully');
    process.exit(results.verdict === 'GO' || results.verdict === 'CONDITIONAL GO' ? 0 : 1);
  })
  .catch(error => {
    console.error('\n❌ Benchmark failed:', error.message);
    console.error(error.stack);
    process.exit(2);
  });
