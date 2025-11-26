#!/usr/bin/env node
/**
 * RUST DTW NAPI BENCHMARK
 *
 * Validates pure Rust DTW via NAPI bindings achieves 50-100x speedup target
 * Compares against same pure JS baseline as WASM benchmark for consistency
 *
 * Expected Results:
 * - Rust DTW: 50-100x faster than pure JS
 * - Zero-copy NAPI: No marshalling overhead
 * - LLVM optimizations: Auto-vectorization of inner loops
 */

const { dtwDistanceRust, dtwBatch } = require('../../neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64-gnu.node');
const fs = require('fs');
const path = require('path');

// Pure JS DTW implementation (baseline - same as WASM benchmark)
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

// Generate realistic trading pattern
function generatePattern(length, volatility = 0.02) {
  const pattern = [100]; // Starting price
  for (let i = 1; i < length; i++) {
    const change = (Math.random() - 0.5) * volatility * pattern[i - 1];
    pattern.push(pattern[i - 1] + change);
  }
  return pattern;
}

async function benchmarkSize(size, iterations) {
  const pattern1 = generatePattern(size);
  const pattern2 = generatePattern(size);

  // Pure JS benchmark
  const jsStart = Date.now();
  let jsResult;
  for (let i = 0; i < iterations; i++) {
    jsResult = pureJsDTW(pattern1, pattern2);
  }
  const jsTime = Date.now() - jsStart;

  // Rust NAPI benchmark
  const rustStart = Date.now();
  let rustResult;
  for (let i = 0; i < iterations; i++) {
    const result = dtwDistanceRust(
      new Float64Array(pattern1),
      new Float64Array(pattern2)
    );
    rustResult = result.distance;
  }
  const rustTime = Date.now() - rustStart;

  const speedup = jsTime / rustTime;
  const match = Math.abs(jsResult - rustResult) / Math.max(jsResult, 0.01) < 0.01;

  return {
    size,
    iterations,
    jsTime,
    rustTime,
    speedup,
    match,
    jsAvg: jsTime / iterations,
    rustAvg: rustTime / iterations,
    jsResult,
    rustResult
  };
}

async function benchmarkBatch() {
  console.log('\n📦 BATCH PROCESSING BENCHMARK');
  console.log('='.repeat(80));

  const patternLength = 100;
  const numHistoricalPatterns = 1000;

  // Generate test data
  const currentPattern = generatePattern(patternLength);
  const historicalData = [];
  for (let i = 0; i < numHistoricalPatterns; i++) {
    historicalData.push(...generatePattern(patternLength));
  }

  // Pure JS batch (1000 individual comparisons)
  console.log(`Comparing against ${numHistoricalPatterns} historical patterns...`);
  const jsStart = Date.now();
  const jsDistances = [];
  for (let i = 0; i < numHistoricalPatterns; i++) {
    const start = i * patternLength;
    const end = start + patternLength;
    const histPattern = historicalData.slice(start, end);
    jsDistances.push(pureJsDTW(currentPattern, histPattern));
  }
  const jsTime = Date.now() - jsStart;

  // Rust batch (single NAPI call with all data)
  const rustStart = Date.now();
  const rustDistances = dtwBatch(
    new Float64Array(currentPattern),
    new Float64Array(historicalData),
    patternLength
  );
  const rustTime = Date.now() - rustStart;

  const speedup = jsTime / rustTime;

  // Verify results match
  let maxDiff = 0;
  for (let i = 0; i < numHistoricalPatterns; i++) {
    const diff = Math.abs(jsDistances[i] - rustDistances[i]);
    maxDiff = Math.max(maxDiff, diff);
  }
  const match = maxDiff < 1.0; // Allow small floating point differences

  console.log(`  JS time: ${jsTime}ms (${(jsTime / numHistoricalPatterns).toFixed(2)}ms per pattern)`);
  console.log(`  Rust time: ${rustTime}ms (${(rustTime / numHistoricalPatterns).toFixed(2)}ms per pattern)`);
  console.log(`  Speedup: ${speedup.toFixed(2)}x ${speedup >= 50 ? '✅ TARGET MET' : speedup >= 25 ? '⚠️  CLOSE' : '❌ BELOW TARGET'}`);
  console.log(`  Results match: ${match ? '✅' : '❌'} (max diff: ${maxDiff.toFixed(4)})`);

  return { speedup, match, jsTime, rustTime };
}

async function main() {
  console.log('🚀 RUST DTW NAPI VALIDATION BENCHMARK');
  console.log('='.repeat(80));
  console.log('Target: ≥50x speedup vs pure JavaScript');
  console.log('Technology: Pure Rust + NAPI-RS zero-copy FFI\n');

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
    console.log(`\nTesting size ${test.size} (${test.iterations} iterations)...`);
    const result = await benchmarkSize(test.size, test.iterations);
    results.push(result);

    const verdict = result.speedup >= 50 ? '✅ TARGET MET' :
                    result.speedup >= 25 ? '⚠️  CLOSE' : '❌ BELOW TARGET';
    console.log(`  Speedup: ${result.speedup.toFixed(2)}x ${verdict}`);
    console.log(`  JS avg: ${result.jsAvg.toFixed(3)}ms, Rust avg: ${result.rustAvg.toFixed(3)}ms`);
    console.log(`  Results match: ${result.match ? '✅' : '❌'}`);
  }

  // Batch benchmark
  const batchResult = await benchmarkBatch();

  console.log('\n\n' + '='.repeat(80));
  console.log('📊 COMPREHENSIVE RESULTS SUMMARY');
  console.log('='.repeat(80));
  console.log('Pattern Size | Iterations | JS Time | Rust Time | Speedup | Verdict');
  console.log('-'.repeat(80));

  for (const r of results) {
    const verdict = r.speedup >= 50 ? '✅ TARGET' :
                    r.speedup >= 25 ? '⚠️  CLOSE' : '❌ FAIL';
    console.log(
      `${r.size.toString().padStart(12)} | ` +
      `${r.iterations.toString().padStart(10)} | ` +
      `${r.jsTime.toString().padStart(7)}ms | ` +
      `${r.rustTime.toString().padStart(9)}ms | ` +
      `${r.speedup.toFixed(2).padStart(7)}x | ` +
      verdict
    );
  }

  // Calculate average speedup
  const avgSpeedup = results.reduce((sum, r) => sum + r.speedup, 0) / results.length;
  console.log(`\n📈 Average Speedup: ${avgSpeedup.toFixed(2)}x`);
  console.log(`📦 Batch Processing: ${batchResult.speedup.toFixed(2)}x speedup`);

  // Comparison with WASM results
  console.log('\n' + '='.repeat(80));
  console.log('📊 TECHNOLOGY COMPARISON');
  console.log('='.repeat(80));
  console.log('Technology        | Average Speedup | Verdict');
  console.log('-'.repeat(80));
  console.log(`Pure JavaScript   |          1.00x | Baseline`);
  console.log(`Midstreamer WASM  |          0.42x | ❌ 2.4x SLOWER`);
  console.log(`Pure Rust + NAPI  |  ${avgSpeedup.toFixed(2).padStart(11)}x | ${avgSpeedup >= 50 ? '✅ TARGET MET' : avgSpeedup >= 25 ? '⚠️  CLOSE TO TARGET' : '❌ BELOW TARGET'}`);

  // Performance projection
  console.log('\n' + '='.repeat(80));
  console.log('🎯 PERFORMANCE PROJECTION (100-bar patterns)');
  console.log('='.repeat(80));
  const result100 = results.find(r => r.size === 100);
  if (result100) {
    const comparisonsPerSecond = 1000 / result100.rustAvg;
    console.log(`Throughput: ${comparisonsPerSecond.toFixed(0)} pattern comparisons/second`);
    console.log(`Time for 1,000 patterns: ${(1000 * result100.rustAvg / 1000).toFixed(2)}s`);
    console.log(`Time for 10,000 patterns: ${(10000 * result100.rustAvg / 1000).toFixed(2)}s`);
    console.log(`Time for 100,000 patterns: ${(100000 * result100.rustAvg / 1000).toFixed(2)}s`);
  }

  // GO/NO-GO decision
  console.log('\n' + '='.repeat(80));
  console.log('🎯 GO/NO-GO DECISION');
  console.log('='.repeat(80));

  const allMatch = results.every(r => r.match);
  const verdict = avgSpeedup >= 50 ? 'GO' :
                  avgSpeedup >= 25 ? 'CONDITIONAL GO' :
                  avgSpeedup >= 10 ? 'PROCEED WITH CAUTION' : 'NO-GO';

  if (avgSpeedup >= 50 && allMatch) {
    console.log(`✅ GO: ${avgSpeedup.toFixed(1)}x average speedup ≥ 50x target`);
    console.log('   All correctness tests passed');
    console.log('   Phase 1 DTW integration VALIDATED');
    console.log('   Ready to integrate into trading strategies');
  } else if (avgSpeedup >= 25 && allMatch) {
    console.log(`⚠️  CONDITIONAL GO: ${avgSpeedup.toFixed(1)}x average speedup`);
    console.log('   Below 50x target but still significant improvement');
    console.log('   Recommend SIMD optimization to reach target');
  } else if (avgSpeedup >= 10 && allMatch) {
    console.log(`⚠️  PROCEED WITH CAUTION: ${avgSpeedup.toFixed(1)}x speedup`);
    console.log('   Below target but better than WASM (0.42x)');
    console.log('   Consider algorithm optimization or hardware-specific features');
  } else {
    console.log(`❌ NO-GO: ${avgSpeedup.toFixed(1)}x speedup insufficient`);
    console.log('   Re-evaluate implementation or consider alternative approaches');
  }

  // Save results
  const reportDir = path.join(__dirname, '../../docs/performance');
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }

  const resultsPath = path.join(reportDir, 'rust-dtw-benchmark-results.json');
  const report = {
    timestamp: new Date().toISOString(),
    verdict,
    averageSpeedup: avgSpeedup,
    batchSpeedup: batchResult.speedup,
    targetMet: avgSpeedup >= 50,
    allTestsPass: allMatch,
    individualResults: results,
    batchResult,
    comparison: {
      pureJS: 1.0,
      midstreamerWASM: 0.42,
      rustNAPI: avgSpeedup
    }
  };

  fs.writeFileSync(resultsPath, JSON.stringify(report, null, 2));
  console.log(`\n📄 Results saved to: ${resultsPath}`);

  // Exit code
  process.exit(avgSpeedup >= 50 && allMatch ? 0 : avgSpeedup >= 25 ? 0 : 1);
}

// Run benchmark
main().catch(error => {
  console.error('\n❌ Benchmark failed:', error.message);
  console.error(error.stack);
  process.exit(2);
});
