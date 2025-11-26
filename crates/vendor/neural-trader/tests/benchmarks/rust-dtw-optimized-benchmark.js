#!/usr/bin/env node
/**
 * OPTIMIZED RUST DTW BENCHMARK
 *
 * Tests performance improvements from:
 * 1. Parallel batch processing (Rayon) - target 2-4x
 * 2. Cache-friendly flat memory layout - target 1.5-2x
 * 3. Memory pooling for repeated operations
 * 4. Adaptive batch selection (auto parallel vs sequential)
 *
 * Combined target: 5-10x total speedup over pure JS
 * Comparison: Baseline (2.65x) → Optimized (5-10x)
 */

const fs = require('fs');
const path = require('path');

// Try to load optimized NAPI bindings
let dtwDistanceRustOptimized, dtwBatchParallel, dtwBatchAdaptive;
try {
  const bindings = require('../../neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64-gnu.node');
  dtwDistanceRustOptimized = bindings.dtwDistanceRustOptimized;
  dtwBatchParallel = bindings.dtwBatchParallel;
  dtwBatchAdaptive = bindings.dtwBatchAdaptive;

  console.log('✅ Loaded optimized Rust DTW bindings');
} catch (error) {
  console.error('❌ Failed to load optimized bindings:', error.message);
  console.error('   Run: cd neural-trader-rust/crates/napi-bindings && npm run build');
  process.exit(1);
}

// Load baseline implementations for comparison
const { dtwDistanceRust, dtwBatch } = require('../../neural-trader-rust/crates/napi-bindings/neural-trader.linux-x64-gnu.node');

// Pure JS baseline
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
  const pattern = [100];
  for (let i = 1; i < length; i++) {
    const change = (Math.random() - 0.5) * volatility * pattern[i - 1];
    pattern.push(pattern[i - 1] + change);
  }
  return pattern;
}

async function benchmarkOptimizedBatch(numPatterns = 1000) {
  console.log(`\n📦 OPTIMIZED BATCH PROCESSING BENCHMARK (${numPatterns} patterns)`);
  console.log('='.repeat(80));

  const patternLength = 100;
  const currentPattern = generatePattern(patternLength);
  const historicalData = [];
  for (let i = 0; i < numPatterns; i++) {
    historicalData.push(...generatePattern(patternLength));
  }

  const currentF64 = new Float64Array(currentPattern);
  const historicalF64 = new Float64Array(historicalData);

  // 1. Pure JS batch (baseline)
  console.log(`\nComparing against ${numPatterns} historical patterns...`);
  const jsStart = Date.now();
  const jsDistances = [];
  for (let i = 0; i < numPatterns; i++) {
    const start = i * patternLength;
    const end = start + patternLength;
    const histPattern = historicalData.slice(start, end);
    jsDistances.push(pureJsDTW(currentPattern, histPattern));
  }
  const jsTime = Date.now() - jsStart;

  // 2. Rust batch (baseline - sequential)
  const rustBaselineStart = Date.now();
  const rustBaselineDistances = dtwBatch(currentF64, historicalF64, patternLength);
  const rustBaselineTime = Date.now() - rustBaselineStart;

  // 3. Rust batch parallel (NEW - Rayon)
  const rustParallelStart = Date.now();
  const rustParallelDistances = dtwBatchParallel(currentF64, historicalF64, patternLength);
  const rustParallelTime = Date.now() - rustParallelStart;

  // 4. Rust batch adaptive (NEW - auto-select)
  const rustAdaptiveStart = Date.now();
  const rustAdaptiveDistances = dtwBatchAdaptive(currentF64, historicalF64, patternLength);
  const rustAdaptiveTime = Date.now() - rustAdaptiveStart;

  // Calculate speedups
  const baselineSpeedup = jsTime / rustBaselineTime;
  const parallelSpeedup = jsTime / rustParallelTime;
  const adaptiveSpeedup = jsTime / rustAdaptiveTime;
  const parallelVsBaseline = rustBaselineTime / rustParallelTime;

  // Verify correctness
  let maxDiff = 0;
  for (let i = 0; i < numPatterns; i++) {
    const diff = Math.abs(jsDistances[i] - rustParallelDistances[i]);
    maxDiff = Math.max(maxDiff, diff);
  }
  const match = maxDiff < 1.0;

  console.log(`\n📊 RESULTS:`);
  console.log(`  Pure JS:              ${jsTime}ms (${(jsTime / numPatterns).toFixed(3)}ms per pattern)`);
  console.log(`  Rust Baseline:        ${rustBaselineTime}ms (${baselineSpeedup.toFixed(2)}x speedup)`);
  console.log(`  Rust Parallel:        ${rustParallelTime}ms (${parallelSpeedup.toFixed(2)}x speedup) ${parallelSpeedup >= 5 ? '✅ TARGET MET' : parallelSpeedup >= 3 ? '⚠️  CLOSE' : '❌ BELOW TARGET'}`);
  console.log(`  Rust Adaptive:        ${rustAdaptiveTime}ms (${adaptiveSpeedup.toFixed(2)}x speedup)`);
  console.log(`  Parallel vs Baseline: ${parallelVsBaseline.toFixed(2)}x improvement`);
  console.log(`  Results match:        ${match ? '✅' : '❌'} (max diff: ${maxDiff.toFixed(4)})`);

  return {
    numPatterns,
    jsTime,
    rustBaselineTime,
    rustParallelTime,
    rustAdaptiveTime,
    baselineSpeedup,
    parallelSpeedup,
    adaptiveSpeedup,
    parallelVsBaseline,
    match
  };
}

async function benchmarkScaling() {
  console.log('\n\n📈 SCALING ANALYSIS');
  console.log('='.repeat(80));
  console.log('Testing how parallel speedup scales with batch size\n');

  const batchSizes = [100, 500, 1000, 2000, 5000, 10000];
  const results = [];

  for (const size of batchSizes) {
    console.log(`Testing batch size: ${size} patterns...`);
    const result = await benchmarkOptimizedBatch(size);
    results.push(result);

    // Brief pause to let system stabilize
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log('\n\n' + '='.repeat(80));
  console.log('📊 SCALING RESULTS SUMMARY');
  console.log('='.repeat(80));
  console.log('Batch Size | Baseline | Parallel | Adaptive | Parallel vs Baseline | Verdict');
  console.log('-'.repeat(80));

  for (const r of results) {
    const verdict = r.parallelSpeedup >= 5 ? '✅ EXCELLENT' :
                    r.parallelSpeedup >= 3 ? '⚠️  GOOD' : '❌ NEEDS WORK';
    console.log(
      `${r.numPatterns.toString().padStart(10)} | ` +
      `${r.baselineSpeedup.toFixed(2).padStart(8)}x | ` +
      `${r.parallelSpeedup.toFixed(2).padStart(8)}x | ` +
      `${r.adaptiveSpeedup.toFixed(2).padStart(8)}x | ` +
      `${r.parallelVsBaseline.toFixed(2).padStart(20)}x | ` +
      verdict
    );
  }

  return results;
}

async function benchmarkSinglePattern() {
  console.log('\n\n🎯 SINGLE PATTERN OPTIMIZED vs BASELINE');
  console.log('='.repeat(80));

  const tests = [
    { size: 50, iterations: 200 },
    { size: 100, iterations: 100 },
    { size: 200, iterations: 50 },
    { size: 500, iterations: 20 },
    { size: 1000, iterations: 10 },
  ];

  const results = [];

  for (const test of tests) {
    const pattern1 = generatePattern(test.size);
    const pattern2 = generatePattern(test.size);

    // Rust baseline
    const rustBaselineStart = Date.now();
    for (let i = 0; i < test.iterations; i++) {
      dtwDistanceRust(new Float64Array(pattern1), new Float64Array(pattern2));
    }
    const rustBaselineTime = Date.now() - rustBaselineStart;

    // Rust optimized (flat layout)
    const rustOptimizedStart = Date.now();
    for (let i = 0; i < test.iterations; i++) {
      dtwDistanceRustOptimized(new Float64Array(pattern1), new Float64Array(pattern2));
    }
    const rustOptimizedTime = Date.now() - rustOptimizedStart;

    const improvement = rustBaselineTime / rustOptimizedTime;

    results.push({
      size: test.size,
      iterations: test.iterations,
      rustBaselineTime,
      rustOptimizedTime,
      improvement
    });

    console.log(`  Size ${test.size}: Baseline ${rustBaselineTime}ms, Optimized ${rustOptimizedTime}ms (${improvement.toFixed(2)}x)`);
  }

  return results;
}

async function main() {
  console.log('🚀 RUST DTW OPTIMIZATION VALIDATION BENCHMARK');
  console.log('='.repeat(80));
  console.log('Target: 5-10x total speedup over pure JavaScript');
  console.log('Optimizations: Parallel (Rayon) + Cache-friendly layout + Memory pooling\n');

  // Test 1: Single pattern optimization (cache-friendly layout)
  const singleResults = await benchmarkSinglePattern();

  // Test 2: Batch processing with different sizes
  const scalingResults = await benchmarkScaling();

  // Analysis
  console.log('\n\n' + '='.repeat(80));
  console.log('🎯 OPTIMIZATION IMPACT ANALYSIS');
  console.log('='.repeat(80));

  const avgCacheImprovement = singleResults.reduce((sum, r) => sum + r.improvement, 0) / singleResults.length;
  const bestParallelSpeedup = Math.max(...scalingResults.map(r => r.parallelSpeedup));
  const avgParallelSpeedup = scalingResults.reduce((sum, r) => sum + r.parallelSpeedup, 0) / scalingResults.length;

  console.log('\n📊 Single Pattern (Cache-Friendly Layout):');
  console.log(`  Average improvement: ${avgCacheImprovement.toFixed(2)}x`);
  console.log(`  Best improvement: ${Math.max(...singleResults.map(r => r.improvement)).toFixed(2)}x`);
  console.log(`  Impact: ${avgCacheImprovement >= 1.5 ? '✅ Significant' : '⚠️  Moderate'} (${((avgCacheImprovement - 1) * 100).toFixed(1)}% faster)`);

  console.log('\n📊 Batch Processing (Parallel Execution):');
  console.log(`  Best speedup: ${bestParallelSpeedup.toFixed(2)}x vs pure JS`);
  console.log(`  Average speedup: ${avgParallelSpeedup.toFixed(2)}x vs pure JS`);
  console.log(`  vs Baseline (2.65x): ${(avgParallelSpeedup / 2.65).toFixed(2)}x improvement`);

  console.log('\n📊 Combined Optimization Impact:');
  const totalImprovement = avgCacheImprovement * (avgParallelSpeedup / 2.65);
  console.log(`  Cache + Parallel: ${totalImprovement.toFixed(2)}x total improvement`);
  console.log(`  Expected final speedup: ${(2.65 * totalImprovement).toFixed(2)}x vs pure JS`);

  // Verdict
  console.log('\n' + '='.repeat(80));
  console.log('🎯 FINAL VERDICT');
  console.log('='.repeat(80));

  if (avgParallelSpeedup >= 5) {
    console.log('✅ SUCCESS: Achieved 5-10x target speedup');
    console.log('   Parallel batch processing significantly outperforms baseline');
    console.log('   Ready for production deployment');
  } else if (avgParallelSpeedup >= 3) {
    console.log('⚠️  GOOD PROGRESS: 3-5x speedup achieved');
    console.log('   Significant improvement over baseline (2.65x)');
    console.log('   Consider GPU acceleration for 10x+ target');
  } else {
    console.log('❌ BELOW TARGET: <3x speedup');
    console.log('   Optimization not meeting expectations');
    console.log('   Review parallel overhead and algorithm');
  }

  // Recommendations
  console.log('\n' + '='.repeat(80));
  console.log('💡 RECOMMENDATIONS');
  console.log('='.repeat(80));

  if (avgParallelSpeedup < 5) {
    console.log('\n1. GPU Acceleration (Next Phase):');
    console.log('   - Expected: 10-50x speedup for large batches');
    console.log('   - CUDA/ROCm for massive parallelism');
    console.log('   - Best for 10,000+ pattern comparisons');

    console.log('\n2. FastDTW Algorithm:');
    console.log('   - O(n) vs O(n²) complexity');
    console.log('   - 10-100x speedup for large patterns');
    console.log('   - Trade accuracy for speed (controlled approximation)');
  }

  console.log('\n3. Production Deployment:');
  if (avgParallelSpeedup >= 3) {
    console.log('   ✅ Use dtw_batch_adaptive for automatic optimization');
    console.log('   ✅ Parallel mode for batches >100 patterns');
    console.log('   ✅ Sequential mode for small batches (<100)');
  } else {
    console.log('   ⚠️  Stick with baseline for now');
    console.log('   ⚠️  Investigate parallel overhead');
  }

  // Save results
  const reportDir = path.join(__dirname, '../../docs/performance');
  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
  }

  const report = {
    timestamp: new Date().toISOString(),
    singlePatternResults: singleResults,
    scalingResults: scalingResults,
    analysis: {
      avgCacheImprovement,
      avgParallelSpeedup,
      bestParallelSpeedup,
      totalImprovement,
      expectedFinalSpeedup: 2.65 * totalImprovement
    },
    verdict: avgParallelSpeedup >= 5 ? 'SUCCESS' :
             avgParallelSpeedup >= 3 ? 'GOOD_PROGRESS' : 'BELOW_TARGET'
  };

  const reportPath = path.join(reportDir, 'rust-dtw-optimized-results.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\n📄 Results saved to: ${reportPath}`);

  // Exit code
  process.exit(avgParallelSpeedup >= 3 ? 0 : 1);
}

main().catch(error => {
  console.error('\n❌ Benchmark failed:', error.message);
  console.error(error.stack);
  process.exit(2);
});
