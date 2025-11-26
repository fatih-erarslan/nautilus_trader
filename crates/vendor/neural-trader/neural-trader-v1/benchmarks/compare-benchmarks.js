#!/usr/bin/env node
/**
 * Compare Benchmark Results - Detects performance regressions
 * Usage: node scripts/compare-benchmarks.js baseline.json current.json
 */

const fs = require('fs');
const path = require('path');
const chalk = require('chalk');
const Table = require('cli-table3');

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length < 2) {
  console.error(chalk.red('❌ Error: Missing arguments'));
  console.log(chalk.yellow('\nUsage: node scripts/compare-benchmarks.js <baseline.json> <current.json>'));
  console.log(chalk.gray('\nExample:'));
  console.log(chalk.gray('  node scripts/compare-benchmarks.js baseline-results.json latest-results.json'));
  process.exit(1);
}

const baselinePath = path.resolve(args[0]);
const currentPath = path.resolve(args[1]);

// Validate files exist
if (!fs.existsSync(baselinePath)) {
  console.error(chalk.red(`❌ Baseline file not found: ${baselinePath}`));
  process.exit(1);
}

if (!fs.existsSync(currentPath)) {
  console.error(chalk.red(`❌ Current file not found: ${currentPath}`));
  process.exit(1);
}

// Load benchmark results
let baseline, current;

try {
  baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
  current = JSON.parse(fs.readFileSync(currentPath, 'utf8'));
} catch (err) {
  console.error(chalk.red('❌ Error parsing JSON files:'), err.message);
  process.exit(1);
}

/**
 * Compare performance metrics
 */
function compareMetrics(baseline, current) {
  const regressions = [];
  const improvements = [];
  const unchanged = [];

  // Assuming both files have results.category structure
  const baselineCategories = baseline.results || {};
  const currentCategories = current.results || {};

  for (const [category, baselineMetrics] of Object.entries(baselineCategories)) {
    const currentMetrics = currentCategories[category] || [];

    if (!Array.isArray(baselineMetrics)) continue;

    baselineMetrics.forEach((baselineMetric) => {
      const currentMetric = currentMetrics.find((m) => m.name === baselineMetric.name);

      if (!currentMetric) {
        regressions.push({
          category,
          name: baselineMetric.name,
          issue: 'Missing in current results',
          baselineHz: baselineMetric.hz,
          currentHz: 0,
          change: -100,
        });
        return;
      }

      const baselineHz = parseFloat(baselineMetric.hz) || 0;
      const currentHz = parseFloat(currentMetric.hz) || 0;

      if (baselineHz === 0) return; // Skip invalid baseline

      const changePercent = ((currentHz - baselineHz) / baselineHz) * 100;

      const comparison = {
        category,
        name: baselineMetric.name,
        baselineHz,
        currentHz,
        change: changePercent,
        baselineTime: parseFloat(baselineMetric.meanMs) || 0,
        currentTime: parseFloat(currentMetric.meanMs) || 0,
      };

      if (changePercent < -10) {
        // >10% slower = regression
        regressions.push(comparison);
      } else if (changePercent > 10) {
        // >10% faster = improvement
        improvements.push(comparison);
      } else {
        unchanged.push(comparison);
      }
    });
  }

  return { regressions, improvements, unchanged };
}

/**
 * Print comparison results
 */
function printResults(regressions, improvements, unchanged) {
  console.log(chalk.bold.magenta('\n╔══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.magenta('║              PERFORMANCE COMPARISON REPORT                       ║'));
  console.log(chalk.bold.magenta('╚══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.cyan(`📊 Baseline: ${path.basename(baselinePath)}`));
  console.log(chalk.cyan(`📊 Current:  ${path.basename(currentPath)}\n`));

  // Summary statistics
  const totalTests = regressions.length + improvements.length + unchanged.length;
  const regressionPercent = ((regressions.length / totalTests) * 100).toFixed(1);
  const improvementPercent = ((improvements.length / totalTests) * 100).toFixed(1);

  const summaryTable = new Table();
  summaryTable.push(
    ['Total Tests', totalTests],
    [chalk.red('Regressions (>10% slower)'), `${regressions.length} (${regressionPercent}%)`],
    [chalk.green('Improvements (>10% faster)'), `${improvements.length} (${improvementPercent}%)`],
    [chalk.gray('Unchanged (±10%)'), `${unchanged.length} (${(100 - regressionPercent - improvementPercent).toFixed(1)}%)`]
  );
  console.log(summaryTable.toString());

  // Print regressions
  if (regressions.length > 0) {
    console.log(chalk.bold.red('\n🔴 PERFORMANCE REGRESSIONS\n'));

    const regressionsTable = new Table({
      head: [
        chalk.white('Category'),
        chalk.white('Function'),
        chalk.white('Baseline (ops/s)'),
        chalk.white('Current (ops/s)'),
        chalk.white('Change'),
        chalk.white('Time Impact'),
      ],
      colWidths: [20, 30, 18, 18, 12, 15],
    });

    regressions
      .sort((a, b) => a.change - b.change) // Worst first
      .forEach((r) => {
        const changeColor = r.change < -25 ? chalk.red : r.change < -15 ? chalk.yellow : chalk.gray;
        const timeDiff = r.currentTime - r.baselineTime;

        regressionsTable.push([
          r.category,
          r.name,
          r.baselineHz.toFixed(2),
          r.currentHz.toFixed(2),
          changeColor(`${r.change.toFixed(1)}%`),
          `+${timeDiff.toFixed(2)}ms`,
        ]);
      });

    console.log(regressionsTable.toString());
  }

  // Print improvements
  if (improvements.length > 0) {
    console.log(chalk.bold.green('\n🟢 PERFORMANCE IMPROVEMENTS\n'));

    const improvementsTable = new Table({
      head: [
        chalk.white('Category'),
        chalk.white('Function'),
        chalk.white('Baseline (ops/s)'),
        chalk.white('Current (ops/s)'),
        chalk.white('Change'),
        chalk.white('Time Saved'),
      ],
      colWidths: [20, 30, 18, 18, 12, 15],
    });

    improvements
      .sort((a, b) => b.change - a.change) // Best first
      .forEach((i) => {
        const changeColor = i.change > 50 ? chalk.green : i.change > 25 ? chalk.cyan : chalk.gray;
        const timeSaved = i.baselineTime - i.currentTime;

        improvementsTable.push([
          i.category,
          i.name,
          i.baselineHz.toFixed(2),
          i.currentHz.toFixed(2),
          changeColor(`+${i.change.toFixed(1)}%`),
          `-${timeSaved.toFixed(2)}ms`,
        ]);
      });

    console.log(improvementsTable.toString());
  }

  // Overall verdict
  console.log(chalk.bold.yellow('\n📋 VERDICT\n'));

  if (regressions.length === 0) {
    console.log(chalk.green('✅ No performance regressions detected!'));
  } else if (regressions.length < 5 && regressions.length / totalTests < 0.1) {
    console.log(chalk.yellow(`⚠️  Minor regressions detected (${regressions.length}/${totalTests})`));
    console.log(chalk.gray('   Review and address if critical path affected'));
  } else {
    console.log(chalk.red(`❌ Significant regressions detected (${regressions.length}/${totalTests})`));
    console.log(chalk.gray('   Investigate and fix before merging'));
  }

  if (improvements.length > 0) {
    console.log(
      chalk.green(`\n🎉 Performance improvements: ${improvements.length} functions faster by >10%!`)
    );
  }

  console.log();

  // Exit code based on regressions
  if (regressions.length > 0 && regressions.length / totalTests > 0.15) {
    // >15% regressions = fail
    process.exit(1);
  }
}

/**
 * Main execution
 */
function main() {
  console.log(chalk.bold.cyan('\n🔍 Comparing benchmark results...\n'));

  const { regressions, improvements, unchanged } = compareMetrics(baseline, current);

  printResults(regressions, improvements, unchanged);
}

// Execute
main();
