/**
 * Console reporter for benchmark results
 */

import { BenchmarkResult, ComparisonResult, RegressionAlert } from '../types';

export class ConsoleReporter {
  private colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
  };

  /**
   * Report single benchmark result
   */
  report(result: BenchmarkResult): void {
    console.log(`\n${this.colors.bright}${this.colors.cyan}Benchmark: ${result.name}${this.colors.reset}`);
    console.log(`${'='.repeat(60)}`);
    console.log(`Iterations:  ${result.iterations}`);
    console.log(`Duration:    ${this.formatDuration(result.duration)}`);
    console.log(`Mean:        ${this.formatDuration(result.mean)}`);
    console.log(`Median:      ${this.formatDuration(result.median)}`);
    console.log(`Std Dev:     ${this.formatDuration(result.stdDev)}`);
    console.log(`Min:         ${this.formatDuration(result.min)}`);
    console.log(`Max:         ${this.formatDuration(result.max)}`);
    console.log(`P95:         ${this.formatDuration(result.p95)}`);
    console.log(`P99:         ${this.formatDuration(result.p99)}`);
    console.log(`Throughput:  ${result.throughput.toFixed(2)} ops/sec`);
    console.log(`\nMemory:`);
    console.log(`  Heap Used:   ${this.formatBytes(result.memory.heapUsed)}`);
    console.log(`  Heap Total:  ${this.formatBytes(result.memory.heapTotal)}`);
    console.log(`  Peak:        ${this.formatBytes(result.memory.peak)}`);
  }

  /**
   * Report comparison results
   */
  reportComparison(comparison: ComparisonResult): void {
    const { baseline, current, improvement, faster, significant } = comparison;

    console.log(`\n${this.colors.bright}${this.colors.cyan}Comparison Results${this.colors.reset}`);
    console.log(`${'='.repeat(60)}`);

    const color = faster ? this.colors.green : this.colors.red;
    const symbol = faster ? '↓' : '↑';
    const label = faster ? 'faster' : 'slower';

    console.log(`\n${this.colors.bright}Performance:${this.colors.reset}`);
    console.log(`  Baseline: ${this.formatDuration(baseline.mean)}`);
    console.log(`  Current:  ${this.formatDuration(current.mean)}`);
    console.log(`  Change:   ${color}${symbol} ${Math.abs(improvement).toFixed(2)}% ${label}${this.colors.reset}`);
    console.log(`  Significant: ${significant ? this.colors.green + 'Yes' : this.colors.yellow + 'No'}${this.colors.reset}`);

    console.log(`\n${this.colors.bright}Memory:${this.colors.reset}`);
    const memColor = comparison.memoryDelta > 0 ? this.colors.red : this.colors.green;
    const memSymbol = comparison.memoryDelta > 0 ? '↑' : '↓';
    console.log(`  Delta:    ${memColor}${memSymbol} ${this.formatBytes(Math.abs(comparison.memoryDelta))}${this.colors.reset}`);
  }

  /**
   * Report regression alerts
   */
  reportRegressions(alerts: RegressionAlert[]): void {
    if (alerts.length === 0) {
      console.log(`\n${this.colors.green}✓ No regressions detected${this.colors.reset}`);
      return;
    }

    console.log(`\n${this.colors.bright}${this.colors.red}⚠ Regression Alerts (${alerts.length})${this.colors.reset}`);
    console.log(`${'='.repeat(60)}`);

    for (const alert of alerts) {
      const severityColor = this.getSeverityColor(alert.severity);
      console.log(`\n${severityColor}[${alert.severity.toUpperCase()}]${this.colors.reset} ${alert.benchmark}`);
      console.log(`  Metric:    ${alert.metric}`);
      console.log(`  Threshold: ${alert.threshold.toFixed(2)}%`);
      console.log(`  Actual:    ${alert.actual.toFixed(2)}%`);
      console.log(`  Time:      ${new Date(alert.timestamp).toISOString()}`);
    }
  }

  /**
   * Report multiple results in table format
   */
  reportTable(results: BenchmarkResult[]): void {
    console.log(`\n${this.colors.bright}Benchmark Results${this.colors.reset}`);
    console.log(`${'='.repeat(100)}`);

    // Header
    console.log(
      this.pad('Name', 30) +
      this.pad('Mean', 12) +
      this.pad('P95', 12) +
      this.pad('Throughput', 15) +
      this.pad('Memory', 15)
    );
    console.log('-'.repeat(100));

    // Rows
    for (const result of results) {
      console.log(
        this.pad(result.name, 30) +
        this.pad(this.formatDuration(result.mean), 12) +
        this.pad(this.formatDuration(result.p95), 12) +
        this.pad(`${result.throughput.toFixed(2)} ops/s`, 15) +
        this.pad(this.formatBytes(result.memory.heapUsed), 15)
      );
    }
  }

  private formatDuration(ms: number): string {
    if (ms < 1) {
      return `${(ms * 1000).toFixed(2)}μs`;
    } else if (ms < 1000) {
      return `${ms.toFixed(2)}ms`;
    } else {
      return `${(ms / 1000).toFixed(2)}s`;
    }
  }

  private formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(2)}${units[unitIndex]}`;
  }

  private getSeverityColor(severity: string): string {
    switch (severity) {
      case 'critical': return this.colors.red + this.colors.bright;
      case 'high': return this.colors.red;
      case 'medium': return this.colors.yellow;
      case 'low': return this.colors.blue;
      default: return this.colors.reset;
    }
  }

  private pad(str: string, length: number): string {
    return str.padEnd(length, ' ');
  }
}
