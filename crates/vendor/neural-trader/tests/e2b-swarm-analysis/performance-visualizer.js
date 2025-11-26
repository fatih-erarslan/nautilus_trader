#!/usr/bin/env node

/**
 * Performance Visualization Tool
 *
 * Creates visual charts and comparisons from benchmark data
 *
 * @module tests/e2b-swarm-analysis/performance-visualizer
 */

const fs = require('fs').promises;
const path = require('path');

/**
 * ASCII Chart Generator
 */
class ASCIIChartGenerator {
  static generateBarChart(data, options = {}) {
    const {
      title = 'Bar Chart',
      width = 60,
      maxValue = Math.max(...data.map(d => d.value))
    } = options;

    const lines = [];

    lines.push(title);
    lines.push('='.repeat(title.length));
    lines.push('');

    data.forEach(item => {
      const barLength = Math.floor((item.value / maxValue) * width);
      const bar = '█'.repeat(barLength);
      const label = item.label.padEnd(20);
      const value = item.value.toFixed(2).padStart(10);

      lines.push(`${label} ${bar} ${value}${item.unit || ''}`);
    });

    return lines.join('\n');
  }

  static generateLineChart(data, options = {}) {
    const {
      title = 'Line Chart',
      height = 15,
      width = 60
    } = options;

    const lines = [];
    lines.push(title);
    lines.push('='.repeat(title.length));
    lines.push('');

    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    // Create grid
    const grid = Array(height).fill(null).map(() => Array(width).fill(' '));

    // Plot points
    data.forEach((point, index) => {
      const x = Math.floor((index / (data.length - 1)) * (width - 1));
      const y = height - 1 - Math.floor(((point.value - min) / range) * (height - 1));

      if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[y][x] = '●';

        // Draw line to previous point
        if (index > 0) {
          const prevX = Math.floor(((index - 1) / (data.length - 1)) * (width - 1));
          const prevY = height - 1 - Math.floor(((data[index - 1].value - min) / range) * (height - 1));

          // Simple line drawing
          const dx = x - prevX;
          const dy = y - prevY;
          const steps = Math.max(Math.abs(dx), Math.abs(dy));

          for (let step = 0; step <= steps; step++) {
            const lineX = prevX + Math.round((dx / steps) * step);
            const lineY = prevY + Math.round((dy / steps) * step);

            if (lineX >= 0 && lineX < width && lineY >= 0 && lineY < height) {
              if (grid[lineY][lineX] === ' ') {
                grid[lineY][lineX] = '─';
              }
            }
          }
        }
      }
    });

    // Add axis labels
    lines.push(`${max.toFixed(1)} ├${'─'.repeat(width - 2)}┤`);

    grid.forEach(row => {
      lines.push(`        │${row.join('')}│`);
    });

    lines.push(`${min.toFixed(1)} └${'─'.repeat(width - 2)}┘`);
    lines.push(`        ${' '.repeat(Math.floor(width / 2) - 5)}Scale${' '.repeat(Math.floor(width / 2) - 5)}`);

    return lines.join('\n');
  }

  static generateComparisonTable(data, options = {}) {
    const { title = 'Comparison' } = options;

    const lines = [];
    lines.push(title);
    lines.push('='.repeat(title.length));
    lines.push('');

    const headers = Object.keys(data[0]);
    const colWidths = headers.map(h =>
      Math.max(h.length, ...data.map(row => String(row[h]).length))
    );

    // Header
    const headerLine = headers.map((h, i) => h.padEnd(colWidths[i])).join(' | ');
    lines.push(headerLine);
    lines.push(headers.map((_, i) => '-'.repeat(colWidths[i])).join('-|-'));

    // Data rows
    data.forEach(row => {
      const rowLine = headers.map((h, i) => String(row[h]).padEnd(colWidths[i])).join(' | ');
      lines.push(rowLine);
    });

    return lines.join('\n');
  }
}

/**
 * Performance Metrics Visualizer
 */
class PerformanceVisualizer {
  constructor(metricsData) {
    this.metrics = metricsData;
  }

  generateTopologyComparisonChart() {
    const topologies = ['mesh', 'hierarchical', 'ring', 'star'];
    const operations = ['init', 'deploy', 'strategy'];

    const lines = [];
    lines.push('\n# Topology Performance Comparison\n');

    operations.forEach(op => {
      const data = topologies.map(topology => {
        const ops = this.metrics.operations.filter(
          o => o.operation === op && o.topology === topology
        );

        const avgDuration = ops.length > 0
          ? ops.reduce((sum, o) => sum + o.duration, 0) / ops.length
          : 0;

        return {
          label: topology,
          value: avgDuration
        };
      });

      lines.push(`\n## ${op.charAt(0).toUpperCase() + op.slice(1)} Performance\n`);
      lines.push(ASCIIChartGenerator.generateBarChart(data, {
        title: `${op} Operation Time (ms)`,
        unit: 'ms'
      }));
    });

    return lines.join('\n');
  }

  generateScalingEfficiencyChart() {
    const lines = [];
    lines.push('\n# Scaling Efficiency Analysis\n');

    // Extract scaling operations
    const scalingOps = this.metrics.operations.filter(op => op.operation === 'scale');

    if (scalingOps.length === 0) {
      return lines.join('\n') + '\nNo scaling data available\n';
    }

    // Group by agent count delta
    const scalingData = {};

    scalingOps.forEach(op => {
      const from = op.metadata?.from || 0;
      const to = op.metadata?.to || 0;
      const delta = Math.abs(to - from);

      if (!scalingData[delta]) {
        scalingData[delta] = [];
      }

      scalingData[delta].push({
        topology: op.topology,
        time: op.duration,
        throughput: (delta / op.duration) * 1000
      });
    });

    // Create throughput chart
    const throughputData = Object.entries(scalingData).map(([delta, ops]) => {
      const avgThroughput = ops.reduce((sum, o) => sum + o.throughput, 0) / ops.length;

      return {
        label: `±${delta} agents`,
        value: avgThroughput
      };
    });

    lines.push(ASCIIChartGenerator.generateBarChart(throughputData, {
      title: 'Scaling Throughput (agents/sec)',
      unit: ' a/s'
    }));

    return lines.join('\n');
  }

  generatePerformanceTrendChart() {
    const lines = [];
    lines.push('\n# Performance Trends Over Time\n');

    // Group operations by time
    const initOps = this.metrics.operations
      .filter(op => op.operation === 'init')
      .map((op, index) => ({
        label: `Op ${index + 1}`,
        value: op.duration
      }));

    if (initOps.length > 5) {
      lines.push(ASCIIChartGenerator.generateLineChart(initOps, {
        title: 'Initialization Time Trend',
        height: 12,
        width: 60
      }));
    }

    return lines.join('\n');
  }

  generateSuccessRateChart() {
    const lines = [];
    lines.push('\n# Success Rate Analysis\n');

    const topologies = ['mesh', 'hierarchical', 'ring', 'star'];

    const data = topologies.map(topology => {
      const ops = this.metrics.operations.filter(o => o.topology === topology);
      const successCount = ops.filter(o => o.success).length;
      const successRate = ops.length > 0 ? (successCount / ops.length) * 100 : 0;

      return {
        label: topology,
        value: successRate
      };
    });

    lines.push(ASCIIChartGenerator.generateBarChart(data, {
      title: 'Success Rate by Topology (%)',
      unit: '%',
      maxValue: 100
    }));

    return lines.join('\n');
  }

  generateComprehensiveReport() {
    const report = [];

    report.push('╔════════════════════════════════════════════════════════════╗');
    report.push('║     E2B Swarm Performance Visualization Report            ║');
    report.push('╚════════════════════════════════════════════════════════════╝');

    report.push(this.generateTopologyComparisonChart());
    report.push(this.generateScalingEfficiencyChart());
    report.push(this.generateSuccessRateChart());
    report.push(this.generatePerformanceTrendChart());

    // Statistics summary
    report.push('\n# Summary Statistics\n');

    const totalOps = this.metrics.operations.length;
    const successCount = this.metrics.operations.filter(o => o.success).length;
    const avgDuration = this.metrics.operations.reduce((sum, o) => sum + o.duration, 0) / totalOps;

    const summaryData = [
      { Metric: 'Total Operations', Value: totalOps },
      { Metric: 'Successful', Value: successCount },
      { Metric: 'Success Rate', Value: `${((successCount / totalOps) * 100).toFixed(2)}%` },
      { Metric: 'Avg Duration', Value: `${avgDuration.toFixed(2)}ms` }
    ];

    report.push(ASCIIChartGenerator.generateComparisonTable(summaryData, {
      title: 'Overall Performance'
    }));

    return report.join('\n');
  }
}

/**
 * Main execution
 */
async function main() {
  const args = process.argv.slice(2);
  const metricsFile = args[0];

  if (!metricsFile) {
    console.error('Usage: performance-visualizer.js <metrics-file.json>');
    process.exit(1);
  }

  try {
    const metricsData = JSON.parse(await fs.readFile(metricsFile, 'utf8'));
    const visualizer = new PerformanceVisualizer(metricsData);

    const report = visualizer.generateComprehensiveReport();
    console.log(report);

    // Save to file
    const outputFile = metricsFile.replace('.json', '-visualization.txt');
    await fs.writeFile(outputFile, report);

    console.log(`\n\n✅ Visualization saved to: ${outputFile}\n`);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  ASCIIChartGenerator,
  PerformanceVisualizer
};
