/**
 * HTML reporter for benchmark results
 */

import { promises as fs } from 'fs';
import { BenchmarkResult, ComparisonResult } from '../types';

export class HTMLReporter {
  /**
   * Generate HTML report
   */
  generate(
    results: BenchmarkResult[],
    options?: {
      title?: string;
      comparisons?: ComparisonResult[];
    }
  ): string {
    const title = options?.title || 'Benchmark Results';

    return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>
    ${this.getCSS()}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
  <div class="container">
    <h1>${title}</h1>
    <div class="metadata">
      <p>Generated: ${new Date().toLocaleString()}</p>
      <p>Platform: ${process.platform} | Node: ${process.version}</p>
    </div>

    <h2>Benchmark Results</h2>
    ${this.generateTable(results)}

    <h2>Performance Charts</h2>
    ${this.generateCharts(results)}

    ${options?.comparisons ? this.generateComparisons(options.comparisons) : ''}
  </div>

  <script>
    ${this.getChartJS(results)}
  </script>
</body>
</html>
    `;
  }

  /**
   * Write HTML report to file
   */
  async writeToFile(html: string, filePath: string): Promise<void> {
    await fs.writeFile(filePath, html, 'utf-8');
  }

  private generateTable(results: BenchmarkResult[]): string {
    return `
      <table class="results-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>Mean</th>
            <th>Median</th>
            <th>P95</th>
            <th>P99</th>
            <th>Throughput</th>
            <th>Memory (Peak)</th>
          </tr>
        </thead>
        <tbody>
          ${results.map(r => `
            <tr>
              <td>${r.name}</td>
              <td>${r.mean.toFixed(2)}ms</td>
              <td>${r.median.toFixed(2)}ms</td>
              <td>${r.p95.toFixed(2)}ms</td>
              <td>${r.p99.toFixed(2)}ms</td>
              <td>${r.throughput.toFixed(2)} ops/s</td>
              <td>${this.formatBytes(r.memory.peak)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }

  private generateCharts(results: BenchmarkResult[]): string {
    return `
      <div class="charts">
        <div class="chart-container">
          <canvas id="meanChart"></canvas>
        </div>
        <div class="chart-container">
          <canvas id="throughputChart"></canvas>
        </div>
        <div class="chart-container">
          <canvas id="memoryChart"></canvas>
        </div>
      </div>
    `;
  }

  private generateComparisons(comparisons: ComparisonResult[]): string {
    return `
      <h2>Comparisons</h2>
      <div class="comparisons">
        ${comparisons.map(c => `
          <div class="comparison ${c.faster ? 'faster' : 'slower'}">
            <h3>${c.current.name}</h3>
            <p class="improvement">
              ${c.faster ? '↓' : '↑'} ${Math.abs(c.improvement).toFixed(2)}%
              ${c.faster ? 'faster' : 'slower'}
            </p>
            <p class="significance ${c.significant ? 'significant' : 'not-significant'}">
              ${c.significant ? 'Statistically significant' : 'Not statistically significant'}
            </p>
          </div>
        `).join('')}
      </div>
    `;
  }

  private getCSS(): string {
    return `
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
      .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
      h1 { color: #333; margin-bottom: 10px; }
      h2 { color: #555; margin: 30px 0 15px; padding-bottom: 10px; border-bottom: 2px solid #eee; }
      .metadata { color: #777; font-size: 14px; margin-bottom: 30px; }
      .results-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
      .results-table th, .results-table td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
      .results-table th { background: #f8f9fa; font-weight: 600; color: #555; }
      .results-table tbody tr:hover { background: #f8f9fa; }
      .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
      .chart-container { background: #f8f9fa; padding: 20px; border-radius: 4px; }
      .comparisons { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
      .comparison { padding: 20px; border-radius: 4px; border: 2px solid #ddd; }
      .comparison.faster { border-color: #28a745; background: #f0f9f3; }
      .comparison.slower { border-color: #dc3545; background: #fef0f1; }
      .improvement { font-size: 24px; font-weight: bold; margin: 10px 0; }
      .faster .improvement { color: #28a745; }
      .slower .improvement { color: #dc3545; }
      .significance { font-size: 14px; color: #666; }
      .significant { font-weight: 600; color: #28a745; }
    `;
  }

  private getChartJS(results: BenchmarkResult[]): string {
    const labels = results.map(r => r.name);
    const means = results.map(r => r.mean);
    const throughputs = results.map(r => r.throughput);
    const memories = results.map(r => r.memory.peak / 1024 / 1024); // Convert to MB

    return `
      // Mean execution time chart
      new Chart(document.getElementById('meanChart'), {
        type: 'bar',
        data: {
          labels: ${JSON.stringify(labels)},
          datasets: [{
            label: 'Mean Execution Time (ms)',
            data: ${JSON.stringify(means)},
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: { y: { beginAtZero: true } }
        }
      });

      // Throughput chart
      new Chart(document.getElementById('throughputChart'), {
        type: 'bar',
        data: {
          labels: ${JSON.stringify(labels)},
          datasets: [{
            label: 'Throughput (ops/sec)',
            data: ${JSON.stringify(throughputs)},
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: { y: { beginAtZero: true } }
        }
      });

      // Memory usage chart
      new Chart(document.getElementById('memoryChart'), {
        type: 'bar',
        data: {
          labels: ${JSON.stringify(labels)},
          datasets: [{
            label: 'Peak Memory (MB)',
            data: ${JSON.stringify(memories)},
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          scales: { y: { beginAtZero: true } }
        }
      });
    `;
  }

  private formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(2)} ${units[unitIndex]}`;
  }
}
