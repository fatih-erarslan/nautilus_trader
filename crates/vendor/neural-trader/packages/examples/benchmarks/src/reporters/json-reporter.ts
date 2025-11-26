/**
 * JSON reporter for benchmark results
 */

import { promises as fs } from 'fs';
import { BenchmarkResult, ComparisonResult, RegressionAlert } from '../types';

export interface JSONReport {
  metadata: {
    generated: number;
    version: string;
    platform: string;
    node: string;
  };
  benchmarks: BenchmarkResult[];
  comparisons?: ComparisonResult[];
  regressions?: RegressionAlert[];
}

export class JSONReporter {
  /**
   * Generate JSON report
   */
  generate(
    results: BenchmarkResult[],
    options?: {
      comparisons?: ComparisonResult[];
      regressions?: RegressionAlert[];
    }
  ): JSONReport {
    return {
      metadata: {
        generated: Date.now(),
        version: process.version,
        platform: process.platform,
        node: process.version
      },
      benchmarks: results,
      comparisons: options?.comparisons,
      regressions: options?.regressions
    };
  }

  /**
   * Write report to file
   */
  async writeToFile(report: JSONReport, filePath: string): Promise<void> {
    const json = JSON.stringify(report, null, 2);
    await fs.writeFile(filePath, json, 'utf-8');
  }

  /**
   * Read report from file
   */
  async readFromFile(filePath: string): Promise<JSONReport> {
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  }
}
