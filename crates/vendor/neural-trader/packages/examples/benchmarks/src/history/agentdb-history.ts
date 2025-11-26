/**
 * AgentDB-backed performance history tracking
 */

import { BenchmarkResult, PerformanceHistory } from '../types';

export interface HistoryQuery {
  benchmark?: string;
  startDate?: number;
  endDate?: number;
  limit?: number;
}

/**
 * Track benchmark history in AgentDB
 */
export class AgentDBHistory {
  private db: any; // AgentDB instance
  private namespace: string;

  constructor(db: any, namespace = 'benchmarks') {
    this.db = db;
    this.namespace = namespace;
  }

  /**
   * Store benchmark result
   */
  async store(result: BenchmarkResult): Promise<void> {
    const key = `${this.namespace}:${result.name}:${result.timestamp}`;

    await this.db.set(key, {
      ...result,
      metadata: {
        stored: Date.now(),
        version: process.version,
        platform: process.platform,
        arch: process.arch
      }
    });

    // Also store in vector database for similarity search
    const vector = this.resultToVector(result);
    await this.db.addVector(key, vector, {
      benchmark: result.name,
      timestamp: result.timestamp
    });
  }

  /**
   * Get benchmark history
   */
  async getHistory(benchmarkName: string, limit = 100): Promise<PerformanceHistory> {
    const pattern = `${this.namespace}:${benchmarkName}:*`;
    const keys = await this.db.keys(pattern);

    // Sort by timestamp
    const sortedKeys = keys.sort().slice(-limit);

    const results: BenchmarkResult[] = [];
    for (const key of sortedKeys) {
      const result = await this.db.get(key);
      if (result) {
        results.push(result);
      }
    }

    if (results.length === 0) {
      return {
        benchmark: benchmarkName,
        results: [],
        trend: 'stable',
        firstRun: 0,
        lastRun: 0
      };
    }

    // Analyze trend
    const trend = this.analyzeTrend(results);

    return {
      benchmark: benchmarkName,
      results,
      trend,
      firstRun: results[0].timestamp,
      lastRun: results[results.length - 1].timestamp
    };
  }

  /**
   * Query benchmark results
   */
  async query(query: HistoryQuery): Promise<BenchmarkResult[]> {
    const pattern = query.benchmark
      ? `${this.namespace}:${query.benchmark}:*`
      : `${this.namespace}:*`;

    let keys = await this.db.keys(pattern);

    // Filter by date range
    if (query.startDate || query.endDate) {
      keys = keys.filter((key: string) => {
        const timestamp = parseInt(key.split(':')[2]);
        if (query.startDate && timestamp < query.startDate) return false;
        if (query.endDate && timestamp > query.endDate) return false;
        return true;
      });
    }

    // Sort and limit
    keys = keys.sort().slice(0, query.limit || 100);

    const results: BenchmarkResult[] = [];
    for (const key of keys) {
      const result = await this.db.get(key);
      if (result) {
        results.push(result);
      }
    }

    return results;
  }

  /**
   * Find similar benchmarks using vector similarity
   */
  async findSimilar(
    result: BenchmarkResult,
    k = 5
  ): Promise<Array<{ result: BenchmarkResult; similarity: number }>> {
    const vector = this.resultToVector(result);

    const similar = await this.db.queryVectors(vector, k, {
      benchmark: { $ne: result.name } // Exclude same benchmark
    });

    return similar.map((item: any) => ({
      result: item.data,
      similarity: item.similarity
    }));
  }

  /**
   * Get baseline result (median of last N runs)
   */
  async getBaseline(benchmarkName: string, runs = 10): Promise<BenchmarkResult | null> {
    const history = await this.getHistory(benchmarkName, runs);

    if (history.results.length === 0) {
      return null;
    }

    // Calculate median result
    const sortedByMean = [...history.results].sort((a, b) => a.mean - b.mean);
    const medianIndex = Math.floor(sortedByMean.length / 2);

    return sortedByMean[medianIndex];
  }

  /**
   * Clear history for a benchmark
   */
  async clear(benchmarkName?: string): Promise<void> {
    const pattern = benchmarkName
      ? `${this.namespace}:${benchmarkName}:*`
      : `${this.namespace}:*`;

    const keys = await this.db.keys(pattern);

    for (const key of keys) {
      await this.db.delete(key);
    }
  }

  /**
   * Export history to JSON
   */
  async export(benchmarkName?: string): Promise<any> {
    const results = await this.query({ benchmark: benchmarkName });

    return {
      exported: Date.now(),
      count: results.length,
      results
    };
  }

  private resultToVector(result: BenchmarkResult): number[] {
    // Convert benchmark result to vector for similarity search
    return [
      result.mean / 1000,           // Normalize mean time
      result.median / 1000,         // Normalize median time
      result.stdDev / 1000,         // Normalize std dev
      result.p95 / 1000,            // Normalize p95
      result.throughput / 1000,     // Normalize throughput
      result.memory.heapUsed / 1e6, // Normalize memory (MB)
      result.iterations / 100       // Normalize iterations
    ];
  }

  private analyzeTrend(results: BenchmarkResult[]): 'improving' | 'degrading' | 'stable' {
    if (results.length < 3) {
      return 'stable';
    }

    const means = results.map(r => r.mean);
    const recentMean = means.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const olderMean = means.slice(0, -3).reduce((a, b) => a + b, 0) / (means.length - 3);

    const change = ((recentMean - olderMean) / olderMean) * 100;

    if (change < -5) return 'improving';  // Lower time is better
    if (change > 5) return 'degrading';
    return 'stable';
  }
}
