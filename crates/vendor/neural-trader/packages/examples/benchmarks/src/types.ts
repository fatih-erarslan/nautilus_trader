/**
 * Types for benchmarking framework
 */

export interface BenchmarkConfig {
  name: string;
  description?: string;
  iterations?: number;
  warmupIterations?: number;
  timeout?: number;
  memoryLimit?: number;
  compareBaseline?: boolean;
}

export interface BenchmarkResult {
  name: string;
  iterations: number;
  duration: number;
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
  p95: number;
  p99: number;
  throughput: number;
  memory: MemoryStats;
  timestamp: number;
}

export interface MemoryStats {
  heapUsed: number;
  heapTotal: number;
  external: number;
  rss: number;
  arrayBuffers: number;
  peak: number;
  leaked?: boolean;
  leakRate?: number;
}

export interface ComparisonResult {
  baseline: BenchmarkResult;
  current: BenchmarkResult;
  improvement: number;
  pValue: number;
  significant: boolean;
  faster: boolean;
  memoryDelta: number;
}

export interface RegressionAlert {
  benchmark: string;
  metric: string;
  threshold: number;
  actual: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: number;
}

export interface PerformanceHistory {
  benchmark: string;
  results: BenchmarkResult[];
  trend: 'improving' | 'degrading' | 'stable';
  firstRun: number;
  lastRun: number;
}

export interface BenchmarkSuite {
  name: string;
  benchmarks: Array<() => Promise<any>>;
  config?: Partial<BenchmarkConfig>;
}

export interface StatisticalSummary {
  mean: number;
  median: number;
  stdDev: number;
  variance: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  p95: number;
  p99: number;
  iqr: number;
  outliers: number[];
}
