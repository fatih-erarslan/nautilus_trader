/**
 * @neural-trader/benchoptimizer - High-performance package benchmarking and optimization
 *
 * TypeScript definitions for Rust/NAPI bindings
 */

/**
 * Benchmark configuration options
 */
export interface BenchConfig {
  /** Number of warmup iterations before measurement */
  warmupIterations?: number;
  /** Number of measurement iterations */
  measureIterations?: number;
  /** Maximum time to spend on benchmarking (seconds) */
  timeout?: number;
  /** Enable memory profiling */
  profileMemory?: boolean;
  /** Enable CPU profiling */
  profileCpu?: boolean;
  /** Custom environment variables */
  env?: Record<string, string>;
}

/**
 * Benchmark result statistics
 */
export interface BenchmarkResult {
  /** Package name or identifier */
  packageName: string;
  /** Mean execution time (milliseconds) */
  mean: number;
  /** Median execution time (milliseconds) */
  median: number;
  /** Standard deviation */
  stddev: number;
  /** 95th percentile */
  p95: number;
  /** 99th percentile */
  p99: number;
  /** Minimum execution time */
  min: number;
  /** Maximum execution time */
  max: number;
  /** Peak memory usage (bytes) */
  memoryUsage: number;
  /** Bundle size (bytes) */
  bundleSize: number;
  /** Number of iterations performed */
  iterations: number;
  /** Total benchmark duration (seconds) */
  duration: number;
  /** Timestamp of benchmark */
  timestamp: string;
}

/**
 * Package validation error or warning
 */
export interface ValidationIssue {
  /** Issue severity level */
  severity: 'error' | 'warning' | 'info';
  /** Issue category */
  category: 'structure' | 'dependencies' | 'configuration' | 'security' | 'performance';
  /** Human-readable description */
  message: string;
  /** File or location where issue was found */
  location?: string;
  /** Suggested fix */
  suggestion?: string;
}

/**
 * Package validation result
 */
export interface ValidationResult {
  /** Overall validation status */
  valid: boolean;
  /** List of errors found */
  errors: ValidationIssue[];
  /** List of warnings found */
  warnings: ValidationIssue[];
  /** List of informational messages */
  info: ValidationIssue[];
  /** Package metadata */
  metadata: {
    name: string;
    version: string;
    hasTests: boolean;
    hasDocs: boolean;
    hasBenchmarks: boolean;
  };
}

/**
 * Optimization suggestion with impact estimate
 */
export interface OptimizationSuggestion {
  /** Optimization category */
  type: 'performance' | 'bundle' | 'dependency' | 'memory' | 'build';
  /** Suggestion severity/priority */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Human-readable description */
  description: string;
  /** Estimated improvement percentage */
  estimatedImprovement: string;
  /** Current metric value */
  currentValue?: string;
  /** Target metric value */
  targetValue?: string;
  /** Implementation steps */
  steps?: string[];
  /** Related documentation URLs */
  references?: string[];
}

/**
 * Comparison result between two benchmark runs
 */
export interface ComparisonResult {
  /** Baseline benchmark name */
  baseline: string;
  /** Current benchmark name */
  current: string;
  /** Performance change percentage (positive = improvement) */
  performanceChange: number;
  /** Memory change percentage (negative = improvement) */
  memoryChange: number;
  /** Bundle size change percentage (negative = improvement) */
  bundleSizeChange: number;
  /** Statistical significance (0-1) */
  significance: number;
  /** Whether the change is statistically significant */
  isSignificant: boolean;
}

/**
 * Main BenchOptimizer class for programmatic usage
 */
export class BenchOptimizer {
  /**
   * Create a new BenchOptimizer instance
   * @param config Optional configuration
   */
  constructor(config?: BenchConfig);

  /**
   * Run benchmark on a package
   * @param packagePath Path to package directory or package.json
   * @returns Benchmark results
   */
  benchmark(packagePath: string): Promise<BenchmarkResult>;

  /**
   * Validate a package structure and configuration
   * @param packagePath Path to package directory
   * @returns Validation results
   */
  validate(packagePath: string): Promise<ValidationResult>;

  /**
   * Analyze package and suggest optimizations
   * @param packagePath Path to package directory
   * @returns List of optimization suggestions
   */
  optimize(packagePath: string): Promise<OptimizationSuggestion[]>;

  /**
   * Compare two benchmark results
   * @param baseline Baseline benchmark result
   * @param current Current benchmark result
   * @returns Comparison analysis
   */
  compare(baseline: BenchmarkResult, current: BenchmarkResult): ComparisonResult;

  /**
   * Update configuration
   * @param config New configuration options
   */
  setConfig(config: BenchConfig): void;
}

/**
 * Benchmark a single package (convenience function)
 * @param packagePath Path to package directory
 * @param iterations Number of iterations (default: 100)
 * @returns Benchmark results
 */
export function benchmarkPackage(
  packagePath: string,
  iterations?: number
): Promise<BenchmarkResult>;

/**
 * Validate a package (convenience function)
 * @param packagePath Path to package directory
 * @returns Validation results
 */
export function validatePackage(packagePath: string): Promise<ValidationResult>;

/**
 * Analyze and suggest optimizations (convenience function)
 * @param packagePath Path to package directory
 * @returns Optimization suggestions
 */
export function optimizePackage(
  packagePath: string
): Promise<OptimizationSuggestion[]>;

/**
 * Benchmark all packages in a directory
 * @param packagesDir Directory containing multiple packages
 * @param options Optional benchmark configuration
 * @returns Array of benchmark results
 */
export function benchmarkAll(
  packagesDir: string,
  options?: BenchConfig
): Promise<BenchmarkResult[]>;

/**
 * Generate a formatted report from benchmark results
 * @param results Benchmark results to include in report
 * @param format Output format (default: 'json')
 * @returns Formatted report string
 */
export function generateReport(
  results: BenchmarkResult | BenchmarkResult[],
  format?: 'json' | 'markdown' | 'html' | 'csv'
): string;

/**
 * Package version information
 */
export const version: string;
export const nativeVersion: string;
