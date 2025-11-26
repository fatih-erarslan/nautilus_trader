# nt-benchoptimizer

High-performance benchmarking and optimization toolkit for Neural Trader packages.

## Features

- 🚀 **Multi-threaded Benchmarking**: Parallel execution using rayon for maximum performance
- 📊 **Statistical Analysis**: Mean, median, standard deviation, p95, p99 percentiles
- 💾 **Memory Profiling**: Track memory usage and detect potential leaks
- 📦 **Bundle Analysis**: Analyze bundle sizes and identify optimization opportunities
- ✅ **Package Validation**: Validate dependencies, TypeScript definitions, and NAPI bindings
- 🎯 **Optimization Suggestions**: Get actionable recommendations with estimated impact
- 🔧 **SIMD Optimizations**: Zero-cost abstractions with SIMD acceleration where possible

## Installation

```bash
npm install @neural-trader/benchoptimizer
```

## Usage

### Benchmark a Package

```typescript
import { benchmarkPackage } from '@neural-trader/benchoptimizer';

const result = await benchmarkPackage('/path/to/package', {
  iterations: 100,
  warmupIterations: 10,
  includeMemoryProfiling: true,
  includeBundleAnalysis: true,
  parallel: true,
});

console.log(`Execution time: ${result.executionTimeMs}ms`);
console.log(`Memory usage: ${result.memoryUsageMb}MB`);
console.log(`Bundle size: ${result.bundleSizeKb}KB`);
```

### Validate a Package

```typescript
import { validatePackage } from '@neural-trader/benchoptimizer';

const report = await validatePackage('/path/to/package', {
  checkDependencies: true,
  checkTypescript: true,
  checkNapiBindings: true,
  strictMode: true,
});

if (!report.isValid) {
  console.error('Validation failed:', report.errors);
}
```

### Optimize a Package

```typescript
import { optimizePackage } from '@neural-trader/benchoptimizer';

const report = await optimizePackage('/path/to/package', {
  analyzeBundle: true,
  analyzeDependencies: true,
  analyzeCodeSplitting: true,
  suggestRefactoring: true,
});

console.log(`Potential savings: ${report.potentialSavingsKb}KB`);
console.log(`Estimated performance gain: ${report.estimatedPerformanceGain}%`);

for (const suggestion of report.suggestions) {
  console.log(`[${suggestion.severity}] ${suggestion.description}`);
  console.log(`Impact: ${suggestion.impact}`);
}
```

### Benchmark All Packages

```typescript
import { benchmarkAll } from '@neural-trader/benchoptimizer';

const results = await benchmarkAll('/path/to/workspace', {
  iterations: 50,
  parallel: true,
});

for (const result of results) {
  console.log(`${result.packageName}: ${result.executionTimeMs}ms`);
}
```

### Generate Report

```typescript
import {
  benchmarkAll,
  validatePackage,
  optimizePackage,
  generateReport,
} from '@neural-trader/benchoptimizer';

// Collect data
const benchmarks = await benchmarkAll('/workspace');
const validations = await Promise.all(
  packages.map(pkg => validatePackage(pkg))
);
const optimizations = await Promise.all(
  packages.map(pkg => optimizePackage(pkg))
);

// Generate report
const reportPath = await generateReport(
  benchmarks,
  validations,
  optimizations,
  './reports/benchmark-report.json',
  'json' // or 'markdown' or 'html'
);

console.log(`Report generated: ${reportPath}`);
```

### Compare Benchmarks

```typescript
import { benchmarkPackage, compareBenchmarks } from '@neural-trader/benchoptimizer';

const baseline = await benchmarkPackage('/path/to/package');
// Make changes...
const current = await benchmarkPackage('/path/to/package');

const comparison = compareBenchmarks(baseline, current);

if (comparison.regressionDetected) {
  console.warn('Performance regression detected!');
  console.log(`Performance delta: ${comparison.performanceDelta}%`);
}
```

### Get System Info

```typescript
import { getSystemInfo } from '@neural-trader/benchoptimizer';

const info = getSystemInfo();
console.log(`CPU cores: ${info.cpuCount}`);
console.log(`Total memory: ${info.totalMemoryGb}GB`);
console.log(`OS: ${info.osVersion}`);
console.log(`Architecture: ${info.architecture}`);
```

## API Reference

### Functions

#### `initialize(config?: string): Promise<void>`

Initialize the benchmarker with optional configuration.

#### `benchmarkPackage(packagePath: string, options?: BenchmarkOptions): Promise<BenchmarkResult>`

Benchmark a single package's performance.

#### `validatePackage(packagePath: string, options?: ValidationOptions): Promise<ValidationReport>`

Validate package structure and dependencies.

#### `optimizePackage(packagePath: string, options?: OptimizationOptions): Promise<OptimizationReport>`

Analyze and suggest optimizations.

#### `benchmarkAll(workspacePath: string, options?: BenchmarkOptions): Promise<BenchmarkResult[]>`

Benchmark all packages in parallel.

#### `generateReport(...): Promise<string>`

Generate comprehensive performance report.

#### `compareBenchmarks(baseline: BenchmarkResult, current: BenchmarkResult): ComparisonReport`

Compare two benchmark results.

#### `getSystemInfo(): SystemInfo`

Get system capabilities and configuration.

## Performance

- **Zero-cost abstractions**: Rust's ownership system ensures memory safety without runtime overhead
- **SIMD optimizations**: Automatic vectorization for numerical computations
- **Multi-threaded execution**: Parallel benchmarking using rayon thread pool
- **Minimal allocations**: Careful memory management for optimal performance
- **Async I/O**: Non-blocking file operations using tokio

## Development

```bash
# Build the crate
cargo build --release

# Run tests
cargo test

# Run benchmarks (requires 'profiling' feature)
cargo bench --features profiling

# Build for Node.js
npm run build
```

## License

MIT

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
