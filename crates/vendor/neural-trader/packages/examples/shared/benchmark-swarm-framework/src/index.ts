/**
 * Benchmark Swarm Framework
 *
 * Generic swarm orchestration for parallel variation exploration,
 * performance benchmarking, and parameter optimization.
 *
 * @packageDocumentation
 */

export { SwarmCoordinator } from './swarm-coordinator';
export type {
  SwarmConfig,
  TaskVariation,
  SwarmResult,
  AgentTask,
} from './swarm-coordinator';

export { BenchmarkRunner } from './benchmark-runner';
export type {
  BenchmarkConfig,
  BenchmarkSuite,
  BenchmarkReport,
} from './benchmark-runner';

export { Optimizer } from './optimizer';
export type {
  OptimizationStrategy,
  OptimizationConfig,
  ParameterSpace,
  OptimizationResult,
} from './optimizer';
