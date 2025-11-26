#!/usr/bin/env node

/**
 * CLI for neural-trader benchmarking framework
 */

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { BenchmarkRunner } from './runners/benchmark-runner';
import { ComparisonRunner } from './runners/comparison-runner';
import { RegressionDetector } from './analyzers/regression-detector';
import { ConsoleReporter } from './reporters/console-reporter';
import { JSONReporter } from './reporters/json-reporter';
import { HTMLReporter } from './reporters/html-reporter';
import { AgentDBHistory } from './history/agentdb-history';
import { MemoryLeakDetector } from './detectors/memory-leak-detector';
import * as path from 'path';

const reporter = new ConsoleReporter();
const jsonReporter = new JSONReporter();
const htmlReporter = new HTMLReporter();

yargs(hideBin(process.argv))
  .command(
    'run [suite]',
    'Run benchmark suite',
    (yargs) => {
      return yargs
        .positional('suite', {
          describe: 'Benchmark suite file path',
          type: 'string'
        })
        .option('iterations', {
          alias: 'i',
          type: 'number',
          description: 'Number of iterations',
          default: 100
        })
        .option('warmup', {
          alias: 'w',
          type: 'number',
          description: 'Warmup iterations',
          default: 10
        })
        .option('output', {
          alias: 'o',
          type: 'string',
          description: 'Output format (console|json|html)',
          default: 'console'
        })
        .option('file', {
          alias: 'f',
          type: 'string',
          description: 'Output file path'
        })
        .option('all', {
          alias: 'a',
          type: 'boolean',
          description: 'Run all benchmark suites',
          default: false
        });
    },
    async (argv) => {
      console.log('Running benchmarks...\n');

      // Load benchmark suite
      const suitePath = argv.suite
        ? path.resolve(process.cwd(), argv.suite)
        : path.resolve(process.cwd(), 'benchmarks');

      try {
        const suite = await import(suitePath);
        const benchmarks = suite.default || suite.benchmarks;

        const results = [];
        for (const bench of benchmarks) {
          const runner = new BenchmarkRunner({
            name: bench.name,
            iterations: argv.iterations as number,
            warmupIterations: argv.warmup as number
          });

          const result = await runner.run(bench.fn);
          results.push(result);

          if (argv.output === 'console') {
            reporter.report(result);
          }
        }

        // Export results
        if (argv.file) {
          if (argv.output === 'json') {
            const report = jsonReporter.generate(results);
            await jsonReporter.writeToFile(report, argv.file as string);
            console.log(`\nResults exported to: ${argv.file}`);
          } else if (argv.output === 'html') {
            const html = htmlReporter.generate(results);
            await htmlReporter.writeToFile(html, argv.file as string);
            console.log(`\nResults exported to: ${argv.file}`);
          }
        }

        if (argv.output === 'console') {
          reporter.reportTable(results);
        }
      } catch (error) {
        console.error('Error running benchmarks:', error);
        process.exit(1);
      }
    }
  )
  .command(
    'compare <baseline> <current>',
    'Compare two implementations',
    (yargs) => {
      return yargs
        .positional('baseline', {
          describe: 'Baseline implementation',
          type: 'string'
        })
        .positional('current', {
          describe: 'Current implementation',
          type: 'string'
        })
        .option('iterations', {
          alias: 'i',
          type: 'number',
          description: 'Number of iterations',
          default: 100
        });
    },
    async (argv) => {
      console.log('Comparing implementations...\n');

      try {
        const baselineMod = await import(path.resolve(process.cwd(), argv.baseline as string));
        const currentMod = await import(path.resolve(process.cwd(), argv.current as string));

        const runner = new ComparisonRunner();
        const result = await runner.compare(
          baselineMod.default,
          currentMod.default,
          {
            name: 'Comparison',
            iterations: argv.iterations as number
          }
        );

        reporter.reportComparison(result);
      } catch (error) {
        console.error('Error comparing implementations:', error);
        process.exit(1);
      }
    }
  )
  .command(
    'detect-regression [benchmark]',
    'Detect performance regressions',
    (yargs) => {
      return yargs
        .positional('benchmark', {
          describe: 'Benchmark name',
          type: 'string'
        })
        .option('threshold', {
          alias: 't',
          type: 'number',
          description: 'Regression threshold (%)',
          default: 10
        });
    },
    async (argv) => {
      console.log('Detecting regressions...\n');

      // This would integrate with AgentDB history
      console.log('Regression detection requires AgentDB history tracking.');
      console.log('Use "neural-bench history" to view historical data.');
    }
  )
  .command(
    'memory-leak [suite]',
    'Detect memory leaks',
    (yargs) => {
      return yargs
        .positional('suite', {
          describe: 'Benchmark suite file path',
          type: 'string'
        })
        .option('iterations', {
          alias: 'i',
          type: 'number',
          description: 'Number of iterations',
          default: 100
        })
        .option('threshold', {
          alias: 't',
          type: 'number',
          description: 'Leak threshold (bytes per iteration)',
          default: 1024
        });
    },
    async (argv) => {
      console.log('Detecting memory leaks...\n');

      const detector = new MemoryLeakDetector();

      try {
        const suitePath = path.resolve(process.cwd(), argv.suite as string);
        const suite = await import(suitePath);
        const benchmarks = suite.default || suite.benchmarks;

        for (const bench of benchmarks) {
          console.log(`Testing: ${bench.name}...`);

          const result = await detector.test(
            bench.fn,
            argv.iterations as number
          );

          if (result.leaked) {
            console.log(`❌ Memory leak detected!`);
            console.log(`   Leak rate: ${(result.leakRate / 1024).toFixed(2)} KB/iteration`);
            console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);

            const prediction = detector.predictMemoryUsage(1000);
            console.log(`   Predicted memory after 1000 iterations: ${(prediction.predicted / 1024 / 1024).toFixed(2)} MB`);
            if (prediction.timeToOOM !== Infinity) {
              console.log(`   Estimated time to OOM: ${prediction.timeToOOM} iterations`);
            }
          } else {
            console.log(`✓ No memory leak detected`);
            console.log(`   Trend: ${result.analysis.trend}`);
          }
          console.log('');
        }
      } catch (error) {
        console.error('Error detecting memory leaks:', error);
        process.exit(1);
      }
    }
  )
  .command(
    'history [benchmark]',
    'View benchmark history',
    (yargs) => {
      return yargs
        .positional('benchmark', {
          describe: 'Benchmark name',
          type: 'string'
        })
        .option('limit', {
          alias: 'l',
          type: 'number',
          description: 'Number of results to show',
          default: 10
        });
    },
    async () => {
      console.log('History tracking requires AgentDB integration.');
      console.log('See documentation for setup instructions.');
    }
  )
  .option('verbose', {
    alias: 'v',
    type: 'boolean',
    description: 'Run with verbose logging'
  })
  .demandCommand(1, 'You need at least one command')
  .help()
  .argv;
