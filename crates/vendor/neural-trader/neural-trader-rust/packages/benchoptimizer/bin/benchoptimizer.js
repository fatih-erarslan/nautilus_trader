#!/usr/bin/env node

const path = require('path');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const chalk = require('chalk');
const ora = require('ora');
const Table = require('cli-table3');
const cliProgress = require('cli-progress');
const fs = require('fs-extra');
const path = require('path');

// Import core functionality
const {
  validatePackage,
  validateAll,
  benchmarkPackage,
  benchmarkAll,
  optimizePackage,
  generateReport,
  compareResults
} = require('../index');

// Color configuration
const colors = {
  success: chalk.green,
  error: chalk.red,
  warning: chalk.yellow,
  info: chalk.blue,
  highlight: chalk.cyan,
  dim: chalk.gray
};

// Disable colors if requested
let useColors = true;

// Helper function to format output
function formatOutput(data, format = 'table', noColor = false) {
  if (noColor) {
    Object.keys(colors).forEach(key => {
      colors[key] = (text) => text;
    });
  }

  switch (format) {
    case 'json':
      return JSON.stringify(data, null, 2);

    case 'markdown':
      return formatMarkdown(data);

    case 'html':
      return formatHtml(data);

    case 'table':
    default:
      return formatTable(data);
  }
}

// Format data as table
function formatTable(data) {
  if (Array.isArray(data)) {
    if (data.length === 0) return 'No data';

    const headers = Object.keys(data[0]);
    const table = new Table({
      head: headers.map(h => colors.highlight(h)),
      style: {
        head: [],
        border: []
      }
    });

    data.forEach(row => {
      table.push(headers.map(h => {
        const value = row[h];
        if (typeof value === 'boolean') {
          return value ? colors.success('✓') : colors.error('✗');
        }
        return String(value);
      }));
    });

    return table.toString();
  }

  // Single object
  const table = new Table({
    style: {
      head: [],
      border: []
    }
  });

  Object.entries(data).forEach(([key, value]) => {
    table.push({
      [colors.highlight(key)]: formatValue(value)
    });
  });

  return table.toString();
}

// Format value with appropriate color
function formatValue(value) {
  if (typeof value === 'boolean') {
    return value ? colors.success('✓ true') : colors.error('✗ false');
  }
  if (typeof value === 'number') {
    return colors.info(value.toString());
  }
  if (Array.isArray(value)) {
    return value.join(', ');
  }
  if (typeof value === 'object' && value !== null) {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

// Format data as markdown
function formatMarkdown(data) {
  let md = '# Benchoptimizer Report\n\n';

  if (Array.isArray(data)) {
    const headers = Object.keys(data[0] || {});
    md += `| ${headers.join(' | ')} |\n`;
    md += `| ${headers.map(() => '---').join(' | ')} |\n`;
    data.forEach(row => {
      md += `| ${headers.map(h => row[h]).join(' | ')} |\n`;
    });
  } else {
    Object.entries(data).forEach(([key, value]) => {
      md += `## ${key}\n\n`;
      md += `${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}\n\n`;
    });
  }

  return md;
}

// Format data as HTML
function formatHtml(data) {
  let html = '<!DOCTYPE html>\n<html>\n<head>\n<title>Benchoptimizer Report</title>\n';
  html += '<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background-color:#4CAF50;color:white;}</style>\n';
  html += '</head>\n<body>\n<h1>Benchoptimizer Report</h1>\n';

  if (Array.isArray(data)) {
    const headers = Object.keys(data[0] || {});
    html += '<table>\n<thead>\n<tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr>\n</thead>\n<tbody>\n';
    data.forEach(row => {
      html += '<tr>';
      headers.forEach(h => html += `<td>${row[h]}</td>`);
      html += '</tr>\n';
    });
    html += '</tbody>\n</table>\n';
  } else {
    html += '<dl>\n';
    Object.entries(data).forEach(([key, value]) => {
      html += `<dt><strong>${key}</strong></dt>\n`;
      html += `<dd>${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</dd>\n`;
    });
    html += '</dl>\n';
  }

  html += '</body>\n</html>';
  return html;
}

// Save output to file
async function saveOutput(content, filepath) {
  await fs.ensureFile(filepath);
  await fs.writeFile(filepath, content, 'utf-8');
}

// Validate command handler
async function handleValidate(argv) {
  const spinner = ora({
    text: 'Validating packages...',
    spinner: 'dots'
  }).start();

  try {
    const packages = argv.packages || [];
    const options = {
      fix: argv.fix,
      strict: argv.strict
    };

    let results;
    if (packages.length === 0) {
      results = await validateAll(options);
    } else {
      results = [];
      for (const pkg of packages) {
        const result = await validatePackage(pkg, options);
        results.push(result);
      }
    }

    spinner.succeed(colors.success('Validation complete'));

    // Format and display results
    const output = formatOutput(results, argv.format, argv.noColor);

    if (!argv.quiet) {
      console.log('\n' + output + '\n');
    }

    // Save to file if requested
    if (argv.output) {
      await saveOutput(output, argv.output);
      console.log(colors.info(`Results saved to: ${argv.output}`));
    }

    // Exit with error code if validation failed
    const hasErrors = results.some(r => r.errors && r.errors.length > 0);
    process.exit(hasErrors ? 1 : 0);

  } catch (error) {
    spinner.fail(colors.error('Validation failed'));
    console.error(colors.error('\nError: ' + error.message));
    if (argv.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Benchmark command handler
async function handleBenchmark(argv) {
  const spinner = ora({
    text: 'Running benchmarks...',
    spinner: 'dots'
  }).start();

  try {
    const packages = argv.packages || [];
    const options = {
      iterations: argv.iterations,
      parallel: argv.parallel,
      warmup: argv.warmup !== false
    };

    let results;
    if (packages.length === 0) {
      spinner.text = 'Benchmarking all packages...';
      results = await benchmarkAll(options);
    } else {
      spinner.text = `Benchmarking ${packages.length} package(s)...`;

      if (options.parallel) {
        // Parallel execution
        results = await Promise.all(
          packages.map(pkg => benchmarkPackage(pkg, options))
        );
      } else {
        // Sequential execution with progress
        results = [];
        const progressBar = new cliProgress.SingleBar({
          format: 'Progress |' + colors.info('{bar}') + '| {percentage}% | {value}/{total} packages',
          barCompleteChar: '\u2588',
          barIncompleteChar: '\u2591',
          hideCursor: true
        });

        if (!argv.quiet) {
          spinner.stop();
          progressBar.start(packages.length, 0);
        }

        for (let i = 0; i < packages.length; i++) {
          const result = await benchmarkPackage(packages[i], options);
          results.push(result);
          if (!argv.quiet) {
            progressBar.update(i + 1);
          }
        }

        if (!argv.quiet) {
          progressBar.stop();
        }
      }
    }

    spinner.succeed(colors.success('Benchmarking complete'));

    // Format and display results
    const output = formatOutput(results, argv.format, argv.noColor);

    if (!argv.quiet) {
      console.log('\n' + output + '\n');

      // Show summary statistics
      const avgTime = results.reduce((sum, r) => sum + (r.avgTime || 0), 0) / results.length;
      console.log(colors.highlight('Summary Statistics:'));
      console.log(`  Average Time: ${colors.info(avgTime.toFixed(2) + 'ms')}`);
      console.log(`  Total Packages: ${colors.info(results.length)}`);
    }

    // Save to file if requested
    if (argv.output) {
      await saveOutput(output, argv.output);
      console.log(colors.info(`Results saved to: ${argv.output}`));
    }

  } catch (error) {
    spinner.fail(colors.error('Benchmarking failed'));
    console.error(colors.error('\nError: ' + error.message));
    if (argv.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Optimize command handler
async function handleOptimize(argv) {
  const spinner = ora({
    text: 'Analyzing packages for optimization...',
    spinner: 'dots'
  }).start();

  try {
    const packages = argv.packages || [];
    const options = {
      apply: argv.apply,
      dryRun: argv.dryRun !== false,
      severity: argv.severity || 'medium'
    };

    let results = [];

    if (packages.length === 0) {
      // Analyze all packages
      spinner.text = 'Analyzing all packages...';
      // Fix: Use dynamic path resolution instead of hardcoded development path
      const packagesDir = path.resolve(__dirname, '../..');
      const allPackages = await fs.readdir(packagesDir);
      for (const pkg of allPackages) {
        const result = await optimizePackage(pkg, options);
        if (result) results.push(result);
      }
    } else {
      for (const pkg of packages) {
        const result = await optimizePackage(pkg, options);
        results.push(result);
      }
    }

    spinner.succeed(colors.success('Optimization analysis complete'));

    // Format and display results
    const output = formatOutput(results, argv.format, argv.noColor);

    if (!argv.quiet) {
      console.log('\n' + output + '\n');

      // Show optimization summary
      const totalOptimizations = results.reduce((sum, r) => sum + (r.optimizations?.length || 0), 0);
      console.log(colors.highlight('Optimization Summary:'));
      console.log(`  Total Suggestions: ${colors.info(totalOptimizations)}`);
      console.log(`  Mode: ${colors.info(options.apply ? 'Applied' : 'Dry Run')}`);
    }

    // Save to file if requested
    if (argv.output) {
      await saveOutput(output, argv.output);
      console.log(colors.info(`Results saved to: ${argv.output}`));
    }

  } catch (error) {
    spinner.fail(colors.error('Optimization failed'));
    console.error(colors.error('\nError: ' + error.message));
    if (argv.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Report command handler
async function handleReport(argv) {
  const spinner = ora({
    text: 'Generating comprehensive report...',
    spinner: 'dots'
  }).start();

  try {
    const options = {
      format: argv.format || 'markdown',
      includeValidation: true,
      includeBenchmarks: true,
      includeOptimizations: true,
      compare: argv.compare
    };

    const report = await generateReport(options);

    spinner.succeed(colors.success('Report generated'));

    const output = formatOutput(report, argv.format, argv.noColor);

    if (!argv.quiet) {
      console.log('\n' + output + '\n');
    }

    // Save to file if requested
    if (argv.output) {
      await saveOutput(output, argv.output);
      console.log(colors.info(`Report saved to: ${argv.output}`));
    } else {
      // Fix: Use dynamic path resolution instead of hardcoded development path
      const defaultPath = path.resolve(__dirname, '../../docs', `benchoptimizer-report.${argv.format === 'html' ? 'html' : 'md'}`);
      await saveOutput(output, defaultPath);
      console.log(colors.info(`Report saved to: ${defaultPath}`));
    }

  } catch (error) {
    spinner.fail(colors.error('Report generation failed'));
    console.error(colors.error('\nError: ' + error.message));
    if (argv.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Compare command handler
async function handleCompare(argv) {
  const spinner = ora({
    text: 'Comparing benchmark results...',
    spinner: 'dots'
  }).start();

  try {
    if (!argv.baseline || !argv.current) {
      throw new Error('Both baseline and current files are required');
    }

    const baselineData = await fs.readJSON(argv.baseline);
    const currentData = await fs.readJSON(argv.current);

    const comparison = await compareResults(baselineData, currentData);

    spinner.succeed(colors.success('Comparison complete'));

    const output = formatOutput(comparison, argv.format, argv.noColor);

    if (!argv.quiet) {
      console.log('\n' + output + '\n');

      // Show comparison summary
      console.log(colors.highlight('Comparison Summary:'));
      if (comparison.improvements) {
        console.log(`  Improvements: ${colors.success(comparison.improvements)}`);
      }
      if (comparison.regressions) {
        console.log(`  Regressions: ${colors.error(comparison.regressions)}`);
      }
    }

    // Save to file if requested
    if (argv.output) {
      await saveOutput(output, argv.output);
      console.log(colors.info(`Comparison saved to: ${argv.output}`));
    }

  } catch (error) {
    spinner.fail(colors.error('Comparison failed'));
    console.error(colors.error('\nError: ' + error.message));
    if (argv.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Main CLI setup
const argv = yargs(hideBin(process.argv))
  .scriptName('benchoptimizer')
  .usage('$0 <command> [options]')
  .version('1.0.0')
  .help()

  // Validate command
  .command(
    'validate [packages..]',
    'Validate package structure and dependencies',
    (yargs) => {
      return yargs
        .positional('packages', {
          describe: 'Packages to validate (leave empty for all)',
          type: 'string',
          array: true
        })
        .option('fix', {
          describe: 'Automatically fix issues',
          type: 'boolean',
          default: false
        })
        .option('strict', {
          describe: 'Enable strict validation mode',
          type: 'boolean',
          default: false
        })
        .example('$0 validate core neural', 'Validate core and neural packages')
        .example('$0 validate --fix', 'Validate all packages and auto-fix issues');
    },
    handleValidate
  )

  // Benchmark command
  .command(
    'benchmark [packages..]',
    'Benchmark package performance',
    (yargs) => {
      return yargs
        .positional('packages', {
          describe: 'Packages to benchmark (leave empty for all)',
          type: 'string',
          array: true
        })
        .option('iterations', {
          alias: 'i',
          describe: 'Number of iterations',
          type: 'number',
          default: 100
        })
        .option('parallel', {
          alias: 'p',
          describe: 'Run benchmarks in parallel',
          type: 'boolean',
          default: false
        })
        .option('warmup', {
          describe: 'Enable warmup runs',
          type: 'boolean',
          default: true
        })
        .example('$0 benchmark core --iterations 1000', 'Benchmark core package with 1000 iterations')
        .example('$0 benchmark --parallel', 'Benchmark all packages in parallel');
    },
    handleBenchmark
  )

  // Optimize command
  .command(
    'optimize [packages..]',
    'Analyze and suggest optimizations',
    (yargs) => {
      return yargs
        .positional('packages', {
          describe: 'Packages to optimize (leave empty for all)',
          type: 'string',
          array: true
        })
        .option('apply', {
          describe: 'Apply optimizations automatically',
          type: 'boolean',
          default: false
        })
        .option('dry-run', {
          describe: 'Show what would be changed without applying',
          type: 'boolean',
          default: true
        })
        .option('severity', {
          describe: 'Minimum severity level',
          type: 'string',
          choices: ['low', 'medium', 'high'],
          default: 'medium'
        })
        .example('$0 optimize core --apply', 'Optimize core package and apply changes')
        .example('$0 optimize --severity high', 'Show only high severity optimizations');
    },
    handleOptimize
  )

  // Report command
  .command(
    'report',
    'Generate comprehensive report',
    (yargs) => {
      return yargs
        .option('compare', {
          describe: 'Baseline file to compare against',
          type: 'string'
        })
        .example('$0 report --format html', 'Generate HTML report')
        .example('$0 report --compare baseline.json', 'Generate report with comparison');
    },
    handleReport
  )

  // Compare command
  .command(
    'compare <baseline> <current>',
    'Compare two benchmark results',
    (yargs) => {
      return yargs
        .positional('baseline', {
          describe: 'Baseline benchmark file (JSON)',
          type: 'string'
        })
        .positional('current', {
          describe: 'Current benchmark file (JSON)',
          type: 'string'
        })
        .example('$0 compare baseline.json current.json', 'Compare two benchmark results');
    },
    handleCompare
  )

  // Global options
  .option('config', {
    alias: 'c',
    describe: 'Load configuration from file',
    type: 'string'
  })
  .option('output', {
    alias: 'o',
    describe: 'Output file path',
    type: 'string'
  })
  .option('format', {
    alias: 'f',
    describe: 'Output format',
    type: 'string',
    choices: ['json', 'table', 'markdown', 'html'],
    default: 'table'
  })
  .option('verbose', {
    alias: 'v',
    describe: 'Verbose output',
    type: 'boolean',
    default: false
  })
  .option('quiet', {
    alias: 'q',
    describe: 'Minimal output',
    type: 'boolean',
    default: false
  })
  .option('no-color', {
    describe: 'Disable colored output',
    type: 'boolean',
    default: false
  })

  .demandCommand(1, 'You need to specify a command')
  .recommendCommands()
  .strict()
  .wrap(null)
  .epilogue('For more information, visit: https://github.com/neural-trader')
  .parse();

// Load config if provided
if (argv.config) {
  try {
    const config = require(path.resolve(argv.config));
    Object.assign(argv, config);
  } catch (error) {
    console.error(colors.error('Failed to load config file: ' + error.message));
    process.exit(1);
  }
}
