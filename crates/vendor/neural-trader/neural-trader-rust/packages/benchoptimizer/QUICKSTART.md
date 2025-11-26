# Benchoptimizer Quick Start

Get started with benchoptimizer in 5 minutes!

## Installation

```bash
cd /workspaces/neural-trader/neural-trader-rust/packages/benchoptimizer
npm install
```

## Quick Commands

### 1. Show Help
```bash
./bin/benchoptimizer.js --help
```

### 2. Validate Packages
```bash
# Validate all packages
./bin/benchoptimizer.js validate

# Validate specific package
./bin/benchoptimizer.js validate core
```

### 3. Benchmark Performance
```bash
# Quick benchmark
./bin/benchoptimizer.js benchmark core --iterations 50

# Detailed benchmark
./bin/benchoptimizer.js benchmark --iterations 1000 --parallel
```

### 4. Optimize Analysis
```bash
# Show optimization suggestions
./bin/benchoptimizer.js optimize core

# Apply optimizations (be careful!)
./bin/benchoptimizer.js optimize --apply --severity low
```

### 5. Generate Report
```bash
# HTML report
./bin/benchoptimizer.js report --format html --output report.html

# Markdown report
./bin/benchoptimizer.js report --format markdown
```

## Output Formats

- `--format table` (default) - Pretty terminal tables
- `--format json` - Machine-readable JSON
- `--format markdown` - GitHub-compatible markdown
- `--format html` - Styled HTML report

## Common Options

- `--output <file>` - Save results to file
- `--quiet` - Minimal output
- `--verbose` - Detailed output
- `--no-color` - Disable colors
- `--parallel` - Parallel execution
- `--iterations <n>` - Number of benchmark iterations

## Example Workflows

### Quick Check
```bash
./bin/benchoptimizer.js validate --fix
./bin/benchoptimizer.js benchmark core --iterations 100
```

### Performance Analysis
```bash
./bin/benchoptimizer.js benchmark --iterations 1000 --output baseline.json
./bin/benchoptimizer.js optimize --severity high
./bin/benchoptimizer.js report --format html --output report.html
```

### CI/CD Integration
```bash
./bin/benchoptimizer.js validate --strict --quiet || exit 1
./bin/benchoptimizer.js benchmark --output results.json --quiet
```

## Next Steps

- Read [README.md](./README.md) for detailed documentation
- Check [examples/USAGE_EXAMPLES.md](./examples/USAGE_EXAMPLES.md) for more examples
- See [DEMO_OUTPUT.md](./DEMO_OUTPUT.md) for sample outputs
- Review [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) for technical details

## Test the CLI

```bash
# Run test suite
./tests/cli-test.sh
```

All features are working and ready to use!
