# Benchoptimizer Usage Examples

## Quick Start

```bash
# Install globally
npm install -g @neural-trader/benchoptimizer

# Or use via npx
npx benchoptimizer --help
```

## Basic Commands

### 1. Validate Packages

```bash
# Validate all packages
benchoptimizer validate

# Validate specific packages
benchoptimizer validate core neural strategies

# Auto-fix issues
benchoptimizer validate --fix

# Strict mode (treats warnings as errors)
benchoptimizer validate --strict

# Output as JSON
benchoptimizer validate --format json --output validation.json
```

### 2. Benchmark Performance

```bash
# Benchmark all packages
benchoptimizer benchmark

# Benchmark specific packages
benchoptimizer benchmark core neural

# Run 1000 iterations
benchoptimizer benchmark --iterations 1000

# Parallel execution (faster)
benchoptimizer benchmark --parallel

# Save results
benchoptimizer benchmark --output benchmark-results.json --format json

# Disable warmup
benchoptimizer benchmark --warmup false
```

### 3. Optimize Packages

```bash
# Analyze all packages
benchoptimizer optimize

# Analyze specific packages
benchoptimizer optimize core neural

# Apply optimizations (WARNING: modifies files)
benchoptimizer optimize --apply

# Dry run (default - shows what would change)
benchoptimizer optimize --dry-run

# Filter by severity
benchoptimizer optimize --severity high
benchoptimizer optimize --severity medium
benchoptimizer optimize --severity low
```

### 4. Generate Reports

```bash
# Generate markdown report
benchoptimizer report --format markdown

# Generate HTML report
benchoptimizer report --format html --output report.html

# Generate JSON report
benchoptimizer report --format json --output report.json

# Compare with baseline
benchoptimizer report --compare baseline.json
```

### 5. Compare Results

```bash
# Compare two benchmark runs
benchoptimizer compare baseline.json current.json

# Output as table (default)
benchoptimizer compare baseline.json current.json --format table

# Output as markdown
benchoptimizer compare baseline.json current.json --format markdown
```

## Advanced Usage

### Using Configuration Files

Create `benchoptimizer.config.json`:

```json
{
  "iterations": 1000,
  "parallel": true,
  "format": "markdown",
  "severity": "medium",
  "output": "./reports/benchmark.md"
}
```

Use it:

```bash
benchoptimizer benchmark --config benchoptimizer.config.json
```

### Workflow Examples

#### Complete Performance Analysis

```bash
# 1. Validate everything is correct
benchoptimizer validate --fix

# 2. Run baseline benchmark
benchoptimizer benchmark --iterations 1000 --output baseline.json

# 3. Analyze optimizations
benchoptimizer optimize --severity medium

# 4. Apply safe optimizations
benchoptimizer optimize --apply --severity low

# 5. Run new benchmark
benchoptimizer benchmark --iterations 1000 --output optimized.json

# 6. Compare results
benchoptimizer compare baseline.json optimized.json

# 7. Generate comprehensive report
benchoptimizer report --format html --output report.html
```

#### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running benchoptimizer validation..."

# Validate with strict mode
if ! benchoptimizer validate --strict --quiet; then
  echo "❌ Validation failed! Please fix issues before committing."
  exit 1
fi

echo "✅ Validation passed!"
exit 0
```

#### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmark and Validate

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      - name: Validate packages
        run: npx benchoptimizer validate --strict

      - name: Run benchmarks
        run: npx benchoptimizer benchmark --iterations 1000 --output benchmark-results.json

      - name: Analyze optimizations
        run: npx benchoptimizer optimize --severity high

      - name: Generate report
        run: npx benchoptimizer report --format markdown --output BENCHMARK.md

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            benchmark-results.json
            BENCHMARK.md

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('BENCHMARK.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### Output Format Examples

#### JSON Output

```bash
benchoptimizer validate core --format json
```

```json
[
  {
    "package": "core",
    "valid": true,
    "errors": [],
    "warnings": ["No test directory found"],
    "info": {
      "name": "@neural-trader/core",
      "version": "1.0.0",
      "dependencies": 5
    }
  }
]
```

#### Table Output (Default)

```bash
benchoptimizer benchmark core --iterations 100
```

```
┌─────────┬────────────┬─────────┬─────────┬─────────┬────────┐
│ package │ iterations │ avgTime │ minTime │ maxTime │ stdDev │
├─────────┼────────────┼─────────┼─────────┼─────────┼────────┤
│ core    │ 100        │ 0.07ms  │ 0.05ms  │ 0.10ms  │ 0.01ms │
└─────────┴────────────┴─────────┴─────────┴─────────┴────────┘
```

#### Markdown Output

```bash
benchoptimizer report --format markdown
```

```markdown
# Benchoptimizer Report

## Validation Results

| Package | Valid | Errors | Warnings |
|---------|-------|--------|----------|
| core    | ✓     | 0      | 1        |
| neural  | ✓     | 0      | 0        |

## Benchmark Results

| Package | Avg Time | Min Time | Max Time |
|---------|----------|----------|----------|
| core    | 0.07ms   | 0.05ms   | 0.10ms   |
| neural  | 0.12ms   | 0.10ms   | 0.15ms   |
```

#### HTML Output

```bash
benchoptimizer report --format html --output report.html
```

Generates a styled HTML report with tables and statistics.

## Common Use Cases

### 1. Package Development

```bash
# Before committing
benchoptimizer validate --fix
benchoptimizer optimize --severity high

# After changes
benchoptimizer benchmark --iterations 500 --output after.json
benchoptimizer compare before.json after.json
```

### 2. Performance Regression Detection

```bash
# Save baseline
benchoptimizer benchmark --output baseline.json

# After changes, compare
benchoptimizer benchmark --output current.json
benchoptimizer compare baseline.json current.json

# If regressions found
if [ $? -ne 0 ]; then
  echo "Performance regression detected!"
  exit 1
fi
```

### 3. Code Review

```bash
# Generate review report
benchoptimizer validate --strict
benchoptimizer optimize --severity medium
benchoptimizer report --format markdown --output REVIEW.md
```

### 4. Documentation Generation

```bash
# Generate comprehensive docs
benchoptimizer validate > validation.txt
benchoptimizer benchmark --iterations 100 > benchmark.txt
benchoptimizer optimize > optimization.txt
benchoptimizer report --format html --output docs/performance.html
```

## Tips and Tricks

### 1. Quiet Mode for Scripts

```bash
# Only show errors
benchoptimizer validate --quiet
echo $?  # Check exit code
```

### 2. Verbose Mode for Debugging

```bash
# Show detailed information
benchoptimizer benchmark --verbose
```

### 3. No Color Output

```bash
# For logs and CI/CD
benchoptimizer validate --no-color > log.txt
```

### 4. Multiple Packages

```bash
# Validate specific packages
benchoptimizer validate core neural strategies execution

# Benchmark core packages only
benchoptimizer benchmark core neural --parallel
```

### 5. Performance Monitoring

```bash
# Create a monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
benchoptimizer benchmark \
  --iterations 1000 \
  --output "benchmarks/$DATE.json" \
  --quiet

benchoptimizer report \
  --format html \
  --output "reports/$DATE.html" \
  --quiet

echo "Report saved to reports/$DATE.html"
EOF

chmod +x monitor.sh
```

## Troubleshooting

### Issue: "Package not found"

```bash
# Check package exists
ls -la /workspaces/neural-trader/neural-trader-rust/packages/

# Use correct package name
benchoptimizer validate core  # not Core or @neural-trader/core
```

### Issue: "Main entry point not found"

```bash
# Check package.json has 'main' field
cat packages/core/package.json | grep main

# Validate package structure
benchoptimizer validate core --verbose
```

### Issue: Slow benchmarks

```bash
# Reduce iterations
benchoptimizer benchmark --iterations 50

# Use parallel mode
benchoptimizer benchmark --parallel

# Disable warmup
benchoptimizer benchmark --warmup false
```

### Issue: Native binding not available

```bash
# Use JavaScript fallback (already automatic)
# For better performance, build native module:
cd packages/benchoptimizer
npm run build
```

## API Reference

See main README.md for programmatic usage examples.
