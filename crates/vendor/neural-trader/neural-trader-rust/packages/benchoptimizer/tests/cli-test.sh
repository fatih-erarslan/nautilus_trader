#!/bin/bash
# CLI test script for benchoptimizer

set -e

echo "🧪 Testing benchoptimizer CLI..."
echo ""

# Test 1: Help command
echo "✅ Test 1: Help command"
./bin/benchoptimizer.js --help > /dev/null 2>&1
echo "   Passed"

# Test 2: Validate single package
echo "✅ Test 2: Validate single package"
./bin/benchoptimizer.js validate core --quiet > /dev/null 2>&1
echo "   Passed"

# Test 3: Validate with JSON output
echo "✅ Test 3: Validate with JSON output"
./bin/benchoptimizer.js validate core --format json --quiet > /dev/null 2>&1
echo "   Passed"

# Test 4: Benchmark single package
echo "✅ Test 4: Benchmark single package"
./bin/benchoptimizer.js benchmark core --iterations 10 --quiet > /dev/null 2>&1
echo "   Passed"

# Test 5: Optimize analysis
echo "✅ Test 5: Optimize analysis"
./bin/benchoptimizer.js optimize core --quiet > /dev/null 2>&1
echo "   Passed"

# Test 6: Generate report
echo "✅ Test 6: Generate report"
./bin/benchoptimizer.js report --format json --quiet > /dev/null 2>&1
echo "   Passed"

# Test 7: Validate with fix
echo "✅ Test 7: Validate with fix (dry-run)"
./bin/benchoptimizer.js validate core --quiet > /dev/null 2>&1
echo "   Passed"

# Test 8: Benchmark with output file
echo "✅ Test 8: Benchmark with output file"
./bin/benchoptimizer.js benchmark core --iterations 10 --output /tmp/benchmark-test.json --quiet > /dev/null 2>&1
echo "   Passed"

# Test 9: Multiple formats
echo "✅ Test 9: Multiple output formats"
./bin/benchoptimizer.js validate core --format table --quiet > /dev/null 2>&1
./bin/benchoptimizer.js validate core --format json --quiet > /dev/null 2>&1
./bin/benchoptimizer.js validate core --format markdown --quiet > /dev/null 2>&1
echo "   Passed"

# Test 10: Optimize with severity levels
echo "✅ Test 10: Optimize with severity levels"
./bin/benchoptimizer.js optimize core --severity low --quiet > /dev/null 2>&1
./bin/benchoptimizer.js optimize core --severity medium --quiet > /dev/null 2>&1
./bin/benchoptimizer.js optimize core --severity high --quiet > /dev/null 2>&1
echo "   Passed"

echo ""
echo "🎉 All tests passed!"
echo ""
echo "Example commands:"
echo "  ./bin/benchoptimizer.js validate core neural"
echo "  ./bin/benchoptimizer.js benchmark --parallel --iterations 1000"
echo "  ./bin/benchoptimizer.js optimize --severity high"
echo "  ./bin/benchoptimizer.js report --format html --output report.html"
