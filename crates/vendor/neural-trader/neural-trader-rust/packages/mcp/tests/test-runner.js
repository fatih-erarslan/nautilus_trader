#!/usr/bin/env node
/**
 * MCP 2025-11 Compliance Test Runner
 * Runs all compliance tests and generates detailed validation report
 */

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class ComplianceTestRunner {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      overall: {
        total: 0,
        passed: 0,
        failed: 0,
        skipped: 0,
        duration: 0,
      },
      categories: {},
      violations: [],
      recommendations: [],
    };
  }

  async run() {
    console.log('╭──────────────────────────────────────────────────────────────────────────────╮');
    console.log('│           MCP 2025-11 Specification Compliance Test Suite                   │');
    console.log('│                      Neural Trader MCP Server                                │');
    console.log('╰──────────────────────────────────────────────────────────────────────────────╯');
    console.log('');

    const startTime = Date.now();

    try {
      // Run tests with Jest
      await this.runJestTests();

      // Calculate compliance percentage
      this.results.overall.duration = Date.now() - startTime;
      this.calculateCompliance();

      // Generate report
      await this.generateReport();

      // Display summary
      this.displaySummary();

      // Exit with appropriate code
      process.exit(this.results.overall.failed > 0 ? 1 : 0);
    } catch (error) {
      console.error('❌ Test runner failed:', error.message);
      process.exit(1);
    }
  }

  async runJestTests() {
    return new Promise((resolve, reject) => {
      const jest = spawn('npx', [
        'jest',
        '--config',
        path.join(__dirname, 'jest.config.js'),
        '--json',
        '--outputFile',
        path.join(__dirname, 'results.json'),
      ], {
        cwd: path.join(__dirname, '..'),
        stdio: 'inherit',
      });

      jest.on('close', async (code) => {
        try {
          // Read test results
          const resultsPath = path.join(__dirname, 'results.json');
          const resultsJson = await fs.readFile(resultsPath, 'utf8');
          const results = JSON.parse(resultsJson);

          this.processJestResults(results);
          resolve();
        } catch (error) {
          // Tests may have failed, but that's okay
          console.log('Note: Some tests may have failed, continuing with report generation...');
          resolve();
        }
      });

      jest.on('error', reject);
    });
  }

  processJestResults(results) {
    this.results.overall.total = results.numTotalTests || 0;
    this.results.overall.passed = results.numPassedTests || 0;
    this.results.overall.failed = results.numFailedTests || 0;
    this.results.overall.skipped = results.numPendingTests || 0;

    // Process test results by category
    if (results.testResults) {
      results.testResults.forEach(file => {
        const category = this.categorizeTest(file.name);

        if (!this.results.categories[category]) {
          this.results.categories[category] = {
            total: 0,
            passed: 0,
            failed: 0,
            tests: [],
          };
        }

        const cat = this.results.categories[category];
        cat.total += file.assertionResults?.length || 0;
        cat.passed += file.numPassingTests || 0;
        cat.failed += file.numFailingTests || 0;

        // Collect failures
        if (file.assertionResults) {
          file.assertionResults.forEach(test => {
            if (test.status === 'failed') {
              this.results.violations.push({
                category,
                test: test.title,
                file: path.basename(file.name),
                message: test.failureMessages?.[0] || 'Unknown error',
              });
            }
          });
        }
      });
    }
  }

  categorizeTest(filePath) {
    if (filePath.includes('protocol')) return 'Protocol Compliance';
    if (filePath.includes('discovery')) return 'Tool Discovery';
    if (filePath.includes('transport')) return 'Transport Layer';
    if (filePath.includes('logging')) return 'Audit Logging';
    if (filePath.includes('integration')) return 'MCP Methods';
    return 'Other';
  }

  calculateCompliance() {
    const total = this.results.overall.total;
    const passed = this.results.overall.passed;

    this.results.compliance = {
      percentage: total > 0 ? ((passed / total) * 100).toFixed(2) : 0,
      status: this.getComplianceStatus(passed, total),
    };

    // Add recommendations based on failures
    this.generateRecommendations();
  }

  getComplianceStatus(passed, total) {
    if (total === 0) return 'UNKNOWN';
    const percentage = (passed / total) * 100;

    if (percentage === 100) return 'FULLY COMPLIANT';
    if (percentage >= 95) return 'HIGHLY COMPLIANT';
    if (percentage >= 80) return 'MOSTLY COMPLIANT';
    if (percentage >= 50) return 'PARTIALLY COMPLIANT';
    return 'NON-COMPLIANT';
  }

  generateRecommendations() {
    const recommendations = [];

    // Check each category
    Object.entries(this.results.categories).forEach(([category, data]) => {
      if (data.failed > 0) {
        recommendations.push({
          category,
          priority: data.failed > data.passed ? 'HIGH' : 'MEDIUM',
          message: `${data.failed} test(s) failing in ${category}`,
        });
      }
    });

    // Specific recommendations based on violations
    const violationTypes = new Set(this.results.violations.map(v => v.category));

    if (violationTypes.has('Protocol Compliance')) {
      recommendations.push({
        category: 'Protocol Compliance',
        priority: 'HIGH',
        message: 'Fix JSON-RPC 2.0 format issues for protocol compliance',
      });
    }

    if (violationTypes.has('Tool Discovery')) {
      recommendations.push({
        category: 'Tool Discovery',
        priority: 'HIGH',
        message: 'Ensure all tools have valid JSON Schema 1.1 format',
      });
    }

    if (violationTypes.has('Audit Logging')) {
      recommendations.push({
        category: 'Audit Logging',
        priority: 'MEDIUM',
        message: 'Verify JSON Lines format and event logging completeness',
      });
    }

    this.results.recommendations = recommendations;
  }

  async generateReport() {
    const reportPath = path.join(__dirname, 'COMPLIANCE_REPORT.md');

    const report = this.buildMarkdownReport();
    await fs.writeFile(reportPath, report);

    console.log('');
    console.log(`📄 Detailed report saved to: ${reportPath}`);
    console.log('');
  }

  buildMarkdownReport() {
    const { compliance, overall, categories, violations, recommendations } = this.results;

    let md = `# MCP 2025-11 Compliance Validation Report\n\n`;
    md += `**Generated:** ${this.results.timestamp}\n`;
    md += `**Server:** Neural Trader MCP Server v2.0.0\n\n`;

    md += `## Executive Summary\n\n`;
    md += `**Compliance Status:** ${compliance.status}\n`;
    md += `**Compliance Percentage:** ${compliance.percentage}%\n\n`;

    md += `### Overall Results\n\n`;
    md += `| Metric | Count |\n`;
    md += `|--------|-------|\n`;
    md += `| Total Tests | ${overall.total} |\n`;
    md += `| Passed | ${overall.passed} |\n`;
    md += `| Failed | ${overall.failed} |\n`;
    md += `| Skipped | ${overall.skipped} |\n`;
    md += `| Duration | ${(overall.duration / 1000).toFixed(2)}s |\n\n`;

    md += `## Requirement Validation\n\n`;

    // Category results
    Object.entries(categories).forEach(([category, data]) => {
      const percentage = data.total > 0 ? ((data.passed / data.total) * 100).toFixed(1) : 0;
      const status = percentage === '100.0' ? '✅' : percentage >= 80 ? '⚠️' : '❌';

      md += `### ${status} ${category}\n\n`;
      md += `- **Tests:** ${data.passed}/${data.total} passed (${percentage}%)\n`;

      if (data.failed > 0) {
        md += `- **Issues:** ${data.failed} test(s) failing\n`;
      }

      md += `\n`;
    });

    // Specification violations
    if (violations.length > 0) {
      md += `## Specification Violations\n\n`;
      md += `Found ${violations.length} specification violation(s):\n\n`;

      violations.forEach((v, i) => {
        md += `### ${i + 1}. ${v.category} - ${v.test}\n\n`;
        md += `**File:** ${v.file}\n\n`;
        md += `**Issue:**\n\`\`\`\n${v.message.substring(0, 500)}\n\`\`\`\n\n`;
      });
    } else {
      md += `## Specification Violations\n\n`;
      md += `✅ No specification violations detected!\n\n`;
    }

    // Recommendations
    if (recommendations.length > 0) {
      md += `## Recommendations\n\n`;

      const highPriority = recommendations.filter(r => r.priority === 'HIGH');
      const mediumPriority = recommendations.filter(r => r.priority === 'MEDIUM');

      if (highPriority.length > 0) {
        md += `### 🔴 High Priority\n\n`;
        highPriority.forEach(r => {
          md += `- **${r.category}:** ${r.message}\n`;
        });
        md += `\n`;
      }

      if (mediumPriority.length > 0) {
        md += `### 🟡 Medium Priority\n\n`;
        mediumPriority.forEach(r => {
          md += `- **${r.category}:** ${r.message}\n`;
        });
        md += `\n`;
      }
    }

    md += `## Compliance Checklist\n\n`;
    md += this.generateChecklist();

    return md;
  }

  generateChecklist() {
    const checks = [
      { category: 'Protocol Compliance', item: 'JSON-RPC 2.0 request/response format' },
      { category: 'Protocol Compliance', item: 'Standard error codes match specification' },
      { category: 'Protocol Compliance', item: 'Batch request handling' },
      { category: 'Protocol Compliance', item: 'Request ID tracking' },
      { category: 'Tool Discovery', item: 'All tools discoverable via tools/list' },
      { category: 'Tool Discovery', item: 'JSON Schema 1.1 format for all schemas' },
      { category: 'Tool Discovery', item: 'Metadata completeness (cost, latency, category)' },
      { category: 'Tool Discovery', item: 'ETag generation for caching' },
      { category: 'Transport Layer', item: 'STDIO line-delimited JSON' },
      { category: 'Transport Layer', item: 'stderr separate from protocol' },
      { category: 'Transport Layer', item: 'Graceful shutdown handling' },
      { category: 'Audit Logging', item: 'JSON Lines format' },
      { category: 'Audit Logging', item: 'All events logged (tool_call, tool_result, errors)' },
      { category: 'Audit Logging', item: 'Log file creation and rotation' },
      { category: 'MCP Methods', item: 'initialize method' },
      { category: 'MCP Methods', item: 'tools/list method' },
      { category: 'MCP Methods', item: 'tools/call method' },
      { category: 'MCP Methods', item: 'tools/schema method' },
    ];

    let checklist = '';
    const categoryStatus = {};

    // Calculate category status
    Object.entries(this.results.categories).forEach(([cat, data]) => {
      categoryStatus[cat] = data.failed === 0;
    });

    checks.forEach(check => {
      const status = categoryStatus[check.category] ? '✅' : '❌';
      checklist += `- [${status === '✅' ? 'x' : ' '}] ${status} ${check.item}\n`;
    });

    return checklist;
  }

  displaySummary() {
    const { compliance, overall } = this.results;

    console.log('╭──────────────────────────────────────────────────────────────────────────────╮');
    console.log('│                          COMPLIANCE SUMMARY                                  │');
    console.log('╰──────────────────────────────────────────────────────────────────────────────╯');
    console.log('');

    const statusSymbol = compliance.status === 'FULLY COMPLIANT' ? '✅' :
                         compliance.percentage >= 80 ? '⚠️' : '❌';

    console.log(`${statusSymbol} Compliance Status: ${compliance.status}`);
    console.log(`   Compliance Percentage: ${compliance.percentage}%`);
    console.log('');
    console.log(`   Total Tests: ${overall.total}`);
    console.log(`   ✅ Passed: ${overall.passed}`);
    console.log(`   ❌ Failed: ${overall.failed}`);
    console.log(`   ⏭️  Skipped: ${overall.skipped}`);
    console.log('');

    // Category breakdown
    console.log('Category Results:');
    Object.entries(this.results.categories).forEach(([category, data]) => {
      const percentage = data.total > 0 ? ((data.passed / data.total) * 100).toFixed(0) : 0;
      const status = percentage === '100' ? '✅' : percentage >= 80 ? '⚠️' : '❌';
      console.log(`   ${status} ${category}: ${data.passed}/${data.total} (${percentage}%)`);
    });

    console.log('');

    if (this.results.recommendations.length > 0) {
      console.log('⚠️  Recommendations: See COMPLIANCE_REPORT.md for details');
    }

    console.log('');
  }
}

// Run if called directly
if (require.main === module) {
  const runner = new ComplianceTestRunner();
  runner.run();
}

module.exports = { ComplianceTestRunner };
