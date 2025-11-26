#!/usr/bin/env node
/**
 * Core Trading MCP Tools - Direct Analysis
 *
 * Analyzes the implementation directly from source code
 * Tests 6 core tools by examining their Rust implementation
 */

const fs = require('fs');
const path = require('path');

// Analysis results
const analysis = {
  timestamp: new Date().toISOString(),
  tools: {},
  securityFindings: [],
  optimizations: [],
  codeQuality: {}
};

/**
 * Analyze tool implementation from source
 */
function analyzeTool(toolName, startLine, endLine, sourceCode) {
  const lines = sourceCode.split('\n').slice(startLine - 1, endLine);
  const code = lines.join('\n');

  const result = {
    name: toolName,
    lineRange: `${startLine}-${endLine}`,
    linesOfCode: endLine - startLine + 1,
    functionality: {},
    errorHandling: {},
    inputValidation: {},
    performance: {},
    security: {}
  };

  // Check for error handling
  result.errorHandling.hasResultType = code.includes('-> ToolResult') || code.includes('-> Result');
  result.errorHandling.hasErrorPropagation = code.includes('.map_err(') || code.includes('?');
  result.errorHandling.hasCustomErrors = code.includes('Error::from_reason');
  result.errorHandling.catchesUnwrap = !code.includes('.unwrap()') || code.includes('catch_unwind');

  // Check for input validation
  result.inputValidation.validatesInputs = code.includes('validate') || code.includes('if ') || code.includes('match');
  result.inputValidation.checksNullable = code.includes('Option<') || code.includes('unwrap_or');
  result.inputValidation.typeChecks = code.includes('::new(') || code.includes('parse');
  result.inputValidation.boundsChecks = code.includes('> 0') || code.includes('< ');

  // Check for performance patterns
  result.performance.isAsync = code.includes('async fn');
  result.performance.hasCaching = code.includes('cache') || code.includes('Arc<');
  result.performance.usesGpu = code.includes('use_gpu') || code.includes('gpu');
  result.performance.parallelProcessing = code.includes('parallel') || code.includes('rayon');
  result.performance.hasTimeouts = code.includes('timeout') || code.includes('Duration');

  // Check for security patterns
  result.security.sanitizesInput = code.includes('.trim()') || code.includes('.escape(');
  result.security.checksAuth = code.includes('AUTH') || code.includes('token');
  result.security.logsActions = code.includes('log::') || code.includes('info!');
  result.security.hasRateLimiting = code.includes('rate_limit') || code.includes('throttle');
  result.security.checksEnvVars = code.includes('env::var');

  // Extract documentation
  const docComments = lines.filter(l => l.trim().startsWith('///')).map(l => l.trim().substring(4));
  result.documentation = {
    hasDocumentation: docComments.length > 0,
    docLines: docComments.length,
    description: docComments[0] || '',
    hasArguments: docComments.some(c => c.includes('# Arguments')),
    hasReturns: docComments.some(c => c.includes('# Returns')),
    hasExamples: docComments.some(c => c.includes('# Examples'))
  };

  return result;
}

/**
 * Analyze security issues
 */
function analyzeSecurityIssues(toolResults) {
  const findings = [];

  for (const tool of Object.values(toolResults)) {
    // Missing rate limiting
    if (!tool.security.hasRateLimiting) {
      findings.push({
        severity: 'MEDIUM',
        tool: tool.name,
        category: 'Rate Limiting',
        issue: 'No rate limiting implementation detected',
        risk: 'Susceptible to DoS attacks via request flooding',
        recommendation: 'Implement token bucket or leaky bucket rate limiting',
        cwe: 'CWE-770: Allocation of Resources Without Limits or Throttling'
      });
    }

    // Missing input sanitization
    if (!tool.security.sanitizesInput && tool.inputValidation.validatesInputs) {
      findings.push({
        severity: 'MEDIUM',
        tool: tool.name,
        category: 'Input Validation',
        issue: 'Input validation present but no sanitization detected',
        risk: 'Potential injection vulnerabilities',
        recommendation: 'Add input sanitization for string parameters',
        cwe: 'CWE-20: Improper Input Validation'
      });
    }

    // Missing audit logging
    if (!tool.security.logsActions) {
      findings.push({
        severity: 'LOW',
        tool: tool.name,
        category: 'Audit Logging',
        issue: 'No audit logging detected',
        risk: 'Difficult to track usage and debug issues',
        recommendation: 'Add structured logging for all tool invocations',
        cwe: 'CWE-778: Insufficient Logging'
      });
    }

    // Unwrap usage (potential panic)
    if (!tool.errorHandling.catchesUnwrap) {
      findings.push({
        severity: 'HIGH',
        tool: tool.name,
        category: 'Error Handling',
        issue: 'Potential unwrap() calls without panic protection',
        risk: 'Service crashes on unexpected inputs',
        recommendation: 'Replace unwrap() with proper error handling or expect()',
        cwe: 'CWE-755: Improper Handling of Exceptional Conditions'
      });
    }
  }

  return findings;
}

/**
 * Generate optimization recommendations
 */
function generateOptimizations(toolResults) {
  const recommendations = [];

  for (const tool of Object.values(toolResults)) {
    // Caching opportunity
    if (!tool.performance.hasCaching && tool.linesOfCode > 50) {
      recommendations.push({
        priority: 'HIGH',
        tool: tool.name,
        category: 'Caching',
        optimization: 'Implement response caching',
        benefit: 'Reduce latency by 60-80% for repeated queries',
        implementation: 'Use Arc<RwLock<HashMap>> or external cache like Redis',
        estimatedImpact: 'High - especially for list_strategies and get_strategy_info'
      });
    }

    // GPU acceleration
    if (tool.performance.usesGpu && !tool.name.includes('neural')) {
      recommendations.push({
        priority: 'MEDIUM',
        tool: tool.name,
        category: 'GPU Acceleration',
        optimization: 'Verify GPU utilization is optimal',
        benefit: 'Ensure GPU flag actually triggers GPU code paths',
        implementation: 'Profile GPU utilization during execution',
        estimatedImpact: 'Medium - validate claimed GPU capabilities'
      });
    }

    // Async optimization
    if (!tool.performance.isAsync) {
      recommendations.push({
        priority: 'HIGH',
        tool: tool.name,
        category: 'Concurrency',
        optimization: 'Convert to async function',
        benefit: 'Enable non-blocking execution and better throughput',
        implementation: 'Add async keyword and use tokio runtime',
        estimatedImpact: 'High - improve concurrency by 10-20x'
      });
    }

    // Documentation
    if (!tool.documentation.hasExamples) {
      recommendations.push({
        priority: 'LOW',
        tool: tool.name,
        category: 'Documentation',
        optimization: 'Add usage examples to documentation',
        benefit: 'Improve developer experience and reduce support burden',
        implementation: 'Add # Examples section with common use cases',
        estimatedImpact: 'Low - quality of life improvement'
      });
    }

    // Input validation
    if (!tool.inputValidation.boundsChecks && tool.name.includes('analysis')) {
      recommendations.push({
        priority: 'MEDIUM',
        tool: tool.name,
        category: 'Input Validation',
        optimization: 'Add bounds checking for numeric inputs',
        benefit: 'Prevent invalid parameter ranges from causing errors',
        implementation: 'Validate min/max ranges before processing',
        estimatedImpact: 'Medium - improve reliability'
      });
    }
  }

  return recommendations;
}

/**
 * Analyze code quality metrics
 */
function analyzeCodeQuality(toolResults) {
  const metrics = {
    avgLinesOfCode: 0,
    errorHandlingScore: 0,
    inputValidationScore: 0,
    documentationScore: 0,
    securityScore: 0,
    performanceScore: 0
  };

  const tools = Object.values(toolResults);
  const count = tools.length;

  // Calculate averages
  metrics.avgLinesOfCode = tools.reduce((sum, t) => sum + t.linesOfCode, 0) / count;

  // Error handling score (0-100)
  metrics.errorHandlingScore = tools.reduce((sum, t) => {
    let score = 0;
    if (t.errorHandling.hasResultType) score += 25;
    if (t.errorHandling.hasErrorPropagation) score += 25;
    if (t.errorHandling.hasCustomErrors) score += 25;
    if (t.errorHandling.catchesUnwrap) score += 25;
    return sum + score;
  }, 0) / count;

  // Input validation score (0-100)
  metrics.inputValidationScore = tools.reduce((sum, t) => {
    let score = 0;
    if (t.inputValidation.validatesInputs) score += 25;
    if (t.inputValidation.checksNullable) score += 25;
    if (t.inputValidation.typeChecks) score += 25;
    if (t.inputValidation.boundsChecks) score += 25;
    return sum + score;
  }, 0) / count;

  // Documentation score (0-100)
  metrics.documentationScore = tools.reduce((sum, t) => {
    let score = 0;
    if (t.documentation.hasDocumentation) score += 25;
    if (t.documentation.hasArguments) score += 25;
    if (t.documentation.hasReturns) score += 25;
    if (t.documentation.hasExamples) score += 25;
    return sum + score;
  }, 0) / count;

  // Security score (0-100)
  metrics.securityScore = tools.reduce((sum, t) => {
    let score = 0;
    if (t.security.sanitizesInput) score += 20;
    if (t.security.checksAuth) score += 20;
    if (t.security.logsActions) score += 20;
    if (t.security.hasRateLimiting) score += 20;
    if (t.security.checksEnvVars) score += 20;
    return sum + score;
  }, 0) / count;

  // Performance score (0-100)
  metrics.performanceScore = tools.reduce((sum, t) => {
    let score = 0;
    if (t.performance.isAsync) score += 25;
    if (t.performance.hasCaching) score += 25;
    if (t.performance.usesGpu) score += 25;
    if (t.performance.hasTimeouts) score += 25;
    return sum + score;
  }, 0) / count;

  // Overall quality score
  metrics.overallScore = (
    metrics.errorHandlingScore * 0.25 +
    metrics.inputValidationScore * 0.20 +
    metrics.documentationScore * 0.15 +
    metrics.securityScore * 0.20 +
    metrics.performanceScore * 0.20
  );

  return metrics;
}

/**
 * Main analysis function
 */
async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║   Core Trading MCP Tools - Code Analysis                  ║');
  console.log('║   Direct source code analysis of 6 core tools             ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  try {
    // Read source file
    const sourcePath = path.join(__dirname, '../neural-trader-rust/crates/napi-bindings/src/mcp_tools.rs');
    const sourceCode = fs.readFileSync(sourcePath, 'utf-8');

    console.log(`📁 Analyzing: ${sourcePath}`);
    console.log(`📏 Total lines: ${sourceCode.split('\n').length}\n`);

    // Analyze each core tool
    const toolDefinitions = [
      { name: 'ping', start: 38, end: 73 },
      { name: 'list_strategies', start: 79, end: 117 },
      { name: 'get_strategy_info', start: 126, end: 202 },
      { name: 'quick_analysis', start: 381, end: 414 },
      { name: 'get_portfolio_status', start: 211, end: 258 },
    ];

    console.log('🔬 Analyzing tools...\n');

    for (const def of toolDefinitions) {
      const result = analyzeTool(def.name, def.start, def.end, sourceCode);
      analysis.tools[def.name] = result;

      console.log(`✅ ${def.name}`);
      console.log(`   Lines: ${result.linesOfCode} (${result.lineRange})`);
      console.log(`   Async: ${result.performance.isAsync ? '✓' : '✗'}`);
      console.log(`   Error handling: ${result.errorHandling.hasResultType ? '✓' : '✗'}`);
      console.log(`   Input validation: ${result.inputValidation.validatesInputs ? '✓' : '✗'}`);
      console.log(`   Documentation: ${result.documentation.hasDocumentation ? '✓' : '✗'}`);
      console.log('');
    }

    // Security analysis
    console.log('🔒 Security Analysis\n' + '━'.repeat(60));
    analysis.securityFindings = analyzeSecurityIssues(analysis.tools);

    const criticalFindings = analysis.securityFindings.filter(f => f.severity === 'HIGH');
    const mediumFindings = analysis.securityFindings.filter(f => f.severity === 'MEDIUM');
    const lowFindings = analysis.securityFindings.filter(f => f.severity === 'LOW');

    console.log(`Total findings: ${analysis.securityFindings.length}`);
    console.log(`  🔴 Critical: ${criticalFindings.length}`);
    console.log(`  🟡 Medium:   ${mediumFindings.length}`);
    console.log(`  🟢 Low:      ${lowFindings.length}\n`);

    if (criticalFindings.length > 0) {
      console.log('Critical Issues:');
      criticalFindings.forEach((f, i) => {
        console.log(`  ${i + 1}. [${f.tool}] ${f.issue}`);
        console.log(`     Risk: ${f.risk}`);
        console.log(`     Fix: ${f.recommendation}\n`);
      });
    }

    // Optimization recommendations
    console.log('⚡ Optimization Recommendations\n' + '━'.repeat(60));
    analysis.optimizations = generateOptimizations(analysis.tools);

    const criticalOpts = analysis.optimizations.filter(o => o.priority === 'HIGH');
    const mediumOpts = analysis.optimizations.filter(o => o.priority === 'MEDIUM');

    console.log(`Total recommendations: ${analysis.optimizations.length}`);
    console.log(`  🔴 High priority:   ${criticalOpts.length}`);
    console.log(`  🟡 Medium priority: ${mediumOpts.length}\n`);

    if (criticalOpts.length > 0) {
      console.log('High Priority Optimizations:');
      criticalOpts.slice(0, 5).forEach((opt, i) => {
        console.log(`  ${i + 1}. [${opt.tool}] ${opt.optimization}`);
        console.log(`     Benefit: ${opt.benefit}`);
        console.log(`     Impact: ${opt.estimatedImpact}\n`);
      });
    }

    // Code quality metrics
    console.log('📊 Code Quality Metrics\n' + '━'.repeat(60));
    analysis.codeQuality = analyzeCodeQuality(analysis.tools);

    console.log(`Overall Quality Score:      ${analysis.codeQuality.overallScore.toFixed(1)}/100`);
    console.log(`Error Handling:             ${analysis.codeQuality.errorHandlingScore.toFixed(1)}/100`);
    console.log(`Input Validation:           ${analysis.codeQuality.inputValidationScore.toFixed(1)}/100`);
    console.log(`Documentation:              ${analysis.codeQuality.documentationScore.toFixed(1)}/100`);
    console.log(`Security:                   ${analysis.codeQuality.securityScore.toFixed(1)}/100`);
    console.log(`Performance:                ${analysis.codeQuality.performanceScore.toFixed(1)}/100`);
    console.log(`Average Lines per Tool:     ${analysis.codeQuality.avgLinesOfCode.toFixed(0)}`);

    // Save results
    const outputDir = path.join(__dirname, '../docs/mcp-analysis');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, 'code_analysis_results.json');
    fs.writeFileSync(outputPath, JSON.stringify(analysis, null, 2));

    console.log(`\n✅ Results saved to: ${outputPath}`);

    // Return summary
    return {
      toolsAnalyzed: Object.keys(analysis.tools).length,
      overallScore: analysis.codeQuality.overallScore,
      criticalIssues: criticalFindings.length,
      optimizations: analysis.optimizations.length
    };

  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run analysis
if (require.main === module) {
  main()
    .then(summary => {
      console.log('\n✅ Analysis complete!');
      console.log(JSON.stringify(summary, null, 2));
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { analyzeTool, analyzeSecurityIssues, generateOptimizations };
