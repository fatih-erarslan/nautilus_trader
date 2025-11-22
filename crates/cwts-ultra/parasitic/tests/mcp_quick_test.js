#!/usr/bin/env node

/**
 * Quick MCP Server Test Suite
 * 
 * Tests all 10 parasitic trading tools without Rust backend dependency
 * Validates core functionality and generates performance report
 */

const WebSocket = require('ws');
const path = require('path');

class QuickMCPTester {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      tools: {},
      websocket: null,
      performance: {},
      summary: {}
    };
  }

  async runQuickTest() {
    console.log('🚀 Quick MCP Server Functionality Test');
    console.log('='.repeat(50));

    // Test WebSocket port
    await this.testWebSocketPort();

    // Test all 10 tools
    await this.testAllTools();

    // Generate report
    this.generateReport();
  }

  async testWebSocketPort() {
    console.log('\n🔗 Testing WebSocket Port 8080...');
    
    try {
      const ws = new WebSocket('ws://localhost:8080');
      
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 3000);

        ws.on('open', () => {
          clearTimeout(timeout);
          ws.close();
          resolve();
        });

        ws.on('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });

      this.results.websocket = {
        status: 'SUCCESS',
        port: 8080,
        connection_time: Date.now()
      };
      
      console.log('✅ WebSocket port 8080 is accessible');
      
    } catch (error) {
      this.results.websocket = {
        status: 'FAILED',
        error: error.message
      };
      console.log(`❌ WebSocket test failed: ${error.message}`);
    }
  }

  async testAllTools() {
    console.log('\n🐝 Testing All 10 Parasitic Trading Tools...');
    
    const tools = [
      { name: 'scan_parasitic_opportunities', args: { min_volume: 100000 } },
      { name: 'detect_whale_nests', args: { min_whale_size: 1000000 } },
      { name: 'identify_zombie_pairs', args: { min_predictability: 0.8 } },
      { name: 'analyze_mycelial_network', args: { correlation_threshold: 0.6 } },
      { name: 'activate_octopus_camouflage', args: { threat_level: 'medium' } },
      { name: 'deploy_anglerfish_lure', args: { lure_pairs: ['BTCUSDT'] } },
      { name: 'track_wounded_pairs', args: { volatility_threshold: 0.05 } },
      { name: 'enter_cryptobiosis', args: { trigger_conditions: {} } },
      { name: 'electric_shock', args: { shock_pairs: ['ADAUSDT'] } },
      { name: 'electroreception_scan', args: { sensitivity: 0.9 } }
    ];

    let successCount = 0;
    let totalExecutionTime = 0;

    for (const tool of tools) {
      const result = await this.testTool(tool);
      if (result.status === 'SUCCESS') successCount++;
      totalExecutionTime += result.execution_time || 0;
    }

    this.results.performance = {
      total_tools: tools.length,
      successful_tools: successCount,
      success_rate: (successCount / tools.length) * 100,
      total_execution_time_ms: totalExecutionTime,
      average_execution_time_ms: totalExecutionTime / tools.length,
      real_implementation_rate: this.calculateRealImplementationRate()
    };

    console.log(`\n📊 Performance Summary:`);
    console.log(`   Success Rate: ${this.results.performance.success_rate.toFixed(1)}%`);
    console.log(`   Average Execution Time: ${this.results.performance.average_execution_time_ms.toFixed(0)}ms`);
  }

  async testTool(tool) {
    console.log(`   Testing ${tool.name}...`);
    
    try {
      const startTime = Date.now();
      
      // Load tool module
      const toolPath = path.join(process.cwd(), 'mcp', 'tools', `${tool.name}.js`);
      const toolModule = require(toolPath);
      
      // Create system state
      const systemState = new Map();
      systemState.set('server_info', {
        name: 'parasitic-trading-mcp',
        version: '2.0.0',
        status: 'active'
      });
      
      // Execute tool with fallback handling
      const result = await toolModule.execute(tool.args, systemState);
      const executionTime = Date.now() - startTime;
      
      const toolResult = {
        status: 'SUCCESS',
        execution_time: executionTime,
        has_result: !!result,
        has_error: !!result.error,
        fallback_used: !!result.fallback_analysis || !!result.fallback_data || !!result.fallback_mode,
        cqgs_compliant: result.cqgs_compliance !== 'failed',
        real_data: !result.fallback_mode && result.performance?.zero_mock_compliance === 1.0
      };

      this.results.tools[tool.name] = toolResult;
      
      const statusIcon = toolResult.has_error ? '⚠️' : '✅';
      const fallbackNote = toolResult.fallback_used ? ' (fallback)' : '';
      console.log(`   ${statusIcon} ${tool.name} completed${fallbackNote} (${executionTime}ms)`);
      
      return toolResult;
      
    } catch (error) {
      const toolResult = {
        status: 'FAILED',
        execution_time: 0,
        error: error.message
      };
      
      this.results.tools[tool.name] = toolResult;
      console.log(`   ❌ ${tool.name} failed: ${error.message}`);
      
      return toolResult;
    }
  }

  calculateRealImplementationRate() {
    const tools = Object.values(this.results.tools);
    const realDataTools = tools.filter(t => t.real_data).length;
    return (realDataTools / tools.length) * 100;
  }

  generateReport() {
    console.log('\n📄 Generating Test Report...');
    
    const toolResults = Object.values(this.results.tools);
    const successfulTools = toolResults.filter(t => t.status === 'SUCCESS').length;
    const failedTools = toolResults.filter(t => t.status === 'FAILED').length;
    const fallbackTools = toolResults.filter(t => t.fallback_used).length;
    const cqgsCompliantTools = toolResults.filter(t => t.cqgs_compliant).length;

    this.results.summary = {
      overall_status: this.determineOverallStatus(),
      tools_tested: 10,
      successful_tools: successfulTools,
      failed_tools: failedTools,
      fallback_tools: fallbackTools,
      success_rate: (successfulTools / 10) * 100,
      cqgs_compliance_rate: (cqgsCompliantTools / 10) * 100,
      websocket_status: this.results.websocket?.status,
      blueprint_compliance: this.assessBlueprintCompliance()
    };

    this.printReport();
  }

  determineOverallStatus() {
    const successRate = this.results.performance?.success_rate || 0;
    const websocketStatus = this.results.websocket?.status;
    
    if (websocketStatus === 'SUCCESS' && successRate >= 90) {
      return 'SUCCESS';
    } else if (websocketStatus === 'SUCCESS' && successRate >= 70) {
      return 'PARTIAL_SUCCESS';
    } else {
      return 'NEEDS_IMPROVEMENT';
    }
  }

  assessBlueprintCompliance() {
    const criteria = {
      all_10_tools_present: Object.keys(this.results.tools).length === 10,
      websocket_port_8080: this.results.websocket?.status === 'SUCCESS',
      tools_executable: this.results.performance?.success_rate >= 80,
      cqgs_compliance: (Object.values(this.results.tools).filter(t => t.cqgs_compliant).length / 10) >= 0.8
    };

    const score = Object.values(criteria).filter(Boolean).length / Object.keys(criteria).length;
    
    return {
      score: score,
      status: score >= 0.8 ? 'COMPLIANT' : 'PARTIAL_COMPLIANCE',
      criteria: criteria
    };
  }

  printReport() {
    console.log('\n' + '='.repeat(60));
    console.log('🐝 PARASITIC MCP SERVER TEST REPORT');
    console.log('='.repeat(60));
    console.log(`Test Date: ${this.results.timestamp}`);
    console.log(`Overall Status: ${this.results.summary.overall_status}`);
    console.log('');
    
    // MCP Server Status
    console.log('MCP SERVER STATUS:');
    console.log(`  WebSocket (port 8080): ${this.results.websocket?.status || 'NOT_TESTED'}`);
    console.log('');
    
    // Tool Execution Results
    console.log('TOOL EXECUTION RESULTS:');
    console.log(`  Total Tools: ${this.results.summary.tools_tested}`);
    console.log(`  Successful: ${this.results.summary.successful_tools}`);
    console.log(`  Failed: ${this.results.summary.failed_tools}`);
    console.log(`  Using Fallback: ${this.results.summary.fallback_tools}`);
    console.log(`  Success Rate: ${this.results.summary.success_rate.toFixed(1)}%`);
    console.log(`  CQGS Compliance: ${this.results.summary.cqgs_compliance_rate.toFixed(1)}%`);
    console.log('');
    
    // Performance Metrics
    if (this.results.performance.total_execution_time_ms) {
      console.log('PERFORMANCE METRICS:');
      console.log(`  Total Execution Time: ${this.results.performance.total_execution_time_ms}ms`);
      console.log(`  Average per Tool: ${this.results.performance.average_execution_time_ms.toFixed(0)}ms`);
      console.log(`  Real Implementation Rate: ${this.results.performance.real_implementation_rate.toFixed(1)}%`);
      console.log('');
    }
    
    // Blueprint Compliance
    console.log('BLUEPRINT COMPLIANCE:');
    console.log(`  Compliance Status: ${this.results.summary.blueprint_compliance.status}`);
    console.log(`  Compliance Score: ${(this.results.summary.blueprint_compliance.score * 100).toFixed(1)}%`);
    console.log(`  All 10 Tools Present: ${this.results.summary.blueprint_compliance.criteria.all_10_tools_present ? 'YES' : 'NO'}`);
    console.log(`  WebSocket Ready: ${this.results.summary.blueprint_compliance.criteria.websocket_port_8080 ? 'YES' : 'NO'}`);
    console.log(`  Tools Executable: ${this.results.summary.blueprint_compliance.criteria.tools_executable ? 'YES' : 'NO'}`);
    console.log(`  CQGS Compliant: ${this.results.summary.blueprint_compliance.criteria.cqgs_compliance ? 'YES' : 'NO'}`);
    console.log('');
    
    // Individual Tool Results
    console.log('INDIVIDUAL TOOL RESULTS:');
    Object.entries(this.results.tools).forEach(([name, result]) => {
      const status = result.status === 'SUCCESS' ? 
        (result.has_error ? '⚠️' : '✅') : '❌';
      const time = result.execution_time ? `${result.execution_time}ms` : 'N/A';
      const notes = [];
      
      if (result.fallback_used) notes.push('fallback');
      if (result.has_error) notes.push('with errors');
      if (!result.cqgs_compliant) notes.push('non-compliant');
      
      const noteStr = notes.length > 0 ? ` (${notes.join(', ')})` : '';
      console.log(`  ${status} ${name} - ${time}${noteStr}`);
    });
    
    console.log('');
    
    // Recommendations
    const recommendations = this.generateRecommendations();
    if (recommendations.length > 0) {
      console.log('RECOMMENDATIONS:');
      recommendations.forEach(rec => {
        console.log(`  - ${rec}`);
      });
      console.log('');
    }
    
    console.log('='.repeat(60));
    
    // Save report to file
    const fs = require('fs');
    fs.writeFileSync('tests/mcp_quick_test_report.json', JSON.stringify(this.results, null, 2));
    console.log('📄 Report saved to: tests/mcp_quick_test_report.json');
  }

  generateRecommendations() {
    const recommendations = [];
    
    if (this.results.websocket?.status === 'FAILED') {
      recommendations.push('Ensure MCP server is running with WebSocket on port 8080');
    }
    
    if (this.results.summary.success_rate < 90) {
      recommendations.push('Investigate and fix failed tool executions');
    }
    
    if (this.results.summary.fallback_tools > 5) {
      recommendations.push('Build Rust backend to enable full functionality');
    }
    
    if (this.results.summary.cqgs_compliance_rate < 80) {
      recommendations.push('Improve CQGS compliance in tool implementations');
    }
    
    return recommendations;
  }
}

// Run the test
if (require.main === module) {
  const tester = new QuickMCPTester();
  tester.runQuickTest().catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
  });
}

module.exports = { QuickMCPTester };