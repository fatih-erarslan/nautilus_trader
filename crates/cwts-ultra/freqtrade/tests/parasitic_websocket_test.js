#!/usr/bin/env node

/**
 * Parasitic MCP Server WebSocket Testing Suite
 * Tests WebSocket functionality on port 8081 and validates tool calls
 */

const WebSocket = require('ws');
const http = require('http');

class ParasiticWebSocketTester {
  constructor() {
    this.testResults = {
      port_8081_connectivity: null,
      tool_calls: {},
      websocket_responses: [],
      server_discovery: null,
      error_log: []
    };
  }

  async runTests() {
    console.log('🧪 Testing Parasitic MCP Server WebSocket on Port 8081');
    console.log('='.repeat(60));
    
    try {
      // Test 1: Server Discovery
      await this.testServerDiscovery();
      
      // Test 2: WebSocket Connection on 8081
      await this.testWebSocketConnection();
      
      // Test 3: Tool Call Testing
      await this.testToolCalls();
      
      // Test 4: Generate Report
      this.generateReport();
      
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      this.testResults.error_log.push({
        timestamp: new Date().toISOString(),
        error: error.message,
        stack: error.stack
      });
    }
  }

  /**
   * Test server discovery
   */
  async testServerDiscovery() {
    console.log('\n🔍 Discovering Parasitic MCP Server...');
    
    try {
      // Check if port 8081 is listening
      const isListening = await this.checkPort(8081);
      
      this.testResults.server_discovery = {
        port_8081_listening: isListening,
        timestamp: new Date().toISOString()
      };
      
      if (isListening) {
        console.log('✅ Port 8081 is listening');
      } else {
        console.log('❌ Port 8081 is not accessible');
      }
      
    } catch (error) {
      this.testResults.server_discovery = {
        error: error.message
      };
      console.log('❌ Server discovery failed:', error.message);
    }
  }

  /**
   * Test WebSocket connection on port 8081
   */
  async testWebSocketConnection() {
    console.log('\n🔌 Testing WebSocket Connection on Port 8081...');
    
    const testUrls = [
      'ws://localhost:8081',
      'ws://localhost:8081/ws',
      'ws://localhost:8081/websocket',
      'ws://localhost:8081/mcp'
    ];
    
    for (const url of testUrls) {
      try {
        console.log(`   Trying ${url}...`);
        
        const connectionResult = await this.testSingleWebSocketConnection(url);
        
        if (connectionResult.success) {
          this.testResults.port_8081_connectivity = {
            status: 'SUCCESS',
            url: url,
            connection_time_ms: connectionResult.connectionTime,
            timestamp: new Date().toISOString()
          };
          
          console.log(`   ✅ Connected successfully to ${url}`);
          console.log(`   ⚡ Connection time: ${connectionResult.connectionTime}ms`);
          return; // Success, stop trying other URLs
        }
        
      } catch (error) {
        console.log(`   ❌ Failed to connect to ${url}: ${error.message}`);
      }
    }
    
    // If we reach here, all connections failed
    this.testResults.port_8081_connectivity = {
      status: 'FAILED',
      error: 'All WebSocket connection attempts failed',
      tested_urls: testUrls
    };
  }

  /**
   * Test single WebSocket connection
   */
  async testSingleWebSocketConnection(url) {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const ws = new WebSocket(url);
      
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('Connection timeout'));
      }, 5000);
      
      ws.on('open', () => {
        const connectionTime = Date.now() - startTime;
        clearTimeout(timeout);
        
        // Test basic communication
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          method: 'initialize',
          id: 1,
          params: {
            protocolVersion: '2024-11-05',
            capabilities: {}
          }
        }));
        
        ws.close();
        resolve({ success: true, connectionTime });
      });
      
      ws.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      
      ws.on('message', (data) => {
        this.testResults.websocket_responses.push({
          timestamp: new Date().toISOString(),
          data: data.toString()
        });
      });
    });
  }

  /**
   * Test MCP tool calls through direct HTTP or available mechanism
   */
  async testToolCalls() {
    console.log('\n🛠️ Testing Parasitic Trading Tools...');
    
    const toolsToTest = [
      'scan_parasitic_opportunities',
      'detect_whale_nests',
      'analyze_mycelial_network',
      'electroreception_scan'
    ];
    
    for (const toolName of toolsToTest) {
      await this.testSingleToolCall(toolName);
    }
  }

  /**
   * Test single tool call
   */
  async testSingleToolCall(toolName) {
    console.log(`   Testing ${toolName}...`);
    
    try {
      // Check if tool file exists and can be loaded
      const toolPath = `/home/kutlu/CWTS/cwts-ultra/parasitic/mcp/tools/${toolName}.js`;
      
      const toolExists = await this.checkFileExists(toolPath);
      
      if (toolExists) {
        // Try to load and test the tool directly
        try {
          const toolModule = require(toolPath);
          
          // Test with sample arguments
          const sampleArgs = this.getSampleArgsForTool(toolName);
          const mockSystemState = new Map();
          mockSystemState.set('server_info', {
            name: 'parasitic-trading-mcp',
            version: '2.0.0',
            status: 'active'
          });
          
          const startTime = Date.now();
          const result = await toolModule.execute(sampleArgs, mockSystemState);
          const executionTime = Date.now() - startTime;
          
          this.testResults.tool_calls[toolName] = {
            status: 'SUCCESS',
            execution_time_ms: executionTime,
            has_result: !!result,
            sample_args: sampleArgs,
            result_preview: this.getResultPreview(result)
          };
          
          console.log(`   ✅ ${toolName} executed successfully (${executionTime}ms)`);
          
        } catch (executionError) {
          this.testResults.tool_calls[toolName] = {
            status: 'EXECUTION_FAILED',
            error: executionError.message,
            tool_exists: true
          };
          console.log(`   ⚠️ ${toolName} exists but execution failed: ${executionError.message}`);
        }
        
      } else {
        this.testResults.tool_calls[toolName] = {
          status: 'NOT_FOUND',
          tool_path: toolPath
        };
        console.log(`   ❌ ${toolName} tool file not found`);
      }
      
    } catch (error) {
      this.testResults.tool_calls[toolName] = {
        status: 'FAILED',
        error: error.message
      };
      console.log(`   ❌ ${toolName} test failed: ${error.message}`);
    }
  }

  /**
   * Get sample arguments for different tools
   */
  getSampleArgsForTool(toolName) {
    const sampleArgs = {
      scan_parasitic_opportunities: {
        min_volume: 100000,
        organisms: ['cuckoo', 'wasp'],
        risk_limit: 0.1
      },
      detect_whale_nests: {
        min_whale_size: 1000000,
        vulnerability_threshold: 0.7
      },
      analyze_mycelial_network: {
        correlation_threshold: 0.6,
        network_depth: 3
      },
      electroreception_scan: {
        sensitivity: 0.9,
        frequency_range: [1, 100]
      }
    };
    
    return sampleArgs[toolName] || {};
  }

  /**
   * Get result preview for reporting
   */
  getResultPreview(result) {
    if (!result) return null;
    
    return {
      has_opportunities: !!result.opportunities,
      opportunity_count: result.opportunities?.length || 0,
      has_performance: !!result.performance,
      cqgs_compliance: result.cqgs_compliance,
      execution_success: !result.error
    };
  }

  /**
   * Check if port is listening
   */
  async checkPort(port) {
    return new Promise((resolve) => {
      const req = http.request({
        hostname: 'localhost',
        port: port,
        method: 'GET',
        timeout: 3000
      }, (res) => {
        resolve(true);
      });
      
      req.on('error', () => {
        resolve(false);
      });
      
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
      
      req.end();
    });
  }

  /**
   * Check if file exists
   */
  async checkFileExists(filePath) {
    try {
      const fs = require('fs');
      await fs.promises.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Generate test report
   */
  generateReport() {
    console.log('\n📊 Test Results Summary');
    console.log('='.repeat(60));
    
    // Server Discovery
    console.log('🔍 Server Discovery:');
    if (this.testResults.server_discovery?.port_8081_listening) {
      console.log('   ✅ Port 8081 is listening');
    } else {
      console.log('   ❌ Port 8081 is not accessible');
    }
    
    // WebSocket Connectivity
    console.log('\n🔌 WebSocket Connectivity (Port 8081):');
    if (this.testResults.port_8081_connectivity?.status === 'SUCCESS') {
      console.log('   ✅ WebSocket connection successful');
      console.log(`   📡 Connected to: ${this.testResults.port_8081_connectivity.url}`);
      console.log(`   ⚡ Connection time: ${this.testResults.port_8081_connectivity.connection_time_ms}ms`);
    } else {
      console.log('   ❌ WebSocket connection failed');
      if (this.testResults.port_8081_connectivity?.error) {
        console.log(`   💥 Error: ${this.testResults.port_8081_connectivity.error}`);
      }
    }
    
    // Tool Calls
    console.log('\n🛠️ Tool Call Results:');
    const toolResults = this.testResults.tool_calls;
    const toolNames = Object.keys(toolResults);
    const successfulTools = toolNames.filter(name => toolResults[name].status === 'SUCCESS');
    
    console.log(`   📊 Tools tested: ${toolNames.length}`);
    console.log(`   ✅ Successful: ${successfulTools.length}`);
    console.log(`   ❌ Failed: ${toolNames.length - successfulTools.length}`);
    
    toolNames.forEach(toolName => {
      const result = toolResults[toolName];
      const status = result.status === 'SUCCESS' ? '✅' : '❌';
      console.log(`   ${status} ${toolName}: ${result.status}`);
      if (result.execution_time_ms) {
        console.log(`      ⚡ Execution time: ${result.execution_time_ms}ms`);
      }
    });
    
    // Overall Assessment
    console.log('\n🎯 Overall Assessment:');
    const serverListening = this.testResults.server_discovery?.port_8081_listening;
    const websocketWorking = this.testResults.port_8081_connectivity?.status === 'SUCCESS';
    const toolsWorking = successfulTools.length > 0;
    
    if (serverListening && websocketWorking && toolsWorking) {
      console.log('   🟢 PASS: Server is functional with working tools');
    } else if (serverListening && (websocketWorking || toolsWorking)) {
      console.log('   🟡 PARTIAL: Server is running but has some issues');
    } else {
      console.log('   🔴 FAIL: Server has significant issues');
    }
    
    // Specific Issues Found
    const issues = [];
    if (!serverListening) issues.push('Port 8081 not listening');
    if (!websocketWorking) issues.push('WebSocket connection failed');
    if (!toolsWorking) issues.push('No tools are working');
    
    if (issues.length > 0) {
      console.log('\n🚨 Issues Identified:');
      issues.forEach(issue => console.log(`   - ${issue}`));
    }
    
    // WebSocket Responses
    if (this.testResults.websocket_responses.length > 0) {
      console.log('\n📨 WebSocket Responses Received:');
      this.testResults.websocket_responses.forEach((response, index) => {
        console.log(`   ${index + 1}. ${response.data}`);
      });
    }
    
    console.log('\n='.repeat(60));
  }
}

// Run the test suite
if (require.main === module) {
  const tester = new ParasiticWebSocketTester();
  tester.runTests().catch(error => {
    console.error('Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = { ParasiticWebSocketTester };