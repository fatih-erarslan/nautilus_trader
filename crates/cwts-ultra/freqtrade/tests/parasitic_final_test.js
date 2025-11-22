#!/usr/bin/env node

/**
 * Final Comprehensive Test for Parasitic MCP Server WebSocket
 * Tests port 8081 connectivity and all 4 required tool calls
 */

const WebSocket = require('ws');

class ParasiticFinalTester {
  constructor() {
    this.results = {
      websocket_connection: null,
      tool_tests: {},
      subscriptions: {},
      performance_metrics: {},
      overall_status: 'PENDING'
    };
    this.ws = null;
    this.messageId = 1;
    this.responses = new Map();
  }

  async runTests() {
    console.log('🧪 Final Test: Parasitic MCP Server WebSocket on Port 8081');
    console.log('Testing: scan_parasitic_opportunities, detect_whale_nests, analyze_mycelial_network, electroreception_scan');
    console.log('='.repeat(80));

    try {
      // Step 1: Test WebSocket Connection
      await this.testWebSocketConnection();

      // Step 2: Test Required Tool Calls
      await this.testRequiredTools();

      // Step 3: Test Subscriptions
      await this.testSubscriptions();

      // Step 4: Generate Final Report
      this.generateFinalReport();

    } catch (error) {
      console.error('❌ Test suite failed:', error.message);
      this.results.overall_status = 'FAILED';
      this.results.error = error.message;
    } finally {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
    }
  }

  async testWebSocketConnection() {
    console.log('\n🔌 Testing WebSocket Connection to ws://localhost:8081...');
    
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      this.ws = new WebSocket('ws://localhost:8081');

      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout (10s)'));
      }, 10000);

      this.ws.on('open', () => {
        const connectionTime = Date.now() - startTime;
        clearTimeout(timeout);
        
        this.results.websocket_connection = {
          status: 'SUCCESS',
          connection_time_ms: connectionTime,
          server_port: 8081,
          protocol: 'WebSocket'
        };

        console.log(`✅ WebSocket connected successfully (${connectionTime}ms)`);
        
        // Setup message handler
        this.ws.on('message', (data) => {
          this.handleMessage(data);
        });

        resolve();
      });

      this.ws.on('error', (error) => {
        clearTimeout(timeout);
        this.results.websocket_connection = {
          status: 'FAILED',
          error: error.message
        };
        reject(error);
      });

      this.ws.on('close', () => {
        console.log('🔌 WebSocket connection closed');
      });
    });
  }

  handleMessage(data) {
    try {
      const message = JSON.parse(data.toString());
      console.log('📥 Received:', JSON.stringify(message, null, 2));
      
      // Store response if it has an ID
      if (message.id) {
        this.responses.set(message.id, message);
      }
    } catch (error) {
      console.error('Message parse error:', error.message);
    }
  }

  async testRequiredTools() {
    console.log('\n🛠️ Testing Required Tool Calls...');
    
    const requiredTools = [
      {
        method: 'scan_parasitic_opportunities',
        params: {
          min_volume: 100000,
          organisms: ['cuckoo', 'wasp'],
          risk_limit: 0.1
        }
      },
      {
        method: 'detect_whale_nests',
        params: {
          min_whale_size: 1000000,
          vulnerability_threshold: 0.7
        }
      },
      {
        method: 'analyze_mycelial_network',
        params: {
          correlation_threshold: 0.6,
          network_depth: 3
        }
      },
      {
        method: 'electroreception_scan',
        params: {
          sensitivity: 0.9,
          frequency_range: [1, 100]
        }
      }
    ];

    for (const tool of requiredTools) {
      await this.testSingleTool(tool);
      // Wait between tool calls to avoid overwhelming the server
      await this.sleep(1500);
    }
  }

  async testSingleTool(tool) {
    const { method, params } = tool;
    console.log(`   🔧 Testing ${method}...`);
    
    const startTime = Date.now();
    const messageId = this.messageId++;
    
    try {
      // Send tool call message
      const message = {
        method: method,
        params: params,
        id: messageId
      };

      this.ws.send(JSON.stringify(message));
      
      // Wait for response
      const response = await this.waitForResponse(messageId, 10000);
      const executionTime = Date.now() - startTime;
      
      // Analyze response
      const success = !response.error;
      const hasData = response && typeof response === 'object' && Object.keys(response).length > 0;
      
      this.results.tool_tests[method] = {
        status: success ? 'SUCCESS' : 'FAILED',
        execution_time_ms: executionTime,
        request_params: params,
        response_received: !!response,
        has_data: hasData,
        error: response?.error,
        response_preview: this.getResponsePreview(response)
      };

      if (success) {
        console.log(`   ✅ ${method} succeeded (${executionTime}ms)`);
        if (response.opportunities?.length) {
          console.log(`      📊 Found ${response.opportunities.length} opportunities`);
        }
        if (response.performance?.execution_time_ms) {
          console.log(`      ⚡ Server execution: ${response.performance.execution_time_ms}ms`);
        }
      } else {
        console.log(`   ❌ ${method} failed: ${response?.error || 'No response'}`);
      }

    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      this.results.tool_tests[method] = {
        status: 'ERROR',
        execution_time_ms: executionTime,
        error: error.message
      };
      
      console.log(`   💥 ${method} error: ${error.message}`);
    }
  }

  async waitForResponse(messageId, timeout = 5000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (this.responses.has(messageId)) {
        const response = this.responses.get(messageId);
        this.responses.delete(messageId); // Clean up
        return response;
      }
      await this.sleep(100);
    }
    
    throw new Error(`Timeout waiting for response to message ${messageId}`);
  }

  getResponsePreview(response) {
    if (!response) return null;
    
    const preview = {};
    
    if (response.opportunities) {
      preview.opportunities_count = response.opportunities.length;
    }
    if (response.whale_nests) {
      preview.whale_nests_count = response.whale_nests.length;
    }
    if (response.network) {
      preview.network_nodes = response.network.nodes?.length || 0;
    }
    if (response.signals) {
      preview.signals_detected = response.signals.length;
    }
    if (response.performance) {
      preview.performance = {
        execution_time: response.performance.execution_time_ms,
        cqgs_compliance: response.performance.cqgs_compliance
      };
    }
    
    return Object.keys(preview).length > 0 ? preview : null;
  }

  async testSubscriptions() {
    console.log('\n📡 Testing WebSocket Subscriptions...');
    
    const subscriptions = ['market_data', 'system_status', 'organism_activity'];
    
    for (const resource of subscriptions) {
      try {
        const message = {
          type: 'subscribe',
          resource: resource
        };
        
        this.ws.send(JSON.stringify(message));
        console.log(`   ✅ Subscription request sent for: ${resource}`);
        
        this.results.subscriptions[resource] = {
          status: 'REQUESTED',
          timestamp: Date.now()
        };
        
      } catch (error) {
        console.log(`   ❌ Subscription failed for ${resource}: ${error.message}`);
        this.results.subscriptions[resource] = {
          status: 'FAILED',
          error: error.message
        };
      }
    }
    
    // Wait a bit for subscription confirmations
    await this.sleep(2000);
  }

  generateFinalReport() {
    console.log('\n📊 Final Test Results');
    console.log('='.repeat(80));
    
    // WebSocket Connection Status
    console.log('🔌 WebSocket Connection:');
    if (this.results.websocket_connection?.status === 'SUCCESS') {
      console.log('   ✅ PASS - Connected successfully to port 8081');
      console.log(`   ⚡ Connection time: ${this.results.websocket_connection.connection_time_ms}ms`);
    } else {
      console.log('   ❌ FAIL - Could not connect to WebSocket');
      if (this.results.websocket_connection?.error) {
        console.log(`   💥 Error: ${this.results.websocket_connection.error}`);
      }
    }

    // Tool Call Results
    console.log('\n🛠️ Required Tool Call Results:');
    const toolNames = Object.keys(this.results.tool_tests);
    const successfulTools = toolNames.filter(name => 
      this.results.tool_tests[name].status === 'SUCCESS'
    );

    console.log(`   📊 Tools tested: ${toolNames.length}/4`);
    console.log(`   ✅ Successful: ${successfulTools.length}/4`);
    console.log(`   ❌ Failed: ${toolNames.length - successfulTools.length}/4`);

    toolNames.forEach(toolName => {
      const result = this.results.tool_tests[toolName];
      const status = result.status === 'SUCCESS' ? '✅' : '❌';
      console.log(`   ${status} ${toolName}`);
      console.log(`      ⏱️  Execution time: ${result.execution_time_ms}ms`);
      
      if (result.response_preview) {
        console.log(`      📊 Response: ${JSON.stringify(result.response_preview)}`);
      }
      
      if (result.error) {
        console.log(`      💥 Error: ${result.error}`);
      }
    });

    // Subscriptions
    console.log('\n📡 Subscription Results:');
    const subscriptionNames = Object.keys(this.results.subscriptions);
    subscriptionNames.forEach(resource => {
      const result = this.results.subscriptions[resource];
      const status = result.status === 'REQUESTED' ? '✅' : '❌';
      console.log(`   ${status} ${resource}: ${result.status}`);
    });

    // Overall Assessment
    console.log('\n🎯 Overall Assessment:');
    const wsWorking = this.results.websocket_connection?.status === 'SUCCESS';
    const allToolsWork = successfulTools.length === 4;
    const mostToolsWork = successfulTools.length >= 3;

    if (wsWorking && allToolsWork) {
      this.results.overall_status = 'FULL_SUCCESS';
      console.log('   🟢 FULL SUCCESS: WebSocket working, all 4 tools operational');
    } else if (wsWorking && mostToolsWork) {
      this.results.overall_status = 'PARTIAL_SUCCESS';
      console.log('   🟡 PARTIAL SUCCESS: WebSocket working, most tools operational');
    } else if (wsWorking && successfulTools.length > 0) {
      this.results.overall_status = 'LIMITED_SUCCESS';
      console.log('   🟠 LIMITED SUCCESS: WebSocket working, some tools operational');
    } else if (wsWorking) {
      this.results.overall_status = 'CONNECTION_ONLY';
      console.log('   🔵 CONNECTION ONLY: WebSocket works but tools failing');
    } else {
      this.results.overall_status = 'FAILED';
      console.log('   🔴 FAILED: WebSocket connection failed');
    }

    // Identify Issues
    console.log('\n🚨 Issues and Gaps Identified:');
    const issues = [];
    
    if (!wsWorking) {
      issues.push('WebSocket connection to port 8081 failed');
    }
    
    toolNames.forEach(toolName => {
      const result = this.results.tool_tests[toolName];
      if (result.status !== 'SUCCESS') {
        issues.push(`Tool "${toolName}" is not working: ${result.error || 'Unknown error'}`);
      }
    });
    
    if (issues.length === 0) {
      console.log('   ✅ No critical issues found - Server is fully functional');
    } else {
      issues.forEach(issue => {
        console.log(`   ❌ ${issue}`);
      });
    }

    // Performance Metrics
    if (successfulTools.length > 0) {
      const avgExecutionTime = toolNames
        .filter(name => this.results.tool_tests[name].status === 'SUCCESS')
        .reduce((sum, name) => sum + this.results.tool_tests[name].execution_time_ms, 0) 
        / successfulTools.length;

      console.log('\n⚡ Performance Metrics:');
      console.log(`   📊 Average tool execution time: ${avgExecutionTime.toFixed(0)}ms`);
      console.log(`   📈 Success rate: ${(successfulTools.length / toolNames.length * 100).toFixed(1)}%`);
    }

    console.log('\n' + '='.repeat(80));
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Run the comprehensive test
if (require.main === module) {
  const tester = new ParasiticFinalTester();
  tester.runTests()
    .then(() => {
      const success = tester.results.overall_status.includes('SUCCESS');
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Test execution failed:', error);
      process.exit(1);
    });
}

module.exports = { ParasiticFinalTester };