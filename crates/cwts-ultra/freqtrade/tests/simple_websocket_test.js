#!/usr/bin/env node

/**
 * Simple WebSocket Test for Parasitic MCP Server
 * Tests basic connectivity and MCP protocol communication
 */

const WebSocket = require('ws');

console.log('🧪 Testing Parasitic MCP Server WebSocket on Port 8081');
console.log('='.repeat(60));

async function testWebSocketConnection() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket('ws://localhost:8081');
    
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error('Connection timeout'));
    }, 10000);
    
    ws.on('open', () => {
      console.log('✅ WebSocket connection established');
      clearTimeout(timeout);
      
      // Send MCP initialize message
      const initMessage = {
        jsonrpc: '2.0',
        method: 'initialize',
        id: 1,
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {
            tools: { listChanged: true }
          }
        }
      };
      
      console.log('📤 Sending initialize message...');
      ws.send(JSON.stringify(initMessage));
    });
    
    ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      console.log('📥 Received message:', JSON.stringify(message, null, 2));
      
      if (message.id === 1) {
        // Response to initialize - now test tool calls
        testToolCalls(ws);
      }
    });
    
    ws.on('error', (error) => {
      clearTimeout(timeout);
      console.log('❌ WebSocket error:', error.message);
      reject(error);
    });
    
    ws.on('close', () => {
      clearTimeout(timeout);
      console.log('🔌 WebSocket connection closed');
      resolve();
    });
  });
}

async function testToolCalls(ws) {
  console.log('\n🛠️ Testing MCP Tool Calls...');
  
  const toolsToTest = [
    {
      name: 'scan_parasitic_opportunities',
      args: {
        min_volume: 100000,
        organisms: ['cuckoo', 'wasp'],
        risk_limit: 0.1
      }
    },
    {
      name: 'detect_whale_nests',
      args: {
        min_whale_size: 1000000,
        vulnerability_threshold: 0.7
      }
    },
    {
      name: 'analyze_mycelial_network',
      args: {
        correlation_threshold: 0.6,
        network_depth: 3
      }
    },
    {
      name: 'electroreception_scan',
      args: {
        sensitivity: 0.9,
        frequency_range: [1, 100]
      }
    }
  ];
  
  for (let i = 0; i < toolsToTest.length; i++) {
    const tool = toolsToTest[i];
    const messageId = i + 2;
    
    const toolMessage = {
      jsonrpc: '2.0',
      method: 'tools/call',
      id: messageId,
      params: {
        name: tool.name,
        arguments: tool.args
      }
    };
    
    console.log(`📤 Testing ${tool.name}...`);
    ws.send(JSON.stringify(toolMessage));
    
    // Wait between tool calls
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  // Close after testing
  setTimeout(() => {
    ws.close();
  }, 5000);
}

// Run the test
testWebSocketConnection()
  .then(() => {
    console.log('\n✅ WebSocket testing completed successfully');
  })
  .catch((error) => {
    console.error('\n❌ WebSocket testing failed:', error.message);
    process.exit(1);
  });