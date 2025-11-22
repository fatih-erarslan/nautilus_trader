#!/usr/bin/env node

/**
 * Debug Test for Parasitic MCP Server
 * Attempts to identify the exact issue with WebSocket tool calls
 */

const WebSocket = require('ws');

console.log('🔍 Parasitic MCP Server Debug Test');
console.log('='.repeat(50));

async function debugWebSocketCommunication() {
  const ws = new WebSocket('ws://localhost:8081');
  
  ws.on('open', () => {
    console.log('✅ WebSocket connected');
    
    // Test 1: Simple ping
    console.log('\n📤 Test 1: Sending simple message');
    ws.send(JSON.stringify({
      test: 'ping',
      timestamp: Date.now()
    }));
    
    setTimeout(() => {
      // Test 2: Tool call with different formats
      console.log('\n📤 Test 2: Direct tool call format');
      ws.send(JSON.stringify({
        method: 'scan_parasitic_opportunities',
        params: {
          min_volume: 100000,
          organisms: ['cuckoo'],
          risk_limit: 0.1
        }
      }));
      
      setTimeout(() => {
        // Test 3: MCP-style tool call
        console.log('\n📤 Test 3: MCP-style tool call');
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          method: 'tools/call',
          id: 123,
          params: {
            name: 'electroreception_scan',
            arguments: {
              sensitivity: 0.9,
              frequency_range: [1, 100]
            }
          }
        }));
        
        setTimeout(() => {
          // Test 4: Subscription test
          console.log('\n📤 Test 4: Subscription test');
          ws.send(JSON.stringify({
            type: 'subscribe',
            resource: 'market_data'
          }));
          
          setTimeout(() => {
            console.log('\n🔌 Closing connection...');
            ws.close();
          }, 3000);
        }, 2000);
      }, 2000);
    }, 2000);
  });
  
  ws.on('message', (data) => {
    console.log('📥 Received:', data.toString());
  });
  
  ws.on('error', (error) => {
    console.log('❌ WebSocket error:', error.message);
  });
  
  ws.on('close', (code, reason) => {
    console.log(`🔌 Connection closed: ${code} ${reason}`);
    console.log('\n📊 Debug Test Complete');
  });
}

debugWebSocketCommunication();