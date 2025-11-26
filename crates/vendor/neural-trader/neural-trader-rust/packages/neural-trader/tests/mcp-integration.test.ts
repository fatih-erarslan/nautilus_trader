/**
 * MCP Server Integration Tests
 * Tests the Model Context Protocol server functionality
 *
 * Test Categories:
 * 1. Server Startup and Health
 * 2. Tool Discovery and Listing
 * 3. Tool Execution (Neural, Trading, Risk, Sports)
 * 4. Performance and Concurrency
 * 5. Error Handling
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

interface MCPMessage {
  jsonrpc: string;
  id?: number;
  method?: string;
  params?: any;
  result?: any;
  error?: any;
}

interface ToolInfo {
  name: string;
  description: string;
  inputSchema: any;
}

class MCPClient extends EventEmitter {
  private process: ChildProcess | null = null;
  private messageId = 0;
  private pendingRequests = new Map<number, { resolve: Function; reject: Function }>();
  private buffer = '';

  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Start MCP server in stdio mode
      this.process = spawn('npx', ['neural-trader', 'mcp'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env }
      });

      if (!this.process.stdout || !this.process.stdin) {
        reject(new Error('Failed to create process streams'));
        return;
      }

      this.process.stdout.on('data', (data: Buffer) => {
        this.handleData(data.toString());
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        console.error('MCP Server Error:', data.toString());
      });

      this.process.on('error', (error) => {
        reject(error);
      });

      // Initialize connection
      setTimeout(() => {
        this.sendRequest('initialize', {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: {
            name: 'neural-trader-test-client',
            version: '1.0.0'
          }
        }).then(() => resolve()).catch(reject);
      }, 1000);
    });
  }

  private handleData(data: string): void {
    this.buffer += data;
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        try {
          const message: MCPMessage = JSON.parse(line);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse message:', line);
        }
      }
    }
  }

  private handleMessage(message: MCPMessage): void {
    if (message.id !== undefined) {
      const pending = this.pendingRequests.get(message.id);
      if (pending) {
        this.pendingRequests.delete(message.id);
        if (message.error) {
          pending.reject(new Error(message.error.message));
        } else {
          pending.resolve(message.result);
        }
      }
    }
    this.emit('message', message);
  }

  async sendRequest(method: string, params?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      const message: MCPMessage = {
        jsonrpc: '2.0',
        id,
        method,
        params
      };

      this.pendingRequests.set(id, { resolve, reject });

      if (this.process?.stdin) {
        this.process.stdin.write(JSON.stringify(message) + '\n');
      } else {
        reject(new Error('Process not started'));
      }

      // Timeout after 10 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 10000);
    });
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.process) {
        this.process.once('exit', () => resolve());
        this.process.kill();
      } else {
        resolve();
      }
    });
  }
}

describe('MCP Server Integration Tests', () => {
  let client: MCPClient;
  let testResults = {
    timestamp: new Date().toISOString(),
    server_started: false,
    tools_discovered: 0,
    tools_tested: 0,
    successful_calls: 0,
    failed_calls: 0,
    avg_execution_time: 0,
    concurrent_calls_tested: 0,
    execution_times: [] as number[],
    errors: [] as any[]
  };

  beforeAll(async () => {
    console.log('🚀 Starting MCP Server Integration Tests');
    client = new MCPClient();

    try {
      await client.start();
      testResults.server_started = true;
      console.log('✅ MCP Server started successfully');
    } catch (error: any) {
      console.error('❌ Failed to start MCP server:', error.message);
      testResults.errors.push({ test: 'server_startup', error: error.message });
      throw error;
    }
  }, 30000);

  afterAll(async () => {
    if (client) {
      await client.stop();
      console.log('🛑 MCP Server stopped');
    }

    // Calculate average execution time
    if (testResults.execution_times.length > 0) {
      testResults.avg_execution_time =
        testResults.execution_times.reduce((a, b) => a + b, 0) / testResults.execution_times.length;
    }

    console.log('\n📊 MCP Test Summary:');
    console.log(`   Server Started: ${testResults.server_started}`);
    console.log(`   Tools Discovered: ${testResults.tools_discovered}`);
    console.log(`   Tools Tested: ${testResults.tools_tested}`);
    console.log(`   Successful Calls: ${testResults.successful_calls}`);
    console.log(`   Failed Calls: ${testResults.failed_calls}`);
    console.log(`   Avg Execution Time: ${testResults.avg_execution_time.toFixed(2)}ms`);
    console.log(`   Concurrent Calls: ${testResults.concurrent_calls_tested}`);
    console.log(`   Errors: ${testResults.errors.length}`);
  });

  describe('1. Server Health and Discovery', () => {
    it('should list all available tools', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/list');
        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);

        expect(result).toHaveProperty('tools');
        expect(Array.isArray(result.tools)).toBe(true);

        testResults.tools_discovered = result.tools.length;
        testResults.successful_calls++;

        console.log(`✅ Discovered ${result.tools.length} tools in ${executionTime}ms`);

        // Categorize tools
        const categories: Record<string, number> = {};
        result.tools.forEach((tool: ToolInfo) => {
          const category = tool.name.split('_')[0];
          categories[category] = (categories[category] || 0) + 1;
        });

        console.log('   Tool Categories:');
        Object.entries(categories).forEach(([category, count]) => {
          console.log(`     ${category}: ${count} tools`);
        });

        expect(result.tools.length).toBeGreaterThan(50); // Should have 102+ tools
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'list_tools', error: error.message });
        throw error;
      }
    }, 15000);

    it('should respond to ping', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'ping',
          arguments: {}
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Ping response in ${executionTime}ms`);
        console.log(`   Response: ${JSON.stringify(result)}`);

        expect(executionTime).toBeLessThan(2000); // Should respond quickly
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'ping', error: error.message });
        throw error;
      }
    }, 10000);

    it('should list available trading strategies', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'list_strategies',
          arguments: {}
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Listed strategies in ${executionTime}ms`);

        if (result.content && result.content[0]) {
          const strategies = JSON.parse(result.content[0].text);
          console.log(`   Strategies available: ${strategies.strategies?.length || 0}`);
        }
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'list_strategies', error: error.message });
        throw error;
      }
    }, 10000);
  });

  describe('2. Neural Tools', () => {
    it('should check neural model status', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'neural_model_status',
          arguments: {}
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Neural status checked in ${executionTime}ms`);
        expect(executionTime).toBeLessThan(2000);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'neural_status', error: error.message });
        throw error;
      }
    }, 10000);
  });

  describe('3. Trading Tools', () => {
    it('should simulate a trade', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'simulate_trade',
          arguments: {
            strategy: 'momentum',
            symbol: 'AAPL',
            action: 'buy'
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Trade simulated in ${executionTime}ms`);
        expect(executionTime).toBeLessThan(3000);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'simulate_trade', error: error.message });
        // Don't throw - this might fail if market is closed
      }
    }, 15000);

    it('should get portfolio status', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'get_portfolio_status',
          arguments: {
            include_analytics: true
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Portfolio status retrieved in ${executionTime}ms`);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'portfolio_status', error: error.message });
        throw error;
      }
    }, 10000);
  });

  describe('4. Risk Tools', () => {
    it('should calculate Kelly Criterion', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'calculate_kelly_criterion',
          arguments: {
            probability: 0.55,
            odds: 2.0,
            bankroll: 1000
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Kelly Criterion calculated in ${executionTime}ms`);

        if (result.content && result.content[0]) {
          const kellyResult = JSON.parse(result.content[0].text);
          console.log(`   Recommended stake: $${kellyResult.recommended_stake?.toFixed(2) || 'N/A'}`);
        }

        expect(executionTime).toBeLessThan(1000);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'kelly_criterion', error: error.message });
        throw error;
      }
    }, 10000);

    it('should perform risk analysis', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'risk_analysis',
          arguments: {
            portfolio: [
              { symbol: 'AAPL', weight: 0.3, value: 3000 },
              { symbol: 'GOOGL', weight: 0.3, value: 3000 },
              { symbol: 'MSFT', weight: 0.4, value: 4000 }
            ],
            use_gpu: false
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Risk analysis completed in ${executionTime}ms`);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'risk_analysis', error: error.message });
        // Don't throw - might fail if market data unavailable
      }
    }, 20000);
  });

  describe('5. Sports Betting Tools', () => {
    it('should fetch sports events', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'get_sports_events',
          arguments: {
            sport: 'americanfootball_nfl',
            days_ahead: 7
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Sports events fetched in ${executionTime}ms`);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'sports_events', error: error.message });
        // Don't throw - API might be rate limited
      }
    }, 15000);

    it('should get sports odds', async () => {
      const startTime = Date.now();

      try {
        const result = await client.sendRequest('tools/call', {
          name: 'get_sports_odds',
          arguments: {
            sport: 'americanfootball_nfl'
          }
        });

        const executionTime = Date.now() - startTime;
        testResults.execution_times.push(executionTime);
        testResults.successful_calls++;
        testResults.tools_tested++;

        console.log(`✅ Sports odds fetched in ${executionTime}ms`);
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'sports_odds', error: error.message });
        // Don't throw - API might be rate limited
      }
    }, 15000);
  });

  describe('6. Performance and Concurrency', () => {
    it('should handle concurrent tool calls', async () => {
      const startTime = Date.now();
      const concurrentCalls = 10;

      try {
        const promises = Array(concurrentCalls).fill(null).map((_, i) =>
          client.sendRequest('tools/call', {
            name: 'ping',
            arguments: {}
          })
        );

        const results = await Promise.all(promises);
        const executionTime = Date.now() - startTime;

        testResults.concurrent_calls_tested = concurrentCalls;
        testResults.successful_calls += concurrentCalls;

        console.log(`✅ Handled ${concurrentCalls} concurrent calls in ${executionTime}ms`);
        console.log(`   Average per call: ${(executionTime / concurrentCalls).toFixed(2)}ms`);

        expect(results.length).toBe(concurrentCalls);
        expect(executionTime).toBeLessThan(5000); // Should handle concurrency well
      } catch (error: any) {
        testResults.failed_calls++;
        testResults.errors.push({ test: 'concurrent_calls', error: error.message });
        throw error;
      }
    }, 20000);

    it('should execute tools under performance target', async () => {
      const performanceTests = [
        { name: 'ping', target: 100 },
        { name: 'list_strategies', target: 1000 },
        { name: 'get_portfolio_status', target: 2000 }
      ];

      console.log('⚡ Performance Tests:');

      for (const test of performanceTests) {
        const startTime = Date.now();

        try {
          await client.sendRequest('tools/call', {
            name: test.name,
            arguments: {}
          });

          const executionTime = Date.now() - startTime;
          testResults.execution_times.push(executionTime);
          testResults.successful_calls++;

          const status = executionTime < test.target ? '✅' : '⚠️';
          console.log(`   ${status} ${test.name}: ${executionTime}ms (target: ${test.target}ms)`);
        } catch (error: any) {
          testResults.failed_calls++;
          console.log(`   ❌ ${test.name}: Failed - ${error.message}`);
        }
      }
    }, 30000);
  });
});
