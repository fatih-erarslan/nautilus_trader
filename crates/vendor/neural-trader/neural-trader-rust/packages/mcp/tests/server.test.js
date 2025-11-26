/**
 * MCP Server Tests
 * Comprehensive test suite for MCP 2025-11 compliance
 */

const { describe, it, before, after } = require('mocha');
const { expect } = require('chai');
const { McpServer } = require('../src/server');
const { JsonRpcHandler } = require('../src/protocol/jsonrpc');
const path = require('path');

describe('MCP Server', function() {
  this.timeout(10000);
  let server;

  before(async function() {
    server = new McpServer({
      toolsDir: path.join(__dirname, '../tools'),
      enableRustBridge: false, // Use stub mode for testing
      enableAuditLog: false,
    });
    await server.initialize();
  });

  after(async function() {
    if (server) {
      await server.stop();
    }
  });

  describe('Server Initialization', function() {
    it('should initialize successfully', function() {
      expect(server).to.exist;
      expect(server.running).to.be.false; // Not started yet
    });

    it('should load tool registry', function() {
      expect(server.registry).to.exist;
      expect(server.registry.tools.size).to.be.greaterThan(0);
    });

    it('should have JSON-RPC handler', function() {
      expect(server.rpc).to.exist;
      expect(server.rpc).to.be.instanceOf(JsonRpcHandler);
    });
  });

  describe('MCP Protocol Methods', function() {
    it('should handle initialize method', async function() {
      const result = await server.handleInitialize({});
      expect(result).to.have.property('protocolVersion', '2025-11');
      expect(result).to.have.property('serverInfo');
      expect(result.serverInfo).to.have.property('name', 'Neural Trader MCP Server');
      expect(result).to.have.property('capabilities');
    });

    it('should handle tools/list method', async function() {
      const result = await server.handleToolsList({});
      expect(result).to.have.property('tools');
      expect(result.tools).to.be.an('array');
      expect(result.tools.length).to.be.greaterThan(0);

      const tool = result.tools[0];
      expect(tool).to.have.property('name');
      expect(tool).to.have.property('description');
      expect(tool).to.have.property('inputSchema');
    });

    it('should handle tools/schema method', async function() {
      const toolName = server.registry.listTools()[0].name;
      const result = await server.handleToolsSchema({ name: toolName });
      expect(result).to.have.property('schema');
      expect(result).to.have.property('etag');
    });

    it('should handle tools/call method in stub mode', async function() {
      const toolName = server.registry.listTools()[0].name;
      const result = await server.handleToolsCall({
        name: toolName,
        arguments: {}
      });
      expect(result).to.have.property('content');
      expect(result.content).to.be.an('array');
      expect(result.content[0]).to.have.property('type', 'text');
    });

    it('should handle server/info method', async function() {
      const result = await server.handleServerInfo({});
      expect(result).to.have.property('name');
      expect(result).to.have.property('version');
      expect(result).to.have.property('protocol', 'MCP 2025-11');
      expect(result).to.have.property('toolsCount');
    });

    it('should handle ping method', async function() {
      const result = await server.handlePing({});
      expect(result).to.have.property('status', 'ok');
      expect(result).to.have.property('timestamp');
      expect(result).to.have.property('uptime');
    });
  });

  describe('Tool Registry', function() {
    it('should have core trading tools', function() {
      const hasTool = server.registry.hasTool('list_strategies');
      expect(hasTool).to.be.true;
    });

    it('should have neural network tools', function() {
      const hasTool = server.registry.hasTool('neural_train');
      expect(hasTool).to.be.true;
    });

    it('should have sports betting tools', function() {
      const hasTool = server.registry.hasTool('get_sports_odds');
      expect(hasTool).to.be.true;
    });

    it('should search tools by keyword', async function() {
      const results = await server.handleToolsSearch({ query: 'neural' });
      expect(results.results).to.be.an('array');
      expect(results.results.length).to.be.greaterThan(0);
    });

    it('should list tool categories', async function() {
      const results = await server.handleToolsCategories({});
      expect(results.categories).to.be.an('array');
      expect(results.categories.length).to.be.greaterThan(0);
    });
  });

  describe('Error Handling', function() {
    it('should throw error for unknown tool', async function() {
      try {
        await server.handleToolsCall({
          name: 'nonexistent_tool',
          arguments: {}
        });
        expect.fail('Should have thrown error');
      } catch (error) {
        expect(error.message).to.include('Tool not found');
      }
    });

    it('should throw error for unknown schema', async function() {
      try {
        await server.handleToolsSchema({ name: 'nonexistent_tool' });
        expect.fail('Should have thrown error');
      } catch (error) {
        expect(error.message).to.include('Tool not found');
      }
    });
  });
});

describe('JSON-RPC Protocol', function() {
  let handler;

  before(function() {
    handler = new JsonRpcHandler();
    handler.register('test_method', async (params) => {
      return { result: 'success', params };
    });
  });

  describe('Request Processing', function() {
    it('should process valid JSON-RPC request', async function() {
      const request = JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'test_method',
        params: { foo: 'bar' }
      });

      const response = await handler.process(request);
      const parsed = JSON.parse(response);

      expect(parsed).to.have.property('jsonrpc', '2.0');
      expect(parsed).to.have.property('id', 1);
      expect(parsed).to.have.property('result');
      expect(parsed.result).to.have.property('result', 'success');
    });

    it('should handle notification (no id)', async function() {
      const request = JSON.stringify({
        jsonrpc: '2.0',
        method: 'test_method',
        params: {}
      });

      const response = await handler.process(request);
      expect(response).to.be.null; // Notifications don't get responses
    });

    it('should return error for parse errors', async function() {
      const request = 'invalid json';
      const response = await handler.process(request);
      const parsed = JSON.parse(response);

      expect(parsed).to.have.property('error');
      expect(parsed.error).to.have.property('code', -32700);
    });

    it('should return error for method not found', async function() {
      const request = JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'unknown_method',
        params: {}
      });

      const response = await handler.process(request);
      const parsed = JSON.parse(response);

      expect(parsed).to.have.property('error');
      expect(parsed.error).to.have.property('code', -32601);
    });
  });

  describe('Batch Processing', function() {
    it('should process batch requests', async function() {
      const request = JSON.stringify([
        { jsonrpc: '2.0', id: 1, method: 'test_method', params: {} },
        { jsonrpc: '2.0', id: 2, method: 'test_method', params: {} }
      ]);

      const response = await handler.process(request);
      const parsed = JSON.parse(response);

      expect(parsed).to.be.an('array');
      expect(parsed).to.have.lengthOf(2);
      expect(parsed[0]).to.have.property('id', 1);
      expect(parsed[1]).to.have.property('id', 2);
    });

    it('should handle mixed batch (success + error)', async function() {
      const request = JSON.stringify([
        { jsonrpc: '2.0', id: 1, method: 'test_method', params: {} },
        { jsonrpc: '2.0', id: 2, method: 'unknown', params: {} }
      ]);

      const response = await handler.process(request);
      const parsed = JSON.parse(response);

      expect(parsed).to.be.an('array');
      expect(parsed[0]).to.have.property('result');
      expect(parsed[1]).to.have.property('error');
    });
  });
});
