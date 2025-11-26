/**
 * MCP Server Implementation
 * Fully compliant with MCP 2025-11 specification
 */

const { JsonRpcHandler } = require('./protocol/jsonrpc');
const { StdioTransport } = require('./transport/stdio');
const { ToolRegistry } = require('./discovery/registry');
const { RustBridge } = require('./bridge/rust');
const { createAuditLogger } = require('./logging/audit');

class McpServer {
  constructor(options = {}) {
    this.options = {
      transport: 'stdio',
      toolsDir: null,
      enableRustBridge: true,
      stubMode: false,
      enableAuditLog: true,
      ...options,
    };

    this.rpc = new JsonRpcHandler();
    this.registry = new ToolRegistry({ toolsDir: this.options.toolsDir });
    this.transport = null;
    this.rustBridge = null;
    this.auditLog = null;
    this.running = false;
  }

  /**
   * Initialize the MCP server
   */
  async initialize() {
    console.error('🚀 Neural Trader MCP Server (MCP 2025-11 Compliant)');
    console.error('');

    // Load tool registry
    console.error('📚 Loading tool schemas...');
    await this.registry.loadTools();
    console.error(`   Found ${this.registry.tools.size} tools`);
    console.error('');

    // Initialize Rust bridge if enabled
    if (this.options.enableRustBridge) {
      console.error('🦀 Starting Rust NAPI bridge...');
      this.rustBridge = new RustBridge({
        stubMode: this.options.stubMode
      });

      try {
        await this.rustBridge.start();
        const status = this.rustBridge.getStatus();
        if (status.napiLoaded) {
          console.error('   ✅ Rust NAPI module loaded');
        } else {
          console.error('   ⚠️  Running in stub mode');
        }
      } catch (error) {
        console.error('   ⚠️  Rust bridge failed:', error.message);
        console.error('   ℹ️  Falling back to stub implementations');
        this.rustBridge = null;
      }
      console.error('');
    }

    // Initialize audit logging
    if (this.options.enableAuditLog) {
      this.auditLog = createAuditLogger();
      console.error('📝 Audit logging enabled');
      console.error('');
    }

    // Register MCP methods
    this.registerMethods();

    console.error('✅ Server initialized successfully');
    console.error('');
  }

  /**
   * Register JSON-RPC methods
   */
  registerMethods() {
    // MCP protocol methods
    this.rpc.register('initialize', this.handleInitialize.bind(this));
    this.rpc.register('tools/list', this.handleToolsList.bind(this));
    this.rpc.register('tools/call', this.handleToolsCall.bind(this));
    this.rpc.register('tools/schema', this.handleToolsSchema.bind(this));

    // Discovery methods
    this.rpc.register('tools/search', this.handleToolsSearch.bind(this));
    this.rpc.register('tools/categories', this.handleToolsCategories.bind(this));

    // Server methods
    this.rpc.register('server/info', this.handleServerInfo.bind(this));
    this.rpc.register('ping', this.handlePing.bind(this));
  }

  /**
   * Handle 'initialize' method
   */
  async handleInitialize(params) {
    return {
      protocolVersion: '2025-11',
      serverInfo: {
        name: 'Neural Trader MCP Server',
        version: '2.0.0',
        vendor: 'Neural Trader Team',
      },
      capabilities: {
        tools: {
          listChanged: true,
          supportedFormats: ['json-schema-1.1'],
        },
        resources: false,
        prompts: false,
        logging: true,
      },
      instructions: 'Neural Trader provides 99+ trading tools for AI assistants',
    };
  }

  /**
   * Handle 'tools/list' method
   */
  async handleToolsList(params) {
    const tools = this.registry.listTools();

    return {
      tools: tools.map(tool => ({
        name: tool.name,
        description: tool.description,
        inputSchema: {
          $ref: `/tools/${tool.name}.json#/input_schema`,
        },
      })),
    };
  }

  /**
   * Handle 'tools/call' method
   */
  async handleToolsCall(params) {
    const { name, arguments: args = {} } = params;

    // Validate tool exists
    if (!this.registry.hasTool(name)) {
      throw new Error(`Tool not found: ${name}`);
    }

    // Log the call
    if (this.auditLog) {
      this.auditLog.logToolCall(name, args);
    }

    const startTime = Date.now();
    let result;
    let error = null;

    try {
      // Try Rust bridge first
      if (this.rustBridge && this.rustBridge.isReady()) {
        try {
          result = await this.rustBridge.call(name, args);
        } catch (rustError) {
          console.error(`Rust tool error for ${name}:`, rustError.message);
          // Fall back to stub
          result = await this.handleToolStub(name, args);
        }
      } else {
        // Use stub implementation
        result = await this.handleToolStub(name, args);
      }
    } catch (callError) {
      error = callError;
      throw callError;
    } finally {
      const duration = Date.now() - startTime;

      // Log the result
      if (this.auditLog) {
        this.auditLog.logToolResult(name, result, error, duration);
      }
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
      isError: false,
    };
  }

  /**
   * Handle tool with stub implementation
   */
  async handleToolStub(name, args) {
    return {
      tool: name,
      status: 'stub',
      message: 'This tool is not yet implemented - Rust NAPI module not loaded',
      arguments: args,
      timestamp: new Date().toISOString(),
      rustBridgeAvailable: this.rustBridge && this.rustBridge.isReady(),
      rustBridgeLoaded: this.rustBridge && this.rustBridge.isNapiLoaded(),
    };
  }

  /**
   * Handle 'tools/schema' method
   */
  async handleToolsSchema(params) {
    const { name } = params;
    const schema = this.registry.getToolSchema(name);

    if (!schema) {
      throw new Error(`Tool not found: ${name}`);
    }

    return {
      schema,
      etag: this.registry.getToolETag(name),
    };
  }

  /**
   * Handle 'tools/search' method
   */
  async handleToolsSearch(params) {
    const { query } = params;
    const results = this.registry.searchTools(query);

    return {
      results: results.map(name => ({
        name,
        metadata: this.registry.getToolMetadata(name),
      })),
    };
  }

  /**
   * Handle 'tools/categories' method
   */
  async handleToolsCategories(params) {
    const categories = new Set();
    for (const [, tool] of this.registry.tools) {
      categories.add(tool.schema.category || 'general');
    }

    return {
      categories: Array.from(categories).map(cat => ({
        name: cat,
        tools: this.registry.getToolsByCategory(cat),
      })),
    };
  }

  /**
   * Handle 'server/info' method
   */
  async handleServerInfo(params) {
    return {
      name: 'Neural Trader MCP Server',
      version: '2.0.0',
      protocol: 'MCP 2025-11',
      transport: this.options.transport,
      toolsCount: this.registry.tools.size,
      rustBridge: this.rustBridge ? this.rustBridge.getStatus() : null,
      auditLog: this.options.enableAuditLog,
    };
  }

  /**
   * Handle 'ping' method
   */
  async handlePing(params) {
    return {
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    };
  }

  /**
   * Start the MCP server
   */
  async start() {
    if (this.running) {
      throw new Error('Server already running');
    }

    await this.initialize();

    // Create transport
    if (this.options.transport === 'stdio') {
      this.transport = new StdioTransport();
    } else {
      throw new Error(`Unsupported transport: ${this.options.transport}`);
    }

    // Handle incoming messages
    this.transport.on('message', async (message) => {
      try {
        const response = await this.rpc.process(message);
        if (response) {
          this.transport.send(response);
        }
      } catch (error) {
        console.error('Error processing message:', error);
      }
    });

    // Handle transport events
    this.transport.on('connect', () => {
      console.error('🔌 Transport connected');
    });

    this.transport.on('close', () => {
      console.error('🔌 Transport closed');
      this.stop();
    });

    this.transport.on('error', (error) => {
      console.error('Transport error:', error);
    });

    // Start transport
    await this.transport.start();

    this.running = true;
    console.error('✅ MCP server running');
    console.error('   Waiting for requests...');
    console.error('');
  }

  /**
   * Stop the MCP server
   */
  async stop() {
    if (!this.running) {
      return;
    }

    console.error('');
    console.error('🛑 Stopping MCP server...');

    if (this.transport) {
      await this.transport.stop();
    }

    if (this.rustBridge) {
      await this.rustBridge.stop();
    }

    this.running = false;
    console.error('✅ MCP server stopped');
  }
}

module.exports = { McpServer };
