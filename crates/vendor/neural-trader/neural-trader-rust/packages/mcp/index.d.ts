/**
 * @neural-trader/mcp - MCP server implementation
 *
 * Type definitions for Neural Trader MCP server
 */

import * as protocol from '@neural-trader/mcp-protocol';

export { protocol };

/**
 * MCP Server configuration
 */
export interface McpServerConfig {
  /** Transport type: 'stdio', 'http', 'websocket' */
  transport?: 'stdio' | 'http' | 'websocket';
  /** Port number for HTTP/WebSocket (default: 3000) */
  port?: number;
  /** Host address (default: 'localhost') */
  host?: string;
  /** Enable CORS for HTTP transport */
  enableCors?: boolean;
  /** Maximum concurrent connections */
  maxConnections?: number;
}

/**
 * MCP Tool definition
 */
export interface McpTool {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

/**
 * Syndicate tool definition
 */
export interface SyndicateTool {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
  handler: (params: any) => Promise<any>;
}

/**
 * MCP Server class
 */
export class McpServer {
  constructor(config?: McpServerConfig);

  /**
   * Start the MCP server
   */
  start(): Promise<void>;

  /**
   * Stop the MCP server
   */
  stop(): Promise<void>;

  /**
   * Register a custom tool
   */
  registerTool(name: string, handler: (params: any) => Promise<any>): void;

  /**
   * List all available tools
   */
  listTools(): Promise<string[]>;

  /**
   * Get syndicate tools
   */
  getSyndicateTools(): SyndicateTool[];

  /**
   * Execute a syndicate tool
   */
  executeSyndicateTool(toolName: string, params: any): Promise<any>;
}

/**
 * Start the MCP server with configuration
 */
export function startServer(config?: McpServerConfig): Promise<McpServer>;
