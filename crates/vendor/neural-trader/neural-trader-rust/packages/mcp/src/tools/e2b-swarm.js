/**
 * E2B Trading Swarm Tools
 *
 * Provides tools for managing distributed trading swarms in E2B cloud sandboxes.
 * These tools enable multi-agent trading systems with coordinated execution,
 * shared state management, and scalable deployment patterns.
 *
 * @module tools/e2b-swarm
 */

const { RustBridge } = require('../bridge/rust');

/**
 * E2B Swarm Tool Definitions
 * All tools follow MCP 2025-11 specification
 */
const E2B_SWARM_TOOLS = {
  /**
   * Initialize E2B trading swarm with specified topology
   */
  init_e2b_swarm: {
    name: 'init_e2b_swarm',
    description: 'Initialize E2B trading swarm with specified topology and configuration. Creates a distributed trading system with multiple coordinated agents.',
    inputSchema: {
      type: 'object',
      properties: {
        topology: {
          type: 'string',
          enum: ['mesh', 'hierarchical', 'ring', 'star'],
          description: 'Network topology for swarm coordination: mesh (peer-to-peer), hierarchical (tree), ring (circular), star (centralized)'
        },
        maxAgents: {
          type: 'number',
          default: 5,
          minimum: 1,
          maximum: 50,
          description: 'Maximum number of trading agents in the swarm'
        },
        strategy: {
          type: 'string',
          default: 'balanced',
          enum: ['balanced', 'aggressive', 'conservative', 'adaptive'],
          description: 'Overall swarm trading strategy'
        },
        sharedMemory: {
          type: 'boolean',
          default: true,
          description: 'Enable shared memory for inter-agent communication'
        },
        autoScale: {
          type: 'boolean',
          default: false,
          description: 'Automatically scale swarm based on market conditions'
        }
      },
      required: ['topology']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  /**
   * Deploy trading agent to E2B swarm
   */
  deploy_trading_agent: {
    name: 'deploy_trading_agent',
    description: 'Deploy a specialized trading agent to the E2B swarm with specific role and capabilities.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        agentType: {
          type: 'string',
          enum: ['market_maker', 'trend_follower', 'arbitrage', 'risk_manager', 'coordinator'],
          description: 'Type of trading agent to deploy'
        },
        symbols: {
          type: 'array',
          items: { type: 'string' },
          description: 'Trading symbols this agent will focus on'
        },
        strategyParams: {
          type: 'object',
          additionalProperties: true,
          description: 'Strategy-specific parameters for the agent'
        },
        resources: {
          type: 'object',
          properties: {
            memory_mb: { type: 'number', default: 512 },
            cpu_count: { type: 'number', default: 1 },
            gpu_enabled: { type: 'boolean', default: false }
          },
          description: 'Resource allocation for the agent sandbox'
        }
      },
      required: ['swarmId', 'agentType', 'symbols']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  /**
   * Get swarm status and health metrics
   */
  get_swarm_status: {
    name: 'get_swarm_status',
    description: 'Get comprehensive status and health metrics for an E2B trading swarm.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        includeMetrics: {
          type: 'boolean',
          default: true,
          description: 'Include detailed performance metrics'
        },
        includeAgents: {
          type: 'boolean',
          default: true,
          description: 'Include individual agent statuses'
        }
      },
      required: ['swarmId']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  /**
   * Scale E2B swarm up or down
   */
  scale_swarm: {
    name: 'scale_swarm',
    description: 'Scale E2B trading swarm by adding or removing agents based on demand.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        targetAgents: {
          type: 'number',
          minimum: 1,
          maximum: 50,
          description: 'Target number of agents in the swarm'
        },
        scaleMode: {
          type: 'string',
          enum: ['immediate', 'gradual', 'adaptive'],
          default: 'gradual',
          description: 'How to perform the scaling operation'
        },
        preserveState: {
          type: 'boolean',
          default: true,
          description: 'Preserve agent state during scaling'
        }
      },
      required: ['swarmId', 'targetAgents']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  /**
   * Execute coordinated strategy across swarm
   */
  execute_swarm_strategy: {
    name: 'execute_swarm_strategy',
    description: 'Execute a coordinated trading strategy across all agents in the swarm.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        strategy: {
          type: 'string',
          description: 'Name of the strategy to execute'
        },
        parameters: {
          type: 'object',
          additionalProperties: true,
          description: 'Strategy parameters'
        },
        coordination: {
          type: 'string',
          enum: ['parallel', 'sequential', 'adaptive'],
          default: 'parallel',
          description: 'Execution coordination mode'
        },
        timeout: {
          type: 'number',
          default: 300,
          description: 'Execution timeout in seconds'
        }
      },
      required: ['swarmId', 'strategy']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  /**
   * Monitor swarm health and performance
   */
  monitor_swarm_health: {
    name: 'monitor_swarm_health',
    description: 'Monitor E2B swarm health with real-time metrics and alerts.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        interval: {
          type: 'number',
          default: 60,
          minimum: 10,
          description: 'Monitoring interval in seconds'
        },
        alerts: {
          type: 'object',
          properties: {
            failureThreshold: { type: 'number', default: 0.2 },
            latencyThreshold: { type: 'number', default: 1000 },
            errorRateThreshold: { type: 'number', default: 0.05 }
          },
          description: 'Alert thresholds for health monitoring'
        },
        includeSystemMetrics: {
          type: 'boolean',
          default: true,
          description: 'Include CPU, memory, and network metrics'
        }
      },
      required: ['swarmId']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  /**
   * Get swarm performance metrics
   */
  get_swarm_metrics: {
    name: 'get_swarm_metrics',
    description: 'Get detailed performance metrics for E2B trading swarm.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        timeRange: {
          type: 'string',
          enum: ['1h', '6h', '24h', '7d', '30d'],
          default: '24h',
          description: 'Time range for metrics'
        },
        metrics: {
          type: 'array',
          items: {
            type: 'string',
            enum: ['latency', 'throughput', 'error_rate', 'success_rate', 'pnl', 'trades', 'all']
          },
          default: ['all'],
          description: 'Specific metrics to retrieve'
        },
        aggregation: {
          type: 'string',
          enum: ['avg', 'min', 'max', 'sum', 'p50', 'p95', 'p99'],
          default: 'avg',
          description: 'Aggregation method for metrics'
        }
      },
      required: ['swarmId']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  /**
   * Shutdown E2B swarm
   */
  shutdown_swarm: {
    name: 'shutdown_swarm',
    description: 'Gracefully shutdown E2B trading swarm and cleanup resources.',
    inputSchema: {
      type: 'object',
      properties: {
        swarmId: {
          type: 'string',
          description: 'Unique identifier of the swarm'
        },
        gracePeriod: {
          type: 'number',
          default: 60,
          description: 'Grace period in seconds for agents to finish tasks'
        },
        saveState: {
          type: 'boolean',
          default: true,
          description: 'Save swarm state before shutdown'
        },
        force: {
          type: 'boolean',
          default: false,
          description: 'Force immediate shutdown without waiting'
        }
      },
      required: ['swarmId']
    },
    metadata: {
      category: 'e2b_swarm',
      cost: 'low',
      latency: 'medium',
      gpu_capable: false
    }
  }
};

/**
 * E2B Swarm Tool Handler
 * Routes tool calls to appropriate Rust NAPI implementations
 */
class E2bSwarmToolHandler {
  constructor(rustBridge) {
    this.bridge = rustBridge;
  }

  /**
   * Initialize E2B swarm
   */
  async initE2bSwarm(params) {
    return await this.bridge.call('init_e2b_swarm', params);
  }

  /**
   * Deploy trading agent to swarm
   */
  async deployTradingAgent(params) {
    return await this.bridge.call('deploy_trading_agent', params);
  }

  /**
   * Get swarm status
   */
  async getSwarmStatus(params) {
    return await this.bridge.call('get_swarm_status', params);
  }

  /**
   * Scale swarm
   */
  async scaleSwarm(params) {
    return await this.bridge.call('scale_swarm', params);
  }

  /**
   * Execute swarm strategy
   */
  async executeSwarmStrategy(params) {
    return await this.bridge.call('execute_swarm_strategy', params);
  }

  /**
   * Monitor swarm health
   */
  async monitorSwarmHealth(params) {
    return await this.bridge.call('monitor_swarm_health', params);
  }

  /**
   * Get swarm metrics
   */
  async getSwarmMetrics(params) {
    return await this.bridge.call('get_swarm_metrics', params);
  }

  /**
   * Shutdown swarm
   */
  async shutdownSwarm(params) {
    return await this.bridge.call('shutdown_swarm', params);
  }
}

/**
 * Register E2B swarm tools with the tool registry
 */
function registerE2bSwarmTools(registry) {
  for (const [name, definition] of Object.entries(E2B_SWARM_TOOLS)) {
    registry.registerTool(name, definition);
  }
}

/**
 * Get all E2B swarm tool definitions
 */
function getE2bSwarmTools() {
  return E2B_SWARM_TOOLS;
}

module.exports = {
  E2B_SWARM_TOOLS,
  E2bSwarmToolHandler,
  registerE2bSwarmTools,
  getE2bSwarmTools
};
