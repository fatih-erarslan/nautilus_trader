/**
 * Swarm coordinator for parallel variation exploration
 * Uses agentic-flow patterns for distributed task execution
 */

import { AgenticFlow, FlowNode, FlowEdge } from 'agentic-flow';

export interface SwarmConfig {
  maxAgents: number;
  topology: 'star' | 'mesh' | 'hierarchical';
  communicationProtocol: 'sync' | 'async';
  timeout?: number; // ms
  retryAttempts?: number;
}

export interface TaskVariation {
  id: string;
  parameters: Record<string, any>;
  priority?: number;
}

export interface SwarmResult<T = any> {
  variationId: string;
  success: boolean;
  result?: T;
  error?: Error;
  metrics: {
    executionTime: number;
    memoryUsed: number;
    cpuUsage?: number;
  };
}

export interface AgentTask<T = any> {
  execute(parameters: Record<string, any>): Promise<T>;
  validate?(result: T): boolean;
}

export class SwarmCoordinator<T = any> {
  private config: Required<SwarmConfig>;
  private activeAgents: Map<string, Promise<SwarmResult<T>>> = new Map();
  private results: SwarmResult<T>[] = [];
  private flow: AgenticFlow;

  constructor(config: SwarmConfig) {
    this.config = {
      maxAgents: config.maxAgents,
      topology: config.topology,
      communicationProtocol: config.communicationProtocol,
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
    };

    this.flow = new AgenticFlow({
      maxConcurrency: this.config.maxAgents,
      retryStrategy: {
        maxAttempts: this.config.retryAttempts,
        backoff: 'exponential',
      },
    });

    this.initializeTopology();
  }

  /**
   * Execute task variations in parallel using swarm
   */
  async executeVariations(
    variations: TaskVariation[],
    task: AgentTask<T>,
    onProgress?: (completed: number, total: number) => void
  ): Promise<SwarmResult<T>[]> {
    const sortedVariations = this.prioritizeVariations(variations);
    const batches = this.createBatches(sortedVariations);

    let completed = 0;
    const total = variations.length;

    for (const batch of batches) {
      const batchPromises = batch.map((variation) =>
        this.executeVariation(variation, task)
      );

      const batchResults = await Promise.allSettled(batchPromises);

      batchResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          this.results.push(result.value);
        } else {
          this.results.push({
            variationId: batch[index].id,
            success: false,
            error: result.reason,
            metrics: {
              executionTime: 0,
              memoryUsed: 0,
            },
          });
        }
        completed++;
        onProgress?.(completed, total);
      });
    }

    return this.results;
  }

  /**
   * Execute single variation with metrics tracking
   */
  private async executeVariation(
    variation: TaskVariation,
    task: AgentTask<T>
  ): Promise<SwarmResult<T>> {
    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed;

    try {
      const result = await this.executeWithTimeout(
        () => task.execute(variation.parameters),
        this.config.timeout
      );

      // Validate result if validator provided
      if (task.validate && !task.validate(result)) {
        throw new Error('Result validation failed');
      }

      const executionTime = Date.now() - startTime;
      const memoryUsed = process.memoryUsage().heapUsed - startMemory;

      return {
        variationId: variation.id,
        success: true,
        result,
        metrics: {
          executionTime,
          memoryUsed,
        },
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const memoryUsed = process.memoryUsage().heapUsed - startMemory;

      return {
        variationId: variation.id,
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        metrics: {
          executionTime,
          memoryUsed,
        },
      };
    }
  }

  /**
   * Execute with timeout
   */
  private async executeWithTimeout<R>(
    fn: () => Promise<R>,
    timeout: number
  ): Promise<R> {
    return Promise.race([
      fn(),
      new Promise<R>((_, reject) =>
        setTimeout(() => reject(new Error('Task timeout')), timeout)
      ),
    ]);
  }

  /**
   * Prioritize variations based on priority field
   */
  private prioritizeVariations(variations: TaskVariation[]): TaskVariation[] {
    return [...variations].sort((a, b) => {
      const priorityA = a.priority ?? 0;
      const priorityB = b.priority ?? 0;
      return priorityB - priorityA; // Higher priority first
    });
  }

  /**
   * Create batches for parallel execution
   */
  private createBatches(variations: TaskVariation[]): TaskVariation[][] {
    const batches: TaskVariation[][] = [];
    for (let i = 0; i < variations.length; i += this.config.maxAgents) {
      batches.push(variations.slice(i, i + this.config.maxAgents));
    }
    return batches;
  }

  /**
   * Initialize swarm topology
   */
  private initializeTopology(): void {
    switch (this.config.topology) {
      case 'star':
        this.initializeStarTopology();
        break;
      case 'mesh':
        this.initializeMeshTopology();
        break;
      case 'hierarchical':
        this.initializeHierarchicalTopology();
        break;
    }
  }

  /**
   * Initialize star topology (coordinator at center)
   */
  private initializeStarTopology(): void {
    const coordinator: FlowNode = {
      id: 'coordinator',
      type: 'coordinator',
      execute: async (data: any) => data,
    };

    this.flow.addNode(coordinator);

    for (let i = 0; i < this.config.maxAgents; i++) {
      const agent: FlowNode = {
        id: `agent-${i}`,
        type: 'worker',
        execute: async (data: any) => data,
      };

      this.flow.addNode(agent);
      this.flow.addEdge({ from: 'coordinator', to: agent.id });
    }
  }

  /**
   * Initialize mesh topology (all agents connected)
   */
  private initializeMeshTopology(): void {
    const agents: FlowNode[] = [];

    for (let i = 0; i < this.config.maxAgents; i++) {
      const agent: FlowNode = {
        id: `agent-${i}`,
        type: 'worker',
        execute: async (data: any) => data,
      };
      agents.push(agent);
      this.flow.addNode(agent);
    }

    // Connect all agents to each other
    for (let i = 0; i < agents.length; i++) {
      for (let j = i + 1; j < agents.length; j++) {
        this.flow.addEdge({ from: agents[i].id, to: agents[j].id });
        this.flow.addEdge({ from: agents[j].id, to: agents[i].id });
      }
    }
  }

  /**
   * Initialize hierarchical topology (tree structure)
   */
  private initializeHierarchicalTopology(): void {
    const coordinator: FlowNode = {
      id: 'coordinator',
      type: 'coordinator',
      execute: async (data: any) => data,
    };

    this.flow.addNode(coordinator);

    const managersCount = Math.ceil(Math.sqrt(this.config.maxAgents));
    const workersPerManager = Math.ceil(this.config.maxAgents / managersCount);

    for (let i = 0; i < managersCount; i++) {
      const manager: FlowNode = {
        id: `manager-${i}`,
        type: 'manager',
        execute: async (data: any) => data,
      };

      this.flow.addNode(manager);
      this.flow.addEdge({ from: 'coordinator', to: manager.id });

      for (let j = 0; j < workersPerManager; j++) {
        const workerIndex = i * workersPerManager + j;
        if (workerIndex >= this.config.maxAgents) break;

        const worker: FlowNode = {
          id: `worker-${workerIndex}`,
          type: 'worker',
          execute: async (data: any) => data,
        };

        this.flow.addNode(worker);
        this.flow.addEdge({ from: manager.id, to: worker.id });
      }
    }
  }

  /**
   * Get swarm statistics
   */
  getStatistics(): {
    totalVariations: number;
    successfulVariations: number;
    failedVariations: number;
    averageExecutionTime: number;
    averageMemoryUsed: number;
    successRate: number;
  } {
    const successful = this.results.filter((r) => r.success);
    const failed = this.results.filter((r) => !r.success);

    const avgExecutionTime =
      this.results.reduce((sum, r) => sum + r.metrics.executionTime, 0) /
      this.results.length;

    const avgMemoryUsed =
      this.results.reduce((sum, r) => sum + r.metrics.memoryUsed, 0) /
      this.results.length;

    return {
      totalVariations: this.results.length,
      successfulVariations: successful.length,
      failedVariations: failed.length,
      averageExecutionTime: avgExecutionTime,
      averageMemoryUsed: avgMemoryUsed,
      successRate: successful.length / this.results.length,
    };
  }

  /**
   * Get best performing variation
   */
  getBestVariation(
    metric: 'executionTime' | 'memoryUsed' = 'executionTime'
  ): SwarmResult<T> | null {
    const successful = this.results.filter((r) => r.success);
    if (successful.length === 0) return null;

    return successful.reduce((best, current) => {
      return current.metrics[metric] < best.metrics[metric] ? current : best;
    });
  }

  /**
   * Clear results
   */
  clear(): void {
    this.results = [];
    this.activeAgents.clear();
  }

  /**
   * Export results
   */
  exportResults(): SwarmResult<T>[] {
    return [...this.results];
  }
}
