/**
 * Base Agent Class
 *
 * Foundation for all specialized accounting agents with:
 * - ReasoningBank integration for learning
 * - Memory coordination via hooks
 * - Performance tracking
 * - Error handling patterns
 */

import { EventEmitter } from 'events';

export interface AgentConfig {
  agentId: string;
  agentType: string;
  enableLearning?: boolean;
  enableMetrics?: boolean;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

export interface AgentTask {
  taskId: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  data: unknown;
  metadata?: Record<string, unknown>;
}

export interface AgentResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: Error;
  metrics?: {
    startTime: number;
    endTime: number;
    duration: number;
    memoryUsed?: number;
  };
  metadata?: Record<string, unknown>;
}

export interface DecisionLog {
  scenario: string;
  decision: string;
  rationale: string;
  outcome: 'SUCCESS' | 'FAILURE' | 'PENDING';
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export abstract class BaseAgent extends EventEmitter {
  protected config: AgentConfig;
  protected isRunning: boolean = false;
  protected decisions: DecisionLog[] = [];
  protected logger: Console = console;

  constructor(config: AgentConfig) {
    super();
    this.config = config;
  }

  /**
   * Learn from experience - placeholder for ReasoningBank integration
   */
  protected async learn(data: Record<string, any>): Promise<void> {
    if (this.config.enableLearning) {
      // Store learning data via hooks in production
      this.logger.debug(`[${this.config.agentId}] Learning:`, data);
    }
  }

  /**
   * Abstract method - must be implemented by each agent
   */
  abstract execute(task: AgentTask): Promise<AgentResult>;

  /**
   * Start the agent
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error(`Agent ${this.config.agentId} is already running`);
    }

    this.isRunning = true;
    this.emit('started', { agentId: this.config.agentId, timestamp: Date.now() });

    if (this.config.logLevel !== 'error') {
      console.log(`[${this.config.agentId}] Agent started`);
    }
  }

  /**
   * Stop the agent
   */
  async stop(): Promise<void> {
    this.isRunning = false;
    this.emit('stopped', { agentId: this.config.agentId, timestamp: Date.now() });

    if (this.config.logLevel !== 'error') {
      console.log(`[${this.config.agentId}] Agent stopped`);
    }
  }

  /**
   * Log a decision for ReasoningBank learning
   */
  protected async logDecision(
    scenario: string,
    decision: string,
    rationale: string,
    outcome: 'SUCCESS' | 'FAILURE' | 'PENDING' = 'PENDING',
    metadata?: Record<string, unknown>
  ): Promise<void> {
    const log: DecisionLog = {
      scenario,
      decision,
      rationale,
      outcome,
      timestamp: Date.now(),
      metadata,
    };

    this.decisions.push(log);
    this.emit('decision', log);

    if (this.config.enableLearning) {
      // Store in ReasoningBank via hooks
      // This would be called via hooks in production
      if (this.config.logLevel === 'debug') {
        console.log(`[${this.config.agentId}] Decision logged:`, {
          scenario,
          decision,
          outcome,
        });
      }
    }
  }

  /**
   * Execute task with metrics tracking
   */
  protected async executeWithMetrics<T>(
    taskFn: () => Promise<T>
  ): Promise<AgentResult<T>> {
    const startTime = Date.now();
    const startMemory = this.config.enableMetrics
      ? process.memoryUsage().heapUsed
      : undefined;

    try {
      const data = await taskFn();
      const endTime = Date.now();
      const endMemory = this.config.enableMetrics
        ? process.memoryUsage().heapUsed
        : undefined;

      return {
        success: true,
        data,
        metrics: {
          startTime,
          endTime,
          duration: endTime - startTime,
          memoryUsed: startMemory && endMemory ? endMemory - startMemory : undefined,
        },
      };
    } catch (error) {
      const endTime = Date.now();

      return {
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        metrics: {
          startTime,
          endTime,
          duration: endTime - startTime,
        },
      };
    }
  }

  /**
   * Get agent status
   */
  getStatus(): {
    agentId: string;
    agentType: string;
    isRunning: boolean;
    decisionCount: number;
  } {
    return {
      agentId: this.config.agentId,
      agentType: this.config.agentType,
      isRunning: this.isRunning,
      decisionCount: this.decisions.length,
    };
  }

  /**
   * Get recent decisions for analysis
   */
  getRecentDecisions(limit: number = 10): DecisionLog[] {
    return this.decisions.slice(-limit);
  }

  /**
   * Clear decision history (for testing or memory management)
   */
  clearDecisions(): void {
    this.decisions = [];
  }
}
