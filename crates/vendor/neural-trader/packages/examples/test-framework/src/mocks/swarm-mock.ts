/**
 * Mock Swarm for testing
 */

import { MockOptions, SwarmMetrics } from '../types';

export interface Agent {
  id: string;
  role: string;
  state: any;
}

export interface SwarmMessage {
  from: string;
  to: string | 'broadcast';
  type: string;
  payload: any;
}

/**
 * Mock Swarm implementation
 */
export class MockSwarm {
  private agents: Map<string, Agent>;
  private messages: SwarmMessage[];
  private options: MockOptions;
  private callCount: number;

  constructor(options: MockOptions = {}) {
    this.agents = new Map();
    this.messages = [];
    this.options = options;
    this.callCount = 0;
  }

  /**
   * Add agent to swarm
   */
  async addAgent(agent: Agent): Promise<void> {
    this.callCount++;
    await this.simulateDelay();
    this.agents.set(agent.id, agent);
  }

  /**
   * Remove agent from swarm
   */
  async removeAgent(agentId: string): Promise<boolean> {
    this.callCount++;
    await this.simulateDelay();
    return this.agents.delete(agentId);
  }

  /**
   * Send message between agents
   */
  async sendMessage(message: SwarmMessage): Promise<void> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();
    this.messages.push(message);
  }

  /**
   * Broadcast message to all agents
   */
  async broadcast(from: string, type: string, payload: any): Promise<void> {
    await this.sendMessage({
      from,
      to: 'broadcast',
      type,
      payload
    });
  }

  /**
   * Run swarm coordination
   */
  async coordinate(iterations = 10): Promise<SwarmMetrics> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();

    const startTime = Date.now();

    // Simulate coordination process
    for (let i = 0; i < iterations; i++) {
      await this.simulateDelay();
    }

    const convergenceTime = Date.now() - startTime;

    return {
      agents: this.agents.size,
      messages: this.messages.length,
      convergenceTime,
      consensusReached: Math.random() > 0.2, // 80% consensus rate
      averageQuality: 0.7 + Math.random() * 0.3
    };
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): Agent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get all agents
   */
  getAllAgents(): Agent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Get messages
   */
  getMessages(): SwarmMessage[] {
    return [...this.messages];
  }

  /**
   * Clear swarm
   */
  clear(): void {
    this.agents.clear();
    this.messages = [];
    this.callCount = 0;
  }

  /**
   * Get call count
   */
  getCallCount(): number {
    return this.callCount;
  }

  private async simulateDelay(): Promise<void> {
    const delay = this.options.delay || 10;
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  private maybeThrowError(): void {
    if (this.options.errorRate && Math.random() < this.options.errorRate) {
      throw new Error('Mock Swarm error');
    }
  }
}

/**
 * Create mock swarm
 */
export function createMockSwarm(options: MockOptions = {}): MockSwarm {
  return new MockSwarm(options);
}
