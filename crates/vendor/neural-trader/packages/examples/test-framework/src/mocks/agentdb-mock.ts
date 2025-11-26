/**
 * Mock AgentDB for testing
 */

import { MockOptions } from '../types';

export interface AgentDBQuery {
  vector: number[];
  k?: number;
  filter?: Record<string, any>;
}

export interface AgentDBResult {
  id: string;
  vector: number[];
  metadata: Record<string, any>;
  distance: number;
}

/**
 * Mock AgentDB implementation
 */
export class MockAgentDB {
  private storage: Map<string, { vector: number[]; metadata: Record<string, any> }>;
  private options: MockOptions;
  private callCount: number;

  constructor(options: MockOptions = {}) {
    this.storage = new Map();
    this.options = options;
    this.callCount = 0;
  }

  /**
   * Add vector to mock database
   */
  async add(
    id: string,
    vector: number[],
    metadata: Record<string, any> = {}
  ): Promise<void> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();

    this.storage.set(id, { vector, metadata });
  }

  /**
   * Query mock database
   */
  async query(query: AgentDBQuery): Promise<AgentDBResult[]> {
    this.callCount++;
    await this.simulateDelay();
    this.maybeThrowError();

    const { vector, k = 10, filter } = query;
    const results: AgentDBResult[] = [];

    for (const [id, data] of this.storage.entries()) {
      // Apply filter if provided
      if (filter) {
        const matches = Object.entries(filter).every(
          ([key, value]) => data.metadata[key] === value
        );
        if (!matches) continue;
      }

      // Calculate cosine distance
      const distance = this.cosineSimilarity(vector, data.vector);

      results.push({
        id,
        vector: data.vector,
        metadata: data.metadata,
        distance
      });
    }

    // Sort by distance and return top k
    return results
      .sort((a, b) => b.distance - a.distance)
      .slice(0, k);
  }

  /**
   * Delete from mock database
   */
  async delete(id: string): Promise<boolean> {
    this.callCount++;
    await this.simulateDelay();
    return this.storage.delete(id);
  }

  /**
   * Clear mock database
   */
  clear(): void {
    this.storage.clear();
    this.callCount = 0;
  }

  /**
   * Get call count
   */
  getCallCount(): number {
    return this.callCount;
  }

  /**
   * Get storage size
   */
  size(): number {
    return this.storage.size;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private async simulateDelay(): Promise<void> {
    if (this.options.delay) {
      await new Promise(resolve => setTimeout(resolve, this.options.delay));
    }
  }

  private maybeThrowError(): void {
    if (this.options.errorRate) {
      if (Math.random() < this.options.errorRate) {
        throw new Error('Mock AgentDB error');
      }
    }
  }
}

/**
 * Create mock AgentDB instance
 */
export function createMockAgentDB(options: MockOptions = {}): MockAgentDB {
  return new MockAgentDB(options);
}
