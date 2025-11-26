/**
 * Experience replay system with AgentDB integration
 * Stores and retrieves past experiences for continuous learning
 */

import { AgentDB } from 'agentdb';

export interface Experience<TState = any, TAction = any, TResult = any> {
  id: string;
  timestamp: Date;
  state: TState;
  action: TAction;
  result: TResult;
  reward: number;
  metadata?: Record<string, any>;
}

export interface ReplayConfig {
  maxSize: number;
  prioritization?: 'uniform' | 'prioritized' | 'temporal';
  dbPath?: string;
  namespace?: string;
}

export interface ReplayBatch<T extends Experience = Experience> {
  experiences: T[];
  weights?: number[];
  indices: number[];
}

export class ExperienceReplay<TExp extends Experience = Experience> {
  private db: AgentDB;
  private config: Required<ReplayConfig>;
  private experienceCount: number = 0;

  constructor(config: ReplayConfig) {
    this.config = {
      maxSize: config.maxSize,
      prioritization: config.prioritization || 'uniform',
      dbPath: config.dbPath || './agentdb',
      namespace: config.namespace || 'experience-replay',
    };

    this.db = new AgentDB({
      path: this.config.dbPath,
      dimensions: 384, // Default embedding dimensions
    });
  }

  /**
   * Store an experience in the replay buffer
   */
  async store(experience: TExp): Promise<void> {
    // Create embedding from experience
    const embedding = await this.createEmbedding(experience);

    // Store in AgentDB
    await this.db.add({
      id: experience.id,
      vector: embedding,
      metadata: {
        ...experience,
        timestamp: experience.timestamp.toISOString(),
      },
    });

    this.experienceCount++;

    // Evict oldest if buffer is full
    if (this.experienceCount > this.config.maxSize) {
      await this.evictOldest();
    }
  }

  /**
   * Store multiple experiences in batch
   */
  async storeBatch(experiences: TExp[]): Promise<void> {
    const embeddings = await Promise.all(
      experiences.map((exp) => this.createEmbedding(exp))
    );

    const documents = experiences.map((exp, i) => ({
      id: exp.id,
      vector: embeddings[i],
      metadata: {
        ...exp,
        timestamp: exp.timestamp.toISOString(),
      },
    }));

    await this.db.addBatch(documents);
    this.experienceCount += experiences.length;

    // Evict if necessary
    while (this.experienceCount > this.config.maxSize) {
      await this.evictOldest();
    }
  }

  /**
   * Sample random experiences from buffer
   */
  async sample(batchSize: number): Promise<ReplayBatch<TExp>> {
    const results = await this.db.query({
      vector: await this.createRandomEmbedding(),
      k: Math.min(batchSize, this.experienceCount),
    });

    const experiences = results.map((result) => this.deserializeExperience(result.metadata));
    const indices = results.map((_, i) => i);

    return {
      experiences,
      indices,
    };
  }

  /**
   * Sample prioritized experiences (higher reward = higher probability)
   */
  async samplePrioritized(batchSize: number, alpha: number = 0.6): Promise<ReplayBatch<TExp>> {
    // Get all experiences (in production, use pagination)
    const all = await this.getAll();

    if (all.length === 0) {
      return { experiences: [], indices: [] };
    }

    // Calculate priorities based on reward
    const priorities = all.map((exp) => Math.abs(exp.reward) ** alpha);
    const totalPriority = priorities.reduce((sum, p) => sum + p, 0);
    const probabilities = priorities.map((p) => p / totalPriority);

    // Sample based on priorities
    const samples: TExp[] = [];
    const indices: number[] = [];
    const weights: number[] = [];

    for (let i = 0; i < Math.min(batchSize, all.length); i++) {
      const index = this.weightedRandomChoice(probabilities);
      samples.push(all[index]);
      indices.push(index);
      weights.push(1 / (all.length * probabilities[index])); // Importance sampling weight
    }

    return {
      experiences: samples,
      indices,
      weights,
    };
  }

  /**
   * Query similar experiences using vector search
   */
  async querySimilar(experience: TExp, k: number = 10): Promise<TExp[]> {
    const embedding = await this.createEmbedding(experience);

    const results = await this.db.query({
      vector: embedding,
      k,
    });

    return results.map((result) => this.deserializeExperience(result.metadata));
  }

  /**
   * Get experiences by time range
   */
  async getByTimeRange(startTime: Date, endTime: Date): Promise<TExp[]> {
    // Filter in application layer (AgentDB doesn't support time range queries directly)
    const all = await this.getAll();

    return all.filter((exp) => {
      const timestamp = new Date(exp.timestamp);
      return timestamp >= startTime && timestamp <= endTime;
    });
  }

  /**
   * Get experiences with reward above threshold
   */
  async getHighReward(threshold: number): Promise<TExp[]> {
    const all = await this.getAll();
    return all.filter((exp) => exp.reward >= threshold);
  }

  /**
   * Get all experiences (use with caution for large buffers)
   */
  async getAll(): Promise<TExp[]> {
    // Query with large k to get all
    const results = await this.db.query({
      vector: await this.createRandomEmbedding(),
      k: this.experienceCount,
    });

    return results.map((result) => this.deserializeExperience(result.metadata));
  }

  /**
   * Clear all experiences
   */
  async clear(): Promise<void> {
    await this.db.clear();
    this.experienceCount = 0;
  }

  /**
   * Get buffer statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    utilizationPercent: number;
  } {
    return {
      size: this.experienceCount,
      maxSize: this.config.maxSize,
      utilizationPercent: (this.experienceCount / this.config.maxSize) * 100,
    };
  }

  /**
   * Create embedding from experience
   */
  private async createEmbedding(experience: TExp): Promise<number[]> {
    // Simple hash-based embedding (in production, use proper embedding model)
    const str = JSON.stringify({
      state: experience.state,
      action: experience.action,
      result: experience.result,
    });

    // Create deterministic embedding from string
    const embedding = new Array(384).fill(0).map((_, i) => {
      let hash = 0;
      for (let j = 0; j < str.length; j++) {
        hash = (hash << 5) - hash + str.charCodeAt(j) + i;
        hash = hash & hash; // Convert to 32-bit integer
      }
      return (hash / 2147483647); // Normalize to [-1, 1]
    });

    return embedding;
  }

  /**
   * Create random embedding for sampling
   */
  private async createRandomEmbedding(): Promise<number[]> {
    return new Array(384).fill(0).map(() => Math.random() * 2 - 1);
  }

  /**
   * Deserialize experience from metadata
   */
  private deserializeExperience(metadata: any): TExp {
    return {
      ...metadata,
      timestamp: new Date(metadata.timestamp),
    } as TExp;
  }

  /**
   * Evict oldest experience
   */
  private async evictOldest(): Promise<void> {
    const all = await this.getAll();

    if (all.length === 0) return;

    // Find oldest
    const oldest = all.reduce((oldest, current) =>
      current.timestamp < oldest.timestamp ? current : oldest
    );

    await this.db.delete(oldest.id);
    this.experienceCount--;
  }

  /**
   * Weighted random choice
   */
  private weightedRandomChoice(weights: number[]): number {
    const random = Math.random();
    let cumulative = 0;

    for (let i = 0; i < weights.length; i++) {
      cumulative += weights[i];
      if (random <= cumulative) {
        return i;
      }
    }

    return weights.length - 1;
  }

  /**
   * Export experiences to JSON
   */
  async export(): Promise<TExp[]> {
    return this.getAll();
  }

  /**
   * Import experiences from JSON
   */
  async import(experiences: TExp[]): Promise<void> {
    await this.storeBatch(experiences);
  }

  /**
   * Close database connection
   */
  async close(): Promise<void> {
    // AgentDB auto-manages connections
  }
}
