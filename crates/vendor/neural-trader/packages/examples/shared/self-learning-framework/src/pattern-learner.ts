/**
 * Pattern learner for identifying recurring patterns and strategies
 * Uses AgentDB vector search for similarity detection
 */

import { AgentDB } from 'agentdb';
import { ExperienceReplay, Experience } from './experience-replay';

export interface Pattern<T = any> {
  id: string;
  name: string;
  description: string;
  template: T;
  occurrences: number;
  successRate: number;
  avgReward: number;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface PatternMatch<T = any> {
  pattern: Pattern<T>;
  similarity: number;
  experience: Experience;
}

export interface LearnerConfig {
  minOccurrences: number;
  similarityThreshold: number;
  confidenceThreshold: number;
  dbPath?: string;
  namespace?: string;
}

export class PatternLearner<T = any> {
  private db: AgentDB;
  private replay: ExperienceReplay;
  private config: Required<LearnerConfig>;
  private patterns: Map<string, Pattern<T>> = new Map();

  constructor(replay: ExperienceReplay, config: LearnerConfig) {
    this.replay = replay;
    this.config = {
      minOccurrences: config.minOccurrences,
      similarityThreshold: config.similarityThreshold,
      confidenceThreshold: config.confidenceThreshold,
      dbPath: config.dbPath || './agentdb',
      namespace: config.namespace || 'pattern-learner',
    };

    this.db = new AgentDB({
      path: this.config.dbPath,
      dimensions: 384,
    });
  }

  /**
   * Learn patterns from experience replay buffer
   */
  async learnPatterns(): Promise<Pattern<T>[]> {
    console.log('🧠 Learning patterns from experience replay...');

    const experiences = await this.replay.getAll();

    if (experiences.length < this.config.minOccurrences) {
      console.log(`   ⚠️  Not enough experiences (${experiences.length} < ${this.config.minOccurrences})`);
      return [];
    }

    // Cluster similar experiences
    const clusters = await this.clusterExperiences(experiences);

    // Extract patterns from clusters
    const newPatterns: Pattern<T>[] = [];

    for (const cluster of clusters) {
      if (cluster.length >= this.config.minOccurrences) {
        const pattern = this.extractPattern(cluster);

        if (pattern.confidence >= this.config.confidenceThreshold) {
          this.patterns.set(pattern.id, pattern);
          newPatterns.push(pattern);

          // Store pattern in AgentDB
          await this.storePattern(pattern);
        }
      }
    }

    console.log(`   ✅ Learned ${newPatterns.length} new patterns`);
    return newPatterns;
  }

  /**
   * Match current state against known patterns
   */
  async matchPatterns(
    currentState: any,
    k: number = 5
  ): Promise<PatternMatch<T>[]> {
    const embedding = await this.createStateEmbedding(currentState);

    const results = await this.db.query({
      vector: embedding,
      k,
    });

    const matches: PatternMatch<T>[] = [];

    for (const result of results) {
      if (result.score >= this.config.similarityThreshold) {
        const patternId = result.metadata.patternId as string;
        const pattern = this.patterns.get(patternId);

        if (pattern) {
          matches.push({
            pattern,
            similarity: result.score,
            experience: result.metadata as any,
          });
        }
      }
    }

    return matches;
  }

  /**
   * Get pattern by ID
   */
  getPattern(id: string): Pattern<T> | undefined {
    return this.patterns.get(id);
  }

  /**
   * Get all patterns
   */
  getAllPatterns(): Pattern<T>[] {
    return Array.from(this.patterns.values());
  }

  /**
   * Get patterns by success rate
   */
  getTopPatterns(k: number = 10): Pattern<T>[] {
    return Array.from(this.patterns.values())
      .sort((a, b) => b.successRate - a.successRate)
      .slice(0, k);
  }

  /**
   * Get patterns by average reward
   */
  getBestRewardPatterns(k: number = 10): Pattern<T>[] {
    return Array.from(this.patterns.values())
      .sort((a, b) => b.avgReward - a.avgReward)
      .slice(0, k);
  }

  /**
   * Update pattern with new occurrence
   */
  async updatePattern(
    patternId: string,
    reward: number,
    success: boolean
  ): Promise<void> {
    const pattern = this.patterns.get(patternId);
    if (!pattern) return;

    // Update statistics
    pattern.occurrences++;
    pattern.avgReward =
      (pattern.avgReward * (pattern.occurrences - 1) + reward) /
      pattern.occurrences;
    pattern.successRate =
      (pattern.successRate * (pattern.occurrences - 1) + (success ? 1 : 0)) /
      pattern.occurrences;

    // Recalculate confidence
    pattern.confidence = this.calculateConfidence(pattern);

    // Update in storage
    await this.storePattern(pattern);
  }

  /**
   * Remove patterns with low confidence
   */
  async prunePatterns(minConfidence?: number): Promise<number> {
    const threshold = minConfidence ?? this.config.confidenceThreshold;
    let pruned = 0;

    for (const [id, pattern] of this.patterns.entries()) {
      if (pattern.confidence < threshold) {
        this.patterns.delete(id);
        await this.db.delete(id);
        pruned++;
      }
    }

    console.log(`   🧹 Pruned ${pruned} low-confidence patterns`);
    return pruned;
  }

  /**
   * Cluster similar experiences
   */
  private async clusterExperiences(
    experiences: Experience[]
  ): Promise<Experience[][]> {
    const clusters: Experience[][] = [];
    const visited = new Set<string>();

    for (const exp of experiences) {
      if (visited.has(exp.id)) continue;

      // Find similar experiences
      const similar = await this.replay.querySimilar(exp, 20);

      const cluster = similar.filter((e) => {
        if (visited.has(e.id)) return false;
        visited.add(e.id);
        return true;
      });

      if (cluster.length >= this.config.minOccurrences) {
        clusters.push(cluster);
      }
    }

    return clusters;
  }

  /**
   * Extract pattern from cluster of similar experiences
   */
  private extractPattern(cluster: Experience[]): Pattern<T> {
    const id = `pattern-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Calculate statistics
    const successfulExperiences = cluster.filter((e) => e.reward > 0);
    const successRate = successfulExperiences.length / cluster.length;
    const avgReward =
      cluster.reduce((sum, e) => sum + e.reward, 0) / cluster.length;

    // Extract template (simplified - take most common action)
    const template = this.extractTemplate(cluster);

    // Calculate confidence based on occurrences and consistency
    const confidence = this.calculateConfidenceFromCluster(cluster);

    return {
      id,
      name: `Pattern ${id}`,
      description: `Pattern learned from ${cluster.length} similar experiences`,
      template,
      occurrences: cluster.length,
      successRate,
      avgReward,
      confidence,
      metadata: {
        firstSeen: cluster[0].timestamp,
        lastSeen: cluster[cluster.length - 1].timestamp,
      },
    };
  }

  /**
   * Extract template from cluster
   */
  private extractTemplate(cluster: Experience[]): T {
    // Simplified: return the action from the best-performing experience
    const best = cluster.reduce((best, current) =>
      current.reward > best.reward ? current : best
    );

    return best.action as T;
  }

  /**
   * Calculate confidence from cluster
   */
  private calculateConfidenceFromCluster(cluster: Experience[]): number {
    const occurrences = cluster.length;
    const variance = this.calculateRewardVariance(cluster);

    // Confidence increases with occurrences and consistency (low variance)
    const occurrenceScore = Math.min(occurrences / 100, 1.0);
    const consistencyScore = 1.0 / (1.0 + variance);

    return (occurrenceScore + consistencyScore) / 2;
  }

  /**
   * Calculate confidence for existing pattern
   */
  private calculateConfidence(pattern: Pattern<T>): number {
    const occurrenceScore = Math.min(pattern.occurrences / 100, 1.0);
    const successScore = pattern.successRate;

    return (occurrenceScore + successScore) / 2;
  }

  /**
   * Calculate reward variance
   */
  private calculateRewardVariance(experiences: Experience[]): number {
    const mean =
      experiences.reduce((sum, e) => sum + e.reward, 0) / experiences.length;
    const variance =
      experiences.reduce((sum, e) => sum + Math.pow(e.reward - mean, 2), 0) /
      experiences.length;

    return variance;
  }

  /**
   * Store pattern in AgentDB
   */
  private async storePattern(pattern: Pattern<T>): Promise<void> {
    const embedding = await this.createPatternEmbedding(pattern);

    await this.db.add({
      id: pattern.id,
      vector: embedding,
      metadata: {
        patternId: pattern.id,
        ...pattern,
      },
    });
  }

  /**
   * Create embedding from pattern
   */
  private async createPatternEmbedding(pattern: Pattern<T>): Promise<number[]> {
    const str = JSON.stringify(pattern.template);

    // Create deterministic embedding
    const embedding = new Array(384).fill(0).map((_, i) => {
      let hash = 0;
      for (let j = 0; j < str.length; j++) {
        hash = (hash << 5) - hash + str.charCodeAt(j) + i;
        hash = hash & hash;
      }
      return hash / 2147483647;
    });

    return embedding;
  }

  /**
   * Create embedding from state
   */
  private async createStateEmbedding(state: any): Promise<number[]> {
    const str = JSON.stringify(state);

    const embedding = new Array(384).fill(0).map((_, i) => {
      let hash = 0;
      for (let j = 0; j < str.length; j++) {
        hash = (hash << 5) - hash + str.charCodeAt(j) + i;
        hash = hash & hash;
      }
      return hash / 2147483647;
    });

    return embedding;
  }

  /**
   * Export patterns to JSON
   */
  export(): Pattern<T>[] {
    return this.getAllPatterns();
  }

  /**
   * Import patterns from JSON
   */
  async import(patterns: Pattern<T>[]): Promise<void> {
    for (const pattern of patterns) {
      this.patterns.set(pattern.id, pattern);
      await this.storePattern(pattern);
    }
  }

  /**
   * Clear all patterns
   */
  async clear(): Promise<void> {
    this.patterns.clear();
    await this.db.clear();
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalPatterns: number;
    avgSuccessRate: number;
    avgReward: number;
    avgConfidence: number;
  } {
    const patterns = this.getAllPatterns();

    if (patterns.length === 0) {
      return {
        totalPatterns: 0,
        avgSuccessRate: 0,
        avgReward: 0,
        avgConfidence: 0,
      };
    }

    return {
      totalPatterns: patterns.length,
      avgSuccessRate:
        patterns.reduce((sum, p) => sum + p.successRate, 0) / patterns.length,
      avgReward:
        patterns.reduce((sum, p) => sum + p.avgReward, 0) / patterns.length,
      avgConfidence:
        patterns.reduce((sum, p) => sum + p.confidence, 0) / patterns.length,
    };
  }
}
