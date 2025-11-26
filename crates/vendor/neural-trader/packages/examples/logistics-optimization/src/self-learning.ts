/**
 * Self-learning system for logistics optimization
 * Uses AgentDB for pattern storage and adaptive learning
 */

import { Solution, TrafficPattern, LearningMetrics, Customer } from './types';

export interface MemoryEntry {
  id: string;
  timestamp: number;
  solution: Solution;
  metrics: LearningMetrics;
  context: {
    numCustomers: number;
    numVehicles: number;
    avgDemand: number;
    timeOfDay: number;
    dayOfWeek: number;
  };
}

export class SelfLearningSystem {
  private memoryStore: Map<string, MemoryEntry>;
  private trafficPatterns: Map<string, TrafficPattern>;
  private episodeHistory: LearningMetrics[];
  private learningRate: number;

  constructor(learningRate: number = 0.1) {
    this.memoryStore = new Map();
    this.trafficPatterns = new Map();
    this.episodeHistory = [];
    this.learningRate = learningRate;
  }

  /**
   * Store solution and learn from it
   */
  async learnFromSolution(
    solution: Solution,
    customers: Customer[],
    metrics: LearningMetrics
  ): Promise<void> {
    const now = new Date();
    const context = {
      numCustomers: customers.length,
      numVehicles: solution.routes.length,
      avgDemand: customers.reduce((sum, c) => sum + c.demand, 0) / customers.length,
      timeOfDay: now.getHours(),
      dayOfWeek: now.getDay()
    };

    const entry: MemoryEntry = {
      id: metrics.episodeId,
      timestamp: metrics.timestamp,
      solution,
      metrics,
      context
    };

    this.memoryStore.set(entry.id, entry);
    this.episodeHistory.push(metrics);

    // Learn traffic patterns from routes
    await this.updateTrafficPatterns(solution);

    // Prune old memories if needed
    if (this.memoryStore.size > 1000) {
      this.pruneMemories();
    }

    console.log(`Learned from episode ${metrics.episodeId}. Memory size: ${this.memoryStore.size}`);
  }

  /**
   * Update traffic patterns based on route performance
   */
  private async updateTrafficPatterns(solution: Solution): Promise<void> {
    const now = new Date();
    const timeOfDay = now.getHours();
    const dayOfWeek = now.getDay();

    for (const route of solution.routes) {
      for (let i = 0; i < route.customers.length - 1; i++) {
        const from = route.customers[i].location.id;
        const to = route.customers[i + 1].location.id;
        const key = `${from}-${to}-${timeOfDay}-${dayOfWeek}`;

        const existing = this.trafficPatterns.get(key);

        if (existing) {
          // Update with exponential moving average
          const newSpeed = existing.avgSpeed * (1 - this.learningRate) +
                          50 * this.learningRate; // Placeholder speed
          const newReliability = existing.reliability * (1 - this.learningRate) +
                                0.9 * this.learningRate; // Placeholder reliability

          this.trafficPatterns.set(key, {
            ...existing,
            avgSpeed: newSpeed,
            reliability: newReliability
          });
        } else {
          // Create new pattern
          this.trafficPatterns.set(key, {
            fromLocationId: from,
            toLocationId: to,
            timeOfDay,
            dayOfWeek,
            avgSpeed: 50, // Default
            reliability: 0.8
          });
        }
      }
    }
  }

  /**
   * Retrieve similar past solutions
   */
  async retrieveSimilarSolutions(
    numCustomers: number,
    numVehicles: number,
    topK: number = 5
  ): Promise<MemoryEntry[]> {
    const candidates: Array<{ entry: MemoryEntry; similarity: number }> = [];

    for (const [id, entry] of this.memoryStore.entries()) {
      const similarity = this.calculateContextSimilarity(
        { numCustomers, numVehicles, avgDemand: 0, timeOfDay: 0, dayOfWeek: 0 },
        entry.context
      );
      candidates.push({ entry, similarity });
    }

    // Sort by similarity and quality
    candidates.sort((a, b) => {
      const similarityDiff = b.similarity - a.similarity;
      if (Math.abs(similarityDiff) < 0.1) {
        // If similar, prefer better quality
        return a.entry.metrics.solutionQuality - b.entry.metrics.solutionQuality;
      }
      return similarityDiff;
    });

    return candidates.slice(0, topK).map(c => c.entry);
  }

  /**
   * Calculate context similarity
   */
  private calculateContextSimilarity(
    ctx1: { numCustomers: number; numVehicles: number; avgDemand: number; timeOfDay: number; dayOfWeek: number },
    ctx2: { numCustomers: number; numVehicles: number; avgDemand: number; timeOfDay: number; dayOfWeek: number }
  ): number {
    const customerDiff = Math.abs(ctx1.numCustomers - ctx2.numCustomers) /
                        Math.max(ctx1.numCustomers, ctx2.numCustomers);
    const vehicleDiff = Math.abs(ctx1.numVehicles - ctx2.numVehicles) /
                       Math.max(ctx1.numVehicles, ctx2.numVehicles);

    // Weighted similarity score
    return 1 - (customerDiff * 0.5 + vehicleDiff * 0.5);
  }

  /**
   * Get traffic prediction for a route segment
   */
  getTrafficPrediction(
    fromLocationId: string,
    toLocationId: string,
    timeOfDay: number,
    dayOfWeek: number
  ): TrafficPattern | null {
    const key = `${fromLocationId}-${toLocationId}-${timeOfDay}-${dayOfWeek}`;
    return this.trafficPatterns.get(key) || null;
  }

  /**
   * Get learning statistics
   */
  getStatistics(): {
    totalEpisodes: number;
    avgSolutionQuality: number;
    avgComputeTime: number;
    improvementRate: number;
    trafficPatternsLearned: number;
  } {
    if (this.episodeHistory.length === 0) {
      return {
        totalEpisodes: 0,
        avgSolutionQuality: 0,
        avgComputeTime: 0,
        improvementRate: 0,
        trafficPatternsLearned: 0
      };
    }

    const avgQuality = this.episodeHistory.reduce((sum, m) => sum + m.solutionQuality, 0) /
                      this.episodeHistory.length;
    const avgTime = this.episodeHistory.reduce((sum, m) => sum + m.computeTime, 0) /
                   this.episodeHistory.length;

    // Calculate improvement rate (last 10 vs first 10)
    let improvementRate = 0;
    if (this.episodeHistory.length >= 20) {
      const first10 = this.episodeHistory.slice(0, 10);
      const last10 = this.episodeHistory.slice(-10);

      const first10Avg = first10.reduce((sum, m) => sum + m.solutionQuality, 0) / 10;
      const last10Avg = last10.reduce((sum, m) => sum + m.solutionQuality, 0) / 10;

      improvementRate = ((first10Avg - last10Avg) / first10Avg) * 100;
    }

    return {
      totalEpisodes: this.episodeHistory.length,
      avgSolutionQuality: avgQuality,
      avgComputeTime: avgTime,
      improvementRate,
      trafficPatternsLearned: this.trafficPatterns.size
    };
  }

  /**
   * Prune old or low-quality memories
   */
  private pruneMemories(): void {
    const entries = Array.from(this.memoryStore.entries());

    // Sort by quality (keep best solutions)
    entries.sort((a, b) => a[1].metrics.solutionQuality - b[1].metrics.solutionQuality);

    // Remove worst 20%
    const removeCount = Math.floor(entries.length * 0.2);
    for (let i = entries.length - removeCount; i < entries.length; i++) {
      this.memoryStore.delete(entries[i][0]);
    }
  }

  /**
   * Export learned patterns
   */
  exportPatterns(): {
    trafficPatterns: TrafficPattern[];
    topSolutions: Solution[];
    statistics: any;
  } {
    const trafficPatterns = Array.from(this.trafficPatterns.values());

    // Get top 10 solutions
    const entries = Array.from(this.memoryStore.values());
    entries.sort((a, b) => a.metrics.solutionQuality - b.metrics.solutionQuality);
    const topSolutions = entries.slice(0, 10).map(e => e.solution);

    return {
      trafficPatterns,
      topSolutions,
      statistics: this.getStatistics()
    };
  }

  /**
   * Import learned patterns
   */
  importPatterns(data: {
    trafficPatterns: TrafficPattern[];
    topSolutions?: Solution[];
  }): void {
    // Import traffic patterns
    for (const pattern of data.trafficPatterns) {
      const key = `${pattern.fromLocationId}-${pattern.toLocationId}-${pattern.timeOfDay}-${pattern.dayOfWeek}`;
      this.trafficPatterns.set(key, pattern);
    }

    console.log(`Imported ${data.trafficPatterns.length} traffic patterns`);
  }

  /**
   * Reset learning state
   */
  reset(): void {
    this.memoryStore.clear();
    this.trafficPatterns.clear();
    this.episodeHistory = [];
    console.log('Learning system reset');
  }
}
