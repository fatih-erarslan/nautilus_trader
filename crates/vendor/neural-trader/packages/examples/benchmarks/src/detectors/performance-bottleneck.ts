/**
 * Performance bottleneck detection
 */

export interface BottleneckReport {
  hotspots: Hotspot[];
  recommendations: string[];
  overallScore: number; // 0-100
}

export interface Hotspot {
  name: string;
  duration: number;
  percentage: number;
  callCount: number;
  avgDuration: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface ProfiledFunction {
  name: string;
  calls: number;
  totalTime: number;
  ownTime: number;
}

export class PerformanceBottleneckDetector {
  private profiles: Map<string, ProfiledFunction> = new Map();
  private callStack: Array<{ name: string; startTime: number }> = [];

  /**
   * Start profiling a function
   */
  start(functionName: string): void {
    this.callStack.push({
      name: functionName,
      startTime: performance.now()
    });
  }

  /**
   * Stop profiling a function
   */
  stop(functionName: string): void {
    const entry = this.callStack.pop();
    if (!entry || entry.name !== functionName) {
      throw new Error(`Mismatched profiling calls: expected ${entry?.name}, got ${functionName}`);
    }

    const duration = performance.now() - entry.startTime;

    const profile = this.profiles.get(functionName) || {
      name: functionName,
      calls: 0,
      totalTime: 0,
      ownTime: 0
    };

    profile.calls++;
    profile.totalTime += duration;
    profile.ownTime += duration;

    // Subtract time from parent
    if (this.callStack.length > 0) {
      const parent = this.callStack[this.callStack.length - 1];
      const parentProfile = this.profiles.get(parent.name);
      if (parentProfile) {
        parentProfile.ownTime -= duration;
      }
    }

    this.profiles.set(functionName, profile);
  }

  /**
   * Profile a function automatically
   */
  async profile<T>(
    name: string,
    fn: () => Promise<T>
  ): Promise<T> {
    this.start(name);
    try {
      return await fn();
    } finally {
      this.stop(name);
    }
  }

  /**
   * Analyze and generate bottleneck report
   */
  analyze(): BottleneckReport {
    const hotspots: Hotspot[] = [];
    const totalTime = Array.from(this.profiles.values())
      .reduce((sum, p) => sum + p.totalTime, 0);

    for (const profile of this.profiles.values()) {
      const percentage = (profile.totalTime / totalTime) * 100;
      const avgDuration = profile.totalTime / profile.calls;

      let severity: 'low' | 'medium' | 'high' | 'critical';
      if (percentage > 50) severity = 'critical';
      else if (percentage > 25) severity = 'high';
      else if (percentage > 10) severity = 'medium';
      else severity = 'low';

      hotspots.push({
        name: profile.name,
        duration: profile.totalTime,
        percentage,
        callCount: profile.calls,
        avgDuration,
        severity
      });
    }

    // Sort by duration
    hotspots.sort((a, b) => b.duration - a.duration);

    // Generate recommendations
    const recommendations = this.generateRecommendations(hotspots);

    // Calculate overall score
    const overallScore = this.calculatePerformanceScore(hotspots);

    return {
      hotspots,
      recommendations,
      overallScore
    };
  }

  /**
   * Clear profiling data
   */
  clear(): void {
    this.profiles.clear();
    this.callStack = [];
  }

  /**
   * Get function profile
   */
  getProfile(functionName: string): ProfiledFunction | undefined {
    return this.profiles.get(functionName);
  }

  private generateRecommendations(hotspots: Hotspot[]): string[] {
    const recommendations: string[] = [];

    for (const hotspot of hotspots) {
      if (hotspot.severity === 'critical' || hotspot.severity === 'high') {
        recommendations.push(
          `Optimize ${hotspot.name}: takes ${hotspot.percentage.toFixed(1)}% of total time`
        );

        if (hotspot.callCount > 1000) {
          recommendations.push(
            `Consider caching or memoization for ${hotspot.name} (called ${hotspot.callCount} times)`
          );
        }

        if (hotspot.avgDuration > 100) {
          recommendations.push(
            `${hotspot.name} has high average duration (${hotspot.avgDuration.toFixed(2)}ms), consider async processing`
          );
        }
      }
    }

    // Check for optimization opportunities
    const highFrequencyCalls = hotspots.filter(h => h.callCount > 500);
    if (highFrequencyCalls.length > 0) {
      recommendations.push(
        'Consider batch processing for frequently called functions'
      );
    }

    const longRunning = hotspots.filter(h => h.avgDuration > 50);
    if (longRunning.length > 0) {
      recommendations.push(
        'Consider parallelizing long-running operations'
      );
    }

    return recommendations;
  }

  private calculatePerformanceScore(hotspots: Hotspot[]): number {
    if (hotspots.length === 0) return 100;

    // Calculate penalty based on severity
    const penalties = hotspots.reduce((sum, h) => {
      const severityPenalty = {
        low: 0,
        medium: 5,
        high: 15,
        critical: 30
      };
      return sum + severityPenalty[h.severity];
    }, 0);

    return Math.max(0, 100 - penalties);
  }
}

/**
 * Create a profiled wrapper for a function
 */
export function profileFunction<T extends (...args: any[]) => any>(
  detector: PerformanceBottleneckDetector,
  name: string,
  fn: T
): T {
  return ((...args: any[]) => {
    return detector.profile(name, () => fn(...args));
  }) as T;
}
