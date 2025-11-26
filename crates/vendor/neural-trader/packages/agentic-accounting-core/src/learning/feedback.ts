/**
 * Feedback Loop System
 * Process feedback and improve agent performance
 */

import { logger } from '../utils/logger';

export interface Feedback {
  id: string;
  agentId: string;
  taskId: string;
  rating: number; // 0-1 scale
  comments?: string;
  metrics?: {
    accuracy?: number;
    speed?: number;
    quality?: number;
  };
  timestamp: Date;
}

export interface PerformanceMetrics {
  agentId: string;
  period: {
    start: Date;
    end: Date;
  };
  averageRating: number;
  totalFeedback: number;
  accuracyTrend: number[];
  speedTrend: number[];
  qualityTrend: number[];
}

export class FeedbackLoopService {
  private feedbackStore: Map<string, Feedback[]> = new Map();

  /**
   * Process feedback for an agent
   */
  async processFeedback(feedback: Feedback): Promise<void> {
    logger.info('Processing feedback', {
      agentId: feedback.agentId,
      rating: feedback.rating
    });

    // Store feedback
    const agentFeedback = this.feedbackStore.get(feedback.agentId) || [];
    agentFeedback.push(feedback);
    this.feedbackStore.set(feedback.agentId, agentFeedback);

    // Analyze feedback
    await this.analyzeFeedback(feedback);

    // Trigger improvements if needed
    if (feedback.rating < 0.5) {
      await this.triggerImprovement(feedback);
    }
  }

  /**
   * Analyze feedback patterns
   */
  private async analyzeFeedback(feedback: Feedback): Promise<void> {
    const agentFeedback = this.feedbackStore.get(feedback.agentId) || [];

    // Calculate trends
    const recentFeedback = agentFeedback.slice(-10);
    const averageRating = recentFeedback.reduce((sum, f) => sum + f.rating, 0) / recentFeedback.length;

    logger.debug('Feedback analysis', {
      agentId: feedback.agentId,
      recentCount: recentFeedback.length,
      averageRating
    });

    // Detect degradation
    if (averageRating < 0.6) {
      logger.warn('Agent performance degradation detected', {
        agentId: feedback.agentId,
        averageRating
      });
    }
  }

  /**
   * Trigger performance improvement
   */
  private async triggerImprovement(feedback: Feedback): Promise<void> {
    logger.info('Triggering performance improvement', {
      agentId: feedback.agentId,
      rating: feedback.rating
    });

    // In production, this would:
    // 1. Analyze failure patterns
    // 2. Adjust agent parameters
    // 3. Trigger retraining
    // 4. Update decision thresholds
  }

  /**
   * Get performance metrics for agent
   */
  async getPerformanceMetrics(
    agentId: string,
    startDate: Date,
    endDate: Date
  ): Promise<PerformanceMetrics> {
    const agentFeedback = this.feedbackStore.get(agentId) || [];

    // Filter by date range
    const periodFeedback = agentFeedback.filter(
      f => f.timestamp >= startDate && f.timestamp <= endDate
    );

    // Calculate metrics
    const averageRating = periodFeedback.reduce((sum, f) => sum + f.rating, 0) / periodFeedback.length || 0;

    const accuracyTrend = periodFeedback.map(f => f.metrics?.accuracy || 0);
    const speedTrend = periodFeedback.map(f => f.metrics?.speed || 0);
    const qualityTrend = periodFeedback.map(f => f.metrics?.quality || 0);

    return {
      agentId,
      period: { start: startDate, end: endDate },
      averageRating,
      totalFeedback: periodFeedback.length,
      accuracyTrend,
      speedTrend,
      qualityTrend
    };
  }

  /**
   * Generate improvement recommendations
   */
  async generateRecommendations(agentId: string): Promise<string[]> {
    const agentFeedback = this.feedbackStore.get(agentId) || [];
    const recommendations: string[] = [];

    if (agentFeedback.length === 0) {
      return ['Insufficient feedback data for recommendations'];
    }

    const recentFeedback = agentFeedback.slice(-20);
    const averageRating = recentFeedback.reduce((sum, f) => sum + f.rating, 0) / recentFeedback.length;

    // Analyze patterns and generate recommendations
    if (averageRating < 0.7) {
      recommendations.push('Review and improve decision-making algorithms');
      recommendations.push('Increase training data quality');
    }

    const lowAccuracy = recentFeedback.filter(f => (f.metrics?.accuracy || 1) < 0.8).length;
    if (lowAccuracy > recentFeedback.length * 0.3) {
      recommendations.push('Focus on improving calculation accuracy');
      recommendations.push('Add more validation checks');
    }

    const slowSpeed = recentFeedback.filter(f => (f.metrics?.speed || 1) < 0.7).length;
    if (slowSpeed > recentFeedback.length * 0.3) {
      recommendations.push('Optimize performance bottlenecks');
      recommendations.push('Consider caching frequently used data');
    }

    return recommendations;
  }

  /**
   * Batch process feedback
   */
  async processBatch(feedbackList: Feedback[]): Promise<void> {
    await Promise.all(feedbackList.map(f => this.processFeedback(f)));
  }

  /**
   * Get all feedback for agent
   */
  getFeedback(agentId: string): Feedback[] {
    return this.feedbackStore.get(agentId) || [];
  }

  /**
   * Clear feedback history
   */
  clearFeedback(agentId?: string): void {
    if (agentId) {
      this.feedbackStore.delete(agentId);
    } else {
      this.feedbackStore.clear();
    }
    logger.info('Feedback history cleared', { agentId });
  }
}
