/**
 * Learning Agent
 * Autonomous agent for continuous system improvement
 * Target: 10%+ accuracy improvement per quarter
 */

import { BaseAgent } from '../base/agent';
import { ReasoningBankService, Trajectory, Verdict } from '@neural-trader/agentic-accounting-core';
import { FeedbackLoopService, Feedback } from '@neural-trader/agentic-accounting-core';

export interface LearningAgentConfig {
  learningRate?: number;
  minSuccessRate?: number;
  feedbackThreshold?: number;
}

export class LearningAgent extends BaseAgent {
  private reasoningBank: ReasoningBankService;
  private feedbackLoop: FeedbackLoopService;
  private learningConfig: LearningAgentConfig;

  constructor(config: LearningAgentConfig = {}) {
    super({
      agentId: 'learning-agent',
      agentType: 'LearningAgent',
      enableLearning: true,
      enableMetrics: true
    });
    this.reasoningBank = new ReasoningBankService();
    this.feedbackLoop = new FeedbackLoopService();
    this.learningConfig = {
      learningRate: config.learningRate || 0.1,
      minSuccessRate: config.minSuccessRate || 0.7,
      feedbackThreshold: config.feedbackThreshold || 0.5
    };
  }

  /**
   * Execute learning task
   */
  async execute(task: {
    action: 'train' | 'process_feedback' | 'analyze_performance' | 'optimize';
    agentId?: string;
    trajectory?: Trajectory;
    feedback?: Feedback;
    startDate?: Date;
    endDate?: Date;
  }): Promise<any> {
    this.logger.info(`Executing learning task: ${task.action}`);

    try {
      switch (task.action) {
        case 'train':
          return await this.trainOnTrajectory(task.trajectory!);
        case 'process_feedback':
          return await this.processFeedback(task.feedback!);
        case 'analyze_performance':
          return await this.analyzePerformance(task.agentId!, task.startDate!, task.endDate!);
        case 'optimize':
          return await this.optimizeAgent(task.agentId!);
        default:
          throw new Error(`Unknown action: ${task.action}`);
      }
    } catch (error) {
      this.logger.error('Learning task failed', { error, task });
      throw error;
    }
  }

  /**
   * Train on agent trajectory
   */
  private async trainOnTrajectory(trajectory: Trajectory): Promise<any> {
    this.logger.info('Training on trajectory', {
      agentId: trajectory.agentId,
      action: trajectory.action
    });

    // Store trajectory in ReasoningBank
    await this.reasoningBank.storeTrajectory(trajectory);

    // Find similar past trajectories
    const similarTrajectories = await this.reasoningBank.findSimilarTrajectories(
      trajectory.context,
      trajectory.agentId
    );

    // Analyze outcomes
    const analysis = {
      trajectory,
      similarCount: similarTrajectories.length,
      patterns: this.extractPatterns(trajectory, similarTrajectories)
    };

    // Log learning
    await this.learn({
      action: 'train_trajectory',
      agentId: trajectory.agentId,
      similarTrajectories: similarTrajectories.length,
      patternsFound: analysis.patterns.length
    });

    return analysis;
  }

  /**
   * Process feedback and improve
   */
  private async processFeedback(feedback: Feedback): Promise<any> {
    this.logger.info('Processing feedback', {
      agentId: feedback.agentId,
      rating: feedback.rating
    });

    // Store feedback
    await this.feedbackLoop.processFeedback(feedback);

    // Learn from feedback in ReasoningBank
    await this.reasoningBank.learnFromFeedback(feedback.taskId, {
      isSuccessful: feedback.rating >= this.config.feedbackThreshold!,
      score: feedback.rating,
      message: feedback.comments || ''
    });

    // Generate recommendations
    const recommendations = await this.feedbackLoop.generateRecommendations(feedback.agentId);

    // Log learning
    await this.learn({
      action: 'process_feedback',
      agentId: feedback.agentId,
      rating: feedback.rating,
      recommendationsGenerated: recommendations.length
    });

    return {
      feedback,
      recommendations,
      processedAt: new Date()
    };
  }

  /**
   * Analyze agent performance
   */
  private async analyzePerformance(
    agentId: string,
    startDate: Date,
    endDate: Date
  ): Promise<any> {
    this.logger.info('Analyzing agent performance', { agentId });

    // Get performance metrics
    const metrics = await this.feedbackLoop.getPerformanceMetrics(agentId, startDate, endDate);

    // Get ReasoningBank metrics
    const rbMetrics = await this.reasoningBank.getAgentMetrics(agentId);

    // Calculate improvement
    const improvement = this.calculateImprovement(metrics);

    const analysis = {
      agentId,
      period: { start: startDate, end: endDate },
      performanceMetrics: metrics,
      reasoningBankMetrics: rbMetrics,
      improvement,
      recommendations: await this.feedbackLoop.generateRecommendations(agentId)
    };

    // Log learning
    await this.learn({
      action: 'analyze_performance',
      agentId,
      averageRating: metrics.averageRating,
      improvement: improvement.percentage
    });

    return analysis;
  }

  /**
   * Optimize agent based on learned patterns
   */
  private async optimizeAgent(agentId: string): Promise<any> {
    this.logger.info('Optimizing agent', { agentId });

    // Get successful patterns
    const patterns = await this.reasoningBank.getSuccessfulPatterns(
      agentId,
      this.learningConfig.minSuccessRate
    );

    // Generate optimization plan
    const optimizationPlan = {
      agentId,
      patternsApplied: patterns.length,
      improvements: [
        'Apply successful decision patterns',
        'Increase accuracy thresholds',
        'Optimize processing speed'
      ],
      expectedImprovement: '10-15%',
      createdAt: new Date()
    };

    // Log learning
    await this.learn({
      action: 'optimize_agent',
      agentId,
      patternsApplied: patterns.length
    });

    return optimizationPlan;
  }

  /**
   * Extract patterns from trajectories
   */
  private extractPatterns(trajectory: Trajectory, similar: Trajectory[]): any[] {
    const patterns: any[] = [];

    // Analyze common features
    if (similar.length > 0) {
      patterns.push({
        type: 'common_context',
        count: similar.length,
        confidence: similar.length / 10
      });
    }

    // Analyze action outcomes
    patterns.push({
      type: 'action_pattern',
      action: trajectory.action,
        context: trajectory.context,
        result: trajectory.result
      });

    return patterns;
  }

  /**
   * Calculate improvement over time
   */
  private calculateImprovement(metrics: any): any {
    const { accuracyTrend, speedTrend, qualityTrend } = metrics;

    const calculateTrendImprovement = (trend: number[]) => {
      if (trend.length < 2) return 0;
      const first = trend.slice(0, Math.floor(trend.length / 2));
      const second = trend.slice(Math.floor(trend.length / 2));
      const firstAvg = first.reduce((a, b) => a + b, 0) / first.length;
      const secondAvg = second.reduce((a, b) => a + b, 0) / second.length;
      return ((secondAvg - firstAvg) / firstAvg) * 100;
    };

    return {
      accuracy: calculateTrendImprovement(accuracyTrend),
      speed: calculateTrendImprovement(speedTrend),
      quality: calculateTrendImprovement(qualityTrend),
      percentage: calculateTrendImprovement(accuracyTrend) // Overall based on accuracy
    };
  }

  /**
   * Run overnight training batch
   */
  async runBatchTraining(trajectories: Trajectory[]): Promise<any> {
    this.logger.info(`Running batch training on ${trajectories.length} trajectories`);

    const results = await Promise.all(
      trajectories.map(t => this.trainOnTrajectory(t))
    );

    return {
      totalTrajectories: trajectories.length,
      trained: results.length,
      timestamp: new Date()
    };
  }

  /**
   * Generate learning report
   */
  async generateLearningReport(agentId: string, period: { start: Date; end: Date }): Promise<any> {
    this.logger.info('Generating learning report', { agentId });

    const performance = await this.analyzePerformance(agentId, period.start, period.end);
    const recommendations = await this.feedbackLoop.generateRecommendations(agentId);

    return {
      agentId,
      period,
      performance,
      recommendations,
      generatedAt: new Date()
    };
  }
}
