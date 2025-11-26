/**
 * @neural-trader/example-healthcare-optimization
 *
 * Healthcare optimization with self-learning patient forecasting and swarm-based staff scheduling.
 * Features:
 * - Patient arrival forecasting with uncertainty quantification
 * - Queue theory optimization for patient flow
 * - Staff scheduling with skill constraints
 * - Swarm intelligence for exploring scheduling heuristics
 * - Memory-persistent hospital metrics
 * - Privacy-preserving (synthetic data only)
 */

import OpenAI from 'openai';
import { AgentDB } from 'agentdb';
import { ArrivalForecaster, type ArrivalForecasterConfig } from './arrival-forecaster.js';
import { Scheduler, type SchedulerConfig } from './scheduler.js';
import { QueueOptimizer, type QueueOptimizerConfig } from './queue-optimizer.js';
import { SwarmCoordinator } from './swarm.js';
import type {
  HealthcareOptimizationConfig,
  Patient,
  StaffMember,
  OptimizationResult,
  ResourcePool,
  TriageDecision
} from './types.js';

export class HealthcareOptimizer {
  private config: HealthcareOptimizationConfig;
  private forecaster: ArrivalForecaster;
  private scheduler: Scheduler;
  private queueOptimizer: QueueOptimizer;
  private swarm: SwarmCoordinator;
  private memory: AgentDB;
  private openai?: OpenAI;

  constructor(config: HealthcareOptimizationConfig) {
    this.config = config;

    // Initialize components
    const forecasterConfig: ArrivalForecasterConfig = {
      agentdbPath: config.agentdbPath,
      enableNapiRS: config.enableNapiRS,
      privacy: config.privacy,
      lookbackDays: 90,
      forecastHorizon: 24,
      confidenceLevel: 0.95
    };
    this.forecaster = new ArrivalForecaster(forecasterConfig);

    const schedulerConfig: SchedulerConfig = {
      planningHorizonDays: 7,
      shiftDuration: 8,
      costPerConstraintViolation: 1000
    };
    this.scheduler = new Scheduler(schedulerConfig);

    const queueConfig: QueueOptimizerConfig = {
      targetUtilization: 0.75,
      maxWaitTime: 30,
      abandonmentThreshold: 120,
      reallocateInterval: 60
    };
    this.queueOptimizer = new QueueOptimizer(queueConfig);

    this.swarm = new SwarmCoordinator(config.swarm, config.agentdbPath);

    this.memory = new AgentDB({
      dbPath: config.agentdbPath,
      enableQuantization: true,
      quantizationBits: 8
    });

    // Initialize OpenRouter if API key provided
    if (config.openRouterApiKey) {
      this.openai = new OpenAI({
        apiKey: config.openRouterApiKey,
        baseURL: 'https://openrouter.ai/api/v1'
      });
    }
  }

  /**
   * Train forecaster with historical data
   */
  async trainForecaster(historicalData: Array<{ timestamp: Date; arrivals: number }>): Promise<void> {
    console.log('🧠 Training arrival forecaster...');
    await this.forecaster.train(historicalData);
    console.log('✅ Forecaster trained');
  }

  /**
   * Add staff member
   */
  addStaff(staff: StaffMember): void {
    this.scheduler.addStaff(staff);
  }

  /**
   * Add resource pool for queue management
   */
  addResourcePool(pool: ResourcePool): void {
    this.queueOptimizer.addResourcePool(pool);
  }

  /**
   * Run full optimization workflow
   */
  async optimize(startDate: Date = new Date()): Promise<OptimizationResult> {
    console.log('\n🏥 Starting healthcare optimization workflow...\n');

    // Step 1: Forecast patient arrivals
    console.log('📊 Step 1: Forecasting patient arrivals...');
    const forecasts = await this.forecaster.forecastHorizon(startDate);
    const totalPredicted = forecasts.reduce((sum, f) => sum + f.predictedArrivals, 0);
    console.log(`  Predicted ${totalPredicted} arrivals over next ${forecasts.length} hours`);

    // Step 2: Optimize schedule using swarm
    console.log('\n🐝 Step 2: Optimizing staff schedule with swarm intelligence...');
    const swarmResult = await this.swarm.optimize(
      forecasts,
      this.config.constraints,
      this.config.optimization,
      {
        planningHorizonDays: 7,
        shiftDuration: 8,
        costPerConstraintViolation: 1000
      },
      startDate
    );

    console.log(`  Explored ${swarmResult.exploredSolutions} solutions in ${swarmResult.iterations} iterations`);
    console.log(`  Best coverage: ${(swarmResult.bestSolution.coverageScore * 100).toFixed(1)}%`);
    console.log(`  Fairness score: ${(swarmResult.bestSolution.fairnessScore * 100).toFixed(1)}%`);

    // Step 3: Simulate queue performance
    console.log('\n⏱️  Step 3: Simulating queue performance...');
    const queueMetrics = await this.simulateQueue(forecasts, swarmResult.bestSolution.shifts);

    // Step 4: Optimize resource allocation
    console.log('\n🎯 Step 4: Optimizing resource allocation...');
    const resourceOptimization = this.queueOptimizer.optimizeResources(this.config.optimization);

    console.log(`  Expected wait time reduction: ${resourceOptimization.expectedImpact.waitTimeReduction.toFixed(1)} minutes`);
    console.log(`  Recommendations: ${resourceOptimization.recommendations.length}`);

    // Calculate overall quality score
    const qualityScore = this.calculateQualityScore(
      swarmResult.bestSolution,
      queueMetrics
    );

    console.log(`\n✨ Optimization complete!`);
    console.log(`  Overall quality score: ${(qualityScore * 100).toFixed(1)}%`);

    // Store results in memory
    await this.storeResults({
      schedule: swarmResult.bestSolution,
      expectedWaitTime: queueMetrics.averageWaitTime,
      expectedUtilization: queueMetrics.utilization,
      totalCost: swarmResult.bestSolution.totalCost,
      qualityScore,
      simulationRuns: swarmResult.iterations
    });

    return {
      schedule: swarmResult.bestSolution,
      expectedWaitTime: queueMetrics.averageWaitTime,
      expectedUtilization: queueMetrics.utilization,
      totalCost: swarmResult.bestSolution.totalCost,
      qualityScore,
      simulationRuns: swarmResult.iterations
    };
  }

  /**
   * Triage patient using OpenRouter AI
   */
  async triagePatient(patient: Pick<Patient, 'chiefComplaint'>): Promise<TriageDecision> {
    if (!this.openai) {
      // Fallback to rule-based triage
      return this.ruleBasedTriage(patient.chiefComplaint);
    }

    try {
      const response = await this.openai.chat.completions.create({
        model: this.config.openRouterModel || 'anthropic/claude-3.5-sonnet',
        messages: [
          {
            role: 'system',
            content: 'You are an emergency department triage nurse. Assess patient acuity (1=critical, 5=non-urgent) and recommend treatment path.'
          },
          {
            role: 'user',
            content: `Chief complaint: ${patient.chiefComplaint}\n\nProvide triage assessment in JSON format: {"acuity": 1-5, "path": "immediate|urgent|standard|fast_track", "reasoning": "explanation"}`
          }
        ],
        response_format: { type: 'json_object' }
      });

      const result = JSON.parse(response.choices[0].message.content || '{}');

      return {
        patientId: 'unknown',
        assignedAcuity: result.acuity,
        recommendedPath: result.path,
        reasoning: result.reasoning,
        confidence: 0.9
      };
    } catch (error) {
      console.error('OpenRouter triage failed, falling back to rules:', error);
      return this.ruleBasedTriage(patient.chiefComplaint);
    }
  }

  /**
   * Rule-based triage fallback
   */
  private ruleBasedTriage(complaint: string): TriageDecision {
    const lower = complaint.toLowerCase();

    const criticalKeywords = ['chest pain', 'stroke', 'severe bleeding', 'unconscious'];
    const urgentKeywords = ['fracture', 'severe pain', 'difficulty breathing', 'high fever'];
    const fastTrackKeywords = ['minor cut', 'cold', 'rash', 'sprain'];

    let acuity: 1 | 2 | 3 | 4 | 5 = 3;
    let path: 'immediate' | 'urgent' | 'standard' | 'fast_track' = 'standard';
    let reasoning = 'Standard triage assessment';

    if (criticalKeywords.some(k => lower.includes(k))) {
      acuity = 1;
      path = 'immediate';
      reasoning = 'Life-threatening condition detected';
    } else if (urgentKeywords.some(k => lower.includes(k))) {
      acuity = 2;
      path = 'urgent';
      reasoning = 'Urgent care required';
    } else if (fastTrackKeywords.some(k => lower.includes(k))) {
      acuity = 4;
      path = 'fast_track';
      reasoning = 'Minor condition suitable for fast track';
    }

    return {
      patientId: 'unknown',
      assignedAcuity: acuity,
      recommendedPath: path,
      reasoning,
      confidence: 0.7
    };
  }

  /**
   * Simulate queue performance
   */
  private async simulateQueue(forecasts: any[], shifts: any[]): Promise<any> {
    // Simplified simulation
    const avgArrivals = forecasts.reduce((sum, f) => sum + f.predictedArrivals, 0) / forecasts.length;
    const avgServiceTime = 30; // minutes

    return {
      timestamp: new Date(),
      queueLength: Math.floor(avgArrivals * 0.3),
      averageWaitTime: avgArrivals * avgServiceTime / (shifts.length * 60) * 60,
      maxWaitTime: avgArrivals * avgServiceTime / (shifts.length * 60) * 120,
      throughput: avgArrivals,
      utilization: Math.min(0.95, avgArrivals / (shifts.length * 2)),
      abandonmentRate: Math.max(0, (avgArrivals - shifts.length * 2) / avgArrivals) * 0.1
    };
  }

  /**
   * Calculate overall quality score
   */
  private calculateQualityScore(solution: any, metrics: any): number {
    const coverageWeight = 0.3;
    const fairnessWeight = 0.2;
    const waitTimeWeight = 0.3;
    const utilizationWeight = 0.2;

    const coverageScore = solution.coverageScore;
    const fairnessScore = solution.fairnessScore;
    const waitTimeScore = Math.max(0, 1 - metrics.averageWaitTime / 60);
    const utilizationScore = 1 - Math.abs(metrics.utilization - 0.75) / 0.75;

    return (
      coverageScore * coverageWeight +
      fairnessScore * fairnessWeight +
      waitTimeScore * waitTimeWeight +
      utilizationScore * utilizationWeight
    );
  }

  /**
   * Store optimization results
   */
  private async storeResults(result: OptimizationResult): Promise<void> {
    await this.memory.store('latest_optimization', {
      timestamp: new Date().toISOString(),
      result
    });
  }

  /**
   * Get historical performance
   */
  async getPerformanceHistory(): Promise<any> {
    return await this.memory.retrieve('optimization_history') || [];
  }

  /**
   * Update with actual outcomes for learning
   */
  async updateWithActuals(
    timestamp: Date,
    actualArrivals: number,
    actualWaitTime: number
  ): Promise<void> {
    await this.forecaster.updateWithActuals(timestamp, actualArrivals);

    // Store actual metrics for learning
    await this.memory.store(`actual:${timestamp.toISOString()}`, {
      timestamp,
      arrivals: actualArrivals,
      waitTime: actualWaitTime
    });
  }
}

// Export all types
export * from './types.js';
export { ArrivalForecaster } from './arrival-forecaster.js';
export { Scheduler } from './scheduler.js';
export { QueueOptimizer } from './queue-optimizer.js';
export { SwarmCoordinator } from './swarm.js';
