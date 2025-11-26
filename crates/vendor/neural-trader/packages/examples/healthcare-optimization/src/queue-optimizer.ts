/**
 * Queue Optimizer
 *
 * Queue theory-based optimization for patient flow and resource allocation.
 * Implements M/M/c and M/G/c models with dynamic resource adjustment.
 */

import type {
  Patient,
  QueueState,
  QueueMetrics,
  ResourcePool,
  OptimizationObjective
} from './types.js';

export interface QueueOptimizerConfig {
  targetUtilization: number; // 0-1
  maxWaitTime: number; // minutes
  abandonmentThreshold: number; // minutes
  reallocateInterval: number; // minutes
}

export class QueueOptimizer {
  private config: QueueOptimizerConfig;
  private state: QueueState;
  private metricsHistory: QueueMetrics[];

  constructor(config: QueueOptimizerConfig) {
    this.config = config;
    this.metricsHistory = [];

    // Initialize empty state
    this.state = {
      waitingPatients: [],
      inServicePatients: [],
      resources: [],
      metrics: this.emptyMetrics()
    };
  }

  /**
   * Add patient to queue with priority
   */
  addPatient(patient: Patient): void {
    // Insert based on acuity (1=highest priority)
    const insertIndex = this.state.waitingPatients.findIndex(
      p => p.acuity > patient.acuity
    );

    if (insertIndex === -1) {
      this.state.waitingPatients.push(patient);
    } else {
      this.state.waitingPatients.splice(insertIndex, 0, patient);
    }

    this.updateMetrics();
  }

  /**
   * Process next patient from queue
   */
  processNextPatient(resourceId: string): Patient | null {
    if (this.state.waitingPatients.length === 0) {
      return null;
    }

    const resource = this.state.resources.find(r => r.id === resourceId);
    if (!resource || resource.available === 0) {
      return null;
    }

    // Get highest priority patient
    const patient = this.state.waitingPatients.shift()!;

    // Move to in-service
    this.state.inServicePatients.push(patient);

    // Update resource availability
    resource.available--;

    this.updateMetrics();
    return patient;
  }

  /**
   * Complete patient service
   */
  completeService(patientId: string, resourceId: string): void {
    const index = this.state.inServicePatients.findIndex(p => p.id === patientId);
    if (index === -1) {
      return;
    }

    // Remove from in-service
    this.state.inServicePatients.splice(index, 1);

    // Free resource
    const resource = this.state.resources.find(r => r.id === resourceId);
    if (resource) {
      resource.available = Math.min(resource.available + 1, resource.capacity);
    }

    this.updateMetrics();
  }

  /**
   * Add resource pool
   */
  addResourcePool(pool: ResourcePool): void {
    this.state.resources.push(pool);
  }

  /**
   * Optimize resource allocation based on queue state
   */
  optimizeResources(objective: OptimizationObjective): {
    recommendations: Array<{
      resourceId: string;
      action: 'increase' | 'decrease' | 'maintain';
      amount: number;
      reason: string;
    }>;
    expectedImpact: {
      waitTimeReduction: number;
      utilizationChange: number;
      costChange: number;
    };
  } {
    const recommendations: Array<{
      resourceId: string;
      action: 'increase' | 'decrease' | 'maintain';
      amount: number;
      reason: string;
    }> = [];

    for (const resource of this.state.resources) {
      const utilization = 1 - (resource.available / resource.capacity);
      const queueLength = this.state.waitingPatients.length;

      // Calculate optimal servers using Erlang C
      const optimalCapacity = this.calculateOptimalCapacity(
        resource,
        queueLength,
        objective
      );

      if (optimalCapacity > resource.capacity) {
        recommendations.push({
          resourceId: resource.id,
          action: 'increase',
          amount: optimalCapacity - resource.capacity,
          reason: `High utilization (${(utilization * 100).toFixed(1)}%) and queue length (${queueLength})`
        });
      } else if (optimalCapacity < resource.capacity && utilization < 0.5) {
        recommendations.push({
          resourceId: resource.id,
          action: 'decrease',
          amount: resource.capacity - optimalCapacity,
          reason: `Low utilization (${(utilization * 100).toFixed(1)}%)`
        });
      } else {
        recommendations.push({
          resourceId: resource.id,
          action: 'maintain',
          amount: 0,
          reason: `Optimal utilization (${(utilization * 100).toFixed(1)}%)`
        });
      }
    }

    // Estimate impact
    const expectedImpact = this.estimateImpact(recommendations);

    return { recommendations, expectedImpact };
  }

  /**
   * Calculate optimal resource capacity using M/M/c model
   */
  private calculateOptimalCapacity(
    resource: ResourcePool,
    queueLength: number,
    objective: OptimizationObjective
  ): number {
    const lambda = this.estimateArrivalRate(); // arrivals per hour
    const mu = 60 / this.estimateServiceTime(); // service rate per hour
    const rho = lambda / mu; // traffic intensity

    // Target wait time based on objective
    const targetWait = this.config.maxWaitTime * (1 - objective.minimizeWaitTime);

    // Use Erlang C formula to find minimum servers
    let c = Math.ceil(rho); // start with minimum
    let waitTime = Infinity;

    while (waitTime > targetWait && c < resource.capacity * 2) {
      c++;
      waitTime = this.erlangCWaitTime(lambda, mu, c);
    }

    // Adjust for utilization target
    const utilizationCapacity = Math.ceil(rho / this.config.targetUtilization);

    // Return max of both constraints
    return Math.max(c, utilizationCapacity);
  }

  /**
   * Erlang C formula for wait time calculation
   */
  private erlangCWaitTime(lambda: number, mu: number, c: number): number {
    const rho = lambda / (c * mu);
    if (rho >= 1) {
      return Infinity; // unstable system
    }

    // Erlang C probability of waiting
    const erlangC = this.erlangCProbability(lambda / mu, c);

    // Expected wait time
    const waitTime = (erlangC / (c * mu - lambda)) * 60; // in minutes

    return waitTime;
  }

  /**
   * Erlang C probability calculation
   */
  private erlangCProbability(a: number, c: number): number {
    // Calculate Erlang C using iterative formula
    let sum = 0;
    for (let k = 0; k < c; k++) {
      sum += Math.pow(a, k) / this.factorial(k);
    }

    const erlangB = Math.pow(a, c) / this.factorial(c);
    const denominator = sum + erlangB * c / (c - a);

    return erlangB / denominator;
  }

  /**
   * Factorial helper
   */
  private factorial(n: number): number {
    if (n <= 1) return 1;
    return n * this.factorial(n - 1);
  }

  /**
   * Estimate arrival rate from recent data
   */
  private estimateArrivalRate(): number {
    if (this.metricsHistory.length < 2) {
      return 10; // default
    }

    // Calculate average throughput
    const recentMetrics = this.metricsHistory.slice(-10);
    const avgThroughput = recentMetrics.reduce((sum, m) => sum + m.throughput, 0) / recentMetrics.length;

    return avgThroughput;
  }

  /**
   * Estimate service time from patients
   */
  private estimateServiceTime(): number {
    const allPatients = [...this.state.waitingPatients, ...this.state.inServicePatients];

    if (allPatients.length === 0) {
      return 30; // default 30 minutes
    }

    const avgServiceTime = allPatients.reduce((sum, p) => sum + p.estimatedServiceTime, 0) / allPatients.length;
    return avgServiceTime;
  }

  /**
   * Estimate impact of resource changes
   */
  private estimateImpact(recommendations: Array<any>): {
    waitTimeReduction: number;
    utilizationChange: number;
    costChange: number;
  } {
    let totalCapacityChange = 0;
    let totalCostChange = 0;

    for (const rec of recommendations) {
      if (rec.action === 'increase') {
        totalCapacityChange += rec.amount;
        totalCostChange += rec.amount * 50; // $50 per hour per resource
      } else if (rec.action === 'decrease') {
        totalCapacityChange -= rec.amount;
        totalCostChange -= rec.amount * 50;
      }
    }

    // Estimate wait time reduction (simplified model)
    const currentWaitTime = this.state.metrics.averageWaitTime;
    const capacityIncreasePct = totalCapacityChange / this.getTotalCapacity();
    const waitTimeReduction = currentWaitTime * capacityIncreasePct * 0.5;

    // Estimate utilization change
    const utilizationChange = -capacityIncreasePct * 0.3;

    return {
      waitTimeReduction,
      utilizationChange,
      costChange: totalCostChange
    };
  }

  /**
   * Get total resource capacity
   */
  private getTotalCapacity(): number {
    return this.state.resources.reduce((sum, r) => sum + r.capacity, 0);
  }

  /**
   * Update queue metrics
   */
  private updateMetrics(): void {
    const now = new Date();
    const queueLength = this.state.waitingPatients.length;
    const inService = this.state.inServicePatients.length;

    // Calculate wait times
    const waitTimes = this.state.waitingPatients.map(p => {
      return (now.getTime() - p.arrivalTime.getTime()) / (1000 * 60);
    });

    const averageWaitTime = waitTimes.length > 0
      ? waitTimes.reduce((a, b) => a + b, 0) / waitTimes.length
      : 0;

    const maxWaitTime = waitTimes.length > 0
      ? Math.max(...waitTimes)
      : 0;

    // Calculate utilization
    const totalCapacity = this.getTotalCapacity();
    const totalAvailable = this.state.resources.reduce((sum, r) => sum + r.available, 0);
    const utilization = totalCapacity > 0 ? 1 - (totalAvailable / totalCapacity) : 0;

    // Estimate throughput from recent history
    const throughput = this.estimateArrivalRate();

    // Calculate abandonment rate (patients waiting > threshold)
    const abandonments = waitTimes.filter(w => w > this.config.abandonmentThreshold).length;
    const abandonmentRate = waitTimes.length > 0 ? abandonments / waitTimes.length : 0;

    this.state.metrics = {
      timestamp: now,
      queueLength,
      averageWaitTime,
      maxWaitTime,
      throughput,
      utilization,
      abandonmentRate
    };

    this.metricsHistory.push(this.state.metrics);

    // Keep only recent history
    if (this.metricsHistory.length > 1000) {
      this.metricsHistory.shift();
    }
  }

  /**
   * Get current queue state
   */
  getState(): QueueState {
    return { ...this.state };
  }

  /**
   * Get metrics history
   */
  getMetricsHistory(): QueueMetrics[] {
    return [...this.metricsHistory];
  }

  /**
   * Create empty metrics object
   */
  private emptyMetrics(): QueueMetrics {
    return {
      timestamp: new Date(),
      queueLength: 0,
      averageWaitTime: 0,
      maxWaitTime: 0,
      throughput: 0,
      utilization: 0,
      abandonmentRate: 0
    };
  }
}
