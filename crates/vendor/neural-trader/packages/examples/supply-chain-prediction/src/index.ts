/**
 * @neural-trader/example-supply-chain-prediction
 *
 * Self-learning demand forecasting and swarm-based inventory optimization
 * with uncertainty quantification for retail, manufacturing, and e-commerce.
 *
 * Features:
 * - Multi-echelon inventory optimization
 * - Demand sensing with conformal prediction
 * - Lead time uncertainty modeling
 * - Safety stock optimization
 * - Swarm exploration of (s,S) policies
 * - Self-learning service level targets
 * - AgentDB memory for seasonal patterns
 *
 * @module @neural-trader/example-supply-chain-prediction
 */

// Core exports
export {
  DemandForecaster,
  DemandPattern,
  DemandForecast,
  ForecastConfig,
} from './demand-forecaster';

export {
  InventoryOptimizer,
  InventoryNode,
  OptimizationResult,
  NetworkOptimization,
  OptimizerConfig,
} from './inventory-optimizer';

export {
  SwarmPolicyOptimizer,
  PolicyParticle,
  SwarmConfig,
  SwarmResult,
} from './swarm-policy';

// Re-exports from dependencies
export type { ConformalConfig } from '@neural-trader/predictor';

/**
 * Complete supply chain optimization system
 */
import { DemandForecaster, ForecastConfig } from './demand-forecaster';
import { InventoryOptimizer, OptimizerConfig, InventoryNode } from './inventory-optimizer';
import { SwarmPolicyOptimizer, SwarmConfig } from './swarm-policy';

export interface SupplyChainConfig {
  forecast: ForecastConfig;
  optimizer: OptimizerConfig;
  swarm: SwarmConfig;
  openRouterApiKey?: string;
}

export class SupplyChainSystem {
  private forecaster: DemandForecaster;
  private optimizer: InventoryOptimizer;
  private swarmOptimizer: SwarmPolicyOptimizer;
  private config: SupplyChainConfig;

  constructor(config: SupplyChainConfig) {
    this.config = config;

    // Initialize components
    this.forecaster = new DemandForecaster(config.forecast);
    this.optimizer = new InventoryOptimizer(this.forecaster, config.optimizer);
    this.swarmOptimizer = new SwarmPolicyOptimizer(
      this.forecaster,
      this.optimizer,
      config.swarm
    );
  }

  /**
   * Initialize supply chain network
   */
  addInventoryNode(node: InventoryNode): void {
    this.optimizer.addNode(node);
  }

  /**
   * Train system on historical data
   */
  async train(historicalData: any[]): Promise<void> {
    await this.forecaster.train(historicalData);
  }

  /**
   * Optimize supply chain with swarm intelligence
   */
  async optimize(productId: string, currentFeatures: any) {
    // Run swarm optimization
    const swarmResult = await this.swarmOptimizer.optimize(productId, currentFeatures);

    // Apply best policy
    const bestPolicy = swarmResult.bestPolicy;

    // Update optimizer config with best parameters
    this.optimizer = new InventoryOptimizer(this.forecaster, {
      ...this.config.optimizer,
      safetyFactor: bestPolicy.safetyFactor,
    });

    // Optimize network with best policy
    const networkOptimization = await this.optimizer.optimizeNetwork(
      productId,
      currentFeatures
    );

    return {
      swarmResult,
      networkOptimization,
      bestPolicy,
      performance: swarmResult.bestFitness,
    };
  }

  /**
   * Get real-time recommendations
   */
  async getRecommendations(productId: string, currentFeatures: any) {
    // Get demand forecast
    const forecast = await this.forecaster.forecast(productId, currentFeatures, 1);

    // Get current optimization
    const optimization = await this.optimizer.optimizeNetwork(productId, currentFeatures);

    // Generate recommendations
    const recommendations = optimization.nodeResults.map((result) => ({
      nodeId: result.nodeId,
      action: result.reorderPoint > 0 ? 'ORDER' : 'HOLD',
      quantity: Math.max(0, result.orderUpToLevel - result.reorderPoint),
      urgency: result.serviceLevel < this.config.optimizer.targetServiceLevel ? 'HIGH' : 'NORMAL',
      reason: this.generateReason(result, forecast),
    }));

    return {
      forecast,
      optimization,
      recommendations,
    };
  }

  /**
   * Update system with new observation
   */
  async update(observation: any): Promise<void> {
    await this.forecaster.update(observation);
  }

  /**
   * Get system metrics
   */
  getMetrics() {
    return {
      forecastCalibration: this.forecaster.getCalibration(),
      networkTopology: this.optimizer.getNetworkTopology(),
      paretoFront: this.swarmOptimizer.getParetoFront(),
    };
  }

  /**
   * Generate recommendation reason
   */
  private generateReason(result: any, forecast: any): string {
    if (result.serviceLevel < this.config.optimizer.targetServiceLevel) {
      return `Service level (${(result.serviceLevel * 100).toFixed(1)}%) below target. Increase stock.`;
    }

    if (forecast.uncertainty > forecast.pointForecast * 0.3) {
      return `High demand uncertainty. Consider increasing safety stock.`;
    }

    if (result.expectedCost > 1000) {
      return `High inventory cost. Consider reducing order quantity.`;
    }

    return 'Maintain current policy.';
  }
}

/**
 * Factory function for easy setup
 */
export function createSupplyChainSystem(
  config: Partial<SupplyChainConfig> = {}
): SupplyChainSystem {
  const defaultConfig: SupplyChainConfig = {
    forecast: {
      alpha: 0.1,
      horizons: [1, 7, 14, 30],
      seasonalityPeriods: [7, 52],
      learningRate: 0.01,
      memoryNamespace: 'supply-chain',
    },
    optimizer: {
      targetServiceLevel: 0.95,
      planningHorizon: 30,
      reviewPeriod: 7,
      safetyFactor: 1.65,
      costWeights: {
        holding: 1,
        ordering: 1,
        shortage: 5,
      },
    },
    swarm: {
      particles: 20,
      iterations: 50,
      inertia: 0.7,
      cognitive: 1.5,
      social: 1.5,
      bounds: {
        reorderPoint: [0, 1000],
        orderUpToLevel: [100, 2000],
        safetyFactor: [1.0, 3.0],
      },
      objectives: {
        costWeight: 0.6,
        serviceLevelWeight: 0.4,
      },
    },
  };

  return new SupplyChainSystem({
    ...defaultConfig,
    ...config,
    forecast: { ...defaultConfig.forecast, ...config.forecast },
    optimizer: { ...defaultConfig.optimizer, ...config.optimizer },
    swarm: { ...defaultConfig.swarm, ...config.swarm },
  });
}

/**
 * Example usage for retail
 */
export async function retailExample() {
  const system = createSupplyChainSystem();

  // Add retail nodes
  system.addInventoryNode({
    nodeId: 'warehouse-1',
    type: 'warehouse',
    level: 1,
    upstreamNodes: ['supplier-1'],
    downstreamNodes: ['store-1', 'store-2'],
    position: { currentStock: 500, onOrder: 0, allocated: 100 },
    costs: { holding: 0.5, ordering: 100, shortage: 50 },
    leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
    capacity: { storage: 10000, throughput: 1000 },
  });

  system.addInventoryNode({
    nodeId: 'store-1',
    type: 'retail',
    level: 2,
    upstreamNodes: ['warehouse-1'],
    downstreamNodes: [],
    position: { currentStock: 100, onOrder: 0, allocated: 0 },
    costs: { holding: 1, ordering: 50, shortage: 100 },
    leadTime: { mean: 3, stdDev: 1, distribution: 'normal' },
    capacity: { storage: 500, throughput: 100 },
  });

  // Optimize
  const result = await system.optimize('product-123', {
    dayOfWeek: 1,
    weekOfYear: 20,
    monthOfYear: 5,
    isHoliday: false,
    promotions: 0,
    priceIndex: 1.0,
  });

  console.log('Optimization Result:', result);
  return result;
}

/**
 * Example usage for manufacturing
 */
export async function manufacturingExample() {
  const system = createSupplyChainSystem({
    optimizer: {
      targetServiceLevel: 0.99, // Higher service level for manufacturing
      planningHorizon: 60,
      reviewPeriod: 14,
      safetyFactor: 2.0,
      costWeights: {
        holding: 2,
        ordering: 5,
        shortage: 100, // Very high shortage cost
      },
    },
  });

  // Add manufacturing nodes
  system.addInventoryNode({
    nodeId: 'raw-materials',
    type: 'supplier',
    level: 0,
    upstreamNodes: [],
    downstreamNodes: ['production-line-1'],
    position: { currentStock: 1000, onOrder: 500, allocated: 200 },
    costs: { holding: 0.2, ordering: 500, shortage: 1000 },
    leadTime: { mean: 14, stdDev: 5, distribution: 'lognormal' },
    capacity: { storage: 50000, throughput: 5000 },
  });

  return system;
}

/**
 * Example usage for e-commerce
 */
export async function ecommerceExample() {
  const system = createSupplyChainSystem({
    forecast: {
      alpha: 0.05, // Tighter prediction intervals
      horizons: [1, 3, 7, 14, 30],
      seasonalityPeriods: [7, 365], // Weekly and yearly
      learningRate: 0.02,
      memoryNamespace: 'ecommerce',
    },
    swarm: {
      particles: 30,
      iterations: 100,
      inertia: 0.6,
      cognitive: 2.0,
      social: 2.0,
      bounds: {
        reorderPoint: [0, 5000],
        orderUpToLevel: [500, 10000],
        safetyFactor: [1.2, 2.5],
      },
      objectives: {
        costWeight: 0.5,
        serviceLevelWeight: 0.5,
      },
    },
  });

  // Add e-commerce nodes
  system.addInventoryNode({
    nodeId: 'fulfillment-center-1',
    type: 'distribution',
    level: 1,
    upstreamNodes: ['supplier-1', 'supplier-2'],
    downstreamNodes: ['customer-zone-1', 'customer-zone-2'],
    position: { currentStock: 2000, onOrder: 1000, allocated: 500 },
    costs: { holding: 0.8, ordering: 200, shortage: 75 },
    leadTime: { mean: 5, stdDev: 1.5, distribution: 'gamma' },
    capacity: { storage: 20000, throughput: 2000 },
  });

  return system;
}
