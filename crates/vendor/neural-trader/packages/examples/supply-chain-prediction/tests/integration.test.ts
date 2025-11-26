/**
 * Integration tests for complete supply chain system
 */

import {
  createSupplyChainSystem,
  SupplyChainSystem,
  retailExample,
  manufacturingExample,
  ecommerceExample,
} from '../src/index';
import { DemandPattern } from '../src/demand-forecaster';

describe('Supply Chain System Integration', () => {
  let system: SupplyChainSystem;

  beforeEach(() => {
    system = createSupplyChainSystem();
  });

  describe('System Initialization', () => {
    it('should create system with default config', () => {
      expect(system).toBeDefined();
    });

    it('should create system with custom config', () => {
      const customSystem = createSupplyChainSystem({
        optimizer: {
          targetServiceLevel: 0.99,
          planningHorizon: 60,
          reviewPeriod: 14,
          safetyFactor: 2.0,
          costWeights: {
            holding: 2,
            ordering: 5,
            shortage: 100,
          },
        },
      });

      expect(customSystem).toBeDefined();
    });
  });

  describe('End-to-End Workflow', () => {
    it('should complete full supply chain optimization', async () => {
      // Setup network
      system.addInventoryNode({
        nodeId: 'warehouse-1',
        type: 'warehouse',
        level: 1,
        upstreamNodes: ['supplier-1'],
        downstreamNodes: ['store-1'],
        position: { currentStock: 500, onOrder: 100, allocated: 50 },
        costs: { holding: 0.5, ordering: 100, shortage: 50 },
        leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
        capacity: { storage: 10000, throughput: 1000 },
      });

      // Train system
      const trainingData = generateTrainingData(100);
      await system.train(trainingData);

      // Optimize
      const result = await system.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.swarmResult).toBeDefined();
      expect(result.networkOptimization).toBeDefined();
      expect(result.bestPolicy).toBeDefined();
      expect(result.performance).toBeDefined();
    });

    it('should provide real-time recommendations', async () => {
      system.addInventoryNode({
        nodeId: 'warehouse-1',
        type: 'warehouse',
        level: 1,
        upstreamNodes: [],
        downstreamNodes: [],
        position: { currentStock: 500, onOrder: 100, allocated: 50 },
        costs: { holding: 0.5, ordering: 100, shortage: 50 },
        leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
        capacity: { storage: 10000, throughput: 1000 },
      });

      const trainingData = generateTrainingData(50);
      await system.train(trainingData);

      const recommendations = await system.getRecommendations('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(recommendations.forecast).toBeDefined();
      expect(recommendations.optimization).toBeDefined();
      expect(recommendations.recommendations).toBeDefined();
      expect(recommendations.recommendations.length).toBeGreaterThan(0);
    });

    it('should update with new observations', async () => {
      const trainingData = generateTrainingData(50);
      await system.train(trainingData);

      const observation: DemandPattern = {
        productId: 'product-1',
        timestamp: Date.now(),
        demand: 150,
        features: {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
      };

      await expect(system.update(observation)).resolves.not.toThrow();
    });

    it('should get system metrics', async () => {
      system.addInventoryNode({
        nodeId: 'warehouse-1',
        type: 'warehouse',
        level: 1,
        upstreamNodes: [],
        downstreamNodes: [],
        position: { currentStock: 500, onOrder: 100, allocated: 50 },
        costs: { holding: 0.5, ordering: 100, shortage: 50 },
        leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
        capacity: { storage: 10000, throughput: 1000 },
      });

      const trainingData = generateTrainingData(50);
      await system.train(trainingData);

      await system.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const metrics = system.getMetrics();

      expect(metrics.forecastCalibration).toBeDefined();
      expect(metrics.networkTopology).toBeDefined();
      expect(metrics.paretoFront).toBeDefined();
    });
  });

  describe('Use Case Examples', () => {
    it('should run retail example', async () => {
      const result = await retailExample();
      expect(result).toBeDefined();
      expect(result.swarmResult).toBeDefined();
      expect(result.networkOptimization).toBeDefined();
    }, 30000); // Longer timeout for full optimization

    it('should run manufacturing example', async () => {
      const manufacturingSystem = await manufacturingExample();
      expect(manufacturingSystem).toBeDefined();
    });

    it('should run e-commerce example', async () => {
      const ecommerceSystem = await ecommerceExample();
      expect(ecommerceSystem).toBeDefined();
    });
  });

  describe('Multi-Echelon Scenarios', () => {
    it('should handle 3-level supply chain', async () => {
      // Level 0: Supplier
      system.addInventoryNode({
        nodeId: 'supplier-1',
        type: 'supplier',
        level: 0,
        upstreamNodes: [],
        downstreamNodes: ['warehouse-1'],
        position: { currentStock: 2000, onOrder: 0, allocated: 500 },
        costs: { holding: 0.2, ordering: 500, shortage: 200 },
        leadTime: { mean: 14, stdDev: 3, distribution: 'normal' },
        capacity: { storage: 50000, throughput: 5000 },
      });

      // Level 1: Warehouse
      system.addInventoryNode({
        nodeId: 'warehouse-1',
        type: 'warehouse',
        level: 1,
        upstreamNodes: ['supplier-1'],
        downstreamNodes: ['store-1', 'store-2'],
        position: { currentStock: 1000, onOrder: 200, allocated: 300 },
        costs: { holding: 0.5, ordering: 200, shortage: 100 },
        leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
        capacity: { storage: 20000, throughput: 2000 },
      });

      // Level 2: Stores
      system.addInventoryNode({
        nodeId: 'store-1',
        type: 'retail',
        level: 2,
        upstreamNodes: ['warehouse-1'],
        downstreamNodes: [],
        position: { currentStock: 100, onOrder: 50, allocated: 0 },
        costs: { holding: 1, ordering: 50, shortage: 75 },
        leadTime: { mean: 3, stdDev: 1, distribution: 'normal' },
        capacity: { storage: 500, throughput: 100 },
      });

      system.addInventoryNode({
        nodeId: 'store-2',
        type: 'retail',
        level: 2,
        upstreamNodes: ['warehouse-1'],
        downstreamNodes: [],
        position: { currentStock: 120, onOrder: 40, allocated: 0 },
        costs: { holding: 1, ordering: 50, shortage: 75 },
        leadTime: { mean: 3, stdDev: 1, distribution: 'normal' },
        capacity: { storage: 500, throughput: 100 },
      });

      const trainingData = generateTrainingData(50);
      await system.train(trainingData);

      const result = await system.optimize('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.networkOptimization.nodeResults).toHaveLength(4);
    }, 30000);
  });

  describe('Seasonal Demand Patterns', () => {
    it('should handle seasonal variations', async () => {
      system.addInventoryNode({
        nodeId: 'warehouse-1',
        type: 'warehouse',
        level: 1,
        upstreamNodes: [],
        downstreamNodes: [],
        position: { currentStock: 500, onOrder: 100, allocated: 50 },
        costs: { holding: 0.5, ordering: 100, shortage: 50 },
        leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
        capacity: { storage: 10000, throughput: 1000 },
      });

      // Generate data with strong seasonality
      const seasonalData = generateSeasonalData(200);
      await system.train(seasonalData);

      // Test in-season forecast
      const peakForecast = await system.getRecommendations('product-1', {
        dayOfWeek: 1,
        weekOfYear: 26, // Mid-year peak
        monthOfYear: 6,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      // Test off-season forecast
      const offPeakForecast = await system.getRecommendations('product-1', {
        dayOfWeek: 1,
        weekOfYear: 2, // Early year low
        monthOfYear: 1,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      // Peak season should have higher forecast
      expect(peakForecast.forecast.pointForecast).toBeGreaterThan(
        offPeakForecast.forecast.pointForecast
      );
    });
  });

  describe('Promotional Events', () => {
    it('should handle promotional spikes', async () => {
      system.addInventoryNode({
        nodeId: 'store-1',
        type: 'retail',
        level: 1,
        upstreamNodes: [],
        downstreamNodes: [],
        position: { currentStock: 200, onOrder: 0, allocated: 0 },
        costs: { holding: 1, ordering: 50, shortage: 100 },
        leadTime: { mean: 3, stdDev: 1, distribution: 'normal' },
        capacity: { storage: 1000, throughput: 200 },
      });

      const trainingData = generateTrainingData(50);
      await system.train(trainingData);

      // No promotion
      const normalForecast = await system.getRecommendations('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      // With promotion
      const promoForecast = await system.getRecommendations('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 1,
        priceIndex: 0.8,
      });

      // Both should succeed
      expect(normalForecast.forecast.pointForecast).toBeGreaterThan(0);
      expect(promoForecast.forecast.pointForecast).toBeGreaterThan(0);
    });
  });
});

/**
 * Helper functions
 */

function generateTrainingData(count: number): DemandPattern[] {
  const data: DemandPattern[] = [];
  const baseDate = new Date('2024-01-01');

  for (let i = 0; i < count; i++) {
    const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    data.push({
      productId: 'product-1',
      timestamp: date.getTime(),
      demand: 100 + Math.random() * 50,
      features: {
        dayOfWeek: date.getDay(),
        weekOfYear: Math.floor(i / 7),
        monthOfYear: date.getMonth(),
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      },
    });
  }

  return data;
}

function generateSeasonalData(count: number): DemandPattern[] {
  const data: DemandPattern[] = [];
  const baseDate = new Date('2024-01-01');

  for (let i = 0; i < count; i++) {
    const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    const weekOfYear = Math.floor(i / 7);

    // Strong seasonality: peak in summer, low in winter
    const seasonal = 1 + 0.5 * Math.sin(((weekOfYear - 13) / 52) * 2 * Math.PI);

    data.push({
      productId: 'product-1',
      timestamp: date.getTime(),
      demand: (100 + Math.random() * 30) * seasonal,
      features: {
        dayOfWeek: date.getDay(),
        weekOfYear,
        monthOfYear: date.getMonth(),
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      },
    });
  }

  return data;
}
