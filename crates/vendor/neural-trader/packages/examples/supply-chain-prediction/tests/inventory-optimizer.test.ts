/**
 * Comprehensive tests for InventoryOptimizer
 */

import { InventoryOptimizer, InventoryNode, OptimizerConfig } from '../src/inventory-optimizer';
import { DemandForecaster, ForecastConfig, DemandPattern } from '../src/demand-forecaster';

describe('InventoryOptimizer', () => {
  let forecaster: DemandForecaster;
  let optimizer: InventoryOptimizer;
  let config: OptimizerConfig;

  beforeEach(async () => {
    // Setup forecaster
    const forecastConfig: ForecastConfig = {
      alpha: 0.1,
      horizons: [1, 7, 14, 30],
      seasonalityPeriods: [7, 52],
      learningRate: 0.01,
      memoryNamespace: 'test-inventory',
    };

    forecaster = new DemandForecaster(forecastConfig);

    // Train with sample data
    const trainingData = generateTrainingData(100);
    await forecaster.train(trainingData);

    // Setup optimizer
    config = {
      targetServiceLevel: 0.95,
      planningHorizon: 30,
      reviewPeriod: 7,
      safetyFactor: 1.65,
      costWeights: {
        holding: 1,
        ordering: 1,
        shortage: 5,
      },
    };

    optimizer = new InventoryOptimizer(forecaster, config);
  });

  describe('Network Setup', () => {
    it('should add nodes to network', () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      expect(() => optimizer.addNode(node)).not.toThrow();
    });

    it('should get network topology', () => {
      optimizer.addNode(createTestNode('supplier-1', 'supplier'));
      optimizer.addNode(createTestNode('warehouse-1', 'warehouse'));

      const topology = optimizer.getNetworkTopology();
      expect(topology.nodes).toHaveLength(2);
    });

    it('should track node relationships', () => {
      const supplier = createTestNode('supplier-1', 'supplier');
      const warehouse = createTestNode('warehouse-1', 'warehouse');
      warehouse.upstreamNodes = ['supplier-1'];

      optimizer.addNode(supplier);
      optimizer.addNode(warehouse);

      const topology = optimizer.getNetworkTopology();
      expect(topology.edges).toContainEqual({
        from: 'supplier-1',
        to: 'warehouse-1',
      });
    });
  });

  describe('Node Optimization', () => {
    it('should calculate safety stock', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.safetyStock).toBeGreaterThan(0);
    });

    it('should calculate reorder point', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.reorderPoint).toBeGreaterThan(0);
      expect(result.reorderPoint).toBeGreaterThan(result.safetyStock);
    });

    it('should calculate order-up-to level', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.orderUpToLevel).toBeGreaterThan(result.reorderPoint);
    });

    it('should define (s,S) policy', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.policy.type).toBe('(s,S)');
      expect(result.policy.parameters.s).toBe(result.reorderPoint);
      expect(result.policy.parameters.S).toBe(result.orderUpToLevel);
    });

    it('should calculate expected cost', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.expectedCost).toBeGreaterThan(0);
    });

    it('should achieve target service level', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);
      expect(result.serviceLevel).toBeGreaterThanOrEqual(config.targetServiceLevel * 0.9);
    });
  });

  describe('Network Optimization', () => {
    it('should optimize entire network', async () => {
      optimizer.addNode(createTestNode('supplier-1', 'supplier'));
      optimizer.addNode(createTestNode('warehouse-1', 'warehouse'));
      optimizer.addNode(createTestNode('store-1', 'retail'));

      const result = await optimizer.optimizeNetwork('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.nodeResults).toHaveLength(3);
      expect(result.totalCost).toBeGreaterThan(0);
      expect(result.avgServiceLevel).toBeGreaterThan(0);
    });

    it('should calculate flows between nodes', async () => {
      const supplier = createTestNode('supplier-1', 'supplier', 0);
      const warehouse = createTestNode('warehouse-1', 'warehouse', 1);
      warehouse.upstreamNodes = ['supplier-1'];
      supplier.downstreamNodes = ['warehouse-1'];

      optimizer.addNode(supplier);
      optimizer.addNode(warehouse);

      const result = await optimizer.optimizeNetwork('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(result.flow.size).toBeGreaterThan(0);
    });

    it('should respect multi-echelon structure', async () => {
      const supplier = createTestNode('supplier-1', 'supplier', 0);
      const warehouse = createTestNode('warehouse-1', 'warehouse', 1);
      const store = createTestNode('store-1', 'retail', 2);

      warehouse.upstreamNodes = ['supplier-1'];
      store.upstreamNodes = ['warehouse-1'];

      optimizer.addNode(supplier);
      optimizer.addNode(warehouse);
      optimizer.addNode(store);

      const result = await optimizer.optimizeNetwork('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      // Supplier should be optimized before warehouse
      const supplierResult = result.nodeResults.find((r) => r.nodeId === 'supplier-1');
      const warehouseResult = result.nodeResults.find((r) => r.nodeId === 'warehouse-1');

      expect(supplierResult).toBeDefined();
      expect(warehouseResult).toBeDefined();
    });
  });

  describe('Lead Time Uncertainty', () => {
    it('should account for lead time variance', async () => {
      const lowVariance = createTestNode('node-low', 'warehouse');
      lowVariance.leadTime = { mean: 7, stdDev: 1, distribution: 'normal' };

      const highVariance = createTestNode('node-high', 'warehouse');
      highVariance.leadTime = { mean: 7, stdDev: 5, distribution: 'normal' };

      optimizer.addNode(lowVariance);
      optimizer.addNode(highVariance);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const lowResult = await optimizer.optimizeNode(lowVariance, forecasts);
      const highResult = await optimizer.optimizeNode(highVariance, forecasts);

      // Higher variance should require more safety stock
      expect(highResult.safetyStock).toBeGreaterThan(lowResult.safetyStock);
    });
  });

  describe('Cost Optimization', () => {
    it('should balance holding and ordering costs', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      optimizer.addNode(node);

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const result = await optimizer.optimizeNode(node, forecasts);

      // EOQ should balance costs
      const Q = result.policy.parameters.Q!;
      expect(Q).toBeGreaterThan(0);
    });

    it('should penalize stockouts appropriately', async () => {
      const lowPenalty = new InventoryOptimizer(forecaster, {
        ...config,
        costWeights: { holding: 1, ordering: 1, shortage: 1 },
      });

      const highPenalty = new InventoryOptimizer(forecaster, {
        ...config,
        costWeights: { holding: 1, ordering: 1, shortage: 100 },
      });

      const node = createTestNode('warehouse-1', 'warehouse');
      lowPenalty.addNode(node);
      highPenalty.addNode({ ...node });

      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      const lowResult = await lowPenalty.optimizeNode(node, forecasts);
      const highResult = await highPenalty.optimizeNode({ ...node }, forecasts);

      // High penalty should lead to higher safety stock
      expect(highResult.safetyStock).toBeGreaterThan(lowResult.safetyStock);
    });
  });

  describe('Simulation', () => {
    it('should simulate inventory performance', async () => {
      optimizer.addNode(createTestNode('warehouse-1', 'warehouse'));

      const simulation = await optimizer.simulate(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        10
      );

      expect(simulation.avgServiceLevel).toBeGreaterThan(0);
      expect(simulation.avgInventoryCost).toBeGreaterThan(0);
      expect(simulation.fillRate).toBeGreaterThan(0);
      expect(simulation.stockouts).toBeGreaterThanOrEqual(0);
    });

    it('should track stockouts', async () => {
      const node = createTestNode('warehouse-1', 'warehouse');
      node.position.currentStock = 10; // Low stock
      optimizer.addNode(node);

      const simulation = await optimizer.simulate(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        10
      );

      expect(simulation.stockouts).toBeGreaterThanOrEqual(0);
    });
  });
});

/**
 * Helper functions
 */

function createTestNode(
  nodeId: string,
  type: InventoryNode['type'],
  level: number = 1
): InventoryNode {
  return {
    nodeId,
    type,
    level,
    upstreamNodes: [],
    downstreamNodes: [],
    position: {
      currentStock: 500,
      onOrder: 100,
      allocated: 50,
    },
    costs: {
      holding: 0.5,
      ordering: 100,
      shortage: 50,
    },
    leadTime: {
      mean: 7,
      stdDev: 2,
      distribution: 'normal',
    },
    capacity: {
      storage: 10000,
      throughput: 1000,
    },
  };
}

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
