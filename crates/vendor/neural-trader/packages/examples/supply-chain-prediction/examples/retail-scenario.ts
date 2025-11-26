/**
 * Retail Supply Chain Example
 *
 * Demonstrates supply chain optimization for retail operations with:
 * - Multi-location inventory management
 * - Seasonal demand patterns
 * - Promotional planning
 * - Service level optimization
 */

import { createSupplyChainSystem } from '../src/index';

async function runRetailScenario() {
  console.log('=== Retail Supply Chain Optimization ===\n');

  // Create system optimized for retail
  const system = createSupplyChainSystem({
    forecast: {
      alpha: 0.1,
      horizons: [1, 7, 14, 30],
      seasonalityPeriods: [7, 52], // Weekly and yearly patterns
      learningRate: 0.02,
      memoryNamespace: 'retail-supply-chain',
    },
    optimizer: {
      targetServiceLevel: 0.95,
      planningHorizon: 30,
      reviewPeriod: 7,
      safetyFactor: 1.65,
      costWeights: {
        holding: 1,
        ordering: 1,
        shortage: 5, // High penalty for stockouts
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
  });

  // Setup retail network
  console.log('Setting up retail network...');

  // Regional Distribution Center
  system.addInventoryNode({
    nodeId: 'rdc-northeast',
    type: 'distribution',
    level: 1,
    upstreamNodes: ['supplier-main'],
    downstreamNodes: ['store-nyc', 'store-boston', 'store-philly'],
    position: { currentStock: 5000, onOrder: 2000, allocated: 1000 },
    costs: { holding: 0.3, ordering: 500, shortage: 100 },
    leadTime: { mean: 7, stdDev: 2, distribution: 'normal' },
    capacity: { storage: 50000, throughput: 5000 },
  });

  // Store 1: NYC (high volume)
  system.addInventoryNode({
    nodeId: 'store-nyc',
    type: 'retail',
    level: 2,
    upstreamNodes: ['rdc-northeast'],
    downstreamNodes: [],
    position: { currentStock: 300, onOrder: 0, allocated: 0 },
    costs: { holding: 2, ordering: 100, shortage: 150 },
    leadTime: { mean: 2, stdDev: 0.5, distribution: 'normal' },
    capacity: { storage: 1000, throughput: 500 },
  });

  // Store 2: Boston (medium volume)
  system.addInventoryNode({
    nodeId: 'store-boston',
    type: 'retail',
    level: 2,
    upstreamNodes: ['rdc-northeast'],
    downstreamNodes: [],
    position: { currentStock: 200, onOrder: 0, allocated: 0 },
    costs: { holding: 1.5, ordering: 100, shortage: 120 },
    leadTime: { mean: 2, stdDev: 0.5, distribution: 'normal' },
    capacity: { storage: 800, throughput: 300 },
  });

  // Store 3: Philly (medium volume)
  system.addInventoryNode({
    nodeId: 'store-philly',
    type: 'retail',
    level: 2,
    upstreamNodes: ['rdc-northeast'],
    downstreamNodes: [],
    position: { currentStock: 180, onOrder: 0, allocated: 0 },
    costs: { holding: 1.5, ordering: 100, shortage: 120 },
    leadTime: { mean: 2, stdDev: 0.5, distribution: 'normal' },
    capacity: { storage: 800, throughput: 300 },
  });

  // Generate historical demand data
  console.log('\nGenerating historical demand data...');
  const historicalData = generateRetailDemand(180); // 6 months

  // Train system
  console.log('Training forecasting models...');
  await system.train(historicalData);

  // Optimize supply chain
  console.log('\nOptimizing inventory policies with swarm intelligence...');
  const optimization = await system.optimize('product-electronics-tablet', {
    dayOfWeek: 1, // Monday
    weekOfYear: 20,
    monthOfYear: 5,
    isHoliday: false,
    promotions: 0,
    priceIndex: 1.0,
  });

  // Display results
  console.log('\n=== Optimization Results ===');
  console.log(`\nBest Policy Found:`);
  console.log(`  Reorder Point: ${optimization.bestPolicy.reorderPoint.toFixed(0)}`);
  console.log(`  Order-Up-To Level: ${optimization.bestPolicy.orderUpToLevel.toFixed(0)}`);
  console.log(`  Safety Factor: ${optimization.bestPolicy.safetyFactor.toFixed(2)}`);

  console.log(`\nPerformance:`);
  console.log(`  Total Cost: $${optimization.networkOptimization.totalCost.toFixed(2)}`);
  console.log(
    `  Average Service Level: ${(optimization.networkOptimization.avgServiceLevel * 100).toFixed(1)}%`
  );

  console.log(`\nSwarm Optimization:`);
  console.log(`  Particles: ${optimization.swarmResult.particles.length}`);
  console.log(`  Iterations: ${optimization.swarmResult.iterations}`);
  console.log(
    `  Best Fitness: ${optimization.swarmResult.bestFitness.combined.toFixed(2)}`
  );

  // Get recommendations for each node
  console.log('\n=== Node Recommendations ===');
  const recommendations = await system.getRecommendations(
    'product-electronics-tablet',
    {
      dayOfWeek: 1,
      weekOfYear: 20,
      monthOfYear: 5,
      isHoliday: false,
      promotions: 0,
      priceIndex: 1.0,
    }
  );

  for (const rec of recommendations.recommendations) {
    console.log(`\n${rec.nodeId}:`);
    console.log(`  Action: ${rec.action}`);
    console.log(`  Quantity: ${rec.quantity.toFixed(0)} units`);
    console.log(`  Urgency: ${rec.urgency}`);
    console.log(`  Reason: ${rec.reason}`);
  }

  // Forecast analysis
  console.log('\n=== Demand Forecast ===');
  console.log(`Point Forecast: ${recommendations.forecast.pointForecast.toFixed(0)} units`);
  console.log(
    `Prediction Interval: [${recommendations.forecast.lowerBound.toFixed(0)}, ${recommendations.forecast.upperBound.toFixed(0)}]`
  );
  console.log(`Confidence: ${(recommendations.forecast.confidence * 100).toFixed(0)}%`);
  console.log(`Uncertainty: ${recommendations.forecast.uncertainty.toFixed(2)}`);

  // Simulate promotional event
  console.log('\n=== Promotional Event Simulation ===');
  console.log('Planning for weekend sale (20% off)...');

  const promoRecommendations = await system.getRecommendations(
    'product-electronics-tablet',
    {
      dayOfWeek: 6, // Saturday
      weekOfYear: 21,
      monthOfYear: 5,
      isHoliday: false,
      promotions: 1,
      priceIndex: 0.8, // 20% off
    }
  );

  console.log(
    `Expected Demand: ${promoRecommendations.forecast.pointForecast.toFixed(0)} units (${((promoRecommendations.forecast.pointForecast / recommendations.forecast.pointForecast - 1) * 100).toFixed(1)}% increase)`
  );

  for (const rec of promoRecommendations.recommendations) {
    if (rec.action === 'ORDER') {
      console.log(`\n${rec.nodeId}: Increase order by ${rec.quantity.toFixed(0)} units`);
    }
  }

  // Get system metrics
  console.log('\n=== System Metrics ===');
  const metrics = system.getMetrics();
  console.log(`Forecast Calibration:`);
  console.log(`  Coverage: ${(metrics.forecastCalibration.coverage * 100).toFixed(1)}%`);
  console.log(
    `  Interval Width: ${metrics.forecastCalibration.intervalWidth.toFixed(2)}`
  );

  console.log(`\nNetwork Topology:`);
  console.log(`  Nodes: ${metrics.networkTopology.nodes.length}`);
  console.log(`  Edges: ${metrics.networkTopology.edges.length}`);

  console.log(`\nPareto Front Solutions: ${metrics.paretoFront.length}`);
}

/**
 * Generate synthetic retail demand data
 */
function generateRetailDemand(days: number) {
  const data = [];
  const baseDate = new Date('2024-01-01');
  const products = [
    'product-electronics-tablet',
    'product-electronics-laptop',
    'product-electronics-phone',
  ];

  for (let i = 0; i < days; i++) {
    const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    const dayOfWeek = date.getDay();
    const weekOfYear = Math.floor(i / 7);
    const monthOfYear = date.getMonth();

    for (const productId of products) {
      // Base demand
      let baseDemand = 100;

      // Seasonal pattern (higher in Q4 for holidays)
      const seasonal = 1 + 0.3 * Math.sin(((monthOfYear - 3) / 12) * 2 * Math.PI);

      // Weekly pattern (higher on weekends)
      const weekly = dayOfWeek >= 5 ? 1.4 : 1.0;

      // Random variation
      const noise = 1 + (Math.random() - 0.5) * 0.3;

      // Promotional spikes (10% of days)
      const isPromo = Math.random() > 0.9;
      const promo = isPromo ? 1.5 : 1.0;

      const demand = baseDemand * seasonal * weekly * promo * noise;

      data.push({
        productId,
        timestamp: date.getTime(),
        demand: Math.max(0, demand),
        features: {
          dayOfWeek,
          weekOfYear,
          monthOfYear,
          isHoliday: false,
          promotions: isPromo ? 1 : 0,
          priceIndex: isPromo ? 0.8 : 1.0,
        },
      });
    }
  }

  return data;
}

// Run the scenario
runRetailScenario()
  .then(() => {
    console.log('\n=== Scenario Complete ===');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
