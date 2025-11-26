# @neural-trader/example-energy-grid-optimization

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fexample-energy-grid-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-energy-grid-optimization)
[![npm downloads](https://img.shields.io/npm/dm/@neural-trader/example-energy-grid-optimization.svg)](https://www.npmjs.com/package/@neural-trader/example-energy-grid-optimization)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/neural-trader/ci.yml?branch=main)](https://github.com/ruvnet/neural-trader/actions)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)]()

Self-learning energy grid optimization with multi-horizon load forecasting, unit commitment optimization, and swarm-based scheduling strategy exploration.

## Features

### 🔮 Multi-Horizon Load Forecasting
- **5-minute to 7-day** forecasting horizons
- **Conformal prediction intervals** with guaranteed coverage
- **Self-learning error correction** using AgentDB memory
- **Weather and calendar** feature engineering
- **Memory-persistent patterns** for continuous improvement

### ⚡ Unit Commitment Optimization
- **Sublinear-time solver** for fast optimization
- **Real-world constraints**:
  - Ramp rates (up/down)
  - Minimum up/down times
  - Spinning reserve requirements
  - Generator capacity limits
- **Battery storage optimization**
- **Renewable energy integration**
- **Demand response coordination**

### 🐝 Swarm-Based Scheduling
- **Multi-strategy exploration** with parallel evaluation
- **Self-learning strategy optimization** via AgentDB
- **Multi-objective optimization**:
  - Cost minimization
  - Renewable utilization maximization
  - Emissions reduction
  - Reliability focus
- **Adaptive parameter tuning**
- **Memory-persistent performance tracking**

## Installation

```bash
npm install @neural-trader/example-energy-grid-optimization
```

### Dependencies
- `@neural-trader/predictor` - Conformal prediction
- `agentdb` - Memory-persistent pattern learning
- `sublinear-time-solver` - Fast optimization

## Quick Start

```typescript
import {
  LoadForecaster,
  UnitCommitmentOptimizer,
  SwarmScheduler,
  ForecastHorizon,
  GeneratorType,
  type GridState,
  type GeneratorUnit,
  type BatteryStorage,
} from '@neural-trader/example-energy-grid-optimization';

// 1. Initialize load forecaster
const forecaster = new LoadForecaster({
  horizons: [
    ForecastHorizon.HOUR_1,
    ForecastHorizon.HOUR_4,
    ForecastHorizon.HOUR_24,
  ],
  historyWindowDays: 30,
  confidenceLevel: 0.95,
  enableErrorCorrection: true,
  correctionUpdateFrequency: 1,
  memoryNamespace: 'grid-load-forecasts',
});

await forecaster.initialize();

// 2. Define current grid state
const currentState: GridState = {
  timestamp: new Date(),
  loadMW: 5000,
  generationMW: 5100,
  renewablePenetration: 25,
  frequency: 60.0,
  voltageStability: 0.98,
  activeGenerators: ['gen-1', 'gen-2', 'gen-3'],
  weather: {
    temperature: 22,
    windSpeed: 8,
    solarIrradiance: 800,
    precipitation: 0,
  },
  dayOfWeek: new Date().getDay(),
  hourOfDay: new Date().getHours(),
  isHoliday: false,
};

// 3. Generate load forecasts
const forecasts = await forecaster.forecast(currentState);
console.log(`Generated ${forecasts.length} forecasts`);

// 4. Define generator fleet
const generators: GeneratorUnit[] = [
  {
    id: 'coal-1',
    type: GeneratorType.COAL,
    minCapacityMW: 100,
    maxCapacityMW: 500,
    rampUpRate: 50,
    rampDownRate: 50,
    minUpTime: 4,
    minDownTime: 4,
    startupCost: 5000,
    shutdownCost: 2000,
    variableCost: 30,
    fixedCost: 500,
    startupTime: 2,
    status: {
      isOnline: true,
      currentOutputMW: 300,
      hoursOnline: 12,
      hoursOffline: 0,
    },
  },
  // Add more generators...
];

// 5. Define battery storage
const batteries: BatteryStorage[] = [
  {
    id: 'battery-1',
    maxPowerMW: 100,
    capacityMWh: 400,
    currentChargeMWh: 200,
    chargeEfficiency: 0.95,
    dischargeEfficiency: 0.95,
    minChargeMWh: 40,
    maxChargeMWh: 400,
    degradationRate: 0.0001,
  },
];

// 6. Initialize swarm scheduler
const scheduler = new SwarmScheduler({
  swarmSize: 8,
  explorationRate: 0.3,
  maxIterations: 10,
  convergenceThreshold: 0.01,
  enableOpenRouter: false,
  memoryNamespace: 'grid-scheduling-strategies',
});

await scheduler.initialize();

// 7. Optimize schedule
const result = await scheduler.optimizeSchedule(
  generators,
  batteries,
  forecasts
);

console.log('Optimization Results:');
console.log(`Strategy: ${result.strategy.name}`);
console.log(`Total Cost: $${result.totalCost.toFixed(2)}`);
console.log(`Renewable Utilization: ${result.renewableUtilization.toFixed(1)}%`);
console.log(`Total Emissions: ${result.totalEmissions.toFixed(2)} tons CO2`);
console.log(`Quality Score: ${result.qualityScore.toFixed(3)}`);
```

## Run Example

```bash
npm run build
node dist/index.js
```

## Architecture

### LoadForecaster

Multi-horizon load forecasting with self-learning error correction:

```typescript
const forecaster = new LoadForecaster({
  horizons: [
    ForecastHorizon.MINUTES_5,
    ForecastHorizon.HOUR_1,
    ForecastHorizon.HOUR_24,
    ForecastHorizon.HOUR_168, // 1 week
  ],
  historyWindowDays: 30,
  confidenceLevel: 0.95,
  enableErrorCorrection: true,
  correctionUpdateFrequency: 1,
  memoryNamespace: 'grid-load-forecasts',
});

// Add historical data
await forecaster.addHistoricalState(gridState);

// Generate forecasts
const forecasts = await forecaster.forecast(currentState);

// Update with actual observations
await forecaster.updateWithActual(forecast, actualLoadMW);

// Get accuracy metrics
const metrics = forecaster.getAccuracyMetrics();
console.log(`MAE: ${metrics.mae.toFixed(2)} MW`);
console.log(`MAPE: ${metrics.mape.toFixed(2)}%`);
```

**Features:**
- Pattern-based forecasting using historical similarity
- Hourly, daily, and weather-based bias corrections
- Exponential moving average for error statistics
- Automatic persistence to AgentDB

### UnitCommitmentOptimizer

Sublinear optimization with real-world grid constraints:

```typescript
const optimizer = new UnitCommitmentOptimizer({
  planningHorizonHours: 24,
  timeStepMinutes: 60,
  reserveMarginPercent: 10,
  maxComputeTimeMs: 5000,
  solverTolerance: 1e-6,
  enableBatteryOptimization: true,
});

optimizer.registerGenerators(generators);
optimizer.registerBatteries(batteries);
optimizer.registerDemandResponse(demandResponsePrograms);

const commitments = await optimizer.optimize(loadForecasts, renewableForecasts);
```

**Constraints:**
- **Load balance**: Generation = Load at every time step
- **Capacity limits**: Min/max output for each generator
- **Ramp rates**: Maximum change in output per hour
- **Min up/down times**: Minimum hours online/offline
- **Spinning reserve**: Extra capacity for contingencies
- **Battery limits**: Charge/discharge rates, state of charge

### SwarmScheduler

Multi-strategy exploration with evolutionary optimization:

```typescript
const scheduler = new SwarmScheduler({
  swarmSize: 10, // Number of parallel strategies
  explorationRate: 0.3, // Exploration vs exploitation
  maxIterations: 20,
  convergenceThreshold: 0.01,
  enableOpenRouter: false, // AI-guided strategy generation
  memoryNamespace: 'grid-scheduling-strategies',
});

await scheduler.initialize(); // Load learned strategies

const result = await scheduler.optimizeSchedule(
  generators,
  batteries,
  loadForecasts,
  renewableForecasts
);

// Get strategy statistics
const stats = scheduler.getStrategyStatistics();
console.log('Top Strategies:');
stats.slice(0, 5).forEach(stat => {
  console.log(`  ${stat.strategyId}: ${stat.avgScore.toFixed(3)}`);
});

// Get best learned strategies
const bestStrategies = scheduler.getBestStrategies(5);
```

**Multi-Objective Optimization:**
- **Cost**: Total generation + startup costs
- **Renewable**: Solar/wind utilization percentage
- **Emissions**: CO2 emissions from fossil fuels
- **Reliability**: Reserve margins and stability

**Strategy Evolution:**
- Elite selection (top 30%)
- Mutation (40% with small random changes)
- Exploration (30% completely random)

## Real-World Applications

### 1. Day-Ahead Market Scheduling

```typescript
// Generate 24-hour load forecasts
const forecasts = await forecaster.forecast(currentState, [
  ForecastHorizon.HOUR_24,
]);

// Optimize unit commitment
const result = await scheduler.optimizeSchedule(
  generators,
  batteries,
  forecasts,
  renewableForecasts
);

// Submit bids to day-ahead market
result.commitments.forEach(commitment => {
  submitMarketBid(commitment.timestamp, commitment.totalGenerationMW);
});
```

### 2. Real-Time Dispatch

```typescript
// Short-horizon forecasts for real-time balancing
const forecasts = await forecaster.forecast(currentState, [
  ForecastHorizon.MINUTES_5,
  ForecastHorizon.MINUTES_15,
]);

// Fast optimization for immediate dispatch
const optimizer = new UnitCommitmentOptimizer({
  planningHorizonHours: 1,
  timeStepMinutes: 5,
  maxComputeTimeMs: 100, // Fast for real-time
  solverTolerance: 1e-4,
  enableBatteryOptimization: true,
});

const commitments = await optimizer.optimize(forecasts);
```

### 3. Renewable Integration Planning

```typescript
// Evaluate impact of new renewable capacity
const newGenerators = [
  ...existingGenerators,
  {
    id: 'wind-2',
    type: GeneratorType.WIND,
    maxCapacityMW: 500, // New 500 MW wind farm
    // ...
  },
];

const result = await scheduler.optimizeSchedule(
  newGenerators,
  batteries,
  forecasts,
  renewableForecasts
);

console.log(`Renewable utilization: ${result.renewableUtilization.toFixed(1)}%`);
console.log(`Cost savings: $${costSavings.toFixed(2)}`);
console.log(`Emissions reduction: ${emissionsReduction.toFixed(2)} tons CO2`);
```

### 4. Battery Storage Arbitrage

```typescript
// Optimize battery charging/discharging for price arbitrage
const result = await scheduler.optimizeSchedule(
  generators,
  batteries,
  forecasts
);

result.commitments.forEach(commitment => {
  commitment.batteryOperations.forEach(op => {
    if (op.chargeMW > 0) {
      console.log(`${op.batteryId}: Charge ${op.chargeMW.toFixed(0)} MW`);
    }
    if (op.dischargeMW > 0) {
      console.log(`${op.batteryId}: Discharge ${op.dischargeMW.toFixed(0)} MW`);
    }
  });
});
```

## Self-Learning Capabilities

### Load Forecasting

The forecaster continuously learns from forecast errors:

```typescript
// Initial forecast
const forecast = await forecaster.forecast(currentState);

// After actual load is observed
await forecaster.updateWithActual(forecast, actualLoadMW);

// Error corrections are automatically applied:
// - Hourly bias: Systematic errors by hour of day
// - Daily bias: Systematic errors by day of week
// - Weather corrections: Temperature/wind/solar adjustments
```

### Strategy Optimization

The swarm scheduler evolves strategies over time:

```typescript
// Run multiple optimization cycles
for (let i = 0; i < 100; i++) {
  const forecasts = await forecaster.forecast(currentState);
  const result = await scheduler.optimizeSchedule(
    generators,
    batteries,
    forecasts
  );

  // Strategies are automatically:
  // 1. Evaluated based on multi-objective quality score
  // 2. Ranked by performance
  // 3. Evolved using genetic algorithm
  // 4. Persisted to AgentDB for future sessions
}

// Best strategies are automatically loaded in next session
const newScheduler = new SwarmScheduler(config);
await newScheduler.initialize(); // Loads learned strategies
```

## Performance

- **Load Forecasting**: <10ms per horizon (with history)
- **Unit Commitment**: 1-5 seconds for 24-hour horizon
- **Swarm Optimization**: 10-30 seconds for 8 strategies
- **Memory Usage**: ~50MB (with 30 days of history)

## Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch

# Benchmarks
npm run bench
```

## Advanced Configuration

### Custom Distance Metrics

```typescript
// Extend LoadForecaster for custom similarity matching
class CustomForecaster extends LoadForecaster {
  protected euclideanDistance(
    a: Record<string, number>,
    b: Record<string, number>
  ): number {
    // Custom distance metric for your domain
    return customDistanceFunction(a, b);
  }
}
```

### OpenRouter Integration

```typescript
const scheduler = new SwarmScheduler({
  swarmSize: 10,
  enableOpenRouter: true,
  openRouterApiKey: process.env.OPENROUTER_API_KEY,
  memoryNamespace: 'grid-scheduling-strategies',
});

// AI-guided strategy generation uses LLMs to propose
// novel optimization strategies based on past performance
```

## License

MIT OR Apache-2.0

## Related Packages

- `@neural-trader/predictor` - Conformal prediction for trading
- `@neural-trader/core` - Core neural trading utilities
- `agentdb` - Vector database for AI agents
- `sublinear-time-solver` - Fast optimization algorithms

## Contributing

See the main [neural-trader repository](https://github.com/ruvnet/neural-trader) for contribution guidelines.

## Support

- [Documentation](https://github.com/ruvnet/neural-trader)
- [Issues](https://github.com/ruvnet/neural-trader/issues)
- [Discussions](https://github.com/ruvnet/neural-trader/discussions)
