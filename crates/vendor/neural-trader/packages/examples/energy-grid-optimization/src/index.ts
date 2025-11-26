/**
 * Energy Grid Optimization Example Package
 *
 * Self-learning energy grid optimization with:
 * - Multi-horizon load forecasting with conformal prediction
 * - Unit commitment optimization using sublinear-time-solver
 * - Swarm-based scheduling strategy exploration
 * - Renewable energy integration (solar, wind)
 * - Battery storage optimization
 * - Memory-persistent pattern learning via AgentDB
 *
 * @packageDocumentation
 */

export { LoadForecaster } from './load-forecaster.js';
export { UnitCommitmentOptimizer } from './unit-commitment.js';
export { SwarmScheduler } from './swarm-scheduler.js';

export type {
  // Time and forecasting
  ForecastHorizon,
  LoadForecast,
  RenewableForecast,
  ForecastErrorCorrection,
  LoadForecasterConfig,

  // Generators and storage
  GeneratorType,
  GeneratorUnit,
  BatteryStorage,

  // Optimization
  UnitCommitment,
  UnitCommitmentConfig,
  DemandResponseProgram,

  // Scheduling
  SchedulingStrategy,
  OptimizationResult,
  SwarmSchedulerConfig,

  // Grid state
  GridState,
} from './types.js';

/**
 * Example usage demonstrating complete grid optimization workflow
 */
export async function runEnergyGridOptimization(): Promise<void> {
  const { LoadForecaster } = await import('./load-forecaster.js');
  const { SwarmScheduler } = await import('./swarm-scheduler.js');
  const {
    ForecastHorizon,
    GeneratorType,
  } = await import('./types.js');

  console.log('=== Energy Grid Optimization Example ===\n');

  // 1. Initialize load forecaster
  console.log('1. Initializing load forecaster...');
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

  // 2. Create current grid state
  const currentState = {
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
  console.log('2. Generating load forecasts...');
  const forecasts = await forecaster.forecast(currentState);
  console.log(`   Generated ${forecasts.length} forecasts`);
  forecasts.forEach(f => {
    console.log(
      `   - ${f.horizon}: ${f.loadMW.toFixed(0)} MW [${f.lowerBound.toFixed(0)}, ${f.upperBound.toFixed(0)}]`
    );
  });

  // 4. Define generator fleet
  console.log('\n3. Defining generator fleet...');
  const generators = [
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
    {
      id: 'natgas-1',
      type: GeneratorType.NATURAL_GAS,
      minCapacityMW: 50,
      maxCapacityMW: 300,
      rampUpRate: 100,
      rampDownRate: 100,
      minUpTime: 2,
      minDownTime: 2,
      startupCost: 2000,
      shutdownCost: 500,
      variableCost: 45,
      fixedCost: 300,
      startupTime: 1,
      status: {
        isOnline: true,
        currentOutputMW: 200,
        hoursOnline: 6,
        hoursOffline: 0,
      },
    },
    {
      id: 'wind-1',
      type: GeneratorType.WIND,
      minCapacityMW: 0,
      maxCapacityMW: 200,
      rampUpRate: 200,
      rampDownRate: 200,
      minUpTime: 0,
      minDownTime: 0,
      startupCost: 0,
      shutdownCost: 0,
      variableCost: 5,
      fixedCost: 50,
      startupTime: 0,
      status: {
        isOnline: true,
        currentOutputMW: 150,
        hoursOnline: 24,
        hoursOffline: 0,
      },
    },
    {
      id: 'solar-1',
      type: GeneratorType.SOLAR,
      minCapacityMW: 0,
      maxCapacityMW: 300,
      rampUpRate: 300,
      rampDownRate: 300,
      minUpTime: 0,
      minDownTime: 0,
      startupCost: 0,
      shutdownCost: 0,
      variableCost: 2,
      fixedCost: 30,
      startupTime: 0,
      status: {
        isOnline: true,
        currentOutputMW: 250,
        hoursOnline: 8,
        hoursOffline: 0,
      },
    },
  ];

  console.log(`   Registered ${generators.length} generators`);

  // 5. Define battery storage
  const batteries = [
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

  console.log(`   Registered ${batteries.length} battery systems`);

  // 6. Initialize swarm scheduler
  console.log('\n4. Initializing swarm scheduler...');
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
  console.log('5. Running swarm optimization...');
  const result = await scheduler.optimizeSchedule(
    generators,
    batteries,
    forecasts,
    []
  );

  console.log('\n=== Optimization Results ===');
  console.log(`Strategy: ${result.strategy.name}`);
  console.log(`Total Cost: $${result.totalCost.toFixed(2)}`);
  console.log(`Renewable Utilization: ${result.renewableUtilization.toFixed(1)}%`);
  console.log(`Total Emissions: ${result.totalEmissions.toFixed(2)} tons CO2`);
  console.log(`Quality Score: ${result.qualityScore.toFixed(3)}`);
  console.log(`Computation Time: ${result.computeTimeMs.toFixed(0)}ms`);
  console.log(`Feasible: ${result.isFeasible ? 'Yes' : 'No'}`);

  // 8. Display commitment schedule
  console.log('\n=== Hourly Commitment Schedule ===');
  result.commitments.slice(0, 5).forEach((commit, idx) => {
    console.log(`\nHour ${idx + 1}: ${commit.timestamp.toLocaleTimeString()}`);
    console.log(`  Load: ${commit.totalLoadMW.toFixed(0)} MW`);
    console.log(`  Generation: ${commit.totalGenerationMW.toFixed(0)} MW`);
    console.log(`  Cost: $${commit.totalCost.toFixed(2)}`);

    commit.commitments
      .filter(c => c.isCommitted)
      .forEach(c => {
        console.log(`    - ${c.generatorId}: ${c.outputMW.toFixed(0)} MW`);
      });

    if (commit.batteryOperations.length > 0) {
      commit.batteryOperations.forEach(b => {
        if (b.chargeMW > 0) {
          console.log(`    - ${b.batteryId}: Charging ${b.chargeMW.toFixed(0)} MW`);
        }
        if (b.dischargeMW > 0) {
          console.log(`    - ${b.batteryId}: Discharging ${b.dischargeMW.toFixed(0)} MW`);
        }
      });
    }
  });

  // 9. Display strategy statistics
  console.log('\n=== Strategy Performance ===');
  const stats = scheduler.getStrategyStatistics();
  stats.slice(0, 3).forEach(stat => {
    console.log(
      `  ${stat.strategyId}: ${stat.avgScore.toFixed(3)} (${stat.count} evaluations)`
    );
  });

  console.log('\n=== Optimization Complete ===');
}

// Run example if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runEnergyGridOptimization().catch(console.error);
}
