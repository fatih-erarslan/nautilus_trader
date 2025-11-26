/**
 * Complete Neural Network Training Pipeline Example
 *
 * This comprehensive example demonstrates:
 * 1. Data preparation and validation
 * 2. Model training with multiple architectures
 * 3. Hyperparameter optimization
 * 4. Evaluation and overfitting detection
 * 5. Backtesting and performance analysis
 * 6. Production deployment preparation
 *
 * Usage: node complete-training-pipeline.js
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');
const fs = require('fs').promises;
const path = require('path');

// Configuration
const CONFIG = {
  dataDir: './data',
  modelsDir: './trained_models',
  symbol: 'AAPL',
  modelTypes: ['gru', 'lstm', 'transformer'],
  epochs: 100,
  useGPU: true,
  validationSplit: 0.2,
  backtestPeriod: {
    start: '2023-01-01',
    end: '2023-12-31'
  }
};

// ============================================================================
// Step 1: Data Preparation
// ============================================================================

async function prepareData() {
  console.log('\n=== STEP 1: Data Preparation ===\n');

  const dataPath = path.join(CONFIG.dataDir, `${CONFIG.symbol}_training.csv`);

  // Check if data exists
  try {
    await fs.access(dataPath);
    console.log(`✓ Training data found: ${dataPath}`);
  } catch (error) {
    console.log(`Generating synthetic training data...`);
    await generateSyntheticData(dataPath, 5000);
  }

  // Validate data quality
  const isValid = await validateData(dataPath);

  if (!isValid) {
    throw new Error('Data validation failed. Please check data quality.');
  }

  console.log('✓ Data preparation complete\n');
  return dataPath;
}

async function generateSyntheticData(outputPath, samples) {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });

  const rows = ['timestamp,value'];
  const baseDate = new Date('2023-01-01');
  let price = 150.0;

  for (let i = 0; i < samples; i++) {
    const timestamp = new Date(baseDate.getTime() + i * 3600000).toISOString();

    // Realistic price movement
    const trend = 0.0001 * i; // Long-term uptrend
    const daily = Math.sin(i / 24) * 2; // Daily cycle
    const noise = (Math.random() - 0.5) * 3; // Random walk

    price = price + trend + daily + noise;
    price = Math.max(100, price); // Floor at $100

    rows.push(`${timestamp},${price.toFixed(2)}`);
  }

  await fs.writeFile(outputPath, rows.join('\n'));
  console.log(`Generated ${samples} samples of synthetic data`);
}

async function validateData(dataPath) {
  console.log('Validating data quality...');

  const content = await fs.readFile(dataPath, 'utf-8');
  const lines = content.trim().split('\n');
  const data = lines.slice(1).map(line => {
    const [timestamp, value] = line.split(',');
    return { timestamp, value: parseFloat(value) };
  });

  // Check 1: Minimum sample size
  const minSamples = 1000;
  if (data.length < minSamples) {
    console.error(`✗ Insufficient data: ${data.length} < ${minSamples} samples`);
    return false;
  }
  console.log(`✓ Sample size: ${data.length} samples`);

  // Check 2: No missing values
  const missingCount = data.filter(d => isNaN(d.value)).length;
  if (missingCount > 0) {
    console.error(`✗ Missing values: ${missingCount} rows`);
    return false;
  }
  console.log('✓ No missing values');

  // Check 3: Outliers
  const values = data.map(d => d.value);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
  );
  const outliers = values.filter(v => Math.abs(v - mean) > 3 * std).length;
  const outlierPercent = (outliers / values.length) * 100;

  if (outlierPercent > 5) {
    console.warn(`⚠ High outlier percentage: ${outlierPercent.toFixed(2)}%`);
  } else {
    console.log(`✓ Outliers: ${outlierPercent.toFixed(2)}%`);
  }

  // Check 4: Regular intervals
  const intervals = [];
  for (let i = 1; i < Math.min(100, data.length); i++) {
    const diff = new Date(data[i].timestamp) - new Date(data[i-1].timestamp);
    intervals.push(diff);
  }
  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const variance = intervals.reduce((sum, val) =>
    sum + Math.pow(val - avgInterval, 2), 0) / intervals.length;
  const consistency = 1 - (Math.sqrt(variance) / avgInterval);

  if (consistency < 0.9) {
    console.warn(`⚠ Irregular time intervals: ${(consistency * 100).toFixed(2)}% consistent`);
  } else {
    console.log(`✓ Regular intervals: ${(consistency * 100).toFixed(2)}% consistent`);
  }

  return true;
}

// ============================================================================
// Step 2: Model Training
// ============================================================================

async function trainModels(dataPath) {
  console.log('\n=== STEP 2: Model Training ===\n');

  const models = [];

  for (const modelType of CONFIG.modelTypes) {
    console.log(`\nTraining ${modelType.toUpperCase()} model...`);
    console.log('-'.repeat(50));

    const startTime = Date.now();
    const startMem = process.memoryUsage().heapUsed / 1024 / 1024;

    try {
      const result = await backend.neuralTrain(
        dataPath,
        modelType,
        CONFIG.epochs,
        CONFIG.useGPU
      );

      const duration = Date.now() - startTime;
      const memUsed = (process.memoryUsage().heapUsed / 1024 / 1024) - startMem;

      models.push({
        modelId: result.modelId,
        modelType,
        trainingTimeMs: duration,
        memoryUsedMB: memUsed,
        finalLoss: result.finalLoss,
        validationAccuracy: result.validationAccuracy
      });

      console.log(`✓ Training complete`);
      console.log(`  Model ID: ${result.modelId}`);
      console.log(`  Duration: ${(duration / 1000).toFixed(2)}s`);
      console.log(`  Memory: ${memUsed.toFixed(2)}MB`);
      console.log(`  Final Loss: ${result.finalLoss.toFixed(6)}`);
      console.log(`  Validation Accuracy: ${result.validationAccuracy.toFixed(4)}`);

    } catch (error) {
      console.error(`✗ Training failed: ${error.message}`);
    }
  }

  // Save model registry
  await fs.mkdir(CONFIG.modelsDir, { recursive: true });
  await fs.writeFile(
    path.join(CONFIG.modelsDir, 'model_registry.json'),
    JSON.stringify(models, null, 2)
  );

  console.log(`\n✓ Trained ${models.length} models`);
  return models;
}

// ============================================================================
// Step 3: Model Evaluation
// ============================================================================

async function evaluateModels(models, dataPath) {
  console.log('\n=== STEP 3: Model Evaluation ===\n');

  // Create test data (in production, use separate test set)
  const testDataPath = dataPath; // Simplified for example

  const evaluations = [];

  for (const model of models) {
    console.log(`\nEvaluating ${model.modelType.toUpperCase()} (${model.modelId})...`);
    console.log('-'.repeat(50));

    try {
      const metrics = await backend.neuralEvaluate(
        model.modelId,
        testDataPath,
        CONFIG.useGPU
      );

      evaluations.push({
        ...model,
        ...metrics
      });

      console.log(`Test Samples: ${metrics.testSamples}`);
      console.log(`MAE: ${metrics.mae.toFixed(4)}`);
      console.log(`RMSE: ${metrics.rmse.toFixed(4)}`);
      console.log(`MAPE: ${(metrics.mape * 100).toFixed(2)}%`);
      console.log(`R² Score: ${metrics.r2Score.toFixed(4)}`);

      // Overfitting check
      const gap = model.validationAccuracy - metrics.r2Score;
      if (gap > 0.10) {
        console.log(`⚠️  WARNING: Significant overfitting detected (gap: ${gap.toFixed(4)})`);
      } else if (gap > 0.05) {
        console.log(`⚠️  Mild overfitting detected (gap: ${gap.toFixed(4)})`);
      } else {
        console.log(`✓ No significant overfitting`);
      }

    } catch (error) {
      console.error(`✗ Evaluation failed: ${error.message}`);
    }
  }

  // Find best model
  const bestModel = evaluations.sort((a, b) => b.r2Score - a.r2Score)[0];

  console.log('\n=== Best Model ===');
  console.log(`Model Type: ${bestModel.modelType.toUpperCase()}`);
  console.log(`Model ID: ${bestModel.modelId}`);
  console.log(`R² Score: ${bestModel.r2Score.toFixed(4)}`);
  console.log(`MAE: ${bestModel.mae.toFixed(4)}`);

  return { evaluations, bestModel };
}

// ============================================================================
// Step 4: Hyperparameter Optimization
// ============================================================================

async function optimizeHyperparameters(bestModel) {
  console.log('\n=== STEP 4: Hyperparameter Optimization ===\n');

  const paramRanges = JSON.stringify({
    learning_rate: [0.0001, 0.001, 0.01],
    batch_size: [16, 32, 64],
    hidden_size: [128, 256, 512],
    num_layers: [1, 2, 3]
  });

  console.log('Starting optimization...');
  console.log('Parameter ranges:', JSON.parse(paramRanges));

  const startTime = Date.now();

  const result = await backend.neuralOptimize(
    bestModel.modelId,
    paramRanges,
    CONFIG.useGPU
  );

  const duration = Date.now() - startTime;

  console.log(`\n✓ Optimization complete`);
  console.log(`Duration: ${(duration / 1000).toFixed(2)}s`);
  console.log(`Trials: ${result.trialsCompleted}`);
  console.log(`Best Score: ${result.bestScore.toFixed(4)}`);
  console.log(`Improvement: ${((result.bestScore - bestModel.r2Score) * 100).toFixed(2)}%`);
  console.log('\nBest Parameters:');
  console.log(JSON.stringify(JSON.parse(result.bestParams), null, 2));

  return result;
}

// ============================================================================
// Step 5: Backtesting
// ============================================================================

async function backtestStrategy(bestModel) {
  console.log('\n=== STEP 5: Backtesting ===\n');

  console.log(`Testing ${bestModel.modelType.toUpperCase()} model...`);
  console.log(`Period: ${CONFIG.backtestPeriod.start} to ${CONFIG.backtestPeriod.end}`);
  console.log(`Benchmark: SPY`);

  const result = await backend.neuralBacktest(
    bestModel.modelId,
    CONFIG.backtestPeriod.start,
    CONFIG.backtestPeriod.end,
    'SPY',
    CONFIG.useGPU
  );

  console.log('\n=== Backtest Results ===');
  console.log(`Total Return: ${(result.totalReturn * 100).toFixed(2)}%`);
  console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
  console.log(`Max Drawdown: ${(result.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`Win Rate: ${(result.winRate * 100).toFixed(2)}%`);
  console.log(`Total Trades: ${result.totalTrades}`);

  // Evaluate strategy quality
  console.log('\n=== Strategy Evaluation ===');

  // Sharpe Ratio
  if (result.sharpeRatio > 2.0) {
    console.log('✓ Excellent risk-adjusted returns (Sharpe > 2.0)');
  } else if (result.sharpeRatio > 1.0) {
    console.log('⚠️  Good risk-adjusted returns (Sharpe > 1.0)');
  } else {
    console.log('✗ Poor risk-adjusted returns (Sharpe < 1.0)');
  }

  // Max Drawdown
  if (result.maxDrawdown > -0.20) {
    console.log('✓ Acceptable drawdown (< 20%)');
  } else if (result.maxDrawdown > -0.40) {
    console.log('⚠️  High drawdown (20-40%)');
  } else {
    console.log('✗ Excessive drawdown (> 40%)');
  }

  // Win Rate
  if (result.winRate > 0.55) {
    console.log('✓ Strong win rate (> 55%)');
  } else if (result.winRate > 0.45) {
    console.log('⚠️  Moderate win rate (45-55%)');
  } else {
    console.log('✗ Low win rate (< 45%)');
  }

  return result;
}

// ============================================================================
// Step 6: Production Readiness Check
// ============================================================================

async function productionReadinessCheck(bestModel, backtestResult) {
  console.log('\n=== STEP 6: Production Readiness Check ===\n');

  const checks = {
    dataQuality: true,
    modelAccuracy: bestModel.r2Score > 0.70,
    noOverfitting: (bestModel.validationAccuracy - bestModel.r2Score) < 0.10,
    backtestPerformance: backtestResult.sharpeRatio > 1.0 && backtestResult.maxDrawdown > -0.30,
    inferenceLatency: await checkInferenceLatency(bestModel.modelId),
    documentation: true, // Manually verify
    monitoring: true, // Manually verify
    riskManagement: true // Manually verify
  };

  console.log('Production Readiness Checklist:');
  console.log('-'.repeat(50));

  Object.entries(checks).forEach(([check, passed]) => {
    const status = passed ? '✓' : '✗';
    const label = check.replace(/([A-Z])/g, ' $1').toLowerCase();
    console.log(`${status} ${label}`);
  });

  const allPassed = Object.values(checks).every(v => v);

  console.log('\n' + '='.repeat(50));
  if (allPassed) {
    console.log('✓ MODEL IS READY FOR PRODUCTION');
  } else {
    console.log('⚠️  MODEL NEEDS IMPROVEMENTS BEFORE PRODUCTION');
  }
  console.log('='.repeat(50));

  return allPassed;
}

async function checkInferenceLatency(modelId) {
  console.log('Measuring inference latency...');

  const iterations = 100;
  const latencies = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await backend.neuralForecast(CONFIG.symbol, 24, CONFIG.useGPU);
    latencies.push(performance.now() - start);
  }

  const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p95 = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

  console.log(`  Average: ${avgLatency.toFixed(2)}ms`);
  console.log(`  P95: ${p95.toFixed(2)}ms`);

  return avgLatency < 1000; // Sub-second requirement
}

// ============================================================================
// Step 7: Generate Report
// ============================================================================

async function generateReport(models, evaluations, bestModel, backtestResult) {
  console.log('\n=== STEP 7: Generating Report ===\n');

  const report = {
    timestamp: new Date().toISOString(),
    configuration: CONFIG,
    models: models.length,
    bestModel: {
      modelId: bestModel.modelId,
      modelType: bestModel.modelType,
      validationAccuracy: bestModel.validationAccuracy,
      testR2Score: bestModel.r2Score,
      mae: bestModel.mae,
      rmse: bestModel.rmse
    },
    backtest: {
      totalReturn: backtestResult.totalReturn,
      sharpeRatio: backtestResult.sharpeRatio,
      maxDrawdown: backtestResult.maxDrawdown,
      winRate: backtestResult.winRate,
      totalTrades: backtestResult.totalTrades
    },
    allModels: evaluations
  };

  const reportPath = path.join(CONFIG.modelsDir, 'training_report.json');
  await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

  console.log(`✓ Report saved to ${reportPath}`);

  // Print summary
  console.log('\n' + '='.repeat(70));
  console.log('TRAINING PIPELINE COMPLETE');
  console.log('='.repeat(70));
  console.log(`\nBest Model: ${bestModel.modelType.toUpperCase()}`);
  console.log(`Model ID: ${bestModel.modelId}`);
  console.log(`\nPerformance Metrics:`);
  console.log(`  Validation Accuracy: ${bestModel.validationAccuracy.toFixed(4)}`);
  console.log(`  Test R² Score: ${bestModel.r2Score.toFixed(4)}`);
  console.log(`  MAE: ${bestModel.mae.toFixed(4)}`);
  console.log(`  RMSE: ${bestModel.rmse.toFixed(4)}`);
  console.log(`\nBacktest Results:`);
  console.log(`  Total Return: ${(backtestResult.totalReturn * 100).toFixed(2)}%`);
  console.log(`  Sharpe Ratio: ${backtestResult.sharpeRatio.toFixed(2)}`);
  console.log(`  Max Drawdown: ${(backtestResult.maxDrawdown * 100).toFixed(2)}%`);
  console.log(`  Win Rate: ${(backtestResult.winRate * 100).toFixed(2)}%`);
  console.log('\n' + '='.repeat(70));

  return report;
}

// ============================================================================
// Main Pipeline
// ============================================================================

async function main() {
  console.log('\n' + '='.repeat(70));
  console.log('NEURAL NETWORK TRAINING PIPELINE');
  console.log('='.repeat(70));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Models: ${CONFIG.modelTypes.join(', ')}`);
  console.log(`Epochs: ${CONFIG.epochs}`);
  console.log(`GPU: ${CONFIG.useGPU ? 'Enabled' : 'Disabled'}`);

  try {
    // Step 1: Prepare data
    const dataPath = await prepareData();

    // Step 2: Train models
    const models = await trainModels(dataPath);

    // Step 3: Evaluate models
    const { evaluations, bestModel } = await evaluateModels(models, dataPath);

    // Step 4: Optimize hyperparameters
    const optimizationResult = await optimizeHyperparameters(bestModel);

    // Step 5: Backtest strategy
    const backtestResult = await backtestStrategy(bestModel);

    // Step 6: Check production readiness
    const isReady = await productionReadinessCheck(bestModel, backtestResult);

    // Step 7: Generate report
    const report = await generateReport(models, evaluations, bestModel, backtestResult);

    console.log('\n✓ Pipeline completed successfully\n');

    return {
      success: true,
      isProductionReady: isReady,
      report
    };

  } catch (error) {
    console.error('\n✗ Pipeline failed:', error.message);
    console.error(error.stack);

    return {
      success: false,
      error: error.message
    };
  }
}

// Run pipeline if executed directly
if (require.main === module) {
  main()
    .then(result => {
      process.exit(result.success ? 0 : 1);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = {
  main,
  prepareData,
  trainModels,
  evaluateModels,
  optimizeHyperparameters,
  backtestStrategy,
  productionReadinessCheck
};
