#!/usr/bin/env node

/**
 * Smoke test for @neural-trader/neuro-divergent
 * Tests basic functionality without requiring extensive setup
 */

const assert = require('assert');

async function runSmokeTest() {
  console.log('🧪 Running smoke tests for @neural-trader/neuro-divergent...\n');

  try {
    // Test 1: Module loading
    console.log('Test 1: Loading module...');
    const { NeuralForecast, listAvailableModels, version, isGpuAvailable } = require('..');
    console.log('✅ Module loaded successfully\n');

    // Test 2: Version check
    console.log('Test 2: Version check...');
    const v = version();
    assert(typeof v === 'string', 'Version should be a string');
    console.log(`✅ Version: ${v}\n`);

    // Test 3: List models
    console.log('Test 3: List available models...');
    const models = listAvailableModels();
    assert(Array.isArray(models), 'Models should be an array');
    assert(models.length > 0, 'Should have at least one model');
    console.log(`✅ Available models: ${models.join(', ')}\n`);

    // Test 4: GPU availability
    console.log('Test 4: Check GPU availability...');
    const hasGpu = isGpuAvailable();
    assert(typeof hasGpu === 'boolean', 'GPU check should return boolean');
    console.log(`✅ GPU available: ${hasGpu}\n`);

    // Test 5: Create forecast instance
    console.log('Test 5: Create NeuralForecast instance...');
    const forecast = new NeuralForecast();
    assert(forecast !== null, 'Forecast instance should not be null');
    console.log('✅ NeuralForecast instance created\n');

    // Test 6: Add model
    console.log('Test 6: Add LSTM model...');
    const modelId = await forecast.addModel({
      modelType: 'LSTM',
      inputSize: 10,
      horizon: 5,
      hiddenSize: 32,
      numLayers: 2,
      dropout: 0.1,
      learningRate: 0.001
    });
    assert(typeof modelId === 'string', 'Model ID should be a string');
    assert(modelId.length > 0, 'Model ID should not be empty');
    console.log(`✅ Model added with ID: ${modelId}\n`);

    // Test 7: Get model config
    console.log('Test 7: Get model configuration...');
    const config = await forecast.getConfig(modelId);
    assert(config !== null, 'Config should not be null');
    assert(config.modelType === 'LSTM', 'Model type should be LSTM');
    assert(config.inputSize === 10, 'Input size should be 10');
    assert(config.horizon === 5, 'Horizon should be 5');
    console.log('✅ Model configuration retrieved\n');

    // Test 8: Prepare time series data
    console.log('Test 8: Prepare time series data...');
    const baseDate = new Date('2024-01-01T00:00:00Z');
    const data = {
      points: Array.from({ length: 50 }, (_, i) => ({
        timestamp: new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000).toISOString(),
        value: 100 + Math.sin(i * 0.1) * 10 + Math.random() * 5
      })),
      frequency: '1D'
    };
    assert(data.points.length === 50, 'Should have 50 data points');
    console.log(`✅ Created ${data.points.length} data points\n`);

    // Test 9: Fit model (basic test, may not converge)
    console.log('Test 9: Fit model (this may take a moment)...');
    const metrics = await forecast.fit(modelId, data);
    assert(Array.isArray(metrics), 'Metrics should be an array');
    assert(metrics.length > 0, 'Should have at least one epoch');
    assert(typeof metrics[0].trainLoss === 'number', 'Train loss should be a number');
    console.log(`✅ Model trained for ${metrics.length} epochs\n`);
    console.log(`   Last epoch loss: ${metrics[metrics.length - 1].trainLoss.toFixed(4)}\n`);

    // Test 10: Make predictions
    console.log('Test 10: Make predictions...');
    const predictions = await forecast.predict(modelId, 5);
    assert(Array.isArray(predictions.predictions), 'Predictions should be an array');
    assert(predictions.predictions.length === 5, 'Should have 5 predictions');
    assert(predictions.modelType === 'LSTM', 'Model type should be LSTM');
    console.log('✅ Predictions made successfully\n');
    console.log(`   Predicted values: ${predictions.predictions.map(p => p.toFixed(2)).join(', ')}\n`);

    // Test 11: Cross-validation
    console.log('Test 11: Cross-validation...');
    const cvResults = await forecast.crossValidation(modelId, data, 3, 1);
    assert(typeof cvResults.mae === 'number', 'MAE should be a number');
    assert(typeof cvResults.mse === 'number', 'MSE should be a number');
    assert(typeof cvResults.rmse === 'number', 'RMSE should be a number');
    assert(typeof cvResults.mape === 'number', 'MAPE should be a number');
    console.log('✅ Cross-validation completed\n');
    console.log(`   MAE: ${cvResults.mae.toFixed(4)}`);
    console.log(`   RMSE: ${cvResults.rmse.toFixed(4)}`);
    console.log(`   MAPE: ${cvResults.mape.toFixed(2)}%\n`);

    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('✅ All smoke tests passed!');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    process.exit(0);

  } catch (error) {
    console.error('\n❌ Smoke test failed:');
    console.error(error.message);
    console.error('\nStack trace:');
    console.error(error.stack);
    process.exit(1);
  }
}

runSmokeTest();
