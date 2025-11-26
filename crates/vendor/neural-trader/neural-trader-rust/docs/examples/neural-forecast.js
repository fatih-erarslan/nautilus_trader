#!/usr/bin/env node

/**
 * Neural Network Forecasting Examples
 *
 * Demonstrates neural network tools for price forecasting.
 */

const { McpServer } = require('@neural-trader/mcp');

async function main() {
  console.log('🧠 Neural Trader MCP - Neural Forecasting Examples\n');

  const server = new McpServer({ transport: 'stdio' });
  await server.start();

  // Example 1: Generate Forecast
  console.log('📈 Example 1: Generate 5-Day Forecast');
  try {
    const forecast = await server.callTool('neural_forecast', {
      symbol: 'AAPL',
      horizon: 5,
      confidenceLevel: 0.95,
      useGpu: true
    });

    console.log('Symbol:', forecast.symbol);
    console.log('Model:', forecast.model_type);
    console.log('Confidence:', (forecast.confidence_level * 100) + '%');
    console.log('\nForecast:');
    forecast.predictions.forEach((pred, i) => {
      console.log(
        `  Day ${i + 1}: $${pred.toFixed(2)} ` +
        `(${forecast.lower_bound[i].toFixed(2)} - ${forecast.upper_bound[i].toFixed(2)})`
      );
    });
    console.log(`\nGeneration time: ${forecast.generation_time_ms}ms`);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 2: List Available Models
  console.log('🤖 Example 2: Available Models');
  const models = await server.callTool('list_model_types', {});
  console.log('Model types:', models.join(', '));
  console.log('');

  // Example 3: Check Model Status
  console.log('📊 Example 3: Model Status');
  try {
    const status = await server.callTool('neural_model_status', {});
    console.log(`Active models: ${status.models.length}`);
    status.models.forEach((model, i) => {
      console.log(`\nModel ${i + 1}:`);
      console.log('  ID:', model.model_id);
      console.log('  Type:', model.model_type);
      console.log('  Status:', model.training_status);
      console.log('  Created:', model.created_at);
      if (model.metrics) {
        console.log('  MAE:', model.metrics.mae?.toFixed(4));
        console.log('  RMSE:', model.metrics.rmse?.toFixed(4));
        console.log('  R²:', model.metrics.r2_score?.toFixed(4));
      }
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 4: Train New Model (simulation)
  console.log('🎓 Example 4: Train Neural Model');
  console.log('Note: This would train a model with real data');
  console.log('Example configuration:');
  const trainingConfig = {
    dataPath: '/data/AAPL_historical.csv',
    modelType: 'NHITS',
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    validationSplit: 0.2,
    useGpu: true
  };
  console.log(JSON.stringify(trainingConfig, null, 2));
  console.log('');

  // Example 5: Neural Backtest
  console.log('📉 Example 5: Neural Strategy Backtest');
  console.log('Note: Requires trained model');
  const backtestConfig = {
    modelId: 'model_20250114_103045',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    benchmark: 'sp500',
    rebalanceFrequency: 'daily',
    useGpu: true
  };
  console.log('Configuration:', JSON.stringify(backtestConfig, null, 2));
  console.log('');

  // Example 6: Optimize Hyperparameters
  console.log('🔧 Example 6: Hyperparameter Optimization');
  console.log('Note: This is a long-running operation');
  const optimizationConfig = {
    modelId: 'model_20250114_103045',
    parameterRanges: {
      learningRate: { min: 0.0001, max: 0.01 },
      hiddenSize: { values: [64, 128, 256] },
      numLayers: { values: [2, 3, 4] },
      dropout: { min: 0.1, max: 0.5 }
    },
    optimizationMetric: 'mae',
    trials: 50,
    useGpu: true
  };
  console.log('Configuration:', JSON.stringify(optimizationConfig, null, 2));
  console.log('');

  // Clean up
  await server.stop();
  console.log('✅ Examples complete');
}

main().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
