#!/usr/bin/env node

/**
 * Basic forecasting example using @neural-trader/neuro-divergent
 *
 * This example demonstrates:
 * - Creating a forecast instance
 * - Adding a model
 * - Training on synthetic data
 * - Making predictions
 */

const { NeuralForecast } = require('..');

async function main() {
  console.log('📊 Basic Neural Forecasting Example\n');

  // Create forecast engine
  console.log('1. Creating NeuralForecast instance...');
  const forecast = new NeuralForecast();
  console.log('✅ Forecast instance created\n');

  // Configure and add a model
  console.log('2. Adding LSTM model...');
  const modelId = await forecast.addModel({
    modelType: 'LSTM',
    inputSize: 20,      // Use 20 past values
    horizon: 10,        // Predict 10 future values
    hiddenSize: 64,     // 64 hidden units
    numLayers: 2,       // 2 LSTM layers
    dropout: 0.1,       // 10% dropout
    learningRate: 0.001 // Learning rate
  });
  console.log(`✅ Model added with ID: ${modelId}\n`);

  // Generate synthetic time series data (sine wave with noise)
  console.log('3. Generating synthetic data...');
  const baseDate = new Date('2024-01-01T00:00:00Z');
  const dataPoints = 100;

  const data = {
    points: Array.from({ length: dataPoints }, (_, i) => {
      const value =
        100 +                           // baseline
        20 * Math.sin(i * 0.1) +       // seasonal pattern
        5 * Math.sin(i * 0.3) +        // weekly pattern
        (Math.random() - 0.5) * 10;    // noise

      return {
        timestamp: new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000).toISOString(),
        value: value
      };
    }),
    frequency: '1D'
  };

  console.log(`✅ Created ${dataPoints} data points`);
  console.log(`   Range: ${Math.min(...data.points.map(p => p.value)).toFixed(2)} to ${Math.max(...data.points.map(p => p.value)).toFixed(2)}\n`);

  // Train the model
  console.log('4. Training model (this may take a moment)...');
  const startTime = Date.now();
  const metrics = await forecast.fit(modelId, data);
  const trainTime = Date.now() - startTime;

  console.log(`✅ Training complete in ${trainTime}ms`);
  console.log(`   Epochs: ${metrics.length}`);
  console.log(`   Initial loss: ${metrics[0].trainLoss.toFixed(4)}`);
  console.log(`   Final loss: ${metrics[metrics.length - 1].trainLoss.toFixed(4)}\n`);

  // Make predictions
  console.log('5. Making predictions...');
  const predictions = await forecast.predict(modelId, 10);

  console.log('✅ Predictions generated:');
  console.log('\n   Time                      Predicted Value');
  console.log('   ─────────────────────     ───────────────');
  predictions.predictions.forEach((pred, i) => {
    console.log(`   ${predictions.timestamps[i]}     ${pred.toFixed(2)}`);
  });

  // Perform cross-validation
  console.log('\n6. Running cross-validation...');
  const cvResults = await forecast.crossValidation(modelId, data, 5, 1);

  console.log('✅ Cross-validation results:');
  console.log(`   MAE:  ${cvResults.mae.toFixed(4)}`);
  console.log(`   MSE:  ${cvResults.mse.toFixed(4)}`);
  console.log(`   RMSE: ${cvResults.rmse.toFixed(4)}`);
  console.log(`   MAPE: ${cvResults.mape.toFixed(2)}%`);

  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('✅ Example completed successfully!');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
}

main().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
