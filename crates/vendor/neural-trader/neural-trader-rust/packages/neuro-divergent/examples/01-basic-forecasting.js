/**
 * Example 1: Basic Time Series Forecasting
 *
 * Demonstrates:
 * - Loading time series data
 * - Training an LSTM model
 * - Generating forecasts
 * - Evaluating accuracy
 */

const { NeuralForecaster, models } = require('@neural-trader/neuro-divergent');

async function basicForecasting() {
    console.log('📊 Basic Time Series Forecasting Example\n');

    // 1. Generate synthetic data (replace with your real data)
    const data = generateSyntheticData({
        series: 1,
        points: 1000,
        frequency: 'H'  // Hourly
    });

    console.log(`✅ Data loaded: ${data.y.length} observations`);

    // 2. Create forecaster with LSTM model
    const forecaster = new NeuralForecaster({
        models: [
            new models.LSTM({
                hiddenSize: 128,      // Hidden layer size
                numLayers: 2,         // Number of LSTM layers
                dropout: 0.1,         // Dropout for regularization
                horizon: 24,          // Forecast 24 steps ahead
                inputSize: 48         // Use 48 historical points
            })
        ],
        frequency: 'H',  // Data frequency
        localScalerType: 'standard'  // Z-score normalization
    });

    console.log('✅ Model created: LSTM(hidden=128, layers=2)');

    // 3. Train the model
    console.log('\n🎯 Training model...');
    const startTime = Date.now();

    await forecaster.fit(data, {
        epochs: 100,          // Training epochs
        batchSize: 32,        // Batch size
        learningRate: 0.001,  // Learning rate
        validationSize: 0.2,  // 20% validation split
        earlyStopping: true,  // Enable early stopping
        patience: 10,         // Early stopping patience
        verbose: true         // Show progress
    });

    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(`✅ Training complete in ${trainingTime.toFixed(2)}s`);

    // 4. Generate forecasts
    console.log('\n🔮 Generating forecasts...');
    const forecasts = await forecaster.predict({
        horizon: 24  // Forecast next 24 hours
    });

    console.log(`✅ Forecasts generated: ${forecasts.LSTM.length} predictions`);
    console.log('\nFirst 5 forecasts:');
    for (let i = 0; i < 5; i++) {
        console.log(`  ${forecasts.ds[i]}: ${forecasts.LSTM[i].toFixed(2)}`);
    }

    // 5. Evaluate on test set (if available)
    if (data.test) {
        const metrics = evaluateForecasts(forecasts.LSTM, data.test.y);
        console.log('\n📈 Model Performance:');
        console.log(`  MAE:  ${metrics.mae.toFixed(2)}`);
        console.log(`  RMSE: ${metrics.rmse.toFixed(2)}`);
        console.log(`  MAPE: ${metrics.mape.toFixed(2)}%`);
        console.log(`  R²:   ${metrics.r2.toFixed(4)}`);
    }

    // 6. Save model checkpoint
    console.log('\n💾 Saving model checkpoint...');
    await forecaster.saveCheckpoint('./checkpoints/lstm_model.safetensors');
    console.log('✅ Model saved successfully');

    return { forecaster, forecasts };
}

/**
 * Generate synthetic time series data
 */
function generateSyntheticData({ series, points, frequency }) {
    const data = {
        unique_id: [],
        ds: [],
        y: []
    };

    const startDate = new Date('2024-01-01');

    for (let i = 0; i < points; i++) {
        // Generate date based on frequency
        const date = new Date(startDate);
        date.setHours(date.getHours() + i);

        // Generate value with trend and seasonality
        const trend = i * 0.1;
        const seasonality = 10 * Math.sin(2 * Math.PI * i / 24);  // Daily pattern
        const noise = (Math.random() - 0.5) * 5;
        const value = 100 + trend + seasonality + noise;

        data.unique_id.push(`series_${series}`);
        data.ds.push(date.toISOString().split('T')[0]);
        data.y.push(value);
    }

    return data;
}

/**
 * Evaluate forecast accuracy
 */
function evaluateForecasts(predictions, actuals) {
    const n = Math.min(predictions.length, actuals.length);
    let sumAbsError = 0;
    let sumSqError = 0;
    let sumAbsPercentError = 0;
    let sumActuals = 0;
    let sumPredictions = 0;

    for (let i = 0; i < n; i++) {
        const error = actuals[i] - predictions[i];
        sumAbsError += Math.abs(error);
        sumSqError += error * error;
        sumAbsPercentError += Math.abs(error / actuals[i]) * 100;
        sumActuals += actuals[i];
        sumPredictions += predictions[i];
    }

    const mae = sumAbsError / n;
    const rmse = Math.sqrt(sumSqError / n);
    const mape = sumAbsPercentError / n;

    // Calculate R²
    const meanActual = sumActuals / n;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < n; i++) {
        ssTot += Math.pow(actuals[i] - meanActual, 2);
        ssRes += Math.pow(actuals[i] - predictions[i], 2);
    }
    const r2 = 1 - (ssRes / ssTot);

    return { mae, rmse, mape, r2 };
}

// Run example
if (require.main === module) {
    basicForecasting()
        .then(() => {
            console.log('\n✅ Example completed successfully!');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Error:', error);
            process.exit(1);
        });
}

module.exports = { basicForecasting, generateSyntheticData, evaluateForecasts };
