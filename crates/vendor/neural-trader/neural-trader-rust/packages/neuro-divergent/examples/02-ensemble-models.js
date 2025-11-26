/**
 * Example 2: Ensemble Forecasting with Multiple Models
 *
 * Demonstrates:
 * - Training multiple models simultaneously
 * - Combining predictions from different architectures
 * - Weighted ensembles
 * - Model comparison
 */

const { NeuralForecaster, models } = require('@neural-trader/neuro-divergent');

async function ensembleForecasting() {
    console.log('🎭 Ensemble Forecasting Example\n');

    // 1. Load or generate data
    const data = require('./data-loader').loadTimeSeriesData();
    console.log(`✅ Data loaded: ${data.y.length} observations`);

    // 2. Create forecaster with multiple models
    const forecaster = new NeuralForecaster({
        models: [
            // Recurrent models
            new models.LSTM({
                name: 'LSTM-128',
                hiddenSize: 128,
                numLayers: 2,
                horizon: 24,
                inputSize: 48
            }),
            new models.GRU({
                name: 'GRU-128',
                hiddenSize: 128,
                numLayers: 2,
                horizon: 24,
                inputSize: 48
            }),

            // Attention-based
            new models.Transformer({
                name: 'Transformer',
                hiddenSize: 256,
                numLayers: 4,
                numHeads: 8,
                horizon: 24,
                inputSize: 96
            }),

            // Specialized architectures
            new models.NBEATS({
                name: 'NBEATS',
                stackTypes: ['trend', 'seasonality'],
                numBlocks: 3,
                horizon: 24,
                inputSize: 96
            }),

            new models.TCN({
                name: 'TCN',
                numChannels: [32, 64, 128],
                kernelSize: 3,
                horizon: 24,
                inputSize: 96
            })
        ],
        frequency: 'H',
        numThreads: 8,       // Train models in parallel
        parallelModels: true
    });

    console.log('✅ Created ensemble with 5 models');

    // 3. Train all models
    console.log('\n🎯 Training ensemble...');
    const startTime = Date.now();

    await forecaster.fit(data, {
        epochs: 50,
        batchSize: 32,
        learningRate: 0.001,
        validationSize: 0.2,
        earlyStopping: true,
        patience: 10,
        verbose: true
    });

    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(`✅ Ensemble trained in ${trainingTime.toFixed(2)}s`);

    // 4. Generate forecasts from all models
    console.log('\n🔮 Generating ensemble forecasts...');
    const forecasts = await forecaster.predict({ horizon: 24 });

    // 5. Individual model predictions
    console.log('\n📊 Individual Model Forecasts:');
    const modelNames = ['LSTM-128', 'GRU-128', 'Transformer', 'NBEATS', 'TCN'];
    for (const modelName of modelNames) {
        if (forecasts[modelName]) {
            const firstForecast = forecasts[modelName][0];
            const lastForecast = forecasts[modelName][forecasts[modelName].length - 1];
            console.log(`  ${modelName.padEnd(15)}: ${firstForecast.toFixed(2)} → ${lastForecast.toFixed(2)}`);
        }
    }

    // 6. Create weighted ensemble
    console.log('\n⚖️  Creating weighted ensemble...');

    // Simple average
    const simpleEnsemble = createSimpleEnsemble(forecasts, modelNames);
    console.log('  Simple average: ✅');

    // Weighted by validation performance
    const weights = await forecaster.getModelWeights();
    const weightedEnsemble = createWeightedEnsemble(forecasts, modelNames, weights);
    console.log('  Weighted by performance: ✅');

    // Median ensemble (robust to outliers)
    const medianEnsemble = createMedianEnsemble(forecasts, modelNames);
    console.log('  Median ensemble: ✅');

    // 7. Compare performance
    if (data.test) {
        console.log('\n📈 Ensemble Performance Comparison:');

        const results = [];
        for (const modelName of modelNames) {
            const metrics = evaluateModel(forecasts[modelName], data.test.y);
            results.push({ model: modelName, ...metrics });
        }

        // Evaluate ensemble methods
        results.push({
            model: 'Simple Ensemble',
            ...evaluateModel(simpleEnsemble, data.test.y)
        });
        results.push({
            model: 'Weighted Ensemble',
            ...evaluateModel(weightedEnsemble, data.test.y)
        });
        results.push({
            model: 'Median Ensemble',
            ...evaluateModel(medianEnsemble, data.test.y)
        });

        // Sort by RMSE (best first)
        results.sort((a, b) => a.rmse - b.rmse);

        console.log('\n  Model              MAE      RMSE     MAPE      R²');
        console.log('  ' + '─'.repeat(60));
        for (const result of results) {
            console.log(
                `  ${result.model.padEnd(15)} ` +
                `${result.mae.toFixed(2).padStart(7)}  ` +
                `${result.rmse.toFixed(2).padStart(7)}  ` +
                `${result.mape.toFixed(2).padStart(6)}%  ` +
                `${result.r2.toFixed(4).padStart(7)}`
            );
        }

        const bestModel = results[0];
        console.log(`\n  🏆 Best model: ${bestModel.model} (RMSE: ${bestModel.rmse.toFixed(2)})`);
    }

    // 8. Model-specific insights
    console.log('\n🔍 Model Insights:');
    const insights = await forecaster.getModelInsights();
    for (const [modelName, insight] of Object.entries(insights)) {
        console.log(`\n  ${modelName}:`);
        console.log(`    Training time: ${insight.trainingTime.toFixed(2)}s`);
        console.log(`    Parameters: ${(insight.numParameters / 1e6).toFixed(2)}M`);
        console.log(`    Memory usage: ${(insight.memoryMB).toFixed(0)}MB`);
        console.log(`    Inference speed: ${insight.predictionsPerSecond.toFixed(0)}/s`);
    }

    return { forecaster, forecasts, simpleEnsemble, weightedEnsemble, medianEnsemble };
}

/**
 * Create simple average ensemble
 */
function createSimpleEnsemble(forecasts, modelNames) {
    const length = forecasts[modelNames[0]].length;
    const ensemble = new Array(length).fill(0);

    for (let i = 0; i < length; i++) {
        let sum = 0;
        for (const modelName of modelNames) {
            sum += forecasts[modelName][i];
        }
        ensemble[i] = sum / modelNames.length;
    }

    return ensemble;
}

/**
 * Create weighted ensemble based on model performance
 */
function createWeightedEnsemble(forecasts, modelNames, weights) {
    const length = forecasts[modelNames[0]].length;
    const ensemble = new Array(length).fill(0);

    // Normalize weights
    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);

    for (let i = 0; i < length; i++) {
        let weightedSum = 0;
        for (const modelName of modelNames) {
            const weight = (weights[modelName] || 1) / totalWeight;
            weightedSum += forecasts[modelName][i] * weight;
        }
        ensemble[i] = weightedSum;
    }

    return ensemble;
}

/**
 * Create median ensemble (robust to outliers)
 */
function createMedianEnsemble(forecasts, modelNames) {
    const length = forecasts[modelNames[0]].length;
    const ensemble = new Array(length);

    for (let i = 0; i < length; i++) {
        const values = modelNames.map(name => forecasts[name][i]).sort((a, b) => a - b);
        const mid = Math.floor(values.length / 2);
        ensemble[i] = values.length % 2 === 0
            ? (values[mid - 1] + values[mid]) / 2
            : values[mid];
    }

    return ensemble;
}

/**
 * Evaluate model performance
 */
function evaluateModel(predictions, actuals) {
    const n = Math.min(predictions.length, actuals.length);
    let sumAbsError = 0;
    let sumSqError = 0;
    let sumAbsPercentError = 0;
    let sumActuals = 0;

    for (let i = 0; i < n; i++) {
        const error = actuals[i] - predictions[i];
        sumAbsError += Math.abs(error);
        sumSqError += error * error;
        sumAbsPercentError += Math.abs(error / actuals[i]) * 100;
        sumActuals += actuals[i];
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
    ensembleForecasting()
        .then(() => {
            console.log('\n✅ Ensemble example completed successfully!');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Error:', error);
            process.exit(1);
        });
}

module.exports = { ensembleForecasting, createSimpleEnsemble, createWeightedEnsemble };
