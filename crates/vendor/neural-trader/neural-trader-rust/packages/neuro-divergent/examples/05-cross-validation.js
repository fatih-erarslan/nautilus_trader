/**
 * Example 5: Cross-Validation and Model Selection
 *
 * Demonstrates:
 * - Time series cross-validation
 * - Multiple model comparison
 * - Hyperparameter tuning
 * - Best model selection
 */

const { NeuralForecaster, models } = require('@neural-trader/neuro-divergent');

async function crossValidationExample() {
    console.log('🔬 Cross-Validation and Model Selection Example\n');

    // 1. Load data
    const data = require('./data-loader').loadTimeSeriesData();
    console.log(`✅ Data loaded: ${data.y.length} observations`);

    // 2. Define model candidates
    const modelCandidates = [
        {
            name: 'LSTM-Small',
            model: new models.LSTM({
                hiddenSize: 64,
                numLayers: 1,
                horizon: 24,
                inputSize: 48
            })
        },
        {
            name: 'LSTM-Medium',
            model: new models.LSTM({
                hiddenSize: 128,
                numLayers: 2,
                horizon: 24,
                inputSize: 48
            })
        },
        {
            name: 'LSTM-Large',
            model: new models.LSTM({
                hiddenSize: 256,
                numLayers: 3,
                horizon: 24,
                inputSize: 48
            })
        },
        {
            name: 'GRU-Medium',
            model: new models.GRU({
                hiddenSize: 128,
                numLayers: 2,
                horizon: 24,
                inputSize: 48
            })
        },
        {
            name: 'Transformer',
            model: new models.Transformer({
                hiddenSize: 256,
                numLayers: 4,
                numHeads: 8,
                horizon: 24,
                inputSize: 96
            })
        },
        {
            name: 'NBEATS',
            model: new models.NBEATS({
                stackTypes: ['trend', 'seasonality'],
                numBlocks: 3,
                horizon: 24,
                inputSize: 96
            })
        }
    ];

    console.log(`📊 Testing ${modelCandidates.length} model configurations`);

    // 3. Cross-validation settings
    const cvConfig = {
        nWindows: 5,      // 5-fold CV
        step: 24,         // Step size (24 hours)
        horizon: 24       // Forecast horizon
    };

    console.log(`\n⚙️  CV Configuration:`);
    console.log(`   Windows: ${cvConfig.nWindows}`);
    console.log(`   Step: ${cvConfig.step}`);
    console.log(`   Horizon: ${cvConfig.horizon}`);

    // 4. Perform cross-validation for each model
    console.log('\n🔄 Running cross-validation...\n');

    const results = [];

    for (const { name, model } of modelCandidates) {
        console.log(`📈 Testing ${name}...`);

        const forecaster = new NeuralForecaster({
            models: [model],
            frequency: 'H'
        });

        try {
            // Train model
            await forecaster.fit(data, {
                epochs: 50,
                batchSize: 32,
                learningRate: 0.001,
                validationSize: 0.2,
                earlyStopping: true,
                patience: 10,
                verbose: false
            });

            // Perform cross-validation
            const cvResults = await forecaster.crossValidation(cvConfig);

            // Calculate metrics
            const metrics = calculateMetrics(cvResults);

            results.push({
                name,
                metrics,
                cvResults
            });

            console.log(`   ✅ MAE: ${metrics.mae.toFixed(2)}, RMSE: ${metrics.rmse.toFixed(2)}`);

        } catch (error) {
            console.error(`   ❌ Error: ${error.message}`);
            results.push({
                name,
                error: error.message
            });
        }
    }

    // 5. Compare results
    console.log('\n📊 Cross-Validation Results:\n');
    console.log('  Model            MAE      RMSE     MAPE     R²       Time(s)');
    console.log('  ' + '─'.repeat(70));

    // Sort by RMSE (ascending)
    results.sort((a, b) => {
        if (!a.metrics || !b.metrics) return 0;
        return a.metrics.rmse - b.metrics.rmse;
    });

    for (const result of results) {
        if (result.error) {
            console.log(`  ${result.name.padEnd(15)} ERROR: ${result.error}`);
        } else {
            const m = result.metrics;
            console.log(
                `  ${result.name.padEnd(15)} ` +
                `${m.mae.toFixed(2).padStart(7)}  ` +
                `${m.rmse.toFixed(2).padStart(7)}  ` +
                `${m.mape.toFixed(2).padStart(6)}% ` +
                `${m.r2.toFixed(4).padStart(7)}  ` +
                `${(m.time || 0).toFixed(1).padStart(7)}`
            );
        }
    }

    // 6. Select best model
    const bestModel = results[0];
    console.log(`\n🏆 Best Model: ${bestModel.name}`);
    console.log(`   MAE: ${bestModel.metrics.mae.toFixed(2)}`);
    console.log(`   RMSE: ${bestModel.metrics.rmse.toFixed(2)}`);
    console.log(`   MAPE: ${bestModel.metrics.mape.toFixed(2)}%`);
    console.log(`   R²: ${bestModel.metrics.r2.toFixed(4)}`);

    // 7. Statistical analysis
    console.log('\n📈 Statistical Analysis:');

    // Calculate confidence intervals
    const ci = calculateConfidenceInterval(bestModel.cvResults.errors, 0.95);
    console.log(`   95% CI for MAE: [${ci.lower.toFixed(2)}, ${ci.upper.toFixed(2)}]`);

    // Test statistical significance
    if (results.length >= 2) {
        const secondBest = results[1];
        const pValue = tTest(
            bestModel.cvResults.errors,
            secondBest.cvResults.errors
        );

        console.log(`\n🔍 ${bestModel.name} vs ${secondBest.name}:`);
        console.log(`   p-value: ${pValue.toFixed(4)}`);
        console.log(`   Significantly better: ${pValue < 0.05 ? '✅ Yes' : '❌ No'}`);
    }

    // 8. Hyperparameter sensitivity analysis
    console.log('\n🎛️  Hyperparameter Sensitivity:');

    const lstmResults = results.filter(r => r.name.startsWith('LSTM'));
    if (lstmResults.length >= 3) {
        console.log('\n   Hidden Size Impact:');
        for (const result of lstmResults) {
            const size = result.name.includes('Small') ? 64 :
                        result.name.includes('Medium') ? 128 : 256;
            console.log(`   ${size}: RMSE = ${result.metrics.rmse.toFixed(2)}`);
        }
    }

    return { results, bestModel };
}

/**
 * Calculate comprehensive metrics from CV results
 */
function calculateMetrics(cvResults) {
    const errors = [];
    const absoluteErrors = [];
    const squaredErrors = [];
    const percentErrors = [];

    for (let i = 0; i < cvResults.actuals.length; i++) {
        const error = cvResults.actuals[i] - cvResults.predictions[i];
        const absError = Math.abs(error);
        const sqError = error * error;
        const pctError = Math.abs(error / cvResults.actuals[i]) * 100;

        errors.push(error);
        absoluteErrors.push(absError);
        squaredErrors.push(sqError);
        percentErrors.push(pctError);
    }

    const n = errors.length;
    const mae = absoluteErrors.reduce((a, b) => a + b, 0) / n;
    const rmse = Math.sqrt(squaredErrors.reduce((a, b) => a + b, 0) / n);
    const mape = percentErrors.reduce((a, b) => a + b, 0) / n;

    // Calculate R²
    const meanActual = cvResults.actuals.reduce((a, b) => a + b, 0) / n;
    const ssTot = cvResults.actuals.reduce(
        (sum, val) => sum + Math.pow(val - meanActual, 2),
        0
    );
    const ssRes = squaredErrors.reduce((a, b) => a + b, 0);
    const r2 = 1 - (ssRes / ssTot);

    return { mae, rmse, mape, r2, errors };
}

/**
 * Calculate confidence interval using t-distribution
 */
function calculateConfidenceInterval(data, confidence) {
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;

    const variance = data.reduce(
        (sum, val) => sum + Math.pow(val - mean, 2),
        0
    ) / (n - 1);

    const std = Math.sqrt(variance);
    const se = std / Math.sqrt(n);

    // t-value for 95% CI with n-1 degrees of freedom (approximation)
    const tValue = 1.96;  // For large n

    return {
        lower: mean - tValue * se,
        upper: mean + tValue * se
    };
}

/**
 * Welch's t-test for unequal variances
 */
function tTest(sample1, sample2) {
    const n1 = sample1.length;
    const n2 = sample2.length;

    const mean1 = sample1.reduce((a, b) => a + b, 0) / n1;
    const mean2 = sample2.reduce((a, b) => a + b, 0) / n2;

    const var1 = sample1.reduce(
        (sum, val) => sum + Math.pow(val - mean1, 2),
        0
    ) / (n1 - 1);

    const var2 = sample2.reduce(
        (sum, val) => sum + Math.pow(val - mean2, 2),
        0
    ) / (n2 - 1);

    const se = Math.sqrt(var1 / n1 + var2 / n2);
    const tStat = Math.abs((mean1 - mean2) / se);

    // Degrees of freedom (Welch-Satterthwaite)
    const df = Math.pow(var1 / n1 + var2 / n2, 2) /
               (Math.pow(var1 / n1, 2) / (n1 - 1) +
                Math.pow(var2 / n2, 2) / (n2 - 1));

    // Approximate p-value (simplified)
    const pValue = 2 * (1 - tCDF(tStat, df));

    return pValue;
}

/**
 * Student's t-distribution CDF (approximation)
 */
function tCDF(t, df) {
    const x = df / (t * t + df);
    return 1 - 0.5 * betaIncomplete(df / 2, 0.5, x);
}

/**
 * Incomplete beta function (approximation)
 */
function betaIncomplete(a, b, x) {
    // Simplified approximation
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return Math.pow(x, a) * Math.pow(1 - x, b);
}

// Run example
if (require.main === module) {
    crossValidationExample()
        .then(() => {
            console.log('\n✅ Cross-validation example completed!');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Error:', error);
            process.exit(1);
        });
}

module.exports = { crossValidationExample, calculateMetrics, tTest };
