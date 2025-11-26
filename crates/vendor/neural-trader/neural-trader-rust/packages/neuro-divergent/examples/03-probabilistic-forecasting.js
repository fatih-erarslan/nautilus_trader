/**
 * Example 3: Probabilistic Forecasting with Uncertainty Quantification
 *
 * Demonstrates:
 * - DeepAR probabilistic model
 * - Confidence intervals
 * - Monte Carlo sampling
 * - Quantile predictions
 */

const { NeuralForecaster, models } = require('@neural-trader/neuro-divergent');

async function probabilisticForecasting() {
    console.log('🎲 Probabilistic Forecasting Example\n');

    // 1. Load data
    const data = require('./data-loader').loadTimeSeriesData();
    console.log(`✅ Data loaded: ${data.y.length} observations`);

    // 2. Create forecaster with DeepAR (probabilistic model)
    const forecaster = new NeuralForecaster({
        models: [
            new models.DeepAR({
                name: 'DeepAR',
                hiddenSize: 128,
                numLayers: 2,
                horizon: 24,
                inputSize: 48,
                dropout: 0.1,
                // Probabilistic settings
                likelihood: 'gaussian',  // or 'negative-binomial', 'student-t'
                numSamples: 1000        // Monte Carlo samples
            })
        ],
        frequency: 'H'
    });

    console.log('✅ DeepAR model created');

    // 3. Train the model
    console.log('\n🎯 Training probabilistic model...');
    await forecaster.fit(data, {
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001,
        validationSize: 0.2,
        earlyStopping: true,
        patience: 10,
        verbose: true
    });

    console.log('✅ Training complete');

    // 4. Generate probabilistic forecasts with confidence intervals
    console.log('\n🔮 Generating probabilistic forecasts...');
    const forecasts = await forecaster.predict({
        horizon: 24,
        level: [80, 90, 95, 99],  // Confidence levels
        numSamples: 1000           // MC samples for accurate quantiles
    });

    // 5. Display forecast with uncertainty bands
    console.log('\n📊 Forecast with Confidence Intervals:\n');
    console.log('  Time         Point   80% CI           90% CI           95% CI           99% CI');
    console.log('  ' + '─'.repeat(95));

    for (let i = 0; i < Math.min(10, forecasts.ds.length); i++) {
        const point = forecasts['DeepAR'][i];
        const lo80 = forecasts['DeepAR-lo-80'][i];
        const hi80 = forecasts['DeepAR-hi-80'][i];
        const lo90 = forecasts['DeepAR-lo-90'][i];
        const hi90 = forecasts['DeepAR-hi-90'][i];
        const lo95 = forecasts['DeepAR-lo-95'][i];
        const hi95 = forecasts['DeepAR-hi-95'][i];
        const lo99 = forecasts['DeepAR-lo-99'][i];
        const hi99 = forecasts['DeepAR-hi-99'][i];

        console.log(
            `  ${forecasts.ds[i]}  ${point.toFixed(1).padStart(6)}  ` +
            `[${lo80.toFixed(1)}, ${hi80.toFixed(1)}]  `.padEnd(18) +
            `[${lo90.toFixed(1)}, ${hi90.toFixed(1)}]  `.padEnd(18) +
            `[${lo95.toFixed(1)}, ${hi95.toFixed(1)}]  `.padEnd(18) +
            `[${lo99.toFixed(1)}, ${hi99.toFixed(1)}]`.padEnd(18)
        );
    }

    // 6. Analyze prediction intervals
    console.log('\n📏 Prediction Interval Analysis:');

    const intervals = [80, 90, 95, 99];
    for (const level of intervals) {
        const lo = forecasts[`DeepAR-lo-${level}`];
        const hi = forecasts[`DeepAR-hi-${level}`];

        // Calculate average interval width
        let totalWidth = 0;
        for (let i = 0; i < lo.length; i++) {
            totalWidth += hi[i] - lo[i];
        }
        const avgWidth = totalWidth / lo.length;

        console.log(`  ${level}% CI: Average width = ${avgWidth.toFixed(2)}`);
    }

    // 7. Risk analysis
    console.log('\n⚠️  Risk Analysis:');

    const point = forecasts['DeepAR'];
    const lo95 = forecasts['DeepAR-lo-95'];
    const hi95 = forecasts['DeepAR-hi-95'];

    // Calculate downside risk (5th percentile)
    const downsideRisk = point.map((p, i) => p - lo95[i]);
    const avgDownsideRisk = downsideRisk.reduce((a, b) => a + b, 0) / downsideRisk.length;

    // Calculate upside potential (95th percentile)
    const upsidePotential = point.map((p, i) => hi95[i] - p);
    const avgUpsidePotential = upsidePotential.reduce((a, b) => a + b, 0) / upsidePotential.length;

    console.log(`  Average downside risk (5th %ile): ${avgDownsideRisk.toFixed(2)}`);
    console.log(`  Average upside potential (95th %ile): ${avgUpsidePotential.toFixed(2)}`);
    console.log(`  Risk-reward ratio: ${(avgUpsidePotential / avgDownsideRisk).toFixed(2)}`);

    // 8. Scenario analysis
    console.log('\n🎭 Scenario Analysis:');

    // Get sample paths (if available)
    const samplePaths = await forecaster.getSamplePaths({
        horizon: 24,
        numPaths: 100
    });

    if (samplePaths) {
        const scenarios = {
            'Pessimistic (10th %ile)': calculateQuantile(samplePaths, 0.10),
            'Conservative (25th %ile)': calculateQuantile(samplePaths, 0.25),
            'Expected (50th %ile)': calculateQuantile(samplePaths, 0.50),
            'Optimistic (75th %ile)': calculateQuantile(samplePaths, 0.75),
            'Very Optimistic (90th %ile)': calculateQuantile(samplePaths, 0.90)
        };

        for (const [scenario, values] of Object.entries(scenarios)) {
            const finalValue = values[values.length - 1];
            console.log(`  ${scenario.padEnd(30)}: ${finalValue.toFixed(2)}`);
        }
    }

    // 9. Probability of exceeding threshold
    console.log('\n🎯 Probability Analysis:');

    const threshold = 120;  // Example threshold
    const exceedanceProb = calculateExceedanceProbability(
        forecasts['DeepAR'],
        forecasts['DeepAR-lo-95'],
        forecasts['DeepAR-hi-95'],
        threshold
    );

    console.log(`  P(value > ${threshold}): ${(exceedanceProb * 100).toFixed(1)}%`);

    // 10. Coverage diagnostics
    if (data.test) {
        console.log('\n📊 Coverage Diagnostics:');

        const coverage = calculateCoverage(
            data.test.y,
            forecasts['DeepAR-lo-80'],
            forecasts['DeepAR-hi-80'],
            forecasts['DeepAR-lo-90'],
            forecasts['DeepAR-hi-90'],
            forecasts['DeepAR-lo-95'],
            forecasts['DeepAR-hi-95']
        );

        console.log(`  80% CI actual coverage: ${(coverage[80] * 100).toFixed(1)}% (target: 80%)`);
        console.log(`  90% CI actual coverage: ${(coverage[90] * 100).toFixed(1)}% (target: 90%)`);
        console.log(`  95% CI actual coverage: ${(coverage[95] * 100).toFixed(1)}% (target: 95%)`);

        // Check calibration
        const wellCalibrated = Object.entries(coverage).every(
            ([level, actual]) => Math.abs(actual - level / 100) < 0.05
        );

        if (wellCalibrated) {
            console.log('  ✅ Model is well-calibrated');
        } else {
            console.log('  ⚠️  Model may need recalibration');
        }
    }

    return { forecaster, forecasts, samplePaths };
}

/**
 * Calculate quantile from sample paths
 */
function calculateQuantile(samplePaths, quantile) {
    const numSteps = samplePaths[0].length;
    const result = new Array(numSteps);

    for (let i = 0; i < numSteps; i++) {
        const values = samplePaths.map(path => path[i]).sort((a, b) => a - b);
        const index = Math.floor(values.length * quantile);
        result[i] = values[index];
    }

    return result;
}

/**
 * Calculate probability of exceeding threshold
 */
function calculateExceedanceProbability(point, lo95, hi95, threshold) {
    // Approximate using normal distribution
    let count = 0;
    for (let i = 0; i < point.length; i++) {
        const mean = point[i];
        const std = (hi95[i] - lo95[i]) / (2 * 1.96);  // 95% CI ≈ ±1.96σ

        // Calculate Z-score
        const z = (threshold - mean) / std;

        // P(X > threshold) = 1 - Φ(z)
        const prob = 1 - normalCDF(z);
        count += prob;
    }

    return count / point.length;
}

/**
 * Normal CDF approximation
 */
function normalCDF(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
}

/**
 * Calculate coverage of confidence intervals
 */
function calculateCoverage(actuals, lo80, hi80, lo90, hi90, lo95, hi95) {
    const n = Math.min(actuals.length, lo80.length);
    let count80 = 0;
    let count90 = 0;
    let count95 = 0;

    for (let i = 0; i < n; i++) {
        if (actuals[i] >= lo80[i] && actuals[i] <= hi80[i]) count80++;
        if (actuals[i] >= lo90[i] && actuals[i] <= hi90[i]) count90++;
        if (actuals[i] >= lo95[i] && actuals[i] <= hi95[i]) count95++;
    }

    return {
        80: count80 / n,
        90: count90 / n,
        95: count95 / n
    };
}

// Run example
if (require.main === module) {
    probabilisticForecasting()
        .then(() => {
            console.log('\n✅ Probabilistic forecasting example completed!');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Error:', error);
            process.exit(1);
        });
}

module.exports = { probabilisticForecasting, calculateQuantile, calculateExceedanceProbability };
