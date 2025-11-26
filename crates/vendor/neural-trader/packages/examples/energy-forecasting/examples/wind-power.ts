/**
 * Example: Wind Power Forecasting
 *
 * Demonstrates wind power forecasting with high variability
 * and conformal prediction uncertainty quantification.
 */

import { EnergyForecaster } from '../src/forecaster';
import { EnergyDomain, ModelType, TimeSeriesPoint } from '../src/types';

async function main() {
  console.log('='.repeat(80));
  console.log('Wind Power Forecasting with Ensemble Models');
  console.log('='.repeat(80));
  console.log();

  // Generate synthetic wind power data (MW)
  console.log('Generating synthetic wind power generation data...');
  const historicalData: TimeSeriesPoint[] = [];
  const installedCapacity = 500; // MW

  for (let day = 0; day < 30; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = Date.now() - (30 - day) * 24 * 3600000 + hour * 3600000;

      // Wind power is more variable and less predictable than solar
      // Simulate wind speed variations
      const baseWindSpeed = 8 + 4 * Math.sin((day / 7) * 2 * Math.PI); // Weekly cycle
      const hourlyVariation = 2 * Math.sin((hour / 24) * 2 * Math.PI);
      const randomGust = (Math.random() - 0.5) * 6;
      const windSpeed = Math.max(0, baseWindSpeed + hourlyVariation + randomGust);

      // Wind power follows cubic relationship with wind speed (simplified)
      let power = 0;
      if (windSpeed > 3 && windSpeed < 25) {
        // Cut-in and cut-out wind speeds
        const efficiency = Math.min(1, Math.pow((windSpeed - 3) / 12, 3));
        power = efficiency * installedCapacity;
      }

      historicalData.push({
        timestamp,
        value: power,
        metadata: {
          hour,
          day,
          windSpeed,
          capacity: installedCapacity
        }
      });
    }
  }

  console.log(`Generated ${historicalData.length} hourly data points (30 days)`);
  console.log();

  // Create forecaster with models suited for high variability
  console.log('Initializing wind power forecaster...');
  const forecaster = new EnergyForecaster(EnergyDomain.WIND, {
    alpha: 0.15, // 85% confidence (wider intervals for high variability)
    horizon: 72, // 72-hour ahead forecast
    seasonalPeriod: 168, // Weekly seasonality for wind patterns
    enableAdaptive: true,
    ensembleConfig: {
      models: [
        ModelType.LSTM, // Good for capturing complex patterns
        ModelType.TRANSFORMER, // Attention mechanism for variable patterns
        ModelType.ARIMA
      ]
    }
  });

  console.log('Training models (LSTM, Transformer, ARIMA)...');
  console.log();

  await forecaster.train(historicalData);

  // Get statistics
  const stats = forecaster.getStats();
  console.log('Training Complete!');
  console.log('-'.repeat(80));
  console.log(`Training points: ${stats.trainingPoints}`);
  console.log(`Models trained: ${stats.ensembleStats.modelCount}`);
  console.log();

  console.log('Model Performance Comparison:');
  stats.ensembleStats.performances.forEach(({ model, performance }) => {
    console.log(`\n  ${model}:`);
    console.log(`    MAPE: ${performance.mape.toFixed(2)}%`);
    console.log(`    RMSE: ${performance.rmse.toFixed(2)} MW`);
    console.log(`    MAE: ${performance.mae.toFixed(2)} MW`);
    console.log(
      `    Coverage: ${(performance.coverage * 100).toFixed(1)}%`
    );
  });

  console.log();
  console.log('='.repeat(80));
  console.log('Generating 72-hour forecast...');
  console.log('='.repeat(80));
  console.log();

  const forecast = await forecaster.forecast(72);

  console.log(`Model used: ${forecast.modelPerformance.modelName}`);
  console.log(
    `Average interval width: ${forecast.modelPerformance.intervalWidth.toFixed(2)} MW`
  );
  console.log();

  // Analyze forecast statistics
  const avgPower =
    forecast.forecasts.reduce((sum, f) => sum + f.pointForecast, 0) /
    forecast.forecasts.length;
  const maxPower = Math.max(...forecast.forecasts.map(f => f.pointForecast));
  const minPower = Math.min(...forecast.forecasts.map(f => f.pointForecast));

  console.log('72-Hour Forecast Summary:');
  console.log('-'.repeat(80));
  console.log(`Average power: ${avgPower.toFixed(2)} MW`);
  console.log(`Maximum power: ${maxPower.toFixed(2)} MW`);
  console.log(`Minimum power: ${minPower.toFixed(2)} MW`);
  console.log(`Capacity factor: ${((avgPower / installedCapacity) * 100).toFixed(1)}%`);

  console.log();
  console.log('Next 24 Hours:');
  console.log('-'.repeat(80));
  console.log(
    'Hour'.padEnd(6) +
      'Forecast (MW)'.padStart(15) +
      'Lower 85%'.padStart(15) +
      'Upper 85%'.padStart(15) +
      'Uncertainty'.padStart(15)
  );
  console.log('-'.repeat(80));

  for (let i = 0; i < 24; i += 3) {
    // Show every 3 hours
    const f = forecast.forecasts[i];
    const uncertainty = (f.interval.width() / f.pointForecast) * 100;

    console.log(
      `${i + 1}`.padEnd(6) +
        `${f.pointForecast.toFixed(2)}`.padStart(15) +
        `${f.interval.lower.toFixed(2)}`.padStart(15) +
        `${f.interval.upper.toFixed(2)}`.padStart(15) +
        `${uncertainty.toFixed(1)}%`.padStart(15)
    );
  }

  console.log();
  console.log('High Variability Periods:');
  const highVariability = forecast.forecasts
    .filter(f => f.interval.width() > avgPower * 0.5)
    .slice(0, 5);

  if (highVariability.length > 0) {
    highVariability.forEach(f => {
      const hour = forecast.forecasts.indexOf(f) + 1;
      console.log(`  Hour ${hour}: Width=${f.interval.width().toFixed(2)} MW`);
    });
  } else {
    console.log('  No periods with exceptionally high variability detected');
  }

  console.log();
  console.log('='.repeat(80));
  console.log('Wind Power Forecasting Complete!');
  console.log('='.repeat(80));
}

main().catch(console.error);
