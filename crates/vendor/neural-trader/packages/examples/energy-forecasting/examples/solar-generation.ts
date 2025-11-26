/**
 * Example: Solar Generation Forecasting
 *
 * Demonstrates multi-step ahead forecasting for solar power generation
 * with conformal prediction intervals and ensemble model selection.
 */

import { EnergyForecaster } from '../src/forecaster';
import { EnergyDomain, ModelType, TimeSeriesPoint } from '../src/types';

async function main() {
  console.log('='.repeat(80));
  console.log('Solar Generation Forecasting with Conformal Prediction');
  console.log('='.repeat(80));
  console.log();

  // Generate synthetic solar generation data (kW)
  // Pattern: Daily cycle with peak at noon, zero at night
  console.log('Generating synthetic solar generation data...');
  const historicalData: TimeSeriesPoint[] = [];
  const installedCapacity = 1000; // kW

  for (let day = 0; day < 30; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = Date.now() - (30 - day) * 24 * 3600000 + hour * 3600000;

      // Solar generation model
      let generation = 0;
      if (hour >= 6 && hour <= 18) {
        // Daylight hours (6 AM to 6 PM)
        const solarAngle = ((hour - 12) / 6) * Math.PI;
        const baseGeneration = Math.cos(solarAngle) * installedCapacity;

        // Add weather variability (cloud cover)
        const cloudiness = 0.8 + Math.random() * 0.2;
        generation = Math.max(0, baseGeneration * cloudiness);
      }

      // Add small random noise
      generation += (Math.random() - 0.5) * 10;
      generation = Math.max(0, generation);

      historicalData.push({
        timestamp,
        value: generation,
        metadata: {
          hour,
          day,
          capacity: installedCapacity
        }
      });
    }
  }

  console.log(`Generated ${historicalData.length} hourly data points (30 days)`);
  console.log();

  // Create and configure forecaster
  console.log('Initializing solar forecaster...');
  const forecaster = new EnergyForecaster(EnergyDomain.SOLAR, {
    alpha: 0.1, // 90% confidence intervals
    horizon: 48, // 48-hour ahead forecast
    seasonalPeriod: 24, // Daily seasonality
    enableAdaptive: true,
    ensembleConfig: {
      models: [
        ModelType.ARIMA,
        ModelType.LSTM,
        ModelType.PROPHET,
        ModelType.TRANSFORMER
      ]
    }
  });

  // Train the forecaster
  console.log('Training ensemble models with swarm exploration...');
  console.log('Models: ARIMA, LSTM, Prophet, Transformer');
  console.log();

  const trainStart = Date.now();
  await forecaster.train(historicalData);
  const trainDuration = Date.now() - trainStart;

  console.log(`Training completed in ${(trainDuration / 1000).toFixed(2)}s`);
  console.log();

  // Display training statistics
  const stats = forecaster.getStats();
  console.log('Training Statistics:');
  console.log('-'.repeat(80));
  console.log(`Domain: ${stats.domain}`);
  console.log(`Training points: ${stats.trainingPoints}`);
  console.log(`Ensemble models: ${stats.ensembleStats.modelCount}`);

  if (stats.seasonalPattern) {
    console.log(
      `Seasonal pattern detected: Period=${stats.seasonalPattern.period}h, Strength=${(
        stats.seasonalPattern.strength * 100
      ).toFixed(1)}%`
    );
  }

  console.log();
  console.log('Model Performance:');
  stats.ensembleStats.performances.forEach(({ model, performance }) => {
    console.log(`  ${model}:`);
    console.log(`    MAPE: ${performance.mape.toFixed(2)}%`);
    console.log(`    RMSE: ${performance.rmse.toFixed(2)} kW`);
    console.log(`    MAE: ${performance.mae.toFixed(2)} kW`);
  });

  console.log();
  console.log('Conformal Predictor Statistics:');
  stats.conformalStats.forEach(({ modelType, stats: cstats }) => {
    console.log(`  ${modelType}:`);
    console.log(`    Implementation: ${cstats.implementationType}`);
    console.log(`    Calibration points: ${cstats.calibrationPoints}`);
    console.log(`    Alpha: ${cstats.alpha}`);
  });

  console.log();
  console.log('='.repeat(80));
  console.log('Generating 48-hour ahead forecast...');
  console.log('='.repeat(80));
  console.log();

  // Generate forecast
  const forecast = await forecaster.forecast(48);

  console.log(`Forecast generated at: ${new Date(forecast.generatedAt).toISOString()}`);
  console.log(`Selected model: ${forecast.modelPerformance.modelName}`);
  console.log(
    `Expected coverage: ${(forecast.forecasts[0].confidence * 100).toFixed(1)}%`
  );
  console.log();

  // Display first 24 hours of forecast
  console.log('First 24 Hours Forecast:');
  console.log('-'.repeat(80));
  console.log(
    'Time'.padEnd(20) +
      'Point'.padStart(12) +
      'Lower'.padStart(12) +
      'Upper'.padStart(12) +
      'Width'.padStart(12)
  );
  console.log('-'.repeat(80));

  for (let i = 0; i < 24; i++) {
    const f = forecast.forecasts[i];
    const time = new Date(f.timestamp).toISOString().substr(11, 5);
    console.log(
      `Hour ${i + 1} (${time})`.padEnd(20) +
        `${f.pointForecast.toFixed(2)}`.padStart(12) +
        `${f.interval.lower.toFixed(2)}`.padStart(12) +
        `${f.interval.upper.toFixed(2)}`.padStart(12) +
        `${f.interval.width().toFixed(2)}`.padStart(12)
    );
  }

  console.log();
  console.log('Peak Generation Forecast:');
  const peakForecast = forecast.forecasts.reduce((max, f) =>
    f.pointForecast > max.pointForecast ? f : max
  );
  console.log(`  Time: ${new Date(peakForecast.timestamp).toISOString()}`);
  console.log(`  Expected: ${peakForecast.pointForecast.toFixed(2)} kW`);
  console.log(
    `  Range: ${peakForecast.interval.lower.toFixed(2)} - ${peakForecast.interval.upper.toFixed(2)} kW`
  );
  console.log(
    `  Confidence: ${(peakForecast.confidence * 100).toFixed(1)}%`
  );

  console.log();
  console.log('Night Period (Zero Generation Expected):');
  const nightForecasts = forecast.forecasts.filter(
    f => new Date(f.timestamp).getHours() >= 20 || new Date(f.timestamp).getHours() <= 5
  );
  if (nightForecasts.length > 0) {
    const avgNight =
      nightForecasts.reduce((sum, f) => sum + f.pointForecast, 0) /
      nightForecasts.length;
    console.log(`  Average night forecast: ${avgNight.toFixed(2)} kW`);
  }

  console.log();
  console.log('Model Performance Metrics:');
  console.log(`  MAPE: ${forecast.modelPerformance.mape.toFixed(2)}%`);
  console.log(`  RMSE: ${forecast.modelPerformance.rmse.toFixed(2)} kW`);
  console.log(`  MAE: ${forecast.modelPerformance.mae.toFixed(2)} kW`);
  console.log(`  Coverage: ${(forecast.modelPerformance.coverage * 100).toFixed(1)}%`);
  console.log(
    `  Avg interval width: ${forecast.modelPerformance.intervalWidth.toFixed(2)} kW`
  );

  console.log();
  console.log('='.repeat(80));
  console.log('Solar Generation Forecasting Complete!');
  console.log('='.repeat(80));
}

main().catch(console.error);
