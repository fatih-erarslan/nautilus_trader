/**
 * Example: Temperature Prediction
 *
 * Demonstrates temperature forecasting with seasonal patterns
 * and weather-aware conformal prediction intervals.
 */

import { EnergyForecaster } from '../src/forecaster';
import { EnergyDomain, ModelType, TimeSeriesPoint } from '../src/types';

async function main() {
  console.log('='.repeat(80));
  console.log('Temperature Prediction with Conformal Intervals');
  console.log('='.repeat(80));
  console.log();

  // Generate synthetic temperature data (°C)
  console.log('Generating synthetic temperature data...');
  const historicalData: TimeSeriesPoint[] = [];
  const baseTemp = 15; // °C average

  for (let day = 0; day < 60; day++) {
    // 2 months of data
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = Date.now() - (60 - day) * 24 * 3600000 + hour * 3600000;

      // Seasonal trend (warming over 2 months)
      const seasonalTrend = (day / 60) * 8;

      // Daily temperature cycle
      const dailyCycle = -5 * Math.cos(((hour - 14) / 24) * 2 * Math.PI); // Peak at 2 PM

      // Weekly pattern (cooler on certain days)
      const weeklyPattern = (day % 7) === 0 ? -2 : 0;

      // Random weather variations
      const weatherNoise = (Math.random() - 0.5) * 3;

      const temperature = baseTemp + seasonalTrend + dailyCycle + weeklyPattern + weatherNoise;

      historicalData.push({
        timestamp,
        value: temperature,
        metadata: {
          hour,
          day,
          season: day < 30 ? 'spring' : 'early-summer'
        }
      });
    }
  }

  console.log(`Generated ${historicalData.length} hourly data points (60 days)`);
  console.log();

  // Create temperature forecaster
  console.log('Initializing temperature forecaster...');
  const forecaster = new EnergyForecaster(EnergyDomain.TEMPERATURE, {
    alpha: 0.1, // 90% confidence intervals
    horizon: 120, // 5-day forecast
    seasonalPeriod: 24, // Daily temperature cycle
    enableAdaptive: true,
    ensembleConfig: {
      models: [
        ModelType.PROPHET, // Good for seasonal patterns
        ModelType.LSTM, // Captures complex patterns
        ModelType.ARIMA,
        ModelType.TRANSFORMER
      ]
    }
  });

  console.log('Training models (Prophet, LSTM, ARIMA, Transformer)...');
  console.log();

  await forecaster.train(historicalData);

  const stats = forecaster.getStats();
  console.log('Training Complete!');
  console.log('-'.repeat(80));
  console.log();

  console.log('Model Performance:');
  stats.ensembleStats.performances.forEach(({ model, performance }) => {
    console.log(`  ${model}:`);
    console.log(`    MAE: ${performance.mae.toFixed(2)}°C`);
    console.log(`    RMSE: ${performance.rmse.toFixed(2)}°C`);
    console.log(`    MAPE: ${performance.mape.toFixed(2)}%`);
  });

  console.log();
  console.log('Seasonal Pattern Detection:');
  if (stats.seasonalPattern) {
    console.log(`  Period: ${stats.seasonalPattern.period} hours`);
    console.log(
      `  Strength: ${(stats.seasonalPattern.strength * 100).toFixed(1)}%`
    );

    // Show daily temperature pattern
    console.log('\n  Typical Daily Pattern:');
    const components = stats.seasonalPattern.components;
    const hours = [0, 6, 12, 18, 23];
    hours.forEach(hour => {
      const component = components[hour];
      console.log(
        `    ${hour.toString().padStart(2, '0')}:00 - ${component >= 0 ? '+' : ''}${component.toFixed(2)}°C`
      );
    });
  }

  console.log();
  console.log('='.repeat(80));
  console.log('Generating 5-day (120-hour) temperature forecast...');
  console.log('='.repeat(80));
  console.log();

  const forecast = await forecaster.forecast(120);

  console.log(`Model selected: ${forecast.modelPerformance.modelName}`);
  console.log(
    `Average interval width: ${forecast.modelPerformance.intervalWidth.toFixed(2)}°C`
  );
  console.log();

  // Daily summary
  console.log('5-Day Temperature Forecast Summary:');
  console.log('-'.repeat(80));
  console.log(
    'Day'.padEnd(8) +
      'Low (°C)'.padStart(12) +
      'High (°C)'.padStart(12) +
      'Avg (°C)'.padStart(12) +
      'Confidence'.padStart(15)
  );
  console.log('-'.repeat(80));

  for (let day = 0; day < 5; day++) {
    const dayForecasts = forecast.forecasts.slice(day * 24, (day + 1) * 24);

    const temps = dayForecasts.map(f => f.pointForecast);
    const low = Math.min(...temps);
    const high = Math.max(...temps);
    const avg = temps.reduce((a, b) => a + b, 0) / temps.length;

    const avgConfidence =
      dayForecasts.reduce((sum, f) => sum + f.confidence, 0) / 24;

    console.log(
      `Day ${day + 1}`.padEnd(8) +
        `${low.toFixed(1)}`.padStart(12) +
        `${high.toFixed(1)}`.padStart(12) +
        `${avg.toFixed(1)}`.padStart(12) +
        `${(avgConfidence * 100).toFixed(1)}%`.padStart(15)
    );
  }

  console.log();
  console.log('Detailed Next 24 Hours:');
  console.log('-'.repeat(80));
  console.log(
    'Hour'.padEnd(6) +
      'Temp (°C)'.padStart(12) +
      'Lower 90%'.padStart(12) +
      'Upper 90%'.padStart(12) +
      'Interval'.padStart(12)
  );
  console.log('-'.repeat(80));

  for (let i = 0; i < 24; i += 3) {
    // Every 3 hours
    const f = forecast.forecasts[i];
    const hour = new Date(f.timestamp).getHours();

    console.log(
      `${hour.toString().padStart(2, '0')}:00`.padEnd(6) +
        `${f.pointForecast.toFixed(1)}`.padStart(12) +
        `${f.interval.lower.toFixed(1)}`.padStart(12) +
        `${f.interval.upper.toFixed(1)}`.padStart(12) +
        `±${(f.interval.width() / 2).toFixed(1)}`.padStart(12)
    );
  }

  console.log();
  console.log('Temperature Extremes:');
  console.log('-'.repeat(80));

  const sortedByTemp = [...forecast.forecasts].sort(
    (a, b) => b.pointForecast - a.pointForecast
  );
  const hottestPeriods = sortedByTemp.slice(0, 3);
  const coldestPeriods = sortedByTemp.slice(-3).reverse();

  console.log('  Hottest periods:');
  hottestPeriods.forEach((f, i) => {
    const hour = forecast.forecasts.indexOf(f);
    const day = Math.floor(hour / 24) + 1;
    const hourOfDay = hour % 24;
    console.log(
      `    ${i + 1}. Day ${day}, ${hourOfDay.toString().padStart(2, '0')}:00 - ` +
        `${f.pointForecast.toFixed(1)}°C [${f.interval.lower.toFixed(1)} - ${f.interval.upper.toFixed(1)}]`
    );
  });

  console.log('\n  Coldest periods:');
  coldestPeriods.forEach((f, i) => {
    const hour = forecast.forecasts.indexOf(f);
    const day = Math.floor(hour / 24) + 1;
    const hourOfDay = hour % 24;
    console.log(
      `    ${i + 1}. Day ${day}, ${hourOfDay.toString().padStart(2, '0')}:00 - ` +
        `${f.pointForecast.toFixed(1)}°C [${f.interval.lower.toFixed(1)} - ${f.interval.upper.toFixed(1)}]`
    );
  });

  console.log();
  console.log('Forecast Quality Metrics:');
  console.log('-'.repeat(80));
  console.log(`  Mean Absolute Error: ${forecast.modelPerformance.mae.toFixed(2)}°C`);
  console.log(`  Root Mean Squared Error: ${forecast.modelPerformance.rmse.toFixed(2)}°C`);
  console.log(`  Mean Absolute Percentage Error: ${forecast.modelPerformance.mape.toFixed(2)}%`);
  console.log(
    `  Expected coverage: ${(forecast.modelPerformance.coverage * 100).toFixed(1)}%`
  );

  // Uncertainty analysis
  const uncertaintyByHorizon = [24, 48, 72, 96, 120].map(horizon => {
    const forecastsAtHorizon = forecast.forecasts.slice(horizon - 24, horizon);
    const avgWidth =
      forecastsAtHorizon.reduce((sum, f) => sum + f.interval.width(), 0) /
      forecastsAtHorizon.length;
    return { horizon, avgWidth };
  });

  console.log();
  console.log('Uncertainty Growth by Forecast Horizon:');
  console.log('-'.repeat(80));
  uncertaintyByHorizon.forEach(({ horizon, avgWidth }) => {
    console.log(`  ${horizon} hours: ±${(avgWidth / 2).toFixed(2)}°C`);
  });

  console.log();
  console.log('='.repeat(80));
  console.log('Temperature Prediction Complete!');
  console.log('='.repeat(80));
}

main().catch(console.error);
