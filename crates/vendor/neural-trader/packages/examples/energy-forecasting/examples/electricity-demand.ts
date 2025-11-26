/**
 * Example: Electricity Demand Forecasting
 *
 * Demonstrates electricity demand forecasting with multiple
 * seasonal patterns (daily, weekly) and trend components.
 */

import { EnergyForecaster } from '../src/forecaster';
import { EnergyDomain, ModelType, TimeSeriesPoint } from '../src/types';

async function main() {
  console.log('='.repeat(80));
  console.log('Electricity Demand Forecasting with Multi-Seasonal Patterns');
  console.log('='.repeat(80));
  console.log();

  // Generate synthetic electricity demand data (MWh)
  console.log('Generating synthetic electricity demand data...');
  const historicalData: TimeSeriesPoint[] = [];
  const baseLoad = 1000; // MWh

  for (let day = 0; day < 30; day++) {
    const isWeekend = day % 7 >= 5; // Saturday and Sunday

    for (let hour = 0; hour < 24; hour++) {
      const timestamp = Date.now() - (30 - day) * 24 * 3600000 + hour * 3600000;

      // Base demand with linear growth trend
      const trend = day * 2; // 2 MWh/day growth

      // Daily pattern: peaks in morning (8 AM) and evening (7 PM)
      let dailyPattern = 0;
      if (hour >= 6 && hour <= 9) {
        // Morning peak
        dailyPattern = 200 * Math.sin(((hour - 6) / 3) * Math.PI);
      } else if (hour >= 17 && hour <= 22) {
        // Evening peak
        dailyPattern = 250 * Math.sin(((hour - 17) / 5) * Math.PI);
      } else if (hour >= 0 && hour <= 5) {
        // Night low
        dailyPattern = -150;
      }

      // Weekly pattern: lower demand on weekends
      const weeklyPattern = isWeekend ? -200 : 0;

      // Random noise
      const noise = (Math.random() - 0.5) * 50;

      const demand = baseLoad + trend + dailyPattern + weeklyPattern + noise;

      historicalData.push({
        timestamp,
        value: Math.max(0, demand),
        metadata: {
          hour,
          day,
          isWeekend,
          baseLoad
        }
      });
    }
  }

  console.log(`Generated ${historicalData.length} hourly data points (30 days)`);
  console.log();

  // Create forecaster optimized for demand patterns
  console.log('Initializing demand forecaster...');
  const forecaster = new EnergyForecaster(EnergyDomain.DEMAND, {
    alpha: 0.1, // 90% confidence intervals
    horizon: 168, // 1-week ahead forecast
    seasonalPeriod: 24, // Primary daily seasonality
    enableAdaptive: true,
    ensembleConfig: {
      models: [
        ModelType.PROPHET, // Excellent for multiple seasonalities
        ModelType.ARIMA,
        ModelType.LSTM,
        ModelType.TRANSFORMER
      ]
    }
  });

  console.log('Training ensemble models...');
  console.log('Models: Prophet, ARIMA, LSTM, Transformer');
  console.log();

  await forecaster.train(historicalData);

  const stats = forecaster.getStats();
  console.log('Training Complete!');
  console.log('-'.repeat(80));
  console.log();

  console.log('Detected Patterns:');
  if (stats.seasonalPattern) {
    console.log(`  Daily seasonality: Period=${stats.seasonalPattern.period}h`);
    console.log(
      `  Pattern strength: ${(stats.seasonalPattern.strength * 100).toFixed(1)}%`
    );
  }

  console.log();
  console.log('Model Performance:');
  stats.ensembleStats.performances.forEach(({ model, performance }) => {
    console.log(`  ${model}: MAPE=${performance.mape.toFixed(2)}%, RMSE=${performance.rmse.toFixed(2)} MWh`);
  });

  console.log();
  console.log('='.repeat(80));
  console.log('Generating 1-week (168-hour) forecast...');
  console.log('='.repeat(80));
  console.log();

  const forecast = await forecaster.forecast(168);

  console.log(`Selected model: ${forecast.modelPerformance.modelName}`);
  console.log();

  // Analyze daily patterns
  console.log('Daily Demand Patterns:');
  console.log('-'.repeat(80));

  for (let day = 0; day < 7; day++) {
    const dayForecasts = forecast.forecasts.slice(day * 24, (day + 1) * 24);
    const dayName = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][
      (new Date().getDay() + day) % 7
    ];

    const avgDemand =
      dayForecasts.reduce((sum, f) => sum + f.pointForecast, 0) / 24;
    const peakDemand = Math.max(...dayForecasts.map(f => f.pointForecast));
    const minDemand = Math.min(...dayForecasts.map(f => f.pointForecast));

    const peakHour = dayForecasts.findIndex(
      f => f.pointForecast === peakDemand
    );
    const minHour = dayForecasts.findIndex(f => f.pointForecast === minDemand);

    console.log(`\n  ${dayName} (Day ${day + 1}):`);
    console.log(`    Average: ${avgDemand.toFixed(2)} MWh`);
    console.log(`    Peak: ${peakDemand.toFixed(2)} MWh at hour ${peakHour}`);
    console.log(`    Min: ${minDemand.toFixed(2)} MWh at hour ${minHour}`);
  }

  console.log();
  console.log('Weekday vs Weekend Analysis:');
  console.log('-'.repeat(80));

  const weekdayForecasts = forecast.forecasts.slice(24, 24 * 6); // Mon-Fri
  const weekendForecasts = [
    ...forecast.forecasts.slice(0, 24), // Sun
    ...forecast.forecasts.slice(24 * 6, 24 * 7) // Sat
  ];

  const avgWeekday =
    weekdayForecasts.reduce((sum, f) => sum + f.pointForecast, 0) /
    weekdayForecasts.length;
  const avgWeekend =
    weekendForecasts.reduce((sum, f) => sum + f.pointForecast, 0) /
    weekendForecasts.length;

  console.log(`  Weekday average: ${avgWeekday.toFixed(2)} MWh`);
  console.log(`  Weekend average: ${avgWeekend.toFixed(2)} MWh`);
  console.log(
    `  Difference: ${((avgWeekday - avgWeekend) / avgWeekday * 100).toFixed(1)}%`
  );

  console.log();
  console.log('Peak Demand Periods:');
  console.log('-'.repeat(80));

  const sortedForecasts = [...forecast.forecasts].sort(
    (a, b) => b.pointForecast - a.pointForecast
  );
  const topPeaks = sortedForecasts.slice(0, 5);

  topPeaks.forEach((peak, i) => {
    const hour = forecast.forecasts.indexOf(peak);
    const day = Math.floor(hour / 24);
    const hourOfDay = hour % 24;
    const dayName = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][
      (new Date().getDay() + day) % 7
    ];

    console.log(
      `  ${i + 1}. ${dayName} ${hourOfDay}:00 - ${peak.pointForecast.toFixed(2)} MWh ` +
        `[${peak.interval.lower.toFixed(2)} - ${peak.interval.upper.toFixed(2)}]`
    );
  });

  console.log();
  console.log('Forecast Uncertainty:');
  console.log('-'.repeat(80));
  const avgWidth =
    forecast.forecasts.reduce((sum, f) => sum + f.interval.width(), 0) /
    forecast.forecasts.length;
  const avgRelativeWidth =
    forecast.forecasts.reduce((sum, f) => sum + f.interval.relativeWidth(), 0) /
    forecast.forecasts.length;

  console.log(`  Average interval width: ${avgWidth.toFixed(2)} MWh`);
  console.log(`  Average relative width: ${avgRelativeWidth.toFixed(2)}%`);
  console.log(
    `  Coverage confidence: ${(forecast.forecasts[0].confidence * 100).toFixed(1)}%`
  );

  console.log();
  console.log('='.repeat(80));
  console.log('Electricity Demand Forecasting Complete!');
  console.log('='.repeat(80));
}

main().catch(console.error);
