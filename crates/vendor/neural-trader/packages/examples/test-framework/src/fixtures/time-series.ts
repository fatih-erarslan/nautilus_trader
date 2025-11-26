/**
 * Time series data fixtures for testing
 */

import { TimeSeriesData } from '../types';

/**
 * Generate synthetic time series with known patterns
 */
export function generateTimeSeriesWithPattern(
  count: number,
  pattern: 'linear' | 'sine' | 'exponential' | 'random' | 'seasonal',
  options: {
    amplitude?: number;
    frequency?: number;
    noise?: number;
    trend?: number;
  } = {}
): TimeSeriesData {
  const {
    amplitude = 10,
    frequency = 0.1,
    noise = 1,
    trend = 0
  } = options;

  const timestamps: number[] = [];
  const values: number[] = [];
  let baseTime = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    timestamps.push(baseTime + i * 3600000);

    let value = 0;
    switch (pattern) {
      case 'linear':
        value = i * trend;
        break;
      case 'sine':
        value = amplitude * Math.sin(2 * Math.PI * frequency * i);
        break;
      case 'exponential':
        value = Math.exp(trend * i);
        break;
      case 'seasonal':
        value = amplitude * Math.sin(2 * Math.PI * i / 24) + i * trend;
        break;
      case 'random':
      default:
        value = Math.random() * amplitude;
    }

    // Add noise
    value += (Math.random() - 0.5) * noise * 2;
    values.push(value);
  }

  return { timestamps, values };
}

/**
 * Generate multivariate time series
 */
export function generateMultivariateTimeSeries(
  count: number,
  features: string[],
  correlations?: number[][]
): TimeSeriesData {
  const timestamps: number[] = [];
  const featureData: Record<string, number[]> = {};

  // Initialize feature arrays
  features.forEach(feature => {
    featureData[feature] = [];
  });

  let baseTime = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    timestamps.push(baseTime + i * 3600000);

    // Generate correlated features
    const baseValues = features.map(() => Math.random());

    features.forEach((feature, idx) => {
      let value = baseValues[idx];

      // Apply correlations if provided
      if (correlations && correlations[idx]) {
        value = correlations[idx].reduce((sum, corr, j) => {
          return sum + corr * baseValues[j];
        }, 0) / correlations[idx].length;
      }

      featureData[feature].push(value * 100);
    });
  }

  return {
    timestamps,
    values: featureData[features[0]], // Primary feature
    features: featureData
  };
}

/**
 * Generate time series with anomalies
 */
export function generateTimeSeriesWithAnomalies(
  count: number,
  anomalyRate = 0.05,
  anomalyMagnitude = 5
): TimeSeriesData {
  const timestamps: number[] = [];
  const values: number[] = [];
  let baseTime = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    timestamps.push(baseTime + i * 3600000);

    // Generate normal value
    let value = 50 + 10 * Math.sin(2 * Math.PI * i / 24) + (Math.random() - 0.5) * 2;

    // Inject anomalies
    if (Math.random() < anomalyRate) {
      value += (Math.random() > 0.5 ? 1 : -1) * anomalyMagnitude * 10;
    }

    values.push(value);
  }

  return { timestamps, values };
}

/**
 * Generate time series with missing data
 */
export function generateTimeSeriesWithMissing(
  count: number,
  missingRate = 0.1
): TimeSeriesData {
  const timestamps: number[] = [];
  const values: number[] = [];
  let baseTime = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    timestamps.push(baseTime + i * 3600000);

    // Generate value or NaN for missing
    const value = Math.random() < missingRate
      ? NaN
      : 50 + 10 * Math.sin(2 * Math.PI * i / 24) + (Math.random() - 0.5) * 2;

    values.push(value);
  }

  return { timestamps, values };
}

/**
 * Standard test fixtures
 */
export const LINEAR_SERIES = generateTimeSeriesWithPattern(100, 'linear', {
  trend: 0.5,
  noise: 0.1
});

export const SINE_SERIES = generateTimeSeriesWithPattern(100, 'sine', {
  amplitude: 20,
  frequency: 0.1,
  noise: 1
});

export const SEASONAL_SERIES = generateTimeSeriesWithPattern(168, 'seasonal', {
  amplitude: 10,
  trend: 0.1,
  noise: 2
});

export const MULTIVARIATE_SERIES = generateMultivariateTimeSeries(
  100,
  ['price', 'volume', 'volatility']
);

export const ANOMALY_SERIES = generateTimeSeriesWithAnomalies(100, 0.05, 5);

export const MISSING_DATA_SERIES = generateTimeSeriesWithMissing(100, 0.1);
