/**
 * Comprehensive tests for DemandForecaster
 */

import { DemandForecaster, DemandPattern, ForecastConfig } from '../src/demand-forecaster';

describe('DemandForecaster', () => {
  let forecaster: DemandForecaster;
  let config: ForecastConfig;
  let trainingData: DemandPattern[];

  beforeEach(() => {
    config = {
      alpha: 0.1,
      horizons: [1, 7, 14, 30],
      seasonalityPeriods: [7, 52],
      learningRate: 0.01,
      memoryNamespace: 'test-supply-chain',
    };

    forecaster = new DemandForecaster(config);

    // Generate synthetic training data
    trainingData = generateSyntheticData(100);
  });

  describe('Training', () => {
    it('should train on historical patterns', async () => {
      await expect(forecaster.train(trainingData)).resolves.not.toThrow();
    });

    it('should extract seasonal patterns', async () => {
      await forecaster.train(trainingData);
      const calibration = forecaster.getCalibration();
      expect(calibration).toBeDefined();
    });

    it('should fit trend models', async () => {
      await forecaster.train(trainingData);
      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );
      expect(forecast.trendComponent).toBeDefined();
    });
  });

  describe('Forecasting', () => {
    beforeEach(async () => {
      await forecaster.train(trainingData);
    });

    it('should generate point forecast', async () => {
      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.pointForecast).toBeGreaterThan(0);
    });

    it('should provide prediction intervals', async () => {
      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.lowerBound).toBeLessThan(forecast.pointForecast);
      expect(forecast.upperBound).toBeGreaterThan(forecast.pointForecast);
      expect(forecast.confidence).toBe(1 - config.alpha);
    });

    it('should handle multi-horizon forecasts', async () => {
      const forecasts = await forecaster.forecastMultiHorizon('product-1', {
        dayOfWeek: 1,
        weekOfYear: 20,
        monthOfYear: 5,
        isHoliday: false,
        promotions: 0,
        priceIndex: 1.0,
      });

      expect(forecasts).toHaveLength(config.horizons.length);
      expect(forecasts[0]!.horizon).toBe(1);
      expect(forecasts[3]!.horizon).toBe(30);
    });

    it('should account for seasonality', async () => {
      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.seasonalComponent).toBeGreaterThan(0);
    });

    it('should quantify uncertainty', async () => {
      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.uncertainty).toBeGreaterThan(0);
    });

    it('should increase uncertainty with horizon', async () => {
      const shortHorizon = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      const longHorizon = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        30
      );

      expect(longHorizon.uncertainty).toBeGreaterThan(shortHorizon.uncertainty);
    });
  });

  describe('Online Learning', () => {
    beforeEach(async () => {
      await forecaster.train(trainingData);
    });

    it('should update with new observations', async () => {
      const observation: DemandPattern = {
        productId: 'product-1',
        timestamp: Date.now(),
        demand: 150,
        features: {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
      };

      await expect(forecaster.update(observation)).resolves.not.toThrow();
    });

    it('should adapt to changing patterns', async () => {
      const initialForecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      // Add high-demand observations
      for (let i = 0; i < 10; i++) {
        await forecaster.update({
          productId: 'product-1',
          timestamp: Date.now() + i * 1000,
          demand: 200,
          features: {
            dayOfWeek: 1,
            weekOfYear: 20,
            monthOfYear: 5,
            isHoliday: false,
            promotions: 0,
            priceIndex: 1.0,
          },
        });
      }

      const updatedForecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      // Forecast should adapt to higher demand
      expect(updatedForecast.pointForecast).toBeGreaterThanOrEqual(
        initialForecast.pointForecast
      );
    });
  });

  describe('Calibration', () => {
    beforeEach(async () => {
      await forecaster.train(trainingData);
    });

    it('should provide calibration metrics', () => {
      const calibration = forecaster.getCalibration();
      expect(calibration).toHaveProperty('coverage');
      expect(calibration).toHaveProperty('intervalWidth');
    });

    it('should achieve target coverage', async () => {
      // Generate test data
      const testData = generateSyntheticData(50);
      let inInterval = 0;

      for (const point of testData) {
        const forecast = await forecaster.forecast(
          point.productId,
          point.features,
          1
        );

        if (
          point.demand >= forecast.lowerBound &&
          point.demand <= forecast.upperBound
        ) {
          inInterval++;
        }
      }

      const coverage = inInterval / testData.length;
      const targetCoverage = 1 - config.alpha;

      // Allow 10% tolerance
      expect(coverage).toBeGreaterThanOrEqual(targetCoverage * 0.9);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty training data', async () => {
      await expect(forecaster.train([])).resolves.not.toThrow();
    });

    it('should handle single product', async () => {
      const singleProduct = trainingData.filter((p) => p.productId === 'product-1');
      await forecaster.train(singleProduct);

      const forecast = await forecaster.forecast(
        'product-1',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.pointForecast).toBeGreaterThan(0);
    });

    it('should handle unknown product', async () => {
      await forecaster.train(trainingData);

      const forecast = await forecaster.forecast(
        'unknown-product',
        {
          dayOfWeek: 1,
          weekOfYear: 20,
          monthOfYear: 5,
          isHoliday: false,
          promotions: 0,
          priceIndex: 1.0,
        },
        1
      );

      expect(forecast.pointForecast).toBeDefined();
    });

    it('should handle extreme demand values', async () => {
      const extremeData: DemandPattern[] = [
        {
          productId: 'product-extreme',
          timestamp: Date.now(),
          demand: 0,
          features: {
            dayOfWeek: 1,
            weekOfYear: 1,
            monthOfYear: 1,
            isHoliday: false,
            promotions: 0,
            priceIndex: 1.0,
          },
        },
        {
          productId: 'product-extreme',
          timestamp: Date.now() + 1000,
          demand: 10000,
          features: {
            dayOfWeek: 2,
            weekOfYear: 2,
            monthOfYear: 1,
            isHoliday: false,
            promotions: 0,
            priceIndex: 1.0,
          },
        },
      ];

      await expect(forecaster.train(extremeData)).resolves.not.toThrow();
    });
  });
});

/**
 * Generate synthetic demand data for testing
 */
function generateSyntheticData(count: number): DemandPattern[] {
  const data: DemandPattern[] = [];
  const baseDate = new Date('2024-01-01');

  for (let i = 0; i < count; i++) {
    const date = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    const dayOfWeek = date.getDay();
    const weekOfYear = getWeekOfYear(date);
    const monthOfYear = date.getMonth();

    // Simulate seasonal pattern
    const seasonal = 1 + 0.3 * Math.sin((weekOfYear / 52) * 2 * Math.PI);

    // Simulate weekly pattern (higher on weekends)
    const weekly = dayOfWeek >= 5 ? 1.2 : 1.0;

    // Base demand with trend
    const trend = 100 + i * 0.5;
    const demand = trend * seasonal * weekly + (Math.random() - 0.5) * 20;

    data.push({
      productId: `product-${(i % 3) + 1}`,
      timestamp: date.getTime(),
      demand: Math.max(0, demand),
      features: {
        dayOfWeek,
        weekOfYear,
        monthOfYear,
        isHoliday: false,
        promotions: Math.random() > 0.9 ? 1 : 0,
        priceIndex: 1 + (Math.random() - 0.5) * 0.2,
      },
    });
  }

  return data;
}

/**
 * Get week of year
 */
function getWeekOfYear(date: Date): number {
  const firstDay = new Date(date.getFullYear(), 0, 1);
  const days = Math.floor((date.getTime() - firstDay.getTime()) / (24 * 60 * 60 * 1000));
  return Math.floor(days / 7);
}
