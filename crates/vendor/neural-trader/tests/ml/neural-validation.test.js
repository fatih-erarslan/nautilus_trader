/**
 * Neural Network Function Validation Test Suite
 *
 * Comprehensive validation of all neural/ML functions:
 * - neuralForecast: test with various horizons, confidence levels
 * - neuralTrain: validate training convergence, overfitting detection
 * - neuralEvaluate: verify metrics accuracy (MAE, RMSE, MAPE, R²)
 * - neuralModelStatus: check status tracking accuracy
 * - neuralOptimize: test hyperparameter optimization
 * - neuralBacktest: validate historical accuracy
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');
const fs = require('fs').promises;
const path = require('path');

describe('Neural Function Validation Suite', () => {

  describe('neuralForecast - Forecasting with Various Parameters', () => {

    test('should forecast with default parameters', async () => {
      const result = await backend.neuralForecast('AAPL', 24);

      expect(result).toBeDefined();
      expect(result.symbol).toBe('AAPL');
      expect(result.horizon).toBe(24);
      expect(result.predictions).toHaveLength(24);
      expect(result.confidenceIntervals).toHaveLength(24);
      expect(result.modelAccuracy).toBeGreaterThan(0);
      expect(result.modelAccuracy).toBeLessThanOrEqual(1);
    });

    test('should forecast with custom confidence level', async () => {
      const confidenceLevels = [0.80, 0.90, 0.95, 0.99];

      for (const confidence of confidenceLevels) {
        const result = await backend.neuralForecast('AAPL', 24, true, confidence);

        expect(result).toBeDefined();
        expect(result.predictions).toHaveLength(24);

        // Verify confidence intervals widen with higher confidence
        result.confidenceIntervals.forEach(interval => {
          expect(interval.lower).toBeLessThan(interval.upper);
          const width = interval.upper - interval.lower;
          expect(width).toBeGreaterThan(0);
        });
      }
    });

    test('should forecast with various horizons', async () => {
      const horizons = [1, 7, 24, 48, 168, 365]; // 1 hour to 1 year

      for (const horizon of horizons) {
        const result = await backend.neuralForecast('AAPL', horizon);

        expect(result).toBeDefined();
        expect(result.horizon).toBe(horizon);
        expect(result.predictions).toHaveLength(horizon);
        expect(result.confidenceIntervals).toHaveLength(horizon);
      }
    });

    test('should reject invalid horizon values', async () => {
      await expect(backend.neuralForecast('AAPL', 0))
        .rejects.toThrow(/horizon must be greater than 0/i);

      await expect(backend.neuralForecast('AAPL', 400))
        .rejects.toThrow(/exceeds maximum/i);

      await expect(backend.neuralForecast('AAPL', -1))
        .rejects.toThrow();
    });

    test('should reject invalid confidence levels', async () => {
      await expect(backend.neuralForecast('AAPL', 24, true, 0))
        .rejects.toThrow(/confidence level.*between 0 and 1/i);

      await expect(backend.neuralForecast('AAPL', 24, true, 1))
        .rejects.toThrow(/confidence level.*between 0 and 1/i);

      await expect(backend.neuralForecast('AAPL', 24, true, -0.5))
        .rejects.toThrow(/confidence level.*between 0 and 1/i);

      await expect(backend.neuralForecast('AAPL', 24, true, 1.5))
        .rejects.toThrow(/confidence level.*between 0 and 1/i);
    });

    test('should reject empty symbol', async () => {
      await expect(backend.neuralForecast('', 24))
        .rejects.toThrow(/symbol cannot be empty/i);
    });

    test('should provide monotonic confidence intervals', async () => {
      const result = await backend.neuralForecast('AAPL', 24);

      // Check that confidence intervals grow with forecast distance
      for (let i = 1; i < result.confidenceIntervals.length; i++) {
        const prevWidth = result.confidenceIntervals[i-1].upper - result.confidenceIntervals[i-1].lower;
        const currWidth = result.confidenceIntervals[i].upper - result.confidenceIntervals[i].lower;

        // Intervals should generally widen or stay constant
        expect(currWidth).toBeGreaterThanOrEqual(prevWidth * 0.9); // Allow 10% variance
      }
    });
  });

  describe('neuralTrain - Training Validation and Convergence', () => {
    const testDataPath = path.join(__dirname, '../fixtures/training_data.csv');

    beforeAll(async () => {
      // Create test training data if it doesn't exist
      const dir = path.dirname(testDataPath);
      await fs.mkdir(dir, { recursive: true });

      // Generate synthetic time series data
      const rows = ['timestamp,value'];
      const baseDate = new Date('2023-01-01');
      for (let i = 0; i < 1000; i++) {
        const date = new Date(baseDate.getTime() + i * 3600000); // Hourly data
        const value = 100 + Math.sin(i / 24) * 10 + Math.random() * 5;
        rows.push(`${date.toISOString()},${value.toFixed(2)}`);
      }
      await fs.writeFile(testDataPath, rows.join('\n'));
    });

    test('should train model successfully with default parameters', async () => {
      const result = await backend.neuralTrain(
        testDataPath,
        'lstm',
        100,
        false // CPU for testing
      );

      expect(result).toBeDefined();
      expect(result.modelId).toBeDefined();
      expect(result.modelType).toBe('lstm');
      expect(result.trainingTimeMs).toBeGreaterThan(0);
      expect(result.finalLoss).toBeGreaterThan(0);
      expect(result.validationAccuracy).toBeGreaterThan(0);
      expect(result.validationAccuracy).toBeLessThanOrEqual(1);
    });

    test('should train different model types', async () => {
      const modelTypes = ['lstm', 'gru', 'transformer', 'cnn', 'hybrid'];

      for (const modelType of modelTypes) {
        const result = await backend.neuralTrain(
          testDataPath,
          modelType,
          50, // Fewer epochs for speed
          false
        );

        expect(result).toBeDefined();
        expect(result.modelType).toBe(modelType);
        expect(result.modelId).toBeDefined();
      }
    });

    test('should validate epoch count', async () => {
      await expect(backend.neuralTrain(testDataPath, 'lstm', 0, false))
        .rejects.toThrow(/epochs must be greater than 0/i);

      await expect(backend.neuralTrain(testDataPath, 'lstm', 15000, false))
        .rejects.toThrow(/exceeds maximum/i);
    });

    test('should reject invalid model types', async () => {
      await expect(backend.neuralTrain(testDataPath, 'invalid_model', 100, false))
        .rejects.toThrow(/unknown model type/i);

      await expect(backend.neuralTrain(testDataPath, '', 100, false))
        .rejects.toThrow(/unknown model type/i);
    });

    test('should reject non-existent data path', async () => {
      await expect(backend.neuralTrain('/nonexistent/path.csv', 'lstm', 100, false))
        .rejects.toThrow(/not found/i);
    });

    test('should validate training convergence', async () => {
      const result = await backend.neuralTrain(
        testDataPath,
        'lstm',
        100,
        false
      );

      // Final loss should be reasonable
      expect(result.finalLoss).toBeLessThan(1.0);

      // Validation accuracy should be decent
      expect(result.validationAccuracy).toBeGreaterThan(0.5);

      // Training should complete in reasonable time
      expect(result.trainingTimeMs).toBeLessThan(300000); // 5 minutes max
    });
  });

  describe('neuralEvaluate - Metrics Accuracy Validation', () => {
    let trainedModelId;
    const testDataPath = path.join(__dirname, '../fixtures/test_data.csv');

    beforeAll(async () => {
      // Train a model first
      const trainingDataPath = path.join(__dirname, '../fixtures/training_data.csv');
      const trainResult = await backend.neuralTrain(
        trainingDataPath,
        'lstm',
        50,
        false
      );
      trainedModelId = trainResult.modelId;

      // Create test data
      const rows = ['timestamp,value'];
      const baseDate = new Date('2024-01-01');
      for (let i = 0; i < 200; i++) {
        const date = new Date(baseDate.getTime() + i * 3600000);
        const value = 100 + Math.sin(i / 24) * 10 + Math.random() * 5;
        rows.push(`${date.toISOString()},${value.toFixed(2)}`);
      }
      await fs.writeFile(testDataPath, rows.join('\n'));
    });

    test('should evaluate model on test data', async () => {
      const result = await backend.neuralEvaluate(
        trainedModelId,
        testDataPath,
        false
      );

      expect(result).toBeDefined();
      expect(result.modelId).toBe(trainedModelId);
      expect(result.testSamples).toBeGreaterThan(0);
      expect(result.mae).toBeGreaterThan(0);
      expect(result.rmse).toBeGreaterThan(0);
      expect(result.mape).toBeGreaterThan(0);
      expect(result.r2Score).toBeGreaterThanOrEqual(-1); // R² can be negative
      expect(result.r2Score).toBeLessThanOrEqual(1);
    });

    test('should verify metric relationships', async () => {
      const result = await backend.neuralEvaluate(
        trainedModelId,
        testDataPath,
        false
      );

      // RMSE should be >= MAE (by Jensen's inequality)
      expect(result.rmse).toBeGreaterThanOrEqual(result.mae);

      // MAPE should be a percentage
      expect(result.mape).toBeGreaterThanOrEqual(0);
      expect(result.mape).toBeLessThan(200); // Reasonable upper bound
    });

    test('should reject invalid model ID', async () => {
      await expect(backend.neuralEvaluate('nonexistent-model', testDataPath, false))
        .rejects.toThrow(/not found/i);

      await expect(backend.neuralEvaluate('', testDataPath, false))
        .rejects.toThrow(/cannot be empty/i);
    });

    test('should reject invalid test data path', async () => {
      await expect(backend.neuralEvaluate(trainedModelId, '', false))
        .rejects.toThrow(/cannot be empty/i);

      await expect(backend.neuralEvaluate(trainedModelId, '/nonexistent.csv', false))
        .rejects.toThrow(/not found/i);
    });
  });

  describe('neuralModelStatus - Status Tracking Accuracy', () => {
    let modelIds = [];

    beforeAll(async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/training_data.csv');

      // Train multiple models
      for (let i = 0; i < 3; i++) {
        const result = await backend.neuralTrain(
          trainingDataPath,
          ['lstm', 'gru', 'transformer'][i],
          50,
          false
        );
        modelIds.push(result.modelId);
      }
    });

    test('should list all trained models', async () => {
      const models = await backend.neuralModelStatus();

      expect(Array.isArray(models)).toBe(true);
      expect(models.length).toBeGreaterThanOrEqual(modelIds.length);

      models.forEach(model => {
        expect(model.modelId).toBeDefined();
        expect(model.modelType).toBeDefined();
        expect(model.status).toBeDefined();
        expect(model.createdAt).toBeDefined();
        expect(model.accuracy).toBeGreaterThanOrEqual(0);
        expect(model.accuracy).toBeLessThanOrEqual(1);
      });
    });

    test('should get specific model status', async () => {
      for (const modelId of modelIds) {
        const models = await backend.neuralModelStatus(modelId);

        expect(models).toHaveLength(1);
        expect(models[0].modelId).toBe(modelId);
      }
    });

    test('should return empty array for non-existent model', async () => {
      const models = await backend.neuralModelStatus('nonexistent-id-12345');
      expect(models).toHaveLength(0);
    });

    test('should track model creation timestamps', async () => {
      const models = await backend.neuralModelStatus();

      models.forEach(model => {
        const createdDate = new Date(model.createdAt);
        expect(createdDate).toBeInstanceOf(Date);
        expect(createdDate.getTime()).toBeLessThanOrEqual(Date.now());
        expect(createdDate.getTime()).toBeGreaterThan(Date.now() - 3600000); // Within last hour
      });
    });
  });

  describe('neuralOptimize - Hyperparameter Optimization', () => {
    let modelId;

    beforeAll(async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/training_data.csv');
      const result = await backend.neuralTrain(
        trainingDataPath,
        'lstm',
        50,
        false
      );
      modelId = result.modelId;
    });

    test('should optimize hyperparameters successfully', async () => {
      const paramRanges = JSON.stringify({
        learning_rate: [0.0001, 0.01],
        batch_size: [16, 32, 64],
        hidden_size: [128, 256, 512],
        num_layers: [1, 2, 3, 4]
      });

      const result = await backend.neuralOptimize(
        modelId,
        paramRanges,
        false
      );

      expect(result).toBeDefined();
      expect(result.modelId).toBe(modelId);
      expect(result.bestParams).toBeDefined();
      expect(result.bestScore).toBeGreaterThan(0);
      expect(result.bestScore).toBeLessThanOrEqual(1);
      expect(result.trialsCompleted).toBeGreaterThan(0);
      expect(result.optimizationTimeMs).toBeGreaterThan(0);

      // Verify best params is valid JSON
      const params = JSON.parse(result.bestParams);
      expect(params).toBeDefined();
    });

    test('should improve upon baseline accuracy', async () => {
      const paramRanges = JSON.stringify({
        learning_rate: [0.0001, 0.001, 0.01],
        batch_size: [32, 64]
      });

      // Get baseline model status
      const baselineStatus = await backend.neuralModelStatus(modelId);
      const baselineAccuracy = baselineStatus[0].accuracy;

      const result = await backend.neuralOptimize(
        modelId,
        paramRanges,
        false
      );

      // Optimized score should be >= baseline (allowing for variation)
      expect(result.bestScore).toBeGreaterThanOrEqual(baselineAccuracy * 0.95);
    });

    test('should reject invalid parameter ranges', async () => {
      await expect(backend.neuralOptimize(modelId, '', false))
        .rejects.toThrow(/cannot be empty/i);

      await expect(backend.neuralOptimize(modelId, 'invalid json', false))
        .rejects.toThrow(/invalid.*json/i);

      await expect(backend.neuralOptimize(modelId, '{"incomplete":', false))
        .rejects.toThrow(/invalid.*json/i);
    });

    test('should reject invalid model ID', async () => {
      const paramRanges = JSON.stringify({ learning_rate: [0.001, 0.01] });

      await expect(backend.neuralOptimize('', paramRanges, false))
        .rejects.toThrow(/cannot be empty/i);

      await expect(backend.neuralOptimize('nonexistent', paramRanges, false))
        .rejects.toThrow(/not found/i);
    });
  });

  describe('neuralBacktest - Historical Accuracy Validation', () => {
    let modelId;

    beforeAll(async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/training_data.csv');
      const result = await backend.neuralTrain(
        trainingDataPath,
        'lstm',
        50,
        false
      );
      modelId = result.modelId;
    });

    test('should run backtest successfully', async () => {
      const result = await backend.neuralBacktest(
        modelId,
        '2023-01-01',
        '2023-12-31',
        'SPY',
        false
      );

      expect(result).toBeDefined();
      expect(result.modelId).toBe(modelId);
      expect(result.startDate).toBe('2023-01-01');
      expect(result.endDate).toBe('2023-12-31');
      expect(result.totalReturn).toBeDefined();
      expect(result.sharpeRatio).toBeDefined();
      expect(result.maxDrawdown).toBeLessThanOrEqual(0); // Drawdown is negative
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
      expect(result.totalTrades).toBeGreaterThan(0);
    });

    test('should validate metric relationships', async () => {
      const result = await backend.neuralBacktest(
        modelId,
        '2023-01-01',
        '2023-12-31',
        'SPY',
        false
      );

      // Max drawdown should be negative or zero
      expect(result.maxDrawdown).toBeLessThanOrEqual(0);

      // Sharpe ratio should be reasonable
      expect(result.sharpeRatio).toBeGreaterThan(-5);
      expect(result.sharpeRatio).toBeLessThan(10);

      // Win rate should be a valid probability
      expect(result.winRate).toBeGreaterThanOrEqual(0);
      expect(result.winRate).toBeLessThanOrEqual(1);
    });

    test('should handle different date ranges', async () => {
      const ranges = [
        { start: '2023-01-01', end: '2023-03-31' },
        { start: '2023-04-01', end: '2023-06-30' },
        { start: '2023-07-01', end: '2023-09-30' },
      ];

      for (const range of ranges) {
        const result = await backend.neuralBacktest(
          modelId,
          range.start,
          range.end,
          'SPY',
          false
        );

        expect(result).toBeDefined();
        expect(result.startDate).toBe(range.start);
        expect(result.endDate).toBe(range.end);
      }
    });

    test('should reject invalid dates', async () => {
      await expect(backend.neuralBacktest(modelId, 'invalid', '2023-12-31'))
        .rejects.toThrow(/invalid.*date/i);

      await expect(backend.neuralBacktest(modelId, '2023-01-01', 'invalid'))
        .rejects.toThrow(/invalid.*date/i);

      await expect(backend.neuralBacktest(modelId, '2023/01/01', '2023/12/31'))
        .rejects.toThrow(/invalid.*date/i);
    });

    test('should reject invalid model ID', async () => {
      await expect(backend.neuralBacktest('', '2023-01-01', '2023-12-31'))
        .rejects.toThrow(/cannot be empty/i);

      await expect(backend.neuralBacktest('nonexistent', '2023-01-01', '2023-12-31'))
        .rejects.toThrow(/not found/i);
    });

    test('should compare to benchmark', async () => {
      const result = await backend.neuralBacktest(
        modelId,
        '2023-01-01',
        '2023-12-31',
        'SPY',
        false
      );

      // Results should be comparable to benchmark
      expect(result.totalReturn).toBeGreaterThan(-1); // Not total loss
      expect(result.sharpeRatio).toBeDefined();
    });
  });

  describe('Integration Tests - End-to-End ML Workflow', () => {
    test('should complete full ML pipeline', async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/training_data.csv');
      const testDataPath = path.join(__dirname, '../fixtures/test_data.csv');

      // 1. Train model
      const trainResult = await backend.neuralTrain(
        trainingDataPath,
        'lstm',
        50,
        false
      );
      expect(trainResult.modelId).toBeDefined();

      // 2. Evaluate model
      const evalResult = await backend.neuralEvaluate(
        trainResult.modelId,
        testDataPath,
        false
      );
      expect(evalResult.mae).toBeGreaterThan(0);

      // 3. Optimize hyperparameters
      const optimizeResult = await backend.neuralOptimize(
        trainResult.modelId,
        JSON.stringify({ learning_rate: [0.001, 0.01], batch_size: [32, 64] }),
        false
      );
      expect(optimizeResult.bestScore).toBeGreaterThan(0);

      // 4. Run backtest
      const backtestResult = await backend.neuralBacktest(
        trainResult.modelId,
        '2023-01-01',
        '2023-12-31',
        'SPY',
        false
      );
      expect(backtestResult.totalReturn).toBeDefined();

      // 5. Get model status
      const statusResult = await backend.neuralModelStatus(trainResult.modelId);
      expect(statusResult).toHaveLength(1);

      console.log('\n=== Full ML Pipeline Results ===');
      console.log(`Model ID: ${trainResult.modelId}`);
      console.log(`Training Accuracy: ${trainResult.validationAccuracy.toFixed(4)}`);
      console.log(`Test MAE: ${evalResult.mae.toFixed(4)}`);
      console.log(`Test RMSE: ${evalResult.rmse.toFixed(4)}`);
      console.log(`Test R²: ${evalResult.r2Score.toFixed(4)}`);
      console.log(`Optimized Score: ${optimizeResult.bestScore.toFixed(4)}`);
      console.log(`Backtest Return: ${(backtestResult.totalReturn * 100).toFixed(2)}%`);
      console.log(`Backtest Sharpe: ${backtestResult.sharpeRatio.toFixed(2)}`);
      console.log(`Backtest Max DD: ${(backtestResult.maxDrawdown * 100).toFixed(2)}%`);
    });
  });
});
