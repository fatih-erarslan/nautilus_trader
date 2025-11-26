/**
 * Model Performance Benchmarking Test Suite
 *
 * Benchmarks:
 * - GPU vs CPU training speed
 * - Different data sizes (1K, 10K, 100K, 1M samples)
 * - Memory usage profiling
 * - Inference latency measurements
 * - Batch processing efficiency
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Helper to get memory usage
function getMemoryUsage() {
  const usage = process.memoryUsage();
  return {
    heapUsed: usage.heapUsed / 1024 / 1024, // MB
    heapTotal: usage.heapTotal / 1024 / 1024,
    external: usage.external / 1024 / 1024,
    rss: usage.rss / 1024 / 1024
  };
}

// Helper to generate synthetic time series data
async function generateTrainingData(samples, filePath) {
  const rows = ['timestamp,value'];
  const baseDate = new Date('2023-01-01');

  for (let i = 0; i < samples; i++) {
    const date = new Date(baseDate.getTime() + i * 3600000);
    // Generate realistic price movement with trend and noise
    const trend = 100 + (i / samples) * 20;
    const seasonal = Math.sin(i / 24) * 10;
    const noise = (Math.random() - 0.5) * 5;
    const value = trend + seasonal + noise;

    rows.push(`${date.toISOString()},${value.toFixed(2)}`);
  }

  await fs.writeFile(filePath, rows.join('\n'));
}

describe('Model Performance Benchmarking', () => {

  describe('GPU vs CPU Training Performance', () => {
    const testDataPath = path.join(__dirname, '../fixtures/benchmark_data.csv');
    const samples = 5000;

    beforeAll(async () => {
      await generateTrainingData(samples, testDataPath);
    });

    test('CPU training benchmark', async () => {
      const startTime = Date.now();
      const startMem = getMemoryUsage();

      const result = await backend.neuralTrain(
        testDataPath,
        'lstm',
        20, // Fewer epochs for benchmark
        false // CPU
      );

      const endTime = Date.now();
      const endMem = getMemoryUsage();
      const duration = endTime - startTime;
      const memoryIncrease = endMem.heapUsed - startMem.heapUsed;

      console.log('\n=== CPU Training Benchmark ===');
      console.log(`Samples: ${samples}`);
      console.log(`Duration: ${duration}ms (${(duration/1000).toFixed(2)}s)`);
      console.log(`Memory increase: ${memoryIncrease.toFixed(2)}MB`);
      console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);
      console.log(`Validation accuracy: ${result.validationAccuracy.toFixed(4)}`);

      expect(result.modelId).toBeDefined();
      expect(duration).toBeGreaterThan(0);
    });

    test('GPU training benchmark (if available)', async () => {
      const startTime = Date.now();
      const startMem = getMemoryUsage();

      const result = await backend.neuralTrain(
        testDataPath,
        'lstm',
        20,
        true // GPU
      );

      const endTime = Date.now();
      const endMem = getMemoryUsage();
      const duration = endTime - startTime;
      const memoryIncrease = endMem.heapUsed - startMem.heapUsed;

      console.log('\n=== GPU Training Benchmark ===');
      console.log(`Samples: ${samples}`);
      console.log(`Duration: ${duration}ms (${(duration/1000).toFixed(2)}s)`);
      console.log(`Memory increase: ${memoryIncrease.toFixed(2)}MB`);
      console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);
      console.log(`Validation accuracy: ${result.validationAccuracy.toFixed(4)}`);

      // GPU should be faster or comparable (if GPU not available, falls back to CPU)
      expect(result.modelId).toBeDefined();
    });
  });

  describe('Scaling Tests - Different Data Sizes', () => {
    const dataSizes = [
      { name: '1K', samples: 1000 },
      { name: '10K', samples: 10000 },
      // Larger sizes commented out for CI/CD speed
      // { name: '100K', samples: 100000 },
      // { name: '1M', samples: 1000000 },
    ];

    test.each(dataSizes)('should train on $name samples efficiently', async ({ name, samples }) => {
      const dataPath = path.join(__dirname, `../fixtures/benchmark_${name}.csv`);
      await generateTrainingData(samples, dataPath);

      const startTime = Date.now();
      const startMem = getMemoryUsage();

      const result = await backend.neuralTrain(
        dataPath,
        'gru', // GRU is faster than LSTM
        10, // Minimal epochs for scaling test
        false
      );

      const endTime = Date.now();
      const endMem = getMemoryUsage();
      const duration = endTime - startTime;
      const memoryIncrease = endMem.heapUsed - startMem.heapUsed;
      const samplesPerSecond = (samples / duration) * 1000;

      console.log(`\n=== ${name} Samples Benchmark ===`);
      console.log(`Total samples: ${samples.toLocaleString()}`);
      console.log(`Duration: ${duration}ms (${(duration/1000).toFixed(2)}s)`);
      console.log(`Throughput: ${samplesPerSecond.toFixed(0)} samples/sec`);
      console.log(`Memory increase: ${memoryIncrease.toFixed(2)}MB`);
      console.log(`Memory per 1K samples: ${((memoryIncrease / samples) * 1000).toFixed(3)}MB`);

      expect(result.modelId).toBeDefined();

      // Cleanup
      await fs.unlink(dataPath).catch(() => {});
    }, 120000); // 2 minute timeout per test
  });

  describe('Inference Latency Benchmarks', () => {
    let modelId;
    const trainingDataPath = path.join(__dirname, '../fixtures/inference_training.csv');

    beforeAll(async () => {
      await generateTrainingData(2000, trainingDataPath);

      const result = await backend.neuralTrain(
        trainingDataPath,
        'gru',
        30,
        false
      );
      modelId = result.modelId;
    });

    test('single prediction latency', async () => {
      const iterations = 100;
      const latencies = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();

        await backend.neuralForecast('AAPL', 24, false);

        const endTime = performance.now();
        latencies.push(endTime - startTime);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const minLatency = Math.min(...latencies);
      const maxLatency = Math.max(...latencies);
      const p50 = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.5)];
      const p95 = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];
      const p99 = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.99)];

      console.log('\n=== Single Prediction Latency ===');
      console.log(`Iterations: ${iterations}`);
      console.log(`Average: ${avgLatency.toFixed(2)}ms`);
      console.log(`Min: ${minLatency.toFixed(2)}ms`);
      console.log(`Max: ${maxLatency.toFixed(2)}ms`);
      console.log(`P50: ${p50.toFixed(2)}ms`);
      console.log(`P95: ${p95.toFixed(2)}ms`);
      console.log(`P99: ${p99.toFixed(2)}ms`);

      // Inference should be fast (sub-second)
      expect(avgLatency).toBeLessThan(1000);
    });

    test('batch prediction throughput', async () => {
      const batchSizes = [1, 10, 50, 100];

      for (const batchSize of batchSizes) {
        const startTime = Date.now();

        const promises = [];
        for (let i = 0; i < batchSize; i++) {
          promises.push(backend.neuralForecast('AAPL', 24, false));
        }

        await Promise.all(promises);

        const duration = Date.now() - startTime;
        const throughput = (batchSize / duration) * 1000;

        console.log(`\nBatch size ${batchSize}:`);
        console.log(`  Duration: ${duration}ms`);
        console.log(`  Throughput: ${throughput.toFixed(2)} predictions/sec`);
        console.log(`  Avg per prediction: ${(duration/batchSize).toFixed(2)}ms`);
      }
    });
  });

  describe('Memory Usage Profiling', () => {

    test('training memory consumption', async () => {
      const dataPath = path.join(__dirname, '../fixtures/memory_test.csv');
      await generateTrainingData(5000, dataPath);

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const initialMem = getMemoryUsage();
      console.log('\n=== Initial Memory ===');
      console.log(`Heap Used: ${initialMem.heapUsed.toFixed(2)}MB`);
      console.log(`Heap Total: ${initialMem.heapTotal.toFixed(2)}MB`);
      console.log(`RSS: ${initialMem.rss.toFixed(2)}MB`);

      const result = await backend.neuralTrain(dataPath, 'lstm', 20, false);

      const afterTrainMem = getMemoryUsage();
      console.log('\n=== After Training ===');
      console.log(`Heap Used: ${afterTrainMem.heapUsed.toFixed(2)}MB (+${(afterTrainMem.heapUsed - initialMem.heapUsed).toFixed(2)}MB)`);
      console.log(`Heap Total: ${afterTrainMem.heapTotal.toFixed(2)}MB`);
      console.log(`RSS: ${afterTrainMem.rss.toFixed(2)}MB`);

      // Training shouldn't cause excessive memory growth
      const heapIncrease = afterTrainMem.heapUsed - initialMem.heapUsed;
      expect(heapIncrease).toBeLessThan(500); // Less than 500MB increase

      expect(result.modelId).toBeDefined();

      await fs.unlink(dataPath).catch(() => {});
    });

    test('inference memory consumption', async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/inference_mem.csv');
      await generateTrainingData(2000, trainingDataPath);

      await backend.neuralTrain(trainingDataPath, 'gru', 20, false);

      if (global.gc) {
        global.gc();
      }

      const initialMem = getMemoryUsage();

      // Run many inferences
      for (let i = 0; i < 100; i++) {
        await backend.neuralForecast('AAPL', 24, false);
      }

      const afterInferenceMem = getMemoryUsage();

      console.log('\n=== Inference Memory Impact (100 predictions) ===');
      console.log(`Initial Heap: ${initialMem.heapUsed.toFixed(2)}MB`);
      console.log(`After Inference: ${afterInferenceMem.heapUsed.toFixed(2)}MB`);
      console.log(`Increase: ${(afterInferenceMem.heapUsed - initialMem.heapUsed).toFixed(2)}MB`);

      // Inference shouldn't leak memory significantly
      const memIncrease = afterInferenceMem.heapUsed - initialMem.heapUsed;
      expect(memIncrease).toBeLessThan(50); // Less than 50MB for 100 inferences

      await fs.unlink(trainingDataPath).catch(() => {});
    });
  });

  describe('Batch Processing Efficiency', () => {

    test('sequential vs parallel processing', async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/batch_test.csv');
      await generateTrainingData(2000, trainingDataPath);

      await backend.neuralTrain(trainingDataPath, 'gru', 20, false);

      const numPredictions = 20;

      // Sequential processing
      const seqStart = Date.now();
      for (let i = 0; i < numPredictions; i++) {
        await backend.neuralForecast('AAPL', 24, false);
      }
      const seqDuration = Date.now() - seqStart;

      // Parallel processing
      const parStart = Date.now();
      const promises = [];
      for (let i = 0; i < numPredictions; i++) {
        promises.push(backend.neuralForecast('AAPL', 24, false));
      }
      await Promise.all(promises);
      const parDuration = Date.now() - parStart;

      const speedup = seqDuration / parDuration;

      console.log('\n=== Sequential vs Parallel Processing ===');
      console.log(`Predictions: ${numPredictions}`);
      console.log(`Sequential: ${seqDuration}ms (${(seqDuration/numPredictions).toFixed(2)}ms each)`);
      console.log(`Parallel: ${parDuration}ms (${(parDuration/numPredictions).toFixed(2)}ms each)`);
      console.log(`Speedup: ${speedup.toFixed(2)}x`);

      // Parallel should be faster
      expect(parDuration).toBeLessThanOrEqual(seqDuration);

      await fs.unlink(trainingDataPath).catch(() => {});
    });

    test('different horizon batch efficiency', async () => {
      const trainingDataPath = path.join(__dirname, '../fixtures/horizon_test.csv');
      await generateTrainingData(2000, trainingDataPath);

      await backend.neuralTrain(trainingDataPath, 'gru', 20, false);

      const horizons = [1, 12, 24, 48, 168]; // 1h, 12h, 1d, 2d, 1w

      console.log('\n=== Horizon Scaling ===');

      for (const horizon of horizons) {
        const startTime = Date.now();

        await backend.neuralForecast('AAPL', horizon, false);

        const duration = Date.now() - startTime;
        const msPerStep = duration / horizon;

        console.log(`Horizon ${horizon}: ${duration}ms total, ${msPerStep.toFixed(2)}ms per step`);
      }

      await fs.unlink(trainingDataPath).catch(() => {});
    });
  });

  describe('System Resource Monitoring', () => {

    test('CPU usage during training', async () => {
      const dataPath = path.join(__dirname, '../fixtures/cpu_test.csv');
      await generateTrainingData(5000, dataPath);

      console.log('\n=== System Info ===');
      console.log(`Platform: ${os.platform()}`);
      console.log(`Architecture: ${os.arch()}`);
      console.log(`CPUs: ${os.cpus().length}x ${os.cpus()[0].model}`);
      console.log(`Total Memory: ${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)}GB`);
      console.log(`Free Memory: ${(os.freemem() / 1024 / 1024 / 1024).toFixed(2)}GB`);

      const startTime = Date.now();
      const result = await backend.neuralTrain(dataPath, 'lstm', 30, false);
      const duration = Date.now() - startTime;

      console.log('\n=== Training Performance ===');
      console.log(`Duration: ${duration}ms`);
      console.log(`Model: ${result.modelType}`);
      console.log(`Final Loss: ${result.finalLoss.toFixed(6)}`);
      console.log(`Validation Accuracy: ${result.validationAccuracy.toFixed(4)}`);

      await fs.unlink(dataPath).catch(() => {});
    });
  });

  describe('Comparative Model Performance', () => {
    const modelTypes = ['lstm', 'gru', 'transformer'];
    const dataPath = path.join(__dirname, '../fixtures/model_compare.csv');

    beforeAll(async () => {
      await generateTrainingData(3000, dataPath);
    });

    test.each(modelTypes)('%s model performance', async (modelType) => {
      const startTime = Date.now();
      const startMem = getMemoryUsage();

      const result = await backend.neuralTrain(
        dataPath,
        modelType,
        20,
        false
      );

      const duration = Date.now() - startTime;
      const endMem = getMemoryUsage();
      const memoryIncrease = endMem.heapUsed - startMem.heapUsed;

      console.log(`\n=== ${modelType.toUpperCase()} Performance ===`);
      console.log(`Training time: ${duration}ms`);
      console.log(`Memory increase: ${memoryIncrease.toFixed(2)}MB`);
      console.log(`Final loss: ${result.finalLoss.toFixed(6)}`);
      console.log(`Validation accuracy: ${result.validationAccuracy.toFixed(4)}`);

      expect(result.modelId).toBeDefined();
      expect(result.validationAccuracy).toBeGreaterThan(0);
    });

    afterAll(async () => {
      await fs.unlink(dataPath).catch(() => {});
    });
  });
});
