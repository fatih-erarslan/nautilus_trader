/**
 * Example 4: Production Deployment Guide
 *
 * Demonstrates:
 * - Model checkpointing and versioning
 * - API endpoint integration
 * - Error handling and monitoring
 * - Performance optimization
 * - Graceful degradation
 */

const { NeuralForecaster, models } = require('@neural-trader/neuro-divergent');
const express = require('express');
const prometheus = require('prom-client');

// Production configuration
const PRODUCTION_CONFIG = {
    modelPath: './models/production',
    checkpointInterval: 3600000,  // 1 hour
    maxConcurrentPredictions: 100,
    cacheTTL: 300000,  // 5 minutes
    healthCheckInterval: 60000,  // 1 minute
};

class ProductionForecaster {
    constructor(config = PRODUCTION_CONFIG) {
        this.config = config;
        this.forecaster = null;
        this.modelVersion = null;
        this.cache = new Map();
        this.metrics = this.initializeMetrics();
        this.isHealthy = false;
    }

    /**
     * Initialize Prometheus metrics
     */
    initializeMetrics() {
        return {
            predictionDuration: new prometheus.Histogram({
                name: 'forecast_prediction_duration_seconds',
                help: 'Duration of prediction requests',
                buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
            }),
            predictionCount: new prometheus.Counter({
                name: 'forecast_predictions_total',
                help: 'Total number of predictions',
                labelNames: ['status']
            }),
            cacheHitRate: new prometheus.Counter({
                name: 'forecast_cache_hits_total',
                help: 'Total cache hits',
                labelNames: ['hit']
            }),
            modelLoadTime: new prometheus.Gauge({
                name: 'forecast_model_load_seconds',
                help: 'Time to load the model'
            })
        };
    }

    /**
     * Initialize forecaster with production settings
     */
    async initialize() {
        try {
            console.log('🚀 Initializing production forecaster...');

            const startTime = Date.now();

            this.forecaster = new NeuralForecaster({
                models: [
                    new models.LSTM({
                        hiddenSize: 128,
                        numLayers: 2,
                        horizon: 24,
                        inputSize: 48
                    }),
                    new models.NBEATS({
                        stackTypes: ['trend', 'seasonality'],
                        numBlocks: 3,
                        horizon: 24,
                        inputSize: 96
                    })
                ],
                frequency: 'H',
                numThreads: 8,
                simdEnabled: true,
                memoryPool: true,
                backend: 'cpu'  // or 'cuda' if GPU available
            });

            // Load checkpoint if exists
            const checkpointPath = `${this.config.modelPath}/latest.safetensors`;
            try {
                await this.forecaster.loadCheckpoint(checkpointPath);
                console.log(`✅ Loaded checkpoint: ${checkpointPath}`);
            } catch (error) {
                console.warn('⚠️  No checkpoint found, starting fresh');
            }

            const loadTime = (Date.now() - startTime) / 1000;
            this.metrics.modelLoadTime.set(loadTime);

            this.isHealthy = true;
            console.log(`✅ Forecaster initialized in ${loadTime.toFixed(2)}s`);

            // Start periodic checkpoint saving
            this.startCheckpointScheduler();

            // Start health monitoring
            this.startHealthMonitoring();

        } catch (error) {
            console.error('❌ Failed to initialize forecaster:', error);
            this.isHealthy = false;
            throw error;
        }
    }

    /**
     * Make predictions with caching and error handling
     */
    async predict(inputData, options = {}) {
        const endTimer = this.metrics.predictionDuration.startTimer();

        try {
            // Generate cache key
            const cacheKey = this.generateCacheKey(inputData, options);

            // Check cache
            if (this.cache.has(cacheKey)) {
                const cached = this.cache.get(cacheKey);
                if (Date.now() - cached.timestamp < this.config.cacheTTL) {
                    this.metrics.cacheHitRate.inc({ hit: 'true' });
                    endTimer();
                    return cached.result;
                }
            }
            this.metrics.cacheHitRate.inc({ hit: 'false' });

            // Make prediction
            const result = await this.forecaster.predict({
                horizon: options.horizon || 24,
                level: options.level || [95],
                ...options
            });

            // Cache result
            this.cache.set(cacheKey, {
                result,
                timestamp: Date.now()
            });

            // Clean old cache entries
            this.cleanCache();

            this.metrics.predictionCount.inc({ status: 'success' });
            endTimer();

            return result;

        } catch (error) {
            this.metrics.predictionCount.inc({ status: 'error' });
            endTimer();

            console.error('Prediction error:', error);

            // Graceful degradation
            return this.getFallbackPrediction(inputData, options);
        }
    }

    /**
     * Train or retrain model
     */
    async train(data, options = {}) {
        try {
            console.log('🎯 Starting model training...');

            await this.forecaster.fit(data, {
                epochs: options.epochs || 100,
                batchSize: options.batchSize || 32,
                learningRate: options.learningRate || 0.001,
                validationSize: 0.2,
                earlyStopping: true,
                patience: 10,
                verbose: true,
                // Production checkpointing
                checkpointPath: this.config.modelPath,
                checkpointFrequency: 10
            });

            // Save final checkpoint
            await this.saveCheckpoint();

            console.log('✅ Training completed');

        } catch (error) {
            console.error('❌ Training failed:', error);
            throw error;
        }
    }

    /**
     * Save model checkpoint
     */
    async saveCheckpoint() {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const checkpointPath = `${this.config.modelPath}/checkpoint-${timestamp}.safetensors`;

            await this.forecaster.saveCheckpoint(checkpointPath);

            // Update latest symlink
            const latestPath = `${this.config.modelPath}/latest.safetensors`;
            await this.forecaster.saveCheckpoint(latestPath);

            console.log(`💾 Checkpoint saved: ${checkpointPath}`);

        } catch (error) {
            console.error('Failed to save checkpoint:', error);
        }
    }

    /**
     * Start periodic checkpoint saving
     */
    startCheckpointScheduler() {
        setInterval(async () => {
            await this.saveCheckpoint();
        }, this.config.checkpointInterval);
    }

    /**
     * Health check monitoring
     */
    startHealthMonitoring() {
        setInterval(async () => {
            try {
                // Simple health check: make a test prediction
                const testData = this.generateTestData();
                await this.predict(testData, { horizon: 1 });
                this.isHealthy = true;
            } catch (error) {
                console.error('Health check failed:', error);
                this.isHealthy = false;
            }
        }, this.config.healthCheckInterval);
    }

    /**
     * Generate cache key from input data
     * Security: Uses SHA256 for secure hashing
     */
    generateCacheKey(inputData, options) {
        const hash = require('crypto')
            .createHash('sha256')
            .update(JSON.stringify({ inputData, options }))
            .digest('hex');
        return hash;
    }

    /**
     * Clean old cache entries
     */
    cleanCache() {
        const now = Date.now();
        for (const [key, value] of this.cache.entries()) {
            if (now - value.timestamp > this.config.cacheTTL) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Fallback prediction for error cases
     */
    getFallbackPrediction(inputData, options) {
        // Simple persistence model (last value)
        const lastValue = inputData.y[inputData.y.length - 1];
        const horizon = options.horizon || 24;

        return {
            ds: Array(horizon).fill(null).map((_, i) => {
                const date = new Date();
                date.setHours(date.getHours() + i);
                return date.toISOString().split('T')[0];
            }),
            fallback: Array(horizon).fill(lastValue)
        };
    }

    /**
     * Generate test data for health checks
     */
    generateTestData() {
        return {
            unique_id: ['test'],
            ds: Array(48).fill(null).map((_, i) => {
                const date = new Date();
                date.setHours(date.getHours() - 48 + i);
                return date.toISOString().split('T')[0];
            }),
            y: Array(48).fill(null).map(() => Math.random() * 100)
        };
    }

    /**
     * Get current health status
     */
    getHealthStatus() {
        return {
            healthy: this.isHealthy,
            modelVersion: this.modelVersion,
            cacheSize: this.cache.size,
            uptime: process.uptime()
        };
    }
}

/**
 * Create Express API server
 */
function createAPIServer(forecaster) {
    const app = express();
    app.use(express.json({ limit: '10mb' }));

    // Prometheus metrics endpoint
    app.get('/metrics', (req, res) => {
        res.set('Content-Type', prometheus.register.contentType);
        res.end(prometheus.register.metrics());
    });

    // Health check endpoint
    app.get('/health', (req, res) => {
        const status = forecaster.getHealthStatus();
        res.status(status.healthy ? 200 : 503).json(status);
    });

    // Prediction endpoint
    app.post('/predict', async (req, res) => {
        try {
            const { data, options } = req.body;

            if (!data || !data.y || !data.ds) {
                return res.status(400).json({
                    error: 'Invalid input data. Required: { data: { y: [...], ds: [...] } }'
                });
            }

            const forecasts = await forecaster.predict(data, options);

            res.json({
                success: true,
                forecasts,
                metadata: {
                    timestamp: new Date().toISOString(),
                    model: forecaster.modelVersion
                }
            });

        } catch (error) {
            console.error('Prediction error:', error);
            res.status(500).json({
                error: 'Prediction failed',
                message: error.message
            });
        }
    });

    // Batch prediction endpoint
    app.post('/predict/batch', async (req, res) => {
        try {
            const { requests } = req.body;

            if (!Array.isArray(requests)) {
                return res.status(400).json({
                    error: 'Invalid input. Expected array of prediction requests.'
                });
            }

            // Limit concurrent predictions
            if (requests.length > forecaster.config.maxConcurrentPredictions) {
                return res.status(413).json({
                    error: `Too many requests. Maximum: ${forecaster.config.maxConcurrentPredictions}`
                });
            }

            // Process batch
            const results = await Promise.all(
                requests.map(req => forecaster.predict(req.data, req.options))
            );

            res.json({
                success: true,
                results,
                count: results.length
            });

        } catch (error) {
            console.error('Batch prediction error:', error);
            res.status(500).json({
                error: 'Batch prediction failed',
                message: error.message
            });
        }
    });

    // Model info endpoint
    app.get('/model/info', (req, res) => {
        res.json({
            version: forecaster.modelVersion,
            models: ['LSTM', 'NBEATS'],
            config: {
                frequency: 'H',
                horizon: 24,
                simdEnabled: true
            }
        });
    });

    return app;
}

/**
 * Start production server
 */
async function startProductionServer() {
    try {
        console.log('🚀 Starting production forecasting server...\n');

        // Initialize forecaster
        const forecaster = new ProductionForecaster();
        await forecaster.initialize();

        // Create API server
        const app = createAPIServer(forecaster);
        const PORT = process.env.PORT || 3000;

        app.listen(PORT, () => {
            console.log(`\n✅ Server running on port ${PORT}`);
            console.log(`   Health: http://localhost:${PORT}/health`);
            console.log(`   Metrics: http://localhost:${PORT}/metrics`);
            console.log(`   Predict: POST http://localhost:${PORT}/predict`);
        });

        // Graceful shutdown
        process.on('SIGTERM', async () => {
            console.log('\n🛑 SIGTERM received, shutting down gracefully...');
            await forecaster.saveCheckpoint();
            process.exit(0);
        });

    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
}

// Run if executed directly
if (require.main === module) {
    startProductionServer();
}

module.exports = { ProductionForecaster, createAPIServer };
