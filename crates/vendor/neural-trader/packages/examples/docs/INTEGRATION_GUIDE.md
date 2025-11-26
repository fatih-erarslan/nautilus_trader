# Integration Guide

This guide demonstrates how to integrate multiple Neural Trader examples together and with external systems.

## Table of Contents

- [Cross-Package Integration](#cross-package-integration)
- [External System Integration](#external-system-integration)
- [Real-time Data Integration](#real-time-data-integration)
- [Database Integration](#database-integration)
- [API Integration](#api-integration)
- [Shared Memory Patterns](#shared-memory-patterns)

---

## Cross-Package Integration

### Market Microstructure + Portfolio Optimization

Combine order book analysis with portfolio optimization:

```typescript
import { createMarketMicrostructure } from '@neural-trader/example-market-microstructure';
import { MeanVarianceOptimizer } from '@neural-trader/example-portfolio-optimization';

async function integratedTrading() {
  // Initialize market microstructure analyzer
  const mm = await createMarketMicrostructure({
    agentDbPath: './market-memory.db',
    useSwarm: true
  });

  // Initialize portfolio optimizer
  const optimizer = new MeanVarianceOptimizer(assets, correlationMatrix);

  // Analyze market conditions
  const orderBook = await fetchOrderBook('BTCUSD');
  const analysis = await mm.analyze(orderBook);

  // Adjust portfolio based on liquidity
  const liquidityFactor = analysis.metrics.liquidityScore;
  const constraints = {
    minWeight: liquidityFactor > 0.7 ? 0.05 : 0.10,
    maxWeight: liquidityFactor > 0.7 ? 0.40 : 0.30,
    targetReturn: 0.14
  };

  const portfolio = optimizer.optimize(constraints);

  // Learn from outcome
  await mm.learn({
    priceMove: portfolio.expectedReturn,
    spreadChange: analysis.metrics.bidAskSpread,
    liquidityChange: liquidityFactor,
    timeHorizon: 86400000 // 1 day
  });

  return { portfolio, analysis };
}
```

### Anomaly Detection + Dynamic Pricing

Use anomaly detection to adjust pricing strategies:

```typescript
import { AnomalyDetector } from '@neural-trader/example-anomaly-detection';
import { DynamicPricingEngine } from '@neural-trader/example-dynamic-pricing';

async function adaptivePricing() {
  const detector = new AnomalyDetector({
    algorithms: ['isolation_forest', 'autoencoder'],
    ensembleMethod: 'voting'
  });

  const pricing = new DynamicPricingEngine({
    algorithm: 'q_learning',
    learningRate: 0.1
  });

  await detector.initialize();
  await pricing.initialize();

  // Monitor demand patterns
  const demandData = await fetchDemandMetrics();
  const anomaly = await detector.detect(demandData);

  if (anomaly.isAnomaly) {
    // Anomalous demand detected - adjust pricing aggressively
    const newPrice = await pricing.optimizePrice({
      currentDemand: demandData.demand,
      competitorPrices: demandData.competitors,
      anomalyConfidence: anomaly.confidence,
      anomalyType: anomaly.anomalyType
    });

    console.log(`Anomaly detected (${anomaly.anomalyType})`);
    console.log(`Adjusted price: ${newPrice} (was ${demandData.currentPrice})`);
  } else {
    // Normal conditions - standard pricing
    const newPrice = await pricing.optimizePrice({
      currentDemand: demandData.demand,
      competitorPrices: demandData.competitors
    });
  }

  return { anomaly, pricing: await pricing.getCurrentStrategy() };
}
```

### Healthcare + Logistics Optimization

Combine patient scheduling with vehicle routing:

```typescript
import { HealthcareOptimizer } from '@neural-trader/example-healthcare-optimization';
import { LogisticsOptimizer } from '@neural-trader/example-logistics-optimization';

async function homeHealthcare() {
  const healthcare = new HealthcareOptimizer('./healthcare-memory.db');
  const logistics = new LogisticsOptimizer({
    agentDbPath: './logistics-memory.db',
    useSwarm: true
  });

  await healthcare.initialize();
  await logistics.initialize();

  // Forecast patient arrivals
  const forecaster = healthcare.createForecaster();
  const expectedPatients = await forecaster.forecast({
    horizon: 24,
    includeEmergencies: true
  });

  // Generate home visit schedule
  const patients = expectedPatients.map((p, i) => ({
    id: `patient-${i}`,
    location: p.location,
    timeWindow: [p.earliestTime, p.latestTime],
    serviceDuration: p.expectedDuration,
    priority: p.urgency
  }));

  // Optimize vehicle routes
  const routes = await logistics.optimizeRoutes({
    locations: patients,
    vehicles: await fetchAvailableVehicles(),
    depots: [{ location: hospitalLocation }]
  });

  // Update both systems with results
  await healthcare.updateSchedule(routes.assignments);
  await logistics.learnFromRoute(routes, {
    completionRate: 0.95,
    avgDelay: 5,
    patientSatisfaction: 0.9
  });

  return { expectedPatients, routes };
}
```

### Energy Grid + Energy Forecasting

Combine forecasting with grid optimization:

```typescript
import { EnergyForecaster } from '@neural-trader/example-energy-forecasting';
import { EnergyGridOptimizer } from '@neural-trader/example-energy-grid-optimization';

async function smartGridManagement() {
  const forecaster = new EnergyForecaster({
    modelType: 'lstm',
    horizons: [1, 6, 24] // 1hr, 6hr, 24hr
  });

  const grid = new EnergyGridOptimizer('./grid-memory.db');

  await forecaster.initialize();
  await grid.initialize();

  // Forecast renewable generation
  const solarForecast = await forecaster.forecastSolar({
    location: gridLocation,
    horizon: 24,
    weatherData: await fetchWeather()
  });

  const windForecast = await forecaster.forecastWind({
    location: gridLocation,
    horizon: 24,
    weatherData: await fetchWeather()
  });

  // Forecast demand
  const loadForecast = await grid.forecastLoad({
    historicalLoad: await fetchHistoricalLoad(),
    temperature: await fetchTemperatureForecast(),
    dayOfWeek: new Date().getDay()
  });

  // Optimize unit commitment
  const schedule = await grid.optimizeUnitCommitment({
    generators: await fetchGenerators(),
    forecast: loadForecast,
    renewableGeneration: {
      solar: solarForecast.values,
      wind: windForecast.values
    },
    batteryStorage: await fetchBatteryStatus()
  });

  // Learn from forecast accuracy
  await forecaster.updateAccuracy({
    solarError: solarForecast.error,
    windError: windForecast.error
  });

  return { schedule, forecasts: { solar: solarForecast, wind: windForecast, load: loadForecast } };
}
```

---

## External System Integration

### Integration with Trading Platforms

```typescript
import { MarketMicrostructure } from '@neural-trader/example-market-microstructure';
import ccxt from 'ccxt';

class TradingPlatformIntegration {
  private exchange: ccxt.Exchange;
  private mm: MarketMicrostructure;

  constructor() {
    this.exchange = new ccxt.binance({
      apiKey: process.env.BINANCE_API_KEY,
      secret: process.env.BINANCE_SECRET
    });
  }

  async initialize() {
    this.mm = await createMarketMicrostructure({
      useSwarm: true,
      agentDbPath: './trading-memory.db'
    });
  }

  async analyzeAndTrade(symbol: string) {
    // Fetch order book from exchange
    const orderBook = await this.exchange.fetchOrderBook(symbol);

    // Convert to our format
    const analysis = await this.mm.analyze({
      bids: orderBook.bids.map(([price, size]) => ({
        price,
        size,
        orders: 1
      })),
      asks: orderBook.asks.map(([price, size]) => ({
        price,
        size,
        orders: 1
      })),
      timestamp: Date.now(),
      symbol
    });

    // Make trading decision
    if (analysis.metrics.orderFlowToxicity < 0.3 &&
        analysis.metrics.liquidityScore > 0.7) {
      // Good conditions for trading
      const order = await this.exchange.createLimitBuyOrder(
        symbol,
        0.001,
        analysis.metrics.midPrice
      );

      return { order, analysis };
    }

    return { order: null, analysis };
  }
}
```

### Integration with Data Warehouses

```typescript
import { PortfolioOptimizationSwarm } from '@neural-trader/example-portfolio-optimization';
import { Client } from 'pg'; // PostgreSQL

class DataWarehouseIntegration {
  private db: Client;
  private swarm: PortfolioOptimizationSwarm;

  async initialize() {
    this.db = new Client({
      host: process.env.DB_HOST,
      database: process.env.DB_NAME,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD
    });

    await this.db.connect();

    this.swarm = new PortfolioOptimizationSwarm();
  }

  async optimizeFromWarehouse(portfolioId: string) {
    // Fetch historical returns from warehouse
    const result = await this.db.query(`
      SELECT
        asset_symbol,
        AVG(daily_return) as expected_return,
        STDDEV(daily_return) as volatility
      FROM asset_returns
      WHERE portfolio_id = $1
        AND date >= NOW() - INTERVAL '252 days'
      GROUP BY asset_symbol
    `, [portfolioId]);

    const assets = result.rows.map(row => ({
      symbol: row.asset_symbol,
      expectedReturn: row.expected_return,
      volatility: row.volatility
    }));

    // Fetch correlation matrix
    const corrResult = await this.db.query(`
      SELECT correlation_matrix
      FROM asset_correlations
      WHERE portfolio_id = $1
      ORDER BY calculated_at DESC
      LIMIT 1
    `, [portfolioId]);

    const correlationMatrix = corrResult.rows[0].correlation_matrix;

    // Run optimization
    const insights = await this.swarm.runBenchmark({
      algorithms: ['mean-variance', 'risk-parity', 'black-litterman'],
      assets,
      correlationMatrix
    });

    // Store results back to warehouse
    await this.db.query(`
      INSERT INTO optimization_results (
        portfolio_id, algorithm, weights, sharpe_ratio, created_at
      ) VALUES ($1, $2, $3, $4, NOW())
    `, [
      portfolioId,
      insights.bestAlgorithm,
      JSON.stringify(insights.bestResult.result.weights),
      insights.bestResult.result.sharpeRatio
    ]);

    return insights;
  }
}
```

### Integration with Message Queues (Kafka)

```typescript
import { AnomalyDetector } from '@neural-trader/example-anomaly-detection';
import { Kafka, Consumer, Producer } from 'kafkajs';

class StreamProcessing {
  private kafka: Kafka;
  private consumer: Consumer;
  private producer: Producer;
  private detector: AnomalyDetector;

  async initialize() {
    this.kafka = new Kafka({
      clientId: 'neural-trader',
      brokers: [process.env.KAFKA_BROKER]
    });

    this.consumer = this.kafka.consumer({ groupId: 'anomaly-detection' });
    this.producer = this.kafka.producer();

    await this.consumer.connect();
    await this.producer.connect();

    this.detector = new AnomalyDetector({
      algorithms: ['isolation_forest'],
      adaptiveThresholds: true
    });

    await this.detector.initialize();
  }

  async processStream() {
    await this.consumer.subscribe({ topic: 'market-data' });

    await this.consumer.run({
      eachMessage: async ({ message }) => {
        const data = JSON.parse(message.value.toString());

        // Detect anomalies in real-time
        const result = await this.detector.detect(data);

        if (result.isAnomaly) {
          // Publish alert to alerts topic
          await this.producer.send({
            topic: 'anomaly-alerts',
            messages: [{
              value: JSON.stringify({
                timestamp: Date.now(),
                data,
                anomaly: result,
                severity: result.confidence > 0.9 ? 'high' : 'medium'
              })
            }]
          });
        }

        // Learn from this data point
        await this.detector.update(data, { isAnomaly: result.isAnomaly });
      }
    });
  }
}
```

---

## Real-time Data Integration

### WebSocket Integration

```typescript
import { MarketMicrostructure } from '@neural-trader/example-market-microstructure';
import WebSocket from 'ws';

class RealtimeAnalysis {
  private ws: WebSocket;
  private mm: MarketMicrostructure;

  async initialize() {
    this.mm = await createMarketMicrostructure({
      useSwarm: true,
      agentDbPath: './realtime-memory.db'
    });

    this.ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth');

    this.ws.on('message', async (data) => {
      const orderBookUpdate = JSON.parse(data.toString());
      await this.processUpdate(orderBookUpdate);
    });

    this.ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  private async processUpdate(update: any) {
    // Convert to our format
    const orderBook = {
      bids: update.bids.map(([price, size]: [string, string]) => ({
        price: parseFloat(price),
        size: parseFloat(size),
        orders: 1
      })),
      asks: update.asks.map(([price, size]: [string, string]) => ({
        price: parseFloat(price),
        size: parseFloat(size),
        orders: 1
      })),
      timestamp: update.E,
      symbol: 'BTCUSDT'
    };

    // Analyze in real-time
    const analysis = await this.mm.analyze(orderBook);

    // Emit events for downstream consumers
    this.emit('analysis', analysis);

    // Check for anomalies
    if (analysis.anomaly?.isAnomaly) {
      this.emit('anomaly', analysis.anomaly);
    }
  }

  private listeners: Record<string, Array<(data: any) => void>> = {};

  on(event: string, callback: (data: any) => void) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  private emit(event: string, data: any) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }
}
```

---

## Database Integration

### PostgreSQL Integration

```typescript
import { SelfLearningOptimizer } from '@neural-trader/example-portfolio-optimization';
import { Pool } from 'pg';

class PostgresIntegration {
  private pool: Pool;
  private optimizer: SelfLearningOptimizer;

  async initialize() {
    this.pool = new Pool({
      host: process.env.POSTGRES_HOST,
      database: process.env.POSTGRES_DB,
      user: process.env.POSTGRES_USER,
      password: process.env.POSTGRES_PASSWORD,
      max: 20
    });

    this.optimizer = new SelfLearningOptimizer('./optimizer-memory.db');
    await this.optimizer.initialize();
  }

  async storeOptimizationResult(result: OptimizationResult) {
    const client = await this.pool.connect();

    try {
      await client.query('BEGIN');

      // Store optimization metadata
      const optResult = await client.query(`
        INSERT INTO optimizations (
          algorithm, sharpe_ratio, risk, return, created_at
        ) VALUES ($1, $2, $3, $4, NOW())
        RETURNING id
      `, [
        result.algorithm,
        result.sharpeRatio,
        result.risk,
        result.expectedReturn
      ]);

      const optId = optResult.rows[0].id;

      // Store weights
      for (let i = 0; i < result.weights.length; i++) {
        await client.query(`
          INSERT INTO portfolio_weights (
            optimization_id, asset_index, weight
          ) VALUES ($1, $2, $3)
        `, [optId, i, result.weights[i]]);
      }

      await client.query('COMMIT');

      return optId;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async loadHistoricalOptimizations(): Promise<OptimizationResult[]> {
    const result = await this.pool.query(`
      SELECT
        o.id,
        o.algorithm,
        o.sharpe_ratio,
        o.risk,
        o.return,
        array_agg(w.weight ORDER BY w.asset_index) as weights
      FROM optimizations o
      JOIN portfolio_weights w ON w.optimization_id = o.id
      WHERE o.created_at >= NOW() - INTERVAL '30 days'
      GROUP BY o.id
      ORDER BY o.created_at DESC
    `);

    return result.rows.map(row => ({
      algorithm: row.algorithm,
      sharpeRatio: row.sharpe_ratio,
      risk: row.risk,
      expectedReturn: row.return,
      weights: row.weights
    }));
  }
}
```

### MongoDB Integration

```typescript
import { PatternLearner } from '@neural-trader/example-market-microstructure';
import { MongoClient, Db } from 'mongodb';

class MongoIntegration {
  private client: MongoClient;
  private db: Db;
  private learner: PatternLearner;

  async initialize() {
    this.client = await MongoClient.connect(process.env.MONGODB_URI);
    this.db = this.client.db('neural-trader');

    this.learner = new PatternLearner({
      agentDbPath: './patterns-memory.db'
    });

    await this.learner.initialize();
  }

  async storePattern(pattern: Pattern) {
    await this.db.collection('patterns').insertOne({
      ...pattern,
      createdAt: new Date(),
      source: 'agentdb'
    });
  }

  async loadRecentPatterns(limit: number = 100): Promise<Pattern[]> {
    return await this.db.collection('patterns')
      .find()
      .sort({ createdAt: -1 })
      .limit(limit)
      .toArray();
  }

  async syncPatterns() {
    // Export patterns from AgentDB
    const patterns = this.learner.getPatterns();

    // Bulk insert to MongoDB
    if (patterns.length > 0) {
      await this.db.collection('patterns').insertMany(
        patterns.map(p => ({
          ...p,
          createdAt: new Date(),
          synced: true
        }))
      );
    }
  }
}
```

---

## API Integration

### REST API Integration

```typescript
import express from 'express';
import { HealthcareOptimizer } from '@neural-trader/example-healthcare-optimization';

const app = express();
app.use(express.json());

let optimizer: HealthcareOptimizer;

app.post('/api/optimize-schedule', async (req, res) => {
  try {
    const { patients, staff, constraints } = req.body;

    if (!optimizer) {
      optimizer = new HealthcareOptimizer('./healthcare-api-memory.db');
      await optimizer.initialize();
    }

    const schedule = await optimizer.optimizeSchedule({
      patients,
      staff,
      constraints
    });

    res.json({
      success: true,
      schedule,
      metrics: optimizer.getMetrics()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/api/forecast/:horizon', async (req, res) => {
  try {
    const horizon = parseInt(req.params.horizon);

    const forecaster = optimizer.createForecaster();
    const forecast = await forecaster.forecast({
      horizon,
      includeEmergencies: true
    });

    res.json({
      success: true,
      forecast
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(3000, () => {
  console.log('API server running on port 3000');
});
```

---

## Shared Memory Patterns

### Cross-Example Memory Sharing

```typescript
import { AgentDB } from 'agentdb';

class SharedMemoryManager {
  private static instance: AgentDB;

  static async getInstance(): Promise<AgentDB> {
    if (!this.instance) {
      this.instance = new AgentDB('./shared-memory.db');
      await this.instance.initialize();
    }
    return this.instance;
  }

  static async storeGlobalInsight(key: string, insight: any) {
    const db = await this.getInstance();
    await db.store({
      collection: 'global_insights',
      data: {
        key,
        insight,
        timestamp: Date.now()
      }
    });
  }

  static async retrieveGlobalInsight(key: string): Promise<any> {
    const db = await this.getInstance();
    const results = await db.query({
      collection: 'global_insights',
      filter: { key }
    });

    return results.length > 0 ? results[0].insight : null;
  }
}

// Usage in market microstructure
const mm = await createMarketMicrostructure();
const analysis = await mm.analyze(orderBook);

if (analysis.anomaly) {
  await SharedMemoryManager.storeGlobalInsight(
    'market_anomaly_detected',
    {
      symbol: orderBook.symbol,
      timestamp: Date.now(),
      anomaly: analysis.anomaly
    }
  );
}

// Usage in portfolio optimization
const recentAnomaly = await SharedMemoryManager.retrieveGlobalInsight(
  'market_anomaly_detected'
);

if (recentAnomaly && Date.now() - recentAnomaly.timestamp < 3600000) {
  // Anomaly detected in last hour - adjust risk
  constraints.maxWeight = 0.20; // More conservative
}
```

---

## Best Practices

1. **Error Handling**: Always wrap integrations in try-catch blocks
2. **Connection Pooling**: Reuse database connections
3. **Async Operations**: Use async/await for non-blocking I/O
4. **Memory Management**: Close connections and cleanup resources
5. **Monitoring**: Log integration metrics and errors
6. **Security**: Never hardcode credentials, use environment variables
7. **Testing**: Create integration tests with mocked external services

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [AgentDB Guide](./AGENTDB_GUIDE.md)

---

Built with ❤️ by the Neural Trader team
