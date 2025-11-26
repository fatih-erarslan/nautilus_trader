# Best Practices Guide

Production-ready best practices for the Neural Trader Backend.

## Table of Contents

1. [General Principles](#general-principles)
2. [Trading Best Practices](#trading-best-practices)
3. [Neural Network Best Practices](#neural-network-best-practices)
4. [Syndicate Management](#syndicate-management)
5. [E2B Swarm Deployment](#e2b-swarm-deployment)
6. [Security Best Practices](#security-best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Error Handling](#error-handling)
9. [Testing & Validation](#testing--validation)
10. [Production Deployment](#production-deployment)

---

## General Principles

### 1. Always Initialize First

```javascript
// ✓ Correct
async function main() {
  await initNeuralTrader();
  // ... rest of code
}

// ✗ Wrong
async function main() {
  // Using functions without initialization
  const result = await executeTrade(/*...*/);  // May fail
}
```

### 2. Use Try-Catch Blocks

```javascript
// ✓ Correct
async function trade() {
  try {
    const result = await executeTrade('momentum', 'AAPL', 'buy', 100);
    return result;
  } catch (error) {
    console.error('Trade failed:', error.message);
    // Handle error appropriately
    throw error;  // Re-throw if needed
  }
}

// ✗ Wrong
async function trade() {
  const result = await executeTrade('momentum', 'AAPL', 'buy', 100);
  return result;  // Unhandled errors
}
```

### 3. Validate Inputs

```javascript
// ✓ Correct
async function executeSafeTrade(symbol, quantity, price) {
  // Validate inputs
  if (!symbol || typeof symbol !== 'string') {
    throw new Error('Invalid symbol');
  }

  if (quantity <= 0 || !Number.isInteger(quantity)) {
    throw new Error('Quantity must be positive integer');
  }

  if (price && price <= 0) {
    throw new Error('Price must be positive');
  }

  return await executeTrade('momentum', symbol, 'buy', quantity);
}
```

### 4. Log Everything

```javascript
// ✓ Correct
const { logAuditEvent } = require('@rUv/neural-trader-backend');

async function trade(symbol, quantity) {
  logAuditEvent(
    'info',
    'trading',
    'trade_initiated',
    'pending',
    userId,
    username,
    ipAddress,
    symbol,
    JSON.stringify({ quantity })
  );

  try {
    const result = await executeTrade('momentum', symbol, 'buy', quantity);

    logAuditEvent(
      'info',
      'trading',
      'trade_executed',
      'success',
      userId,
      username,
      ipAddress,
      symbol,
      JSON.stringify({ orderId: result.orderId })
    );

    return result;
  } catch (error) {
    logAuditEvent(
      'error',
      'trading',
      'trade_failed',
      'failure',
      userId,
      username,
      ipAddress,
      symbol,
      JSON.stringify({ error: error.message })
    );
    throw error;
  }
}
```

---

## Trading Best Practices

### 1. Always Simulate First

```javascript
// ✓ Correct
async function smartTrade(strategy, symbol, action, quantity) {
  // Simulate first
  const simulation = await simulateTrade(strategy, symbol, action, true);

  // Check simulation results
  if (simulation.expectedReturn < 5) {
    console.log('Expected return too low - skipping trade');
    return null;
  }

  if (simulation.riskScore > 0.5) {
    console.log('Risk too high - skipping trade');
    return null;
  }

  // Execute only if simulation looks good
  return await executeTrade(strategy, symbol, action, quantity);
}
```

### 2. Set Risk Limits

```javascript
// ✓ Correct
class RiskManagedTrader {
  constructor() {
    this.maxDailyLoss = 5000;
    this.maxPositionSize = 10000;
    this.dailyPnL = 0;
  }

  async trade(symbol, quantity) {
    // Check daily loss limit
    if (Math.abs(this.dailyPnL) >= this.maxDailyLoss) {
      throw new Error('Daily loss limit reached');
    }

    // Check position size
    const positionValue = quantity * 175;  // Estimate
    if (positionValue > this.maxPositionSize) {
      throw new Error('Position size exceeds limit');
    }

    return await executeTrade('momentum', symbol, 'buy', quantity);
  }
}
```

### 3. Use GPU Acceleration

```javascript
// ✓ Correct - Use GPU for intensive operations
const backtest = await runBacktest(
  'momentum',
  'AAPL',
  '2023-01-01',
  '2024-01-01',
  true  // GPU acceleration
);

// ✓ Correct - Use GPU for analysis
const analysis = await quickAnalysis('AAPL', true);

// ✗ Wrong - Missing GPU acceleration (much slower)
const backtest = await runBacktest(
  'momentum',
  'AAPL',
  '2023-01-01',
  '2024-01-01',
  false
);
```

### 4. Backtest Thoroughly

```javascript
// ✓ Correct
async function validateStrategy(strategy, symbol) {
  const periods = [
    { start: '2020-01-01', end: '2020-12-31' },
    { start: '2021-01-01', end: '2021-12-31' },
    { start: '2022-01-01', end: '2022-12-31' },
    { start: '2023-01-01', end: '2023-12-31' }
  ];

  const results = [];

  for (const period of periods) {
    const result = await runBacktest(
      strategy,
      symbol,
      period.start,
      period.end,
      true
    );
    results.push(result);
  }

  // Check consistency
  const avgSharpe = results.reduce((sum, r) => sum + r.sharpeRatio, 0) / results.length;

  if (avgSharpe < 1.0) {
    console.log('⚠ Strategy shows inconsistent performance');
    return false;
  }

  return true;
}
```

### 5. Monitor Portfolio Continuously

```javascript
// ✓ Correct
async function monitorPortfolio() {
  const portfolio = await getPortfolioStatus(true);

  // Check for alerts
  if (portfolio.dailyPnl < -5000) {
    console.log('⚠ ALERT: Daily loss exceeds $5,000');
    // Send notification, stop trading, etc.
  }

  if (portfolio.totalReturn < -10) {
    console.log('⚠ ALERT: Portfolio down more than 10%');
    // Take action
  }

  return portfolio;
}

// Run every 5 minutes
setInterval(monitorPortfolio, 300000);
```

---

## Neural Network Best Practices

### 1. Split Data Properly

```javascript
// ✓ Correct
function prepareData(allData) {
  const total = allData.length;

  const trainEnd = Math.floor(total * 0.7);
  const valEnd = Math.floor(total * 0.85);

  return {
    train: allData.slice(0, trainEnd),        // 70%
    validation: allData.slice(trainEnd, valEnd),  // 15%
    test: allData.slice(valEnd)               // 15%
  };
}
```

### 2. Monitor Overfitting

```javascript
// ✓ Correct
async function trainWithValidation(dataPath, modelType) {
  const training = await neuralTrain(dataPath, modelType, 150, true);

  console.log(`Training Loss: ${training.finalLoss}`);
  console.log(`Validation Accuracy: ${training.validationAccuracy}%`);

  // Check for overfitting
  if (training.validationAccuracy < 85) {
    console.log('⚠ Low validation accuracy - possible overfitting');
    console.log('Consider: reducing complexity, adding regularization, more data');
  }

  return training;
}
```

### 3. Use Ensemble Models

```javascript
// ✓ Correct
async function ensembleForecast(symbol, horizon) {
  // Train multiple models
  const lstm = await neuralTrain('./data.csv', 'lstm', 150, true);
  const gru = await neuralTrain('./data.csv', 'gru', 150, true);
  const transformer = await neuralTrain('./data.csv', 'transformer', 100, true);

  // Get predictions
  const pred1 = await neuralForecast(symbol, horizon, true);
  const pred2 = await neuralForecast(symbol, horizon, true);
  const pred3 = await neuralForecast(symbol, horizon, true);

  // Ensemble (weighted average)
  const ensemble = pred1.predictions.map((p, i) => {
    return (p * 0.4 + pred2.predictions[i] * 0.3 + pred3.predictions[i] * 0.3);
  });

  return ensemble;
}
```

### 4. Optimize Hyperparameters

```javascript
// ✓ Correct
async function findBestModel(dataPath, modelType) {
  const baseModel = await neuralTrain(dataPath, modelType, 100, true);

  const paramRanges = JSON.stringify({
    learning_rate: [0.0001, 0.001, 0.01],
    hidden_units: [64, 128, 256],
    dropout: [0.1, 0.2, 0.3]
  });

  const optimization = await neuralOptimize(
    baseModel.modelId,
    paramRanges,
    true
  );

  console.log('Best parameters found:', JSON.parse(optimization.bestParams));

  return optimization;
}
```

### 5. Retrain Regularly

```javascript
// ✓ Correct
async function maintainModel(modelId) {
  // Retrain weekly with new data
  setInterval(async () => {
    console.log('Retraining model with latest data...');

    // Fetch latest data
    const newData = await fetchLatestMarketData();

    // Retrain
    const training = await neuralTrain(
      './latest_data.csv',
      'lstm',
      150,
      true
    );

    // Evaluate
    const evaluation = await neuralEvaluate(
      training.modelId,
      './test_data.csv',
      true
    );

    console.log(`New model accuracy: ${evaluation.r2Score}`);

    // Update production model if better
    // ...
  }, 7 * 24 * 60 * 60 * 1000);  // Weekly
}
```

---

## Syndicate Management

### 1. Clear Governance Rules

```javascript
// ✓ Correct
const syndicateRules = {
  memberRoles: {
    lead_investor: {
      minContribution: 100000,
      permissions: ['all'],
      profitShare: 'proportional',
      votingWeight: 2.0
    },
    senior_analyst: {
      minContribution: 50000,
      permissions: ['analyze', 'propose', 'vote'],
      profitShare: 'performance_weighted',
      votingWeight: 1.5
    },
    member: {
      minContribution: 10000,
      permissions: ['view', 'vote'],
      profitShare: 'proportional',
      votingWeight: 1.0
    }
  },
  riskLimits: {
    maxSingleBet: 0.05,  // 5% of bankroll
    maxDailyExposure: 0.20,  // 20%
    stopLossDaily: 0.10  // 10%
  },
  distributions: {
    model: 'hybrid',  // 50% capital, 30% performance, 20% equal
    frequency: 'monthly'
  }
};
```

### 2. Use Kelly Criterion

```javascript
// ✓ Correct
async function allocateSyndicateFunds(syndicateId, opportunity) {
  const kelly = await calculateKellyCriterion(
    opportunity.probability,
    opportunity.odds,
    totalBankroll
  );

  // Use fractional Kelly (more conservative)
  const stake = kelly.suggestedStake * 0.5;  // Half Kelly

  // Adjust for confidence
  const adjustedStake = stake * opportunity.confidence;

  return adjustedStake;
}
```

### 3. Democratic Decision Making

```javascript
// ✓ Correct
async function majorDecisions(syndicateId, decision) {
  const votingSystem = new VotingSystem(syndicateId);

  // Create vote for important decisions
  const vote = votingSystem.createVote(
    'strategy_change',
    JSON.stringify(decision),
    leadInvestorId,
    48  // 48 hours
  );

  // Require supermajority for major changes
  const results = JSON.parse(votingSystem.getVoteResults(vote.vote_id));

  if (results.approval_percentage >= 66.67) {  // 2/3 majority
    // Implement change
    return true;
  } else {
    // Rejected
    return false;
  }
}
```

### 4. Transparent Reporting

```javascript
// ✓ Correct
async function monthlyReport(syndicateId) {
  const status = await getSyndicateStatus(syndicateId);
  const manager = new MemberManager(syndicateId);

  const report = {
    period: new Date().toISOString(),
    totalCapital: status.totalCapital,
    activeBets: status.activeBets,
    totalProfit: status.totalProfit,
    roi: status.roi,
    memberPerformance: []
  };

  const members = JSON.parse(manager.listMembers(true));

  for (const member of members) {
    const performance = JSON.parse(
      manager.getMemberPerformanceReport(member.member_id)
    );
    report.memberPerformance.push(performance);
  }

  // Send to all members
  sendReport(report);

  return report;
}
```

---

## E2B Swarm Deployment

### 1. Choose Appropriate Topology

```javascript
// ✓ Correct - Choose based on use case
const topologyGuide = {
  // High coordination, low latency
  'star': {
    useCase: 'High-frequency trading',
    latency: 'Lowest',
    fault_tolerance: 'Low'
  },

  // Balanced
  'mesh': {
    useCase: 'General trading, good balance',
    latency: 'Medium',
    fault_tolerance: 'High'
  },

  // Structured hierarchy
  'hierarchical': {
    useCase: 'Complex strategies with coordination',
    latency: 'Medium',
    fault_tolerance: 'Medium'
  },

  // Sequential processing
  'ring': {
    useCase: 'Pipeline processing',
    latency: 'High',
    fault_tolerance: 'Low'
  }
};

// Example: HFT uses star
const hftSwarm = await initE2bSwarm('star', config);

// Example: General trading uses mesh
const generalSwarm = await initE2bSwarm('mesh', config);
```

### 2. Enable Auto-Scaling

```javascript
// ✓ Correct
const config = JSON.stringify({
  topology: 0,
  maxAgents: 20,
  distributionStrategy: 4,  // Adaptive
  enableGpu: true,
  autoScaling: true,  // ✓ Enable auto-scaling
  minAgents: 5,
  maxMemoryMb: 1024
});
```

### 3. Monitor Health

```javascript
// ✓ Correct
async function continuousMonitoring(swarmId) {
  setInterval(async () => {
    const health = await monitorSwarmHealth();

    if (health.status !== 'healthy') {
      console.log('⚠ Swarm health degraded:', health);

      // Take action
      if (health.errorRate > 10) {
        // Too many errors - restart degraded agents
        await restartFailedAgents(swarmId);
      }

      if (health.cpuUsage > 90) {
        // Scale up
        await scaleSwarm(swarmId, health.healthyAgents + 2);
      }
    }
  }, 60000);  // Every minute
}
```

### 4. Implement Fault Tolerance

```javascript
// ✓ Correct
async function faultTolerantExecution(swarmId, strategy, symbols) {
  let attempts = 0;
  const maxAttempts = 3;

  while (attempts < maxAttempts) {
    try {
      const execution = await executeSwarmStrategy(
        swarmId,
        strategy,
        symbols
      );

      if (execution.status === 'success') {
        return execution;
      }

      attempts++;
      console.log(`Attempt ${attempts} failed, retrying...`);

      // Wait before retry
      await new Promise(r => setTimeout(r, 5000));

    } catch (error) {
      attempts++;
      console.error(`Attempt ${attempts} error:`, error.message);

      if (attempts >= maxAttempts) {
        throw error;
      }
    }
  }
}
```

### 5. Resource Limits

```javascript
// ✓ Correct
const config = JSON.stringify({
  topology: 0,
  maxAgents: 20,
  distributionStrategy: 4,
  enableGpu: true,
  autoScaling: true,
  minAgents: 5,
  maxMemoryMb: 1024,  // ✓ Set memory limit
  timeoutSecs: 300    // ✓ Set timeout
});
```

---

## Security Best Practices

See [Security Guide](./security.md) for complete details.

### Key Points:

1. **Always use authentication**
2. **Enable rate limiting**
3. **Log all security events**
4. **Sanitize inputs**
5. **Use HTTPS in production**
6. **Rotate API keys regularly**
7. **Implement CORS properly**
8. **Monitor for suspicious activity**

---

## Performance Optimization

### 1. Use GPU Acceleration

```javascript
// ✓ Best performance
const result = await quickAnalysis('AAPL', true);
const backtest = await runBacktest('momentum', 'AAPL', start, end, true);
const training = await neuralTrain(data, 'lstm', 150, true);
```

### 2. Batch Operations

```javascript
// ✓ Correct - Batch analysis
const symbols = ['AAPL', 'GOOGL', 'MSFT'];
const results = await Promise.all(
  symbols.map(s => quickAnalysis(s, true))
);

// ✗ Wrong - Sequential
for (const symbol of symbols) {
  await quickAnalysis(symbol, true);  // Slow
}
```

### 3. Cache Results

```javascript
// ✓ Correct
const cache = new Map();

async function cachedAnalysis(symbol) {
  if (cache.has(symbol)) {
    const cached = cache.get(symbol);
    const age = Date.now() - cached.timestamp;

    // Use cache if less than 5 minutes old
    if (age < 300000) {
      return cached.data;
    }
  }

  const result = await quickAnalysis(symbol, true);

  cache.set(symbol, {
    data: result,
    timestamp: Date.now()
  });

  return result;
}
```

---

## Error Handling

### 1. Graceful Degradation

```javascript
// ✓ Correct
async function resilientAnalysis(symbol) {
  try {
    // Try GPU first
    return await quickAnalysis(symbol, true);
  } catch (gpuError) {
    console.warn('GPU failed, falling back to CPU:', gpuError.message);

    try {
      // Fall back to CPU
      return await quickAnalysis(symbol, false);
    } catch (cpuError) {
      console.error('Both GPU and CPU failed:', cpuError.message);
      throw cpuError;
    }
  }
}
```

### 2. Retry Logic

```javascript
// ✓ Correct
async function retryOperation(operation, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      const delay = Math.pow(2, i) * 1000;  // Exponential backoff
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

// Usage
const result = await retryOperation(
  () => executeTrade('momentum', 'AAPL', 'buy', 100)
);
```

---

## Testing & Validation

### 1. Unit Tests

```javascript
// ✓ Correct
const assert = require('assert');

async function testAnalysis() {
  const result = await quickAnalysis('AAPL', false);

  assert(result.symbol === 'AAPL');
  assert(['bullish', 'bearish', 'neutral'].includes(result.trend));
  assert(result.volatility >= 0);

  console.log('✓ Analysis test passed');
}

testAnalysis();
```

### 2. Integration Tests

```javascript
// ✓ Correct
async function testTradingWorkflow() {
  await initNeuralTrader();

  const analysis = await quickAnalysis('AAPL', true);
  assert(analysis.symbol === 'AAPL');

  const simulation = await simulateTrade('momentum', 'AAPL', 'buy', true);
  assert(simulation.strategy === 'momentum');

  console.log('✓ Integration test passed');
}
```

### 3. Load Testing

```javascript
// ✓ Correct
async function loadTest(concurrentRequests = 100) {
  const start = Date.now();

  const promises = Array(concurrentRequests).fill().map(() =>
    quickAnalysis('AAPL', true)
  );

  await Promise.all(promises);

  const duration = Date.now() - start;
  const rps = (concurrentRequests / duration) * 1000;

  console.log(`Load test: ${rps.toFixed(2)} requests/second`);
}
```

---

## Production Deployment

### 1. Environment Variables

```javascript
// ✓ Correct
require('dotenv').config();

const config = {
  jwtSecret: process.env.JWT_SECRET,
  apiKey: process.env.API_KEY,
  enableGpu: process.env.ENABLE_GPU === 'true',
  logLevel: process.env.LOG_LEVEL || 'info'
};

await initNeuralTrader(JSON.stringify(config));
```

### 2. Process Management

```javascript
// ✓ Correct
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');

  // Close positions
  // Stop swarms
  // Save state

  await shutdown();
  process.exit(0);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  // Log error
  // Alert ops team
  process.exit(1);
});
```

### 3. Health Checks

```javascript
// ✓ Correct
app.get('/health', async (req, res) => {
  try {
    const health = await healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});
```

---

## Checklist

Before deploying to production:

- [ ] All inputs validated
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] Security measures in place
- [ ] GPU acceleration enabled
- [ ] Backups configured
- [ ] Monitoring set up
- [ ] Alerts configured
- [ ] Documentation complete
- [ ] Tests passing
- [ ] Performance benchmarked

---

**For more details, see:**
- [Security Guide](./security.md)
- [API Reference](/docs/api-reference/complete-api-reference.md)
- [Examples](/docs/examples/)
