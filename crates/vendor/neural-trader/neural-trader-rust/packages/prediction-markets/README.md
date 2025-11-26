# @neural-trader/prediction-markets

[![npm version](https://img.shields.io/npm/v/@neural-trader/prediction-markets.svg)](https://www.npmjs.com/package/@neural-trader/prediction-markets)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ruvnet/neural-trader)

Decentralized prediction markets integration for Neural Trader with Polymarket, PredictIt, and Augur support. Calculate expected value, find mispriced markets, and optimize position sizing with Kelly Criterion.

## Features

- **Polymarket Integration**: Trade on the largest decentralized prediction market
- **PredictIt Support**: Access regulated political prediction markets
- **Augur v2**: Decentralized oracle-based markets
- **Expected Value Calculator**: Find positive EV opportunities
- **Kelly Criterion**: Optimal position sizing for prediction markets
- **Probability Analysis**: Compare market prices to true probabilities
- **Risk Management**: Position limits and drawdown controls
- **Market Maker**: Provide liquidity and earn fees
- **Arbitrage Detection**: Cross-market arbitrage opportunities

## Installation

```bash
npm install @neural-trader/prediction-markets @neural-trader/core @neural-trader/risk
```

## Quick Start

### Expected Value Calculation

```typescript
import { PredictionMarketAnalyzer } from '@neural-trader/prediction-markets';

const analyzer = new PredictionMarketAnalyzer({
  platform: 'polymarket',
  minExpectedValue: 0.05  // 5% minimum EV
});

// Analyze a political market
const market = await analyzer.getMarket('will-trump-win-2024');

// Your model's probability estimate
const trueProbability = 0.52;  // 52% chance

// Market is pricing at 45% (0.45)
const marketPrice = market.yesPrice;  // $0.45 per share

// Calculate expected value
const ev = analyzer.calculateEV(trueProbability, marketPrice);

console.log(`Market Price: $${marketPrice}`);
console.log(`True Probability: ${(trueProbability * 100).toFixed(1)}%`);
console.log(`Expected Value: ${(ev * 100).toFixed(2)}%`);

if (ev > 0.05) {
  console.log('Positive EV opportunity found!');
}
```

### Kelly Criterion for Prediction Markets

```typescript
import { PredictionMarketRisk } from '@neural-trader/prediction-markets';

const risk = new PredictionMarketRisk({
  confidenceLevel: 0.95,
  maxPositionSize: 0.10
});

// Kelly calculation for binary market
const kelly = risk.calculateKelly({
  trueProbability: 0.55,    // Your estimate: 55%
  marketPrice: 0.45,         // Market price: $0.45
  maxPayout: 1.0             // Binary market pays $1
});

console.log(`Full Kelly: ${(kelly.kellyFraction * 100).toFixed(2)}%`);
console.log(`Half Kelly: ${(kelly.halfKelly * 100).toFixed(2)}%`);

// With $10,000 portfolio
const portfolio = 10000;
const positionSize = portfolio * kelly.halfKelly;
console.log(`Recommended position: $${positionSize.toFixed(2)}`);

// Number of shares to buy
const shares = positionSize / marketPrice;
console.log(`Buy ${Math.floor(shares)} shares at $${marketPrice}`);
```

## Real-World Use Cases

### 1. Political Markets (Polymarket)

```typescript
import { PolymarketClient, PredictionMarketAnalyzer } from '@neural-trader/prediction-markets';

const client = new PolymarketClient({
  apiKey: process.env.POLYMARKET_API_KEY,
  network: 'polygon'
});

const analyzer = new PredictionMarketAnalyzer({
  platform: 'polymarket',
  minExpectedValue: 0.03
});

// Analyze 2024 Presidential Election
const market = await client.getMarket('2024-presidential-election');

console.log('=== Market Overview ===');
console.log(`Title: ${market.title}`);
console.log(`Volume: $${market.volume.toLocaleString()}`);
console.log(`Liquidity: $${market.liquidity.toLocaleString()}`);

// Get current prices
const outcomes = await client.getOutcomes(market.id);
for (const outcome of outcomes) {
  console.log(`${outcome.name}: $${outcome.price.toFixed(3)}`);
}

// Your probabilistic model
const modelProbabilities = {
  'Biden': 0.48,
  'Trump': 0.45,
  'Other': 0.07
};

// Find mispriced outcomes
const opportunities = [];
for (const outcome of outcomes) {
  const trueProbability = modelProbabilities[outcome.name];
  const marketPrice = outcome.price;
  const ev = (trueProbability / marketPrice) - 1;

  if (ev > 0.03) {  // 3% minimum EV
    opportunities.push({
      outcome: outcome.name,
      marketPrice,
      trueProbability,
      expectedValue: ev
    });
  }
}

// Rank by expected value
opportunities.sort((a, b) => b.expectedValue - a.expectedValue);

console.log('\n=== Opportunities ===');
for (const opp of opportunities) {
  console.log(`${opp.outcome}:`);
  console.log(`  Market: ${(opp.marketPrice * 100).toFixed(1)}%`);
  console.log(`  Model: ${(opp.trueProbability * 100).toFixed(1)}%`);
  console.log(`  EV: +${(opp.expectedValue * 100).toFixed(2)}%`);
}
```

### 2. Sports Markets with Risk Management

```typescript
import {
  PolymarketClient,
  PredictionMarketRisk
} from '@neural-trader/prediction-markets';
import { PortfolioRiskManager } from '@neural-trader/risk';

const client = new PolymarketClient({ network: 'polygon' });
const marketRisk = new PredictionMarketRisk({ maxPositionSize: 0.15 });
const portfolioRisk = new PortfolioRiskManager({ maxDrawdown: 0.20 });

// Super Bowl market
const market = await client.getMarket('super-bowl-2024-winner');

// Your ML model's predictions
const modelPredictions = {
  'Chiefs': 0.28,
  '49ers': 0.22,
  'Eagles': 0.18,
  'Ravens': 0.15,
  // ... other teams
};

const portfolio = 50000;  // $50,000 portfolio
const positions = [];

for (const [team, probability] of Object.entries(modelPredictions)) {
  const outcome = market.outcomes.find(o => o.name === team);
  if (!outcome) continue;

  const marketPrice = outcome.price;
  const ev = (probability / marketPrice) - 1;

  // Only bet if EV > 5%
  if (ev < 0.05) continue;

  // Kelly calculation
  const kelly = marketRisk.calculateKelly({
    trueProbability: probability,
    marketPrice: marketPrice,
    maxPayout: 1.0
  });

  // Raw position size (Half Kelly)
  let positionSize = portfolio * kelly.halfKelly;

  // Apply risk limits
  positionSize = Math.min(
    positionSize,
    portfolio * 0.15,  // Max 15% per position
    outcome.liquidity * 0.05  // Max 5% of market liquidity
  );

  positions.push({
    team,
    probability,
    marketPrice,
    expectedValue: ev,
    positionSize,
    shares: Math.floor(positionSize / marketPrice)
  });
}

// Check portfolio risk
const totalExposure = positions.reduce((sum, p) => sum + p.positionSize, 0);
const portfolioUtilization = totalExposure / portfolio;

console.log('=== Position Allocation ===');
console.log(`Total Portfolio: $${portfolio.toLocaleString()}`);
console.log(`Total Exposure: $${totalExposure.toLocaleString()}`);
console.log(`Utilization: ${(portfolioUtilization * 100).toFixed(1)}%`);

console.log('\n=== Positions ===');
for (const pos of positions) {
  console.log(`${pos.team}:`);
  console.log(`  Size: $${pos.positionSize.toFixed(2)}`);
  console.log(`  Shares: ${pos.shares}`);
  console.log(`  Price: $${pos.marketPrice.toFixed(3)}`);
  console.log(`  EV: +${(pos.expectedValue * 100).toFixed(2)}%`);
  console.log(`  Expected Profit: $${(pos.positionSize * pos.expectedValue).toFixed(2)}`);
}
```

### 3. Market Making Strategy

```typescript
import { PolymarketMarketMaker } from '@neural-trader/prediction-markets';

const marketMaker = new PolymarketMarketMaker({
  wallet: process.env.WALLET_PRIVATE_KEY,
  network: 'polygon',
  minSpread: 0.02,        // 2% minimum spread
  maxExposure: 10000,     // $10,000 max exposure
  rebalanceThreshold: 0.05 // Rebalance at 5% imbalance
});

// Provide liquidity to political market
await marketMaker.provideLiquidity({
  marketId: 'will-trump-win-2024',
  capital: 5000,
  targetSpread: 0.03,  // 3% spread
  inventory: {
    yes: 2500,  // $2,500 in YES shares
    no: 2500    // $2,500 in NO shares
  }
});

// Monitor and rebalance
marketMaker.on('trade', async (trade) => {
  console.log(`Trade executed: ${trade.side} ${trade.shares} @ $${trade.price}`);

  // Check if rebalancing needed
  const inventory = await marketMaker.getInventory(trade.marketId);
  const imbalance = Math.abs(inventory.yes - inventory.no) / (inventory.yes + inventory.no);

  if (imbalance > 0.05) {
    console.log(`Rebalancing: ${(imbalance * 100).toFixed(1)}% imbalance`);
    await marketMaker.rebalance(trade.marketId);
  }
});

// Track fees earned
marketMaker.on('fee', (fee) => {
  console.log(`Fee earned: $${fee.amount.toFixed(2)}`);
});

// Performance metrics
setInterval(async () => {
  const stats = await marketMaker.getStats();
  console.log('\n=== Market Maker Performance ===');
  console.log(`Total Fees: $${stats.totalFees.toFixed(2)}`);
  console.log(`Volume: $${stats.totalVolume.toLocaleString()}`);
  console.log(`Net P&L: $${stats.netPnL.toFixed(2)}`);
  console.log(`Sharpe Ratio: ${stats.sharpeRatio.toFixed(2)}`);
}, 60000);  // Every minute
```

### 4. Cross-Market Arbitrage

```typescript
import {
  PolymarketClient,
  PredictItClient,
  ArbitrageDetector
} from '@neural-trader/prediction-markets';

const polymarket = new PolymarketClient({ network: 'polygon' });
const predictit = new PredictItClient({ apiKey: process.env.PREDICTIT_API_KEY });
const arbitrage = new ArbitrageDetector({
  minProfitMargin: 0.02,  // 2% minimum profit
  maxSlippage: 0.01        // 1% max slippage
});

// Find arbitrage between platforms
const opportunities = await arbitrage.findCrossMarketArbitrage({
  event: '2024-presidential-election',
  outcome: 'Biden wins',
  platforms: ['polymarket', 'predictit']
});

for (const arb of opportunities) {
  console.log('=== Arbitrage Opportunity ===');
  console.log(`Event: ${arb.event}`);
  console.log(`Outcome: ${arb.outcome}`);

  console.log(`\nBuy on ${arb.buyPlatform}:`);
  console.log(`  Price: $${arb.buyPrice.toFixed(3)}`);
  console.log(`  Size: $${arb.buySize.toFixed(2)}`);

  console.log(`\nSell on ${arb.sellPlatform}:`);
  console.log(`  Price: $${arb.sellPrice.toFixed(3)}`);
  console.log(`  Size: $${arb.sellSize.toFixed(2)}`);

  console.log(`\nProfit:`);
  console.log(`  Margin: ${(arb.profitMargin * 100).toFixed(2)}%`);
  console.log(`  Amount: $${arb.profit.toFixed(2)}`);
  console.log(`  ROI: ${(arb.roi * 100).toFixed(2)}%`);

  // Execute if profitable after fees
  if (arb.netProfit > 50) {  // Minimum $50 profit
    await arbitrage.executeArbitrage(arb);
  }
}
```

## Expected Value (EV) Deep Dive

### Binary Market EV Formula

```
EV = (P_true × Payout) - Cost
EV% = (P_true / P_market) - 1

Where:
P_true = Your probability estimate
P_market = Market price (implied probability)
Payout = $1.00 for binary markets
Cost = Market price
```

### Example Calculations

```typescript
// Scenario 1: Strong Edge
// True probability: 60%, Market price: $0.45
const ev1 = (0.60 / 0.45) - 1;  // 33.3% EV
// For $1000 bet: Expected profit = $333

// Scenario 2: Moderate Edge
// True probability: 52%, Market price: $0.48
const ev2 = (0.52 / 0.48) - 1;  // 8.3% EV
// For $1000 bet: Expected profit = $83

// Scenario 3: Negative Edge (avoid)
// True probability: 45%, Market price: $0.50
const ev3 = (0.45 / 0.50) - 1;  // -10% EV
// For $1000 bet: Expected loss = -$100
```

### Kelly Criterion for Prediction Markets

```typescript
// Simplified Kelly for binary markets
function calculateKelly(trueProbability: number, marketPrice: number): number {
  const q = 1 - trueProbability;  // Probability of losing
  const b = (1 - marketPrice) / marketPrice;  // Odds

  const kellyFraction = (trueProbability * b - q) / b;
  return Math.max(0, kellyFraction);  // Never bet negative
}

// Example: 55% probability, $0.45 market price
const kelly = calculateKelly(0.55, 0.45);
// Full Kelly: 20% of bankroll
// Half Kelly: 10% of bankroll (recommended)
```

## Risk Management

### Position Sizing Framework

```typescript
import {
  PredictionMarketRisk,
  PortfolioRiskManager
} from '@neural-trader/prediction-markets';

const marketRisk = new PredictionMarketRisk({
  maxPositionSize: 0.15,      // 15% max per position
  maxMarketExposure: 0.30,    // 30% max per market
  maxPlatformExposure: 0.50   // 50% max per platform
});

const portfolioRisk = new PortfolioRiskManager({
  maxDrawdown: 0.20,          // 20% max drawdown
  targetVolatility: 0.15,     // 15% target volatility
  maxLeverage: 1.0            // No leverage
});

async function sizePosition(
  market: Market,
  trueProbability: number,
  portfolio: number
): Promise<number> {
  // Kelly calculation
  const kelly = marketRisk.calculateKelly({
    trueProbability,
    marketPrice: market.price,
    maxPayout: 1.0
  });

  // Start with Half Kelly
  let size = portfolio * kelly.halfKelly;

  // Apply position limits
  size = Math.min(
    size,
    portfolio * 0.15,  // 15% max
    market.liquidity * 0.05,  // 5% of liquidity
    10000  // $10k hard cap
  );

  // Check portfolio constraints
  const currentExposure = await portfolioRisk.getTotalExposure();
  const availableCapital = portfolio - currentExposure;
  size = Math.min(size, availableCapital * 0.5);

  // Risk-adjusted based on confidence
  const confidence = await marketRisk.assessConfidence(market);
  size *= confidence;

  return size;
}
```

### Correlation Analysis

```typescript
// Avoid correlated positions
const markets = [
  { id: 'biden-wins-2024', outcome: 'yes' },
  { id: 'democrats-win-senate', outcome: 'yes' },
  { id: 'democrats-win-house', outcome: 'yes' }
];

const correlation = await marketRisk.analyzeCorrelation(markets);

if (correlation.max > 0.7) {
  console.log('Warning: High correlation detected');
  console.log('Consider reducing position sizes');
}
```

## API Reference

### PredictionMarketAnalyzer

```typescript
class PredictionMarketAnalyzer {
  constructor(config: AnalyzerConfig);

  getMarket(id: string): Promise<Market>;

  calculateEV(
    trueProbability: number,
    marketPrice: number
  ): number;

  findOpportunities(options: {
    minEV: number;
    maxPrice: number;
    categories?: string[];
  }): Promise<Opportunity[]>;
}

interface Market {
  id: string;
  title: string;
  description: string;
  outcomes: Outcome[];
  volume: number;
  liquidity: number;
  endDate: Date;
}
```

### PolymarketClient

```typescript
class PolymarketClient {
  constructor(config: ClientConfig);

  getMarket(id: string): Promise<Market>;

  buyShares(
    marketId: string,
    outcome: string,
    shares: number
  ): Promise<Transaction>;

  sellShares(
    marketId: string,
    outcome: string,
    shares: number
  ): Promise<Transaction>;

  getBalance(): Promise<Balance>;
}
```

### PredictionMarketRisk

```typescript
class PredictionMarketRisk {
  constructor(config: RiskConfig);

  calculateKelly(params: {
    trueProbability: number;
    marketPrice: number;
    maxPayout: number;
  }): KellyResult;

  validatePosition(
    size: number,
    market: Market
  ): Promise<ValidationResult>;

  assessConfidence(
    market: Market
  ): Promise<number>;
}
```

## Supported Platforms

| Platform | Type | Network | Features |
|----------|------|---------|----------|
| Polymarket | Decentralized | Polygon | Largest liquidity, lowest fees |
| PredictIt | Regulated | Centralized | Political markets, US-based |
| Augur | Decentralized | Ethereum | Permissionless, any market |
| Gnosis | Decentralized | Gnosis Chain | Conditional tokens |

## Best Practices

1. **Verify Probabilities**: Use multiple models to estimate true probabilities
2. **Account for Fees**: Polymarket: 2%, PredictIt: 10% + withdrawal fees
3. **Check Liquidity**: Avoid markets with <$10k liquidity
4. **Use Half Kelly**: Reduces volatility while maintaining growth
5. **Diversify**: Spread capital across uncorrelated markets
6. **Monitor Events**: Track news that might change probabilities
7. **Exit Strategy**: Set profit targets and stop-losses

## Examples

See `/examples` directory for:
- `polymarket-political-trading.ts` - Political market analysis
- `sports-prediction-markets.ts` - Sports betting markets
- `market-making.ts` - Liquidity provision strategy
- `cross-market-arbitrage.ts` - Multi-platform arbitrage
- `kelly-criterion-prediction.ts` - Optimal position sizing

## Dependencies

- `@neural-trader/core` - Core trading engine
- `@neural-trader/risk` - Risk management
- `ethers` - Ethereum/Polygon interaction
- `@polymarket/sdk` - Polymarket API
- `axios` - HTTP requests

## License

MIT OR Apache-2.0
