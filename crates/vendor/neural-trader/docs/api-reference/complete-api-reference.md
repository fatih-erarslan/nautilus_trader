# Neural Trader Backend - Complete API Reference

**Version:** 2.1.1
**Platform:** Multi-platform (Linux, macOS, Windows)
**Total Functions:** 70+

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Trading Functions](#core-trading-functions)
4. [Neural Network & AI](#neural-network--ai)
5. [Sports Betting & Arbitrage](#sports-betting--arbitrage)
6. [Syndicate Management](#syndicate-management)
7. [E2B Swarm Deployment](#e2b-swarm-deployment)
8. [Security & Authentication](#security--authentication)
9. [Risk Analysis & Portfolio Management](#risk-analysis--portfolio-management)
10. [Enums & Constants](#enums--constants)
11. [Error Handling](#error-handling)

---

## Overview

The Neural Trader Backend is a high-performance Rust-based trading engine with Node.js bindings (NAPI-RS). It provides comprehensive trading capabilities including:

- **Multi-strategy trading** with GPU acceleration
- **Neural network training & prediction** for market forecasting
- **Sports betting arbitrage** detection
- **Syndicate management** for collaborative trading
- **Distributed swarm deployment** via E2B sandboxes
- **Enterprise security** with RBAC, rate limiting, and audit logging

---

## Quick Start

### Installation

```bash
npm install @rUv/neural-trader-backend
```

### Basic Setup

```javascript
const {
  initNeuralTrader,
  getSystemInfo,
  listStrategies,
  quickAnalysis
} = require('@rUv/neural-trader-backend');

async function main() {
  // Initialize the system
  const initResult = await initNeuralTrader();
  console.log('Initialized:', initResult);

  // Get system capabilities
  const systemInfo = getSystemInfo();
  console.log('System Info:', systemInfo);
  // Output: { version: '2.1.1', features: [...], totalTools: 70+ }

  // List available strategies
  const strategies = await listStrategies();
  console.log('Available strategies:', strategies);
}

main();
```

---

## Core Trading Functions

### 1. `listStrategies(): Promise<StrategyInfo[]>`

Lists all available trading strategies with GPU capability information.

**Returns:** Array of strategy objects

**Example:**
```javascript
const strategies = await listStrategies();
strategies.forEach(s => {
  console.log(`${s.name}: ${s.description} (GPU: ${s.gpuCapable})`);
});
```

**Response:**
```javascript
[
  {
    name: "momentum",
    description: "Momentum-based trend following",
    gpuCapable: true
  },
  {
    name: "mean_reversion",
    description: "Mean reversion strategy",
    gpuCapable: true
  }
]
```

---

### 2. `getStrategyInfo(strategy: string): Promise<string>`

Get detailed information about a specific trading strategy.

**Parameters:**
- `strategy` (string): Strategy name (e.g., "momentum", "mean_reversion")

**Returns:** JSON string with strategy details

**Example:**
```javascript
const info = await getStrategyInfo("momentum");
const details = JSON.parse(info);
console.log(details);
```

---

### 3. `quickAnalysis(symbol: string, useGpu?: boolean): Promise<MarketAnalysis>`

Perform rapid market analysis for a trading symbol.

**Parameters:**
- `symbol` (string): Trading symbol (e.g., "AAPL", "BTC-USD")
- `useGpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:** Market analysis object

**Example:**
```javascript
const analysis = await quickAnalysis("AAPL", true);
console.log(analysis);
```

**Response:**
```javascript
{
  symbol: "AAPL",
  trend: "bullish",
  volatility: 0.25,
  volumeTrend: "increasing",
  recommendation: "BUY"
}
```

---

### 4. `simulateTrade(strategy: string, symbol: string, action: string, useGpu?: boolean): Promise<TradeSimulation>`

Simulate a trade without executing it. Useful for backtesting and risk assessment.

**Parameters:**
- `strategy` (string): Strategy to use
- `symbol` (string): Trading symbol
- `action` (string): "buy" or "sell"
- `useGpu` (boolean, optional): Enable GPU acceleration

**Returns:** Simulation results

**Example:**
```javascript
const simulation = await simulateTrade("momentum", "AAPL", "buy", true);
console.log(`Expected return: ${simulation.expectedReturn}%`);
console.log(`Risk score: ${simulation.riskScore}`);
console.log(`Execution time: ${simulation.executionTimeMs}ms`);
```

**Response:**
```javascript
{
  strategy: "momentum",
  symbol: "AAPL",
  action: "buy",
  expectedReturn: 12.5,
  riskScore: 0.35,
  executionTimeMs: 125
}
```

---

### 5. `getPortfolioStatus(includeAnalytics?: boolean): Promise<PortfolioStatus>`

Get current portfolio status with optional advanced analytics.

**Parameters:**
- `includeAnalytics` (boolean, optional): Include detailed analytics

**Returns:** Portfolio status object

**Example:**
```javascript
const portfolio = await getPortfolioStatus(true);
console.log(`Total Value: $${portfolio.totalValue}`);
console.log(`Daily P&L: $${portfolio.dailyPnl}`);
console.log(`Total Return: ${portfolio.totalReturn}%`);
```

**Response:**
```javascript
{
  totalValue: 125000.50,
  cash: 25000.00,
  positions: 15,
  dailyPnl: 2500.75,
  totalReturn: 18.5
}
```

---

### 6. `executeTrade(strategy: string, symbol: string, action: string, quantity: number, orderType?: string, limitPrice?: number): Promise<TradeExecution>`

Execute a live trade with the specified parameters.

**Parameters:**
- `strategy` (string): Trading strategy to use
- `symbol` (string): Trading symbol
- `action` (string): "buy" or "sell"
- `quantity` (number): Number of shares/contracts
- `orderType` (string, optional): "market" or "limit" (default: "market")
- `limitPrice` (number, optional): Limit price for limit orders

**Returns:** Trade execution result

**Example:**
```javascript
// Market order
const trade = await executeTrade(
  "momentum",
  "AAPL",
  "buy",
  100
);

// Limit order
const limitTrade = await executeTrade(
  "momentum",
  "AAPL",
  "buy",
  100,
  "limit",
  175.50
);

console.log(`Order ID: ${trade.orderId}`);
console.log(`Status: ${trade.status}`);
console.log(`Fill Price: $${trade.fillPrice}`);
```

**Response:**
```javascript
{
  orderId: "ORD-12345678",
  strategy: "momentum",
  symbol: "AAPL",
  action: "buy",
  quantity: 100,
  status: "filled",
  fillPrice: 175.25
}
```

---

### 7. `runBacktest(strategy: string, symbol: string, startDate: string, endDate: string, useGpu?: boolean): Promise<BacktestResult>`

Run comprehensive historical backtesting for a strategy.

**Parameters:**
- `strategy` (string): Strategy name
- `symbol` (string): Trading symbol
- `startDate` (string): Start date (YYYY-MM-DD)
- `endDate` (string): End date (YYYY-MM-DD)
- `useGpu` (boolean, optional): Enable GPU acceleration

**Returns:** Backtest results with performance metrics

**Example:**
```javascript
const backtest = await runBacktest(
  "momentum",
  "AAPL",
  "2023-01-01",
  "2024-01-01",
  true
);

console.log(`Total Return: ${backtest.totalReturn}%`);
console.log(`Sharpe Ratio: ${backtest.sharpeRatio}`);
console.log(`Max Drawdown: ${backtest.maxDrawdown}%`);
console.log(`Win Rate: ${backtest.winRate}%`);
console.log(`Total Trades: ${backtest.totalTrades}`);
```

**Response:**
```javascript
{
  strategy: "momentum",
  symbol: "AAPL",
  startDate: "2023-01-01",
  endDate: "2024-01-01",
  totalReturn: 32.5,
  sharpeRatio: 1.85,
  maxDrawdown: -12.3,
  totalTrades: 156,
  winRate: 58.5
}
```

---

## Neural Network & AI

### 8. `neuralForecast(symbol: string, horizon: number, useGpu?: boolean, confidenceLevel?: number): Promise<NeuralForecast>`

Generate neural network price forecasts with confidence intervals.

**Parameters:**
- `symbol` (string): Trading symbol
- `horizon` (number): Forecast horizon (number of periods ahead)
- `useGpu` (boolean, optional): Enable GPU acceleration
- `confidenceLevel` (number, optional): Confidence level (0.90 - 0.99, default: 0.95)

**Returns:** Forecast predictions with confidence intervals

**Example:**
```javascript
const forecast = await neuralForecast("AAPL", 30, true, 0.95);

console.log(`Symbol: ${forecast.symbol}`);
console.log(`Horizon: ${forecast.horizon} days`);
console.log(`Model Accuracy: ${forecast.modelAccuracy}%`);

forecast.predictions.forEach((pred, i) => {
  const ci = forecast.confidenceIntervals[i];
  console.log(`Day ${i+1}: $${pred} (${ci.lower} - ${ci.upper})`);
});
```

**Response:**
```javascript
{
  symbol: "AAPL",
  horizon: 30,
  predictions: [175.50, 176.25, 177.10, ...],
  confidenceIntervals: [
    { lower: 170.25, upper: 180.75 },
    { lower: 171.00, upper: 181.50 },
    ...
  ],
  modelAccuracy: 89.5
}
```

---

### 9. `neuralTrain(dataPath: string, modelType: string, epochs?: number, useGpu?: boolean): Promise<TrainingResult>`

Train a neural forecasting model with custom data.

**Parameters:**
- `dataPath` (string): Path to training data (CSV format)
- `modelType` (string): Model architecture ("lstm", "gru", "transformer")
- `epochs` (number, optional): Training epochs (default: 100)
- `useGpu` (boolean, optional): Enable GPU training

**Returns:** Training result with model ID

**Example:**
```javascript
const training = await neuralTrain(
  "./data/aapl_historical.csv",
  "lstm",
  150,
  true
);

console.log(`Model ID: ${training.modelId}`);
console.log(`Training Time: ${training.trainingTimeMs}ms`);
console.log(`Final Loss: ${training.finalLoss}`);
console.log(`Validation Accuracy: ${training.validationAccuracy}%`);
```

**Response:**
```javascript
{
  modelId: "MDL-abc123",
  modelType: "lstm",
  trainingTimeMs: 45000,
  finalLoss: 0.0025,
  validationAccuracy: 92.3
}
```

---

### 10. `neuralEvaluate(modelId: string, testData: string, useGpu?: boolean): Promise<EvaluationResult>`

Evaluate a trained neural model on test data.

**Parameters:**
- `modelId` (string): Model identifier from training
- `testData` (string): Path to test dataset
- `useGpu` (boolean, optional): Use GPU for evaluation

**Returns:** Evaluation metrics

**Example:**
```javascript
const evaluation = await neuralEvaluate(
  "MDL-abc123",
  "./data/test_set.csv",
  true
);

console.log(`Test Samples: ${evaluation.testSamples}`);
console.log(`MAE: ${evaluation.mae}`);
console.log(`RMSE: ${evaluation.rmse}`);
console.log(`MAPE: ${evaluation.mape}%`);
console.log(`R² Score: ${evaluation.r2Score}`);
```

---

### 11. `neuralModelStatus(modelId?: string): Promise<ModelStatus[]>`

Get status of neural models (all models or specific model).

**Parameters:**
- `modelId` (string, optional): Specific model ID (omit for all models)

**Returns:** Array of model status objects

**Example:**
```javascript
// Get all models
const allModels = await neuralModelStatus();

// Get specific model
const model = await neuralModelStatus("MDL-abc123");
```

---

### 12. `neuralOptimize(modelId: string, parameterRanges: string, useGpu?: boolean): Promise<OptimizationResult>`

Optimize neural model hyperparameters using grid/random search.

**Parameters:**
- `modelId` (string): Model to optimize
- `parameterRanges` (string): JSON string of parameter ranges
- `useGpu` (boolean, optional): Use GPU for optimization

**Returns:** Optimization results with best parameters

**Example:**
```javascript
const paramRanges = JSON.stringify({
  learning_rate: [0.001, 0.01, 0.1],
  hidden_units: [64, 128, 256],
  dropout: [0.1, 0.2, 0.3]
});

const optimization = await neuralOptimize(
  "MDL-abc123",
  paramRanges,
  true
);

console.log(`Best Parameters: ${optimization.bestParams}`);
console.log(`Best Score: ${optimization.bestScore}`);
console.log(`Trials Completed: ${optimization.trialsCompleted}`);
```

---

### 13. `neuralBacktest(modelId: string, startDate: string, endDate: string, benchmark?: string, useGpu?: boolean): Promise<BacktestResult>`

Backtest a neural model against historical data.

**Parameters:**
- `modelId` (string): Trained model ID
- `startDate` (string): Start date (YYYY-MM-DD)
- `endDate` (string): End date (YYYY-MM-DD)
- `benchmark` (string, optional): Benchmark index (default: "sp500")
- `useGpu` (boolean, optional): Use GPU acceleration

**Returns:** Backtest performance metrics

---

## Sports Betting & Arbitrage

### 14. `getSportsEvents(sport: string, daysAhead?: number): Promise<SportsEvent[]>`

Get upcoming sports events for betting analysis.

**Parameters:**
- `sport` (string): Sport type ("nfl", "nba", "mlb", "soccer", etc.)
- `daysAhead` (number, optional): Days to look ahead (default: 7)

**Returns:** Array of upcoming events

**Example:**
```javascript
const events = await getSportsEvents("nfl", 3);

events.forEach(event => {
  console.log(`${event.homeTeam} vs ${event.awayTeam}`);
  console.log(`Start: ${event.startTime}`);
  console.log(`Event ID: ${event.eventId}`);
});
```

---

### 15. `getSportsOdds(sport: string): Promise<BettingOdds[]>`

Fetch current betting odds from multiple bookmakers.

**Parameters:**
- `sport` (string): Sport type

**Returns:** Array of odds from different bookmakers

**Example:**
```javascript
const odds = await getSportsOdds("nfl");

odds.forEach(odd => {
  console.log(`${odd.market} - ${odd.bookmaker}`);
  console.log(`Home: ${odd.homeOdds} | Away: ${odd.awayOdds}`);
});
```

---

### 16. `findSportsArbitrage(sport: string, minProfitMargin?: number): Promise<ArbitrageOpportunity[]>`

Detect arbitrage opportunities across bookmakers.

**Parameters:**
- `sport` (string): Sport type
- `minProfitMargin` (number, optional): Minimum profit margin % (default: 0.01 = 1%)

**Returns:** Array of arbitrage opportunities

**Example:**
```javascript
const arbs = await findSportsArbitrage("nfl", 0.02); // 2% min profit

arbs.forEach(arb => {
  console.log(`Event: ${arb.eventId}`);
  console.log(`Profit Margin: ${arb.profitMargin}%`);
  console.log(`Bet Home: ${arb.betHome.bookmaker} @ ${arb.betHome.odds} - $${arb.betHome.stake}`);
  console.log(`Bet Away: ${arb.betAway.bookmaker} @ ${arb.betAway.odds} - $${arb.betAway.stake}`);
});
```

**Response:**
```javascript
{
  eventId: "evt-12345",
  profitMargin: 2.5,
  betHome: {
    bookmaker: "Bookmaker A",
    odds: 2.10,
    stake: 476.19
  },
  betAway: {
    bookmaker: "Bookmaker B",
    odds: 2.15,
    stake: 523.81
  }
}
```

---

### 17. `calculateKellyCriterion(probability: number, odds: number, bankroll: number): Promise<KellyCriterion>`

Calculate optimal bet size using Kelly Criterion formula.

**Parameters:**
- `probability` (number): Win probability (0.0 - 1.0)
- `odds` (number): Decimal odds
- `bankroll` (number): Total bankroll

**Returns:** Kelly calculation results

**Example:**
```javascript
const kelly = await calculateKellyCriterion(0.55, 2.0, 10000);

console.log(`Kelly Fraction: ${kelly.kellyFraction}`);
console.log(`Suggested Stake: $${kelly.suggestedStake}`);
```

**Response:**
```javascript
{
  probability: 0.55,
  odds: 2.0,
  bankroll: 10000,
  kellyFraction: 0.10,
  suggestedStake: 1000
}
```

---

### 18. `executeSportsBet(marketId: string, selection: string, stake: number, odds: number, validateOnly?: boolean): Promise<BetExecution>`

Execute or validate a sports bet.

**Parameters:**
- `marketId` (string): Market identifier
- `selection` (string): Selection (team/outcome)
- `stake` (number): Bet amount
- `odds` (number): Decimal odds
- `validateOnly` (boolean, optional): Only validate without executing

**Returns:** Bet execution result

**Example:**
```javascript
// Validate first
const validation = await executeSportsBet(
  "mkt-12345",
  "Team A",
  100,
  2.50,
  true
);

// Execute if valid
if (validation.status === "valid") {
  const bet = await executeSportsBet(
    "mkt-12345",
    "Team A",
    100,
    2.50,
    false
  );
  console.log(`Bet placed: ${bet.betId}`);
  console.log(`Potential return: $${bet.potentialReturn}`);
}
```

---

## Syndicate Management

### 19. `createSyndicate(syndicateId: string, name: string, description?: string): Promise<Syndicate>`

Create a new investment syndicate for collaborative trading.

**Parameters:**
- `syndicateId` (string): Unique syndicate identifier
- `name` (string): Syndicate name
- `description` (string, optional): Description

**Returns:** Syndicate object

**Example:**
```javascript
const syndicate = await createSyndicate(
  "syn-001",
  "Elite Traders Group",
  "Professional trading syndicate focusing on momentum strategies"
);

console.log(`Created: ${syndicate.name}`);
console.log(`ID: ${syndicate.syndicateId}`);
console.log(`Members: ${syndicate.memberCount}`);
```

---

### 20. `addSyndicateMember(syndicateId: string, name: string, email: string, role: string, initialContribution: number): Promise<SyndicateMember>`

Add a member to a syndicate with initial capital contribution.

**Parameters:**
- `syndicateId` (string): Syndicate ID
- `name` (string): Member name
- `email` (string): Member email
- `role` (string): Member role ("lead_investor", "senior_analyst", etc.)
- `initialContribution` (number): Initial capital in dollars

**Returns:** Member object

**Example:**
```javascript
const member = await addSyndicateMember(
  "syn-001",
  "John Doe",
  "john@example.com",
  "senior_analyst",
  50000
);

console.log(`Member ID: ${member.memberId}`);
console.log(`Role: ${member.role}`);
console.log(`Contribution: $${member.contribution}`);
console.log(`Profit Share: ${member.profitShare}%`);
```

---

### 21. `getSyndicateStatus(syndicateId: string): Promise<SyndicateStatus>`

Get current syndicate status and performance metrics.

**Parameters:**
- `syndicateId` (string): Syndicate identifier

**Returns:** Syndicate status

**Example:**
```javascript
const status = await getSyndicateStatus("syn-001");

console.log(`Total Capital: $${status.totalCapital}`);
console.log(`Active Bets: ${status.activeBets}`);
console.log(`Total Profit: $${status.totalProfit}`);
console.log(`ROI: ${status.roi}%`);
console.log(`Members: ${status.memberCount}`);
```

---

### 22. `allocateSyndicateFunds(syndicateId: string, opportunities: string, strategy?: string): Promise<FundAllocation>`

Allocate syndicate funds across betting opportunities using advanced strategies.

**Parameters:**
- `syndicateId` (string): Syndicate ID
- `opportunities` (string): JSON array of betting opportunities
- `strategy` (string, optional): Allocation strategy ("kelly_criterion", "fixed_percentage", etc.)

**Returns:** Fund allocation result

**Example:**
```javascript
const opportunities = JSON.stringify([
  {
    sport: "nfl",
    event: "Patriots vs Chiefs",
    betType: "moneyline",
    selection: "Patriots",
    odds: 2.5,
    probability: 0.45,
    edge: 0.125
  },
  {
    sport: "nba",
    event: "Lakers vs Warriors",
    betType: "spread",
    selection: "Lakers -3.5",
    odds: 1.91,
    probability: 0.55,
    edge: 0.05
  }
]);

const allocation = await allocateSyndicateFunds(
  "syn-001",
  opportunities,
  "kelly_criterion"
);

console.log(`Total Allocated: $${allocation.totalAllocated}`);
console.log(`Expected Return: ${allocation.expectedReturn}%`);
console.log(`Risk Score: ${allocation.riskScore}`);

allocation.allocations.forEach(alloc => {
  console.log(`Opportunity ${alloc.opportunityId}: $${alloc.amount}`);
});
```

---

### 23. `distributeSyndicateProfits(syndicateId: string, totalProfit: number, model?: string): Promise<ProfitDistribution>`

Distribute profits to syndicate members based on chosen model.

**Parameters:**
- `syndicateId` (string): Syndicate ID
- `totalProfit` (number): Total profit to distribute
- `model` (string, optional): Distribution model ("proportional", "performance_weighted", "hybrid")

**Returns:** Profit distribution breakdown

**Example:**
```javascript
const distribution = await distributeSyndicateProfits(
  "syn-001",
  25000,
  "hybrid"
);

console.log(`Total Distributed: $${distribution.totalProfit}`);
console.log(`Distribution Date: ${distribution.distributionDate}`);

distribution.distributions.forEach(dist => {
  console.log(`Member ${dist.memberId}: $${dist.amount} (${dist.percentage}%)`);
});
```

---

### 24. Fund Allocation Engine (Class)

Advanced bankroll management with automated allocation.

**Example:**
```javascript
const { FundAllocationEngine, AllocationStrategy } = require('@rUv/neural-trader-backend');

const engine = new FundAllocationEngine("syn-001", "100000");

const opportunity = {
  sport: "nfl",
  event: "Patriots vs Chiefs",
  betType: "moneyline",
  selection: "Patriots",
  odds: 2.5,
  probability: 0.48,
  edge: 0.20,
  confidence: 0.75,
  modelAgreement: 0.85,
  timeUntilEventSecs: 86400,
  liquidity: 0.95,
  isLive: false,
  isParlay: false
};

const allocation = engine.allocateFunds(
  opportunity,
  AllocationStrategy.KellyCriterion
);

console.log(`Allocated: $${allocation.amount}`);
console.log(`% of Bankroll: ${allocation.percentageOfBankroll}%`);
console.log(`Reasoning: ${allocation.reasoning}`);
console.log(`Approval Required: ${allocation.approvalRequired}`);

if (allocation.warnings.length > 0) {
  console.log('Warnings:', allocation.warnings);
}

// Update exposure after bet
engine.updateExposure(JSON.stringify({
  sport: "nfl",
  amount: allocation.amount,
  isLive: false
}));

// Get exposure summary
const exposure = engine.getExposureSummary();
console.log('Exposure:', JSON.parse(exposure));
```

---

### 25. Profit Distribution System (Class)

**Example:**
```javascript
const { ProfitDistributionSystem, DistributionModel } = require('@rUv/neural-trader-backend');

const distributor = new ProfitDistributionSystem("syn-001");

const members = JSON.stringify([
  {
    member_id: "mem-001",
    capital_contribution: 50000,
    performance_score: 0.85,
    bets_won: 120,
    bets_lost: 80,
    tier: "Gold"
  },
  {
    member_id: "mem-002",
    capital_contribution: 30000,
    performance_score: 0.72,
    bets_won: 90,
    bets_lost: 110,
    tier: "Silver"
  }
]);

const distribution = distributor.calculateDistribution(
  "25000",
  members,
  DistributionModel.Hybrid
);

console.log('Distribution:', JSON.parse(distribution));
```

---

### 26. Member Manager (Class)

Comprehensive member management with performance tracking.

**Example:**
```javascript
const { MemberManager, MemberRole } = require('@rUv/neural-trader-backend');

const manager = new MemberManager("syn-001");

// Add new member
const memberJson = manager.addMember(
  "John Doe",
  "john@example.com",
  MemberRole.SeniorAnalyst,
  "50000"
);
const member = JSON.parse(memberJson);
console.log(`Added member: ${member.member_id}`);

// Update role
manager.updateMemberRole(
  member.member_id,
  MemberRole.LeadInvestor,
  "admin-001"
);

// Track bet outcome
manager.trackBetOutcome(
  member.member_id,
  JSON.stringify({
    bet_id: "bet-001",
    won: true,
    profit: 500,
    stake: 1000
  })
);

// Get performance report
const report = manager.getMemberPerformanceReport(member.member_id);
console.log('Performance:', JSON.parse(report));

// Get total capital
const totalCapital = manager.getTotalCapital();
console.log(`Total Syndicate Capital: $${totalCapital}`);

// List all members
const allMembers = manager.listMembers(true); // active only
console.log('Members:', JSON.parse(allMembers));
```

---

### 27. Voting System (Class)

Democratic decision-making for syndicates.

**Example:**
```javascript
const { VotingSystem } = require('@rUv/neural-trader-backend');

const voting = new VotingSystem("syn-001");

// Create a vote
const voteJson = voting.createVote(
  "strategy_change",
  JSON.stringify({
    proposal: "Switch to more aggressive risk profile",
    new_max_bet: 0.10,
    current_max_bet: 0.05
  }),
  "mem-001",
  48 // 48-hour voting period
);
const vote = JSON.parse(voteJson);
console.log(`Created vote: ${vote.vote_id}`);

// Cast votes
voting.castVote(vote.vote_id, "mem-001", "approve", 1.5);
voting.castVote(vote.vote_id, "mem-002", "approve", 1.0);
voting.castVote(vote.vote_id, "mem-003", "reject", 1.0);

// Get results
const results = voting.getVoteResults(vote.vote_id);
console.log('Results:', JSON.parse(results));

// Finalize vote
const finalResult = voting.finalizeVote(vote.vote_id);
console.log('Final:', JSON.parse(finalResult));

// List active votes
const activeVotes = voting.listActiveVotes();
console.log('Active Votes:', JSON.parse(activeVotes));
```

---

### 28. Collaboration Hub (Class)

Communication and coordination platform for syndicate members.

**Example:**
```javascript
const { CollaborationHub } = require('@rUv/neural-trader-backend');

const hub = new CollaborationHub("syn-001");

// Create channel
const channelJson = hub.createChannel(
  "trade-alerts",
  "Real-time trading alerts and opportunities",
  "alerts"
);
const channel = JSON.parse(channelJson);

// Add members
hub.addMemberToChannel(channel.channel_id, "mem-001");
hub.addMemberToChannel(channel.channel_id, "mem-002");

// Post message
const messageId = hub.postMessage(
  channel.channel_id,
  "mem-001",
  "Found great arbitrage opportunity in NFL game Patriots vs Chiefs!",
  "alert",
  []
);

// Get messages
const messages = hub.getChannelMessages(channel.channel_id, 50);
console.log('Messages:', JSON.parse(messages));

// List channels
const channels = hub.listChannels();
console.log('Channels:', JSON.parse(channels));
```

---

## E2B Swarm Deployment

### 29. `initE2bSwarm(topology: string, config: string): Promise<SwarmInit>`

Initialize distributed trading swarm with E2B sandboxes.

**Parameters:**
- `topology` (string): "mesh", "hierarchical", "ring", or "star"
- `config` (string): JSON configuration

**Returns:** Swarm initialization result

**Example:**
```javascript
const { SwarmTopology, DistributionStrategy } = require('@rUv/neural-trader-backend');

const config = JSON.stringify({
  topology: SwarmTopology.Mesh,
  maxAgents: 10,
  distributionStrategy: DistributionStrategy.Adaptive,
  enableGpu: true,
  autoScaling: true,
  minAgents: 3,
  maxMemoryMb: 512,
  timeoutSecs: 300
});

const swarm = await initE2bSwarm("mesh", config);

console.log(`Swarm ID: ${swarm.swarmId}`);
console.log(`Topology: ${swarm.topology}`);
console.log(`Agent Count: ${swarm.agentCount}`);
console.log(`Status: ${swarm.status}`);
```

---

### 30. `deployTradingAgent(sandboxId: string, agentType: string, symbols: string[], params?: string): Promise<AgentDeployment>`

Deploy a specialized trading agent to a sandbox.

**Parameters:**
- `sandboxId` (string): Target sandbox ID
- `agentType` (string): Agent type ("momentum", "mean_reversion", "neural", etc.)
- `symbols` (string[]): Trading symbols to monitor
- `params` (string, optional): JSON strategy parameters

**Returns:** Agent deployment details

**Example:**
```javascript
const agent = await deployTradingAgent(
  "sandbox-abc123",
  "momentum",
  ["AAPL", "GOOGL", "MSFT"],
  JSON.stringify({
    lookback_period: 20,
    threshold: 0.02,
    stop_loss: 0.05
  })
);

console.log(`Agent ID: ${agent.agentId}`);
console.log(`Status: ${agent.status}`);
console.log(`Symbols: ${agent.symbols.join(', ')}`);
```

---

### 31. `getSwarmStatus(swarmId?: string): Promise<SwarmStatus>`

Get real-time swarm health and performance.

**Example:**
```javascript
const status = await getSwarmStatus("swarm-xyz789");

console.log(`Active Agents: ${status.activeAgents}`);
console.log(`Idle Agents: ${status.idleAgents}`);
console.log(`Failed Agents: ${status.failedAgents}`);
console.log(`Total Trades: ${status.totalTrades}`);
console.log(`Total P&L: $${status.totalPnl}`);
console.log(`Uptime: ${status.uptimeSecs}s`);
```

---

### 32. `scaleSwarm(swarmId: string, targetCount: number): Promise<ScaleResult>`

Dynamically scale swarm agent count.

**Example:**
```javascript
const scaleResult = await scaleSwarm("swarm-xyz789", 15);

console.log(`Previous Count: ${scaleResult.previousCount}`);
console.log(`New Count: ${scaleResult.newCount}`);
console.log(`Agents Added: ${scaleResult.agentsAdded}`);
console.log(`Agents Removed: ${scaleResult.agentsRemoved}`);
console.log(`Status: ${scaleResult.status}`);
```

---

### 33. `executeSwarmStrategy(swarmId: string, strategy: string, symbols: string[]): Promise<SwarmExecution>`

Execute a trading strategy across the entire swarm.

**Example:**
```javascript
const execution = await executeSwarmStrategy(
  "swarm-xyz789",
  "momentum",
  ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
);

console.log(`Execution ID: ${execution.executionId}`);
console.log(`Strategy: ${execution.strategy}`);
console.log(`Agents Used: ${execution.agentsUsed}`);
console.log(`Expected Return: ${execution.expectedReturn}%`);
console.log(`Risk Score: ${execution.riskScore}`);
```

---

### 34. `getSwarmPerformance(swarmId: string): Promise<SwarmPerformance>`

Get comprehensive swarm performance analytics.

**Example:**
```javascript
const performance = await getSwarmPerformance("swarm-xyz789");

console.log(`Total Return: ${performance.totalReturn}%`);
console.log(`Sharpe Ratio: ${performance.sharpeRatio}`);
console.log(`Max Drawdown: ${performance.maxDrawdown}%`);
console.log(`Win Rate: ${performance.winRate}%`);
console.log(`Total Trades: ${performance.totalTrades}`);
console.log(`Avg Trade Duration: ${performance.avgTradeDuration}s`);
console.log(`Profit Factor: ${performance.profitFactor}`);
```

---

### 35. `monitorSwarmHealth(): Promise<SwarmHealth>`

Monitor system-wide swarm health.

**Example:**
```javascript
const health = await monitorSwarmHealth();

console.log(`Status: ${health.status}`);
console.log(`CPU Usage: ${health.cpuUsage}%`);
console.log(`Memory Usage: ${health.memoryUsage}%`);
console.log(`Avg Response Time: ${health.avgResponseTime}ms`);
console.log(`Healthy Agents: ${health.healthyAgents}`);
console.log(`Degraded Agents: ${health.degradedAgents}`);
console.log(`Error Rate: ${health.errorRate}/min`);
```

---

## Security & Authentication

### 36. `initAuth(jwtSecret?: string): string`

Initialize the authentication system.

**Example:**
```javascript
const result = initAuth("your-secret-key-here");
console.log(result); // "Authentication system initialized"
```

---

### 37. `createApiKey(username: string, role: string, rateLimit?: number, expiresInDays?: number): string`

Create a new API key with RBAC.

**Parameters:**
- `username` (string): Username
- `role` (string): "read_only", "user", "admin", or "service"
- `rateLimit` (number, optional): Requests per minute (default: 60)
- `expiresInDays` (number, optional): Expiration in days (default: 365)

**Returns:** API key string

**Example:**
```javascript
const apiKey = createApiKey("john_doe", "user", 100, 90);
console.log(`API Key: ${apiKey}`);
// Save this securely - it won't be shown again!
```

---

### 38. `validateApiKey(apiKey: string): AuthUser`

Validate an API key and get user information.

**Example:**
```javascript
try {
  const user = validateApiKey(apiKey);
  console.log(`User ID: ${user.userId}`);
  console.log(`Username: ${user.username}`);
  console.log(`Role: ${user.role}`);
  console.log(`Last Activity: ${user.lastActivity}`);
} catch (error) {
  console.error('Invalid API key');
}
```

---

### 39. `generateToken(apiKey: string): string`

Generate a JWT token from an API key.

**Example:**
```javascript
const token = generateToken(apiKey);
console.log(`JWT Token: ${token}`);
// Use this token for authenticated requests
```

---

### 40. `validateToken(token: string): AuthUser`

Validate a JWT token.

**Example:**
```javascript
try {
  const user = validateToken(jwtToken);
  console.log('Token valid for user:', user.username);
} catch (error) {
  console.error('Invalid or expired token');
}
```

---

### 41. `checkAuthorization(apiKey: string, operation: string, requiredRole: string): boolean`

Check if a user is authorized for an operation.

**Example:**
```javascript
const canTrade = checkAuthorization(
  apiKey,
  "execute_trade",
  "user"
);

if (canTrade) {
  // Execute trade
} else {
  console.error('Insufficient permissions');
}
```

---

### 42. Rate Limiting

**Example:**
```javascript
const { initRateLimiter, checkRateLimit, getRateLimitStats } = require('@rUv/neural-trader-backend');

// Initialize with custom config
initRateLimiter({
  maxRequestsPerMinute: 100,
  burstSize: 20,
  windowDurationSecs: 60
});

// Check rate limit
const allowed = checkRateLimit("user-123", 1);
if (allowed) {
  // Process request
} else {
  console.log('Rate limit exceeded');
}

// Get stats
const stats = getRateLimitStats("user-123");
console.log(`Tokens Available: ${stats.tokensAvailable}/${stats.maxTokens}`);
console.log(`Success Rate: ${stats.successRate}%`);
```

---

### 43. DDoS Protection

**Example:**
```javascript
const { checkDdosProtection, blockIp, getBlockedIps } = require('@rUv/neural-trader-backend');

// Check if IP is safe
const safe = checkDdosProtection("192.168.1.100", 50);
if (!safe) {
  blockIp("192.168.1.100");
}

// Get blocked IPs
const blockedIps = getBlockedIps();
console.log('Blocked IPs:', blockedIps);
```

---

### 44. Audit Logging

**Example:**
```javascript
const {
  initAuditLogger,
  logAuditEvent,
  getAuditEvents,
  getAuditStatistics
} = require('@rUv/neural-trader-backend');

// Initialize
initAuditLogger(10000, true, true);

// Log event
logAuditEvent(
  "security",
  "authentication",
  "api_key_created",
  "success",
  "user-123",
  "john_doe",
  "192.168.1.100",
  "api_keys",
  JSON.stringify({ role: "user", rate_limit: 100 })
);

// Get recent events
const events = getAuditEvents(50);
events.forEach(event => {
  console.log(`[${event.timestamp}] ${event.action} - ${event.outcome}`);
});

// Get statistics
const stats = getAuditStatistics();
console.log('Audit Stats:', JSON.parse(stats));
```

---

### 45. Input Sanitization

**Example:**
```javascript
const {
  sanitizeInput,
  validateTradingParams,
  validateEmailFormat,
  checkSecurityThreats
} = require('@rUv/neural-trader-backend');

// Sanitize user input
const clean = sanitizeInput("<script>alert('xss')</script>");
console.log(clean); // Escaped output

// Validate trading parameters
const valid = validateTradingParams("AAPL", 100, 175.50);
if (!valid) {
  console.error('Invalid trading parameters');
}

// Validate email
if (!validateEmailFormat(email)) {
  console.error('Invalid email format');
}

// Check for security threats
const threats = checkSecurityThreats(userInput);
if (threats.length > 0) {
  console.error('Security threats detected:', threats);
}
```

---

## Risk Analysis & Portfolio Management

### 46. `riskAnalysis(portfolio: string, useGpu?: boolean): Promise<RiskAnalysis>`

Comprehensive portfolio risk analysis with GPU acceleration.

**Example:**
```javascript
const portfolio = JSON.stringify({
  positions: [
    { symbol: "AAPL", quantity: 100, cost_basis: 150 },
    { symbol: "GOOGL", quantity: 50, cost_basis: 2800 },
    { symbol: "MSFT", quantity: 75, cost_basis: 380 }
  ]
});

const risk = await riskAnalysis(portfolio, true);

console.log(`VaR (95%): $${risk.var95}`);
console.log(`CVaR (95%): $${risk.cvar95}`);
console.log(`Sharpe Ratio: ${risk.sharpeRatio}`);
console.log(`Max Drawdown: ${risk.maxDrawdown}%`);
console.log(`Beta: ${risk.beta}`);
```

---

### 47. `optimizeStrategy(strategy: string, symbol: string, parameterRanges: string, useGpu?: boolean): Promise<StrategyOptimization>`

Optimize strategy parameters using GPU-accelerated search.

**Example:**
```javascript
const paramRanges = JSON.stringify({
  lookback_period: [10, 20, 30, 50],
  threshold: [0.01, 0.02, 0.03, 0.05],
  stop_loss: [0.02, 0.03, 0.05, 0.07]
});

const optimization = await optimizeStrategy(
  "momentum",
  "AAPL",
  paramRanges,
  true
);

console.log(`Best Parameters: ${optimization.bestParams}`);
console.log(`Best Sharpe: ${optimization.bestSharpe}`);
console.log(`Optimization Time: ${optimization.optimizationTimeMs}ms`);
```

---

### 48. `portfolioRebalance(targetAllocations: string, currentPortfolio?: string): Promise<RebalanceResult>`

Calculate portfolio rebalancing trades.

**Example:**
```javascript
const targetAlloc = JSON.stringify({
  "AAPL": 0.30,
  "GOOGL": 0.25,
  "MSFT": 0.25,
  "AMZN": 0.20
});

const currentPort = JSON.stringify({
  "AAPL": 0.35,
  "GOOGL": 0.20,
  "MSFT": 0.30,
  "AMZN": 0.15
});

const rebalance = await portfolioRebalance(targetAlloc, currentPort);

console.log(`Target Achieved: ${rebalance.targetAchieved}`);
console.log(`Estimated Cost: $${rebalance.estimatedCost}`);

rebalance.tradesNeeded.forEach(trade => {
  console.log(`${trade.action.toUpperCase()} ${trade.quantity} ${trade.symbol}`);
});
```

---

### 49. `correlationAnalysis(symbols: string[], useGpu?: boolean): Promise<CorrelationMatrix>`

Analyze asset correlations with GPU acceleration.

**Example:**
```javascript
const symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];

const correlation = await correlationAnalysis(symbols, true);

console.log(`Symbols: ${correlation.symbols.join(', ')}`);
console.log(`Analysis Period: ${correlation.analysisPeriod}`);

// Print correlation matrix
correlation.matrix.forEach((row, i) => {
  const symbol = correlation.symbols[i];
  const corrs = row.map((val, j) =>
    `${correlation.symbols[j]}: ${val.toFixed(2)}`
  ).join(', ');
  console.log(`${symbol}: ${corrs}`);
});
```

---

## Enums & Constants

### AllocationStrategy

```typescript
enum AllocationStrategy {
  KellyCriterion = 0,       // Kelly Criterion with fractional betting
  FixedPercentage = 1,      // Fixed percentage allocation
  DynamicConfidence = 2,    // Dynamic confidence-based
  RiskParity = 3,           // Risk parity allocation
  Martingale = 4,           // Martingale strategy
  AntiMartingale = 5        // Anti-martingale strategy
}
```

### DistributionModel

```typescript
enum DistributionModel {
  Proportional = 0,         // Pure proportional
  PerformanceWeighted = 1,  // Performance-weighted
  Tiered = 2,               // Tiered distribution
  Hybrid = 3                // Hybrid (50% capital, 30% performance, 20% equal)
}
```

### MemberRole

```typescript
enum MemberRole {
  LeadInvestor = 0,         // Full control
  SeniorAnalyst = 1,        // Advanced permissions
  JuniorAnalyst = 2,        // Limited permissions
  ContributingMember = 3,   // Basic permissions
  Observer = 4              // Read-only
}
```

### SwarmTopology

```typescript
enum SwarmTopology {
  Mesh = 0,                 // Fully connected peer-to-peer
  Hierarchical = 1,         // Tree-structured leader-follower
  Ring = 2,                 // Circular formation
  Star = 3                  // Centralized hub with spokes
}
```

### AgentType

```typescript
enum AgentType {
  Momentum = 0,             // Momentum trading
  MeanReversion = 1,        // Mean reversion
  Pairs = 2,                // Pairs trading
  Neural = 3,               // Neural network-based
  Arbitrage = 4             // Arbitrage detection
}
```

---

## Error Handling

All async functions can throw errors. Always use try-catch blocks:

```javascript
try {
  const result = await executeTrade("momentum", "AAPL", "buy", 100);
  console.log('Trade executed:', result);
} catch (error) {
  console.error('Trade failed:', error.message);

  // Handle specific error types
  if (error.message.includes('Insufficient funds')) {
    console.log('Add more capital to your account');
  } else if (error.message.includes('Rate limit')) {
    console.log('Slow down your requests');
  } else if (error.message.includes('Invalid API key')) {
    console.log('Check your authentication credentials');
  }
}
```

### Common Error Types

- **Authentication Errors**: Invalid API key or expired token
- **Authorization Errors**: Insufficient permissions for operation
- **Rate Limit Errors**: Too many requests
- **Validation Errors**: Invalid parameters or input data
- **Execution Errors**: Trade execution failures
- **System Errors**: Internal server errors

---

## Best Practices

1. **Always use GPU acceleration** for intensive operations (backtesting, neural training, risk analysis)
2. **Implement rate limiting** in production environments
3. **Validate all inputs** before passing to functions
4. **Log audit events** for security compliance
5. **Use try-catch blocks** for error handling
6. **Sanitize user inputs** to prevent injection attacks
7. **Rotate API keys** regularly
8. **Monitor swarm health** in distributed deployments
9. **Test strategies** thoroughly before live trading
10. **Keep bankroll rules** conservative for syndicates

---

## Performance Optimization

- **GPU Acceleration**: Enable for operations processing large datasets
- **Batch Operations**: Group multiple operations to reduce overhead
- **Caching**: Results are cached where appropriate
- **Connection Pooling**: Reuse connections for better performance
- **Async/Await**: All operations are non-blocking
- **Parallel Execution**: Swarms run agents in parallel

---

## Support & Resources

- **Documentation**: [GitHub Repository](https://github.com/ruvnet/neural-trader)
- **Issues**: Report bugs and request features
- **Examples**: See `/docs/examples/` directory
- **Community**: Join discussions and get help

---

**Last Updated:** 2025-11-15
**API Version:** 2.1.1
**License:** MIT
