# Syndicate Quick Start Guide

Get started with @neural-trader/syndicate in 5 minutes!

## Installation

```bash
npm install @neural-trader/syndicate
```

## Step 1: Create a Syndicate (30 seconds)

```typescript
import { SyndicateManager } from '@neural-trader/syndicate';

const manager = new SyndicateManager();

const syndicate = await manager.createSyndicate({
  id: 'my-first-syndicate',
  name: 'Sports Betting Syndicate',
  initialCapital: 100000,
  config: {
    maxSingleBet: 0.05,      // 5% max per bet
    maxDailyExposure: 0.20,  // 20% max daily
    minReserve: 0.10,        // 10% reserve
    votingQuorum: 0.60       // 60% quorum
  }
});

console.log(`Created syndicate: ${syndicate.id}`);
```

## Step 2: Add Members (1 minute)

```typescript
import { MemberRole } from '@neural-trader/syndicate';

// Add lead investor
await syndicate.addMember({
  name: 'Alice',
  email: 'alice@example.com',
  role: MemberRole.LeadInvestor,
  initialContribution: 40000
});

// Add analyst
await syndicate.addMember({
  name: 'Bob',
  email: 'bob@example.com',
  role: MemberRole.Analyst,
  initialContribution: 30000
});

// Add member
await syndicate.addMember({
  name: 'Carol',
  email: 'carol@example.com',
  role: MemberRole.Member,
  initialContribution: 30000
});

console.log(`Members: ${syndicate.members.length}`);
```

## Step 3: Make Your First Allocation (2 minutes)

```typescript
import { AllocationStrategy } from '@neural-trader/syndicate';

// Define opportunity
const opportunity = {
  sport: 'NFL',
  event: 'Chiefs vs Bills',
  market: 'Moneyline',
  selection: 'Chiefs',
  odds: 2.10,
  probability: 0.55,
  confidence: 0.85,
  bookmaker: 'DraftKings'
};

// Allocate using Kelly Criterion
const allocation = await syndicate.allocateFunds({
  opportunity,
  strategy: AllocationStrategy.KellyCriterion,
  fractionalKelly: 0.25  // Quarter Kelly
});

console.log(`Bet size: $${allocation.amount}`);
console.log(`Expected value: $${allocation.expectedValue}`);
```

## Step 4: Distribute Profits (1 minute)

```typescript
import { DistributionModel } from '@neural-trader/syndicate';

// After winning, distribute profits
const profit = 10000;

const distributions = await syndicate.distributeProfits({
  profit,
  model: DistributionModel.Proportional
});

distributions.forEach(dist => {
  console.log(`${dist.memberName}: $${dist.amount}`);
});
```

## Step 5: Check Status (30 seconds)

```typescript
const status = await syndicate.getStatus();

console.log(`Total Capital: $${status.totalCapital}`);
console.log(`ROI: ${status.roi}%`);
console.log(`Win Rate: ${status.winRate}%`);
console.log(`Sharpe Ratio: ${status.sharpeRatio}`);
```

## Complete Example

```typescript
import {
  SyndicateManager,
  MemberRole,
  AllocationStrategy,
  DistributionModel
} from '@neural-trader/syndicate';

async function quickStart() {
  // 1. Create syndicate
  const manager = new SyndicateManager();
  const syndicate = await manager.createSyndicate({
    id: 'my-syndicate',
    name: 'NFL Betting Syndicate',
    initialCapital: 100000,
    config: {
      maxSingleBet: 0.05,
      maxDailyExposure: 0.20,
      minReserve: 0.10,
      votingQuorum: 0.60
    }
  });

  // 2. Add members
  await syndicate.addMember({
    name: 'Alice',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 40000
  });

  await syndicate.addMember({
    name: 'Bob',
    email: 'bob@example.com',
    role: MemberRole.Analyst,
    initialContribution: 30000
  });

  await syndicate.addMember({
    name: 'Carol',
    email: 'carol@example.com',
    role: MemberRole.Member,
    initialContribution: 30000
  });

  // 3. Allocate funds
  const allocation = await syndicate.allocateFunds({
    opportunity: {
      sport: 'NFL',
      event: 'Chiefs vs Bills',
      odds: 2.10,
      probability: 0.55,
      confidence: 0.85,
      bookmaker: 'DraftKings'
    },
    strategy: AllocationStrategy.KellyCriterion,
    fractionalKelly: 0.25
  });

  console.log(`Allocated: $${allocation.amount}`);

  // 4. Distribute profits
  const distributions = await syndicate.distributeProfits({
    profit: 10000,
    model: DistributionModel.Proportional
  });

  // 5. Check status
  const status = await syndicate.getStatus();
  console.log(`ROI: ${status.roi}%`);
}

quickStart().catch(console.error);
```

## Next Steps

- Read the [complete README](./README.md)
- Learn about [Kelly Criterion](./KELLY_CRITERION_GUIDE.md)
- Explore [Governance](./GOVERNANCE_GUIDE.md)
- Check out [Examples](./examples/)

## Common Use Cases

### Conservative Betting
```typescript
// Use Quarter Kelly with low risk
const allocation = await syndicate.allocateFunds({
  opportunity,
  strategy: AllocationStrategy.KellyCriterion,
  fractionalKelly: 0.125  // Eighth Kelly
});
```

### Fixed Percentage
```typescript
// Simple 2% flat betting
const allocation = await syndicate.allocateFunds({
  opportunity,
  strategy: AllocationStrategy.FixedPercentage,
  percentage: 0.02
});
```

### Performance-Based Distribution
```typescript
// Reward top performers
const distributions = await syndicate.distributeProfits({
  profit: 10000,
  model: DistributionModel.PerformanceBased,
  performanceWeights: new Map([
    ['alice-001', 0.50],  // Top performer gets 50%
    ['bob-001', 0.30],
    ['carol-001', 0.20]
  ])
});
```

## Help & Support

- **Documentation**: https://docs.neural-trader.com/syndicate
- **Issues**: https://github.com/neural-trader/neural-trader/issues
- **Discord**: https://discord.gg/neural-trader

---

Happy betting! 🎲
