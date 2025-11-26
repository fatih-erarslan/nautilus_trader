# Syndicate Management & Sports Betting Examples

Complete examples for creating and managing investment syndicates with sports betting focus.

## Table of Contents

1. [Creating a Basic Syndicate](#creating-a-basic-syndicate)
2. [Sports Betting Arbitrage](#sports-betting-arbitrage)
3. [Kelly Criterion Bankroll Management](#kelly-criterion-bankroll-management)
4. [Automated Fund Allocation](#automated-fund-allocation)
5. [Profit Distribution](#profit-distribution)
6. [Democratic Voting System](#democratic-voting-system)
7. [Complete Syndicate Trading System](#complete-syndicate-trading-system)

---

## 1. Creating a Basic Syndicate

Set up a complete syndicate with members and roles.

```javascript
const {
  createSyndicate,
  addSyndicateMember,
  getSyndicateStatus,
  MemberRole
} = require('@rUv/neural-trader-backend');

async function createBasicSyndicate() {
  console.log('=== Creating Investment Syndicate ===\n');

  // Step 1: Create the syndicate
  console.log('1. Creating syndicate...');
  const syndicate = await createSyndicate(
    'syn-elite-001',
    'Elite Sports Betting Syndicate',
    'Professional sports betting group focusing on NFL, NBA, and arbitrage opportunities'
  );

  console.log(`   ✓ Syndicate created: ${syndicate.name}`);
  console.log(`   ID: ${syndicate.syndicateId}`);
  console.log(`   Created: ${syndicate.createdAt}`);

  // Step 2: Add lead investor
  console.log('\n2. Adding members...\n');
  const lead = await addSyndicateMember(
    syndicate.syndicateId,
    'John Smith',
    'john@example.com',
    'lead_investor',
    100000  // $100,000 initial contribution
  );

  console.log(`   Lead Investor: ${lead.name}`);
  console.log(`     Member ID: ${lead.memberId}`);
  console.log(`     Contribution: $${lead.contribution.toLocaleString()}`);
  console.log(`     Profit Share: ${lead.profitShare}%`);

  // Step 3: Add senior analysts
  const analyst1 = await addSyndicateMember(
    syndicate.syndicateId,
    'Sarah Johnson',
    'sarah@example.com',
    'senior_analyst',
    50000
  );

  console.log(`\n   Senior Analyst: ${analyst1.name}`);
  console.log(`     Contribution: $${analyst1.contribution.toLocaleString()}`);
  console.log(`     Profit Share: ${analyst1.profitShare}%`);

  const analyst2 = await addSyndicateMember(
    syndicate.syndicateId,
    'Mike Chen',
    'mike@example.com',
    'senior_analyst',
    50000
  );

  console.log(`\n   Senior Analyst: ${analyst2.name}`);
  console.log(`     Contribution: $${analyst2.contribution.toLocaleString()}`);
  console.log(`     Profit Share: ${analyst2.profitShare}%`);

  // Step 4: Add contributing members
  const members = [
    { name: 'Alice Brown', email: 'alice@example.com', contribution: 25000 },
    { name: 'Bob Wilson', email: 'bob@example.com', contribution: 30000 },
    { name: 'Carol Davis', email: 'carol@example.com', contribution: 20000 }
  ];

  console.log(`\n   Contributing Members:`);
  for (const memberData of members) {
    const member = await addSyndicateMember(
      syndicate.syndicateId,
      memberData.name,
      memberData.email,
      'contributing_member',
      memberData.contribution
    );

    console.log(`     - ${member.name}: $${member.contribution.toLocaleString()} (${member.profitShare}%)`);
  }

  // Step 5: Check syndicate status
  console.log('\n3. Syndicate Status:');
  const status = await getSyndicateStatus(syndicate.syndicateId);

  console.log(`   Total Capital: $${status.totalCapital.toLocaleString()}`);
  console.log(`   Members: ${status.memberCount}`);
  console.log(`   Active Bets: ${status.activeBets}`);
  console.log(`   ROI: ${status.roi}%`);

  return syndicate;
}

createBasicSyndicate();
```

**Output:**
```
=== Creating Investment Syndicate ===

1. Creating syndicate...
   ✓ Syndicate created: Elite Sports Betting Syndicate
   ID: syn-elite-001
   Created: 2025-01-15T10:30:00Z

2. Adding members...

   Lead Investor: John Smith
     Member ID: mem-001
     Contribution: $100,000
     Profit Share: 36.36%

   Senior Analyst: Sarah Johnson
     Contribution: $50,000
     Profit Share: 18.18%

   Senior Analyst: Mike Chen
     Contribution: $50,000
     Profit Share: 18.18%

   Contributing Members:
     - Alice Brown: $25,000 (9.09%)
     - Bob Wilson: $30,000 (10.91%)
     - Carol Davis: $20,000 (7.27%)

3. Syndicate Status:
   Total Capital: $275,000
   Members: 6
   Active Bets: 0
   ROI: 0.00%
```

---

## 2. Sports Betting Arbitrage

Find and execute arbitrage opportunities.

```javascript
const {
  getSportsEvents,
  getSportsOdds,
  findSportsArbitrage,
  executeSportsBet
} = require('@rUv/neural-trader-backend');

async function findArbitrageOpportunities() {
  console.log('=== Sports Betting Arbitrage Scanner ===\n');

  const sports = ['nfl', 'nba', 'mlb'];

  for (const sport of sports) {
    console.log(`\n--- ${sport.toUpperCase()} ---\n`);

    // Step 1: Get upcoming events
    const events = await getSportsEvents(sport, 3);
    console.log(`Found ${events.length} upcoming events`);

    // Step 2: Get odds from multiple bookmakers
    const odds = await getSportsOdds(sport);
    console.log(`Collected odds from ${new Set(odds.map(o => o.bookmaker)).size} bookmakers`);

    // Step 3: Find arbitrage opportunities
    const arbitrages = await findSportsArbitrage(sport, 0.01);  // Min 1% profit

    if (arbitrages.length > 0) {
      console.log(`\n✓ Found ${arbitrages.length} arbitrage opportunities:\n`);

      for (const arb of arbitrages) {
        const event = events.find(e => e.eventId === arb.eventId);
        if (event) {
          console.log(`${event.homeTeam} vs ${event.awayTeam}`);
          console.log(`Start: ${event.startTime}`);
        }

        console.log(`Profit Margin: ${arb.profitMargin.toFixed(2)}%`);
        console.log(`\nBet Distribution:`);
        console.log(`  Home: ${arb.betHome.bookmaker}`);
        console.log(`    Odds: ${arb.betHome.odds}`);
        console.log(`    Stake: $${arb.betHome.stake.toFixed(2)}`);
        console.log(`  Away: ${arb.betAway.bookmaker}`);
        console.log(`    Odds: ${arb.betAway.odds}`);
        console.log(`    Stake: $${arb.betAway.stake.toFixed(2)}`);

        const totalStake = arb.betHome.stake + arb.betAway.stake;
        const guaranteedProfit = totalStake * (arb.profitMargin / 100);

        console.log(`\nTotal Stake: $${totalStake.toFixed(2)}`);
        console.log(`Guaranteed Profit: $${guaranteedProfit.toFixed(2)}`);
        console.log(`---`);
      }
    } else {
      console.log('✗ No arbitrage opportunities found');
    }
  }
}

findArbitrageOpportunities();
```

**Execute Arbitrage:**
```javascript
async function executeArbitrage(arbitrage, validateOnly = true) {
  console.log('\n=== Executing Arbitrage ===\n');

  // Validate both sides first
  console.log('1. Validating bets...');

  const homeBet = await executeSportsBet(
    arbitrage.eventId,
    'home',
    arbitrage.betHome.stake,
    arbitrage.betHome.odds,
    true  // Validate only
  );

  const awayBet = await executeSportsBet(
    arbitrage.eventId,
    'away',
    arbitrage.betAway.stake,
    arbitrage.betAway.odds,
    true
  );

  if (homeBet.status === 'valid' && awayBet.status === 'valid') {
    console.log('   ✓ Both bets validated successfully');

    if (!validateOnly) {
      // Execute for real
      console.log('\n2. Executing bets...');

      const homeExecution = await executeSportsBet(
        arbitrage.eventId,
        'home',
        arbitrage.betHome.stake,
        arbitrage.betHome.odds,
        false
      );

      const awayExecution = await executeSportsBet(
        arbitrage.eventId,
        'away',
        arbitrage.betAway.stake,
        arbitrage.betAway.odds,
        false
      );

      console.log(`\n   ✓ Home bet: ${homeExecution.betId}`);
      console.log(`     Potential return: $${homeExecution.potentialReturn.toFixed(2)}`);

      console.log(`\n   ✓ Away bet: ${awayExecution.betId}`);
      console.log(`     Potential return: $${awayExecution.potentialReturn.toFixed(2)}`);

      const guaranteedProfit = Math.min(homeExecution.potentialReturn, awayExecution.potentialReturn) -
                               (arbitrage.betHome.stake + arbitrage.betAway.stake);

      console.log(`\n   Guaranteed Profit: $${guaranteedProfit.toFixed(2)}`);
    }
  } else {
    console.log('   ✗ Validation failed - odds may have changed');
  }
}
```

---

## 3. Kelly Criterion Bankroll Management

Optimal bet sizing using Kelly Criterion.

```javascript
const {
  calculateKellyCriterion,
  getSportsOdds
} = require('@rUv/neural-trader-backend');

async function kellyBankrollManagement() {
  console.log('=== Kelly Criterion Bankroll Management ===\n');

  const bankroll = 100000;  // $100,000 syndicate bankroll

  // Example opportunities
  const opportunities = [
    {
      event: 'Patriots vs Chiefs',
      selection: 'Patriots',
      odds: 2.5,
      estimatedProbability: 0.45,  // 45% win probability
      confidence: 0.85
    },
    {
      event: 'Lakers vs Warriors',
      selection: 'Lakers',
      odds: 1.91,
      estimatedProbability: 0.55,
      confidence: 0.90
    },
    {
      event: 'Yankees vs Red Sox',
      selection: 'Yankees',
      odds: 1.75,
      estimatedProbability: 0.60,
      confidence: 0.75
    }
  ];

  console.log(`Total Bankroll: $${bankroll.toLocaleString()}\n`);

  for (const opp of opportunities) {
    console.log(`\n${opp.event} - ${opp.selection}`);
    console.log(`Odds: ${opp.odds}`);
    console.log(`Win Probability: ${(opp.estimatedProbability * 100).toFixed(0)}%`);
    console.log(`Confidence: ${(opp.confidence * 100).toFixed(0)}%`);

    // Calculate Kelly Criterion
    const kelly = await calculateKellyCriterion(
      opp.estimatedProbability,
      opp.odds,
      bankroll
    );

    console.log(`\nKelly Criterion:`);
    console.log(`  Kelly Fraction: ${(kelly.kellyFraction * 100).toFixed(2)}%`);
    console.log(`  Full Kelly Stake: $${kelly.suggestedStake.toLocaleString()}`);

    // Fractional Kelly (more conservative)
    const fractionalKelly = kelly.suggestedStake * 0.5;  // Half Kelly
    const quarterKelly = kelly.suggestedStake * 0.25;    // Quarter Kelly

    console.log(`  Half Kelly: $${fractionalKelly.toLocaleString()}`);
    console.log(`  Quarter Kelly: $${quarterKelly.toLocaleString()}`);

    // Confidence-adjusted stake
    const adjustedStake = kelly.suggestedStake * opp.confidence;
    console.log(`  Confidence-Adjusted: $${adjustedStake.toLocaleString()}`);

    // Calculate expected value
    const ev = (opp.estimatedProbability * (opp.odds - 1) - (1 - opp.estimatedProbability)) * adjustedStake;
    console.log(`\nExpected Value: $${ev.toFixed(2)}`);

    // Recommendation
    if (ev > 0 && kelly.kellyFraction > 0) {
      console.log(`✓ RECOMMENDED BET: $${adjustedStake.toLocaleString()}`);
    } else {
      console.log(`✗ NEGATIVE EXPECTED VALUE - SKIP`);
    }
  }

  // Portfolio-level Kelly
  console.log('\n\n=== Portfolio-Level Kelly Allocation ===\n');

  let totalKelly = 0;
  const allocations = [];

  for (const opp of opportunities) {
    const kelly = await calculateKellyCriterion(
      opp.estimatedProbability,
      opp.odds,
      bankroll
    );

    if (kelly.kellyFraction > 0) {
      totalKelly += kelly.kellyFraction;
      allocations.push({
        event: opp.event,
        fraction: kelly.kellyFraction,
        stake: kelly.suggestedStake
      });
    }
  }

  console.log(`Total Kelly Fraction: ${(totalKelly * 100).toFixed(2)}%`);

  if (totalKelly > 1.0) {
    console.log('⚠ Over-leveraged - normalizing allocations...\n');

    // Normalize to 100% of bankroll
    allocations.forEach(alloc => {
      alloc.normalizedFraction = alloc.fraction / totalKelly;
      alloc.normalizedStake = bankroll * alloc.normalizedFraction;

      console.log(`${alloc.event}:`);
      console.log(`  Original: ${(alloc.fraction * 100).toFixed(2)}% ($${alloc.stake.toLocaleString()})`);
      console.log(`  Normalized: ${(alloc.normalizedFraction * 100).toFixed(2)}% ($${alloc.normalizedStake.toLocaleString()})`);
    });
  }
}

kellyBankrollManagement();
```

---

## 4. Automated Fund Allocation

Advanced fund allocation engine with risk management.

```javascript
const {
  FundAllocationEngine,
  AllocationStrategy
} = require('@rUv/neural-trader-backend');

async function automatedFundAllocation() {
  console.log('=== Automated Fund Allocation ===\n');

  const syndicateId = 'syn-elite-001';
  const totalBankroll = '275000';  // $275,000

  // Create allocation engine
  const engine = new FundAllocationEngine(syndicateId, totalBankroll);

  console.log(`Syndicate: ${syndicateId}`);
  console.log(`Total Bankroll: $${totalBankroll}\n`);

  // Define betting opportunities
  const opportunities = [
    {
      sport: 'nfl',
      event: 'Patriots vs Chiefs - AFC Championship',
      betType: 'moneyline',
      selection: 'Patriots',
      odds: 2.5,
      probability: 0.48,
      edge: 0.20,
      confidence: 0.85,
      modelAgreement: 0.90,
      timeUntilEventSecs: 86400 * 2,  // 2 days
      liquidity: 0.95,
      isLive: false,
      isParlay: false
    },
    {
      sport: 'nba',
      event: 'Lakers vs Warriors',
      betType: 'spread',
      selection: 'Lakers -3.5',
      odds: 1.91,
      probability: 0.55,
      edge: 0.05,
      confidence: 0.75,
      modelAgreement: 0.80,
      timeUntilEventSecs: 3600 * 6,  // 6 hours
      liquidity: 0.90,
      isLive: false,
      isParlay: false
    },
    {
      sport: 'mlb',
      event: 'Yankees vs Red Sox',
      betType: 'total',
      selection: 'Over 8.5',
      odds: 2.0,
      probability: 0.52,
      edge: 0.04,
      confidence: 0.70,
      modelAgreement: 0.75,
      timeUntilEventSecs: 86400,  // 1 day
      liquidity: 0.85,
      isLive: false,
      isParlay: false
    }
  ];

  console.log('=== Allocation Results ===\n');

  const strategies = [
    AllocationStrategy.KellyCriterion,
    AllocationStrategy.FixedPercentage,
    AllocationStrategy.DynamicConfidence,
    AllocationStrategy.RiskParity
  ];

  const strategyNames = {
    [AllocationStrategy.KellyCriterion]: 'Kelly Criterion',
    [AllocationStrategy.FixedPercentage]: 'Fixed Percentage',
    [AllocationStrategy.DynamicConfidence]: 'Dynamic Confidence',
    [AllocationStrategy.RiskParity]: 'Risk Parity'
  };

  for (const strategy of strategies) {
    console.log(`\n--- ${strategyNames[strategy]} ---\n`);

    let totalAllocated = 0;

    for (const opp of opportunities) {
      const allocation = engine.allocateFunds(opp, strategy);

      console.log(`${opp.event}`);
      console.log(`  Selection: ${opp.selection}`);
      console.log(`  Odds: ${opp.odds}`);
      console.log(`  Edge: ${(opp.edge * 100).toFixed(2)}%`);

      const amount = parseFloat(allocation.amount);
      console.log(`\n  Allocated: $${amount.toLocaleString()}`);
      console.log(`  % of Bankroll: ${allocation.percentageOfBankroll.toFixed(2)}%`);
      console.log(`  Reasoning: ${allocation.reasoning}`);

      if (allocation.approvalRequired) {
        console.log(`  ⚠ APPROVAL REQUIRED`);
      }

      if (allocation.warnings.length > 0) {
        console.log(`  Warnings:`);
        allocation.warnings.forEach(w => console.log(`    - ${w}`));
      }

      const riskMetrics = JSON.parse(allocation.riskMetrics);
      console.log(`\n  Risk Metrics:`);
      console.log(`    Expected ROI: ${riskMetrics.expected_roi}%`);
      console.log(`    Risk Score: ${riskMetrics.risk_score}`);

      totalAllocated += amount;

      // Update exposure
      engine.updateExposure(JSON.stringify({
        sport: opp.sport,
        amount: allocation.amount,
        isLive: opp.isLive
      }));

      console.log('');
    }

    console.log(`Total Allocated: $${totalAllocated.toLocaleString()}`);
    console.log(`% of Bankroll: ${(totalAllocated / parseFloat(totalBankroll) * 100).toFixed(2)}%`);

    // Get exposure summary
    const exposure = JSON.parse(engine.getExposureSummary());
    console.log(`\nExposure Summary:`);
    console.log(`  Total Exposure: $${exposure.total_exposure.toLocaleString()}`);
    console.log(`  Daily Exposure: $${exposure.daily_exposure.toLocaleString()}`);
    console.log(`  By Sport:`, exposure.by_sport);
  }
}

automatedFundAllocation();
```

---

## 5. Profit Distribution

Distribute profits among syndicate members.

```javascript
const {
  ProfitDistributionSystem,
  DistributionModel,
  MemberManager
} = require('@rUv/neural-trader-backend');

async function distributeProfits() {
  console.log('=== Syndicate Profit Distribution ===\n');

  const syndicateId = 'syn-elite-001';
  const totalProfit = '25000';  // $25,000 profit this period

  // Create distribution system
  const distributor = new ProfitDistributionSystem(syndicateId);

  // Member data with performance metrics
  const members = [
    {
      member_id: 'mem-001',
      name: 'John Smith',
      capital_contribution: 100000,
      performance_score: 0.85,
      bets_won: 120,
      bets_lost: 80,
      total_profit: 15000,
      tier: 'Platinum'
    },
    {
      member_id: 'mem-002',
      name: 'Sarah Johnson',
      capital_contribution: 50000,
      performance_score: 0.92,
      bets_won: 95,
      bets_lost: 55,
      total_profit: 12000,
      tier: 'Gold'
    },
    {
      member_id: 'mem-003',
      name: 'Mike Chen',
      capital_contribution: 50000,
      performance_score: 0.78,
      bets_won: 85,
      bets_lost: 90,
      total_profit: 8000,
      tier: 'Gold'
    },
    {
      member_id: 'mem-004',
      name: 'Alice Brown',
      capital_contribution: 25000,
      performance_score: 0.70,
      bets_won: 60,
      bets_lost: 70,
      total_profit: 3500,
      tier: 'Silver'
    }
  ];

  console.log(`Total Profit to Distribute: $${parseFloat(totalProfit).toLocaleString()}\n`);

  // Test different distribution models
  const models = [
    DistributionModel.Proportional,
    DistributionModel.PerformanceWeighted,
    DistributionModel.Hybrid
  ];

  const modelNames = {
    [DistributionModel.Proportional]: 'Proportional (Capital-Based)',
    [DistributionModel.PerformanceWeighted]: 'Performance-Weighted',
    [DistributionModel.Hybrid]: 'Hybrid (50% Capital, 30% Performance, 20% Equal)'
  };

  for (const model of models) {
    console.log(`\n=== ${modelNames[model]} ===\n`);

    const distribution = distributor.calculateDistribution(
      totalProfit,
      JSON.stringify(members),
      model
    );

    const result = JSON.parse(distribution);

    console.log('Distribution Breakdown:\n');

    result.distributions.forEach(dist => {
      const member = members.find(m => m.member_id === dist.member_id);
      console.log(`${member.name}:`);
      console.log(`  Capital: $${member.capital_contribution.toLocaleString()}`);
      console.log(`  Performance Score: ${(member.performance_score * 100).toFixed(0)}%`);
      console.log(`  Win Rate: ${(member.bets_won / (member.bets_won + member.bets_lost) * 100).toFixed(2)}%`);
      console.log(`  Distribution: $${dist.amount.toFixed(2)} (${dist.percentage.toFixed(2)}%)`);
      console.log('');
    });

    console.log(`Total Distributed: $${result.total_distributed.toFixed(2)}`);
    console.log(`Distribution Model: ${result.model}`);
  }

  // Member performance comparison
  console.log('\n=== Member Performance Analysis ===\n');

  members.forEach(member => {
    const roi = (member.total_profit / member.capital_contribution) * 100;
    const winRate = (member.bets_won / (member.bets_won + member.bets_lost)) * 100;

    console.log(`${member.name} (${member.tier}):`);
    console.log(`  ROI: ${roi.toFixed(2)}%`);
    console.log(`  Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`  Performance Score: ${(member.performance_score * 100).toFixed(0)}%`);
    console.log(`  Total Bets: ${member.bets_won + member.bets_lost}`);
    console.log('');
  });
}

distributeProfits();
```

---

## 6. Democratic Voting System

Syndicate decision-making through voting.

```javascript
const {
  VotingSystem
} = require('@rUv/neural-trader-backend');

async function syndicateVoting() {
  console.log('=== Syndicate Voting System ===\n');

  const syndicateId = 'syn-elite-001';
  const voting = new VotingSystem(syndicateId);

  // Step 1: Create a vote for strategy change
  console.log('1. Creating Vote: Change Risk Profile\n');

  const voteJson = voting.createVote(
    'strategy_change',
    JSON.stringify({
      proposal: 'Increase maximum bet size from 5% to 10% of bankroll',
      rationale: 'Strong performance suggests we can handle higher risk',
      proposed_max_bet: 0.10,
      current_max_bet: 0.05,
      expected_roi_increase: '15%'
    }),
    'mem-001',  // Proposed by lead investor
    48  // 48-hour voting period
  );

  const vote = JSON.parse(voteJson);
  console.log(`Vote Created: ${vote.vote_id}`);
  console.log(`Type: ${vote.vote_type}`);
  console.log(`Voting Period: ${vote.voting_period_hours} hours`);
  console.log(`Ends: ${vote.end_time}\n`);

  // Step 2: Members cast votes with different weights
  console.log('2. Casting Votes:\n');

  const votes = [
    { memberId: 'mem-001', decision: 'approve', weight: 2.0, name: 'John Smith (Lead)' },
    { memberId: 'mem-002', decision: 'approve', weight: 1.5, name: 'Sarah Johnson (Sr Analyst)' },
    { memberId: 'mem-003', decision: 'reject', weight: 1.5, name: 'Mike Chen (Sr Analyst)' },
    { memberId: 'mem-004', decision: 'approve', weight: 1.0, name: 'Alice Brown' },
    { memberId: 'mem-005', decision: 'approve', weight: 1.0, name: 'Bob Wilson' },
    { memberId: 'mem-006', decision: 'reject', weight: 1.0, name: 'Carol Davis' }
  ];

  for (const v of votes) {
    const cast = voting.castVote(vote.vote_id, v.memberId, v.decision, v.weight);
    console.log(`✓ ${v.name}: ${v.decision.toUpperCase()} (weight: ${v.weight})`);
  }

  // Step 3: Get current results
  console.log('\n3. Current Vote Results:\n');

  const results = JSON.parse(voting.getVoteResults(vote.vote_id));

  console.log(`Total Votes Cast: ${results.total_votes}`);
  console.log(`Approve: ${results.approve_count} (weight: ${results.approve_weight})`);
  console.log(`Reject: ${results.reject_count} (weight: ${results.reject_weight})`);
  console.log(`Abstain: ${results.abstain_count} (weight: ${results.abstain_weight})`);
  console.log(`\nApproval Percentage: ${results.approval_percentage.toFixed(2)}%`);

  // Step 4: Finalize vote
  console.log('\n4. Finalizing Vote:\n');

  const finalResult = JSON.parse(voting.finalizeVote(vote.vote_id));

  console.log(`Status: ${finalResult.status}`);
  console.log(`Outcome: ${finalResult.outcome}`);
  console.log(`Final Approval: ${finalResult.final_approval_percentage.toFixed(2)}%`);

  if (finalResult.outcome === 'approved') {
    console.log(`\n✓ VOTE PASSED - Implementing strategy change`);
  } else {
    console.log(`\n✗ VOTE FAILED - Maintaining current strategy`);
  }

  // Step 5: List all active votes
  console.log('\n5. Active Votes:\n');

  const activeVotes = JSON.parse(voting.listActiveVotes());
  activeVotes.forEach(v => {
    console.log(`Vote ID: ${v.vote_id}`);
    console.log(`Type: ${v.vote_type}`);
    console.log(`Status: ${v.status}`);
    console.log(`Ends: ${v.end_time}`);
    console.log('');
  });
}

syndicateVoting();
```

---

## 7. Complete Syndicate Trading System

Full-featured syndicate management system.

```javascript
const {
  createSyndicate,
  addSyndicateMember,
  FundAllocationEngine,
  ProfitDistributionSystem,
  MemberManager,
  VotingSystem,
  CollaborationHub,
  AllocationStrategy,
  DistributionModel,
  MemberRole,
  getSportsEvents,
  findSportsArbitrage
} = require('@rUv/neural-trader-backend');

class CompleteSyndicateSystem {
  constructor(syndicateId, name) {
    this.syndicateId = syndicateId;
    this.name = name;
    this.totalCapital = 0;
  }

  async initialize() {
    console.log('=== Initializing Syndicate System ===\n');

    // Create syndicate
    const syndicate = await createSyndicate(
      this.syndicateId,
      this.name,
      'Professional sports betting syndicate'
    );

    // Initialize managers
    this.memberManager = new MemberManager(this.syndicateId);
    this.votingSystem = new VotingSystem(this.syndicateId);
    this.collaborationHub = new CollaborationHub(this.syndicateId);

    // Create communication channels
    const tradingChannel = this.collaborationHub.createChannel(
      'trade-alerts',
      'Real-time trading opportunities',
      'alerts'
    );

    const discussionChannel = this.collaborationHub.createChannel(
      'general-discussion',
      'Member discussions and strategy',
      'discussion'
    );

    console.log('✓ Syndicate initialized');
    console.log('✓ Communication channels created\n');

    return syndicate;
  }

  async addMembers() {
    console.log('=== Adding Syndicate Members ===\n');

    const memberData = [
      { name: 'John Smith', email: 'john@example.com', role: MemberRole.LeadInvestor, capital: 100000 },
      { name: 'Sarah Johnson', email: 'sarah@example.com', role: MemberRole.SeniorAnalyst, capital: 50000 },
      { name: 'Mike Chen', email: 'mike@example.com', role: MemberRole.SeniorAnalyst, capital: 50000 },
      { name: 'Alice Brown', email: 'alice@example.com', role: MemberRole.ContributingMember, capital: 25000 },
      { name: 'Bob Wilson', email: 'bob@example.com', role: MemberRole.ContributingMember, capital: 30000 }
    ];

    for (const data of memberData) {
      const memberJson = this.memberManager.addMember(
        data.name,
        data.email,
        data.role,
        data.capital.toString()
      );

      const member = JSON.parse(memberJson);
      this.totalCapital += data.capital;

      console.log(`✓ Added: ${data.name} - $${data.capital.toLocaleString()}`);
    }

    console.log(`\nTotal Capital: $${this.totalCapital.toLocaleString()}\n`);
  }

  async findOpportunities() {
    console.log('=== Scanning for Opportunities ===\n');

    const opportunities = [];

    // Scan NFL
    const nflArbs = await findSportsArbitrage('nfl', 0.01);
    opportunities.push(...nflArbs.map(a => ({ ...a, sport: 'nfl' })));

    // Scan NBA
    const nbaArbs = await findSportsArbitrage('nba', 0.01);
    opportunities.push(...nbaArbs.map(a => ({ ...a, sport: 'nba' })));

    console.log(`Found ${opportunities.length} opportunities\n`);

    return opportunities;
  }

  async allocateAndExecute(opportunities) {
    console.log('=== Allocating Funds ===\n');

    const engine = new FundAllocationEngine(
      this.syndicateId,
      this.totalCapital.toString()
    );

    for (const opp of opportunities.slice(0, 5)) {  // Top 5 opportunities
      // Convert arbitrage to betting opportunity format
      const bettingOpp = {
        sport: opp.sport,
        event: opp.eventId,
        betType: 'arbitrage',
        selection: 'both sides',
        odds: 2.0,  // Simplified
        probability: 0.95,  // High probability for arbitrage
        edge: opp.profitMargin / 100,
        confidence: 0.95,
        modelAgreement: 1.0,
        timeUntilEventSecs: 86400,
        liquidity: 0.95,
        isLive: false,
        isParlay: false
      };

      const allocation = engine.allocateFunds(
        bettingOpp,
        AllocationStrategy.KellyCriterion
      );

      console.log(`Opportunity: ${opp.eventId}`);
      console.log(`  Profit Margin: ${opp.profitMargin.toFixed(2)}%`);
      console.log(`  Allocated: $${allocation.amount}`);
      console.log(`  Approval Required: ${allocation.approvalRequired}`);

      // Post to collaboration channel
      this.collaborationHub.postMessage(
        'trade-channel-id',
        'system',
        `New arbitrage opportunity: ${opp.eventId}. Profit: ${opp.profitMargin.toFixed(2)}%. Allocated: $${allocation.amount}`,
        'alert',
        []
      );

      console.log('');
    }
  }

  async distributeProfits(totalProfit) {
    console.log('=== Distributing Profits ===\n');

    const distributor = new ProfitDistributionSystem(this.syndicateId);

    // Get all members
    const membersJson = this.memberManager.listMembers(true);
    const members = JSON.parse(membersJson);

    // Calculate distribution
    const distribution = distributor.calculateDistribution(
      totalProfit.toString(),
      membersJson,
      DistributionModel.Hybrid
    );

    const result = JSON.parse(distribution);

    console.log(`Total Profit: $${totalProfit.toLocaleString()}\n`);

    result.distributions.forEach(dist => {
      const member = members.find(m => m.member_id === dist.member_id);
      console.log(`${member.name}: $${dist.amount.toFixed(2)} (${dist.percentage.toFixed(2)}%)`);
    });

    console.log('');
  }

  async run() {
    await this.initialize();
    await this.addMembers();

    const opportunities = await this.findOpportunities();
    await this.allocateAndExecute(opportunities);

    // Simulate profit
    await this.distributeProfits(25000);

    console.log('=== Syndicate System Complete ===');
  }
}

// Run the system
const syndicate = new CompleteSyndicateSystem(
  'syn-elite-001',
  'Elite Sports Betting Syndicate'
);

syndicate.run();
```

---

## Best Practices

1. **Start with clear rules** for capital contribution and profit sharing
2. **Use Kelly Criterion** for optimal bet sizing
3. **Implement voting** for major decisions
4. **Track member performance** individually
5. **Maintain transparent** allocation and distribution
6. **Set risk limits** per sport and bet type
7. **Use arbitrage** when available for guaranteed profits
8. **Regular reporting** to all members
9. **Emergency procedures** for large losses
10. **Legal compliance** with gambling regulations

---

**Next Steps:**
- Review [E2B Swarm Examples](./swarm-examples.md)
- Check [Best Practices Guide](../guides/best-practices.md)
- Explore [Security Guidelines](../guides/security.md)
