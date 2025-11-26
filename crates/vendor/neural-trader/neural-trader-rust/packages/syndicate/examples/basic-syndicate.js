/**
 * Basic Syndicate Example
 *
 * Demonstrates:
 * - Creating a syndicate
 * - Adding members
 * - Making allocations with Kelly Criterion
 * - Distributing profits
 * - Checking status
 */

const {
  SyndicateManager,
  MemberRole,
  AllocationStrategy,
  DistributionModel
} = require('@neural-trader/syndicate');

async function basicSyndicateExample() {
  console.log('=== Basic Syndicate Example ===\n');

  // Step 1: Create syndicate manager
  console.log('Step 1: Creating Syndicate Manager...');
  const manager = new SyndicateManager();

  // Step 2: Create syndicate
  console.log('Step 2: Creating Syndicate...');
  const syndicate = await manager.createSyndicate({
    id: 'sports-syndicate-001',
    name: 'Elite Sports Betting Syndicate',
    initialCapital: 100000,
    config: {
      maxSingleBet: 0.05,        // 5% max bet size
      maxDailyExposure: 0.20,    // 20% max daily exposure
      minReserve: 0.10,          // 10% reserve requirement
      votingQuorum: 0.60,        // 60% quorum for votes
      votingPeriod: 48,          // 48 hour voting period
      withdrawalDelay: 72,       // 72 hour withdrawal delay
      performanceFee: 0.20,      // 20% performance fee
      managementFee: 0.02        // 2% management fee
    }
  });

  console.log(`✓ Syndicate created: ${syndicate.id}`);
  console.log(`  Total capital: $${syndicate.totalCapital.toLocaleString()}\n`);

  // Step 3: Add members
  console.log('Step 3: Adding Members...');

  const alice = await syndicate.addMember({
    name: 'Alice Johnson',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 40000
  });
  console.log(`✓ Added ${alice.name} as Lead Investor ($40,000)`);

  const bob = await syndicate.addMember({
    name: 'Bob Smith',
    email: 'bob@example.com',
    role: MemberRole.SeniorAnalyst,
    initialContribution: 30000
  });
  console.log(`✓ Added ${bob.name} as Senior Analyst ($30,000)`);

  const carol = await syndicate.addMember({
    name: 'Carol Davis',
    email: 'carol@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 20000
  });
  console.log(`✓ Added ${carol.name} as Contributing Member ($20,000)`);

  const david = await syndicate.addMember({
    name: 'David Lee',
    email: 'david@example.com',
    role: MemberRole.JuniorAnalyst,
    initialContribution: 10000
  });
  console.log(`✓ Added ${david.name} as Junior Analyst ($10,000)\n`);

  // Step 4: Allocate funds using Kelly Criterion
  console.log('Step 4: Making First Allocation...');

  const opportunity1 = {
    sport: 'NFL',
    event: 'Kansas City Chiefs vs Buffalo Bills',
    market: 'Moneyline',
    selection: 'Chiefs',
    odds: 2.10,
    probability: 0.55,
    confidence: 0.85,
    bookmaker: 'DraftKings',
    metadata: {
      analysis: 'Chiefs strong at home, Bills key injuries',
      edgeCalculation: 'Market 47.6%, our model 55%'
    }
  };

  const allocation1 = await syndicate.allocateFunds({
    opportunity: opportunity1,
    strategy: AllocationStrategy.KellyCriterion,
    fractionalKelly: 0.25  // Quarter Kelly for safety
  });

  console.log(`✓ Allocation created:`);
  console.log(`  Sport: ${opportunity1.sport}`);
  console.log(`  Event: ${opportunity1.event}`);
  console.log(`  Odds: ${opportunity1.odds}`);
  console.log(`  Bet size: $${allocation1.amount.toLocaleString()}`);
  console.log(`  Kelly %: ${(allocation1.kellyPercentage * 100).toFixed(2)}%`);
  console.log(`  Fractional %: ${(allocation1.percentage * 100).toFixed(2)}%`);
  console.log(`  Expected value: $${allocation1.expectedValue.toLocaleString()}`);
  console.log(`  Risk level: ${allocation1.riskLevel}\n`);

  // Step 5: Make another allocation
  console.log('Step 5: Making Second Allocation...');

  const opportunity2 = {
    sport: 'NBA',
    event: 'Lakers vs Celtics',
    market: 'Over/Under',
    selection: 'Over 225.5',
    odds: 1.91,
    probability: 0.53,
    confidence: 0.75,
    bookmaker: 'FanDuel',
    metadata: {
      analysis: 'Both teams playing at fast pace',
      trend: 'Over hitting 65% last 10 games'
    }
  };

  const allocation2 = await syndicate.allocateFunds({
    opportunity: opportunity2,
    strategy: AllocationStrategy.KellyCriterion,
    fractionalKelly: 0.25
  });

  console.log(`✓ Allocation created:`);
  console.log(`  Sport: ${opportunity2.sport}`);
  console.log(`  Event: ${opportunity2.event}`);
  console.log(`  Bet size: $${allocation2.amount.toLocaleString()}`);
  console.log(`  Expected value: $${allocation2.expectedValue.toLocaleString()}\n`);

  // Step 6: Simulate winning and distribute profits
  console.log('Step 6: Distributing Profits...');

  const profit = 12500;  // Won $12,500

  const distributions = await syndicate.distributeProfits({
    profit,
    model: DistributionModel.Proportional
  });

  console.log(`✓ Distributed $${profit.toLocaleString()} profit:`);
  distributions.forEach(dist => {
    console.log(`  ${dist.memberName}: $${dist.amount.toLocaleString()} (${(dist.percentage * 100).toFixed(1)}%)`);
  });
  console.log();

  // Step 7: Check syndicate status
  console.log('Step 7: Checking Syndicate Status...');

  const status = await syndicate.getStatus();

  console.log(`\n=== Syndicate Status ===`);
  console.log(`ID: ${status.id}`);
  console.log(`Name: ${status.name}`);
  console.log(`Status: ${status.status}`);
  console.log(`Members: ${status.memberCount}`);
  console.log(`\nCapital:`);
  console.log(`  Total: $${status.totalCapital.toLocaleString()}`);
  console.log(`  Available: $${status.availableCapital.toLocaleString()}`);
  console.log(`  Invested: $${status.totalInvested.toLocaleString()}`);
  console.log(`\nPerformance:`);
  console.log(`  Total P&L: $${status.totalProfitLoss.toLocaleString()}`);
  console.log(`  ROI: ${status.roi.toFixed(2)}%`);
  console.log(`  Win Rate: ${status.winRate.toFixed(1)}%`);
  console.log(`  Sharpe Ratio: ${status.sharpeRatio.toFixed(2)}`);
  console.log(`  Max Drawdown: ${status.maxDrawdown.toFixed(2)}%`);
  console.log(`\nActivity:`);
  console.log(`  Active Bets: ${status.activeBets}`);
  console.log(`  Total Bets: ${status.totalBets}`);
  console.log(`  Created: ${status.created.toLocaleDateString()}`);
  console.log(`  Last Activity: ${status.lastActivity.toLocaleDateString()}`);

  // Step 8: Get member performance
  console.log(`\n=== Member Performance ===`);

  for (const member of [alice, bob, carol, david]) {
    const performance = await syndicate.getMemberPerformance(member.id);

    console.log(`\n${performance.memberName} (${performance.role}):`);
    console.log(`  Tier: ${performance.tier}`);
    console.log(`  Contribution: $${performance.totalContribution.toLocaleString()}`);
    console.log(`  Current Equity: $${performance.currentEquity.toLocaleString()}`);
    console.log(`  Total Profit: $${performance.totalProfit.toLocaleString()}`);
    console.log(`  ROI: ${performance.roi.toFixed(2)}%`);
    console.log(`  Win Rate: ${performance.winRate.toFixed(1)}%`);
    console.log(`  Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
  }

  console.log('\n=== Example Complete ===');
}

// Run example
if (require.main === module) {
  basicSyndicateExample()
    .then(() => {
      console.log('\n✓ Example completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { basicSyndicateExample };
