const assert = require('assert');
const {
  createSyndicate,
  calculateKelly,
  calculateKellyFractional,
  AllocationStrategy,
  DistributionModel,
  MemberRole,
  MemberTier
} = require('..');

async function runTests() {
  console.log('Running @neural-trader/syndicate tests...\n');

  // Test 1: Kelly Criterion calculation
  console.log('Test 1: Kelly Criterion calculation');
  const kelly = calculateKelly(0.55, 2.0);
  console.log(`  Kelly for 55% win rate at 2.0 odds: ${(kelly * 100).toFixed(2)}%`);
  assert(kelly > 0 && kelly < 1, 'Kelly should be between 0 and 1');
  console.log('  ✓ Passed\n');

  // Test 2: Fractional Kelly
  console.log('Test 2: Fractional Kelly calculation');
  const fullKelly = calculateKelly(0.60, 2.5);
  const halfKelly = calculateKellyFractional(0.60, 2.5, 0.5);
  console.log(`  Full Kelly: ${(fullKelly * 100).toFixed(2)}%`);
  console.log(`  Half Kelly: ${(halfKelly * 100).toFixed(2)}%`);
  assert(Math.abs(halfKelly - fullKelly * 0.5) < 0.0001, 'Half Kelly should be half of full Kelly');
  console.log('  ✓ Passed\n');

  // Test 3: Syndicate creation
  console.log('Test 3: Syndicate creation');
  const syndicate = await createSyndicate('test-syndicate-001', '100000.00');
  assert(syndicate, 'Syndicate should be created');
  console.log('  Created syndicate with $100,000 bankroll');
  console.log('  ✓ Passed\n');

  // Test 4: Add members
  console.log('Test 4: Add syndicate members');
  const member1Id = await syndicate.addMember(
    'Alice Johnson',
    'alice@example.com',
    MemberRole.LeadInvestor,
    '40000.00'
  );
  const member2Id = await syndicate.addMember(
    'Bob Smith',
    'bob@example.com',
    MemberRole.SeniorAnalyst,
    '30000.00'
  );
  const member3Id = await syndicate.addMember(
    'Carol Davis',
    'carol@example.com',
    MemberRole.JuniorAnalyst,
    '20000.00'
  );
  console.log(`  Added 3 members: ${member1Id}, ${member2Id}, ${member3Id}`);

  const members = await syndicate.getMembers(true);
  assert(members.length === 3, 'Should have 3 members');
  console.log('  ✓ Passed\n');

  // Test 5: Fund allocation
  console.log('Test 5: Fund allocation');
  const opportunity = {
    id: 'bet-001',
    sport: 'Basketball',
    event: 'Lakers vs Celtics',
    betType: 'Moneyline',
    odds: 2.1,
    probability: 0.55,
    edge: 0.055,
    confidence: 0.8
  };

  const allocation = await syndicate.allocateFunds(
    opportunity,
    AllocationStrategy.KellyCriterion
  );

  console.log(`  Allocated: $${allocation.allocatedAmount}`);
  console.log(`  Kelly %: ${(allocation.kellyPercentage * 100).toFixed(2)}%`);
  console.log(`  Expected Value: ${allocation.expectedValue.toFixed(2)}`);
  assert(allocation.approved, 'Allocation should be approved');
  console.log('  ✓ Passed\n');

  // Test 6: Profit distribution
  console.log('Test 6: Profit distribution');
  const profit = '5000.00';
  const distribution = await syndicate.distributeProfits(
    profit,
    DistributionModel.Proportional
  );

  let totalDistributed = 0;
  for (const [memberId, amount] of distribution.entries()) {
    const member = await syndicate.getMember(memberId);
    const parsedAmount = parseFloat(amount);
    totalDistributed += parsedAmount;
    console.log(`  ${member.name}: $${amount}`);
  }

  assert(Math.abs(totalDistributed - parseFloat(profit)) < 0.01, 'Total distributed should equal profit');
  console.log('  ✓ Passed\n');

  // Test 7: Bankroll status
  console.log('Test 7: Bankroll status');
  const bankroll = await syndicate.getBankrollStatus();
  console.log(`  Total: $${bankroll.total}`);
  console.log(`  Available: $${bankroll.available}`);
  console.log(`  Allocated: $${bankroll.allocated}`);
  console.log(`  Reserve: $${bankroll.reserve}`);
  assert(parseFloat(bankroll.total) > 0, 'Total bankroll should be positive');
  console.log('  ✓ Passed\n');

  // Test 8: Risk metrics
  console.log('Test 8: Risk metrics calculation');
  const riskMetrics = await syndicate.getRiskMetrics();
  console.log(`  Total Exposure: $${riskMetrics.totalExposure}`);
  console.log(`  Daily Exposure: $${riskMetrics.dailyExposure}`);
  console.log(`  Sharpe Ratio: ${riskMetrics.sharpeRatio.toFixed(2)}`);
  console.log(`  Max Drawdown: ${(riskMetrics.maxDrawdown * 100).toFixed(2)}%`);
  console.log('  ✓ Passed\n');

  // Test 9: Member performance
  console.log('Test 9: Member performance statistics');
  const member1 = await syndicate.getMember(member1Id);
  console.log(`  ${member1.name} Statistics:`);
  console.log(`    Performance Score: ${member1.performanceScore.toFixed(2)}`);
  console.log(`    Role: ${member1.role}`);
  console.log(`    Tier: ${member1.tier}`);
  console.log(`    Capital: $${member1.capitalContribution}`);
  console.log('  ✓ Passed\n');

  // Test 10: Performance report
  console.log('Test 10: Performance report generation');
  const report = await syndicate.generatePerformanceReport();
  console.log(`  Period: ${report.period.start.toISOString().split('T')[0]} to ${report.period.end.toISOString().split('T')[0]}`);
  console.log(`  Starting Bankroll: $${report.bankroll.starting}`);
  console.log(`  Ending Bankroll: $${report.bankroll.ending}`);
  console.log(`  Total Bets: ${report.betting.totalBets}`);
  console.log(`  Active Members: ${report.members.activeMembers}`);
  console.log('  ✓ Passed\n');

  console.log('✓ All tests passed!');
}

runTests().catch(error => {
  console.error('Test failed:', error);
  process.exit(1);
});
