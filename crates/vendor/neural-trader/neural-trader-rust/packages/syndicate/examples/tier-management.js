/**
 * Tier Management Example
 *
 * Demonstrates:
 * - Membership tier system
 * - Tier benefits and requirements
 * - Automatic tier upgrades/downgrades
 * - Tier-based voting weights
 * - Performance bonuses by tier
 */

const {
  SyndicateManager,
  MemberRole,
  MemberTier,
  DistributionModel
} = require('@neural-trader/syndicate');

async function tierManagementExample() {
  console.log('=== Tier Management Example ===\n');

  // Create syndicate
  const manager = new SyndicateManager();
  const syndicate = await manager.createSyndicate({
    id: 'tier-demo',
    name: 'Tier Management Demo',
    initialCapital: 500000,
    config: {
      tierSystem: {
        enabled: true,
        autoUpgrade: true,
        reviewPeriod: 'monthly'
      }
    }
  });

  console.log('✓ Syndicate created with tier system enabled\n');

  // Example 1: Tier Requirements
  console.log('Example 1: Tier Requirements and Benefits');
  console.log('=========================================\n');

  const tierInfo = syndicate.getTierInfo();

  console.log('Platinum Tier:');
  console.log(`  Minimum Capital: $${tierInfo.platinum.minCapital.toLocaleString()}`);
  console.log(`  Vote Weight: ${tierInfo.platinum.voteWeight}x`);
  console.log(`  Performance Bonus: ${tierInfo.platinum.performanceBonus}%`);
  console.log(`  Fee Discount: ${tierInfo.platinum.feeDiscount}%`);
  console.log(`  Benefits:`);
  tierInfo.platinum.benefits.forEach(b => console.log(`    - ${b}`));

  console.log('\nGold Tier:');
  console.log(`  Minimum Capital: $${tierInfo.gold.minCapital.toLocaleString()}`);
  console.log(`  Vote Weight: ${tierInfo.gold.voteWeight}x`);
  console.log(`  Performance Bonus: ${tierInfo.gold.performanceBonus}%`);
  console.log(`  Fee Discount: ${tierInfo.gold.feeDiscount}%`);
  console.log(`  Benefits:`);
  tierInfo.gold.benefits.forEach(b => console.log(`    - ${b}`));

  console.log('\nSilver Tier:');
  console.log(`  Minimum Capital: $${tierInfo.silver.minCapital.toLocaleString()}`);
  console.log(`  Vote Weight: ${tierInfo.silver.voteWeight}x`);
  console.log(`  Performance Bonus: ${tierInfo.silver.performanceBonus}%`);
  console.log(`  Fee Discount: ${tierInfo.silver.feeDiscount}%`);

  console.log('\nBronze Tier:');
  console.log(`  Minimum Capital: $${tierInfo.bronze.minCapital.toLocaleString()}`);
  console.log(`  Vote Weight: ${tierInfo.bronze.voteWeight}x`);
  console.log(`  Performance Bonus: ${tierInfo.bronze.performanceBonus}%`);
  console.log(`  Fee Discount: ${tierInfo.bronze.feeDiscount}%\n`);

  // Example 2: Add Members with Different Tiers
  console.log('\nExample 2: Add Members with Different Tiers');
  console.log('==========================================\n');

  const alice = await syndicate.addMember({
    name: 'Alice - Platinum',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 150000  // Platinum tier
  });
  console.log(`✓ ${alice.name} added - Tier: ${alice.tier}`);

  const bob = await syndicate.addMember({
    name: 'Bob - Gold',
    email: 'bob@example.com',
    role: MemberRole.SeniorAnalyst,
    initialContribution: 80000  // Gold tier
  });
  console.log(`✓ ${bob.name} added - Tier: ${bob.tier}`);

  const carol = await syndicate.addMember({
    name: 'Carol - Silver',
    email: 'carol@example.com',
    role: MemberRole.JuniorAnalyst,
    initialContribution: 40000  // Silver tier
  });
  console.log(`✓ ${carol.name} added - Tier: ${carol.tier}`);

  const david = await syndicate.addMember({
    name: 'David - Bronze',
    email: 'david@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 15000  // Bronze tier
  });
  console.log(`✓ ${david.name} added - Tier: ${david.tier}\n`);

  // Example 3: Tier-Based Voting Weights
  console.log('\nExample 3: Tier-Based Voting Weights');
  console.log('===================================\n');

  const votingPower = await syndicate.calculateVotingPower();

  console.log('Voting Power Distribution:');
  votingPower.members.forEach(m => {
    console.log(`  ${m.name}:`);
    console.log(`    Tier: ${m.tier}`);
    console.log(`    Capital: $${m.capital.toLocaleString()}`);
    console.log(`    Tier Weight: ${m.tierWeight}x`);
    console.log(`    Voting Power: ${(m.votingPower * 100).toFixed(1)}%`);
  });

  console.log(`\nTotal Voting Power: ${(votingPower.total * 100).toFixed(0)}%\n`);

  // Example 4: Tier-Based Profit Distribution
  console.log('\nExample 4: Tier-Based Profit Distribution');
  console.log('========================================\n');

  const profit = 50000;
  const distributions = await syndicate.distributeProfits({
    profit,
    model: DistributionModel.Tiered
  });

  console.log(`Distributing $${profit.toLocaleString()} profit (Tiered model):\n`);
  distributions.forEach(dist => {
    console.log(`${dist.memberName}:`);
    console.log(`  Base Share: $${dist.baseShare.toLocaleString()}`);
    console.log(`  Tier Bonus: +${dist.tierBonusPercent}% ($${dist.tierBonus.toLocaleString()})`);
    console.log(`  Total: $${dist.amount.toLocaleString()}`);
    console.log(`  Percentage: ${(dist.percentage * 100).toFixed(1)}%\n`);
  });

  // Example 5: Automatic Tier Upgrade
  console.log('\nExample 5: Automatic Tier Upgrade');
  console.log('================================\n');

  console.log(`David current status:`);
  console.log(`  Tier: ${david.tier}`);
  console.log(`  Capital: $${david.equity.toLocaleString()}\n`);

  // David adds more capital
  console.log('David adds $15,000 more capital...');
  await syndicate.addMemberContribution({
    memberId: david.id,
    amount: 15000
  });

  // Check for tier upgrade
  const davidUpdated = await syndicate.getMember(david.id);
  console.log(`\nDavid updated status:`);
  console.log(`  Tier: ${davidUpdated.tier}`);
  console.log(`  Capital: $${davidUpdated.equity.toLocaleString()}`);
  console.log(`  Upgrade: ${davidUpdated.tier !== david.tier ? 'Yes! Bronze → Silver' : 'No'}\n`);

  // Example 6: Tier Review and Adjustment
  console.log('\nExample 6: Monthly Tier Review');
  console.log('=============================\n');

  console.log('Running monthly tier review...\n');
  const tierReview = await syndicate.conductTierReview();

  console.log('Tier Review Results:');
  tierReview.changes.forEach(change => {
    console.log(`  ${change.memberName}:`);
    console.log(`    Current Tier: ${change.currentTier}`);
    console.log(`    New Tier: ${change.newTier}`);
    console.log(`    Reason: ${change.reason}`);
    console.log(`    Action: ${change.action}\n`);
  });

  console.log(`Total Changes: ${tierReview.changes.length}`);
  console.log(`Upgrades: ${tierReview.upgrades}`);
  console.log(`Downgrades: ${tierReview.downgrades}`);
  console.log(`No Change: ${tierReview.noChange}\n`);

  // Example 7: Tier Benefits Comparison
  console.log('\nExample 7: Tier Benefits Comparison');
  console.log('==================================\n');

  const comparison = await syndicate.compareTierBenefits({
    currentTier: MemberTier.Silver,
    targetTier: MemberTier.Gold,
    memberCapital: 40000
  });

  console.log('Silver → Gold Upgrade Analysis:');
  console.log(`  Additional Capital Needed: $${comparison.additionalCapital.toLocaleString()}`);
  console.log(`  Total Capital Required: $${comparison.requiredCapital.toLocaleString()}\n`);

  console.log('Additional Benefits:');
  comparison.additionalBenefits.forEach(b => {
    console.log(`  ✓ ${b}`);
  });

  console.log('\nFinancial Impact (Annual):');
  console.log(`  Performance Bonus: +${comparison.performanceBonusIncrease}%`);
  console.log(`  Estimated Annual Gain: $${comparison.estimatedAnnualGain.toLocaleString()}`);
  console.log(`  Fee Savings: $${comparison.annualFeeSavings.toLocaleString()}`);
  console.log(`  Total Benefit: $${comparison.totalAnnualBenefit.toLocaleString()}`);
  console.log(`  ROI on Additional Capital: ${comparison.roiOnAdditionalCapital.toFixed(2)}%\n`);

  // Example 8: Tier-Based Fee Structure
  console.log('\nExample 8: Tier-Based Fee Structure');
  console.log('==================================\n');

  const feeAnalysis = await syndicate.analyzeFees();

  console.log('Fee Structure by Tier:\n');
  feeAnalysis.tiers.forEach(tier => {
    console.log(`${tier.name}:`);
    console.log(`  Management Fee: ${tier.managementFee}%`);
    console.log(`  Performance Fee: ${tier.performanceFee}%`);
    console.log(`  Discount: ${tier.discount}%`);
    console.log(`  Effective Mgmt Fee: ${tier.effectiveManagementFee}%`);
    console.log(`  Effective Perf Fee: ${tier.effectivePerformanceFee}%\n`);
  });

  console.log('Annual Fee Projection:');
  feeAnalysis.members.forEach(m => {
    console.log(`  ${m.name} (${m.tier}):`);
    console.log(`    Management Fee: $${m.annualManagementFee.toLocaleString()}`);
    console.log(`    Performance Fee: $${m.annualPerformanceFee.toLocaleString()}`);
    console.log(`    Total Fees: $${m.totalAnnualFees.toLocaleString()}`);
    console.log(`    Savings vs Bronze: $${m.savingsVsBronze.toLocaleString()}\n`);
  });

  // Example 9: Tier Progression Path
  console.log('\nExample 9: Member Tier Progression Path');
  console.log('======================================\n');

  const progression = await syndicate.getMemberTierProgression(carol.id);

  console.log(`${carol.name}'s Tier Progression:\n`);
  console.log('History:');
  progression.history.forEach(h => {
    console.log(`  ${h.date.toLocaleDateString()}: ${h.tier} (Capital: $${h.capital.toLocaleString()})`);
  });

  console.log('\nNext Tier Goal:');
  console.log(`  Current: ${progression.currentTier}`);
  console.log(`  Target: ${progression.nextTier}`);
  console.log(`  Current Capital: $${progression.currentCapital.toLocaleString()}`);
  console.log(`  Required Capital: $${progression.requiredCapital.toLocaleString()}`);
  console.log(`  Additional Needed: $${progression.additionalNeeded.toLocaleString()}`);
  console.log(`  Progress: ${progression.progress.toFixed(1)}%`);
  console.log(`  Estimated Time: ${progression.estimatedMonths} months\n`);

  // Example 10: Tier Statistics
  console.log('\nExample 10: Syndicate Tier Statistics');
  console.log('====================================\n');

  const stats = await syndicate.getTierStatistics();

  console.log('Tier Distribution:');
  stats.distribution.forEach(tier => {
    const bar = '█'.repeat(Math.floor(tier.percentage / 2));
    console.log(`  ${tier.name.padEnd(10)} ${bar} ${tier.count} members (${tier.percentage.toFixed(1)}%)`);
  });

  console.log('\nCapital by Tier:');
  stats.capitalByTier.forEach(tier => {
    console.log(`  ${tier.name.padEnd(10)} $${tier.capital.toLocaleString().padStart(12)} (${tier.percentage.toFixed(1)}%)`);
  });

  console.log('\nPerformance by Tier:');
  stats.performanceByTier.forEach(tier => {
    console.log(`  ${tier.name.padEnd(10)} ROI: ${tier.roi.toFixed(2)}%  Win Rate: ${tier.winRate.toFixed(1)}%  Sharpe: ${tier.sharpeRatio.toFixed(2)}`);
  });

  // Summary
  console.log('\n\n=== Tier Management Summary ===');
  console.log(`Total Members: ${stats.totalMembers}`);
  console.log(`Platinum: ${stats.distribution[0].count} members`);
  console.log(`Gold: ${stats.distribution[1].count} members`);
  console.log(`Silver: ${stats.distribution[2].count} members`);
  console.log(`Bronze: ${stats.distribution[3].count} members`);
  console.log(`Average Tier: ${stats.averageTier}`);
  console.log(`Tier Mobility: ${stats.tierMobility}% (last 6 months)`);

  console.log('\n=== Example Complete ===');
}

// Run example
if (require.main === module) {
  tierManagementExample()
    .then(() => {
      console.log('\n✓ Tier management example completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { tierManagementExample };
