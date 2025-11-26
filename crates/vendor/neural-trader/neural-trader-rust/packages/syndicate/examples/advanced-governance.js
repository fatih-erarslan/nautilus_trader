/**
 * Advanced Governance Example
 *
 * Demonstrates:
 * - Creating proposals and votes
 * - Different vote types
 * - Voting with weighted votes
 * - Vote resolution and execution
 * - Dispute resolution
 */

const {
  SyndicateManager,
  MemberRole,
  VoteType,
  AllocationStrategy
} = require('@neural-trader/syndicate');

async function advancedGovernanceExample() {
  console.log('=== Advanced Governance Example ===\n');

  // Setup syndicate
  const manager = new SyndicateManager();
  const syndicate = await manager.createSyndicate({
    id: 'governance-demo',
    name: 'Governance Demo Syndicate',
    initialCapital: 200000,
    config: {
      maxSingleBet: 0.05,
      maxDailyExposure: 0.20,
      votingQuorum: 0.60,
      votingPeriod: 48
    }
  });

  // Add members
  const alice = await syndicate.addMember({
    name: 'Alice',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 80000
  });

  const bob = await syndicate.addMember({
    name: 'Bob',
    email: 'bob@example.com',
    role: MemberRole.SeniorAnalyst,
    initialContribution: 60000
  });

  const carol = await syndicate.addMember({
    name: 'Carol',
    email: 'carol@example.com',
    role: MemberRole.JuniorAnalyst,
    initialContribution: 40000
  });

  const david = await syndicate.addMember({
    name: 'David',
    email: 'david@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 20000
  });

  console.log('✓ Syndicate and members created\n');

  // Example 1: Strategy Vote
  console.log('Example 1: Strategy Change Vote');
  console.log('================================\n');

  const strategyVote = await syndicate.createVote({
    type: VoteType.Strategy,
    proposal: 'Increase fractional Kelly from 0.25 to 0.30',
    description: `
      Current Strategy: Quarter Kelly (0.25)
      Proposed Strategy: 0.30 Kelly

      Rationale:
      - Win rate improved from 52% to 58% over last 100 bets
      - Sharpe ratio increased from 1.2 to 1.8
      - More confident in probability estimates

      Expected Impact:
      - Average bet size: +20%
      - Expected annual return: +15%
      - Volatility: +10%
      - Max drawdown: +5%

      Risk Mitigation:
      - Maintain stop-loss at 10% daily loss
      - Reduce to Quarter Kelly if drawdown exceeds 15%
      - Review after 50 bets
    `,
    options: ['approve', 'reject', 'defer_for_review'],
    duration: 48,
    quorum: 0.70
  });

  console.log(`✓ Strategy vote created: ${strategyVote.id}`);
  console.log(`  Proposal: ${strategyVote.proposal}`);
  console.log(`  Options: ${strategyVote.options.join(', ')}`);
  console.log(`  Quorum: ${(strategyVote.quorum * 100)}%`);
  console.log(`  Duration: ${strategyVote.duration} hours\n`);

  // Members cast votes
  console.log('Voting in progress...');
  await syndicate.castVote(strategyVote.id, alice.id, 'approve');
  console.log(`  ${alice.name} voted: approve (40% weight)`);

  await syndicate.castVote(strategyVote.id, bob.id, 'approve');
  console.log(`  ${bob.name} voted: approve (30% weight)`);

  await syndicate.castVote(strategyVote.id, carol.id, 'defer_for_review');
  console.log(`  ${carol.name} voted: defer_for_review (20% weight)`);

  await syndicate.castVote(strategyVote.id, david.id, 'reject');
  console.log(`  ${david.name} voted: reject (10% weight)\n`);

  // Check vote status
  const strategyStatus = await strategyVote.getStatus();
  console.log('Vote Status:');
  console.log(`  Participation: ${(strategyStatus.participation * 100).toFixed(1)}%`);
  console.log(`  Quorum met: ${strategyStatus.quorumMet ? 'Yes' : 'No'}`);
  console.log(`  Results:`);
  console.log(`    Approve: ${(strategyStatus.results.approve * 100).toFixed(1)}%`);
  console.log(`    Reject: ${(strategyStatus.results.reject * 100).toFixed(1)}%`);
  console.log(`    Defer: ${(strategyStatus.results.defer_for_review * 100).toFixed(1)}%`);
  console.log(`  Outcome: ${strategyStatus.approved ? 'APPROVED' : 'REJECTED'}\n`);

  if (strategyStatus.approved) {
    await strategyVote.execute();
    console.log('✓ Strategy change executed\n');
  }

  // Example 2: Large Allocation Vote
  console.log('\nExample 2: Large Allocation Vote');
  console.log('=================================\n');

  const allocationVote = await syndicate.createVote({
    type: VoteType.Allocation,
    proposal: 'Allocate $40,000 to Super Bowl prop bet',
    description: `
      Opportunity: Travis Kelce Over 6.5 Receptions
      Bookmaker: DraftKings
      Odds: 1.95
      Our Probability: 68%
      Kelly Recommendation: 15.2%
      Proposed Allocation: $40,000 (20% of bankroll)

      Analysis:
      - Kelce averages 7.8 receptions in playoffs
      - 49ers weak vs TEs (28th ranked)
      - Weather conditions favorable
      - Mahomes relies heavily on Kelce in big games

      Risk Assessment:
      - Largest single bet to date
      - Exceeds normal 5% limit (requires approval)
      - High confidence play (68% vs 51.3% implied)
      - 16.7% edge

      Expected Value: $6,320
    `,
    options: [
      'approve_$40000',
      'approve_$30000',
      'approve_$20000',
      'reject'
    ],
    duration: 24,  // Urgent - game soon
    quorum: 0.60
  });

  console.log(`✓ Allocation vote created: ${allocationVote.id}`);
  console.log(`  Amount: $40,000 (20% of bankroll)`);
  console.log(`  Urgency: High (24 hour window)\n`);

  // Voting
  await syndicate.castVote(allocationVote.id, alice.id, 'approve_$40000');
  console.log(`  ${alice.name} voted: approve_$40000`);

  await syndicate.castVote(allocationVote.id, bob.id, 'approve_$30000');
  console.log(`  ${bob.name} voted: approve_$30000`);

  await syndicate.castVote(allocationVote.id, carol.id, 'approve_$40000');
  console.log(`  ${carol.name} voted: approve_$40000`);

  await syndicate.castVote(allocationVote.id, david.id, 'reject');
  console.log(`  ${david.name} voted: reject\n`);

  const allocationStatus = await allocationVote.getStatus();
  console.log('Vote Result:');
  console.log(`  Winning option: ${allocationStatus.winningOption}`);
  console.log(`  Approved: ${allocationStatus.approved ? 'Yes' : 'No'}\n`);

  // Example 3: Member Role Change Vote
  console.log('\nExample 3: Member Promotion Vote');
  console.log('=================================\n');

  const memberVote = await syndicate.createVote({
    type: VoteType.Member,
    proposal: 'Promote Carol to Senior Analyst',
    description: `
      Member: Carol Davis
      Current Role: Analyst
      Proposed Role: Senior Analyst

      Performance:
      - 18 months as Analyst
      - 64% win rate on recommendations
      - +$89,000 profit contribution
      - Strongest performer in NBA markets
      - Consistent accurate probability estimates

      Qualifications:
      - Deep statistical analysis skills
      - Strong track record
      - Leadership in research initiatives
      - Mentoring junior members

      New Responsibilities:
      - Independent allocation authority
      - Strategy modification rights
      - Access to private syndicate data
      - Higher performance bonus tier
    `,
    options: ['approve', 'reject', 'defer_6_months'],
    duration: 72,
    quorum: 0.75  // High quorum for member changes
  });

  console.log(`✓ Member vote created: ${memberVote.id}`);
  console.log(`  Type: Promotion`);
  console.log(`  Quorum: 75% (high for member changes)\n`);

  // Example 4: Emergency Withdrawal Vote
  console.log('\nExample 4: Emergency Withdrawal Vote');
  console.log('====================================\n');

  const withdrawalVote = await syndicate.createVote({
    type: VoteType.Withdrawal,
    proposal: 'Approve Bob emergency withdrawal of $30,000',
    description: `
      Member: Bob Smith
      Current Equity: $72,000
      Withdrawal Amount: $30,000
      Remaining Equity: $42,000
      Reason: Medical emergency in family

      Impact Analysis:
      - Syndicate capital: $225,000 → $195,000 (-13.3%)
      - Reserve ratio: 12% → 10.5% (above minimum)
      - Active positions: Sufficient liquidity
      - No forced liquidation required

      Timeline:
      - If approved: Funds in 24 hours
      - No penalty for emergency withdrawal
      - Automatic tier adjustment: Gold → Silver

      Member Standing:
      - Good standing, no violations
      - Consistent contributor
      - May return capital when situation improves
    `,
    options: [
      'approve_full_$30000',
      'approve_partial_$20000',
      'defer_7_days',
      'reject'
    ],
    duration: 12,  // Emergency fast-track
    quorum: 0.60
  });

  console.log(`✓ Emergency withdrawal vote created: ${withdrawalVote.id}`);
  console.log(`  Type: Emergency`);
  console.log(`  Duration: 12 hours (fast-tracked)\n`);

  // Example 5: Constitutional Amendment
  console.log('\nExample 5: Constitutional Amendment');
  console.log('===================================\n');

  const amendmentVote = await syndicate.createVote({
    type: VoteType.Amendment,
    proposal: 'Reduce standard voting quorum from 60% to 50%',
    description: `
      Current Rule: 60% quorum for standard votes
      Proposed Rule: 50% quorum for standard votes

      Rationale:
      - 60% difficult to achieve consistently
      - Average participation: 57%
      - Many votes expiring without resolution
      - Slowing critical decisions

      Safeguards Maintained:
      - Member votes: Still 75% quorum
      - Amendments: Still 80% quorum
      - Supermajority: Still required for major changes
      - Only affects routine votes

      Expected Impact:
      - 30% more votes reaching resolution
      - Faster decision-making
      - Improved member engagement
      - Still requires majority approval

      Implementation:
      - Effective immediately upon approval
      - 30-day trial period
      - Review and adjust if needed
    `,
    options: ['approve', 'reject', 'reduce_to_55%'],
    duration: 96,  // 4 days for constitutional changes
    quorum: 0.80   // Very high bar for amendments
  });

  console.log(`✓ Amendment vote created: ${amendmentVote.id}`);
  console.log(`  Type: Constitutional Amendment`);
  console.log(`  Quorum: 80% (highest level)`);
  console.log(`  Duration: 96 hours (extended)\n`);

  // Example 6: Vote with Discussion Period
  console.log('\nExample 6: Vote with Discussion Period');
  console.log('======================================\n');

  const discussion = await syndicate.createDiscussion({
    topic: 'Should we expand into soccer betting?',
    duration: 24,
    participants: [alice.id, bob.id, carol.id, david.id]
  });

  console.log(`✓ Discussion created: ${discussion.id}`);
  console.log(`  Topic: ${discussion.topic}`);
  console.log(`  Duration: 24 hours before voting\n`);

  // Add comments
  await discussion.addComment({
    memberId: alice.id,
    comment: 'Soccer has better market inefficiencies than US sports'
  });
  console.log(`  ${alice.name}: Supports expansion`);

  await discussion.addComment({
    memberId: bob.id,
    comment: 'Need to hire soccer specialist first, we lack expertise'
  });
  console.log(`  ${bob.name}: Concerns about expertise`);

  await discussion.addComment({
    memberId: carol.id,
    comment: 'I can lead soccer analysis, played college soccer and follow EPL closely'
  });
  console.log(`  ${carol.name}: Volunteers to lead\n`);

  // Summary
  console.log('\n=== Governance Summary ===');
  console.log(`Total Votes Created: 6`);
  console.log(`Average Participation: 95%`);
  console.log(`Votes Passed: 4`);
  console.log(`Votes Rejected: 1`);
  console.log(`Votes Pending: 1`);
  console.log(`Discussions: 1 active`);

  console.log('\n=== Example Complete ===');
}

// Run example
if (require.main === module) {
  advancedGovernanceExample()
    .then(() => {
      console.log('\n✓ Governance example completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { advancedGovernanceExample };
