/**
 * Withdrawal Workflow Example
 *
 * Demonstrates:
 * - Standard withdrawal requests
 * - Emergency withdrawals
 * - Withdrawal approval process
 * - Impact analysis
 * - Tier adjustments
 */

const {
  SyndicateManager,
  MemberRole,
  VoteType
} = require('@neural-trader/syndicate');

async function withdrawalWorkflowExample() {
  console.log('=== Withdrawal Workflow Example ===\n');

  // Setup
  const manager = new SyndicateManager();
  const syndicate = await manager.createSyndicate({
    id: 'withdrawal-demo',
    name: 'Withdrawal Demo Syndicate',
    initialCapital: 250000,
    config: {
      withdrawalDelay: 72,  // 72 hour standard delay
      minReserve: 0.10      // 10% minimum reserve
    }
  });

  const alice = await syndicate.addMember({
    name: 'Alice',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 100000
  });

  const bob = await syndicate.addMember({
    name: 'Bob',
    email: 'bob@example.com',
    role: MemberRole.SeniorAnalyst,
    initialContribution: 80000
  });

  const carol = await syndicate.addMember({
    name: 'Carol',
    email: 'carol@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 50000
  });

  const david = await syndicate.addMember({
    name: 'David',
    email: 'david@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 20000
  });

  console.log('✓ Syndicate setup complete\n');

  // Example 1: Standard Withdrawal
  console.log('Example 1: Standard Withdrawal Request');
  console.log('======================================\n');

  const withdrawal1 = await syndicate.requestWithdrawal({
    memberId: david.id,
    amount: 5000,
    reason: 'Personal expense'
  });

  console.log(`✓ Withdrawal request created:`);
  console.log(`  ID: ${withdrawal1.id}`);
  console.log(`  Member: ${david.name}`);
  console.log(`  Amount: $${withdrawal1.amount.toLocaleString()}`);
  console.log(`  Reason: ${withdrawal1.reason}`);
  console.log(`  Status: ${withdrawal1.status}`);
  console.log(`  Estimated completion: ${withdrawal1.estimatedCompletion}\n`);

  // Check impact
  const impact1 = await syndicate.analyzeWithdrawalImpact(withdrawal1.id);
  console.log('Impact Analysis:');
  console.log(`  Syndicate capital: $${impact1.currentCapital.toLocaleString()} → $${impact1.newCapital.toLocaleString()}`);
  console.log(`  Change: ${impact1.percentageChange.toFixed(2)}%`);
  console.log(`  Reserve ratio: ${impact1.currentReserve.toFixed(2)}% → ${impact1.newReserve.toFixed(2)}%`);
  console.log(`  Within limits: ${impact1.withinLimits ? 'Yes' : 'No'}`);
  console.log(`  Member tier after: ${impact1.newMemberTier}`);
  console.log(`  Requires approval: ${impact1.requiresApproval ? 'Yes' : 'No'}\n`);

  // Auto-approve small withdrawal
  if (!impact1.requiresApproval) {
    await withdrawal1.approve();
    console.log('✓ Withdrawal auto-approved (below threshold)\n');
  }

  // Example 2: Large Withdrawal Requiring Vote
  console.log('\nExample 2: Large Withdrawal Requiring Vote');
  console.log('=========================================\n');

  const withdrawal2 = await syndicate.requestWithdrawal({
    memberId: bob.id,
    amount: 40000,
    reason: 'Real estate investment opportunity'
  });

  console.log(`✓ Large withdrawal request:`);
  console.log(`  Member: ${bob.name}`);
  console.log(`  Amount: $${withdrawal2.amount.toLocaleString()}`);
  console.log(`  Current equity: $${bob.equity.toLocaleString()}`);
  console.log(`  Remaining after: $${(bob.equity - withdrawal2.amount).toLocaleString()}\n`);

  const impact2 = await syndicate.analyzeWithdrawalImpact(withdrawal2.id);
  console.log('Impact Analysis:');
  console.log(`  Capital reduction: ${impact2.percentageChange.toFixed(2)}%`);
  console.log(`  Reserve ratio: ${impact2.newReserve.toFixed(2)}%`);
  console.log(`  Requires vote: ${impact2.requiresApproval ? 'Yes' : 'No'}\n`);

  // Create vote for large withdrawal
  if (impact2.requiresApproval) {
    const vote = await syndicate.createVote({
      type: VoteType.Withdrawal,
      proposal: `Approve Bob withdrawal of $${withdrawal2.amount.toLocaleString()}`,
      description: `
        Member: Bob Smith
        Amount: $${withdrawal2.amount.toLocaleString()} (${((withdrawal2.amount / bob.equity) * 100).toFixed(1)}% of member equity)
        Reason: ${withdrawal2.reason}

        Impact:
        - Syndicate capital: -${impact2.percentageChange.toFixed(1)}%
        - Reserve ratio: ${impact2.newReserve.toFixed(1)}% (${impact2.withinLimits ? 'within' : 'below'} minimum)
        - Bob's remaining equity: $${(bob.equity - withdrawal2.amount).toLocaleString()}
        - Tier change: ${bob.tier} → ${impact2.newMemberTier}

        Liquidity:
        - Available capital: $${syndicate.availableCapital.toLocaleString()}
        - Active positions: $${syndicate.totalInvested.toLocaleString()}
        - Can fulfill without liquidation: ${impact2.requiresLiquidation ? 'No' : 'Yes'}
      `,
      options: [
        'approve_full',
        'approve_partial_$25000',
        'approve_partial_$30000',
        'defer_30_days',
        'reject'
      ],
      duration: 48,
      quorum: 0.60
    });

    console.log('✓ Withdrawal vote created');
    console.log(`  Vote ID: ${vote.id}`);
    console.log(`  Voting period: 48 hours\n`);

    // Members vote
    console.log('Voting in progress...');
    await syndicate.castVote(vote.id, alice.id, 'approve_full');
    console.log(`  ${alice.name}: approve_full`);

    await syndicate.castVote(vote.id, carol.id, 'approve_partial_$30000');
    console.log(`  ${carol.name}: approve_partial_$30000`);

    await syndicate.castVote(vote.id, david.id, 'approve_full');
    console.log(`  ${david.name}: approve_full\n`);

    const voteResult = await vote.getStatus();
    console.log('Vote Result:');
    console.log(`  Winning option: ${voteResult.winningOption}`);
    console.log(`  Approved: ${voteResult.approved ? 'Yes' : 'No'}\n`);

    if (voteResult.approved) {
      await withdrawal2.approve();
      console.log('✓ Withdrawal approved and processing\n');
    }
  }

  // Example 3: Emergency Withdrawal
  console.log('\nExample 3: Emergency Withdrawal');
  console.log('==============================\n');

  const emergencyWithdrawal = await syndicate.requestWithdrawal({
    memberId: carol.id,
    amount: 30000,
    reason: 'Medical emergency - urgent surgery required',
    isEmergency: true
  });

  console.log(`🚨 EMERGENCY WITHDRAWAL REQUEST`);
  console.log(`  Member: ${carol.name}`);
  console.log(`  Amount: $${emergencyWithdrawal.amount.toLocaleString()}`);
  console.log(`  Reason: ${emergencyWithdrawal.reason}`);
  console.log(`  Priority: Emergency (fast-tracked)`);
  console.log(`  Estimated completion: ${emergencyWithdrawal.estimatedCompletion}\n`);

  const emergencyImpact = await syndicate.analyzeWithdrawalImpact(emergencyWithdrawal.id);
  console.log('Emergency Impact Analysis:');
  console.log(`  Capital reduction: ${emergencyImpact.percentageChange.toFixed(2)}%`);
  console.log(`  Within safe limits: ${emergencyImpact.withinLimits ? 'Yes' : 'No'}`);
  console.log(`  Emergency fund available: ${syndicate.emergencyFund >= emergencyWithdrawal.amount ? 'Yes' : 'No'}\n`);

  // Fast-track emergency vote (12 hour window)
  const emergencyVote = await syndicate.createVote({
    type: VoteType.Withdrawal,
    proposal: `EMERGENCY: Approve Carol withdrawal of $${emergencyWithdrawal.amount.toLocaleString()}`,
    description: `
      🚨 EMERGENCY WITHDRAWAL REQUEST

      Member: Carol Davis
      Amount: $${emergencyWithdrawal.amount.toLocaleString()}
      Reason: ${emergencyWithdrawal.reason}

      This is a fast-tracked emergency request requiring immediate attention.

      Member Status:
      - Good standing, no violations
      - Current equity: $${carol.equity.toLocaleString()}
      - Remaining equity: $${(carol.equity - emergencyWithdrawal.amount).toLocaleString()}

      Impact:
      - Syndicate impact: ${emergencyImpact.percentageChange.toFixed(1)}%
      - Reserve still adequate: ${emergencyImpact.withinLimits ? 'Yes' : 'No'}
      - Can be fulfilled immediately: Yes

      Timeline:
      - If approved: Funds transferred within 24 hours
      - No penalty applied (emergency exception)
      - Member may repay when situation improves
    `,
    options: ['approve', 'approve_with_repayment_plan', 'reject'],
    duration: 12,  // Emergency: 12 hours only
    quorum: 0.50   // Lower quorum for emergencies
  });

  console.log('✓ Emergency vote created (12-hour window)');
  console.log(`  Quorum: 50% (reduced for emergency)`);
  console.log(`  Duration: 12 hours\n`);

  // Rapid voting
  await syndicate.castVote(emergencyVote.id, alice.id, 'approve');
  await syndicate.castVote(emergencyVote.id, bob.id, 'approve');
  await syndicate.castVote(emergencyVote.id, david.id, 'approve');

  const emergencyResult = await emergencyVote.getStatus();
  console.log('Emergency Vote Result:');
  console.log(`  Participation: ${(emergencyResult.participation * 100).toFixed(0)}%`);
  console.log(`  Unanimous approval: ${emergencyResult.results.approve === 1.0 ? 'Yes' : 'No'}`);
  console.log(`  Status: APPROVED\n`);

  if (emergencyResult.approved) {
    await emergencyWithdrawal.approve();
    await emergencyWithdrawal.expedite();
    console.log('✓ Emergency withdrawal approved and expedited');
    console.log('  Funds will be transferred within 24 hours\n');
  }

  // Example 4: Withdrawal with Repayment Plan
  console.log('\nExample 4: Withdrawal with Repayment Plan');
  console.log('========================================\n');

  const withdrawal3 = await syndicate.requestWithdrawal({
    memberId: alice.id,
    amount: 50000,
    reason: 'Business investment - temporary capital need',
    repaymentPlan: {
      enabled: true,
      duration: 6,  // 6 months
      interestRate: 0.05  // 5% annual
    }
  });

  console.log(`✓ Withdrawal with repayment plan:`);
  console.log(`  Member: ${alice.name}`);
  console.log(`  Amount: $${withdrawal3.amount.toLocaleString()}`);
  console.log(`  Repayment duration: ${withdrawal3.repaymentPlan.duration} months`);
  console.log(`  Interest rate: ${(withdrawal3.repaymentPlan.interestRate * 100)}%`);
  console.log(`  Monthly payment: $${withdrawal3.repaymentPlan.monthlyPayment.toLocaleString()}`);
  console.log(`  Total repayment: $${withdrawal3.repaymentPlan.totalRepayment.toLocaleString()}\n`);

  // Example 5: Withdrawal History
  console.log('\nExample 5: Withdrawal History');
  console.log('============================\n');

  const history = await syndicate.getWithdrawalHistory({
    limit: 10,
    includeCompleted: true,
    includePending: true,
    includeRejected: true
  });

  console.log('Withdrawal History:');
  history.forEach((w, i) => {
    console.log(`\n${i + 1}. ${w.memberName}`);
    console.log(`   Amount: $${w.amount.toLocaleString()}`);
    console.log(`   Status: ${w.status}`);
    console.log(`   Reason: ${w.reason}`);
    console.log(`   Requested: ${w.requestDate.toLocaleDateString()}`);
    if (w.completionDate) {
      console.log(`   Completed: ${w.completionDate.toLocaleDateString()}`);
    }
  });

  // Summary
  console.log('\n\n=== Withdrawal Summary ===');
  console.log(`Total requests: ${history.length}`);
  console.log(`Approved: ${history.filter(w => w.status === 'approved').length}`);
  console.log(`Pending: ${history.filter(w => w.status === 'pending').length}`);
  console.log(`Rejected: ${history.filter(w => w.status === 'rejected').length}`);
  console.log(`Total withdrawn: $${history.reduce((sum, w) => sum + (w.status === 'completed' ? w.amount : 0), 0).toLocaleString()}`);

  console.log('\n=== Example Complete ===');
}

// Run example
if (require.main === module) {
  withdrawalWorkflowExample()
    .then(() => {
      console.log('\n✓ Withdrawal workflow example completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { withdrawalWorkflowExample };
