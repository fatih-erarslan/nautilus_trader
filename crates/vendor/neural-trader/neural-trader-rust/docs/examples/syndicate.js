#!/usr/bin/env node

/**
 * Syndicate Management Examples
 *
 * Demonstrates collaborative trading syndicate features.
 */

const { McpServer } = require('@neural-trader/mcp');

async function main() {
  console.log('👥 Neural Trader MCP - Syndicate Management Examples\n');

  const server = new McpServer({ transport: 'stdio' });
  await server.start();

  const syndicateId = 'alpha-demo-001';

  // Example 1: Create Syndicate
  console.log('🏦 Example 1: Create Trading Syndicate');
  try {
    const syndicate = await server.callTool('create_syndicate', {
      syndicateId,
      name: 'Alpha Demo Syndicate',
      description: 'Professional sports betting with Kelly Criterion'
    });
    console.log('Syndicate created:');
    console.log('  ID:', syndicate.syndicate_id);
    console.log('  Name:', syndicate.name);
    console.log('  Created:', syndicate.created_at);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 2: Add Members
  console.log('👤 Example 2: Add Syndicate Members');
  const members = [
    {
      name: 'Alice Johnson',
      email: 'alice@example.com',
      role: 'senior_analyst',
      contribution: 25000
    },
    {
      name: 'Bob Smith',
      email: 'bob@example.com',
      role: 'junior_analyst',
      contribution: 15000
    },
    {
      name: 'Carol Davis',
      email: 'carol@example.com',
      role: 'quantitative_trader',
      contribution: 35000
    }
  ];

  for (const memberData of members) {
    try {
      const member = await server.callTool('add_syndicate_member', {
        syndicateId,
        ...memberData,
        initialContribution: memberData.contribution
      });
      console.log(`Added ${member.name}:`);
      console.log(`  Role: ${member.role}`);
      console.log(`  Contribution: $${member.contribution.toLocaleString()}`);
      console.log(`  Ownership: ${member.ownership_pct.toFixed(2)}%`);
    } catch (error) {
      console.error(`Error adding ${memberData.name}:`, error.message);
    }
  }
  console.log('');

  // Example 3: Check Syndicate Status
  console.log('📊 Example 3: Syndicate Status');
  try {
    const status = await server.callTool('get_syndicate_status', {
      syndicateId
    });
    console.log('Status:');
    console.log('  Members:', status.member_count);
    console.log('  Total Bankroll:', `$${status.total_bankroll.toLocaleString()}`);
    console.log('  Available Capital:', `$${status.available_capital.toLocaleString()}`);
    console.log('  Active Bets:', status.active_bets);
    console.log('  Total Bets:', status.total_bets_placed);
    console.log('  P&L:', `$${status.total_profit_loss.toLocaleString()}`);
    console.log('  ROI:', `${(status.roi * 100).toFixed(2)}%`);
    console.log('  Win Rate:', `${(status.win_rate * 100).toFixed(2)}%`);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 4: Allocate Funds (Kelly Criterion)
  console.log('💰 Example 4: Allocate Funds Using Kelly Criterion');
  const opportunities = [
    {
      id: 'nfl_001',
      sport: 'NFL',
      event: 'Chiefs vs Eagles',
      odds: 2.15,
      probability: 0.52,
      edge: 0.045,
      confidence: 0.85
    },
    {
      id: 'nba_002',
      sport: 'NBA',
      event: 'Lakers vs Warriors',
      odds: 1.95,
      probability: 0.55,
      edge: 0.055,
      confidence: 0.80
    },
    {
      id: 'mlb_003',
      sport: 'MLB',
      event: 'Yankees vs Red Sox',
      odds: 2.30,
      probability: 0.48,
      edge: 0.035,
      confidence: 0.75
    }
  ];

  try {
    const allocation = await server.callTool('allocate_syndicate_funds', {
      syndicateId,
      opportunities,
      strategy: 'kelly_criterion'
    });

    console.log(`Strategy: ${allocation.strategy}`);
    console.log(`Total Allocated: $${allocation.total_allocated.toLocaleString()}`);
    console.log(`Remaining Capital: $${allocation.remaining_capital.toLocaleString()}`);
    console.log(`Portfolio Kelly: ${(allocation.portfolio_kelly * 100).toFixed(2)}%`);
    console.log('\nAllocations:');

    allocation.allocations.forEach((alloc, i) => {
      const opp = opportunities[i];
      console.log(`\n  ${opp.event}:`);
      console.log(`    Stake: $${alloc.recommended_stake.toLocaleString()}`);
      console.log(`    Kelly %: ${(alloc.kelly_percentage * 100).toFixed(2)}%`);
      console.log(`    Expected Value: $${alloc.expected_value.toFixed(2)}`);
      console.log(`    Odds: ${opp.odds}`);
      console.log(`    Edge: ${(opp.edge * 100).toFixed(2)}%`);
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 5: Simulate Portfolio
  console.log('🎲 Example 5: Monte Carlo Portfolio Simulation');
  try {
    const simulation = await server.callTool('simulate_syndicate_allocation', {
      syndicateId,
      opportunities,
      strategies: ['kelly_criterion', 'fractional_kelly'],
      monteCarloSimulations: 10000
    });

    console.log('Simulation Results:');
    console.log('  Simulations:', simulation.total_simulations);
    console.log('  Mean Return:', `$${simulation.mean_return.toLocaleString()}`);
    console.log('  Median Return:', `$${simulation.median_return.toLocaleString()}`);
    console.log('  Best Case:', `$${simulation.best_case.toLocaleString()}`);
    console.log('  Worst Case:', `$${simulation.worst_case.toLocaleString()}`);
    console.log('  Win Probability:', `${(simulation.win_probability * 100).toFixed(2)}%`);
    console.log('  Risk of Ruin:', `${(simulation.risk_of_ruin * 100).toFixed(4)}%`);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 6: Distribute Profits
  console.log('💸 Example 6: Distribute Profits');
  try {
    const distribution = await server.callTool('distribute_syndicate_profits', {
      syndicateId,
      totalProfit: 15000,
      model: 'hybrid'  // 70% capital, 30% performance
    });

    console.log('Distribution Model:', distribution.distribution_model);
    console.log('Total Profit:', `$${distribution.total_profit.toLocaleString()}`);
    console.log('\nDistributions:');

    distribution.distributions.forEach(dist => {
      console.log(`\n  ${dist.member_id}:`);
      console.log(`    Amount: $${dist.amount.toLocaleString()}`);
      console.log(`    Percentage: ${dist.percentage.toFixed(2)}%`);
      console.log(`    Basis: ${dist.basis}`);
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 7: Create Governance Vote
  console.log('🗳️  Example 7: Create Governance Vote');
  try {
    const vote = await server.callTool('create_syndicate_vote', {
      syndicateId,
      voteType: 'strategy_change',
      proposal: 'Increase maximum single bet from 5% to 7% of bankroll',
      options: ['Approve', 'Reject'],
      durationHours: 48
    });

    console.log('Vote created:');
    console.log('  ID:', vote.vote_id);
    console.log('  Type:', vote.vote_type);
    console.log('  Proposal:', vote.proposal);
    console.log('  Expires:', vote.expires_at);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Example 8: Member Performance
  console.log('📈 Example 8: Member Performance Metrics');
  try {
    const performance = await server.callTool('get_syndicate_member_performance', {
      syndicateId,
      memberId: 'member_001'
    });

    console.log('Performance:');
    console.log('  Total Contributed:', `$${performance.total_contributed.toLocaleString()}`);
    console.log('  Current Balance:', `$${performance.current_balance.toLocaleString()}`);
    console.log('  Total Return:', `$${performance.total_return.toLocaleString()}`);
    console.log('  ROI:', `${(performance.roi * 100).toFixed(2)}%`);
    console.log('  Sharpe Ratio:', performance.sharpe_ratio.toFixed(2));
    console.log('  Win Rate:', `${(performance.win_rate * 100).toFixed(2)}%`);
  } catch (error) {
    console.error('Error:', error.message);
  }
  console.log('');

  // Clean up
  await server.stop();
  console.log('✅ Examples complete');
}

main().catch(error => {
  console.error('❌ Error:', error.message);
  process.exit(1);
});
