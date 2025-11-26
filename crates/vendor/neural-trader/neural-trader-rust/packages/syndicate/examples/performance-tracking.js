/**
 * Performance Tracking Example
 *
 * Demonstrates:
 * - Real-time performance metrics
 * - Member performance analytics
 * - Historical tracking
 * - Benchmark comparison
 * - Performance reports
 */

const {
  SyndicateManager,
  MemberRole,
  AllocationStrategy
} = require('@neural-trader/syndicate');

async function performanceTrackingExample() {
  console.log('=== Performance Tracking Example ===\n');

  // Setup syndicate with history
  const manager = new SyndicateManager();
  const syndicate = await manager.createSyndicate({
    id: 'performance-demo',
    name: 'Performance Tracking Demo',
    initialCapital: 500000
  });

  // Add members
  const alice = await syndicate.addMember({
    name: 'Alice',
    email: 'alice@example.com',
    role: MemberRole.LeadInvestor,
    initialContribution: 200000
  });

  const bob = await syndicate.addMember({
    name: 'Bob',
    email: 'bob@example.com',
    role: MemberRole.SeniorAnalyst,
    initialContribution: 150000
  });

  const carol = await syndicate.addMember({
    name: 'Carol',
    email: 'carol@example.com',
    role: MemberRole.JuniorAnalyst,
    initialContribution: 100000
  });

  const david = await syndicate.addMember({
    name: 'David',
    email: 'david@example.com',
    role: MemberRole.ContributingMember,
    initialContribution: 50000
  });

  console.log('✓ Syndicate setup complete\n');

  // Simulate some betting history
  console.log('Simulating betting history...');
  await simulateBettingHistory(syndicate);
  console.log('✓ History simulated\n');

  // Example 1: Real-Time Syndicate Performance
  console.log('Example 1: Real-Time Syndicate Performance');
  console.log('=========================================\n');

  const performance = await syndicate.getPerformance();

  console.log('Overall Performance:');
  console.log(`  Total Bets: ${performance.totalBets}`);
  console.log(`  Wins: ${performance.wins} (${performance.winRate.toFixed(1)}%)`);
  console.log(`  Losses: ${performance.losses}`);
  console.log(`  Total Wagered: $${performance.totalWagered.toLocaleString()}`);
  console.log(`  Total Won: $${performance.totalWon.toLocaleString()}`);
  console.log(`  Total Lost: $${performance.totalLost.toLocaleString()}`);
  console.log(`  Net Profit: $${performance.netProfit.toLocaleString()}`);
  console.log(`  ROI: ${performance.roi.toFixed(2)}%`);
  console.log(`  Yield: ${performance.yield.toFixed(2)}%\n`);

  console.log('Risk-Adjusted Metrics:');
  console.log(`  Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
  console.log(`  Sortino Ratio: ${performance.sortinoRatio.toFixed(2)}`);
  console.log(`  Calmar Ratio: ${performance.calmarRatio.toFixed(2)}`);
  console.log(`  Max Drawdown: ${performance.maxDrawdown.toFixed(2)}%`);
  console.log(`  Current Drawdown: ${performance.currentDrawdown.toFixed(2)}%`);
  console.log(`  Volatility: ${performance.volatility.toFixed(2)}%\n`);

  console.log('Betting Stats:');
  console.log(`  Average Bet: $${performance.averageBet.toLocaleString()}`);
  console.log(`  Average Win: $${performance.averageWin.toLocaleString()}`);
  console.log(`  Average Loss: $${performance.averageLoss.toLocaleString()}`);
  console.log(`  Profit Factor: ${performance.profitFactor.toFixed(2)}`);
  console.log(`  Kelly Criterion: ${(performance.averageKelly * 100).toFixed(2)}%`);
  console.log(`  Average Odds: ${performance.averageOdds.toFixed(2)}\n`);

  // Example 2: Member Performance Analytics
  console.log('\nExample 2: Member Performance Analytics');
  console.log('======================================\n');

  for (const member of [alice, bob, carol, david]) {
    const memberPerf = await syndicate.getMemberPerformance(member.id);

    console.log(`${memberPerf.memberName} (${memberPerf.role}):`);
    console.log(`  Tier: ${memberPerf.tier}`);
    console.log(`  Capital:`);
    console.log(`    Initial: $${memberPerf.initialContribution.toLocaleString()}`);
    console.log(`    Current: $${memberPerf.currentEquity.toLocaleString()}`);
    console.log(`    Change: ${memberPerf.capitalChange >= 0 ? '+' : ''}$${memberPerf.capitalChange.toLocaleString()}`);
    console.log(`  Performance:`);
    console.log(`    Total Profit: $${memberPerf.totalProfit.toLocaleString()}`);
    console.log(`    ROI: ${memberPerf.roi.toFixed(2)}%`);
    console.log(`    Win Rate: ${memberPerf.winRate.toFixed(1)}%`);
    console.log(`    Sharpe Ratio: ${memberPerf.sharpeRatio.toFixed(2)}`);
    console.log(`  Activity:`);
    console.log(`    Total Bets: ${memberPerf.totalBets}`);
    console.log(`    Average Bet: $${memberPerf.averageBetSize.toLocaleString()}`);
    console.log(`    Largest Win: $${memberPerf.largestWin.toLocaleString()}`);
    console.log(`    Largest Loss: $${memberPerf.largestLoss.toLocaleString()}`);
    console.log();
  }

  // Example 3: Historical Performance
  console.log('\nExample 3: Historical Performance');
  console.log('================================\n');

  const monthlyPerf = await syndicate.getMonthlyPerformance({
    months: 12
  });

  console.log('Last 12 Months Performance:');
  monthlyPerf.forEach(month => {
    console.log(`\n${month.month}:`);
    console.log(`  Bets: ${month.bets}`);
    console.log(`  Win Rate: ${month.winRate.toFixed(1)}%`);
    console.log(`  Profit: $${month.profit.toLocaleString()}`);
    console.log(`  ROI: ${month.roi.toFixed(2)}%`);
    console.log(`  Sharpe: ${month.sharpeRatio.toFixed(2)}`);
  });

  // Chart monthly profit
  console.log('\n\nMonthly Profit Chart:');
  const maxProfit = Math.max(...monthlyPerf.map(m => m.profit));
  monthlyPerf.forEach(month => {
    const barLength = Math.floor((month.profit / maxProfit) * 50);
    const bar = '█'.repeat(Math.max(0, barLength));
    console.log(`${month.month.substring(0, 7)} ${bar} $${month.profit.toLocaleString()}`);
  });

  // Example 4: Benchmark Comparison
  console.log('\n\nExample 4: Benchmark Comparison');
  console.log('==============================\n');

  const benchmark = await syndicate.compareToBenchmark({
    benchmarkName: 'S&P 500',
    period: '1y'
  });

  console.log('vs S&P 500:');
  console.log(`  Syndicate ROI: ${benchmark.syndicateROI.toFixed(2)}%`);
  console.log(`  Benchmark ROI: ${benchmark.benchmarkROI.toFixed(2)}%`);
  console.log(`  Alpha: ${benchmark.alpha.toFixed(2)}%`);
  console.log(`  Beta: ${benchmark.beta.toFixed(2)}`);
  console.log(`  Correlation: ${benchmark.correlation.toFixed(2)}`);
  console.log(`  Outperformance: ${benchmark.outperformance ? 'Yes' : 'No'}\n`);

  console.log('Risk Comparison:');
  console.log(`  Syndicate Sharpe: ${benchmark.syndicateSharpe.toFixed(2)}`);
  console.log(`  Benchmark Sharpe: ${benchmark.benchmarkSharpe.toFixed(2)}`);
  console.log(`  Syndicate Volatility: ${benchmark.syndicateVolatility.toFixed(2)}%`);
  console.log(`  Benchmark Volatility: ${benchmark.benchmarkVolatility.toFixed(2)}%`);
  console.log(`  Information Ratio: ${benchmark.informationRatio.toFixed(2)}\n`);

  // Example 5: Performance by Sport
  console.log('\nExample 5: Performance by Sport');
  console.log('==============================\n');

  const sportPerf = await syndicate.getPerformanceBySport();

  console.log('Sport Performance Breakdown:');
  sportPerf.forEach(sport => {
    console.log(`\n${sport.name}:`);
    console.log(`  Bets: ${sport.bets}`);
    console.log(`  Win Rate: ${sport.winRate.toFixed(1)}%`);
    console.log(`  Profit: $${sport.profit.toLocaleString()}`);
    console.log(`  ROI: ${sport.roi.toFixed(2)}%`);
    console.log(`  Average Odds: ${sport.averageOdds.toFixed(2)}`);
    console.log(`  Sharpe: ${sport.sharpeRatio.toFixed(2)}`);
  });

  // Find best and worst sports
  const bestSport = sportPerf.reduce((best, sport) =>
    sport.roi > best.roi ? sport : best
  );
  const worstSport = sportPerf.reduce((worst, sport) =>
    sport.roi < worst.roi ? sport : worst
  );

  console.log(`\nBest Sport: ${bestSport.name} (${bestSport.roi.toFixed(2)}% ROI)`);
  console.log(`Worst Sport: ${worstSport.name} (${worstSport.roi.toFixed(2)}% ROI)\n`);

  // Example 6: Performance Attribution
  console.log('\nExample 6: Performance Attribution');
  console.log('=================================\n');

  const attribution = await syndicate.getPerformanceAttribution();

  console.log('Profit Attribution:');
  console.log(`  Selection: ${attribution.selection.toFixed(2)}% (${attribution.selectionProfit.toLocaleString()})`);
  console.log(`  Sizing: ${attribution.sizing.toFixed(2)}% (${attribution.sizingProfit.toLocaleString()})`);
  console.log(`  Timing: ${attribution.timing.toFixed(2)}% (${attribution.timingProfit.toLocaleString()})`);
  console.log(`  Other: ${attribution.other.toFixed(2)}% (${attribution.otherProfit.toLocaleString()})\n`);

  // Example 7: Generate Performance Report
  console.log('\nExample 7: Generate Performance Report');
  console.log('=====================================\n');

  const report = await syndicate.generatePerformanceReport({
    period: 'quarterly',
    format: 'detailed',
    includeCharts: true,
    includeMembers: true,
    includeSports: true
  });

  console.log('Performance Report Generated:');
  console.log(`  Period: ${report.period}`);
  console.log(`  Generated: ${report.generatedAt.toLocaleString()}`);
  console.log(`  Pages: ${report.pages}`);
  console.log(`  Sections: ${report.sections.join(', ')}\n`);

  // Export report
  console.log('Exporting report...');
  await report.exportPDF('/tmp/performance-report.pdf');
  console.log('✓ PDF exported to /tmp/performance-report.pdf');

  await report.exportCSV('/tmp/performance-data.csv');
  console.log('✓ CSV exported to /tmp/performance-data.csv');

  await report.exportJSON('/tmp/performance-data.json');
  console.log('✓ JSON exported to /tmp/performance-data.json\n');

  // Example 8: Live Performance Dashboard
  console.log('\nExample 8: Live Performance Dashboard');
  console.log('====================================\n');

  console.log('Starting live dashboard...');
  const dashboard = await syndicate.startLiveDashboard({
    refreshInterval: 60,  // 60 seconds
    metrics: [
      'capital',
      'roi',
      'winRate',
      'sharpeRatio',
      'activeBets',
      'recentBets'
    ]
  });

  console.log('✓ Dashboard started (refreshing every 60 seconds)');
  console.log('  Access at: http://localhost:3000/dashboard');
  console.log('  API endpoint: http://localhost:3000/api/performance\n');

  // Summary
  console.log('=== Performance Summary ===');
  console.log(`Total Profit: $${performance.netProfit.toLocaleString()}`);
  console.log(`Overall ROI: ${performance.roi.toFixed(2)}%`);
  console.log(`Win Rate: ${performance.winRate.toFixed(1)}%`);
  console.log(`Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
  console.log(`Best Performer: ${alice.name} (${alice.roi.toFixed(2)}% ROI)`);
  console.log(`Best Sport: ${bestSport.name} (${bestSport.roi.toFixed(2)}% ROI)`);

  console.log('\n=== Example Complete ===');
}

// Helper function to simulate betting history
async function simulateBettingHistory(syndicate) {
  const sports = ['NFL', 'NBA', 'MLB', 'NHL', 'Soccer'];
  const numBets = 100;

  for (let i = 0; i < numBets; i++) {
    const sport = sports[Math.floor(Math.random() * sports.length)];
    const odds = 1.8 + Math.random() * 1.5;  // 1.8 to 3.3
    const probability = 0.4 + Math.random() * 0.3;  // 40% to 70%
    const won = Math.random() < probability;

    await syndicate.recordBet({
      sport,
      odds,
      probability,
      won,
      stake: 1000 + Math.random() * 4000  // $1k to $5k
    });
  }
}

// Run example
if (require.main === module) {
  performanceTrackingExample()
    .then(() => {
      console.log('\n✓ Performance tracking example completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n✗ Example failed:', error);
      process.exit(1);
    });
}

module.exports = { performanceTrackingExample };
