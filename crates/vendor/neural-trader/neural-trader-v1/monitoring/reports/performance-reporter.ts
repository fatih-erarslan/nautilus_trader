/**
 * Performance Report Generator
 * Aggregates metrics and generates comprehensive performance reports
 */

import * as fs from 'fs/promises';
import * as path from 'path';

interface TradeStatistics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  largestWin: number;
  largestLoss: number;
}

interface PortfolioMetrics {
  initialValue: number;
  currentValue: number;
  totalReturn: number;
  totalReturnPercentage: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  maxDrawdownPercentage: number;
  calmarRatio: number;
  volatility: number;
}

interface ResourceUtilization {
  averageCpu: number;
  peakCpu: number;
  averageMemory: number;
  peakMemory: number;
  diskUsage: number;
  networkBandwidth: number;
  costEstimate: number;
}

interface CoordinationMetrics {
  totalMessages: number;
  averageLatency: number;
  peakLatency: number;
  quicSyncSuccess: number;
  quicSyncFailures: number;
  consensusEvents: number;
  averageConsensusTime: number;
  networkEfficiency: number;
}

interface AgentPerformance {
  agentId: string;
  sandboxId: string;
  trades: TradeStatistics;
  performance: {
    return: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
  resources: {
    avgCpu: number;
    avgMemory: number;
  };
  uptime: number;
  errors: number;
}

export class PerformanceReporter {
  private deploymentId: string;
  private reportDir: string;

  constructor(
    deploymentId: string = 'neural-trader-1763096012878',
    reportDir: string = '/workspaces/neural-trader/monitoring/reports/output'
  ) {
    this.deploymentId = deploymentId;
    this.reportDir = reportDir;
  }

  public async generateFullReport(
    agentData: AgentPerformance[],
    startTime: Date,
    endTime: Date
  ): Promise<string> {
    await fs.mkdir(this.reportDir, { recursive: true });

    const report = {
      metadata: {
        deploymentId: this.deploymentId,
        generatedAt: new Date(),
        periodStart: startTime,
        periodEnd: endTime,
        duration: this.formatDuration(endTime.getTime() - startTime.getTime())
      },
      summary: this.generateSummary(agentData),
      tradeStatistics: this.aggregateTradeStatistics(agentData),
      portfolioMetrics: this.calculatePortfolioMetrics(agentData),
      resourceUtilization: this.analyzeResourceUtilization(agentData),
      coordinationMetrics: this.getCoordinationMetrics(),
      agentPerformance: this.rankAgentPerformance(agentData),
      recommendations: this.generateRecommendations(agentData)
    };

    // Generate multiple formats
    await this.saveJsonReport(report);
    await this.saveHtmlReport(report);
    await this.saveMarkdownReport(report);
    await this.saveCsvReport(agentData);

    console.log(`\n📊 Performance report generated: ${this.reportDir}`);

    return path.join(this.reportDir, 'report.json');
  }

  private generateSummary(agentData: AgentPerformance[]): any {
    const totalTrades = agentData.reduce((sum, a) => sum + a.trades.totalTrades, 0);
    const totalWins = agentData.reduce((sum, a) => sum + a.trades.winningTrades, 0);
    const avgReturn = agentData.reduce((sum, a) => sum + a.performance.return, 0) / agentData.length;

    return {
      totalAgents: agentData.length,
      activeAgents: agentData.filter(a => a.uptime > 0).length,
      totalTrades,
      overallWinRate: totalWins / totalTrades,
      averageReturn: avgReturn,
      bestPerformer: this.findBestPerformer(agentData),
      worstPerformer: this.findWorstPerformer(agentData)
    };
  }

  private aggregateTradeStatistics(agentData: AgentPerformance[]): TradeStatistics {
    const totalTrades = agentData.reduce((sum, a) => sum + a.trades.totalTrades, 0);
    const winningTrades = agentData.reduce((sum, a) => sum + a.trades.winningTrades, 0);
    const losingTrades = agentData.reduce((sum, a) => sum + a.trades.losingTrades, 0);

    const totalWinAmount = agentData.reduce((sum, a) =>
      sum + (a.trades.averageWin * a.trades.winningTrades), 0);
    const totalLossAmount = agentData.reduce((sum, a) =>
      sum + (a.trades.averageLoss * a.trades.losingTrades), 0);

    return {
      totalTrades,
      winningTrades,
      losingTrades,
      winRate: winningTrades / totalTrades,
      averageWin: totalWinAmount / winningTrades,
      averageLoss: Math.abs(totalLossAmount / losingTrades),
      profitFactor: Math.abs(totalWinAmount / totalLossAmount),
      largestWin: Math.max(...agentData.map(a => a.trades.largestWin)),
      largestLoss: Math.min(...agentData.map(a => a.trades.largestLoss))
    };
  }

  private calculatePortfolioMetrics(agentData: AgentPerformance[]): PortfolioMetrics {
    const initialValue = 100000; // Starting portfolio value
    const returns = agentData.map(a => a.performance.return);
    const totalReturn = returns.reduce((sum, r) => sum + r, 0);
    const currentValue = initialValue + totalReturn;

    const avgReturn = totalReturn / agentData.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);

    const riskFreeRate = 0.02; // 2% annual risk-free rate
    const sharpeRatio = (avgReturn - riskFreeRate) / volatility;

    const negativeReturns = returns.filter(r => r < 0);
    const downvideDeviation = Math.sqrt(
      negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length
    );
    const sortinoRatio = (avgReturn - riskFreeRate) / downvideDeviation;

    const maxDrawdown = Math.min(...agentData.map(a => a.performance.maxDrawdown));
    const calmarRatio = avgReturn / Math.abs(maxDrawdown);

    return {
      initialValue,
      currentValue,
      totalReturn,
      totalReturnPercentage: (totalReturn / initialValue) * 100,
      sharpeRatio,
      sortinoRatio,
      maxDrawdown,
      maxDrawdownPercentage: (maxDrawdown / initialValue) * 100,
      calmarRatio,
      volatility
    };
  }

  private analyzeResourceUtilization(agentData: AgentPerformance[]): ResourceUtilization {
    const cpuValues = agentData.map(a => a.resources.avgCpu);
    const memoryValues = agentData.map(a => a.resources.avgMemory);

    // Estimate costs (mock values)
    const hourlyRate = 0.05; // $0.05 per agent per hour
    const totalHours = agentData.reduce((sum, a) => sum + (a.uptime / 3600), 0);

    return {
      averageCpu: cpuValues.reduce((sum, v) => sum + v, 0) / cpuValues.length,
      peakCpu: Math.max(...cpuValues),
      averageMemory: memoryValues.reduce((sum, v) => sum + v, 0) / memoryValues.length,
      peakMemory: Math.max(...memoryValues),
      diskUsage: 2.5, // GB
      networkBandwidth: 150, // MB
      costEstimate: totalHours * hourlyRate
    };
  }

  private getCoordinationMetrics(): CoordinationMetrics {
    // Mock coordination metrics - replace with actual data
    return {
      totalMessages: 15420,
      averageLatency: 45, // ms
      peakLatency: 230,
      quicSyncSuccess: 2840,
      quicSyncFailures: 3,
      consensusEvents: 156,
      averageConsensusTime: 150, // ms
      networkEfficiency: 0.978 // 97.8%
    };
  }

  private rankAgentPerformance(agentData: AgentPerformance[]): AgentPerformance[] {
    return [...agentData].sort((a, b) => b.performance.sharpeRatio - a.performance.sharpeRatio);
  }

  private findBestPerformer(agentData: AgentPerformance[]): string {
    const best = agentData.reduce((best, current) =>
      current.performance.sharpeRatio > best.performance.sharpeRatio ? current : best
    );
    return best.agentId;
  }

  private findWorstPerformer(agentData: AgentPerformance[]): string {
    const worst = agentData.reduce((worst, current) =>
      current.performance.sharpeRatio < worst.performance.sharpeRatio ? current : worst
    );
    return worst.agentId;
  }

  private generateRecommendations(agentData: AgentPerformance[]): string[] {
    const recommendations: string[] = [];

    const portfolio = this.calculatePortfolioMetrics(agentData);
    const trades = this.aggregateTradeStatistics(agentData);
    const resources = this.analyzeResourceUtilization(agentData);

    // Performance recommendations
    if (portfolio.sharpeRatio < 1.0) {
      recommendations.push('⚠️  Sharpe ratio below 1.0 - consider adjusting risk parameters');
    }

    if (trades.winRate < 0.5) {
      recommendations.push('⚠️  Win rate below 50% - review trading strategies');
    }

    if (portfolio.maxDrawdownPercentage < -20) {
      recommendations.push('⚠️  Maximum drawdown exceeds 20% - implement stricter risk controls');
    }

    // Resource recommendations
    if (resources.averageCpu > 70) {
      recommendations.push('💡 High CPU usage detected - consider optimizing algorithms');
    }

    if (resources.averageMemory > 70) {
      recommendations.push('💡 High memory usage - review memory management');
    }

    // Agent recommendations
    const underperformers = agentData.filter(a => a.performance.sharpeRatio < 0);
    if (underperformers.length > 0) {
      recommendations.push(`⚠️  ${underperformers.length} agents with negative Sharpe ratio - investigate and retrain`);
    }

    // Coordination recommendations
    const coordination = this.getCoordinationMetrics();
    if (coordination.quicSyncFailures > 10) {
      recommendations.push('⚠️  High QUIC sync failure rate - check network connectivity');
    }

    if (recommendations.length === 0) {
      recommendations.push('✅ All systems operating within optimal parameters');
    }

    return recommendations;
  }

  private async saveJsonReport(report: any): Promise<void> {
    const filepath = path.join(this.reportDir, 'report.json');
    await fs.writeFile(filepath, JSON.stringify(report, null, 2));
    console.log(`  ✅ JSON report: ${filepath}`);
  }

  private async saveHtmlReport(report: any): Promise<void> {
    const html = this.generateHtml(report);
    const filepath = path.join(this.reportDir, 'report.html');
    await fs.writeFile(filepath, html);
    console.log(`  ✅ HTML report: ${filepath}`);
  }

  private async saveMarkdownReport(report: any): Promise<void> {
    const md = this.generateMarkdown(report);
    const filepath = path.join(this.reportDir, 'report.md');
    await fs.writeFile(filepath, md);
    console.log(`  ✅ Markdown report: ${filepath}`);
  }

  private async saveCsvReport(agentData: AgentPerformance[]): Promise<void> {
    const headers = 'Agent ID,Sandbox ID,Total Trades,Win Rate,Return,Sharpe Ratio,Max Drawdown,Avg CPU,Avg Memory,Uptime\n';
    const rows = agentData.map(a =>
      `${a.agentId},${a.sandboxId},${a.trades.totalTrades},${(a.trades.winRate * 100).toFixed(2)}%,${a.performance.return.toFixed(2)},${a.performance.sharpeRatio.toFixed(2)},${a.performance.maxDrawdown.toFixed(2)},${a.resources.avgCpu.toFixed(1)}%,${a.resources.avgMemory.toFixed(1)}%,${a.uptime}s`
    ).join('\n');

    const filepath = path.join(this.reportDir, 'agents.csv');
    await fs.writeFile(filepath, headers + rows);
    console.log(`  ✅ CSV report: ${filepath}`);
  }

  private generateHtml(report: any): string {
    return `<!DOCTYPE html>
<html>
<head>
  <title>Neural Trader Performance Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    .metric { display: inline-block; margin: 10px 20px 10px 0; }
    .metric-label { font-weight: bold; color: #7f8c8d; }
    .metric-value { font-size: 24px; color: #2c3e50; }
    .positive { color: #27ae60; }
    .negative { color: #e74c3c; }
    table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
    th { background: #3498db; color: white; }
    tr:hover { background: #ecf0f1; }
    .recommendation { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Neural Trader Performance Report</h1>
    <p><strong>Deployment ID:</strong> ${report.metadata.deploymentId}</p>
    <p><strong>Period:</strong> ${new Date(report.metadata.periodStart).toLocaleString()} - ${new Date(report.metadata.periodEnd).toLocaleString()}</p>

    <h2>Portfolio Summary</h2>
    <div class="metric">
      <div class="metric-label">Total Return</div>
      <div class="metric-value ${report.portfolioMetrics.totalReturn >= 0 ? 'positive' : 'negative'}">
        ${report.portfolioMetrics.totalReturn >= 0 ? '+' : ''}$${report.portfolioMetrics.totalReturn.toLocaleString()}
      </div>
    </div>
    <div class="metric">
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-value">${report.portfolioMetrics.sharpeRatio.toFixed(2)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value">${(report.tradeStatistics.winRate * 100).toFixed(1)}%</div>
    </div>

    <h2>Recommendations</h2>
    ${report.recommendations.map((rec: string) => `<div class="recommendation">${rec}</div>`).join('')}

    <h2>Agent Performance</h2>
    <table>
      <tr>
        <th>Agent</th>
        <th>Trades</th>
        <th>Win Rate</th>
        <th>Return</th>
        <th>Sharpe</th>
      </tr>
      ${report.agentPerformance.map((agent: AgentPerformance) => `
        <tr>
          <td>${agent.agentId}</td>
          <td>${agent.trades.totalTrades}</td>
          <td>${(agent.trades.winRate * 100).toFixed(1)}%</td>
          <td class="${agent.performance.return >= 0 ? 'positive' : 'negative'}">
            $${agent.performance.return.toFixed(2)}
          </td>
          <td>${agent.performance.sharpeRatio.toFixed(2)}</td>
        </tr>
      `).join('')}
    </table>
  </div>
</body>
</html>`;
  }

  private generateMarkdown(report: any): string {
    return `# Neural Trader Performance Report

**Deployment ID:** ${report.metadata.deploymentId}
**Generated:** ${new Date(report.metadata.generatedAt).toLocaleString()}
**Period:** ${new Date(report.metadata.periodStart).toLocaleString()} - ${new Date(report.metadata.periodEnd).toLocaleString()}

## Executive Summary

- **Total Agents:** ${report.summary.totalAgents}
- **Total Trades:** ${report.summary.totalTrades}
- **Overall Win Rate:** ${(report.summary.overallWinRate * 100).toFixed(1)}%
- **Average Return:** $${report.summary.averageReturn.toFixed(2)}
- **Best Performer:** ${report.summary.bestPerformer}

## Portfolio Metrics

| Metric | Value |
|--------|-------|
| Initial Value | $${report.portfolioMetrics.initialValue.toLocaleString()} |
| Current Value | $${report.portfolioMetrics.currentValue.toLocaleString()} |
| Total Return | $${report.portfolioMetrics.totalReturn.toFixed(2)} (${report.portfolioMetrics.totalReturnPercentage.toFixed(2)}%) |
| Sharpe Ratio | ${report.portfolioMetrics.sharpeRatio.toFixed(2)} |
| Max Drawdown | ${report.portfolioMetrics.maxDrawdownPercentage.toFixed(2)}% |
| Volatility | ${report.portfolioMetrics.volatility.toFixed(2)} |

## Trade Statistics

- **Total Trades:** ${report.tradeStatistics.totalTrades}
- **Winning Trades:** ${report.tradeStatistics.winningTrades}
- **Losing Trades:** ${report.tradeStatistics.losingTrades}
- **Win Rate:** ${(report.tradeStatistics.winRate * 100).toFixed(1)}%
- **Profit Factor:** ${report.tradeStatistics.profitFactor.toFixed(2)}

## Resource Utilization

- **Average CPU:** ${report.resourceUtilization.averageCpu.toFixed(1)}%
- **Average Memory:** ${report.resourceUtilization.averageMemory.toFixed(1)}%
- **Estimated Cost:** $${report.resourceUtilization.costEstimate.toFixed(2)}

## Recommendations

${report.recommendations.map((rec: string) => `- ${rec}`).join('\n')}
`;
  }

  private formatDuration(ms: number): string {
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    return `${hours}h ${minutes}m`;
  }
}

// CLI Entry Point
if (require.main === module) {
  // Generate sample report
  const reporter = new PerformanceReporter();

  const sampleData: AgentPerformance[] = Array.from({ length: 5 }, (_, i) => ({
    agentId: `agent-${i + 1}`,
    sandboxId: `sandbox-${i + 1}`,
    trades: {
      totalTrades: Math.floor(Math.random() * 100) + 50,
      winningTrades: Math.floor(Math.random() * 60) + 30,
      losingTrades: Math.floor(Math.random() * 40) + 20,
      winRate: Math.random() * 0.3 + 0.5,
      averageWin: Math.random() * 500 + 200,
      averageLoss: -(Math.random() * 300 + 100),
      profitFactor: Math.random() * 2 + 1,
      largestWin: Math.random() * 2000 + 500,
      largestLoss: -(Math.random() * 1500 + 400)
    },
    performance: {
      return: Math.random() * 10000 - 2000,
      sharpeRatio: Math.random() * 2 + 0.5,
      maxDrawdown: -(Math.random() * 5000 + 1000)
    },
    resources: {
      avgCpu: Math.random() * 60 + 20,
      avgMemory: Math.random() * 50 + 30
    },
    uptime: Math.floor(Math.random() * 86400) + 3600,
    errors: Math.floor(Math.random() * 5)
  }));

  const startTime = new Date(Date.now() - 86400000); // 24 hours ago
  const endTime = new Date();

  reporter.generateFullReport(sampleData, startTime, endTime)
    .then(filepath => {
      console.log(`\n✅ Report generation complete!`);
      console.log(`📁 Reports available at: ${filepath}`);
    })
    .catch(error => {
      console.error('Report generation failed:', error);
      process.exit(1);
    });
}

export default PerformanceReporter;
