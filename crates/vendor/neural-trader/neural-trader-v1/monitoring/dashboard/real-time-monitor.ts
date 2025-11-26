/**
 * Real-Time Monitoring Dashboard for Neural Trader Swarm
 * Tracks deployment ID: neural-trader-1763096012878
 */

import { EventEmitter } from 'events';
import blessed from 'blessed';
import contrib from 'blessed-contrib';

interface AgentMetrics {
  id: string;
  sandboxId: string;
  status: 'active' | 'idle' | 'error' | 'offline';
  cpu: number;
  memory: number;
  trades: {
    total: number;
    wins: number;
    losses: number;
    winRate: number;
  };
  performance: {
    sharpeRatio: number;
    totalReturn: number;
    maxDrawdown: number;
  };
  lastSync: Date;
  responseTime: number;
}

interface SwarmMetrics {
  deploymentId: string;
  topology: string;
  totalAgents: number;
  activeAgents: number;
  quicSyncInterval: number;
  uptime: number;
  totalTrades: number;
  aggregatePerformance: {
    portfolioValue: number;
    totalPnL: number;
    sharpeRatio: number;
    successRate: number;
  };
}

export class RealtimeMonitorDashboard extends EventEmitter {
  private screen: any;
  private grid: any;
  private widgets: {
    swarmStatus: any;
    agentTable: any;
    performanceChart: any;
    tradeLog: any;
    cpuGauge: any;
    memoryGauge: any;
    alertBox: any;
    metricsBar: any;
  };

  private agentMetrics: Map<string, AgentMetrics> = new Map();
  private swarmMetrics: SwarmMetrics;
  private performanceHistory: Array<{ time: string; value: number }> = [];
  private alerts: string[] = [];

  constructor(deploymentId: string = 'neural-trader-1763096012878') {
    super();

    this.swarmMetrics = {
      deploymentId,
      topology: 'mesh',
      totalAgents: 5,
      activeAgents: 0,
      quicSyncInterval: 5000,
      uptime: 0,
      totalTrades: 0,
      aggregatePerformance: {
        portfolioValue: 100000,
        totalPnL: 0,
        sharpeRatio: 0,
        successRate: 0
      }
    };

    this.initializeScreen();
    this.createWidgets();
    this.setupEventHandlers();
    this.startRefreshLoop();
  }

  private initializeScreen(): void {
    this.screen = blessed.screen({
      smartCSR: true,
      title: 'Neural Trader Swarm Monitor'
    });

    this.grid = new contrib.grid({
      rows: 12,
      cols: 12,
      screen: this.screen
    });

    // Exit handler
    this.screen.key(['escape', 'q', 'C-c'], () => {
      return process.exit(0);
    });
  }

  private createWidgets(): void {
    // Swarm Status Panel (top-left)
    this.widgets = {} as any;

    this.widgets.swarmStatus = this.grid.set(0, 0, 3, 6, blessed.box, {
      label: ' Swarm Status ',
      tags: true,
      border: { type: 'line' },
      style: {
        border: { fg: 'cyan' },
        label: { fg: 'cyan', bold: true }
      }
    });

    // Metrics Bar (top-right)
    this.widgets.metricsBar = this.grid.set(0, 6, 3, 6, contrib.bar, {
      label: ' Agent Performance ',
      barWidth: 8,
      barSpacing: 2,
      maxHeight: 10,
      style: {
        border: { fg: 'green' }
      }
    });

    // Agent Table (middle-left)
    this.widgets.agentTable = this.grid.set(3, 0, 5, 6, contrib.table, {
      keys: true,
      fg: 'white',
      selectedFg: 'white',
      selectedBg: 'blue',
      interactive: false,
      label: ' Agent Status ',
      width: '100%',
      height: '100%',
      border: { type: 'line', fg: 'cyan' },
      columnSpacing: 2,
      columnWidth: [15, 12, 8, 8, 10, 10]
    });

    // Performance Chart (middle-right)
    this.widgets.performanceChart = this.grid.set(3, 6, 5, 6, contrib.line, {
      style: {
        line: 'yellow',
        text: 'green',
        baseline: 'black'
      },
      xLabelPadding: 3,
      xPadding: 5,
      showLegend: true,
      wholeNumbersOnly: false,
      label: ' Portfolio Performance '
    });

    // CPU Gauge (bottom-left)
    this.widgets.cpuGauge = this.grid.set(8, 0, 2, 3, contrib.gauge, {
      label: ' Avg CPU Usage ',
      stroke: 'green',
      fill: 'white'
    });

    // Memory Gauge (bottom-middle-left)
    this.widgets.memoryGauge = this.grid.set(8, 3, 2, 3, contrib.gauge, {
      label: ' Avg Memory Usage ',
      stroke: 'blue',
      fill: 'white'
    });

    // Trade Log (bottom-right)
    this.widgets.tradeLog = this.grid.set(8, 6, 4, 6, contrib.log, {
      fg: 'green',
      selectedFg: 'green',
      label: ' Trade Execution Log ',
      border: { type: 'line', fg: 'yellow' }
    });

    // Alert Box (bottom)
    this.widgets.alertBox = this.grid.set(10, 0, 2, 6, blessed.box, {
      label: ' Alerts ',
      tags: true,
      border: { type: 'line' },
      style: {
        border: { fg: 'red' },
        label: { fg: 'red', bold: true }
      },
      scrollable: true,
      alwaysScroll: true,
      scrollbar: {
        ch: ' ',
        inverse: true
      }
    });
  }

  private setupEventHandlers(): void {
    this.on('agent-update', (metrics: AgentMetrics) => {
      this.agentMetrics.set(metrics.id, metrics);
      this.updateDisplay();
    });

    this.on('swarm-update', (metrics: Partial<SwarmMetrics>) => {
      this.swarmMetrics = { ...this.swarmMetrics, ...metrics };
      this.updateDisplay();
    });

    this.on('trade-executed', (trade: any) => {
      this.logTrade(trade);
    });

    this.on('alert', (message: string) => {
      this.addAlert(message);
    });
  }

  private updateDisplay(): void {
    this.updateSwarmStatus();
    this.updateAgentTable();
    this.updatePerformanceChart();
    this.updateGauges();
    this.updateMetricsBar();
    this.screen.render();
  }

  private updateSwarmStatus(): void {
    const status = `
{cyan-fg}Deployment ID:{/cyan-fg} ${this.swarmMetrics.deploymentId}
{cyan-fg}Topology:{/cyan-fg} ${this.swarmMetrics.topology.toUpperCase()}
{cyan-fg}Total Agents:{/cyan-fg} ${this.swarmMetrics.totalAgents}
{green-fg}Active Agents:{/green-fg} ${this.swarmMetrics.activeAgents}
{cyan-fg}QUIC Sync:{/cyan-fg} ${this.swarmMetrics.quicSyncInterval}ms
{cyan-fg}Uptime:{/cyan-fg} ${this.formatUptime(this.swarmMetrics.uptime)}

{yellow-fg}Portfolio Metrics:{/yellow-fg}
  Value: {green-fg}$${this.swarmMetrics.aggregatePerformance.portfolioValue.toLocaleString()}{/green-fg}
  P&L: ${this.formatPnL(this.swarmMetrics.aggregatePerformance.totalPnL)}
  Sharpe: {cyan-fg}${this.swarmMetrics.aggregatePerformance.sharpeRatio.toFixed(2)}{/cyan-fg}
  Success Rate: {green-fg}${(this.swarmMetrics.aggregatePerformance.successRate * 100).toFixed(1)}%{/green-fg}
`;

    this.widgets.swarmStatus.setContent(status);
  }

  private updateAgentTable(): void {
    const headers = ['Agent ID', 'Status', 'CPU %', 'Mem %', 'Win Rate', 'Sharpe'];
    const data = Array.from(this.agentMetrics.values()).map(agent => [
      agent.id.substring(0, 12),
      this.formatStatus(agent.status),
      agent.cpu.toFixed(1),
      agent.memory.toFixed(1),
      `${(agent.trades.winRate * 100).toFixed(1)}%`,
      agent.performance.sharpeRatio.toFixed(2)
    ]);

    this.widgets.agentTable.setData({
      headers,
      data
    });
  }

  private updatePerformanceChart(): void {
    const series = [{
      title: 'Portfolio Value',
      x: this.performanceHistory.map(p => p.time),
      y: this.performanceHistory.map(p => p.value),
      style: { line: 'yellow' }
    }];

    this.widgets.performanceChart.setData(series);
  }

  private updateGauges(): void {
    const agents = Array.from(this.agentMetrics.values());

    if (agents.length > 0) {
      const avgCpu = agents.reduce((sum, a) => sum + a.cpu, 0) / agents.length;
      const avgMemory = agents.reduce((sum, a) => sum + a.memory, 0) / agents.length;

      this.widgets.cpuGauge.setPercent(Math.round(avgCpu));
      this.widgets.memoryGauge.setPercent(Math.round(avgMemory));
    }
  }

  private updateMetricsBar(): void {
    const agents = Array.from(this.agentMetrics.values());

    if (agents.length > 0) {
      const barData = {
        titles: agents.map(a => a.id.substring(0, 8)),
        data: agents.map(a => Math.round(a.trades.winRate * 100))
      };

      this.widgets.metricsBar.setData(barData);
    }
  }

  private logTrade(trade: any): void {
    const timestamp = new Date().toISOString().substring(11, 19);
    const log = `[${timestamp}] ${trade.agent}: ${trade.action} ${trade.symbol} @ $${trade.price}`;
    this.widgets.tradeLog.log(log);
  }

  private addAlert(message: string): void {
    const timestamp = new Date().toISOString().substring(11, 19);
    const alert = `{red-fg}[${timestamp}]{/red-fg} ${message}`;
    this.alerts.unshift(alert);
    this.alerts = this.alerts.slice(0, 10); // Keep last 10 alerts

    this.widgets.alertBox.setContent(this.alerts.join('\n'));
  }

  private formatStatus(status: string): string {
    const colors: Record<string, string> = {
      active: '{green-fg}ACTIVE{/green-fg}',
      idle: '{yellow-fg}IDLE{/yellow-fg}',
      error: '{red-fg}ERROR{/red-fg}',
      offline: '{gray-fg}OFFLINE{/gray-fg}'
    };
    return colors[status] || status;
  }

  private formatPnL(pnl: number): string {
    const color = pnl >= 0 ? 'green-fg' : 'red-fg';
    const sign = pnl >= 0 ? '+' : '';
    return `{${color}}${sign}$${pnl.toLocaleString()}{/${color}}`;
  }

  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  }

  private startRefreshLoop(): void {
    setInterval(() => {
      this.swarmMetrics.uptime++;
      this.updateDisplay();
    }, 1000);
  }

  public render(): void {
    this.screen.render();
  }

  public updateAgentMetrics(metrics: AgentMetrics): void {
    this.emit('agent-update', metrics);
  }

  public updateSwarmMetrics(metrics: Partial<SwarmMetrics>): void {
    this.emit('swarm-update', metrics);
  }

  public logTradeExecution(trade: any): void {
    this.emit('trade-executed', trade);
  }

  public raiseAlert(message: string): void {
    this.emit('alert', message);
  }

  public addPerformancePoint(value: number): void {
    const time = new Date().toISOString().substring(11, 19);
    this.performanceHistory.push({ time, value });

    // Keep last 50 points
    if (this.performanceHistory.length > 50) {
      this.performanceHistory.shift();
    }
  }
}

// CLI Entry Point
if (require.main === module) {
  const dashboard = new RealtimeMonitorDashboard();

  // Simulate data updates for demo
  setInterval(() => {
    for (let i = 1; i <= 5; i++) {
      dashboard.updateAgentMetrics({
        id: `agent-${i}`,
        sandboxId: `sandbox-${i}`,
        status: Math.random() > 0.1 ? 'active' : 'idle',
        cpu: Math.random() * 80 + 10,
        memory: Math.random() * 70 + 20,
        trades: {
          total: Math.floor(Math.random() * 100),
          wins: Math.floor(Math.random() * 60),
          losses: Math.floor(Math.random() * 40),
          winRate: Math.random() * 0.4 + 0.5
        },
        performance: {
          sharpeRatio: Math.random() * 2 + 0.5,
          totalReturn: Math.random() * 20 - 5,
          maxDrawdown: Math.random() * -10
        },
        lastSync: new Date(),
        responseTime: Math.random() * 100 + 20
      });
    }

    dashboard.addPerformancePoint(100000 + Math.random() * 10000 - 5000);
  }, 2000);

  dashboard.render();
}

export default RealtimeMonitorDashboard;
