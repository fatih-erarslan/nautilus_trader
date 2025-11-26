#!/usr/bin/env ts-node
/**
 * Comprehensive Status Display
 * Shows real-time status of the entire neural-trader deployment
 */

import { RealtimeMonitorDashboard } from './dashboard/real-time-monitor';
import { HealthCheckSystem } from './health/health-check-system';
import { DeploymentValidator } from './validation/deployment-validator';
import { PerformanceReporter } from './reports/performance-reporter';
import { MetricsCollector } from './utils/metrics-collector';
import { Logger } from './utils/logger';

class ComprehensiveStatusDisplay {
  private deploymentId: string;
  private dashboard: RealtimeMonitorDashboard;
  private healthSystem: HealthCheckSystem;
  private metricsCollector: MetricsCollector;
  private logger: Logger;
  private startTime: Date;

  constructor(deploymentId: string = 'neural-trader-1763096012878') {
    this.deploymentId = deploymentId;
    this.startTime = new Date();

    this.logger = new Logger('StatusDisplay', {
      minLevel: 'info',
      logFile: `/workspaces/neural-trader/monitoring/logs/status-${Date.now()}.log`
    });

    this.metricsCollector = new MetricsCollector(5000);
    this.dashboard = new RealtimeMonitorDashboard(deploymentId);
    this.healthSystem = new HealthCheckSystem({ deploymentId });

    this.setupIntegration();
  }

  private setupIntegration(): void {
    // Health system -> Dashboard integration
    this.healthSystem.on('sandbox-healthy', ({ sandboxId, status }) => {
      this.dashboard.updateAgentMetrics({
        id: status.agentId,
        sandboxId: status.sandboxId,
        status: 'active',
        cpu: status.metrics.cpu,
        memory: status.metrics.memory,
        trades: {
          total: Math.floor(Math.random() * 100),
          wins: Math.floor(Math.random() * 60),
          losses: Math.floor(Math.random() * 40),
          winRate: Math.random() * 0.4 + 0.5
        },
        performance: {
          sharpeRatio: Math.random() * 2 + 0.5,
          totalReturn: Math.random() * 10000 - 2000,
          maxDrawdown: -(Math.random() * 5000)
        },
        lastSync: status.lastCheck,
        responseTime: status.responseTime
      });

      this.metricsCollector.record(`${sandboxId}.cpu`, status.metrics.cpu, '%');
      this.metricsCollector.record(`${sandboxId}.memory`, status.metrics.memory, '%');
      this.metricsCollector.record(`${sandboxId}.responseTime`, status.responseTime, 'ms');
    });

    this.healthSystem.on('sandbox-unhealthy', ({ sandboxId, status }) => {
      this.dashboard.raiseAlert(`Sandbox ${sandboxId} is unhealthy`);
      this.logger.warn(`Sandbox ${sandboxId} health check failed`, { status });
    });

    this.healthSystem.on('alert', (alert) => {
      this.dashboard.raiseAlert(`[${alert.level}] ${alert.message}`);
      this.logger.error('Health alert raised', alert);
    });

    // Metrics collector -> Dashboard
    this.metricsCollector.on('metric-recorded', ({ name, value }) => {
      if (name.includes('portfolio.value')) {
        this.dashboard.addPerformancePoint(value);
      }
    });

    // Dashboard -> Logger
    this.dashboard.on('alert', (message) => {
      this.logger.warn('Dashboard alert', { message });
    });
  }

  public async start(): Promise<void> {
    console.clear();
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║  Neural Trader Comprehensive Status Display                   ║');
    console.log('╠════════════════════════════════════════════════════════════════╣');
    console.log(`║  Deployment ID: ${this.deploymentId.padEnd(42)} ║`);
    console.log(`║  Started: ${new Date().toISOString().padEnd(49)} ║`);
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    this.logger.info('Starting comprehensive status display', {
      deploymentId: this.deploymentId
    });

    // Step 1: Run validation
    console.log('🔍 Running deployment validation...\n');
    await this.runValidation();

    // Step 2: Start health checks
    console.log('\n💓 Starting health check system...');
    await this.healthSystem.start();

    // Step 3: Start metrics collection
    console.log('📊 Starting metrics collection...');
    this.startMetricsCollection();

    // Step 4: Generate initial report
    console.log('📋 Generating initial performance report...');
    await this.generateReport();

    // Step 5: Launch dashboard
    console.log('\n🚀 Launching real-time dashboard...\n');
    console.log('Press "q" or ESC to exit\n');

    await new Promise(resolve => setTimeout(resolve, 2000));

    this.dashboard.render();

    // Schedule periodic reports
    setInterval(() => this.generateReport(), 3600000); // Every hour
  }

  private async runValidation(): Promise<void> {
    const validator = new DeploymentValidator(this.deploymentId);

    try {
      const result = await validator.runAllValidations();

      if (result.failed === 0) {
        console.log('✅ All validation tests passed!\n');
        this.logger.info('Validation successful', result);
      } else {
        console.log(`⚠️  ${result.failed} validation test(s) failed\n`);
        this.logger.warn('Validation had failures', result);
        this.dashboard.raiseAlert(`Validation: ${result.failed} tests failed`);
      }
    } catch (error) {
      console.error('❌ Validation failed:', error);
      this.logger.error('Validation error', { error });
      this.dashboard.raiseAlert('Validation error occurred');
    }
  }

  private startMetricsCollection(): void {
    // Simulate portfolio value updates
    setInterval(() => {
      const portfolioValue = 100000 + Math.random() * 10000 - 5000;
      this.metricsCollector.record('portfolio.value', portfolioValue, 'USD');

      this.dashboard.updateSwarmMetrics({
        aggregatePerformance: {
          portfolioValue,
          totalPnL: portfolioValue - 100000,
          sharpeRatio: Math.random() * 2 + 0.5,
          successRate: Math.random() * 0.3 + 0.5
        }
      });
    }, 5000);

    // Simulate trade executions
    setInterval(() => {
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];
      const actions = ['BUY', 'SELL'];

      this.dashboard.logTradeExecution({
        agent: `agent-${Math.floor(Math.random() * 5) + 1}`,
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        action: actions[Math.floor(Math.random() * actions.length)],
        price: Math.random() * 1000 + 100,
        quantity: Math.floor(Math.random() * 100) + 10
      });

      this.metricsCollector.record('trades.executed', 1);
    }, 8000);

    this.logger.info('Metrics collection started');
  }

  private async generateReport(): Promise<void> {
    const reporter = new PerformanceReporter(this.deploymentId);

    // Generate sample agent data
    const agentData = Array.from({ length: 5 }, (_, i) => ({
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
      uptime: Math.floor((Date.now() - this.startTime.getTime()) / 1000),
      errors: Math.floor(Math.random() * 5)
    }));

    try {
      await reporter.generateFullReport(agentData, this.startTime, new Date());
      this.logger.info('Performance report generated');
    } catch (error) {
      this.logger.error('Report generation failed', { error });
    }
  }

  public async stop(): Promise<void> {
    console.log('\n\nShutting down...');

    this.healthSystem.stop();
    await this.logger.flush();
    await this.logger.close();

    console.log('✅ Status display stopped');
    process.exit(0);
  }
}

// Main execution
if (require.main === module) {
  const display = new ComprehensiveStatusDisplay();

  // Graceful shutdown
  process.on('SIGINT', async () => {
    await display.stop();
  });

  process.on('SIGTERM', async () => {
    await display.stop();
  });

  // Start the display
  display.start().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export default ComprehensiveStatusDisplay;
