/**
 * Neural Trader Monitoring Infrastructure
 * Entry point for all monitoring components
 */

export { RealtimeMonitorDashboard } from './dashboard/real-time-monitor';
export { HealthCheckSystem } from './health/health-check-system';
export { DeploymentValidator } from './validation/deployment-validator';
export { PerformanceReporter } from './reports/performance-reporter';

// Re-export types
export type {
  AgentMetrics,
  SwarmMetrics
} from './dashboard/real-time-monitor';

export type {
  HealthCheckConfig,
  HealthStatus
} from './health/health-check-system';

export type {
  ValidationResult
} from './validation/deployment-validator';

export type {
  TradeStatistics,
  PortfolioMetrics,
  ResourceUtilization,
  CoordinationMetrics,
  AgentPerformance
} from './reports/performance-reporter';

/**
 * Quick start example:
 *
 * ```typescript
 * import { RealtimeMonitorDashboard, HealthCheckSystem } from '@neural-trader/monitoring';
 *
 * const dashboard = new RealtimeMonitorDashboard('neural-trader-1763096012878');
 * const healthSystem = new HealthCheckSystem();
 *
 * healthSystem.on('sandbox-unhealthy', ({ sandboxId, status }) => {
 *   dashboard.raiseAlert(`Sandbox ${sandboxId} is unhealthy`);
 * });
 *
 * await healthSystem.start();
 * dashboard.render();
 * ```
 */
