# Neural Trader Monitoring Infrastructure

Comprehensive monitoring, health checking, validation, and performance reporting for the Neural Trader swarm deployment.

## 📊 Components

### 1. Real-Time Dashboard (`dashboard/real-time-monitor.ts`)
Interactive terminal-based monitoring dashboard with live metrics.

**Features:**
- Real-time swarm status display
- Individual agent health tracking
- Performance metrics (Win Rate, Sharpe Ratio, P&L)
- Resource utilization monitoring (CPU, Memory)
- Trade execution logs
- Alert notifications
- Beautiful blessed-contrib visualizations

**Usage:**
```bash
npm run dashboard
```

**Display Sections:**
- Swarm Status Panel: Deployment info, topology, uptime
- Metrics Bar: Agent performance comparison
- Agent Table: Detailed agent status
- Performance Chart: Portfolio value over time
- CPU/Memory Gauges: Resource utilization
- Trade Log: Real-time trade execution
- Alert Box: System alerts and warnings

### 2. Health Check System (`health/health-check-system.ts`)
Automated health monitoring with configurable intervals and alerting.

**Features:**
- Sandbox responsiveness checks (every 60 seconds)
- Agent process monitoring
- QUIC sync verification (5-second intervals)
- Trading API connectivity tests
- Resource threshold monitoring
- Consecutive failure tracking
- Automatic alert generation

**Usage:**
```bash
npm run health-check
```

**Health Checks:**
1. Sandbox ping/response
2. Agent process status
3. QUIC synchronization activity
4. External API connectivity
5. CPU/Memory/Disk resources

**Alert Thresholds:**
- CPU: 90%
- Memory: 85%
- Consecutive Failures: 3
- Response Time: 5000ms

### 3. Deployment Validator (`validation/deployment-validator.ts`)
Comprehensive validation test suite for deployment verification.

**Features:**
- Sandbox availability verification
- Inter-agent communication tests
- Distributed memory sync validation
- Trading API connectivity checks
- QUIC sync interval validation
- Resource limit compliance
- Failover mechanism testing

**Usage:**
```bash
npm run validate
# or
npm test
```

**Test Categories:**
1. **Sandbox Tests**: Verify all 5 sandboxes running
2. **Communication Tests**: Mesh topology validation, message passing
3. **Memory Tests**: Distributed sync, QUIC propagation
4. **API Tests**: Trading API connectivity per agent
5. **QUIC Tests**: Sync interval (5s), recent sync activity
6. **Resource Tests**: CPU/Memory/Disk within limits
7. **Failover Tests**: Agent failure detection and recovery

### 4. Performance Reporter (`reports/performance-reporter.ts`)
Generates comprehensive performance reports in multiple formats.

**Features:**
- Trade statistics aggregation
- Portfolio metrics calculation
- Resource utilization analysis
- Coordination efficiency metrics
- Agent performance ranking
- Automated recommendations

**Usage:**
```bash
npm run report
```

**Report Formats:**
- **JSON**: Machine-readable data (`report.json`)
- **HTML**: Interactive web report (`report.html`)
- **Markdown**: Documentation format (`report.md`)
- **CSV**: Spreadsheet data (`agents.csv`)

**Metrics Included:**
- Trade Statistics: Win rate, profit factor, avg win/loss
- Portfolio Metrics: Sharpe ratio, max drawdown, total return
- Resource Usage: CPU, memory, disk, network
- Coordination: QUIC sync, latency, consensus time
- Agent Rankings: Performance-based leaderboard

## 🚀 Quick Start

### Installation
```bash
cd /workspaces/neural-trader/monitoring
npm install
```

### Run Everything
```bash
# Dashboard + Health Checks (concurrent)
npm run monitor-all

# Individual components
npm run dashboard      # Real-time dashboard
npm run health-check   # Health monitoring
npm run validate       # Validation tests
npm run report         # Generate performance report
```

### Build
```bash
npm run build         # Compile TypeScript
npm run typecheck     # Type checking only
```

## 📁 Directory Structure

```
monitoring/
├── dashboard/
│   └── real-time-monitor.ts      # Interactive dashboard
├── health/
│   └── health-check-system.ts    # Health monitoring
├── validation/
│   └── deployment-validator.ts   # Validation tests
├── reports/
│   ├── performance-reporter.ts   # Report generator
│   └── output/                   # Generated reports
├── utils/                        # Shared utilities
├── package.json
├── tsconfig.json
├── jest.config.js
└── README.md
```

## 🔧 Configuration

### Health Check Configuration
```typescript
const config = {
  deploymentId: 'neural-trader-1763096012878',
  sandboxIds: ['sandbox-1', 'sandbox-2', 'sandbox-3', 'sandbox-4', 'sandbox-5'],
  checkInterval: 60000,        // 60 seconds
  timeout: 10000,              // 10 seconds
  maxRetries: 3,
  quicSyncInterval: 5000       // 5 seconds
};
```

### Alert Thresholds
```typescript
const alertThresholds = {
  cpu: 90,                     // 90% CPU
  memory: 85,                  // 85% memory
  consecutiveFailures: 3,      // 3 failures
  responseTime: 5000           // 5 seconds
};
```

## 📊 Sample Output

### Dashboard
```
╔══════════════════════════════════════════════════════════════╗
║  Neural Trader Swarm Monitor                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Deployment: neural-trader-1763096012878                     ║
║  Topology: MESH | Agents: 5/5 | Uptime: 2h 15m 43s          ║
║                                                              ║
║  Portfolio: $105,234 | P&L: +$5,234 | Sharpe: 1.87          ║
╚══════════════════════════════════════════════════════════════╝
```

### Health Check
```
[2025-01-14T10:30:45] Running health checks...
  ✅ sandbox-1: All checks passed (Response: 45ms)
  ✅ sandbox-2: All checks passed (Response: 52ms)
  ✅ sandbox-3: All checks passed (Response: 38ms)
  ⚠️  sandbox-4: High CPU usage (92%)
  ✅ sandbox-5: All checks passed (Response: 41ms)

Health check complete: 4/5 healthy
```

### Validation
```
🔍 Starting deployment validation...

📦 Validating sandboxes...
  ✅ Sandbox sandbox-1 is running (87ms)
  ✅ Sandbox sandbox-2 is running (92ms)

🔗 Validating inter-agent communication...
  ✅ Mesh topology communication (145ms)
  ✅ Agent message passing (1023ms)

============================================================
📋 VALIDATION REPORT
============================================================
Deployment ID: neural-trader-1763096012878
Total Tests: 18
Passed: 18 ✅
Failed: 0 ❌
Success Rate: 100.0%
============================================================
```

## 🎯 Integration with Neural Trader

### Event Integration
```typescript
// Listen to monitoring events
healthSystem.on('sandbox-unhealthy', ({ sandboxId, status }) => {
  // Trigger failover
  swarmCoordinator.failover(sandboxId);
});

dashboard.on('alert', (alert) => {
  // Send notification
  notificationService.send(alert);
});
```

### API Integration
```typescript
// Expose monitoring API
app.get('/api/monitoring/status', async (req, res) => {
  const status = healthSystem.getAggregateStatus();
  res.json(status);
});

app.get('/api/monitoring/report', async (req, res) => {
  const report = await reporter.generateFullReport(agentData, startTime, endTime);
  res.json(report);
});
```

## 🔍 Troubleshooting

### Dashboard not rendering?
```bash
# Ensure blessed dependencies installed
npm install blessed blessed-contrib
```

### Health checks failing?
```bash
# Check E2B API connectivity
# Verify sandbox IDs are correct
# Check network connectivity
```

### Validation tests timing out?
```bash
# Increase test timeout in jest.config.js
testTimeout: 60000  // 60 seconds
```

## 📈 Performance Targets

- **Dashboard Refresh**: < 1 second
- **Health Check Cycle**: 60 seconds
- **Validation Suite**: < 30 seconds
- **Report Generation**: < 5 seconds

## 🛡️ Best Practices

1. **Always run health checks** before deployment
2. **Monitor dashboard** during active trading
3. **Validate deployment** after any changes
4. **Generate reports** daily for analysis
5. **Set up alerts** for critical metrics
6. **Archive reports** for historical analysis

## 📝 Dependencies

- `blessed`: Terminal UI framework
- `blessed-contrib`: Charts and widgets
- `axios`: HTTP client
- `jest`: Testing framework
- `typescript`: Type safety

## 🤝 Contributing

When adding monitoring features:
1. Follow existing patterns
2. Add comprehensive error handling
3. Include event emission for integration
4. Document configuration options
5. Add tests for validation

## 📄 License

MIT - Part of Neural Trader project
