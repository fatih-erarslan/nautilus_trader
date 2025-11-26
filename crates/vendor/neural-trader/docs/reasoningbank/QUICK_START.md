# ReasoningBank Learning Dashboard - Quick Start

## 🚀 Installation

No additional installation needed! The dashboard is included with neural-trader.

```bash
cd /workspaces/neural-trader
```

## 📊 Quick Examples

### 1. Generate Demo Data (First Time)

```bash
# Generate sample learning data
node src/reasoningbank/demo-data-generator.js docs/reasoningbank/demo-data.json
```

### 2. View Quick Stats

```bash
# Display key metrics
node scripts/e2b-swarm-cli.js learning stats -s docs/reasoningbank/demo-data.json
```

Output:
```
Quick Stats
════════════════════════════════════════
Episodes:       100
Accuracy:       93.67%
Patterns:       39
Active Agents:  5
Top Topology:   Mesh
════════════════════════════════════════
```

### 3. View Analytics

```bash
# Show convergence predictions and recommendations
node scripts/e2b-swarm-cli.js learning analytics -s docs/reasoningbank/demo-data.json
```

### 4. Generate HTML Dashboard

```bash
# Create interactive HTML dashboard (opens in browser)
node scripts/e2b-swarm-cli.js learning dashboard -s docs/reasoningbank/demo-data.json
```

### 5. Live Dashboard Mode

```bash
# Real-time updates in terminal (2s intervals, 60s duration)
node scripts/e2b-swarm-cli.js learning dashboard --live -s docs/reasoningbank/demo-data.json
```

### 6. Generate Report

```bash
# Create markdown report
node scripts/e2b-swarm-cli.js learning report --format markdown -s docs/reasoningbank/demo-data.json

# Create HTML report
node scripts/e2b-swarm-cli.js learning report --format html -s docs/reasoningbank/demo-data.json
```

### 7. Run Interactive Demo

```bash
# Run comprehensive demo showing all features
node examples/reasoningbank-dashboard-demo.js
```

## 🎨 Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `learning dashboard` | Generate HTML dashboard | `--live` for terminal mode |
| `learning stats` | Show quick statistics | `-a agent-001` for specific agent |
| `learning analytics` | Display analytics | Shows predictions & recommendations |
| `learning report` | Generate report | `-f html/markdown/json` |
| `learning export` | Export data | `-o path/to/output.json` |

## 📁 File Locations

Generated files are saved to:

- **Dashboards**: `/docs/reasoningbank/dashboards/`
- **Reports**: `/docs/reasoningbank/reports/`
- **Demo Data**: `/docs/reasoningbank/demo-data.json`

## 💡 Common Use Cases

### Scenario 1: Quick Health Check

```bash
# See if learning is progressing well
node scripts/e2b-swarm-cli.js learning stats -s docs/reasoningbank/demo-data.json
```

### Scenario 2: Detailed Analysis

```bash
# Generate HTML dashboard for deep dive
node scripts/e2b-swarm-cli.js learning dashboard -s docs/reasoningbank/demo-data.json

# Get optimization recommendations
node scripts/e2b-swarm-cli.js learning analytics -s docs/reasoningbank/demo-data.json
```

### Scenario 3: Documentation

```bash
# Create markdown report for documentation
node scripts/e2b-swarm-cli.js learning report --format markdown \
  -s docs/reasoningbank/demo-data.json \
  -o docs/reports/learning-analysis.md
```

### Scenario 4: Real-Time Monitoring

```bash
# Monitor learning progress in real-time
node scripts/e2b-swarm-cli.js learning dashboard --live -i 1000 -d 300000
```

## 🔧 Programmatic Usage

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');

// Create dashboard
const dashboard = new LearningDashboard({
  targetAccuracy: 0.90
});

// Update with data
dashboard.updateMetrics({
  learningCurve: [{ episode: 1, accuracy: 0.5 }],
  // ... more data
});

// Display visualizations
console.log(await dashboard.displayLearningCurve());
console.log(await dashboard.predictConvergence());

// Export
await dashboard.exportHTML('./dashboard.html');
```

## 📚 Next Steps

1. **Read the full guide**: [LEARNING_DASHBOARD_GUIDE.md](./LEARNING_DASHBOARD_GUIDE.md)
2. **Run the demo**: `node examples/reasoningbank-dashboard-demo.js`
3. **Explore exports**: Check `docs/reasoningbank/dashboards/` and `reports/`

## 🆘 Troubleshooting

**Issue**: "No data available"
```bash
# Solution: Generate demo data first
node src/reasoningbank/demo-data-generator.js docs/reasoningbank/demo-data.json
```

**Issue**: Charts not showing
```bash
# Solution: Ensure output directory exists
mkdir -p docs/reasoningbank/{dashboards,reports}
```

**Issue**: Command not found
```bash
# Solution: Run from project root
cd /workspaces/neural-trader
node scripts/e2b-swarm-cli.js learning --help
```

## ✨ Features Highlight

- ✅ **ASCII Charts**: Beautiful terminal visualizations
- ✅ **Interactive HTML**: Chart.js-powered dashboards
- ✅ **Predictive Analytics**: Convergence forecasting
- ✅ **Bottleneck Detection**: Identify learning issues
- ✅ **Multi-format Export**: HTML, Markdown, JSON
- ✅ **Real-time Updates**: Live dashboard mode
- ✅ **Agent Analysis**: Skill matrices and comparisons
- ✅ **Knowledge Graphs**: Pattern relationship mapping

## 🎯 Quick Tips

1. **Start with stats**: Always run `learning stats` first for overview
2. **Use HTML for sharing**: Generate HTML dashboards for presentations
3. **Monitor convergence**: Check analytics regularly during training
4. **Export for analysis**: Save JSON for further data processing
5. **Live mode for debugging**: Use `--live` to watch learning progress

---

**Need Help?** See [LEARNING_DASHBOARD_GUIDE.md](./LEARNING_DASHBOARD_GUIDE.md) for detailed documentation.
