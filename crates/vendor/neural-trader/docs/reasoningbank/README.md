# ReasoningBank E2B Swarm Learning Dashboard

> Real-time visualization and analytics for adaptive learning in distributed agent swarms

## 🎯 Overview

The ReasoningBank Learning Dashboard provides comprehensive visualization and analytics for E2B swarm learning processes. It combines real-time metrics tracking, predictive analytics, and multi-format reporting to help you understand and optimize agent learning performance.

## ✨ Key Features

### 📊 Real-Time Visualizations
- **Learning Curves**: Track accuracy progression over episodes
- **Decision Quality**: Monitor decision-making quality trends
- **Pattern Discovery**: Visualize pattern growth and diversity
- **Agent Skills**: Analyze skill matrices and specialization
- **Knowledge Graphs**: Map pattern relationships and dependencies

### 📈 Comparative Analysis
- **Topology Comparison**: Compare learning performance across topologies (mesh, hierarchical, ring, star)
- **Strategy Effectiveness**: Evaluate different learning strategies (Q-learning, Policy Gradient, Actor-Critic, PPO)
- **Agent Performance**: Benchmark individual agent capabilities

### 🔮 Predictive Analytics
- **Convergence Prediction**: Estimate episodes needed to reach target accuracy
- **Bottleneck Detection**: Identify learning plateaus and inefficiencies
- **Optimization Recommendations**: Get actionable improvement suggestions

### 💾 Multi-Format Export
- **HTML Dashboards**: Interactive charts powered by Chart.js
- **Markdown Reports**: Documentation-ready analysis
- **JSON Data**: Raw data for pipeline integration
- **ASCII Charts**: Beautiful terminal visualizations

## 🚀 Quick Start

### 1. Generate Demo Data

```bash
node src/reasoningbank/demo-data-generator.js docs/reasoningbank/demo-data.json
```

### 2. View Dashboard

```bash
# Interactive HTML dashboard (opens in browser)
node scripts/e2b-swarm-cli.js learning dashboard -s docs/reasoningbank/demo-data.json

# Live terminal dashboard
node scripts/e2b-swarm-cli.js learning dashboard --live -s docs/reasoningbank/demo-data.json
```

### 3. Get Quick Stats

```bash
node scripts/e2b-swarm-cli.js learning stats -s docs/reasoningbank/demo-data.json
```

### 4. Generate Report

```bash
node scripts/e2b-swarm-cli.js learning report --format html -s docs/reasoningbank/demo-data.json
```

## 📚 Documentation

- **[Quick Start Guide](./QUICK_START.md)** - Get started in 5 minutes
- **[Complete Guide](./LEARNING_DASHBOARD_GUIDE.md)** - Comprehensive documentation
- **[API Reference](./LEARNING_DASHBOARD_GUIDE.md#api-reference)** - Programmatic usage

## 🎨 Visualization Examples

### Learning Curve (ASCII)
```
Learning Curve - Accuracy Over Episodes
========================================

  0.95 │                                                    ●●●●
  0.85 │                                            ●●●●●●●
  0.75 │                                    ●●●●●●●
  0.65 │                            ●●●●●●●
  0.55 │                    ●●●●●●●
  0.45 │            ●●●●●●●
  0.35 │    ●●●●●●●
  0.25 │●●●●
       └────────────────────────────────────────────────────────
         Episodes
```

### Agent Skills (ASCII)
```
Agent Skills Matrix (agent-001)
================================

pattern-recognition │█████████████████████████████████████████████ 0.85
decision-making     │██████████████████████████████████████ 0.78
exploration         │████████████████████████████ 0.65
exploitation        │████████████████████████████████████████████████ 0.92
coordination        │███████████████████████████████ 0.70
```

### HTML Dashboard

The HTML dashboard provides interactive visualizations with:
- ✅ Responsive design
- ✅ Interactive Chart.js charts
- ✅ Real-time updates
- ✅ Beautiful gradient UI
- ✅ Comprehensive analytics

![Dashboard Preview](./assets/dashboard-preview.png)

## 📋 CLI Commands

| Command | Description | Options |
|---------|-------------|---------|
| `learning dashboard` | Generate HTML or live terminal dashboard | `--live`, `-s <source>` |
| `learning stats` | Show quick statistics | `-a <agent>`, `-s <source>` |
| `learning analytics` | Display analytics and predictions | `-s <source>` |
| `learning report` | Generate formatted report | `-f <format>`, `-o <output>` |
| `learning export` | Export raw data | `-f <format>`, `-o <output>` |

## 💻 Programmatic Usage

### Basic Example

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');

// Create dashboard
const dashboard = new LearningDashboard({
  updateInterval: 1000,
  historySize: 100,
  targetAccuracy: 0.95
});

// Update with learning data
dashboard.updateMetrics({
  learningCurve: [
    { episode: 1, accuracy: 0.45 },
    { episode: 2, accuracy: 0.52 }
  ],
  decisionQuality: [
    { score: 0.6, timestamp: '2025-01-01T00:00:00Z' }
  ],
  patternGrowth: [
    { count: 5, timestamp: '2025-01-01T00:00:00Z' }
  ],
  agentSkills: {
    'agent-001': {
      'pattern-recognition': 0.85,
      'decision-making': 0.78
    }
  }
});

// Display visualizations
console.log(await dashboard.displayLearningCurve());
console.log(await dashboard.predictConvergence());

// Export
await dashboard.exportHTML('./dashboard.html');
await dashboard.exportMarkdown('./report.md');
```

### Advanced Example

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');
const fs = require('fs').promises;

async function analyzeLearning() {
  // Load data
  const data = JSON.parse(await fs.readFile('./learning-data.json', 'utf8'));

  // Create dashboard
  const dashboard = new LearningDashboard({ targetAccuracy: 0.90 });
  dashboard.updateMetrics(data);

  // Generate comprehensive report
  const report = await dashboard.generateCompleteReport();
  console.log(report);

  // Get analytics
  const convergence = await dashboard.predictConvergence();
  const bottlenecks = await dashboard.identifyBottlenecks();
  const recommendations = await dashboard.recommendOptimizations();

  // Export
  await dashboard.exportHTML('./docs/dashboard.html');

  return { convergence, bottlenecks, recommendations };
}
```

## 🎯 Use Cases

### 1. Training Monitoring
Monitor learning progress in real-time during agent training:
```bash
node scripts/e2b-swarm-cli.js learning dashboard --live
```

### 2. Performance Analysis
Analyze completed training runs to identify optimization opportunities:
```bash
node scripts/e2b-swarm-cli.js learning analytics -s training-data.json
```

### 3. Comparative Studies
Compare different topologies, strategies, or configurations:
```bash
node scripts/e2b-swarm-cli.js learning report --format html -s comparison-data.json
```

### 4. Documentation
Generate reports for documentation and presentations:
```bash
node scripts/e2b-swarm-cli.js learning report --format markdown -o docs/analysis.md
```

## 📊 Data Format

The dashboard accepts JSON data with the following structure:

```json
{
  "learningCurve": [
    { "episode": 1, "accuracy": 0.45, "loss": 0.55 }
  ],
  "decisionQuality": [
    { "score": 0.6, "confidence": 0.7 }
  ],
  "patternGrowth": [
    { "count": 5, "newPatterns": 2 }
  ],
  "agentSkills": {
    "agent-001": {
      "pattern-recognition": 0.85,
      "decision-making": 0.78
    }
  },
  "knowledgeGraph": {
    "nodes": [
      { "id": "p1", "pattern": "Pattern description", "confidence": 0.92 }
    ],
    "edges": [
      { "from": "p1", "to": "p2", "weight": 0.65 }
    ]
  },
  "topologies": [
    { "topology": "Mesh", "accuracy": 0.89, "episodes": 45 }
  ],
  "strategies": [
    { "strategy": "Q-Learning", "effectiveness": 0.84 }
  ],
  "agents": [
    { "agent": "Agent-001", "performance": 0.87 }
  ]
}
```

## 🛠️ Development

### Running Tests

```bash
# Run comprehensive demo
node examples/reasoningbank-dashboard-demo.js

# Generate demo data
node src/reasoningbank/demo-data-generator.js

# Test CLI commands
node scripts/e2b-swarm-cli.js learning --help
```

### Project Structure

```
src/reasoningbank/
├── learning-dashboard.js      # Main dashboard class
├── dashboard-cli.js           # CLI integration
└── demo-data-generator.js     # Demo data generation

docs/reasoningbank/
├── README.md                  # This file
├── QUICK_START.md            # Quick start guide
├── LEARNING_DASHBOARD_GUIDE.md # Complete guide
├── dashboards/               # Generated HTML dashboards
├── reports/                  # Generated reports
└── demo-data.json           # Demo learning data

examples/
└── reasoningbank-dashboard-demo.js  # Interactive demo
```

## 🔧 Configuration

### Dashboard Options

```javascript
{
  updateInterval: 1000,    // Update interval in milliseconds
  historySize: 100,        // Maximum history points to retain
  targetAccuracy: 0.95     // Target accuracy for predictions
}
```

### CLI Options

```bash
# Dashboard options
--live                    # Live terminal mode
-i, --interval <ms>      # Update interval (default: 2000)
-d, --duration <ms>      # Display duration (default: 60000)
-s, --source <file>      # Data source file

# Report options
-f, --format <format>    # Output format (html/markdown/json)
-o, --output <path>      # Output file path

# Agent stats options
-a, --agent <id>         # Specific agent ID
```

## 📈 Analytics Features

### Convergence Prediction

Estimates episodes needed to reach target accuracy based on current learning rate:

```
Convergence Prediction
=====================

Current Accuracy: 87.45%
Target Accuracy: 95.00%
Avg Improvement: 0.0234% per episode
Estimated Episodes to Target: 323
```

### Bottleneck Analysis

Identifies learning inefficiencies and plateaus:

```
Learning Bottleneck Analysis
============================

1. [HIGH] Learning Plateau
   Learning has plateaued. Consider increasing exploration.

2. [MEDIUM] Low Pattern Diversity
   Few new patterns discovered. Increase exploration diversity.
```

### Optimization Recommendations

Provides actionable improvement suggestions:

```
Optimization Recommendations
============================

1. [HIGH] Learning Rate
   Increase exploration rate or implement curriculum learning
   Expected Impact: +15-30% faster convergence

2. [MEDIUM] Topology
   Mesh topology shows best performance.
   Expected Impact: +12% accuracy improvement
```

## 🎓 Best Practices

1. **Regular Monitoring**: Check dashboard every 10-20 episodes
2. **Data Retention**: Keep historical data for trend analysis
3. **Comparative Analysis**: Test multiple configurations
4. **Export Regularly**: Save dashboards and reports for documentation
5. **Act on Recommendations**: Implement optimization suggestions

## 🤝 Integration

### E2B Swarm Integration

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');
const { E2BSwarmManager } = require('./src/e2b/swarm-manager');

// Initialize swarm and dashboard
const swarm = new E2BSwarmManager();
const dashboard = new LearningDashboard();

// Hook into learning events
swarm.on('episode_complete', (episode, metrics) => {
  dashboard.updateMetrics({
    learningCurve: [{ episode: episode.number, accuracy: episode.accuracy }],
    decisionQuality: [{ score: metrics.qualityScore }],
    patternGrowth: [{ count: metrics.patternsDiscovered }]
  });

  // Display every 10 episodes
  if (episode.number % 10 === 0) {
    console.log(await dashboard.displayLearningCurve());
    console.log(await dashboard.predictConvergence());
  }
});
```

### CI/CD Integration

```yaml
# .github/workflows/training.yml
- name: Train and analyze
  run: |
    node train-agents.js > training-data.json
    node scripts/e2b-swarm-cli.js learning report \
      --format markdown \
      -s training-data.json \
      -o reports/training-analysis.md

- name: Upload dashboard
  uses: actions/upload-artifact@v2
  with:
    name: learning-dashboard
    path: docs/reasoningbank/dashboards/
```

## 🆘 Troubleshooting

**Issue**: No data available
```bash
# Solution: Generate demo data
node src/reasoningbank/demo-data-generator.js docs/reasoningbank/demo-data.json
```

**Issue**: Charts not rendering in HTML
- Ensure internet connection for Chart.js CDN
- Check browser console for errors
- Verify data format is correct

**Issue**: CLI command not found
```bash
# Solution: Run from project root
cd /workspaces/neural-trader
node scripts/e2b-swarm-cli.js learning --help
```

## 📝 License

MIT License - see [LICENSE](../../LICENSE) for details

## 🙏 Acknowledgments

- Chart.js for beautiful interactive charts
- ASCII chart rendering inspired by asciichart
- ReasoningBank methodology for adaptive learning

## 📧 Support

- **Documentation**: Check [LEARNING_DASHBOARD_GUIDE.md](./LEARNING_DASHBOARD_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
- **Examples**: See `examples/reasoningbank-dashboard-demo.js`

---

**Built with ❤️ for the Neural Trader project**
